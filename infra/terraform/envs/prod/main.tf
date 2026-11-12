provider "google" {
  project = var.project_id
  region  = var.region
}

# The one object-store root (spec 0.6): landing, bronze, silver, gold, reports and run manifests.
# No retention policy: bronze reloads and gold promotion rewrite objects. Versioning is the protection.
module "data_bucket" {
  source              = "../../modules/gcs"
  name                = "${var.project_id}-data"
  location            = var.region
  quarantine_ttl_days = 30
}

# DVC's remote (README, "Data versioning with DVC"): the toy pipeline's data versions. A bucket of its own, so
# whoever runs `dvc push` never needs write access to the production data above.
module "dvc_bucket" {
  source   = "../../modules/gcs"
  name     = "${var.project_id}-dvc"
  location = var.region
}

module "state_bucket" {
  source   = "../../modules/gcs"
  name     = "${var.project_id}-tfstate"
  location = var.region
}

# Postgres dumps taken on the VM before every stop (spec §14 decision 27). A bucket of its own, so clearing the data
# bucket never deletes them. No versioning, since every dump has a new name; dumps expire after 180 days.
module "backups_bucket" {
  source            = "../../modules/gcs"
  name              = "${var.project_id}-backups"
  location          = var.region
  versioning        = false
  delete_after_days = 180
}

# MLflow's artifact store (ML spec §14.1, §16.3): model packages, plots and reports, written through the MLflow server
# (--serve-artifacts). A bucket of its own with no expiry: registered models must survive a wipe of the data bucket
# (D11). Versioning keeps an overwritten artifact; GCS's default 7-day soft delete lets a person restore a deleted one
# (sa-pipeline can delete old versions, so versioning alone doesn't protect against a compromised VM).
module "mlflow_bucket" {
  source   = "../../modules/gcs"
  name     = "${var.project_id}-mlflow"
  location = var.region
}

# What the pipeline VM runs as. gcsfs checks the bucket itself before writing, which needs
# storage.buckets.get; objectAdmin alone makes every write fail with "bucket does not exist".
resource "google_service_account" "pipeline" {
  account_id   = "sa-pipeline"
  display_name = "Pipeline runtime (VM)"
}

resource "google_storage_bucket_iam_member" "pipeline_objects" {
  bucket = module.data_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipeline.email}"
}

resource "google_storage_bucket_iam_member" "pipeline_bucket_reader" {
  bucket = module.data_bucket.name
  role   = "roles/storage.legacyBucketReader"
  member = "serviceAccount:${google_service_account.pipeline.email}"
}

# The VM may only upload dumps, not read, list, overwrite or delete them: a broken or compromised VM cannot destroy its
# backups, and a compromised task container (it gets the VM's identity) cannot read them. A person restores with their
# own login (infra/vm/README.md). objectViewer was dropped by hand first, since CI refuses deletes; the for_each of
# one keeps objectCreator's existing state address.
resource "google_storage_bucket_iam_member" "pipeline_backups" {
  for_each = toset(["roles/storage.objectCreator"])

  bucket = module.backups_bucket.name
  role   = each.key
  member = "serviceAccount:${google_service_account.pipeline.email}"
}

# The MLflow server runs as sa-pipeline (the VM's identity; on a laptop, the impersonated ADC file) and writes, reads
# and deletes artifacts (an overwrite and `mlflow gc` need delete, so objectCreator + objectViewer would 403).
# objectUser, not objectAdmin: no IAM rights on the bucket's objects. The storage client checks the bucket first, so
# legacyBucketReader as on the data bucket.
resource "google_storage_bucket_iam_member" "pipeline_mlflow" {
  for_each = toset(["roles/storage.objectUser", "roles/storage.legacyBucketReader"])

  bucket = module.mlflow_bucket.name
  role   = each.key
  member = "serviceAccount:${google_service_account.pipeline.email}"
}

# The VM's logging and monitoring agents write as sa-pipeline; without these its logs and metrics are dropped.
resource "google_project_iam_member" "pipeline_runtime" {
  for_each = toset(["roles/logging.logWriter", "roles/monitoring.metricWriter"])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.pipeline.email}"
}

module "wif" {
  source = "../../modules/wif"

  project_id        = var.project_id
  project_number    = var.project_number
  github_repository = var.github_repository
  github_subject    = var.github_subject
  apply_branches    = var.apply_branches
  state_bucket_name = module.state_bucket.name
}

# Every image CI builds (pipeline, airflow), tagged with the full git sha. The VM pulls the sha it is told to run.
module "artifact_registry" {
  source        = "../../modules/artifact_registry"
  location      = var.region
  repository_id = "precursor"
}

resource "google_artifact_registry_repository_iam_member" "pipeline_pulls" {
  location   = module.artifact_registry.location
  repository = module.artifact_registry.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.pipeline.email}"
}

# CD pushes the images as this account. Like sa-tf-apply, only a run on one of apply_branches can act as it,
# never a pull request, so code that has not been reviewed and merged can never publish an image.
resource "google_service_account" "deployer" {
  account_id   = "sa-deployer"
  display_name = "CD image push (merged branches only)"

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_artifact_registry_repository_iam_member" "deployer_pushes" {
  location   = module.artifact_registry.location
  repository = module.artifact_registry.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_from_branch" {
  for_each = toset(var.apply_branches)

  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "${module.wif.subject_prefix}:ref:refs/heads/${each.key}"
}

locals {
  # The VM renders these into its .env at deploy time; the comment is the variable each becomes. The two alert
  # addresses aren't secret, but email addresses stay out of this public repository.
  vm_secrets = toset([
    "airflow-db-password",    # AIRFLOW_DB_PASSWORD     Postgres role `airflow`
    "pi-meta-db-password",    # PI_META_DB_PASSWORD     Postgres role `pi_meta` (pipeline metadata)
    "airflow-fernet-key",     # AIRFLOW_FERNET_KEY      never change it once set: it decrypts stored connections
    "airflow-jwt-secret",     # AIRFLOW_JWT_SECRET
    "airflow-api-secret-key", # AIRFLOW_API_SECRET_KEY  docker-compose.vm.yaml; a laptop reuses the JWT secret
    "smtp-password",          # PI_ALERT_SMTP_PASSWORD  the alert sender's Gmail app password
    "alert-smtp-user",        # PI_ALERT_SMTP_USER      the alert sender's Gmail address
    "alert-email-to",         # PI_ALERT_EMAIL_TO       the alert recipients, comma-separated
    "mlflow-db-password",     # MLFLOW_DB_PASSWORD      Postgres role `mlflow` (ML spec §16.1); hex only, it goes in a URL
  ])
}

# Empty containers only. A person adds each value with `gcloud secrets versions add` (gcp-bootstrap.md), so no
# value ever passes through Terraform or its state. sa-pipeline may read each one, and nothing else.
resource "google_secret_manager_secret" "vm" {
  for_each = local.vm_secrets

  secret_id = each.key
  labels    = { app = "precursor", consumer = "airflow-vm" }

  # One region for every resource, like the buckets.
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  # Deleting a secret deletes its values with it; the Fernet key in particular cannot be recreated.
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_secret_manager_secret_iam_member" "pipeline_reads" {
  for_each = local.vm_secrets

  secret_id = google_secret_manager_secret.vm[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.pipeline.email}"
}

# The VM's own network, so it never shares the default network's open rules. The VM gets no external IP.
resource "google_compute_network" "vpc" {
  name                    = "pi-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

resource "google_compute_subnetwork" "main" {
  name          = "pi-subnet"
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = "10.10.0.0/24"

  # GCS, Artifact Registry and Secret Manager are reached without an external IP.
  private_ip_google_access = true
}

resource "google_compute_router" "nat" {
  name    = "pi-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

# Outbound internet for a VM with no external IP: the pipeline downloads the CMS archive from data.cms.gov, and
# compose pulls postgres and the Docker proxy from Docker Hub.
resource "google_compute_router_nat" "nat" {
  name                               = "pi-nat"
  router                             = google_compute_router.nat.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# The only inbound rule: SSH from Google's IAP range to VMs tagged pi-airflow. The Airflow UI is reached through an
# SSH port forward, so 8080 is never opened; everything else inbound is denied by default.
resource "google_compute_firewall" "iap_ssh" {
  name          = "pi-allow-iap-ssh"
  network       = google_compute_network.vpc.id
  direction     = "INGRESS"
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["pi-airflow"]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

# Postgres's data disk (spec §14 decision 27). A disk of its own, outside the VM, so the database survives a replaced
# VM or boot disk; the VM's startup script formats it once and mounts it at /mnt/disks/pgdata. Backups are the
# pg_dumps in the backups bucket (infra/vm/README.md), not snapshots.
resource "google_compute_disk" "pgdata" {
  name   = "pi-airflow-pgdata"
  zone   = var.zone
  type   = "pd-balanced"
  size   = 10
  labels = { app = "precursor", role = "postgres-data" }

  lifecycle {
    prevent_destroy = true

    # One region for everything: the subnet and NAT are in var.region.
    precondition {
      condition     = startswith(var.zone, "${var.region}-")
      error_message = "var.zone must be a zone of var.region."
    }
  }
}

# The Airflow VM (spec §14 decisions 26–27). It only builds the machine: what runs on it comes from the images that
# the release pointer pi-release names (set by the Deploy workflow, T22, never by Terraform). It is started and
# stopped by people; a stopped VM costs only its disks.
resource "google_compute_instance" "airflow" {
  name         = "pi-airflow"
  zone         = var.zone
  machine_type = var.vm_machine_type
  tags         = [one(google_compute_firewall.iap_ssh.target_tags)] # the only rule that lets anything in: IAP SSH
  labels       = { app = "precursor", role = "airflow" }

  deletion_protection = true
  # A new machine type, service account, scope or Shielded VM setting needs a stopped VM. Never let an apply stop it:
  # it would kill a running backfill. On a running VM such an apply fails instead; stop the VM and re-run the job.
  allow_stopping_for_update = false

  boot_disk {
    auto_delete = true
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 30
      type  = "pd-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.pgdata.id
    device_name = "pgdata" # /dev/disk/by-id/google-pgdata in infra/vm/startup.sh
  }

  # No access_config, so no external IP: egress through Cloud NAT, Google APIs through Private Google Access.
  network_interface {
    subnetwork = google_compute_subnetwork.main.id
  }

  # Tokens from the metadata server for sa-pipeline; what it may do is set by IAM alone.
  service_account {
    email  = google_service_account.pipeline.email
    scopes = ["cloud-platform"]
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  scheduling {
    provisioning_model  = "STANDARD" # never Spot: a preemption would kill a backfill
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
  }

  # Files go to the VM as metadata values, read by the startup script at every boot. Changing one updates the
  # metadata in place; metadata_startup_script would replace the VM instead. No secret may go here: every process on
  # the VM, task containers included, can read the instance metadata.
  metadata = {
    enable-oslogin         = "TRUE" # people log in with their Google identity through IAP; no SSH keys
    block-project-ssh-keys = "TRUE"
    # pi-deploy reports the deployed sha there; informational only, since anything on the VM may write them.
    enable-guest-attributes = "TRUE"
    startup-script          = file("${path.module}/../../../vm/startup.sh")
    pi-pg-backup            = file("${path.module}/../../../vm/pi-pg-backup")
    pi-pg-restore           = file("${path.module}/../../../vm/pi-pg-restore")
    pi-pg-backup-service    = file("${path.module}/../../../vm/pi-pg-backup.service")
    pi-deploy               = file("${path.module}/../../../vm/pi-deploy")
    pi-deploy-service       = file("${path.module}/../../../vm/pi-deploy.service")
    pi-busy                 = file("${path.module}/../../../vm/pi-busy")
    pi-alert                = file("${path.module}/../../../vm/pi-alert")
    # The idle shutdown and pi-down (T21)
    pi-down                  = file("${path.module}/../../../vm/pi-down")
    pi-idle-shutdown-service = file("${path.module}/../../../vm/pi-idle-shutdown.service")
    pi-idle-shutdown-timer   = file("${path.module}/../../../vm/pi-idle-shutdown.timer")
  }
  # (terraform.yml also runs on infra/vm/**, so a change to one of these files gets its own plan and apply.)

  lifecycle {
    ignore_changes = [
      # The image family moves on; a new Debian image must not replace the VM. Packages are updated in place on the VM
      # (apt; check that unattended-upgrades is enabled, infra/README.md "The VM").
      boot_disk[0].initialize_params[0].image,
      # The release pointer belongs to the Deploy workflow (T22).
      metadata["pi-release"],
    ]
  }
}
