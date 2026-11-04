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

# The VM may upload a dump and list or download one for a restore, but not delete or overwrite one (no objectAdmin):
# a broken or compromised VM cannot destroy its own backups.
resource "google_storage_bucket_iam_member" "pipeline_backups" {
  for_each = toset(["roles/storage.objectCreator", "roles/storage.objectViewer"])

  bucket = module.backups_bucket.name
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
  # The VM renders these into its .env at deploy time; the comment is the variable each becomes.
  vm_secrets = toset([
    "airflow-db-password",    # AIRFLOW_DB_PASSWORD     Postgres role `airflow`
    "pi-meta-db-password",    # PI_META_DB_PASSWORD     Postgres role `pi_meta` (pipeline metadata)
    "airflow-fernet-key",     # AIRFLOW_FERNET_KEY      never change it once set: it decrypts stored connections
    "airflow-jwt-secret",     # AIRFLOW_JWT_SECRET
    "airflow-api-secret-key", # AIRFLOW_API_SECRET_KEY  compose still reuses the JWT secret; a later task splits them
    "smtp-password",          # PI_ALERT_SMTP_PASSWORD  the alert sender's Gmail app password
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
