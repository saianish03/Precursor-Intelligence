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
