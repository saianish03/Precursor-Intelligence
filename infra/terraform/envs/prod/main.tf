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
  apply_branches    = var.apply_branches
  state_bucket_name = module.state_bucket.name
}
