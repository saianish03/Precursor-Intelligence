output "data_bucket" {
  value = module.data_bucket.name
}

output "dvc_bucket" {
  value = module.dvc_bucket.name
}

output "pipeline_service_account" {
  value = google_service_account.pipeline.email
}

output "workload_identity_provider" {
  description = "Goes into .github/workflows/terraform.yml."
  value       = module.wif.provider_name
}

output "artifact_registry_url" {
  description = "Image prefix for CD and the VM: <url>/<image>:<git sha>."
  value       = module.artifact_registry.url
}

output "deployer_service_account" {
  description = "What the CD workflow logs in as to push images."
  value       = google_service_account.deployer.email
}
