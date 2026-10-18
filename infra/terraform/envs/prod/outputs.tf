output "data_bucket" {
  value = module.data_bucket.name
}

output "pipeline_service_account" {
  value = google_service_account.pipeline.email
}

output "workload_identity_provider" {
  description = "Goes into .github/workflows/terraform.yml."
  value       = module.wif.provider_name
}
