output "data_bucket" {
  value = module.data_bucket.name
}

output "dvc_bucket" {
  value = module.dvc_bucket.name
}

output "backups_bucket" {
  value = module.backups_bucket.name
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

output "vm_secret_ids" {
  description = "The VM's secrets; their values are added by hand (gcp-bootstrap.md)."
  value       = sort(keys(google_secret_manager_secret.vm))
}

output "network" {
  value = google_compute_network.vpc.name
}

output "subnetwork" {
  value = google_compute_subnetwork.main.name
}

output "vm_network_tag" {
  description = "The VM must carry this tag, or the IAP SSH rule does not reach it."
  value       = one(google_compute_firewall.iap_ssh.target_tags)
}

output "vm" {
  description = "Reach it with: gcloud compute ssh <name> --zone <zone> --tunnel-through-iap"
  value       = { name = google_compute_instance.airflow.name, zone = google_compute_instance.airflow.zone }
}

output "vm_data_disk" {
  description = "Postgres's data disk; it outlives the VM."
  value       = google_compute_disk.pgdata.name
}
