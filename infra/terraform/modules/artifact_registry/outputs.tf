output "name" {
  value = google_artifact_registry_repository.this.name
}

output "location" {
  value = google_artifact_registry_repository.this.location
}

output "url" {
  description = "Docker URL prefix: <url>/<image>:<git sha>."
  value       = "${var.location}-docker.pkg.dev/${google_artifact_registry_repository.this.project}/${var.repository_id}"
}
