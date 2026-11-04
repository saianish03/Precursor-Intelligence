resource "google_artifact_registry_repository" "this" {
  location      = var.location
  repository_id = var.repository_id
  description   = "Pipeline and Airflow images, tagged with the full git sha"
  format        = "DOCKER"

  # A sha tag can never be moved to other bytes, so the VM runs exactly the commit it was told to.
  docker_config {
    immutable_tags = true
  }

  cleanup_policy_dry_run = false

  # KEEP wins over DELETE: the 5 newest versions of each image always stay, for rollback; older ones go after 30
  # days, and untagged layers left by rebuilds after 7.
  cleanup_policies {
    id     = "keep-5-newest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  cleanup_policies {
    id     = "delete-after-30-days"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "2592000s"
    }
  }

  cleanup_policies {
    id     = "delete-untagged-after-7-days"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s"
    }
  }

  # Losing the repository loses every released image and with it the way back to an older release.
  lifecycle {
    prevent_destroy = true
  }
}
