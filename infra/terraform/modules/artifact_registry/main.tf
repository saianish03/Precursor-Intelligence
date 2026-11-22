resource "google_artifact_registry_repository" "this" {
  location      = var.location
  repository_id = var.repository_id
  description   = "Pipeline, Airflow and MLflow images, tagged with the full git sha"
  format        = "DOCKER"

  # A sha tag can never be moved to other bytes, so the VM runs exactly the commit it was told to.
  docker_config {
    immutable_tags = true
  }

  cleanup_policy_dry_run = false

  # KEEP wins over DELETE: the 5 newest versions of each image always stay, for rollback; older ones go after 7 days
  # (a third image per release, ML Phase 2 item 2, would otherwise grow the storage bill), and untagged layers left
  # by rebuilds after 7.
  cleanup_policies {
    id     = "keep-5-newest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  # main's releases stay whatever their age (CD tags them main-<sha> on a push to main), so the VM can always deploy
  # or roll back to them (docs/runbooks/vm.md section 3).
  cleanup_policies {
    id     = "keep-main-releases"
    action = "KEEP"
    condition {
      tag_state    = "TAGGED"
      tag_prefixes = ["main-"]
    }
  }

  cleanup_policies {
    id     = "delete-after-7-days"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "604800s"
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
