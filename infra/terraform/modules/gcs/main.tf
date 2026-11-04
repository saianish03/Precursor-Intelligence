resource "google_storage_bucket" "this" {
  name     = var.name
  location = var.location

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = var.versioning
  }

  # Only the data bucket sets this: scratch objects under quarantine/ are deleted after N days.
  dynamic "lifecycle_rule" {
    for_each = var.quarantine_ttl_days == null ? [] : [var.quarantine_ttl_days]
    content {
      action {
        type = "Delete"
      }
      condition {
        age            = lifecycle_rule.value
        matches_prefix = ["quarantine/"]
      }
    }
  }

  # Only the backups bucket sets this: every object expires, whatever its name.
  dynamic "lifecycle_rule" {
    for_each = var.delete_after_days == null ? [] : [var.delete_after_days]
    content {
      action {
        type = "Delete"
      }
      condition {
        age = lifecycle_rule.value
      }
    }
  }

  # Losing either bucket loses data or the Terraform state. Destroying one must be a deliberate, reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}
