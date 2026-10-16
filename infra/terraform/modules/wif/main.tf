# Keyless GitHub Actions login, plus the two Terraform identities. A pull request runs as the read-only
# `plan` account; only a run on one of `apply_branches` can act as the `apply` account. Anyone with write
# access to the repository can edit a workflow, so the split is what keeps a pull request from changing
# infrastructure.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_iam_workload_identity_pool_provider" "github_repo" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-repo"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository=='${var.github_repository}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_service_account" "plan" {
  account_id   = "sa-tf-plan"
  display_name = "Terraform plan (PRs, read-only)"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_service_account" "apply" {
  account_id   = "sa-tf-apply"
  display_name = "Terraform apply (merged branches only)"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

locals {
  # One workflow identity is exactly one token subject: no wildcards.
  subject = "principal://iam.googleapis.com/projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/subject/repo:${var.github_repository}"
}

resource "google_service_account_iam_member" "plan_from_pull_request" {
  service_account_id = google_service_account.plan.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "${local.subject}:pull_request"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_service_account_iam_member" "apply_from_branch" {
  for_each = toset(var.apply_branches)

  service_account_id = google_service_account.apply.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "${local.subject}:ref:refs/heads/${each.key}"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

# viewer cannot read IAM policies, which `plan` needs to refresh bucket and account bindings.
resource "google_project_iam_member" "plan" {
  for_each = toset(["roles/viewer", "roles/iam.securityReviewer"])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.plan.email}"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_project_iam_member" "apply" {
  for_each = toset(["roles/editor", "roles/resourcemanager.projectIamAdmin"])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.apply.email}"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}

# The plan account only reads the state. It runs the pull request's own code, so write access here would let
# any PR author rewrite or delete the state that the next apply trusts. Plans on PRs run with -lock=false for
# the same reason. Only the apply account, reachable from dev and main alone, writes.
resource "google_storage_bucket_iam_member" "state" {
  for_each = {
    plan  = { email = google_service_account.plan.email, role = "roles/storage.objectViewer" }
    apply = { email = google_service_account.apply.email, role = "roles/storage.objectAdmin" }
  }

  bucket = var.state_bucket_name
  role   = each.value.role
  member = "serviceAccount:${each.value.email}"

  # These are what lets CI log in and apply. Deleting one through Terraform would cut the branch Terraform
  # is standing on, so the plan fails instead; changing them is a deliberate, hand-reviewed edit.
  lifecycle {
    prevent_destroy = true
  }
}
