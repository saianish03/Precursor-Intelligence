# Keyless GitHub Actions login, plus the two Terraform identities. A pull request runs as the read-only
# `plan` account; only terraform.yml's runs on one of `apply_branches` can act as the `apply` account. Anyone with write
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

# sa-tf-apply can't update this provider (no iam.workloadIdentityPoolProviders.update): a change to it is made by an
# Owner with gcloud first, exactly as written here, and CI's next plan then shows no change (infra/README.md, "Which
# workflow can act as which account").
resource "google_iam_workload_identity_pool_provider" "github_repo" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-repo"

  # attribute.workflow is the token's subject, then the workflow file and branch that asked for it (job_workflow_ref):
  # "repo:OWNER@ID/REPO@ID:ref:refs/heads/dev:OWNER/REPO/.github/workflows/deploy.yml@refs/heads/dev". A binding on it
  # ties an account to one workflow, so a compromised action in one workflow can't act as another workflow's account.
  # The subject part keeps GitHub's permanent ids (a re-created or squatted repository name never matches). Both
  # claims are strings in every GitHub Actions token: the mapping has nothing that can fail to evaluate.
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.workflow"   = "assertion.sub + ':' + assertion.job_workflow_ref"
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
  # One workflow identity is exactly one token subject: no wildcards. GitHub's subject carries the owner and
  # repository ids (see var.github_subject); the plain repository name is only used in the provider condition.
  subject = "principal://iam.googleapis.com/projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/subject/${var.github_subject}"
  # The runs of one workflow file on one branch: append "/<subject>:ref:refs/heads/<branch>:<job_workflow_ref>".
  workflow = "principalSet://iam.googleapis.com/projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/attribute.workflow"
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

# The apply account from terraform.yml alone, on apply_branches.
resource "google_service_account_iam_member" "apply_from_workflow" {
  for_each = toset(var.apply_branches)

  service_account_id = google_service_account.apply.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "${local.workflow}/${var.github_subject}:ref:refs/heads/${each.key}:${var.github_repository}/.github/workflows/terraform.yml@refs/heads/${each.key}"

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

# editor holds no setIamPolicy at all, and projectIamAdmin only the project's own, so apply could create a secret,
# the image repository or a bucket but not grant access to it. securityAdmin adds the setIamPolicy of single
# resources without any data access: it cannot read a secret value or an object (checked 2026-11-04). It adds no
# real power either, since projectIamAdmin could already grant apply any role.
resource "google_project_iam_member" "apply" {
  for_each = toset(["roles/editor", "roles/resourcemanager.projectIamAdmin", "roles/iam.securityAdmin"])

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
