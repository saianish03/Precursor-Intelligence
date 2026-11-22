output "provider_name" {
  description = "Value for the workflow's workload_identity_provider."
  value       = google_iam_workload_identity_pool_provider.github_repo.name
}

output "plan_service_account" {
  value = google_service_account.plan.email
}

output "apply_service_account" {
  value = google_service_account.apply.email
}

output "workflow_principal_set" {
  description = "principalSet:// prefix of attribute.workflow; append /<subject>:ref:refs/heads/<branch>:<job_workflow_ref>."
  value       = local.workflow
}

output "subject_prefix" {
  description = "principal:// prefix of this repository's GitHub token subjects; append :ref:refs/heads/<branch> or :pull_request."
  value       = local.subject
}
