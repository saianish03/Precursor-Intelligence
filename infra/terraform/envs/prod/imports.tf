# Adopts the resources created by hand in docs/runbooks/gcp-bootstrap.md, so the first apply changes nothing.
# Each block does nothing once its resource is in the state; they can be deleted after the first apply.

import {
  to = module.data_bucket.google_storage_bucket.this
  id = "precursor-intel-mlops-data"
}

import {
  to = module.state_bucket.google_storage_bucket.this
  id = "precursor-intel-mlops-tfstate"
}

import {
  to = google_service_account.pipeline
  id = "projects/precursor-intel-mlops/serviceAccounts/sa-pipeline@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = google_storage_bucket_iam_member.pipeline_objects
  id = "b/precursor-intel-mlops-data roles/storage.objectAdmin serviceAccount:sa-pipeline@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = google_storage_bucket_iam_member.pipeline_bucket_reader
  id = "b/precursor-intel-mlops-data roles/storage.legacyBucketReader serviceAccount:sa-pipeline@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_iam_workload_identity_pool.github
  id = "projects/precursor-intel-mlops/locations/global/workloadIdentityPools/github"
}

import {
  to = module.wif.google_iam_workload_identity_pool_provider.github_repo
  id = "projects/precursor-intel-mlops/locations/global/workloadIdentityPools/github/providers/github-repo"
}

import {
  to = module.wif.google_service_account.plan
  id = "projects/precursor-intel-mlops/serviceAccounts/sa-tf-plan@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_service_account.apply
  id = "projects/precursor-intel-mlops/serviceAccounts/sa-tf-apply@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_service_account_iam_member.plan_from_pull_request
  id = "projects/precursor-intel-mlops/serviceAccounts/sa-tf-plan@precursor-intel-mlops.iam.gserviceaccount.com roles/iam.workloadIdentityUser principal://iam.googleapis.com/projects/852874974077/locations/global/workloadIdentityPools/github/subject/repo:saianish03@89864897/Precursor-Intelligence@1402453226:pull_request"
}

import {
  to = module.wif.google_service_account_iam_member.apply_from_branch["dev"]
  id = "projects/precursor-intel-mlops/serviceAccounts/sa-tf-apply@precursor-intel-mlops.iam.gserviceaccount.com roles/iam.workloadIdentityUser principal://iam.googleapis.com/projects/852874974077/locations/global/workloadIdentityPools/github/subject/repo:saianish03@89864897/Precursor-Intelligence@1402453226:ref:refs/heads/dev"
}

import {
  to = module.wif.google_service_account_iam_member.apply_from_branch["main"]
  id = "projects/precursor-intel-mlops/serviceAccounts/sa-tf-apply@precursor-intel-mlops.iam.gserviceaccount.com roles/iam.workloadIdentityUser principal://iam.googleapis.com/projects/852874974077/locations/global/workloadIdentityPools/github/subject/repo:saianish03@89864897/Precursor-Intelligence@1402453226:ref:refs/heads/main"
}

import {
  to = module.wif.google_project_iam_member.plan["roles/viewer"]
  id = "precursor-intel-mlops roles/viewer serviceAccount:sa-tf-plan@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_project_iam_member.plan["roles/iam.securityReviewer"]
  id = "precursor-intel-mlops roles/iam.securityReviewer serviceAccount:sa-tf-plan@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_project_iam_member.apply["roles/editor"]
  id = "precursor-intel-mlops roles/editor serviceAccount:sa-tf-apply@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_project_iam_member.apply["roles/resourcemanager.projectIamAdmin"]
  id = "precursor-intel-mlops roles/resourcemanager.projectIamAdmin serviceAccount:sa-tf-apply@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_storage_bucket_iam_member.state["plan"]
  id = "b/precursor-intel-mlops-tfstate roles/storage.objectViewer serviceAccount:sa-tf-plan@precursor-intel-mlops.iam.gserviceaccount.com"
}

import {
  to = module.wif.google_storage_bucket_iam_member.state["apply"]
  id = "b/precursor-intel-mlops-tfstate roles/storage.objectAdmin serviceAccount:sa-tf-apply@precursor-intel-mlops.iam.gserviceaccount.com"
}
