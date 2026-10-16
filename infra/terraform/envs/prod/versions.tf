terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # The state lives in a bucket created by hand (docs/runbooks/gcp-bootstrap.md). Each environment uses its
  # own prefix, and `plan` and `apply` take a lock on it, so two runs cannot write at once.
  backend "gcs" {
    bucket = "precursor-intel-mlops-tfstate"
    prefix = "prod"
  }
}
