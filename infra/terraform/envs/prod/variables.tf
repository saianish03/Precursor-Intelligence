variable "project_id" {
  type    = string
  default = "precursor-intel-mlops"
}

variable "project_number" {
  type    = string
  default = "852874974077"
}

variable "region" {
  description = "One region for every resource, so data and compute never cross regions."
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  type    = string
  default = "saianish03/Precursor-Intelligence"
}

variable "github_subject" {
  description = "Token subject prefix, with GitHub's numeric ids (owner 89864897, repository 1402453226). Read the real value from a run's login claims if it ever changes."
  type        = string
  default     = "repo:saianish03@89864897/Precursor-Intelligence@1402453226"
}

variable "apply_branches" {
  description = "Merges to these branches run `terraform apply`."
  type        = list(string)
  default     = ["dev", "main"]
}

variable "zone" {
  description = "The VM and its data disk; a disk can only attach to a VM in its own zone."
  type        = string
  default     = "us-central1-a"
}

variable "vm_machine_type" {
  description = "Spec §14 decision 26: e2-highmem-4 (4 vCPU, 32 GB) since T25 (2026-11-20), whose full-archive gold was OOM-killed at 10.46 GB under the 16 GB e2-standard-4's 10g task limit; gold holds the whole dataset in memory. Change it only while the VM is stopped."
  type        = string
  default     = "e2-highmem-4"
}
