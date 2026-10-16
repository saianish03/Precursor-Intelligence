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

variable "apply_branches" {
  description = "Merges to these branches run `terraform apply`."
  type        = list(string)
  default     = ["dev", "main"]
}
