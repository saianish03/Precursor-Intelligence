variable "project_id" {
  type = string
}

variable "project_number" {
  type = string
}

variable "github_repository" {
  description = "owner/name of the only repository allowed to log in."
  type        = string
}

variable "apply_branches" {
  description = "Branches whose workflow runs may act as the apply account."
  type        = list(string)
}

variable "state_bucket_name" {
  type = string
}
