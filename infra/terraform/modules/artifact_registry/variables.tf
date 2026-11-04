variable "location" {
  description = "Repository region."
  type        = string
}

variable "repository_id" {
  description = "Repository name, the last part of the Docker URL."
  type        = string
}
