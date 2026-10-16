variable "name" {
  description = "Globally unique bucket name."
  type        = string
}

variable "location" {
  description = "Bucket region."
  type        = string
}

variable "versioning" {
  description = "Keep old object versions, so an overwrite or delete can be recovered."
  type        = bool
  default     = true
}

variable "quarantine_ttl_days" {
  description = "Delete objects under quarantine/ after this many days. Null = no lifecycle rule."
  type        = number
  default     = null
}
