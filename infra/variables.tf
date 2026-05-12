variable "project_id" {
  description = "GCP project ID for the prototype"
  type        = string
}

variable "region" {
  description = "GCP region for regional resources (Cloud Run, Artifact Registry, etc.)"
  type        = string
  default     = "us-central1"
}

variable "env" {
  description = "Deployment environment label (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "name_prefix" {
  description = "Prefix applied to resource names to keep them human-readable and namespaced"
  type        = string
  default     = "sqc"
}
