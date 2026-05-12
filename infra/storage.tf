locals {
  artifacts_bucket_name = "${var.project_id}-${var.name_prefix}-${var.env}-artifacts"
}

resource "google_storage_bucket" "artifacts" {
  name     = local.artifacts_bucket_name
  location = var.region
  project  = var.project_id

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 30
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    env     = var.env
    purpose = "ml-artifacts"
  }
}
