resource "google_artifact_registry_repository" "docker" {
  repository_id = "${var.name_prefix}-docker"
  location      = var.region
  project       = var.project_id
  format        = "DOCKER"
  description   = "Docker images for the ${var.name_prefix} serving stack"

  cleanup_policies {
    id     = "keep-most-recent-10"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }

  cleanup_policies {
    id     = "delete-everything-else"
    action = "DELETE"
    condition {
      tag_state = "ANY"
    }
  }

  labels = {
    env     = var.env
    purpose = "ml-serving"
  }

  depends_on = [google_project_service.enabled]
}
