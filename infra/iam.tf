# -----------------------------------------------------------------------------
# Service accounts — one per workload, each scoped to least privilege.
# -----------------------------------------------------------------------------

resource "google_service_account" "training" {
  account_id   = "${var.name_prefix}-training"
  display_name = "Training — Vertex AI Custom Training + Pipelines"
  description  = "Runs DistilBERT fine-tuning and orchestrated pipelines"
  project      = var.project_id
}

resource "google_service_account" "serving" {
  account_id   = "${var.name_prefix}-serving"
  display_name = "Serving — Cloud Run inference"
  description  = "Reads model artifacts, writes prediction audit rows"
  project      = var.project_id
}

resource "google_service_account" "labeling" {
  account_id   = "${var.name_prefix}-labeling"
  display_name = "Labeling — Gemini teacher labeling"
  description  = "Calls Gemini on Vertex AI to label training data"
  project      = var.project_id
}

# -----------------------------------------------------------------------------
# Resource-scoped IAM bindings (preferred — narrower blast radius).
# -----------------------------------------------------------------------------

# Artifacts bucket: training + labeling read/write; serving read-only.
resource "google_storage_bucket_iam_member" "training_artifacts_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.training.email}"
}

resource "google_storage_bucket_iam_member" "labeling_artifacts_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.labeling.email}"
}

resource "google_storage_bucket_iam_member" "serving_artifacts_viewer" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.serving.email}"
}

# Audit dataset: training + serving write rows.
resource "google_bigquery_dataset_iam_member" "training_audit_editor" {
  dataset_id = google_bigquery_dataset.audit.dataset_id
  project    = var.project_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.training.email}"
}

resource "google_bigquery_dataset_iam_member" "serving_audit_editor" {
  dataset_id = google_bigquery_dataset.audit.dataset_id
  project    = var.project_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.serving.email}"
}

# Artifact Registry: training pulls images for jobs.
resource "google_artifact_registry_repository_iam_member" "training_registry_reader" {
  location   = google_artifact_registry_repository.docker.location
  repository = google_artifact_registry_repository.docker.repository_id
  project    = var.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.training.email}"
}

# -----------------------------------------------------------------------------
# Project-level bindings — only where GCP's IAM model doesn't support narrower
# scoping. We grant exactly one narrow role each; never roles/editor or
# roles/owner.
# -----------------------------------------------------------------------------

# Vertex AI / Gemini: aiplatform.user is project-scoped only.
resource "google_project_iam_member" "training_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.training.email}"
}

resource "google_project_iam_member" "labeling_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.labeling.email}"
}

# Cloud Run observability: logs and metrics writers are project-scoped.
resource "google_project_iam_member" "serving_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.serving.email}"
}

resource "google_project_iam_member" "serving_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.serving.email}"
}
