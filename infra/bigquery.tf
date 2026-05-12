resource "google_bigquery_dataset" "audit" {
  dataset_id  = "${var.name_prefix}_${var.env}_audit"
  location    = var.region
  project     = var.project_id
  description = "Audit log of model predictions for ${var.name_prefix} (${var.env})"

  labels = {
    env     = var.env
    purpose = "ml-audit"
  }
}

resource "google_bigquery_table" "predictions" {
  dataset_id          = google_bigquery_dataset.audit.dataset_id
  table_id            = "predictions"
  project             = var.project_id
  deletion_protection = false
  description         = "One row per served prediction. Schema per design_doc.md §4.7"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["model_version"]

  schema = jsonencode([
    { name = "request_id", type = "STRING", mode = "REQUIRED", description = "Unique ID per inference request" },
    { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED", description = "Server time the prediction was made" },
    { name = "model_version", type = "STRING", mode = "REQUIRED", description = "Identifier of the model that produced the prediction" },
    { name = "input_text_redacted", type = "STRING", mode = "NULLABLE", description = "Input text after PII redaction" },
    { name = "predicted_intent", type = "STRING", mode = "REQUIRED", description = "Top-1 intent class" },
    { name = "confidence", type = "FLOAT64", mode = "REQUIRED", description = "Temperature-calibrated softmax probability for the top-1 class" },
    { name = "top_k_intents", type = "STRING", mode = "REPEATED", description = "Top-K intent class labels in confidence order" },
    { name = "top_k_confidences", type = "FLOAT64", mode = "REPEATED", description = "Top-K confidence scores aligned with top_k_intents" },
    { name = "latency_ms", type = "FLOAT64", mode = "NULLABLE", description = "End-to-end serving latency in milliseconds" },
  ])

  labels = {
    env     = var.env
    purpose = "ml-audit"
  }
}
