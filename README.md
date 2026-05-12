# Support Query Classifier

Intent classification via LLM distillation, for Meridian Bank's customer support
query routing.

> **Status:** Prototype — under active development. See the Trello board for
> the current task breakdown.

A frontier LLM (Gemini 2.5 Flash, the "teacher") labels historical chat data
offline. A small encoder model (DistilBERT, the "student") is fine-tuned on
those labels and deployed behind a FastAPI service on Cloud Run. The teacher
is never called at inference time. The architecture trades a small amount of
accuracy for ~100× lower per-query cost, sub-100ms inference, and no external
API calls in the serving path.

## Read first

- [`problem_brief.md`](problem_brief.md) — the client's brief
- [`design_doc.md`](design_doc.md) — proposed approach, design decisions, and
  trade-offs

## Repository layout

```
src/
  data/         Banking77 loader, golden set management
  pii/          Regex redaction (DLP-swappable interface)
  labeling/     Teacher labeling via Gemini on Vertex AI
  training/     DistilBERT fine-tune + embedding+LR baseline + temperature scaling
  evaluation/   Metrics, calibration, three-way comparison harness
  serving/      FastAPI inference app + BigQuery audit writer
  pipelines/    Vertex AI Pipeline definitions
configs/        YAML configs (hyperparams, thresholds, prompts)
tests/          pytest unit tests
infra/          Terraform for GCP infrastructure
notebooks/      Dev-only exploration (not delivered)
```

## Prerequisites

| Tool         | Purpose                          | Install                                                   |
| ------------ | -------------------------------- | --------------------------------------------------------- |
| Python 3.11  | Project language                 | `pyenv`, `asdf`, or `brew install python@3.11`            |
| `uv`         | Dependency + venv management     | `brew install uv`                                         |
| Docker       | Build/run the serving image      | `brew install colima docker docker-buildx`                |
| `terraform`  | Provision GCP infrastructure     | `releases.hashicorp.com/terraform` (≥ 1.10)               |
| `gcloud`     | Authenticate to GCP              | `brew install --cask gcloud-cli`                          |

A GCP project with billing enabled is required for `make tf-*` targets.

## Quickstart

```bash
# 1. Install Python deps (uv creates .venv/ automatically)
uv sync

# 2. Verify the project is healthy
make test    # pytest + coverage
make lint    # ruff check + mypy

# 3. Run the placeholder serving app in Docker
colima start                    # if not already running
make docker-build
make docker-run                 # serves on :8080
curl http://localhost:8080/healthz   # → {"status":"ok"}
```

`make help` lists all available targets.

## Infrastructure

GCP resources are managed declaratively under [`infra/`](infra/). After
authenticating to your project, run:

```bash
gcloud auth login                              # human session
gcloud auth application-default login          # ADC for Terraform/SDKs
gcloud config set project YOUR_PROJECT_ID

cd infra
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars                       # set project_id, region, env

# from repo root:
make tf-init
make tf-plan
make tf-apply
```

What gets provisioned (see `infra/*.tf` for specifics):

- GCS artifacts bucket with versioning + lifecycle pruning
- BigQuery audit dataset with a `predictions` table (partitioned, clustered)
- Artifact Registry Docker repository
- Eight required Google Cloud APIs
- Three per-workload service accounts (`sqc-training`, `sqc-serving`,
  `sqc-labeling`) with least-privilege resource-scoped IAM

## Cost note

Empty Phase 1 infrastructure costs ≈ $0/month. Real spend starts when later
phases push data, call Gemini, and run training jobs. Set a project-level
budget alert before scaling up — see
[`design_doc.md`](design_doc.md) §8 "Cost monitoring" for the production
hardening checklist.
