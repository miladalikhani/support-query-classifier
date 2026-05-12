# Solution Design: Intent Classification via LLM Distillation

**Project:** Meridian Bank — customer support query routing
**Document type:** Design and scope
**Companion document:** `problem_brief.md`

---

## 1. Approach in one paragraph

Use a frontier LLM (the "teacher") to label a large slice of historical chat data, then fine-tune a small encoder model (the "student") on those labels. The student is what gets deployed; the teacher is never called at inference time. This addresses the brief's three core constraints simultaneously: cost (small model on CPU is ~100-1000x cheaper per inference), latency (sub-100ms vs 1-5s for a remote LLM call), and data residency (no external API calls at serve time). The trade-off is some accuracy loss between teacher and student, which is measured and reported honestly.

## 2. Architecture

```
                ┌─────────────────────────────────────────────────────────────┐
                │  TRAINING TIME (offline, pipeline-driven, runs as needed)   │
                └─────────────────────────────────────────────────────────────┘

  Raw data  ──►  PII redact  ──►  Teacher labeling  ──►  Train student  ──►  Evaluate  ──►  Register
  (GCS)          (DLP/regex)      (Gemini on Vertex)     (DistilBERT)        (vs golden)    (Vertex AI
                                                                                            Model Registry)

                ┌─────────────────────────────────────────────────────────────┐
                │  INFERENCE TIME (online, customer-facing, scale-to-zero)    │
                └─────────────────────────────────────────────────────────────┘

  Chat message  ──►  PII redact  ──►  Cloud Run service  ──►  Student model  ──►  {intent, confidence, top_k}
                                      (FastAPI + temp.                            │
                                       scaling)                                   ▼
                                                                            Audit log
                                                                           (BigQuery)
```

The teacher operates only on historical data at training time. The student runs in a stateless Cloud Run container at inference time. Every prediction is logged to BigQuery for audit and future retraining signal.

## 3. Scope

### In scope

- End-to-end working prototype on Banking77 as a proxy dataset (77 intent classes, ~13k examples — closely matches the brief's "70-80 categories")
- Vertex AI Pipeline orchestrating: label → train → evaluate → register
- Deployed inference service on Cloud Run with a documented API
- Calibration via temperature scaling, applied at serve time
- Evaluation harness covering accuracy, per-class F1, calibration, latency, and cost
- Embedding-based baseline (sentence embeddings + logistic regression) as comparison
- BigQuery audit logging for predictions
- PII handling discussion with a working regex-based redactor in the serving path
- Terraform for core GCP infrastructure

### Explicitly out of scope (acknowledged, deferred)

- Soft-label distillation (matching teacher logits) — using hard-label distillation
- Active learning or selective labeling — labeling a fixed stratified sample
- Multi-language support — Banking77 is English-only; brief defers other languages
- Cloud DLP integration — using regex redaction as a stand-in; production would use DLP
- Production hardening: authentication, rate limiting, autoscaling tuning, full monitoring
- Feature Store, DVC, or any heavy data versioning beyond GCS paths
- CI/CD pipeline for code — clean repo with tests, but no GitHub Actions
- Drift detection and automated retraining triggers
- A/B testing or canary deployment infrastructure
- Distilled small LLM (e.g. Gemma 2B fine-tuned) — overkill for 77-way classification; mentioned for completeness

### Stand-in mapping (prototype vs real engagement)

| Brief says | Prototype uses | Note |
|---|---|---|
| Bank's historical chat data | Banking77 dataset | Public proxy with same task, same domain, similar scale |
| Frontier LLM (flagship commercial model) | Gemini 2.5 Flash on Vertex | Cheaper than a true frontier model; pattern is identical, cost-savings ratio in a real engagement would be larger |
| ~80 internal intent categories | 77 Banking77 categories | Effectively the same scale |
| 30k messages/day production volume | Test set + load test sample | Demonstrates per-query cost/latency; full-scale validation deferred |
| Cloud DLP for PII handling | Regex redaction | Same architectural position in the pipeline; production swap is mechanical |

## 4. Design decisions

### 4.1 Student model: fine-tuned small encoder

**Choice:** DistilBERT (`distilbert-base-uncased`), fine-tuned with a classification head.

**Considered:**
- *Embedding + linear classifier* (sentence-transformer + logistic regression): lower accuracy ceiling, but extremely cheap and has a useful property — only the head needs retraining when the taxonomy changes. Building this as a secondary baseline.
- *Distilled small LLM* (e.g., Gemma 2B): overkill for 77-way classification; the latency/cost story gets worse without an obvious accuracy win for this task shape.

**Rationale:** Best accuracy/cost trade-off for short-text intent classification. ~66M parameters, fine-tunes in 15-30 minutes on a T4, runs comfortably on CPU at inference time. Well-supported tooling (HuggingFace).

### 4.2 Teacher labeling strategy

**Choice:** Gemini 2.5 Flash on Vertex AI. Label a stratified sample of the training data via structured-output prompts. **Deliberately ignore the dataset's gold labels during labeling** — the whole point is to simulate the real-world scenario where the client doesn't have labels.

**Why this matters:** If we used the gold labels for training, we wouldn't be doing distillation — we'd be doing supervised learning. The interesting measurement is the *distillation gap*: how much accuracy the student loses relative to the teacher. By holding out the gold labels as a final evaluation set, we can measure:
- Teacher accuracy (teacher vs gold on test set) — sets the ceiling
- Student accuracy (student vs gold on test set) — what we ship
- Student vs teacher agreement — the distillation gap itself

**Details:**
- Prompt the teacher with the 77 class names plus short natural-language descriptions
- Use Gemini's JSON mode for guaranteed parseable output
- On the gold-labeled test set, measure teacher accuracy first to validate the prompt before labeling at scale
- Stratified sample of ~3-5k training examples (predictable cost, manageable runtime)

### 4.3 Calibration: temperature scaling at serve time

**Choice:** Apply post-hoc temperature scaling on a held-out validation set, save the temperature parameter alongside the model, apply it at inference time before returning confidence.

**Why this isn't optional:** The brief asks for a "confidence signal" to drive routing decisions ("high-confidence → auto-route, low-confidence → human triage"). Raw softmax probabilities from a fine-tuned transformer are systematically overconfident. Without calibration, the threshold isn't meaningful — you'd be routing on noise.

Temperature scaling is the right technique here: it's a single-parameter fix, it doesn't change the predicted class, it's fit in seconds on a validation set, and it materially improves Expected Calibration Error (ECE) on classification tasks.

### 4.4 Confidence-based routing: per-class thresholds

**Choice:** Recommend per-class confidence thresholds rather than a single global threshold.

**Rationale:** With 77 imbalanced classes, a single threshold is the wrong shape. Some intents are easy and the model is reliably confident; some are inherently ambiguous (e.g., `transfer_not_received_by_recipient` vs `pending_transfer`) and the model's confidence on them is lower even when correct. A global threshold either over-routes ambiguous intents to humans (under-utilizing the model) or under-routes confident ones (introducing errors).

Per-class thresholds are calibrated on the validation set to hit a target precision per class. Long-tail classes (those with very few training examples) naturally end up with higher thresholds, which is the right behavior — when in doubt, route to a human.

### 4.5 Long-tail strategy

**Choice:** Higher confidence thresholds for low-support classes; accept they route to humans more often. No class weighting in the loss, no oversampling, no hierarchical classification.

**Rationale:** The simplest answer that's honest about what's happening. Class-weighted loss and oversampling both risk degrading the head classes for marginal gains on the tail. Hierarchical classification adds taxonomy design complexity and another set of decisions to defend. Letting the routing logic handle it — via thresholds — keeps the model simple and pushes the policy decision (how often is "ask a human" acceptable?) to the right layer.

This is also what production teams actually do.

### 4.6 PII handling

**Choice:** Redact before the model sees the input. Use a regex-based redactor in the prototype (account numbers, card numbers, emails, phone numbers); architect for a Cloud DLP swap in production.

**The decision worth defending:** Redact-before-inference vs redact-before-storage.

- *Redact-before-inference* (chosen): customer message is scrubbed before it reaches the model. Slightly hurts accuracy (the model loses some signal). Safest posture: no PII in model inputs, no PII in audit logs, no PII anywhere downstream.
- *Redact-before-storage only*: model sees raw text (slightly better accuracy), but PII lives in serving memory and risks ending up in logs and audit trails.

For a banking client, the safer posture wins. Redact early.

### 4.7 Audit logging

**Choice:** Every prediction writes a row to BigQuery via a fire-and-forget async write from the serving container. Schema: `request_id, timestamp, model_version, input_text_redacted, predicted_intent, confidence, top_k_intents, top_k_confidences, latency_ms`.

**Why BigQuery:** The brief calls this out as non-negotiable. BigQuery handles the write volume trivially, gives us SQL for ad-hoc analysis, integrates with Looker Studio for monitoring dashboards, and is the natural home for drift analysis later. Cheap to bolt on now; painful to retrofit.

The async write means audit logging doesn't add to user-facing latency.

### 4.8 Serving: Cloud Run

**Choice:** Cloud Run, CPU-only, `min_instances=0` (scale-to-zero).

**Considered and rejected:** Vertex AI Endpoints. No scale-to-zero. Always-on cost is the most common way GCP ML demos rack up surprise bills, and there's no benefit for CPU inference on a small encoder.

**Acknowledged limitation:** Cold start on a transformer container is 5-15 seconds when scaled to zero. In production, `min_instances=1` during business hours fixes this. For the prototype this is acceptable but called out honestly.

### 4.9 Orchestration: Vertex AI Pipelines

**Choice:** Kubeflow Pipelines via the Vertex AI Pipelines runner. Components for: load data → redact → label (teacher) → train (student) → evaluate → conditional register.

**Rationale:** Native GCP orchestration, reproducible runs, automatic lineage tracking, integrates with Model Registry and Experiments. This is the on-brand choice for a Datatonic engagement.

**Conditional registration:** The pipeline only registers a new model version if evaluation metrics clear thresholds (e.g., top-1 accuracy within X percentage points of the previous version). Prevents accidentally promoting a degraded model.

### 4.10 Evaluation harness — separate from training

**Choice:** Evaluation lives in its own module, owns the locked golden test set, and reports a fixed set of metrics for any model version.

**Why separated:** The training code should not be able to see, touch, or accidentally leak the golden eval set. Keeping evaluation as an independent module with its own data ownership is the cleanest enforcement.

**Metrics reported:**

| Metric | Why |
|---|---|
| Top-1 accuracy (overall and per-class) | Primary; per-class catches tail degradation |
| Top-5 accuracy | Useful for "suggest top-3 to agent" UX |
| Macro F1 | 77 imbalanced classes; macro F1 weighs all classes equally |
| Per-class confusion matrix | Surfaces ambiguous pairs for analysis |
| Expected Calibration Error (ECE) | Validates that confidence is meaningful |
| Reliability diagram | Visual companion to ECE |
| Inference latency (P50, P95, P99) | Measured on Cloud Run with realistic payloads |
| Cost per 1k inferences | Computed explicitly: teacher vs student |

**Three models compared head-to-head:**
1. Teacher (Gemini Flash) — sets the ceiling
2. Student (fine-tuned DistilBERT) — primary deliverable
3. Baseline (embeddings + logistic regression) — sanity check and architectural alternative

### 4.11 Taxonomy stability

**Design property worth naming:** The architecture supports stable encoder + retrainable head. If the bank adds new intent categories (new products, etc.), only the classification head needs retraining — the embedding encoder is unchanged. This makes the embedding+LR baseline more than a sanity check: it's a viable production option specifically when taxonomy drift is frequent.

The fine-tuned DistilBERT path requires full retraining for taxonomy changes. Worth flagging as a trade-off.

## 5. Trade-offs worth a real conversation

These are the trade-offs an informed stakeholder would push on, and the answers I'd give.

**Teacher labels vs hand labels.**
The teacher is cheap and plentiful but inherits its own biases — especially on the long tail. Right answer: teacher for training, hand-labeled golden set for evaluation (never touches training). This catches "the student inherited the teacher's mistakes" because errors that exist in both teacher and student show up against the gold labels but not against teacher-derived metrics alone.

**Accuracy vs cost vs latency.**
The brief implies these are independent constraints. They're not. Pushing the student smaller (TinyBERT, MiniLM) buys latency and cost at some accuracy cost. Going bigger (full BERT-base, RoBERTa) buys accuracy at latency cost. DistilBERT is the conventional sweet spot; the evaluation will show the actual numbers and where each lands.

**Calibration is doing real work.**
A model that's 92% accurate but uncalibrated is worse for routing than a 90% accurate well-calibrated model. The cost of bad calibration is *invisible* — it shows up as worse routing decisions downstream, not as a metric the model team owns. Worth being insistent about.

**The 300ms latency budget is loose.**
A quantized DistilBERT on CPU should hit sub-50ms inference, leaving 250ms for network, preprocessing, and audit logging. This constraint is less binding than it looks. Worth flagging because it means we have headroom — for example, we could afford to redact-before-inference rather than skipping redaction for latency reasons.

**Static vs evolving taxonomy.**
The brief mentions drift. If the taxonomy changes frequently, the embedding+LR baseline is more attractive (cheap retraining of just the head). If it's stable, fine-tuned DistilBERT wins on accuracy. Worth surfacing this as a discussion with the client.

## 6. Tech stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Standard for ML; required by training/serving stack |
| Dependency management | `uv` | Fast, reproducible, modern |
| ML framework | PyTorch + HuggingFace Transformers | Standard, well-supported on Vertex |
| Embedding baseline | `sentence-transformers` | Drop-in encoders, pre-trained |
| Dataset access | HuggingFace `datasets` | Direct Banking77 access |
| API serving | FastAPI + Uvicorn | Lightweight, fast, async |
| Container | Docker (slim Python base) | Standard for Cloud Run |
| Pipeline SDK | `kfp` + Vertex AI SDK | Native to Vertex Pipelines |
| Teacher inference | `google-genai` SDK | Vertex AI Gemini access |
| Config | `pydantic` + YAML | Typed, validated |
| Logging | `structlog` | Structured JSON for Cloud Logging |
| Audit writes | `google-cloud-bigquery` | Async streaming inserts |
| Testing | `pytest` | Standard |
| Code quality | `ruff` (lint + format), `mypy` | Demonstrates engineering rigor |
| IaC | Terraform | Core GCP resources |

| GCP service | Used for |
|---|---|
| Cloud Storage (GCS) | Raw data, labeled data, model artifacts |
| Vertex AI Pipelines | End-to-end orchestration |
| Vertex AI Gemini API | Teacher labeling |
| Vertex AI Custom Training | Student training (T4 GPU) |
| Vertex AI Model Registry | Model versioning |
| Vertex AI Experiments | Metric tracking per run |
| Cloud Run | Serving (scale-to-zero) |
| Artifact Registry | Docker images |
| BigQuery | Audit log of predictions |
| Cloud Logging | Application logs |

## 7. Repository structure

```
project/
├── README.md
├── pyproject.toml
├── Dockerfile
├── Makefile
│
├── src/
│   ├── data/                  # Banking77 loading, golden set management
│   ├── pii/                   # Redaction (regex; DLP-swappable interface)
│   ├── labeling/              # Teacher labeling logic, prompt design
│   ├── training/              # Fine-tuning, temperature scaling
│   ├── evaluation/            # Metrics, calibration, comparison harness
│   ├── serving/               # FastAPI app, audit-log writer
│   └── pipelines/             # Vertex AI Pipeline definitions
│
├── configs/                   # YAML configs for hyperparams, thresholds
├── tests/                     # pytest unit tests
├── infra/                     # Terraform
└── notebooks/                 # Dev-only exploration, not delivered
```

## 8. What production would add (beyond this prototype)

For the "what would you finish if this were a real engagement" conversation:

- **Cloud DLP** in place of regex redaction; tested coverage across PII types relevant to banking
- **Authentication** in front of the Cloud Run endpoint (IAP, API Gateway, or service-to-service IAM)
- **Monitoring + alerting**: latency SLOs, error rate, prediction distribution drift, audit log lag
- **Automated drift detection**: compare current-week prediction distribution against training distribution; alert on KL divergence above threshold
- **Retraining triggers**: scheduled (e.g., monthly) plus drift-triggered; gated by golden-set evaluation
- **A/B testing**: traffic splitting in Cloud Run between model versions; statistical comparison on routing outcomes
- **Soft-label distillation**: if accuracy needs another push, request teacher logits via a top-k probabilities prompt
- **Active labeling**: when the model is unconfident on a new example, prioritize it for human labeling; use as future training data
- **CI/CD**: GitHub Actions for tests, linting, image builds, Terraform plan/apply gates
- **Cost monitoring**: budgets and alerts per environment
- **Min-instances during business hours** to eliminate cold-start latency for real traffic

---

*Decisions made during implementation that diverge from this design should be recorded with a brief rationale.*