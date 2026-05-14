"""FastAPI app for the Cloud Run inference endpoint.

Three endpoints:
  - `POST /predict` — runs the [P6][T1] inference path, schedules a
    fire-and-forget audit write, returns the labelled response.
  - `GET /healthz`  — liveness; cheap, always 200.
  - `GET /readyz`   — readiness; 503 until the model bundle is loaded.

The model and audit writer are loaded once in the FastAPI lifespan and
made available to handlers via dependency-injection. Cloud Run is
configured with one request per container instance so no locking is
needed around module-level state.

Production wiring of the BigQuery audit writer arrives with [P6][T3];
this file ships with a no-op writer that logs but does not persist, so
the service is safe to run end-to-end during development without BQ
credentials.
"""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.pii.regex_redactor import RegexRedactor
from src.serving.audit import AuditRow, AuditWriter, NoopAuditWriter
from src.serving.inference import InferenceService

log = structlog.get_logger()

_MAX_MESSAGE_LEN = 2000

# Module-level state populated by lifespan. Cloud Run runs one request
# at a time per container instance ([P6][T5] sets
# max_instance_request_concurrency=1), so no synchronisation is needed.
_inference_service: InferenceService | None = None
_audit_writer: AuditWriter = NoopAuditWriter()


# ---- Dependencies (overridable in tests) -----------------------------------


def get_inference() -> InferenceService:
    if _inference_service is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _inference_service


def get_audit() -> AuditWriter:
    return _audit_writer


# ---- Request / response models ---------------------------------------------


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_MESSAGE_LEN)


class PredictResponse(BaseModel):
    request_id: str
    predicted_intent: str
    confidence: float
    top_k_intents: list[str]
    top_k_confidences: list[float]
    model_version: str
    latency_ms: float


# ---- Lifespan: load the bundle once ----------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _inference_service
    bundle_uri = os.environ["BUNDLE_GCS_URI"]
    model_version = os.environ.get("MODEL_VERSION", "unknown")
    pii_salt = os.environ["PII_SALT"]
    _inference_service = InferenceService(
        bundle_uri,
        redactor=RegexRedactor(salt=pii_salt.encode("utf-8")),
        model_version=model_version,
    )
    log.info(
        "inference_service_loaded",
        bundle_uri=bundle_uri,
        model_version=model_version,
    )
    yield
    log.info("shutting_down")


app = FastAPI(title="Support Query Classifier", version="0.2.0", lifespan=lifespan)


def _safe_audit_write(audit: AuditWriter, row: AuditRow) -> None:
    """Defense in depth: a buggy writer must never crash the background task.

    The `AuditWriter` contract says implementations swallow their own errors;
    this wrapper is the second line of defence so a contract violation in a
    future writer cannot escalate into a server-side error or kill the
    background-task runner.
    """
    try:
        audit.write(row)
    except Exception as e:
        log.error(
            "audit_write_unhandled_exception",
            error=str(e),
            request_id=row.request_id,
        )


# ---- Routes ----------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(
    _: Annotated[InferenceService, Depends(get_inference)],
) -> dict[str, str]:
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    background: BackgroundTasks,
    inference: Annotated[InferenceService, Depends(get_inference)],
    audit: Annotated[AuditWriter, Depends(get_audit)],
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> PredictResponse:
    request_id = x_request_id or str(uuid.uuid4())
    result = inference.predict(body.message)

    background.add_task(
        _safe_audit_write,
        audit,
        AuditRow(
            request_id=request_id,
            timestamp=datetime.now(UTC),
            model_version=result.model_version,
            input_text_redacted=result.redacted_message,
            predicted_intent=result.predicted_intent,
            confidence=result.confidence,
            top_k_intents=result.top_k_intents,
            top_k_confidences=result.top_k_confidences,
            latency_ms=result.latency_ms,
        ),
    )

    return PredictResponse(
        request_id=request_id,
        predicted_intent=result.predicted_intent,
        confidence=result.confidence,
        top_k_intents=result.top_k_intents,
        top_k_confidences=result.top_k_confidences,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
    )
