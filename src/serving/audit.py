"""Audit-writer interface and a no-op implementation.

`AuditRow` is the in-process shape of one row written to the BigQuery
`predictions` table per inference call (schema per design_doc.md §4.7).
The Protocol decouples the FastAPI handler from any specific writer; the
production-grade BigQuery-backed implementation is added separately in
[P6][T3]. The `NoopAuditWriter` here is the safe default for dev and
tests — it logs the call but does not persist.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class AuditRow:
    """One BigQuery predictions row.

    Field names and types mirror the table schema provisioned in
    `infra/bigquery.tf`; the audit writer is expected to write the row
    verbatim with no transformation.
    """

    request_id: str
    timestamp: datetime
    model_version: str
    input_text_redacted: str
    predicted_intent: str
    confidence: float
    top_k_intents: list[str]
    top_k_confidences: list[float]
    latency_ms: float


@runtime_checkable
class AuditWriter(Protocol):
    """Anything that can persist an `AuditRow` without blocking the caller."""

    def write(self, row: AuditRow) -> None: ...


class NoopAuditWriter:
    """Logs the row instead of writing it. Default for dev runs."""

    def write(self, row: AuditRow) -> None:
        log.info(
            "audit_row_skipped",
            request_id=row.request_id,
            predicted_intent=row.predicted_intent,
            model_version=row.model_version,
        )
