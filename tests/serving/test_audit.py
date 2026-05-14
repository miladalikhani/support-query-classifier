"""Tests for src/serving/audit.py — interface + NoopAuditWriter.

The BigQuery-backed AuditWriter is exercised in [P6][T3]'s tests; here we
only validate the shared dataclass and the no-op default.
"""

from dataclasses import fields
from datetime import UTC, datetime

from src.serving.audit import AuditRow, AuditWriter, NoopAuditWriter


def _make_row() -> AuditRow:
    return AuditRow(
        request_id="r-1",
        timestamp=datetime.now(UTC),
        model_version="test-v1",
        input_text_redacted="hello",
        predicted_intent="card_arrival",
        confidence=0.91,
        top_k_intents=["card_arrival", "lost_or_stolen_card"],
        top_k_confidences=[0.91, 0.04],
        latency_ms=12.3,
    )


def test_audit_row_has_all_bigquery_schema_fields() -> None:
    """Field names must match the BQ schema in infra/bigquery.tf exactly."""
    bq_schema_fields = {
        "request_id",
        "timestamp",
        "model_version",
        "input_text_redacted",
        "predicted_intent",
        "confidence",
        "top_k_intents",
        "top_k_confidences",
        "latency_ms",
    }
    actual = {f.name for f in fields(AuditRow)}
    assert actual == bq_schema_fields


def test_audit_row_is_frozen() -> None:
    """Caller mutation must not leak into the audit writer's buffer."""
    import dataclasses

    row = _make_row()
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        row.predicted_intent = "x"  # type: ignore[misc]


def test_noop_audit_writer_does_not_raise() -> None:
    """The default writer must accept any row silently."""
    writer = NoopAuditWriter()
    writer.write(_make_row())


def test_noop_audit_writer_satisfies_protocol() -> None:
    """NoopAuditWriter is a structural match for the AuditWriter Protocol."""
    writer: AuditWriter = NoopAuditWriter()
    writer.write(_make_row())
