import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from src.pii import NoopRedactor, RedactionResult, RedactionSpan, Redactor


def test_noop_redactor_satisfies_protocol() -> None:
    assert isinstance(NoopRedactor(), Redactor)


def test_noop_redactor_returns_input_unchanged() -> None:
    result = NoopRedactor().redact("hello world")
    assert result.redacted_text == "hello world"
    assert result.spans == ()


def test_redaction_span_is_frozen() -> None:
    span = RedactionSpan(start=0, end=5, kind="email", original_hash="abc123")
    with pytest.raises(FrozenInstanceError):
        span.start = 10  # type: ignore[misc]


def test_redaction_span_is_hashable_and_set_friendly() -> None:
    a = RedactionSpan(start=0, end=5, kind="email", original_hash="hash1")
    b = RedactionSpan(start=0, end=5, kind="email", original_hash="hash1")
    c = RedactionSpan(start=0, end=5, kind="phone", original_hash="hash1")
    assert hash(a) == hash(b)
    assert {a, b, c} == {a, c}


def test_redaction_result_round_trips_through_json() -> None:
    """Phase 6 will serialize this structure to BigQuery via JSON."""
    result = RedactionResult(
        redacted_text="hi [EMAIL]",
        spans=(RedactionSpan(start=3, end=8, kind="email", original_hash="hash1"),),
    )
    payload = json.loads(json.dumps(asdict(result)))
    assert payload == {
        "redacted_text": "hi [EMAIL]",
        "spans": [{"start": 3, "end": 8, "kind": "email", "original_hash": "hash1"}],
    }
