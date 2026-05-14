"""Tests for src/serving/inference.py."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.adapters import PredictionBatch
from src.pii.interface import RedactionResult, RedactionSpan
from src.pii.noop import NoopRedactor
from src.serving.inference import (
    InferenceService,
    PredictionResponse,
    _materialize_bundle,
    _parse_gs_uri,
)

NUM_CLASSES = 77


# ---- Fixture bundle (local) ----


def _write_label_maps(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    id_to_label = {i: f"class_{i}" for i in range(NUM_CLASSES)}
    payload = {
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "label_to_id": {v: k for k, v in id_to_label.items()},
    }
    (bundle_dir / "label_maps.json").write_text(json.dumps(payload))


def _make_stub_adapter(seed: int = 0) -> MagicMock:
    """Mocked DistilbertAdapter that returns deterministic top-5 predictions."""
    rng = np.random.default_rng(seed)
    probs_row = rng.dirichlet(np.ones(NUM_CLASSES))
    top_indices = np.argsort(-probs_row)[:5][None, :].astype(np.int64)
    top_probs = probs_row[top_indices[0]][None, :]

    adapter = MagicMock()
    adapter.predict.return_value = PredictionBatch(
        probs=probs_row[None, :],
        top_k_indices=top_indices,
        top_k_probs=top_probs,
        per_example_latency_ms=np.array([10.0]),
        total_wall_time_s=0.01,
    )
    return adapter


def _make_service(
    tmp_path: Path,
    redactor: Any = None,
    seed: int = 0,
) -> InferenceService:
    """Build an InferenceService backed by a stub adapter + label_maps fixture."""
    bundle_dir = tmp_path / "bundle"
    _write_label_maps(bundle_dir)
    stub = _make_stub_adapter(seed=seed)
    with patch(
        "src.serving.inference.DistilbertAdapter", return_value=stub
    ):
        return InferenceService(
            bundle_dir,
            redactor=redactor or NoopRedactor(),
            model_version="test-v1",
        )


# ---- InferenceService.predict ----


def test_predict_returns_documented_response_shape(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    response = service.predict("hello, where is my card?")
    assert isinstance(response, PredictionResponse)
    assert response.predicted_intent.startswith("class_")
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.top_k_intents) == 5
    assert len(response.top_k_confidences) == 5
    assert response.top_k_intents[0] == response.predicted_intent
    assert response.top_k_confidences[0] == response.confidence
    assert response.model_version == "test-v1"


def test_predict_top_k_confidences_descend(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    response = service.predict("a billing question")
    assert all(
        response.top_k_confidences[i] >= response.top_k_confidences[i + 1]
        for i in range(4)
    )


def test_predict_latency_is_populated_and_non_negative(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    response = service.predict("a question")
    assert response.latency_ms >= 0.0


# ---- PII redaction ----


class _FakeRedactor:
    """Replaces every input with a constant token + records the call."""

    def __init__(self) -> None:
        self.called_with: list[str] = []

    def redact(self, text: str) -> RedactionResult:
        self.called_with.append(text)
        return RedactionResult(
            redacted_text="[REDACTED]",
            spans=(RedactionSpan(start=0, end=len(text), kind="all", original_hash="x"),),
        )


def test_predict_redacts_before_handing_to_model(tmp_path: Path) -> None:
    redactor = _FakeRedactor()
    service = _make_service(tmp_path, redactor=redactor)
    response = service.predict("my email is user@example.com")
    assert redactor.called_with == ["my email is user@example.com"]
    assert response.redacted_message == "[REDACTED]"
    # And verify the adapter saw the REDACTED text, not the raw one.
    underlying_adapter = service._adapter  # type: ignore[attr-defined]
    underlying_adapter.predict.assert_called_once_with(["[REDACTED]"])


def test_predict_noop_redactor_passes_text_through_unchanged(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    response = service.predict("plain message without PII")
    assert response.redacted_message == "plain message without PII"


# ---- Bundle materialization ----


def test_parse_gs_uri_with_prefix() -> None:
    assert _parse_gs_uri("gs://my-bucket/models/distilbert/v1") == (
        "my-bucket",
        "models/distilbert/v1",
    )


def test_parse_gs_uri_bucket_only() -> None:
    assert _parse_gs_uri("gs://my-bucket") == ("my-bucket", "")


def test_materialize_local_path_returns_input(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    assert _materialize_bundle(bundle) == bundle


def test_materialize_missing_local_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _materialize_bundle(tmp_path / "does_not_exist")


def test_materialize_gcs_uri_downloads_blobs(tmp_path: Path) -> None:
    """The GCS path lists blobs and writes each one to a temp directory."""

    def _fake_download(local_path: str) -> None:
        Path(local_path).write_text("contents")

    blob_a = MagicMock()
    blob_a.name = "models/distilbert/v1/manifest.json"
    blob_a.download_to_filename.side_effect = _fake_download
    blob_b = MagicMock()
    blob_b.name = "models/distilbert/v1/model.safetensors"
    blob_b.download_to_filename.side_effect = _fake_download

    fake_client = MagicMock()
    fake_client.list_blobs.return_value = [blob_a, blob_b]

    with patch("google.cloud.storage.Client", return_value=fake_client):
        local_dir = _materialize_bundle("gs://my-bucket/models/distilbert/v1")

    assert local_dir.is_dir()
    assert (local_dir / "manifest.json").read_text() == "contents"
    assert (local_dir / "model.safetensors").read_text() == "contents"


def test_materialize_gcs_uri_raises_when_empty(tmp_path: Path) -> None:
    fake_client = MagicMock()
    fake_client.list_blobs.return_value = []
    with (
        patch("google.cloud.storage.Client", return_value=fake_client),
        pytest.raises(FileNotFoundError, match="No objects found"),
    ):
        _materialize_bundle("gs://empty-bucket/nothing")
