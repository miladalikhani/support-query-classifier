"""Tests for src/serving/app.py.

We bypass the FastAPI lifespan in tests (no `with TestClient(app):`) so
the app does not try to read env vars or load a real bundle. The
inference and audit dependencies are overridden via FastAPI's
`dependency_overrides`.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app, get_audit, get_inference
from src.serving.audit import AuditRow
from src.serving.inference import PredictionResponse


def _make_prediction(message_version: str = "test-v1") -> PredictionResponse:
    return PredictionResponse(
        predicted_intent="card_arrival",
        confidence=0.92,
        top_k_intents=["card_arrival", "lost_or_stolen_card", "card_delivery_estimate",
                       "card_not_working", "supported_cards_and_currencies"],
        top_k_confidences=[0.92, 0.04, 0.02, 0.01, 0.01],
        redacted_message="when does my card arrive",
        latency_ms=14.3,
        model_version=message_version,
    )


@pytest.fixture
def mocks() -> Iterator[tuple[MagicMock, MagicMock]]:
    """Wire mocked inference + audit dependencies; tear down after the test."""
    mock_inference = MagicMock()
    mock_inference.predict.return_value = _make_prediction()
    mock_audit = MagicMock()
    app.dependency_overrides[get_inference] = lambda: mock_inference
    app.dependency_overrides[get_audit] = lambda: mock_audit
    yield mock_inference, mock_audit
    app.dependency_overrides.clear()


# ---- /healthz ----


def test_healthz_returns_200_unconditionally() -> None:
    """Liveness must not depend on model load state."""
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---- /readyz ----


def test_readyz_503_when_no_model_loaded() -> None:
    """Without an inference override, get_inference raises 503."""
    client = TestClient(app)
    response = client.get("/readyz")
    assert response.status_code == 503


def test_readyz_200_when_model_loaded(mocks: tuple[MagicMock, MagicMock]) -> None:
    client = TestClient(app)
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


# ---- /predict happy path ----


def test_predict_returns_documented_response_shape(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    client = TestClient(app)
    response = client.post("/predict", json={"message": "where is my card?"})
    assert response.status_code == 200
    body = response.json()
    expected_keys = {
        "request_id",
        "predicted_intent",
        "confidence",
        "top_k_intents",
        "top_k_confidences",
        "model_version",
        "latency_ms",
    }
    assert set(body) == expected_keys
    assert body["predicted_intent"] == "card_arrival"
    assert body["confidence"] == pytest.approx(0.92)
    assert len(body["top_k_intents"]) == 5
    assert len(body["top_k_confidences"]) == 5
    assert body["model_version"] == "test-v1"


def test_predict_calls_inference_with_raw_message(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    """The handler forwards the raw message — redaction happens inside InferenceService."""
    mock_inference, _ = mocks
    client = TestClient(app)
    client.post("/predict", json={"message": "my card 4242424242424242"})
    mock_inference.predict.assert_called_once_with("my card 4242424242424242")


# ---- Request validation ----


def test_predict_rejects_missing_message(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    client = TestClient(app)
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_empty_message(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    client = TestClient(app)
    response = client.post("/predict", json={"message": ""})
    assert response.status_code == 422


def test_predict_rejects_message_over_max_length(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    client = TestClient(app)
    response = client.post("/predict", json={"message": "x" * 2001})
    assert response.status_code == 422


# ---- Request ID handling ----


def test_predict_generates_uuid_request_id_when_not_supplied(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    client = TestClient(app)
    a = client.post("/predict", json={"message": "hi"}).json()
    b = client.post("/predict", json={"message": "hi"}).json()
    assert a["request_id"] != b["request_id"]
    # UUID4 is 36 chars with hyphens
    assert len(a["request_id"]) == 36


def test_predict_echoes_inbound_request_id_header(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"message": "hi"},
        headers={"X-Request-ID": "caller-12345"},
    )
    assert response.json()["request_id"] == "caller-12345"


# ---- Audit fire-and-forget ----


def test_predict_schedules_one_audit_write(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_audit = mocks
    client = TestClient(app)
    response = client.post("/predict", json={"message": "ping"})
    assert response.status_code == 200
    mock_audit.write.assert_called_once()
    (row,), _ = mock_audit.write.call_args
    assert isinstance(row, AuditRow)
    assert row.request_id == response.json()["request_id"]
    assert row.predicted_intent == "card_arrival"
    assert row.input_text_redacted == "when does my card arrive"
    assert row.model_version == "test-v1"
    assert len(row.top_k_intents) == 5


def test_predict_does_not_propagate_audit_failure(
    mocks: tuple[MagicMock, MagicMock],
) -> None:
    """Audit raising in the background should not break the response.

    BackgroundTasks runs after the response is sent, so even if the writer
    raises, the client has already received its 200.
    """
    _, mock_audit = mocks
    mock_audit.write.side_effect = RuntimeError("BQ down")
    client = TestClient(app)
    response = client.post("/predict", json={"message": "ping"})
    assert response.status_code == 200
