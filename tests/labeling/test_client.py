from unittest.mock import MagicMock, patch

import pytest

from src.labeling.client import TeacherClient, TeacherResponse, from_env


@pytest.fixture(autouse=True)
def _reset_call_count() -> None:
    TeacherClient.reset_call_count()
    yield
    TeacherClient.reset_call_count()


@pytest.fixture
def mock_response() -> MagicMock:
    response = MagicMock()
    response.text = '{"intent": "card_arrival", "confidence": 0.9}'
    response.parsed = {"intent": "card_arrival", "confidence": 0.9}
    response.usage_metadata.prompt_token_count = 120
    response.usage_metadata.candidates_token_count = 18
    return response


@pytest.fixture
def patched_genai(mock_response: MagicMock):
    with patch("src.labeling.client.genai.Client") as mock_class:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.return_value = mock_response
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance


def test_constructor_wires_genai_for_vertex(patched_genai: tuple) -> None:
    mock_class, _ = patched_genai
    TeacherClient(project_id="test-project", location="us-east1", model="gemini-2.5-flash")
    mock_class.assert_called_once_with(vertexai=True, project="test-project", location="us-east1")


def test_generate_returns_complete_teacher_response(patched_genai: tuple) -> None:
    client = TeacherClient(project_id="p")
    resp = client.generate("hi")
    assert isinstance(resp, TeacherResponse)
    assert resp.text == '{"intent": "card_arrival", "confidence": 0.9}'
    assert resp.input_tokens == 120
    assert resp.output_tokens == 18
    assert resp.latency_ms >= 0
    assert resp.model == "gemini-2.5-flash"


def test_generate_invokes_sdk_with_expected_args(patched_genai: tuple) -> None:
    _, mock_instance = patched_genai
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    client = TeacherClient(project_id="p")
    client.generate("hello", response_schema=schema, temperature=0.2)
    args = mock_instance.models.generate_content.call_args
    assert args.kwargs["model"] == "gemini-2.5-flash"
    assert args.kwargs["contents"] == "hello"
    config = args.kwargs["config"]
    assert config.temperature == 0.2
    assert config.response_mime_type == "application/json"
    assert config.response_schema == schema


def test_cost_guard_raises_after_threshold(patched_genai: tuple) -> None:
    client = TeacherClient(project_id="p", max_calls_per_run=2)
    client.generate("call 1")
    client.generate("call 2")
    with pytest.raises(RuntimeError, match="Cost guard"):
        client.generate("call 3")


def test_cost_guard_is_process_wide(patched_genai: tuple) -> None:
    """Two TeacherClient instances share the same counter."""
    a = TeacherClient(project_id="p", max_calls_per_run=2)
    b = TeacherClient(project_id="p", max_calls_per_run=2)
    a.generate("from a")
    b.generate("from b")
    with pytest.raises(RuntimeError, match="Cost guard"):
        a.generate("over limit")


def test_cost_usd_computes_against_published_pricing() -> None:
    resp = TeacherResponse(
        text="",
        parsed=None,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_ms=0,
        model="gemini-2.5-flash",
    )
    # 1M input @ $0.30 + 1M output @ $2.50 = $2.80
    assert abs(resp.cost_usd() - 2.80) < 1e-9


def test_from_env_reads_required_and_optional_vars(
    monkeypatch: pytest.MonkeyPatch, patched_genai: tuple
) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "my-proj")
    monkeypatch.setenv("GCP_REGION", "europe-west1")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-flash-experimental")
    monkeypatch.setenv("MAX_CALLS_PER_RUN", "7")
    client = from_env()
    assert client._project_id == "my-proj"
    assert client._location == "europe-west1"
    assert client._model == "gemini-flash-experimental"
    assert client._max_calls == 7


def test_from_env_raises_without_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Disable .env loading so the local .env doesn't smuggle a value in.
    monkeypatch.setattr("src.labeling.client.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID"):
        from_env()
