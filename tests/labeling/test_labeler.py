from unittest.mock import MagicMock, patch

import pytest

from src.labeling.client import TeacherResponse
from src.labeling.labeler import (
    MAX_RETRIES,
    UNKNOWN_INTENT_ID,
    LabeledMessage,
    label_message,
)


def _response(
    parsed: dict | None = None,
    text: str = '{"intent": "card_arrival"}',
    tokens: tuple[int, int] = (100, 5),
) -> TeacherResponse:
    return TeacherResponse(
        text=text,
        parsed=parsed if parsed is not None else {"intent": "card_arrival"},
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        latency_ms=123.4,
        model="gemini-2.5-flash",
    )


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.generate.return_value = _response()
    return client


def test_happy_path_returns_complete_labeled_message(mock_client: MagicMock) -> None:
    label_to_id = {"card_arrival": 11, "lost_or_stolen_card": 41}
    result = label_message(mock_client, "where is my card?", "tax", label_to_id)

    assert isinstance(result, LabeledMessage)
    assert result.text == "where is my card?"
    assert result.teacher_intent_name == "card_arrival"
    assert result.teacher_intent_id == 11
    assert result.input_tokens == 100
    assert result.output_tokens == 5
    assert result.latency_ms == 123.4
    assert result.model_version == "gemini-2.5-flash"
    assert result.error is None


def test_unknown_intent_recorded_with_sentinel_id(mock_client: MagicMock) -> None:
    mock_client.generate.return_value = _response(
        parsed={"intent": "totally_made_up_class"},
        text='{"intent": "totally_made_up_class"}',
    )
    result = label_message(mock_client, "huh?", "tax", {"card_arrival": 11})

    assert result.teacher_intent_id == UNKNOWN_INTENT_ID
    assert result.teacher_intent_name == "totally_made_up_class"
    assert result.error == "unknown_intent"


def test_parse_failure_recorded_with_sentinel_id(mock_client: MagicMock) -> None:
    mock_client.generate.return_value = _response(parsed={"intent": 42})
    result = label_message(mock_client, "msg", "tax", {"card_arrival": 11})

    assert result.teacher_intent_id == UNKNOWN_INTENT_ID
    assert result.error == "parse_error"


def test_retries_then_succeeds(mock_client: MagicMock) -> None:
    transient = Exception("503 service unavailable")
    mock_client.generate.side_effect = [transient, _response()]

    with patch("src.labeling.labeler.time.sleep") as mock_sleep:
        result = label_message(mock_client, "msg", "tax", {"card_arrival": 11})

    assert result.error is None
    assert result.teacher_intent_id == 11
    assert mock_client.generate.call_count == 2
    mock_sleep.assert_called_once()


def test_gives_up_after_max_retries(mock_client: MagicMock) -> None:
    mock_client.generate.side_effect = Exception("503 service unavailable")

    with (
        patch("src.labeling.labeler.time.sleep"),
        pytest.raises(Exception, match="503"),
    ):
        label_message(mock_client, "msg", "tax", {"card_arrival": 11})

    assert mock_client.generate.call_count == MAX_RETRIES


def test_does_not_retry_on_non_transient(mock_client: MagicMock) -> None:
    mock_client.generate.side_effect = ValueError("bad prompt")

    with pytest.raises(ValueError, match="bad prompt"):
        label_message(mock_client, "msg", "tax", {"card_arrival": 11})

    assert mock_client.generate.call_count == 1


def test_recognises_429_via_status_code_attribute(mock_client: MagicMock) -> None:
    err = Exception("rate exceeded")
    err.code = 429  # type: ignore[attr-defined]
    mock_client.generate.side_effect = [err, _response()]

    with patch("src.labeling.labeler.time.sleep"):
        result = label_message(mock_client, "msg", "tax", {"card_arrival": 11})

    assert result.error is None
    assert mock_client.generate.call_count == 2
