import pytest

from src.labeling.prompt import (
    TeacherPrediction,
    build_prompt,
    build_response_schema,
    parse_prediction,
)


def test_build_prompt_includes_message_and_taxonomy() -> None:
    taxonomy = "- foo: Customer asks about foo.\n- bar: Customer asks about bar."
    prompt = build_prompt("how do I foo?", taxonomy)
    assert "how do I foo?" in prompt
    assert "- foo: Customer asks about foo." in prompt
    assert "- bar: Customer asks about bar." in prompt


def test_build_prompt_is_deterministic() -> None:
    a = build_prompt("msg", "taxonomy")
    b = build_prompt("msg", "taxonomy")
    assert a == b


def test_response_schema_enumerates_intents_and_marks_required() -> None:
    schema = build_response_schema(["card_arrival", "lost_or_stolen_card"])
    assert schema["type"] == "object"
    assert schema["properties"]["intent"]["type"] == "string"
    assert schema["properties"]["intent"]["enum"] == ["card_arrival", "lost_or_stolen_card"]
    assert schema["required"] == ["intent"]


def test_response_schema_sorts_intents_for_determinism() -> None:
    schema = build_response_schema(["zeta", "alpha", "mu"])
    assert schema["properties"]["intent"]["enum"] == ["alpha", "mu", "zeta"]


def test_parse_prediction_happy_path() -> None:
    result = parse_prediction(
        {"intent": "card_arrival"},
        valid_intents={"card_arrival", "lost_or_stolen_card"},
    )
    assert isinstance(result, TeacherPrediction)
    assert result.intent == "card_arrival"


def test_parse_prediction_ignores_extra_fields() -> None:
    """Gemini might emit extra fields; we accept and ignore them."""
    result = parse_prediction(
        {"intent": "card_arrival", "reasoning": "the user mentioned a card"},
        valid_intents={"card_arrival"},
    )
    assert result.intent == "card_arrival"


def test_parse_prediction_raises_on_non_dict_input() -> None:
    with pytest.raises(ValueError, match="Expected dict"):
        parse_prediction("card_arrival", valid_intents={"card_arrival"})


def test_parse_prediction_raises_on_missing_intent() -> None:
    with pytest.raises(ValueError, match="Missing required field 'intent'"):
        parse_prediction({}, valid_intents={"x"})


def test_parse_prediction_raises_on_wrong_intent_type() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        parse_prediction({"intent": 42}, valid_intents={"x"})


def test_parse_prediction_raises_on_unknown_intent() -> None:
    with pytest.raises(ValueError, match="Unknown intent"):
        parse_prediction({"intent": "made_up_class"}, valid_intents={"card_arrival"})
