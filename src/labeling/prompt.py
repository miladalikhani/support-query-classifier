"""Prompt and structured-output schema for the teacher (Gemini).

The teacher returns a single field — `intent` — chosen from the closed set of
Banking77 classes. Self-reported LLM confidence is intentionally NOT requested
because it's poorly calibrated and never read downstream (see design_doc.md
§4.2-§4.4: routing confidence comes from the temperature-scaled student, not
from the teacher).
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TeacherPrediction:
    intent: str


def build_response_schema(valid_intents: list[str]) -> dict[str, Any]:
    """JSON Schema constraining the teacher's response.

    `enum` forces Gemini's structured-output mode to pick from the closed set.
    The parser still re-validates against `valid_intents` as defense in depth.
    """
    return {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": sorted(valid_intents),
                "description": "The chosen intent class name.",
            },
        },
        "required": ["intent"],
    }


def build_prompt(message: str, taxonomy_block: str) -> str:
    """Assemble the teacher prompt: instruction + taxonomy + customer message."""
    return (
        "You are an expert intent classifier for a retail bank's customer support "
        "chat queue. Read the customer message below and select the single best "
        "intent label from the closed set of classes.\n"
        "\n"
        "# Intent classes\n"
        f"{taxonomy_block}\n"
        "\n"
        "# Customer message\n"
        f"{message}\n"
        "\n"
        "Respond with JSON containing the chosen intent name (must match one of "
        "the class names above exactly)."
    )


def parse_prediction(raw: Any, valid_intents: set[str]) -> TeacherPrediction:
    """Strictly validate Gemini's response and return a TeacherPrediction.

    Raises ValueError on: non-dict input, missing `intent`, wrong type, or an
    intent not in `valid_intents`.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict response, got {type(raw).__name__}")
    if "intent" not in raw:
        raise ValueError("Missing required field 'intent'")
    intent = raw["intent"]
    if not isinstance(intent, str):
        raise ValueError(f"Field 'intent' must be a string, got {type(intent).__name__}")
    if intent not in valid_intents:
        raise ValueError(f"Unknown intent: {intent!r}")
    return TeacherPrediction(intent=intent)
