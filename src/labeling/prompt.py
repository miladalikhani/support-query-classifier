"""Prompt and structured-output schema for the teacher (Gemini).

The teacher returns a single field — `intent` — chosen from the closed set of
Banking77 classes. Self-reported LLM confidence is intentionally NOT requested
because it's poorly calibrated and never read downstream (see design_doc.md
§4.2-§4.4: routing confidence comes from the temperature-scaled student, not
from the teacher).

Prompt versioning: bump PROMPT_VERSION whenever you change the instruction
text, the taxonomy descriptions, the schema shape, the model, or the
sampling parameters. `prompt_fingerprint()` derives a content hash that
catches accidental drift even if the version label wasn't bumped.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.labeling.taxonomy import BANKING77_DESCRIPTIONS

PROMPT_VERSION = "3"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.0

INSTRUCTION_HEADER = (
    "You are an expert intent classifier for a retail bank's customer support "
    "chat queue. Read the customer message below and select the single best "
    "intent label from the closed set of classes."
)
INSTRUCTION_FOOTER = (
    "Respond with JSON containing the chosen intent name (must match one of "
    "the class names above exactly)."
)


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
        f"{INSTRUCTION_HEADER}\n"
        "\n"
        "# Intent classes\n"
        f"{taxonomy_block}\n"
        "\n"
        "# Customer message\n"
        f"{message}\n"
        "\n"
        f"{INSTRUCTION_FOOTER}"
    )


def prompt_fingerprint(
    *,
    valid_intents: list[str],
    descriptions: dict[str, str] = BANKING77_DESCRIPTIONS,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """16-char SHA-256 fingerprint of everything that determines teacher behavior.

    Two runs with the same fingerprint are guaranteed to send the same prompt
    and the same schema to the same model with the same sampling settings.
    If the fingerprint drifts but PROMPT_VERSION hasn't been bumped, someone
    forgot to bump.
    """
    canonical = {
        "version": prompt_version,
        "model": model,
        "temperature": temperature,
        "instruction_header": INSTRUCTION_HEADER,
        "instruction_footer": INSTRUCTION_FOOTER,
        "intents_with_descriptions": [
            [name, descriptions[name]] for name in sorted(valid_intents)
        ],
        "schema": build_response_schema(valid_intents),
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


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
