"""Single-message teacher labeling unit.

Wraps client + prompt + parser into one callable that returns a
`LabeledMessage` with all telemetry attached. Retries on transient errors
(HTTP 429/5xx, timeouts) with exponential backoff + jitter. Unknown
intent names from the teacher are recorded with `teacher_intent_id=-1`
and `error="unknown_intent"` rather than raised — Phase 4 will filter
those out at training time.
"""

import random
import sys
import time
from dataclasses import dataclass
from typing import Any

import structlog

from src.labeling.client import TeacherClient, TeacherResponse
from src.labeling.prompt import (
    build_prompt,
    build_response_schema,
    parse_prediction,
)

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0
UNKNOWN_INTENT_ID = -1

_log = structlog.get_logger()


@dataclass(frozen=True)
class LabeledMessage:
    text: str
    teacher_intent_name: str
    teacher_intent_id: int  # UNKNOWN_INTENT_ID if the teacher returned a class not in the taxonomy
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_version: str
    error: str | None


def label_message(
    client: TeacherClient,
    message: str,
    taxonomy_block: str,
    label_to_id: dict[str, int],
) -> LabeledMessage:
    """Label one message with the teacher."""
    prompt = build_prompt(message, taxonomy_block)
    schema = build_response_schema(list(label_to_id.keys()))
    valid_intents = set(label_to_id.keys())

    response = _call_with_retry(client, prompt, schema)

    try:
        prediction = parse_prediction(response.parsed, valid_intents)
        teacher_intent_name = prediction.intent
        teacher_intent_id = label_to_id[prediction.intent]
        error: str | None = None
    except ValueError as e:
        msg = str(e)
        if "Unknown intent" in msg:
            raw = response.parsed if isinstance(response.parsed, dict) else {}
            teacher_intent_name = str(raw.get("intent", "")) if isinstance(raw, dict) else ""
            error = "unknown_intent"
        else:
            teacher_intent_name = ""
            error = "parse_error"
        teacher_intent_id = UNKNOWN_INTENT_ID
        _log.warning("parse_failed", error=msg, raw_text=response.text[:200])

    _log.info(
        "labeled",
        teacher_intent=teacher_intent_name,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=int(response.latency_ms),
        error=error,
    )

    return LabeledMessage(
        text=message,
        teacher_intent_name=teacher_intent_name,
        teacher_intent_id=teacher_intent_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        model_version=response.model,
        error=error,
    )


def _call_with_retry(
    client: TeacherClient,
    prompt: str,
    schema: dict[str, Any],
) -> TeacherResponse:
    """Call client.generate, retrying on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.generate(prompt, response_schema=schema)
        except Exception as e:
            if not _is_transient(e) or attempt == MAX_RETRIES - 1:
                raise
            backoff = min(INITIAL_BACKOFF_S * (2**attempt), MAX_BACKOFF_S)
            sleep_s = backoff + random.uniform(0, backoff * 0.5)
            _log.warning(
                "transient_error_retry",
                attempt=attempt + 1,
                error=str(e),
                sleep_s=round(sleep_s, 2),
            )
            time.sleep(sleep_s)
    raise AssertionError("unreachable: retry loop should have raised or returned")


def _is_transient(error: Exception) -> bool:
    """Heuristic: is this exception worth retrying?"""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if isinstance(code, int) and code in (408, 429, 500, 502, 503, 504):
        return True
    msg = str(error).lower()
    return any(
        s in msg
        for s in (
            "rate limit",
            "timeout",
            "deadline",
            "unavailable",
            "service unavailable",
            "internal error",
        )
    )


def _smoke() -> None:
    """Label one Banking77 message via the real Gemini client."""
    from src.data.banking77 import load_banking77
    from src.labeling.client import from_env
    from src.labeling.taxonomy import format_class_list

    splits = load_banking77()
    taxonomy = format_class_list(splits.id_to_label)
    sample = splits.train.iloc[0]
    message = sample["text"]
    gold_label_id = int(sample["label"])
    gold_label_name = splits.id_to_label[gold_label_id]

    client = from_env()
    labeled = label_message(client, message, taxonomy, splits.label_to_id)

    print(f"message:       {message!r}")
    print(f"gold label:    {gold_label_name} (id={gold_label_id})")
    print(f"teacher said:  {labeled.teacher_intent_name} (id={labeled.teacher_intent_id})")
    print(f"agree:         {labeled.teacher_intent_id == gold_label_id}")
    print(f"tokens:        {labeled.input_tokens} in / {labeled.output_tokens} out")
    print(f"latency:       {labeled.latency_ms:.0f} ms")
    print(f"error:         {labeled.error}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    else:
        print("Usage: uv run python -m src.labeling.labeler smoke")
        sys.exit(1)
