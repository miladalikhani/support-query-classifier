"""Vertex AI Gemini client used to call the teacher model.

Wraps `google.genai.Client` for Vertex AI access. Includes a process-wide
cost guard that raises after a configurable number of generate calls so a
runaway loop can't quietly burn through the budget.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MAX_CALLS_PER_RUN = 50

# Gemini 2.5 Flash pricing (text), USD per 1M tokens (as of design-doc date).
USD_PER_M_INPUT_TOKENS = 0.075
USD_PER_M_OUTPUT_TOKENS = 0.30


@dataclass(frozen=True)
class TeacherResponse:
    text: str
    parsed: Any | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str

    def cost_usd(self) -> float:
        return (
            self.input_tokens * USD_PER_M_INPUT_TOKENS
            + self.output_tokens * USD_PER_M_OUTPUT_TOKENS
        ) / 1_000_000


class TeacherClient:
    """Wraps google.genai.Client for Vertex AI access to Gemini.

    Process-wide cost guard: every instance shares a class-level call counter.
    Once exceeded, all `generate` calls raise until the counter is reset.
    """

    _call_count: int = 0

    def __init__(
        self,
        project_id: str,
        location: str = DEFAULT_LOCATION,
        model: str = DEFAULT_MODEL,
        max_calls_per_run: int = DEFAULT_MAX_CALLS_PER_RUN,
    ) -> None:
        self._project_id = project_id
        self._location = location
        self._model = model
        self._max_calls = max_calls_per_run
        self._client = genai.Client(vertexai=True, project=project_id, location=location)

    def generate(
        self,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> TeacherResponse:
        TeacherClient._call_count += 1
        if TeacherClient._call_count > self._max_calls:
            raise RuntimeError(
                f"Cost guard: {self._max_calls} generate() calls already made in this "
                f"process. Increase max_calls_per_run if this is intended."
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if response_schema else None,
            response_schema=response_schema,
        )

        start = time.monotonic()
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.monotonic() - start) * 1000

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return TeacherResponse(
            text=response.text or "",
            parsed=getattr(response, "parsed", None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self._model,
        )

    @classmethod
    def reset_call_count(cls) -> None:
        cls._call_count = 0

    @classmethod
    def call_count(cls) -> int:
        return cls._call_count


def from_env() -> TeacherClient:
    """Build a TeacherClient from environment variables.

    Loads `.env` from the current working directory first; values already
    present in the environment are not overridden.

    Required: `GCP_PROJECT_ID`.
    Optional: `GCP_REGION` (default us-central1), `GEMINI_MODEL`
    (default gemini-2.5-flash), `MAX_CALLS_PER_RUN` (default 50).
    """
    load_dotenv()
    try:
        project_id = os.environ["GCP_PROJECT_ID"]
    except KeyError as e:
        raise RuntimeError(
            "GCP_PROJECT_ID must be set. Example: export GCP_PROJECT_ID=datatonic-496102"
        ) from e
    return TeacherClient(
        project_id=project_id,
        location=os.getenv("GCP_REGION", DEFAULT_LOCATION),
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        max_calls_per_run=int(os.getenv("MAX_CALLS_PER_RUN", str(DEFAULT_MAX_CALLS_PER_RUN))),
    )


def _smoke() -> None:
    client = from_env()
    resp = client.generate("Reply with the single word: pong")
    print(f"model:     {resp.model}")
    print(f"response:  {resp.text!r}")
    print(f"tokens:    {resp.input_tokens} in / {resp.output_tokens} out")
    print(f"latency:   {resp.latency_ms:.0f} ms")
    print(f"cost:      ${resp.cost_usd():.6f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    else:
        print("Usage: uv run python -m src.labeling.client smoke")
        sys.exit(1)
