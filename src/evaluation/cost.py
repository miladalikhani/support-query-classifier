"""Cost-per-1k computation for the evaluation harness.

Loads dated pricing from `configs/cost/YYYY-MM.yaml` so report numbers
trace back to a specific rate sheet rather than to a hard-coded constant
nobody can verify a year from now. Bump pricing by writing a new YAML
and updating DEFAULT_PRICING_PATH.

Teacher cost is token-billed (Gemini on Vertex AI) and is hardware-
independent: Gemini latency is the same whether the call originates
from a laptop or from Cloud Run, because the work happens on Google's
infrastructure either way.

Student and baseline cost is wall-time billed (Cloud Run, request-based).
The latency that goes into the headline number must come from a Cloud
Run load test, not the eval adapter's in-process timing — laptop cores
are typically 2-4x faster per second than the vCPUs that Cloud Run
sells, and container overhead adds further skew. See
`src.evaluation.serving_latency.load_cloud_run_latencies` for the Phase 6
artifact the report should consume; until that file exists, dev latency
is a sanity check, not a billing claim.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_PRICING_PATH = Path(__file__).parents[2] / "configs" / "cost" / "2026-05.yaml"


def _load_pricing(path: Path = DEFAULT_PRICING_PATH) -> dict[str, Any]:
    """Read the dated pricing YAML."""
    return yaml.safe_load(path.read_text())


_PRICING: dict[str, Any] = _load_pricing()
PRICING_DATE: str = _PRICING["pricing_date"]
GEMINI_FLASH_PRICING: dict[str, float] = _PRICING["gemini_flash"]
CLOUD_RUN_PRICING: dict[str, float] = _PRICING["cloud_run"]


def teacher_cost_per_1k(
    prompt_tokens: np.ndarray,
    completion_tokens: np.ndarray,
    *,
    pricing: dict[str, float] | None = None,
) -> float:
    """Average dollar cost per 1k Gemini calls across input + output tokens."""
    p = pricing if pricing is not None else GEMINI_FLASH_PRICING
    prompt = np.asarray(prompt_tokens, dtype=np.float64)
    completion = np.asarray(completion_tokens, dtype=np.float64)
    n = prompt.shape[0]
    if n == 0:
        return 0.0
    input_cost = float(prompt.sum()) * p["usd_per_million_input_tokens"] / 1e6
    output_cost = float(completion.sum()) * p["usd_per_million_output_tokens"] / 1e6
    return (input_cost + output_cost) / n * 1000.0


def student_cost_per_1k(
    per_example_latency_ms: np.ndarray,
    vcpu: float,
    memory_gib: float,
    *,
    pricing: dict[str, float] | None = None,
) -> float:
    """Average Cloud Run dollar cost per 1k requests at the given instance shape.

    Cost = (per-example wall time * (vCPU + memory rates)) + per-request fee,
    averaged across the input and scaled to 1k. Sequential-request assumption
    means concurrency would lower this number.
    """
    p = pricing if pricing is not None else CLOUD_RUN_PRICING
    latency = np.asarray(per_example_latency_ms, dtype=np.float64)
    n = latency.shape[0]
    if n == 0:
        return 0.0
    seconds = latency / 1000.0
    compute_cost = seconds.sum() * (
        vcpu * p["usd_per_vcpu_second"] + memory_gib * p["usd_per_gib_second"]
    )
    request_cost = n * p["usd_per_million_requests"] / 1e6
    return (compute_cost + request_cost) / n * 1000.0


def format_cost_breakdown(
    *,
    teacher: float,
    student: float,
    baseline: float | None = None,
) -> dict[str, Any]:
    """Side-by-side cost-per-1k summary with reduction ratios for the report."""
    out: dict[str, Any] = {
        "pricing_date": PRICING_DATE,
        "teacher_cost_per_1k_usd": round(teacher, 6),
        "student_cost_per_1k_usd": round(student, 6),
        "student_cost_reduction_vs_teacher_x": (
            round(teacher / student, 1) if student > 0 else None
        ),
    }
    if baseline is not None:
        out["baseline_cost_per_1k_usd"] = round(baseline, 6)
        out["baseline_cost_reduction_vs_teacher_x"] = (
            round(teacher / baseline, 1) if baseline > 0 else None
        )
    return out
