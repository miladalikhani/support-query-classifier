"""Tests for src/evaluation/cost.py."""

import numpy as np
import pytest
import yaml

from src.evaluation.cost import (
    CLOUD_RUN_PRICING,
    DEFAULT_PRICING_PATH,
    GEMINI_FLASH_PRICING,
    PRICING_DATE,
    format_cost_breakdown,
    student_cost_per_1k,
    teacher_cost_per_1k,
)

# ---- Pricing constants ----


def test_pricing_constants_match_yaml() -> None:
    """Single source of truth: module constants must match the dated YAML."""
    pricing = yaml.safe_load(DEFAULT_PRICING_PATH.read_text())
    assert pricing["pricing_date"] == PRICING_DATE
    assert pricing["gemini_flash"] == GEMINI_FLASH_PRICING
    assert pricing["cloud_run"] == CLOUD_RUN_PRICING


def test_gemini_pricing_has_expected_keys() -> None:
    assert {"usd_per_million_input_tokens", "usd_per_million_output_tokens"} <= set(
        GEMINI_FLASH_PRICING
    )


def test_cloud_run_pricing_has_expected_keys() -> None:
    assert {
        "usd_per_vcpu_second",
        "usd_per_gib_second",
        "usd_per_million_requests",
    } <= set(CLOUD_RUN_PRICING)


# ---- teacher_cost_per_1k ----


def test_teacher_cost_matches_token_math() -> None:
    """One call: 1000 in @ $0.30/M + 10 out @ $2.50/M = $0.000325. Per 1k: $0.325."""
    pricing = {
        "usd_per_million_input_tokens": 0.30,
        "usd_per_million_output_tokens": 2.50,
    }
    cost = teacher_cost_per_1k(
        np.array([1000]), np.array([10]), pricing=pricing
    )
    expected_per_call = (1000 * 0.30 + 10 * 2.50) / 1_000_000
    assert cost == pytest.approx(expected_per_call * 1000.0)


def test_teacher_cost_averages_across_calls() -> None:
    """Two calls of identical tokens → same per-1k as one call."""
    pricing = {
        "usd_per_million_input_tokens": 0.30,
        "usd_per_million_output_tokens": 2.50,
    }
    one = teacher_cost_per_1k(np.array([1000]), np.array([10]), pricing=pricing)
    two = teacher_cost_per_1k(
        np.array([1000, 1000]), np.array([10, 10]), pricing=pricing
    )
    assert one == pytest.approx(two)


def test_teacher_cost_zero_examples() -> None:
    assert teacher_cost_per_1k(np.array([]), np.array([])) == 0.0


def test_teacher_cost_uses_module_pricing_by_default() -> None:
    """Without an explicit `pricing=`, the module-level YAML values apply."""
    explicit = teacher_cost_per_1k(
        np.array([500]), np.array([20]), pricing=GEMINI_FLASH_PRICING
    )
    implicit = teacher_cost_per_1k(np.array([500]), np.array([20]))
    assert explicit == pytest.approx(implicit)


# ---- student_cost_per_1k ----


def test_student_cost_scales_linearly_with_latency() -> None:
    pricing = {
        "usd_per_vcpu_second": 0.000024,
        "usd_per_gib_second": 0.0000025,
        "usd_per_million_requests": 0.40,
    }
    base = student_cost_per_1k(
        np.array([100.0]), vcpu=1.0, memory_gib=2.0, pricing=pricing
    )
    doubled = student_cost_per_1k(
        np.array([200.0]), vcpu=1.0, memory_gib=2.0, pricing=pricing
    )
    # Compute scales 2x; the per-request fixed fee does not.
    fixed = pricing["usd_per_million_requests"] / 1e6 * 1000.0
    assert doubled - fixed == pytest.approx(2 * (base - fixed))


def test_student_cost_includes_per_request_fee() -> None:
    """Zero-latency 'requests' still cost the per-request fee."""
    pricing = {
        "usd_per_vcpu_second": 0.000024,
        "usd_per_gib_second": 0.0000025,
        "usd_per_million_requests": 0.40,
    }
    cost = student_cost_per_1k(
        np.array([0.0, 0.0]), vcpu=1.0, memory_gib=2.0, pricing=pricing
    )
    # Per-request cost per 1k = 0.40 / 1e6 * 1000 = $0.0004
    assert cost == pytest.approx(0.40 / 1_000_000 * 1000.0)


def test_student_cost_zero_examples() -> None:
    assert student_cost_per_1k(np.array([]), vcpu=1.0, memory_gib=2.0) == 0.0


def test_student_cost_against_known_value() -> None:
    """100ms request at 1 vCPU + 2 GiB:

    compute = 0.1 s * (1 * 0.000024 + 2 * 0.0000025) = 0.1 * 0.000029 = $0.0000029
    request = 0.40 / 1e6 = $0.0000004
    per call = $0.0000033 -> per 1k = $0.0033
    """
    pricing = {
        "usd_per_vcpu_second": 0.000024,
        "usd_per_gib_second": 0.0000025,
        "usd_per_million_requests": 0.40,
    }
    cost = student_cost_per_1k(
        np.array([100.0]), vcpu=1.0, memory_gib=2.0, pricing=pricing
    )
    assert cost == pytest.approx(0.0033, abs=1e-6)


# ---- format_cost_breakdown ----


def test_format_cost_breakdown_minimal_two_models() -> None:
    out = format_cost_breakdown(teacher=0.010, student=0.0005)
    assert out["pricing_date"] == PRICING_DATE
    assert out["teacher_cost_per_1k_usd"] == 0.010
    assert out["student_cost_per_1k_usd"] == 0.0005
    assert out["student_cost_reduction_vs_teacher_x"] == 20.0


def test_format_cost_breakdown_includes_baseline_when_given() -> None:
    out = format_cost_breakdown(teacher=0.010, student=0.0005, baseline=0.0002)
    assert out["baseline_cost_per_1k_usd"] == 0.0002
    assert out["baseline_cost_reduction_vs_teacher_x"] == 50.0


def test_format_cost_breakdown_handles_zero_student() -> None:
    """No divide-by-zero when a model has no measured cost."""
    out = format_cost_breakdown(teacher=0.010, student=0.0)
    assert out["student_cost_reduction_vs_teacher_x"] is None
