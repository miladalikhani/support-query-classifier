"""Tests for src/evaluation/serving_latency.py."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.serving_latency import (
    CloudRunLatencies,
    load_cloud_run_latencies,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_round_trip_loads_all_fields(tmp_path: Path) -> None:
    payload = {
        "model": "distilbert",
        "deployment_run_id": "2026-05-15T12-00-00Z",
        "vcpu": 1.0,
        "memory_gib": 2.0,
        "per_example_latency_ms": [110.4, 120.1, 130.0],
        "sampled_at_utc": "2026-05-15T12-30-00Z",
    }
    path = _write(tmp_path / "latencies.json", payload)
    loaded = load_cloud_run_latencies(path)
    assert isinstance(loaded, CloudRunLatencies)
    assert loaded.model == "distilbert"
    assert loaded.deployment_run_id == "2026-05-15T12-00-00Z"
    assert loaded.vcpu == 1.0
    assert loaded.memory_gib == 2.0
    np.testing.assert_allclose(
        loaded.per_example_latency_ms, [110.4, 120.1, 130.0]
    )
    assert loaded.sampled_at_utc == "2026-05-15T12-30-00Z"


def test_raises_on_missing_required_key(tmp_path: Path) -> None:
    payload = {
        "model": "distilbert",
        # deployment_run_id intentionally missing
        "vcpu": 1.0,
        "memory_gib": 2.0,
        "per_example_latency_ms": [100.0],
        "sampled_at_utc": "...",
    }
    path = _write(tmp_path / "broken.json", payload)
    with pytest.raises(ValueError, match="deployment_run_id"):
        load_cloud_run_latencies(path)


def test_raises_on_empty_latency_array(tmp_path: Path) -> None:
    payload = {
        "model": "baseline_minilm_lr",
        "deployment_run_id": "...",
        "vcpu": 1.0,
        "memory_gib": 2.0,
        "per_example_latency_ms": [],
        "sampled_at_utc": "...",
    }
    path = _write(tmp_path / "empty.json", payload)
    with pytest.raises(ValueError, match="zero samples"):
        load_cloud_run_latencies(path)


def test_loaded_latencies_compose_with_cost_module(tmp_path: Path) -> None:
    """The loaded array is directly consumable by student_cost_per_1k."""
    from src.evaluation.cost import student_cost_per_1k

    payload = {
        "model": "distilbert",
        "deployment_run_id": "...",
        "vcpu": 1.0,
        "memory_gib": 2.0,
        "per_example_latency_ms": [100.0, 100.0],
        "sampled_at_utc": "...",
    }
    path = _write(tmp_path / "cloud.json", payload)
    loaded = load_cloud_run_latencies(path)
    cost = student_cost_per_1k(
        loaded.per_example_latency_ms,
        vcpu=loaded.vcpu,
        memory_gib=loaded.memory_gib,
    )
    assert cost > 0.0
