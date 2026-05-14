"""Load Cloud Run load-test latency files for the headline report.

In-process latency captured by the model adapters reflects whichever
machine ran the eval (the developer's laptop). That number does not
match what production will pay for: Cloud Run vCPUs are slower than
modern laptop cores, container overhead is real, and concurrency
changes per-request billing. Reporting laptop latency as a production
claim understates cost and overstates speed.

The right number comes from a Cloud Run load test (Phase 6): deploy
the serving image, hit it with a representative workload, dump the
per-request wall times to JSON. This module loads that file so the
evaluation report can substitute it for the adapter-measured latency
when computing the headline cost and P50/P95/P99.

Expected JSON shape:

    {
        "model": "distilbert",
        "deployment_run_id": "2026-05-15T...",
        "vcpu": 1.0,
        "memory_gib": 2.0,
        "per_example_latency_ms": [123.4, 130.1, ...],
        "sampled_at_utc": "2026-05-15T..."
    }
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REQUIRED_KEYS = (
    "model",
    "deployment_run_id",
    "vcpu",
    "memory_gib",
    "per_example_latency_ms",
    "sampled_at_utc",
)


@dataclass(frozen=True)
class CloudRunLatencies:
    """Per-request wall-time samples from a Cloud Run deployment."""

    model: str
    deployment_run_id: str
    vcpu: float
    memory_gib: float
    per_example_latency_ms: np.ndarray
    sampled_at_utc: str


def load_cloud_run_latencies(path: Path) -> CloudRunLatencies:
    """Read a Cloud Run latency JSON file produced by the load test."""
    data = json.loads(path.read_text())
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"Cloud Run latency file {path} is missing required keys: {missing}"
        )
    latencies = np.asarray(data["per_example_latency_ms"], dtype=np.float64)
    if latencies.size == 0:
        raise ValueError(
            f"Cloud Run latency file {path} has zero samples — load test never ran?"
        )
    return CloudRunLatencies(
        model=str(data["model"]),
        deployment_run_id=str(data["deployment_run_id"]),
        vcpu=float(data["vcpu"]),
        memory_gib=float(data["memory_gib"]),
        per_example_latency_ms=latencies,
        sampled_at_utc=str(data["sampled_at_utc"]),
    )
