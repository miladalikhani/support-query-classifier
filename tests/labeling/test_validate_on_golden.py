from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.labeling.client import TeacherResponse
from src.labeling.labeler import LabeledMessage
from src.labeling.validate_on_golden import (
    _parse_args,
    _percentiles,
    compute_report,
    estimate_cost_usd,
    label_concurrently,
    stratified_subsample,
)


def _make_labeled(
    intent_id: int, name: str = "x", in_tok: int = 100, out_tok: int = 5
) -> LabeledMessage:
    return LabeledMessage(
        text="msg",
        teacher_intent_name=name,
        teacher_intent_id=intent_id,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=100.0,
        model_version="gemini-2.5-flash",
        error=None,
    )


# ---------- Stratified subsample ----------


def test_stratified_subsample_preserves_class_proportions() -> None:
    df = pd.DataFrame({"text": [f"m{i}" for i in range(100)], "label": [i % 5 for i in range(100)]})
    subset = stratified_subsample(df, n=20, seed=42)
    assert len(subset) == 20
    # Original is 20 per class; expect ~4 per class in a stratified split
    counts = subset["label"].value_counts().sort_index()
    assert all(c == 4 for c in counts), f"expected 4 per class, got {counts.to_dict()}"


def test_stratified_subsample_returns_all_when_n_exceeds_df() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    subset = stratified_subsample(df, n=100)
    assert len(subset) == 2


def test_stratified_subsample_falls_back_to_random_for_small_n() -> None:
    df = pd.DataFrame({"text": [f"m{i}" for i in range(100)], "label": [i % 5 for i in range(100)]})
    # n=3, but there are 5 classes, so stratified is impossible
    subset = stratified_subsample(df, n=3, seed=42)
    assert len(subset) == 3


def test_stratified_subsample_is_deterministic() -> None:
    df = pd.DataFrame({"text": [f"m{i}" for i in range(50)], "label": [i % 5 for i in range(50)]})
    a = stratified_subsample(df, n=10, seed=42)
    b = stratified_subsample(df, n=10, seed=42)
    pd.testing.assert_frame_equal(a, b)


# ---------- Cost estimate ----------


def test_estimate_cost_scales_linearly_with_n() -> None:
    taxonomy = "x" * 4000  # ~1000 tokens
    cost_100, _, _ = estimate_cost_usd(100, taxonomy)
    cost_300, _, _ = estimate_cost_usd(300, taxonomy)
    assert abs(cost_300 - 3 * cost_100) < 1e-9


# ---------- Percentiles ----------


def test_percentiles_handles_single_value() -> None:
    assert _percentiles([42.0]) == {"p50": 42.0, "p95": 42.0, "p99": 42.0}


def test_percentiles_handles_empty() -> None:
    assert _percentiles([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_percentiles_orders_correctly() -> None:
    p = _percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert p["p50"] <= p["p95"] <= p["p99"]


# ---------- Report ----------


def test_compute_report_counts_accuracy_correctly() -> None:
    id_to_label = {0: "a", 1: "b"}
    results = [
        (0, _make_labeled(0), 0),  # correct
        (1, _make_labeled(1), 1),  # correct
        (2, _make_labeled(0), 1),  # incorrect
        (3, _make_labeled(-1), 0),  # error, excluded from accuracy
    ]
    report = compute_report(
        results,
        id_to_label,
        "model-x",
        wall_time_s=5.0,
        run_id="r1",
        prompt_version="1",
        prompt_fp="abc123",
    )
    # 2 correct out of 3 valid = 0.6667
    assert report["ceiling_top1_accuracy"] == pytest.approx(0.6667, abs=1e-3)
    assert report["subset_size"] == 4
    assert report["model"] == "model-x"
    assert report["wall_time_s"] == 5.0
    assert report["prompt"] == {"version": "1", "fingerprint": "abc123"}


def test_compute_report_per_class_breakdown() -> None:
    id_to_label = {0: "a", 1: "b"}
    results = [
        (0, _make_labeled(0), 0),  # a correct
        (1, _make_labeled(1), 0),  # a incorrect
        (2, _make_labeled(1), 1),  # b correct
        (3, _make_labeled(1), 1),  # b correct
    ]
    report = compute_report(
        results,
        id_to_label,
        "m",
        wall_time_s=1.0,
        run_id="r",
        prompt_version="1",
        prompt_fp="fp",
    )
    per_class = report["per_class"]
    assert per_class["a"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert per_class["b"] == {"n": 2, "correct": 2, "accuracy": 1.0}


def test_compute_report_aggregates_tokens_and_cost() -> None:
    results = [
        (0, _make_labeled(0, in_tok=1_000_000, out_tok=1_000_000), 0),
    ]
    report = compute_report(
        results,
        {0: "a"},
        "m",
        wall_time_s=1.0,
        run_id="r",
        prompt_version="1",
        prompt_fp="fp",
    )
    # 1M input @ $0.075 + 1M output @ $0.30 = $0.375
    assert report["cost_usd"] == pytest.approx(0.375, abs=1e-3)
    assert report["tokens"]["input_total"] == 1_000_000
    assert report["tokens"]["output_total"] == 1_000_000


# ---------- Concurrent labeling ----------


def test_label_concurrently_calls_labeler_for_each_input() -> None:
    client = MagicMock()
    examples = [("msg1", 0), ("msg2", 1), ("msg3", 0)]

    fake_response = TeacherResponse(
        text="",
        parsed={"intent": "a"},
        input_tokens=10,
        output_tokens=2,
        latency_ms=5,
        model="m",
    )
    client.generate.return_value = fake_response

    with patch("src.labeling.validate_on_golden.label_message") as mock_label:
        mock_label.side_effect = lambda c, msg, tax, lookup: _make_labeled(
            lookup.get("a", -1), name="a"
        )
        results = label_concurrently(client, examples, "tax", {"a": 0}, concurrency=2)

    assert len(results) == 3
    assert mock_label.call_count == 3
    # Results are sorted by input index
    assert [idx for idx, _, _ in results] == [0, 1, 2]


def test_label_concurrently_halts_on_cost_cap() -> None:
    client = MagicMock()
    examples = [("m", 0)] * 10

    # Each call is "expensive": 0.5 USD per call.
    expensive = _make_labeled(0, in_tok=1_000_000 * 3, out_tok=1_000_000 * 1)
    # input: 3M * 0.075/M = 0.225
    # output: 1M * 0.30/M = 0.300
    # total per call: 0.525

    with patch("src.labeling.validate_on_golden.label_message", return_value=expensive):
        results = label_concurrently(
            client, examples, "tax", {"x": 0}, concurrency=1, max_cost_usd=1.0
        )

    # Should halt after ~2 calls (cumulative cost passes 1.0 after the 2nd)
    assert len(results) < len(examples)


# ---------- CLI ----------


def test_parse_args_defaults() -> None:
    args = _parse_args([])
    assert args.max_examples == 300
    assert args.max_cost_usd == 1.0
    assert args.concurrency == 5
    assert args.yes is False
    assert args.no_upload is False


def test_parse_args_custom() -> None:
    args = _parse_args(["--max-examples", "100", "--max-cost-usd", "0.5", "--yes", "--no-upload"])
    assert args.max_examples == 100
    assert args.max_cost_usd == 0.5
    assert args.yes is True
    assert args.no_upload is True
