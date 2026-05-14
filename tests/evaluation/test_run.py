"""Tests for src/evaluation/run.py: end-to-end with mock adapters."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.adapters import PredictionBatch, TeacherAdapter
from src.evaluation.golden import GoldenSet
from src.evaluation.run import evaluate, render_report_md
from src.evaluation.val import ValSet

N_CLASSES = 4
N_EXAMPLES = 20
ID_TO_LABEL = {0: "alpha", 1: "beta", 2: "gamma", 3: "delta"}


def _one_hot(labels: list[int]) -> np.ndarray:
    probs = np.zeros((len(labels), N_CLASSES))
    for i, lbl in enumerate(labels):
        probs[i, lbl] = 1.0
    return probs


def _make_fixture_sets() -> tuple[ValSet, GoldenSet]:
    rng = np.random.default_rng(0)
    val_labels = rng.integers(0, N_CLASSES, N_EXAMPLES).tolist()
    val_df = pd.DataFrame({"text": [f"val_{i}" for i in range(N_EXAMPLES)], "label": val_labels})
    val_set = ValSet(
        examples=val_df,
        id_to_label=ID_TO_LABEL,
        label_to_id={v: k for k, v in ID_TO_LABEL.items()},
    )
    golden_labels = rng.integers(0, N_CLASSES, N_EXAMPLES).tolist()
    golden_df = pd.DataFrame(
        {"text": [f"gold_{i}" for i in range(N_EXAMPLES)], "label": golden_labels}
    )
    golden_set = GoldenSet(
        examples=golden_df,
        id_to_label=ID_TO_LABEL,
        label_to_id={v: k for k, v in ID_TO_LABEL.items()},
        version="test",
    )
    return val_set, golden_set


def _make_classifier_adapter(name: str, seed: int = 0) -> object:
    """Returns an object whose .predict produces shape-(n, N_CLASSES) probs."""

    class _Mock:
        def __init__(self) -> None:
            self.name = name
            self._rng = np.random.default_rng(seed)

        def predict(self, texts: list[str]) -> PredictionBatch:
            n = len(texts)
            logits = self._rng.standard_normal((n, N_CLASSES))
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            top_indices = np.argsort(-probs, axis=1)
            top_probs = np.take_along_axis(probs, top_indices, axis=1)
            return PredictionBatch(
                probs=probs,
                top_k_indices=top_indices,
                top_k_probs=top_probs,
                per_example_latency_ms=np.full(n, 50.0),
                total_wall_time_s=0.05 * n,
            )

    return _Mock()


def _make_teacher_adapter() -> TeacherAdapter:
    """Real TeacherAdapter wired to a fake client; never hits the network."""
    # We can't easily monkey-patch label_message here, so instead we subclass.
    label_to_id = {v: k for k, v in ID_TO_LABEL.items()}

    class _StubbedTeacher(TeacherAdapter):
        def __init__(self) -> None:
            super().__init__(
                client=object(), label_to_id=label_to_id, taxonomy_block=""
            )
            self._rng = np.random.default_rng(1)

        def predict(self, texts: list[str]) -> PredictionBatch:
            n = len(texts)
            probs = _one_hot(list(self._rng.integers(0, N_CLASSES, n)))
            top_indices = np.argsort(-probs, axis=1)
            top_probs = np.take_along_axis(probs, top_indices, axis=1)
            return PredictionBatch(
                probs=probs,
                top_k_indices=top_indices,
                top_k_probs=top_probs,
                per_example_latency_ms=np.full(n, 800.0),
                total_wall_time_s=0.8 * n,
                prompt_tokens=np.full(n, 200, dtype=np.int64),
                completion_tokens=np.full(n, 5, dtype=np.int64),
            )

    return _StubbedTeacher()


# ---- evaluate ----


def test_evaluate_writes_results_and_report(tmp_path: Path) -> None:
    val, golden = _make_fixture_sets()
    adapters = [
        _make_classifier_adapter("distilbert", seed=0),
        _make_classifier_adapter("baseline_minilm_lr", seed=1),
        _make_teacher_adapter(),
    ]
    results = evaluate(adapters, val, golden, tmp_path)  # type: ignore[arg-type]
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "report.md").exists()
    # Reliability diagrams for classifiers only.
    assert (tmp_path / "reliability_distilbert.png").exists()
    assert (tmp_path / "reliability_baseline_minilm_lr.png").exists()
    # Per-classifier threshold files.
    assert (tmp_path / "distilbert_thresholds.json").exists()
    assert (tmp_path / "baseline_minilm_lr_thresholds.json").exists()

    # The returned dict matches what was written to results.json.
    on_disk = json.loads((tmp_path / "results.json").read_text())
    assert on_disk["models"].keys() == results["models"].keys()


def test_results_json_has_expected_top_level_keys(tmp_path: Path) -> None:
    val, golden = _make_fixture_sets()
    adapters = [_make_classifier_adapter("distilbert"), _make_teacher_adapter()]
    evaluate(adapters, val, golden, tmp_path)  # type: ignore[arg-type]
    payload = json.loads((tmp_path / "results.json").read_text())
    expected = {
        "run_id",
        "generated_at_utc",
        "git_sha",
        "git_dirty",
        "pricing_date",
        "target_precision",
        "serving_shape",
        "label_noise_band_pp",
        "golden_n_examples",
        "val_n_examples",
        "num_classes",
        "models",
        "cost_summary",
    }
    assert expected <= set(payload.keys())


def test_classifier_model_records_thresholds_and_metrics(tmp_path: Path) -> None:
    val, golden = _make_fixture_sets()
    evaluate(
        [_make_classifier_adapter("distilbert")],  # type: ignore[list-item]
        val,
        golden,
        tmp_path,
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    model = payload["models"]["distilbert"]
    assert model["kind"] == "calibrated_classifier"
    metrics = model["metrics"]
    assert {
        "top_1_accuracy",
        "top_5_accuracy",
        "macro_f1",
        "ece",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "latency_source",
        "cost_per_1k_usd",
    } <= set(metrics)
    assert metrics["latency_source"] == "laptop_dev"
    assert "thresholds" in model
    assert "per_class" in model
    assert "top_confusion_pairs" in model


def test_teacher_model_records_token_cost(tmp_path: Path) -> None:
    val, golden = _make_fixture_sets()
    evaluate([_make_teacher_adapter()], val, golden, tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text())
    teacher = payload["models"]["teacher"]
    assert teacher["kind"] == "llm_teacher"
    assert teacher["metrics"]["latency_source"] == "gemini_api_synchronous"
    # 200 input + 5 output tokens per call, all 20 calls — non-zero cost.
    assert teacher["metrics"]["cost_per_1k_usd"] > 0.0
    # No thresholds key for teacher.
    assert "thresholds" not in teacher


def test_cost_summary_includes_reduction_ratios(tmp_path: Path) -> None:
    val, golden = _make_fixture_sets()
    adapters = [_make_classifier_adapter("distilbert"), _make_teacher_adapter()]
    evaluate(adapters, val, golden, tmp_path)  # type: ignore[arg-type]
    payload = json.loads((tmp_path / "results.json").read_text())
    summary = payload["cost_summary"]
    assert "teacher_cost_per_1k_usd" in summary
    assert "student_cost_per_1k_usd" in summary
    assert "student_cost_reduction_vs_teacher_x" in summary


# ---- render_report_md ----


def test_render_report_md_handles_empty_class_support(tmp_path: Path) -> None:
    """A class with no support in the sample shouldn't crash the markdown writer."""
    val, golden = _make_fixture_sets()
    # Force every example to label 0; classes 1,2,3 will have zero support.
    val.examples["label"] = 0
    golden.examples["label"] = 0
    results = evaluate(
        [_make_classifier_adapter("distilbert")],  # type: ignore[list-item]
        val,
        golden,
        tmp_path,
    )
    md = render_report_md(results)
    # Sanity: it's a non-trivial string and contains the headline header.
    assert "Headline Metrics" in md
    assert "distilbert" in md


def test_render_report_md_flags_laptop_latency(tmp_path: Path) -> None:
    val, golden = _make_fixture_sets()
    results = evaluate(
        [_make_classifier_adapter("distilbert")],  # type: ignore[list-item]
        val,
        golden,
        tmp_path,
    )
    md = render_report_md(results)
    assert "laptop_dev" in md or "Caveat" in md


def test_render_report_md_when_only_classifiers(tmp_path: Path) -> None:
    """Cost summary should fall back to a 'skipped' note if teacher absent."""
    val, golden = _make_fixture_sets()
    results = evaluate(
        [_make_classifier_adapter("distilbert")],  # type: ignore[list-item]
        val,
        golden,
        tmp_path,
    )
    md = render_report_md(results)
    assert "Cost Reduction" in md
