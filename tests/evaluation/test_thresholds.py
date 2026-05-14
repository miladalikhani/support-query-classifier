"""Tests for src/evaluation/thresholds.py."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.thresholds import (
    ESCALATE_ALL_THRESHOLD,
    apply_thresholds,
    fit_per_class_thresholds,
    save_thresholds,
    summarize_thresholds,
)

# ---- fit_per_class_thresholds ----


def _two_class_probs(
    n: int,
    class_a_acc_above_T: float,
    T: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic: class 0 is correct only above confidence T at rate class_a_acc_above_T.

    Above T, fraction `class_a_acc_above_T` of predictions are class 0 (and correct).
    Below T, only half are correct.
    """
    rng = np.random.default_rng(seed)
    confidences = rng.uniform(0.5, 1.0, n)
    above_T_mask = confidences >= T
    is_correct = np.zeros(n, dtype=bool)
    is_correct[above_T_mask] = rng.random(above_T_mask.sum()) < class_a_acc_above_T
    is_correct[~above_T_mask] = rng.random((~above_T_mask).sum()) < 0.5
    probs = np.zeros((n, 2))
    probs[:, 0] = confidences
    probs[:, 1] = 1.0 - confidences
    true_labels = np.where(is_correct, 0, 1)
    return probs, true_labels


def test_fit_returns_finite_threshold_when_target_achievable() -> None:
    """If precision above some T is consistently above target, the fitter finds it."""
    probs, true_labels = _two_class_probs(n=500, class_a_acc_above_T=0.98, T=0.7, seed=42)
    result = fit_per_class_thresholds(probs, true_labels, target_precision=0.95)
    # Class 0's threshold should be a real cutoff in (0, 1], not the escalate sentinel.
    assert result.thresholds[0] < 1.0


def test_fit_hopeless_class_returns_escalate_sentinel() -> None:
    """A class with confident-but-wrong predictions gets threshold = 1.01."""
    n = 200
    probs = np.zeros((n, 3))
    probs[:, 0] = 0.9  # model is very confident about class 0
    probs[:, 1] = 0.05
    probs[:, 2] = 0.05
    true_labels = np.ones(n, dtype=int)  # but the truth is always class 1
    result = fit_per_class_thresholds(probs, true_labels, target_precision=0.95)
    assert result.thresholds[0] == ESCALATE_ALL_THRESHOLD


def test_fit_class_with_no_predictions_returns_escalate_sentinel() -> None:
    """Model never predicts this class → threshold defaults to 1.01."""
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.95, 0.03, 0.02],
        ]
    )
    true_labels = np.array([0, 0, 0])
    result = fit_per_class_thresholds(probs, true_labels, target_precision=0.95)
    assert result.thresholds[1] == ESCALATE_ALL_THRESHOLD
    assert result.thresholds[2] == ESCALATE_ALL_THRESHOLD


def test_fit_deterministic_on_same_data() -> None:
    """Same val data → identical threshold dict."""
    probs, true_labels = _two_class_probs(n=300, class_a_acc_above_T=0.97, T=0.6, seed=7)
    a = fit_per_class_thresholds(probs, true_labels, target_precision=0.95)
    b = fit_per_class_thresholds(probs, true_labels, target_precision=0.95)
    assert a.thresholds == b.thresholds


def test_fit_rejects_invalid_target_precision() -> None:
    probs = np.array([[0.9, 0.1]])
    true_labels = np.array([0])
    with pytest.raises(ValueError):
        fit_per_class_thresholds(probs, true_labels, target_precision=0.0)
    with pytest.raises(ValueError):
        fit_per_class_thresholds(probs, true_labels, target_precision=1.5)


def test_fit_records_provenance_fields() -> None:
    probs, true_labels = _two_class_probs(n=100, class_a_acc_above_T=0.98, T=0.7)
    result = fit_per_class_thresholds(probs, true_labels, target_precision=0.9)
    assert result.target_precision == 0.9
    assert result.n_val_examples == 100
    assert "T" in result.fitted_at_utc  # ISO 8601


# ---- apply_thresholds ----


def test_apply_accepted_predictions_clear_their_class_threshold() -> None:
    """Every prediction in the accepted_mask must have confidence >= its class threshold."""
    probs = np.array(
        [
            [0.95, 0.03, 0.02],
            [0.70, 0.20, 0.10],
            [0.10, 0.85, 0.05],
            [0.40, 0.30, 0.30],
        ]
    )
    thresholds = {0: 0.9, 1: 0.6, 2: 0.8}
    preds, accepted = apply_thresholds(probs, thresholds)
    np.testing.assert_array_equal(preds, [0, 0, 1, 0])
    np.testing.assert_array_equal(accepted, [True, False, True, False])


def test_apply_falls_back_to_escalate_for_unknown_class() -> None:
    """If a class is missing from thresholds, predictions of it never accept."""
    probs = np.array([[0.1, 0.9]])
    preds, accepted = apply_thresholds(probs, {0: 0.5})  # class 1 missing
    assert preds[0] == 1
    assert not accepted[0]


# ---- summarize_thresholds ----


def test_summarize_shape_and_columns() -> None:
    probs, true_labels = _two_class_probs(n=200, class_a_acc_above_T=0.96, T=0.7)
    result = fit_per_class_thresholds(probs, true_labels, target_precision=0.9)
    id_to_label = {0: "alpha", 1: "beta"}
    summary = summarize_thresholds(result.thresholds, probs, true_labels, id_to_label)
    assert set(summary.columns) == {
        "class_name",
        "fitted_threshold",
        "val_precision",
        "val_recall",
        "n_accepted",
    }
    assert list(summary["class_name"]) == ["alpha", "beta"]


def test_summarize_val_precision_meets_target_for_fitted_classes() -> None:
    """Accepted predictions at the fitted cutoff should hit target precision on val."""
    probs, true_labels = _two_class_probs(
        n=500, class_a_acc_above_T=0.99, T=0.7, seed=11
    )
    target = 0.9
    result = fit_per_class_thresholds(probs, true_labels, target_precision=target)
    summary = summarize_thresholds(
        result.thresholds, probs, true_labels, {0: "alpha", 1: "beta"}
    )
    alpha = summary[summary["class_name"] == "alpha"].iloc[0]
    if alpha["n_accepted"] > 0:
        assert alpha["val_precision"] >= target


# ---- save_thresholds ----


def test_save_thresholds_writes_well_formed_json(tmp_path: Path) -> None:
    probs, true_labels = _two_class_probs(n=100, class_a_acc_above_T=0.97, T=0.7)
    result = fit_per_class_thresholds(probs, true_labels, target_precision=0.9)
    id_to_label = {0: "alpha", 1: "beta"}
    summary = summarize_thresholds(result.thresholds, probs, true_labels, id_to_label)
    out = tmp_path / "thresholds.json"
    save_thresholds(result, summary, out)

    payload = json.loads(out.read_text())
    assert payload["target_precision"] == 0.9
    assert payload["n_val_examples"] == 100
    assert payload["escalate_all_threshold"] == ESCALATE_ALL_THRESHOLD
    assert set(payload["thresholds"]) == {"0", "1"}
    assert len(payload["per_class_summary"]) == 2


def test_save_thresholds_creates_parent_dir(tmp_path: Path) -> None:
    probs, true_labels = _two_class_probs(n=50, class_a_acc_above_T=0.97, T=0.7)
    result = fit_per_class_thresholds(probs, true_labels, target_precision=0.9)
    summary = summarize_thresholds(
        result.thresholds, probs, true_labels, {0: "alpha", 1: "beta"}
    )
    out = tmp_path / "nested" / "dir" / "thresholds.json"
    save_thresholds(result, summary, out)
    assert out.exists()
