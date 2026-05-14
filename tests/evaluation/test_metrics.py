"""Tests for src/evaluation/metrics.py."""

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_ece,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    reliability_diagram,
    top_confusion_pairs,
    top_k_accuracy,
)


def _one_hot(label_ids: list[int], n_classes: int) -> np.ndarray:
    probs = np.zeros((len(label_ids), n_classes))
    for i, lbl in enumerate(label_ids):
        probs[i, lbl] = 1.0
    return probs


# ---- top_k_accuracy ----


def test_top_k_accuracy_perfect_when_truth_always_in_top_k() -> None:
    probs = _one_hot([0, 1, 2, 3], n_classes=5)
    true_labels = np.array([0, 1, 2, 3])
    assert top_k_accuracy(probs, true_labels, k=1) == 1.0
    assert top_k_accuracy(probs, true_labels, k=3) == 1.0


def test_top_k_accuracy_zero_when_truth_never_in_top_k() -> None:
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.1, 0.0],
            [0.7, 0.1, 0.1, 0.1, 0.0],
        ]
    )
    true_labels = np.array([4, 4])
    assert top_k_accuracy(probs, true_labels, k=2) == 0.0


def test_top_k_accuracy_handles_ties_via_top_2() -> None:
    """Truth not in argmax but in top-2 → top-2 accuracy is 1.0."""
    probs = np.array([[0.5, 0.4, 0.1]])
    true_labels = np.array([1])
    assert top_k_accuracy(probs, true_labels, k=1) == 0.0
    assert top_k_accuracy(probs, true_labels, k=2) == 1.0


def test_top_k_accuracy_clamps_k_to_n_classes() -> None:
    probs = _one_hot([0], n_classes=3)
    assert top_k_accuracy(probs, np.array([0]), k=100) == 1.0


def test_top_k_accuracy_rejects_non_positive_k() -> None:
    probs = _one_hot([0], n_classes=3)
    with pytest.raises(ValueError):
        top_k_accuracy(probs, np.array([0]), k=0)


# ---- macro_f1 ----


def test_macro_f1_perfect_predictions() -> None:
    probs = _one_hot([0, 1, 2, 0, 1, 2], n_classes=3)
    true_labels = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(probs, true_labels) == pytest.approx(1.0)


def test_macro_f1_completely_wrong_predictions() -> None:
    probs = _one_hot([1, 0], n_classes=2)
    true_labels = np.array([0, 1])
    assert macro_f1(probs, true_labels) == 0.0


# ---- per_class_metrics ----


def test_per_class_metrics_returns_one_row_per_class() -> None:
    id_to_label = {0: "a", 1: "b", 2: "c"}
    probs = _one_hot([0, 1, 2, 0, 1], n_classes=3)
    true_labels = np.array([0, 1, 2, 0, 1])
    df = per_class_metrics(probs, true_labels, id_to_label)
    assert len(df) == 3
    assert list(df["class_name"]) == ["a", "b", "c"]


def test_per_class_metrics_support_sums_to_n() -> None:
    id_to_label = {0: "a", 1: "b", 2: "c"}
    probs = _one_hot([0, 1, 2, 0, 1], n_classes=3)
    true_labels = np.array([0, 1, 2, 0, 1])
    df = per_class_metrics(probs, true_labels, id_to_label)
    assert df["support"].sum() == 5


def test_per_class_metrics_perfect_predictions() -> None:
    id_to_label = {0: "a", 1: "b"}
    probs = _one_hot([0, 1, 0, 1], n_classes=2)
    true_labels = np.array([0, 1, 0, 1])
    df = per_class_metrics(probs, true_labels, id_to_label)
    np.testing.assert_allclose(df["precision"], 1.0)
    np.testing.assert_allclose(df["recall"], 1.0)
    np.testing.assert_allclose(df["f1"], 1.0)
    np.testing.assert_allclose(df["top1_accuracy"], 1.0)


def test_per_class_metrics_includes_classes_with_zero_support() -> None:
    """A class never appearing in true_labels still gets a (zero) row."""
    id_to_label = {0: "a", 1: "b", 2: "c"}
    probs = _one_hot([0, 1, 0], n_classes=3)
    true_labels = np.array([0, 1, 0])
    df = per_class_metrics(probs, true_labels, id_to_label)
    c_row = df[df["class_name"] == "c"].iloc[0]
    assert c_row["support"] == 0
    assert c_row["top1_accuracy"] == 0.0


# ---- confusion_matrix ----


def test_confusion_matrix_shape_matches_probs_classes() -> None:
    probs = _one_hot([0, 1, 2], n_classes=3)
    true_labels = np.array([0, 1, 2])
    cm = confusion_matrix(probs, true_labels, normalize=False)
    assert cm.shape == (3, 3)


def test_confusion_matrix_normalized_rows_sum_to_one() -> None:
    probs = _one_hot([0, 1, 1, 2, 2], n_classes=3)
    true_labels = np.array([0, 1, 2, 2, 2])
    cm = confusion_matrix(probs, true_labels, normalize=True)
    # Each row with support should sum to 1; class 0 has support 1 → row sums 1.
    row_sums = cm.sum(axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0])


def test_confusion_matrix_identity_on_perfect_predictions() -> None:
    probs = _one_hot([0, 1, 2, 0, 1, 2], n_classes=3)
    true_labels = np.array([0, 1, 2, 0, 1, 2])
    cm = confusion_matrix(probs, true_labels, normalize=True)
    np.testing.assert_allclose(cm, np.eye(3))


def test_confusion_matrix_zero_row_for_absent_class() -> None:
    """If true_labels never includes class 2, row 2 of normalized cm is all-zero."""
    probs = _one_hot([0, 1], n_classes=3)
    true_labels = np.array([0, 1])
    cm = confusion_matrix(probs, true_labels, normalize=True)
    np.testing.assert_allclose(cm[2], 0.0)


# ---- top_confusion_pairs ----


def test_top_confusion_pairs_excludes_diagonal_and_sorts_descending() -> None:
    cm = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.5, 0.1],
            [0.0, 0.6, 0.4],
        ]
    )
    id_to_label = {0: "a", 1: "b", 2: "c"}
    pairs = top_confusion_pairs(cm, id_to_label, top_n=3)
    assert len(pairs) == 3
    # Highest off-diagonal is cm[2,1] = 0.6; next cm[1,0] = 0.4; next cm[0,1] = 0.2.
    assert pairs[0] == ("c", "b", pytest.approx(0.6))
    # Descending order overall.
    values = [v for _, _, v in pairs]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    # No diagonal entries.
    for true_name, pred_name, _ in pairs:
        assert true_name != pred_name


def test_top_confusion_pairs_respects_top_n() -> None:
    cm = np.eye(4) * 0.0  # all-zero matrix
    cm[0, 1] = 0.3
    cm[1, 2] = 0.4
    id_to_label = {0: "a", 1: "b", 2: "c", 3: "d"}
    pairs = top_confusion_pairs(cm, id_to_label, top_n=1)
    assert len(pairs) == 1
    assert pairs[0] == ("b", "c", pytest.approx(0.4))


# ---- reliability_diagram ----


def test_reliability_diagram_writes_non_empty_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n, k = 200, 5
    logits = rng.standard_normal((n, k))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    true_labels = rng.integers(0, k, n)
    out = tmp_path / "rel.png"
    reliability_diagram(probs, true_labels, out_path=out, n_bins=10)
    assert out.exists()
    assert out.stat().st_size > 0


def test_reliability_diagram_creates_parent_dir(tmp_path: Path) -> None:
    """Parent directory should be created if it does not exist."""
    rng = np.random.default_rng(1)
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    true_labels = np.array([0, 0])
    out = tmp_path / "nested" / "dir" / "rel.png"
    reliability_diagram(probs, true_labels, out_path=out, n_bins=5)
    assert out.exists()
    _ = rng  # unused; left to keep numpy import consistent across tests


# ---- compute_ece re-export ----


def test_compute_ece_is_re_exported() -> None:
    """The runner imports `compute_ece` from this module rather than calibration."""
    n = 50
    probs = np.zeros((n, 3))
    probs[:, 1] = 1.0
    labels = np.ones(n, dtype=int)
    assert compute_ece(probs, labels, n_bins=10) < 1e-9
