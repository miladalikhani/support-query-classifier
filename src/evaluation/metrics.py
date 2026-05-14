"""Pure metric functions over `(probs, true_labels)`.

No model loading, no I/O beyond the optional `reliability_diagram` PNG
output. The evaluation runner wires these into a report; each function
here can be called independently from a notebook or future drift job.

`compute_ece` is re-exported from `src.training.calibration` so callers
have a single metrics import surface.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix as _sk_confusion_matrix,
)
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)

from src.training.calibration import compute_ece

__all__ = [
    "compute_ece",
    "confusion_matrix",
    "macro_f1",
    "per_class_metrics",
    "reliability_diagram",
    "top_confusion_pairs",
    "top_k_accuracy",
]


def top_k_accuracy(probs: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Fraction of rows whose true label is among the model's top-k predictions."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, probs.shape[1])
    top = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.any(top == np.asarray(true_labels)[:, None], axis=1).mean())


def macro_f1(probs: np.ndarray, true_labels: np.ndarray) -> float:
    """Macro-averaged F1 over argmax predictions across all probs columns."""
    preds = probs.argmax(axis=1)
    labels = np.arange(probs.shape[1])
    return float(
        f1_score(true_labels, preds, labels=labels, average="macro", zero_division=0)
    )


def per_class_metrics(
    probs: np.ndarray,
    true_labels: np.ndarray,
    id_to_label: dict[int, str],
) -> pd.DataFrame:
    """One row per class: support, precision, recall, f1, top1_accuracy.

    `top1_accuracy` is the fraction of examples in the class that the
    model ranks correctly as its top-1 prediction — algebraically equal
    to recall under argmax, kept as a separate column because the
    report addresses both audiences (precision/recall for ML readers,
    top-1 accuracy for product readers).
    """
    preds = probs.argmax(axis=1)
    true_labels = np.asarray(true_labels)
    class_ids = sorted(id_to_label.keys())
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        preds,
        labels=class_ids,
        zero_division=0,
    )
    rows = []
    for cls_id, p, r, f, s in zip(class_ids, precision, recall, f1, support, strict=True):
        mask = true_labels == cls_id
        top1 = float((preds[mask] == cls_id).mean()) if mask.any() else 0.0
        rows.append(
            {
                "class_name": id_to_label[cls_id],
                "support": int(s),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "top1_accuracy": top1,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix(
    probs: np.ndarray,
    true_labels: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Row-normalized (true → predicted) confusion matrix over argmax preds."""
    n_classes = probs.shape[1]
    preds = probs.argmax(axis=1)
    cm = _sk_confusion_matrix(true_labels, preds, labels=np.arange(n_classes)).astype(
        np.float64
    )
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Classes absent from the true labels become zero rows after normalisation.
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    return cm


def top_confusion_pairs(
    cm: np.ndarray,
    id_to_label: dict[int, str],
    top_n: int = 20,
) -> list[tuple[str, str, float]]:
    """Worst off-diagonal cells, sorted by value descending.

    Returns `(true_class_name, predicted_class_name, value)` tuples — used by
    the report to surface ambiguous class pairs.
    """
    n = cm.shape[0]
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, -np.inf)
    flat_order = np.argsort(off_diag, axis=None)[::-1]
    pairs: list[tuple[str, str, float]] = []
    for flat_idx in flat_order:
        if len(pairs) >= top_n:
            break
        i, j = divmod(int(flat_idx), n)
        value = float(cm[i, j])
        if not np.isfinite(off_diag[i, j]):
            break
        pairs.append((id_to_label[i], id_to_label[j], value))
    return pairs


def reliability_diagram(
    probs: np.ndarray,
    true_labels: np.ndarray,
    out_path: Path,
    n_bins: int = 15,
) -> None:
    """Write a reliability-diagram PNG: bin accuracy vs bin mean-confidence."""
    # Force the non-interactive backend so this is safe to call in CI / headless.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == np.asarray(true_labels)).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_n = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            bin_acc[i] = correct[mask].mean()
            bin_conf[i] = confidences[mask].mean()
            bin_n[i] = int(mask.sum())

    centres = (edges[:-1] + edges[1:]) / 2
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.bar(
        centres,
        bin_acc,
        width=width,
        edgecolor="black",
        alpha=0.7,
        label="accuracy",
    )
    populated = bin_n > 0
    ax.scatter(
        bin_conf[populated],
        bin_acc[populated],
        s=40,
        color="red",
        zorder=3,
        label="bin mean confidence",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
