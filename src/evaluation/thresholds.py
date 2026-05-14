"""Per-class confidence thresholds for auto-route vs human-triage routing.

Each class gets its own confidence cutoff: predictions with confidence at
or above the cutoff are auto-routed to the matching specialist queue;
predictions below it are escalated to a human triage agent.

Thresholds are fit per-class because Banking77 classes have very
different difficulty profiles. A distinctive class like `card_arrival`
is reliable at 0.7 confidence, while ambiguous pairs (e.g.,
`transfer_into_account` vs `top_up_by_bank_transfer`) can be wrong 30%
of the time at 0.95. One global threshold either auto-routes the hard
classes incorrectly or escalates the easy ones unnecessarily.

The fit walks each class's val predictions sorted by descending
confidence and finds the lowest confidence at which the running
precision still meets `target_precision`. If precision never meets
target (or the class has no val support), the threshold falls back to
1.01 — guaranteed to never auto-route, surfacing the class for human
review in the report.

Discipline: thresholds are fit on val, never golden. Golden is the
held-out evaluation set and any fitting against it would compromise
every number the report cites. Golden is used to measure per-threshold
precision and recall after fitting, but never to choose the cutoff.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ESCALATE_ALL_THRESHOLD = 1.01


@dataclass(frozen=True)
class ThresholdFitResult:
    """Output of fit_per_class_thresholds plus the val provenance for the report."""

    thresholds: dict[int, float]
    target_precision: float
    fitted_at_utc: str
    n_val_examples: int


def fit_per_class_thresholds(
    val_probs: np.ndarray,
    val_true_labels: np.ndarray,
    target_precision: float = 0.95,
) -> ThresholdFitResult:
    """Fit a confidence cutoff per class on val using a running-precision walk."""
    if not 0.0 < target_precision <= 1.0:
        raise ValueError(
            f"target_precision must be in (0, 1], got {target_precision}"
        )
    n_classes = val_probs.shape[1]
    val_true_labels = np.asarray(val_true_labels)
    predictions = val_probs.argmax(axis=1)
    thresholds: dict[int, float] = {}
    for c in range(n_classes):
        thresholds[c] = _fit_one_class(
            val_probs[:, c], predictions == c, val_true_labels == c, target_precision
        )
    return ThresholdFitResult(
        thresholds=thresholds,
        target_precision=float(target_precision),
        fitted_at_utc=datetime.now(UTC).isoformat(),
        n_val_examples=int(val_probs.shape[0]),
    )


def _fit_one_class(
    class_confidences: np.ndarray,
    predicted_as_class: np.ndarray,
    is_true_class: np.ndarray,
    target_precision: float,
) -> float:
    """Sort the candidates by confidence desc, return lowest cutoff meeting target."""
    candidate_mask = predicted_as_class
    if not candidate_mask.any():
        return ESCALATE_ALL_THRESHOLD

    confidences = class_confidences[candidate_mask]
    correct = is_true_class[candidate_mask].astype(np.float64)
    order = np.argsort(-confidences, kind="stable")
    confidences = confidences[order]
    correct = correct[order]

    cumulative_correct = np.cumsum(correct)
    running_precision = cumulative_correct / np.arange(1, len(confidences) + 1)
    meets_target = running_precision >= target_precision
    if not meets_target.any():
        return ESCALATE_ALL_THRESHOLD
    last_qualifying_idx = int(np.flatnonzero(meets_target).max())
    return float(confidences[last_qualifying_idx])


def apply_thresholds(
    probs: np.ndarray,
    thresholds: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Argmax predictions and a mask of which ones cleared their class threshold."""
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    per_pred_threshold = np.asarray(
        [thresholds.get(int(p), ESCALATE_ALL_THRESHOLD) for p in predictions]
    )
    accepted_mask = confidences >= per_pred_threshold
    return predictions, accepted_mask


def summarize_thresholds(
    thresholds: dict[int, float],
    val_probs: np.ndarray,
    val_true_labels: np.ndarray,
    id_to_label: dict[int, str],
) -> pd.DataFrame:
    """Per-class fit summary: threshold, val precision/recall, n_accepted."""
    val_true_labels = np.asarray(val_true_labels)
    predictions, accepted = apply_thresholds(val_probs, thresholds)

    rows: list[dict[str, Any]] = []
    for cls_id in sorted(id_to_label):
        threshold = thresholds.get(cls_id, ESCALATE_ALL_THRESHOLD)
        accepted_as_c = accepted & (predictions == cls_id)
        true_is_c = val_true_labels == cls_id
        n_accepted = int(accepted_as_c.sum())
        n_true_c = int(true_is_c.sum())
        n_correct_accepted = int((accepted_as_c & true_is_c).sum())
        precision = (
            float(n_correct_accepted) / n_accepted if n_accepted > 0 else 0.0
        )
        recall = float(n_correct_accepted) / n_true_c if n_true_c > 0 else 0.0
        rows.append(
            {
                "class_name": id_to_label[cls_id],
                "fitted_threshold": float(threshold),
                "val_precision": precision,
                "val_recall": recall,
                "n_accepted": n_accepted,
            }
        )
    return pd.DataFrame(rows)


def save_thresholds(
    fit_result: ThresholdFitResult,
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write thresholds + per-class val summary to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_precision": fit_result.target_precision,
        "fitted_at_utc": fit_result.fitted_at_utc,
        "n_val_examples": fit_result.n_val_examples,
        "escalate_all_threshold": ESCALATE_ALL_THRESHOLD,
        "thresholds": {str(k): v for k, v in sorted(fit_result.thresholds.items())},
        "per_class_summary": summary.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path
