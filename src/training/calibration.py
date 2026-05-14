"""Temperature scaling and Expected Calibration Error.

Neural classifiers and many other models tend to be over-confident: their
top-class softmax probability is systematically higher than the rate at
which that prediction is correct. Temperature scaling is a post-hoc fix
that rescales logits by a single learned scalar `T`:

    calibrated_probs = softmax(logits / T)

When `T > 1` the distribution softens (over-confidence is pulled down);
when `T < 1` it sharpens. The scalar is fit by minimising the NLL of the
true labels under the scaled distribution.

The Expected Calibration Error (ECE) measures the average gap between
predicted confidence and observed accuracy across confidence bins. Lower
is better; a perfectly calibrated model has ECE = 0.

These helpers are intentionally model-agnostic: they take raw logits as
input, not a particular model class. Whatever produces logits — a fitted
transformer head, an sklearn `LogisticRegression.decision_function`,
anything — can be calibrated by the same code.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy.optimize import minimize
from scipy.special import logsumexp

log = structlog.get_logger()

DEFAULT_N_BINS = 15
_TEMPERATURE_BOUNDS = (0.05, 100.0)


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    ece_pre: float
    ece_post: float
    val_accuracy_vs_truth: float
    val_accuracy_vs_teacher: float
    n_val: int
    n_bins: int
    fitted_at_utc: str


# ---------- Core calibration math ----------


def calibrate_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over `logits / temperature`. Numerically stable."""
    scaled = logits / float(temperature)
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return np.exp(log_probs)


def _neg_log_likelihood(
    temperature: np.ndarray,
    logits: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Mean NLL of `true_labels` under softmax(logits / temperature[0])."""
    t = float(temperature[0])
    scaled = logits / t
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return -log_probs[np.arange(len(true_labels)), true_labels].mean()


def fit_temperature(logits: np.ndarray, true_labels: np.ndarray) -> float:
    """Return the scalar T minimising NLL of `true_labels` under scaled logits."""
    result = minimize(
        _neg_log_likelihood,
        x0=np.array([1.0]),
        args=(logits, true_labels),
        method="L-BFGS-B",
        bounds=[_TEMPERATURE_BOUNDS],
    )
    return float(result.x[0])


def compute_ece(probs: np.ndarray, true_labels: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> float:
    """Expected Calibration Error using equal-width confidence bins.

    For each bin, takes |mean_confidence - mean_accuracy|, weighted by the
    fraction of examples in that bin.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == true_labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Map each confidence into a bin index in [0, n_bins-1].
    bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)

    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            bin_conf = float(confidences[mask].mean())
            bin_acc = float(accuracies[mask].mean())
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


# ---------- Glue: load val + run DistilBERT ----------


def _load_val(val_uri: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (texts, teacher_labels, truth_labels) from a labels parquet.

    Rejects parquets that aren't the val split, and drops unknown-intent rows
    so the returned arrays have only usable evaluation examples.
    """
    df = pd.read_parquet(val_uri)
    splits_present = sorted(df["split"].unique().tolist())
    if splits_present != ["val"]:
        raise ValueError(
            f"Expected split='val' parquet at {val_uri}, got splits {splits_present}"
        )
    df = df[df["teacher_intent_id"] != -1].reset_index(drop=True)
    return (
        df["text"].tolist(),
        df["teacher_intent_id"].astype(int).to_numpy(),
        df["gold_label_id"].astype(int).to_numpy(),
    )


def _distilbert_logits(model_dir: Path, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Run the saved DistilBERT on every text and return concatenated logits."""
    # Heavy imports kept local so module import is cheap for tests.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            out = model(**enc)
            chunks.append(out.logits.numpy())
    return np.concatenate(chunks, axis=0)


# ---------- CLI ----------


def calibrate_distilbert(
    model_dir: Path,
    val_uri: str,
    n_bins: int = DEFAULT_N_BINS,
) -> CalibrationResult:
    """End-to-end: get DistilBERT logits on val, fit T, report ECE before and after."""
    texts, teacher_labels, true_labels = _load_val(val_uri)
    log.info(
        "calibration_inputs", model_dir=str(model_dir), n_val=len(texts), n_bins=n_bins
    )

    logits = _distilbert_logits(model_dir, texts)
    log.info("logits_computed", shape=tuple(logits.shape))

    preds = logits.argmax(axis=1)
    vs_truth = float((preds == true_labels).mean())
    vs_teacher = float((preds == teacher_labels).mean())
    log.info(
        "val_accuracy", vs_truth=round(vs_truth, 4), vs_teacher=round(vs_teacher, 4)
    )

    ece_pre = compute_ece(calibrate_logits(logits, 1.0), true_labels, n_bins)
    temperature = fit_temperature(logits, true_labels)
    ece_post = compute_ece(calibrate_logits(logits, temperature), true_labels, n_bins)
    log.info(
        "calibration_fit",
        T=round(temperature, 4),
        ece_pre=round(ece_pre, 4),
        ece_post=round(ece_post, 4),
    )

    return CalibrationResult(
        temperature=temperature,
        ece_pre=ece_pre,
        ece_post=ece_post,
        val_accuracy_vs_truth=vs_truth,
        val_accuracy_vs_teacher=vs_teacher,
        n_val=len(texts),
        n_bins=n_bins,
        fitted_at_utc=datetime.now(UTC).isoformat(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help=(
            "Local directory containing a trained DistilBERT "
            "(config.json, model.safetensors, tokenizer)."
        ),
    )
    parser.add_argument(
        "--val-uri",
        required=True,
        help="URI (gs:// or local) to the val labels parquet. Must carry gold_label_id.",
    )
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = calibrate_distilbert(args.model_dir, args.val_uri, n_bins=args.n_bins)

    out_path = args.model_dir / "temperature.json"
    out_path.write_text(
        json.dumps(
            {
                "T": result.temperature,
                "ece_pre": result.ece_pre,
                "ece_post": result.ece_post,
                "val_accuracy_vs_truth": result.val_accuracy_vs_truth,
                "val_accuracy_vs_teacher": result.val_accuracy_vs_teacher,
                "n_val": result.n_val,
                "n_bins": result.n_bins,
                "fitted_at_utc": result.fitted_at_utc,
            },
            indent=2,
        )
    )

    print()
    print("=" * 60)
    print(f"Model dir:                  {args.model_dir}")
    print(f"Val examples:               {result.n_val}")
    print(f"Val accuracy vs truth:      {result.val_accuracy_vs_truth:.4f}")
    print(f"Val accuracy vs teacher:    {result.val_accuracy_vs_teacher:.4f}")
    print(f"Temperature T:              {result.temperature:.4f}")
    print(f"ECE pre-scaling:            {result.ece_pre:.4f}")
    print(f"ECE post-scaling:           {result.ece_post:.4f}")
    print(f"ECE improvement:            {result.ece_pre - result.ece_post:+.4f}")
    print(f"Wrote:                      {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
