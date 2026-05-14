"""Tests for src/training/calibration.py."""

import numpy as np
import pytest
from scipy.special import softmax

from src.training.calibration import (
    calibrate_logits,
    compute_ece,
    fit_temperature,
)

# ---- calibrate_logits ----


def test_calibrate_logits_with_t_equal_one_matches_softmax() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    probs = calibrate_logits(logits, 1.0)
    np.testing.assert_allclose(probs, softmax(logits, axis=1), atol=1e-9)


def test_calibrate_logits_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [10.0, -5.0, 0.0]])
    probs = calibrate_logits(logits, 2.5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-9)


def test_higher_temperature_softens_distribution() -> None:
    logits = np.array([[10.0, 0.0, 0.0]])
    sharp = calibrate_logits(logits, 1.0)
    soft = calibrate_logits(logits, 5.0)
    assert sharp[0].max() > soft[0].max()


# ---- fit_temperature ----


def test_fit_temperature_on_overconfident_logits_returns_value_above_one() -> None:
    rng = np.random.default_rng(42)
    n, k = 500, 5
    true_labels = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k)) * 0.5
    logits[np.arange(n), true_labels] += 1.0
    overconfident = logits * 10.0
    t = fit_temperature(overconfident, true_labels)
    assert t > 2.0


def test_fit_temperature_on_underconfident_logits_returns_value_below_one() -> None:
    rng = np.random.default_rng(0)
    n, k = 500, 5
    true_labels = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k)) * 0.5
    logits[np.arange(n), true_labels] += 3.0
    underconfident = logits * 0.1
    t = fit_temperature(underconfident, true_labels)
    assert t < 1.0


# ---- compute_ece ----


def test_compute_ece_zero_when_predictions_are_perfectly_calibrated() -> None:
    """Confidence 1.0 with 100% accuracy → ECE = 0."""
    n = 100
    probs = np.zeros((n, 3))
    probs[:, 1] = 1.0
    true_labels = np.ones(n, dtype=int)
    assert compute_ece(probs, true_labels, n_bins=10) < 1e-9


def test_compute_ece_high_when_predictions_are_overconfident() -> None:
    """Confidence 1.0 with 0% accuracy → ECE close to 1.0."""
    n = 100
    probs = np.zeros((n, 3))
    probs[:, 1] = 1.0  # always predict class 1 with full confidence
    true_labels = np.zeros(n, dtype=int)  # but the truth is class 0 always
    ece = compute_ece(probs, true_labels, n_bins=10)
    assert ece > 0.9


def test_temperature_scaling_reduces_ece_on_overconfident_logits() -> None:
    rng = np.random.default_rng(0)
    n, k = 2000, 4
    true_labels = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k)) * 0.5
    logits[np.arange(n), true_labels] += 1.0
    overconfident = logits * 8.0

    ece_pre = compute_ece(calibrate_logits(overconfident, 1.0), true_labels, n_bins=15)
    t = fit_temperature(overconfident, true_labels)
    ece_post = compute_ece(calibrate_logits(overconfident, t), true_labels, n_bins=15)

    assert ece_post < ece_pre
    # Sanity: the lift should be material for very-overconfident logits, not noise.
    assert ece_pre - ece_post > 0.05


def test_compute_ece_handles_small_inputs() -> None:
    probs = np.array([[0.6, 0.4]])
    labels = np.array([0])
    ece = compute_ece(probs, labels, n_bins=10)
    # The single example sits at confidence 0.6 in a bin; bin accuracy is 1.0
    # so |0.6 - 1.0| * weight 1.0 = 0.4
    assert ece == pytest.approx(0.4, abs=1e-9)
