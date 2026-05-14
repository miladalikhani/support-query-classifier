"""Tests for src/evaluation/adapters.py."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.adapters import (
    DEFAULT_TOP_K,
    NUM_CLASSES,
    BaselineAdapter,
    DistilbertAdapter,
    PredictionBatch,
    TeacherAdapter,
    _softmax_with_temperature,
    _top_k,
)
from src.training.baseline import BaselineModel
from src.training.persistence import save_baseline_bundle

# ---- Pure helpers ----


def test_softmax_with_temperature_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [10.0, -5.0, 0.0]])
    probs = _softmax_with_temperature(logits, 2.5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-9)


def test_softmax_with_temperature_softens_at_higher_t() -> None:
    logits = np.array([[10.0, 0.0, 0.0]])
    sharp = _softmax_with_temperature(logits, 1.0)
    soft = _softmax_with_temperature(logits, 5.0)
    assert sharp[0].max() > soft[0].max()


def test_softmax_with_temperature_handles_neg_inf() -> None:
    """Logits with -inf entries must still produce a finite, normalized row."""
    logits = np.array([[1.0, -np.inf, 2.0]])
    probs = _softmax_with_temperature(logits, 1.0)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 1] == 0.0


def test_top_k_returns_descending_indices() -> None:
    probs = np.array([[0.1, 0.5, 0.2, 0.05, 0.15], [0.4, 0.3, 0.1, 0.1, 0.1]])
    indices, top_probs = _top_k(probs, k=3)
    assert indices.shape == (2, 3)
    assert top_probs.shape == (2, 3)
    # Each row must be sorted descending and consistent with `probs`.
    for row in range(probs.shape[0]):
        np.testing.assert_array_equal(
            top_probs[row], np.sort(probs[row])[::-1][:3]
        )
        np.testing.assert_array_equal(
            top_probs[row], probs[row, indices[row]]
        )


def test_top_k_clamps_to_num_classes() -> None:
    probs = np.array([[0.6, 0.4]])
    indices, top_probs = _top_k(probs, k=10)
    assert indices.shape == (1, 2)
    assert top_probs.shape == (1, 2)


# ---- PredictionBatch shape invariants ----


def test_prediction_batch_holds_all_fields() -> None:
    """Default construction with arrays produces a coherent object."""
    n, k = 4, 3
    probs = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES)
    batch = PredictionBatch(
        probs=probs,
        top_k_indices=np.zeros((n, k), dtype=np.int64),
        top_k_probs=np.zeros((n, k)),
        per_example_latency_ms=np.zeros(n),
        total_wall_time_s=0.0,
    )
    assert batch.probs.shape == (n, NUM_CLASSES)
    assert batch.prompt_tokens is None
    assert batch.completion_tokens is None


# ---- BaselineAdapter ----


def _make_baseline_bundle(tmp_path: Path) -> Path:
    """Build a real baseline bundle via save_baseline_bundle for round-trip testing."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8))
    y = rng.integers(0, NUM_CLASSES, 60).tolist()
    clf = LogisticRegression(max_iter=200, random_state=0).fit(X, y)
    model = BaselineModel(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        classifier=clf,
        id_to_label={i: f"class_{i}" for i in range(NUM_CLASSES)},
        label_to_id={f"class_{i}": i for i in range(NUM_CLASSES)},
        n_train_examples=60,
        sklearn_version="1.x",
        trained_at_utc=datetime.now(UTC).isoformat(),
    )
    bundle_dir = tmp_path / "baseline_bundle"
    with patch("src.training.persistence._git_info", return_value=("x", False)):
        save_baseline_bundle(
            model,
            bundle_dir,
            teacher_train_uri="gs://b/v1/train/R/labels.parquet",
            teacher_val_uri="gs://b/v1/val/R/labels.parquet",
            prompt_version="3",
            prompt_fingerprint="fp",
            training_config={"encoder_name": model.encoder_name},
            val_accuracy_vs_teacher=0.5,
            val_accuracy_vs_truth=0.4,
            ece_pre=0.1,
            ece_post=0.05,
            temperature=1.0,
        )
    return bundle_dir


class _StubEncoder:
    """Stand-in for SentenceTransformer that returns deterministic embeddings."""

    def __init__(self, _name: str, dim: int = 8) -> None:
        self._dim = dim

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        rng = np.random.default_rng(len(texts))
        return rng.standard_normal((len(texts), self._dim))


def test_baseline_adapter_produces_valid_prediction_batch(tmp_path: Path) -> None:
    bundle_dir = _make_baseline_bundle(tmp_path)
    with patch("sentence_transformers.SentenceTransformer", _StubEncoder):
        adapter = BaselineAdapter(bundle_dir, batch_size=2)
    assert adapter.name == "baseline_minilm_lr"

    batch = adapter.predict(["alpha", "beta", "gamma"])
    assert batch.probs.shape == (3, NUM_CLASSES)
    np.testing.assert_allclose(batch.probs.sum(axis=1), 1.0, rtol=1e-9)
    assert batch.top_k_indices.shape == (3, DEFAULT_TOP_K)
    assert batch.top_k_probs.shape == (3, DEFAULT_TOP_K)
    assert (batch.per_example_latency_ms >= 0).all()
    assert batch.total_wall_time_s >= 0


def test_baseline_adapter_top_k_consistent_with_probs(tmp_path: Path) -> None:
    bundle_dir = _make_baseline_bundle(tmp_path)
    with patch("sentence_transformers.SentenceTransformer", _StubEncoder):
        adapter = BaselineAdapter(bundle_dir)
    batch = adapter.predict(["text"])
    np.testing.assert_array_equal(
        batch.top_k_probs[0], batch.probs[0, batch.top_k_indices[0]]
    )
    # Descending order.
    assert np.all(np.diff(batch.top_k_probs[0]) <= 0)


# ---- DistilbertAdapter ----


class _FakeLogits:
    """Pretends to be a torch tensor enough for `.detach().cpu().numpy()`."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self) -> "_FakeLogits":
        return self

    def cpu(self) -> "_FakeLogits":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeOutput:
    def __init__(self, logits: _FakeLogits) -> None:
        self.logits = logits


class _FakeModel:
    """Deterministic 77-way classifier head over a fixed weight vector."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self._w = rng.standard_normal(NUM_CLASSES)

    def __call__(self, **inputs: Any) -> _FakeOutput:
        ids = inputs["input_ids"]
        batch = int(ids.shape[0]) if hasattr(ids, "shape") else len(ids)
        logits = np.broadcast_to(self._w, (batch, NUM_CLASSES)).copy()
        return _FakeOutput(_FakeLogits(logits))

    def eval(self) -> None:
        return None


class _FakeTokenizer:
    def __call__(self, texts: list[str], **_: Any) -> dict[str, Any]:
        import torch

        n = len(texts)
        return {
            "input_ids": torch.zeros((n, 4), dtype=torch.long),
            "attention_mask": torch.ones((n, 4), dtype=torch.long),
        }


def test_distilbert_adapter_produces_valid_prediction_batch(tmp_path: Path) -> None:
    fake_bundle = {
        "model": _FakeModel(),
        "tokenizer": _FakeTokenizer(),
        "temperature": 1.0,
        "id_to_label": {i: f"class_{i}" for i in range(NUM_CLASSES)},
        "label_to_id": {f"class_{i}": i for i in range(NUM_CLASSES)},
        "manifest": {},
    }
    with patch(
        "src.training.persistence.load_distilbert_bundle", return_value=fake_bundle
    ):
        adapter = DistilbertAdapter(tmp_path / "fake_bundle", batch_size=2)
    assert adapter.name == "distilbert"

    batch = adapter.predict(["a", "b", "c"])
    assert batch.probs.shape == (3, NUM_CLASSES)
    np.testing.assert_allclose(batch.probs.sum(axis=1), 1.0, rtol=1e-6)
    assert batch.top_k_indices.shape == (3, DEFAULT_TOP_K)
    assert (batch.per_example_latency_ms >= 0).all()


def test_distilbert_adapter_applies_temperature(tmp_path: Path) -> None:
    """Higher T should soften the predicted distribution."""

    def make(t: float) -> DistilbertAdapter:
        fake_bundle: dict[str, Any] = {
            "model": _FakeModel(),
            "tokenizer": _FakeTokenizer(),
            "temperature": t,
            "id_to_label": {i: f"class_{i}" for i in range(NUM_CLASSES)},
            "label_to_id": {f"class_{i}": i for i in range(NUM_CLASSES)},
            "manifest": {},
        }
        with patch(
            "src.training.persistence.load_distilbert_bundle",
            return_value=fake_bundle,
        ):
            return DistilbertAdapter(tmp_path / f"b_{t}")

    sharp = make(1.0).predict(["x"]).probs[0].max()
    soft = make(5.0).predict(["x"]).probs[0].max()
    assert sharp > soft


# ---- TeacherAdapter ----


class _FakeLabeledMessage:
    def __init__(
        self,
        teacher_intent_id: int,
        latency_ms: float = 42.0,
        input_tokens: int = 100,
        output_tokens: int = 5,
    ) -> None:
        self.teacher_intent_id = teacher_intent_id
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def test_teacher_adapter_records_one_hot_and_tokens() -> None:
    label_to_id = {f"class_{i}": i for i in range(NUM_CLASSES)}
    adapter = TeacherAdapter(
        client=object(), label_to_id=label_to_id, taxonomy_block="..."
    )
    fake_labels = [
        _FakeLabeledMessage(teacher_intent_id=3),
        _FakeLabeledMessage(teacher_intent_id=42),
    ]
    with patch("src.labeling.labeler.label_message", side_effect=fake_labels):
        batch = adapter.predict(["msg_a", "msg_b"])

    assert batch.probs.shape == (2, NUM_CLASSES)
    np.testing.assert_allclose(batch.probs.sum(axis=1), 1.0)
    assert batch.probs[0, 3] == 1.0
    assert batch.probs[1, 42] == 1.0
    assert batch.prompt_tokens is not None
    assert batch.completion_tokens is not None
    np.testing.assert_array_equal(batch.prompt_tokens, [100, 100])
    np.testing.assert_array_equal(batch.completion_tokens, [5, 5])


def test_teacher_adapter_unknown_intent_uses_uniform() -> None:
    """Teacher returning unknown intent → uniform distribution (rows still sum to 1)."""
    label_to_id = {f"class_{i}": i for i in range(NUM_CLASSES)}
    adapter = TeacherAdapter(
        client=object(), label_to_id=label_to_id, taxonomy_block="..."
    )
    fake_labels = [_FakeLabeledMessage(teacher_intent_id=-1)]
    with patch("src.labeling.labeler.label_message", side_effect=fake_labels):
        batch = adapter.predict(["unknown"])
    np.testing.assert_allclose(batch.probs[0], 1.0 / NUM_CLASSES)
    assert pytest.approx(batch.probs.sum()) == 1.0
