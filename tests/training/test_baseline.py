"""Tests for src/training/baseline.py."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.baseline import (
    BaselineModel,
    _cache_key,
    _embed,
    _parse_args,
    predict_proba,
    train_baseline,
)
from src.training.data import TrainingData


def _make_data(n_train: int = 20, n_val: int = 8) -> TrainingData:
    return TrainingData(
        train_texts=[f"text_{i}" for i in range(n_train)],
        train_labels=[i % 4 for i in range(n_train)],
        val_texts=[f"val_{i}" for i in range(n_val)],
        val_teacher_labels=[i % 4 for i in range(n_val)],
        val_true_labels=[(i + 1) % 4 for i in range(n_val)],
        id_to_label={i: f"class_{i}" for i in range(77)},
        label_to_id={f"class_{i}": i for i in range(77)},
        prompt_version="3",
        prompt_fingerprint="fp",
    )


def _deterministic_embed(
    texts: list[str], encoder_name: str, cache_dir: Path
) -> np.ndarray:
    """Stand-in for _embed: deterministic features from text content."""
    return np.array(
        [[float(ord(t[-1])), float(len(t)), float(len(t.split())), 1.0] for t in texts]
    )


# ---- Cache key ----


def test_cache_key_is_deterministic() -> None:
    k1 = _cache_key("enc", ["a", "b", "c"])
    k2 = _cache_key("enc", ["a", "b", "c"])
    assert k1 == k2
    assert len(k1) == 16


def test_cache_key_changes_with_texts() -> None:
    assert _cache_key("enc", ["a", "b"]) != _cache_key("enc", ["a", "c"])


def test_cache_key_changes_with_encoder() -> None:
    assert _cache_key("enc1", ["a"]) != _cache_key("enc2", ["a"])


def test_cache_key_changes_with_text_order() -> None:
    assert _cache_key("enc", ["a", "b"]) != _cache_key("enc", ["b", "a"])


# ---- Embed caching ----


@pytest.fixture
def patched_sentence_transformer() -> Iterator[MagicMock]:
    """Stub SentenceTransformer so tests don't download or load a real model."""
    with patch("sentence_transformers.SentenceTransformer") as ST:
        instance = MagicMock()
        instance.encode = lambda texts, **kw: np.array(
            [[float(len(t)), 1.0] for t in texts]
        )
        ST.return_value = instance
        yield ST


def test_embed_writes_cache_on_miss(
    patched_sentence_transformer: MagicMock, tmp_path: Path
) -> None:
    embeddings = _embed(["hi", "hello"], "enc", tmp_path)
    assert embeddings.shape == (2, 2)
    cached = list(tmp_path.glob("*.npy"))
    assert len(cached) == 1


def test_embed_reads_cache_on_hit(
    patched_sentence_transformer: MagicMock, tmp_path: Path
) -> None:
    _embed(["hi"], "enc", tmp_path)
    patched_sentence_transformer.reset_mock()

    embeddings = _embed(["hi"], "enc", tmp_path)

    assert embeddings.shape == (1, 2)
    patched_sentence_transformer.assert_not_called()


# ---- train_baseline / predict_proba ----


def test_train_baseline_returns_model_with_metadata(tmp_path: Path) -> None:
    data = _make_data(n_train=20)
    with patch("src.training.baseline._embed", side_effect=_deterministic_embed):
        model = train_baseline(data, cache_dir=tmp_path)

    assert isinstance(model, BaselineModel)
    assert model.n_train_examples == 20
    assert model.encoder_name.startswith("sentence-transformers/")
    assert len(model.id_to_label) == 77
    assert model.sklearn_version  # populated


def test_train_baseline_round_trip_predicts(tmp_path: Path) -> None:
    """Train then predict on new texts; outputs are well-formed probabilities."""
    data = _make_data(n_train=40, n_val=8)
    with patch("src.training.baseline._embed", side_effect=_deterministic_embed):
        model = train_baseline(data, cache_dir=tmp_path)
        probs = predict_proba(
            model, ["new_text_alpha", "new_text_beta"], cache_dir=tmp_path
        )

    # 4 distinct classes appear in train labels → 4 columns of probabilities
    assert probs.shape == (2, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)


# ---- CLI ----


def test_parse_args_requires_train_and_val_uris() -> None:
    with pytest.raises(SystemExit):
        _parse_args([])
    with pytest.raises(SystemExit):
        _parse_args(["--train-uri", "x"])


def test_parse_args_defaults() -> None:
    args = _parse_args(["--train-uri", "t.parquet", "--val-uri", "v.parquet"])
    assert args.train_uri == "t.parquet"
    assert args.val_uri == "v.parquet"
    assert args.encoder.startswith("sentence-transformers/")


# ---- predict_proba shape ----


def test_predict_proba_first_row_is_a_distribution(tmp_path: Path) -> None:
    data = _make_data()
    with patch("src.training.baseline._embed", side_effect=_deterministic_embed):
        model = train_baseline(data, cache_dir=tmp_path)
        probs = predict_proba(model, ["single_text"], cache_dir=tmp_path)

    assert probs.shape[0] == 1
    assert probs[0].min() >= 0.0
    assert probs[0].max() <= 1.0
    assert abs(probs[0].sum() - 1.0) < 1e-6
