"""Tests for src/training/distilbert.py.

Deliberately avoid loading real DistilBERT weights or running the training
loop. The end-to-end fine-tune is exercised by the smoke run on real data,
not by these unit tests.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.training.distilbert import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DistilbertConfig,
    _build_dataset,
    _compute_metrics,
    _parse_args,
)

# ---- Config defaults ----


def test_config_defaults_are_sane() -> None:
    cfg = DistilbertConfig()
    assert cfg.model_name == DEFAULT_MODEL_NAME
    assert cfg.max_length == DEFAULT_MAX_LENGTH
    assert 0 < cfg.learning_rate < 1
    assert cfg.num_train_epochs >= 1
    assert cfg.per_device_train_batch_size >= 1
    assert 0 <= cfg.label_smoothing_factor < 1
    assert cfg.early_stopping_patience >= 0


# ---- _compute_metrics ----


def test_compute_metrics_returns_perfect_accuracy_on_match() -> None:
    logits = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 10.0, 0.0],
        ]
    )
    labels = np.array([0, 1, 2, 1])
    assert _compute_metrics((logits, labels)) == {"accuracy": 1.0}


def test_compute_metrics_returns_zero_on_total_mismatch() -> None:
    logits = np.array([[10.0, 0.0], [10.0, 0.0]])
    labels = np.array([1, 1])
    assert _compute_metrics((logits, labels)) == {"accuracy": 0.0}


def test_compute_metrics_handles_partial_correct() -> None:
    logits = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    labels = np.array([0, 1, 1])  # last prediction is wrong
    result = _compute_metrics((logits, labels))
    assert result["accuracy"] == pytest.approx(2 / 3, abs=1e-6)


# ---- _build_dataset ----


@pytest.fixture
def fake_tokenizer() -> MagicMock:
    """Tokenizer-shaped callable that emits deterministic ids and respects max_length."""

    def encode(texts: list[str], **kw: object) -> dict[str, list[list[int]]]:
        max_len = int(kw.get("max_length", 128))  # type: ignore[arg-type]
        return {
            "input_ids": [list(range(min(len(t), max_len))) for t in texts],
            "attention_mask": [[1] * min(len(t), max_len) for t in texts],
        }

    tok = MagicMock(side_effect=encode)
    return tok


def test_build_dataset_carries_labels_and_tokenized_columns(
    fake_tokenizer: MagicMock,
) -> None:
    ds = _build_dataset(["hi", "hello"], [1, 2], fake_tokenizer, max_length=128)
    assert set(ds.column_names) == {"input_ids", "attention_mask", "labels"}
    assert ds["labels"] == [1, 2]
    assert len(ds["input_ids"]) == 2


def test_build_dataset_truncates_to_max_length(fake_tokenizer: MagicMock) -> None:
    long_text = "a" * 500
    ds = _build_dataset([long_text], [0], fake_tokenizer, max_length=64)
    assert len(ds["input_ids"][0]) == 64


# ---- CLI ----


def test_parse_args_requires_train_and_val_uris() -> None:
    with pytest.raises(SystemExit):
        _parse_args([])
    with pytest.raises(SystemExit):
        _parse_args(["--train-uri", "t.parquet"])


def test_parse_args_defaults() -> None:
    args = _parse_args(["--train-uri", "t.parquet", "--val-uri", "v.parquet"])
    assert args.train_uri == "t.parquet"
    assert args.val_uri == "v.parquet"
    assert args.model_name == DEFAULT_MODEL_NAME
    assert args.epochs == 4
    assert args.batch_size == 32
    assert args.learning_rate == 5e-5
    assert args.label_smoothing == 0.1
    assert args.seed == 42


def test_parse_args_custom_hparams() -> None:
    args = _parse_args(
        [
            "--train-uri",
            "t",
            "--val-uri",
            "v",
            "--epochs",
            "10",
            "--batch-size",
            "16",
            "--learning-rate",
            "1e-4",
            "--label-smoothing",
            "0.0",
            "--seed",
            "7",
        ]
    )
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.learning_rate == 1e-4
    assert args.label_smoothing == 0.0
    assert args.seed == 7
