"""Tests for src/training/data.py."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.training.data import TrainingData, load_training_data


def _make_parquet(
    tmp_path: Path,
    name: str,
    split: str,
    n: int,
    teacher_ids: list[int] | None = None,
    gold_ids: list[int] | None = None,
    prompt_version: str = "3",
    prompt_fingerprint: str = "ab411cfdce238586",
) -> Path:
    tids = teacher_ids if teacher_ids is not None else [0] * n
    gids = gold_ids if gold_ids is not None else [0] * n
    df = pd.DataFrame(
        {
            "text": [f"msg_{i}" for i in range(n)],
            "teacher_intent_name": ["card_arrival"] * n,
            "teacher_intent_id": tids,
            "input_tokens": [100] * n,
            "output_tokens": [10] * n,
            "latency_ms": [100.0] * n,
            "model_version": ["gemini-2.5-flash"] * n,
            "error": [None] * n,
            "gold_label_id": gids,
            "gold_label_name": ["card_arrival"] * n,
            "correct": [True] * n,
            "split": [split] * n,
            "prompt_version": [prompt_version] * n,
            "prompt_fingerprint": [prompt_fingerprint] * n,
        }
    )
    path = tmp_path / name
    df.to_parquet(path, index=False)
    return path


@pytest.fixture(autouse=True)
def mock_banking77_loader() -> Iterator[MagicMock]:
    """Stub load_banking77 to avoid downloading the HF dataset in tests."""
    with patch("src.training.data.load_banking77") as mock:
        mock.return_value = MagicMock(
            id_to_label={i: f"class_{i}" for i in range(77)},
            label_to_id={f"class_{i}": i for i in range(77)},
        )
        yield mock


def test_load_training_data_returns_expected_shapes(tmp_path: Path) -> None:
    train_p = _make_parquet(tmp_path, "train.parquet", split="train", n=10)
    val_p = _make_parquet(tmp_path, "val.parquet", split="val", n=5)

    data = load_training_data(str(train_p), str(val_p))

    assert isinstance(data, TrainingData)
    assert len(data.train_texts) == 10
    assert len(data.train_labels) == 10
    assert len(data.val_texts) == 5
    assert len(data.val_teacher_labels) == 5
    assert len(data.val_true_labels) == 5
    assert data.prompt_version == "3"
    assert data.prompt_fingerprint == "ab411cfdce238586"
    assert len(data.id_to_label) == 77


def test_load_training_data_drops_unknown_intent_rows(tmp_path: Path) -> None:
    train_p = _make_parquet(
        tmp_path,
        "train.parquet",
        split="train",
        n=5,
        teacher_ids=[0, 1, -1, 2, -1],
    )
    val_p = _make_parquet(tmp_path, "val.parquet", split="val", n=3)

    data = load_training_data(str(train_p), str(val_p))

    assert len(data.train_texts) == 3
    assert -1 not in data.train_labels


def test_load_training_data_preserves_both_val_label_columns(tmp_path: Path) -> None:
    """Val carries two independent label columns; both must survive the load."""
    train_p = _make_parquet(tmp_path, "train.parquet", split="train", n=2)
    val_p = _make_parquet(
        tmp_path,
        "val.parquet",
        split="val",
        n=3,
        teacher_ids=[1, 2, 3],
        gold_ids=[1, 5, 3],
    )

    data = load_training_data(str(train_p), str(val_p))

    assert data.val_teacher_labels == [1, 2, 3]
    assert data.val_true_labels == [1, 5, 3]


def test_load_training_data_rejects_wrong_split_in_train_file(tmp_path: Path) -> None:
    train_p = _make_parquet(tmp_path, "train.parquet", split="val", n=5)
    val_p = _make_parquet(tmp_path, "val.parquet", split="val", n=3)

    with pytest.raises(ValueError, match="Expected only split='train'"):
        load_training_data(str(train_p), str(val_p))


def test_load_training_data_rejects_wrong_split_in_val_file(tmp_path: Path) -> None:
    train_p = _make_parquet(tmp_path, "train.parquet", split="train", n=5)
    val_p = _make_parquet(tmp_path, "val.parquet", split="train", n=3)

    with pytest.raises(ValueError, match="Expected only split='val'"):
        load_training_data(str(train_p), str(val_p))


def test_load_training_data_rejects_mismatched_fingerprints(tmp_path: Path) -> None:
    train_p = _make_parquet(
        tmp_path, "train.parquet", split="train", n=5, prompt_fingerprint="fp_1"
    )
    val_p = _make_parquet(
        tmp_path, "val.parquet", split="val", n=3, prompt_fingerprint="fp_2"
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_training_data(str(train_p), str(val_p))


def test_load_training_data_rejects_mismatched_versions(tmp_path: Path) -> None:
    train_p = _make_parquet(
        tmp_path, "train.parquet", split="train", n=5, prompt_version="3"
    )
    val_p = _make_parquet(
        tmp_path, "val.parquet", split="val", n=3, prompt_version="2"
    )

    with pytest.raises(ValueError, match="version mismatch"):
        load_training_data(str(train_p), str(val_p))


def test_load_training_data_rejects_mixed_fingerprints_within_split(
    tmp_path: Path,
) -> None:
    """One labeling run produces one fingerprint; mixed values indicate a bug."""
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "teacher_intent_name": ["x", "y"],
            "teacher_intent_id": [0, 1],
            "input_tokens": [100, 100],
            "output_tokens": [10, 10],
            "latency_ms": [100.0, 100.0],
            "model_version": ["gemini-2.5-flash"] * 2,
            "error": [None, None],
            "gold_label_id": [0, 1],
            "gold_label_name": ["x", "y"],
            "correct": [True, True],
            "split": ["train", "train"],
            "prompt_version": ["3", "3"],
            "prompt_fingerprint": ["fp_a", "fp_b"],  # mixed
        }
    )
    train_p = tmp_path / "train.parquet"
    df.to_parquet(train_p, index=False)
    val_p = _make_parquet(tmp_path, "val.parquet", split="val", n=2)

    with pytest.raises(ValueError, match="Mixed prompt fingerprints"):
        load_training_data(str(train_p), str(val_p))
