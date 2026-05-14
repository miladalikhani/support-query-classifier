"""Tests for src/training/persistence.py."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.training.baseline import BaselineModel
from src.training.persistence import (
    _build_manifest,
    _extract_run_id,
    _hash_config,
    load_baseline_bundle,
    save_baseline_bundle,
)

# ---- _hash_config ----


def test_hash_config_is_deterministic() -> None:
    h1 = _hash_config({"a": 1, "b": 2})
    h2 = _hash_config({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 16


def test_hash_config_changes_with_values() -> None:
    assert _hash_config({"a": 1}) != _hash_config({"a": 2})


# ---- _extract_run_id ----


def test_extract_run_id_from_gcs_uri() -> None:
    uri = "gs://bucket/labeling/teacher_labels/v1/train/2026-05-12T14-22-28Z/labels.parquet"
    assert _extract_run_id(uri) == "2026-05-12T14-22-28Z"


def test_extract_run_id_from_local_path() -> None:
    assert _extract_run_id("data/teacher_labels/val/2026-05-12T08-55-21Z/labels.parquet") == (
        "2026-05-12T08-55-21Z"
    )


# ---- _build_manifest ----


def test_build_manifest_has_required_keys() -> None:
    with patch("src.training.persistence._git_info", return_value=("abc123", False)):
        manifest = _build_manifest(
            model_name="distilbert",
            teacher_train_uri="gs://b/labeling/teacher_labels/v1/train/T1/labels.parquet",
            teacher_val_uri="gs://b/labeling/teacher_labels/v1/val/V1/labels.parquet",
            prompt_version="3",
            prompt_fingerprint="ab411cfdce238586",
            training_config={"lr": 5e-5},
            val_accuracy_vs_teacher=0.89,
            val_accuracy_vs_truth=0.80,
            ece_pre=0.04,
            ece_post=0.02,
            temperature=1.5,
            trained_at_utc="2026-05-14T00:00:00Z",
        )
    expected_top = {
        "run_id",
        "schema_version",
        "model_name",
        "git_sha",
        "git_dirty",
        "training_config_hash",
        "teacher_labels",
        "training_metrics",
        "trained_at_utc",
        "bundled_at_utc",
        "framework_versions",
    }
    assert set(manifest.keys()) >= expected_top
    assert manifest["teacher_labels"]["train_run_id"] == "T1"
    assert manifest["teacher_labels"]["val_run_id"] == "V1"
    assert manifest["teacher_labels"]["prompt_fingerprint"] == "ab411cfdce238586"
    assert manifest["training_metrics"]["temperature"] == 1.5
    assert manifest["git_sha"] == "abc123"


# ---- Baseline round-trip ----


def _make_synthetic_baseline_model() -> BaselineModel:
    """A tiny BaselineModel with a real (trivial) LogisticRegression."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = (rng.integers(0, 3, 40)).tolist()
    clf = LogisticRegression(max_iter=200, random_state=0).fit(X, y)
    return BaselineModel(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        classifier=clf,
        id_to_label={i: f"class_{i}" for i in range(77)},
        label_to_id={f"class_{i}": i for i in range(77)},
        n_train_examples=40,
        sklearn_version="1.x",
        trained_at_utc=datetime.now(UTC).isoformat(),
    )


def test_save_baseline_bundle_writes_expected_files(tmp_path: Path) -> None:
    model = _make_synthetic_baseline_model()
    bundle_dir = tmp_path / "baseline_bundle"
    with patch("src.training.persistence._git_info", return_value=("deadbeef", True)):
        save_baseline_bundle(
            model,
            bundle_dir,
            teacher_train_uri="gs://b/v1/train/RA/labels.parquet",
            teacher_val_uri="gs://b/v1/val/RB/labels.parquet",
            prompt_version="3",
            prompt_fingerprint="fp",
            training_config={"encoder_name": model.encoder_name},
            val_accuracy_vs_teacher=0.88,
            val_accuracy_vs_truth=0.81,
            ece_pre=0.03,
            ece_post=0.03,
            temperature=1.0,
        )
    expected = {
        "manifest.json",
        "label_maps.json",
        "temperature.json",
        "encoder_name.txt",
        "classifier.joblib",
    }
    assert {p.name for p in bundle_dir.iterdir()} == expected


def test_baseline_save_load_round_trips_predictions(tmp_path: Path) -> None:
    """Loaded classifier must produce identical predictions to the original."""
    model = _make_synthetic_baseline_model()
    bundle_dir = tmp_path / "bundle"
    with patch("src.training.persistence._git_info", return_value=("x", False)):
        save_baseline_bundle(
            model,
            bundle_dir,
            teacher_train_uri="gs://b/v1/train/R/labels.parquet",
            teacher_val_uri="gs://b/v1/val/R/labels.parquet",
            prompt_version="3",
            prompt_fingerprint="fp",
            training_config={"encoder_name": model.encoder_name},
            val_accuracy_vs_teacher=0.88,
            val_accuracy_vs_truth=0.81,
            ece_pre=0.03,
            ece_post=0.03,
            temperature=1.0,
        )

    loaded = load_baseline_bundle(bundle_dir)
    assert loaded["encoder_name"] == model.encoder_name
    assert loaded["temperature"] == 1.0
    assert loaded["id_to_label"][0] == "class_0"
    assert len(loaded["id_to_label"]) == 77

    rng = np.random.default_rng(7)
    probe = rng.standard_normal((5, 4))
    np.testing.assert_array_equal(
        loaded["classifier"].predict(probe), model.classifier.predict(probe)
    )
    np.testing.assert_allclose(
        loaded["classifier"].predict_proba(probe),
        model.classifier.predict_proba(probe),
        rtol=1e-9,
    )


def test_load_baseline_bundle_raises_on_missing_manifest(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "broken"
    bundle_dir.mkdir()
    (bundle_dir / "temperature.json").write_text(json.dumps({"T": 1.0}))
    with pytest.raises(FileNotFoundError):
        load_baseline_bundle(bundle_dir)


# ---- CLI parsing ----


def test_parse_args_requires_subcommand() -> None:
    from src.training.persistence import _parse_args

    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_distilbert_subcommand() -> None:
    from src.training.persistence import _parse_args

    args = _parse_args(
        [
            "distilbert",
            "--run-dir",
            "data/models/distilbert/X",
            "--train-uri",
            "gs://b/train.parquet",
            "--val-uri",
            "gs://b/val.parquet",
        ]
    )
    assert args.command == "distilbert"
    assert args.upload is False


def test_parse_args_baseline_subcommand_with_upload() -> None:
    from src.training.persistence import _parse_args

    args = _parse_args(
        [
            "baseline",
            "--train-uri",
            "gs://b/train.parquet",
            "--val-uri",
            "gs://b/val.parquet",
            "--upload",
        ]
    )
    assert args.command == "baseline"
    assert args.upload is True
    assert args.encoder.startswith("sentence-transformers/")


# ---- Mock-based check that MagicMock baseline survives round-trip ----


def test_save_baseline_records_encoder_name_in_file(tmp_path: Path) -> None:
    model = _make_synthetic_baseline_model()
    bundle_dir = tmp_path / "bundle"
    with patch("src.training.persistence._git_info", return_value=("x", False)):
        save_baseline_bundle(
            model,
            bundle_dir,
            teacher_train_uri="gs://b/v1/train/R/labels.parquet",
            teacher_val_uri="gs://b/v1/val/R/labels.parquet",
            prompt_version="3",
            prompt_fingerprint="fp",
            training_config={"encoder_name": model.encoder_name},
            val_accuracy_vs_teacher=0.88,
            val_accuracy_vs_truth=0.81,
            ece_pre=0.03,
            ece_post=0.03,
            temperature=1.0,
        )
    contents = (bundle_dir / "encoder_name.txt").read_text().strip()
    assert contents == model.encoder_name


# Suppress unused-imports lints for MagicMock when patch is the only consumer above.
_unused = MagicMock
