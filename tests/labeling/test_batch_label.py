"""Tests for batch_label.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.labeling.batch_label import (
    DEFAULT_COST_CAPS,
    VALID_SPLITS,
    _checkpoint_path,
    _parse_args,
    _read_checkpoint,
    _write_checkpoint_row,
    build_predictions_dataframe,
    build_prompt_snapshot,
    compute_summary,
    label_split,
    load_split,
)
from src.labeling.labeler import LabeledMessage


def _make_labeled(
    intent_id: int = 0,
    name: str = "x",
    in_tok: int = 100,
    out_tok: int = 5,
    error: str | None = None,
) -> LabeledMessage:
    return LabeledMessage(
        text="msg",
        teacher_intent_name=name,
        teacher_intent_id=intent_id,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=100.0,
        model_version="gemini-2.5-flash",
        error=error,
    )


# ---------- Split loaders ----------


@patch("src.labeling.batch_label.load_banking77")
def test_load_split_returns_val_from_banking77(mock_b77: MagicMock) -> None:
    val_df = pd.DataFrame({"text": ["v"], "label": [0]})
    mock_b77.return_value = MagicMock(val=val_df)
    df = load_split("val")
    pd.testing.assert_frame_equal(df, val_df)


@patch("src.labeling.batch_label.load_banking77")
def test_load_split_returns_train_from_banking77(mock_b77: MagicMock) -> None:
    train_df = pd.DataFrame({"text": ["t"], "label": [1]})
    mock_b77.return_value = MagicMock(train=train_df)
    df = load_split("train")
    pd.testing.assert_frame_equal(df, train_df)


@patch("src.labeling.batch_label.load_golden")
def test_load_split_returns_golden_from_golden_set(mock_golden: MagicMock) -> None:
    golden_df = pd.DataFrame({"text": ["g"], "label": [2]})
    mock_golden.return_value = MagicMock(examples=golden_df)
    df = load_split("golden")
    pd.testing.assert_frame_equal(df, golden_df)


def test_load_split_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        load_split("test")


# ---------- Checkpoint ----------


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.jsonl"
    _write_checkpoint_row(path, 0, _make_labeled(intent_id=11, name="a"), 0)
    _write_checkpoint_row(path, 1, _make_labeled(intent_id=22, name="b"), 1)

    results = _read_checkpoint(path)
    assert len(results) == 2
    idx0, lm0, g0 = results[0]
    assert idx0 == 0
    assert g0 == 0
    assert lm0.teacher_intent_id == 11
    assert lm0.teacher_intent_name == "a"


def test_read_checkpoint_returns_empty_when_missing(tmp_path: Path) -> None:
    assert _read_checkpoint(tmp_path / "nope.jsonl") == []


def test_checkpoint_path_includes_split_and_run_id() -> None:
    path = _checkpoint_path("val", "2026-05-12T00-00-00Z")
    assert "val" in str(path)
    assert "2026-05-12T00-00-00Z" in str(path)


# ---------- label_split ----------


def test_label_split_skips_indices_in_skip_set(tmp_path: Path) -> None:
    client = MagicMock()
    examples = [("m1", 0), ("m2", 1), ("m3", 0)]

    with patch("src.labeling.batch_label.label_message") as mock_label:
        mock_label.side_effect = lambda c, m, t, lookup: _make_labeled(
            intent_id=lookup.get("a", -1), name="a"
        )
        results = label_split(
            client=client,
            examples=examples,
            taxonomy_block="tax",
            label_to_id={"a": 0},
            checkpoint_path=tmp_path / "ck.jsonl",
            concurrency=2,
            max_cost_usd=10.0,
            skip_indices={1},
        )

    assert mock_label.call_count == 2
    assert sorted(idx for idx, _, _ in results) == [0, 2]


def test_label_split_writes_to_checkpoint(tmp_path: Path) -> None:
    client = MagicMock()
    examples = [("m1", 0), ("m2", 1)]
    ckpt = tmp_path / "ck.jsonl"

    with patch("src.labeling.batch_label.label_message") as mock_label:
        mock_label.side_effect = lambda c, m, t, lookup: _make_labeled()
        label_split(
            client=client,
            examples=examples,
            taxonomy_block="tax",
            label_to_id={"x": 0},
            checkpoint_path=ckpt,
            concurrency=1,
            max_cost_usd=10.0,
        )

    assert ckpt.exists()
    persisted = _read_checkpoint(ckpt)
    assert len(persisted) == 2


def test_label_split_halts_on_cost_cap(tmp_path: Path) -> None:
    client = MagicMock()
    examples = [("m", 0)] * 10

    # Each call: 1M input @ $0.30 + 100k output @ $2.50 = $0.30 + $0.25 = $0.55
    expensive = _make_labeled(intent_id=0, in_tok=1_000_000, out_tok=100_000)

    with patch("src.labeling.batch_label.label_message", return_value=expensive):
        results = label_split(
            client=client,
            examples=examples,
            taxonomy_block="tax",
            label_to_id={"x": 0},
            checkpoint_path=tmp_path / "ck.jsonl",
            concurrency=1,
            max_cost_usd=1.0,
        )

    # Cumulative cost passes $1.0 after the 2nd call; loop exits.
    assert len(results) < len(examples)


# ---------- Summary ----------


def test_compute_summary_has_per_split_fields() -> None:
    results = [
        (0, _make_labeled(intent_id=0, name="a", in_tok=1000, out_tok=10), 0),
        (1, _make_labeled(intent_id=1, name="b", in_tok=1000, out_tok=10), 1),
    ]
    summary = compute_summary(
        results,
        {0: "a", 1: "b"},
        split="val",
        model_version="m",
        wall_time_s=1.0,
        run_id="r",
        prompt_version="3",
        prompt_fp="fp",
    )
    assert summary["split"] == "val"
    assert summary["rows"] == 2
    assert summary["errors"] == 0
    assert summary["prompt"] == {"version": "3", "fingerprint": "fp"}
    assert summary["teacher_vs_gold_accuracy"] == 1.0
    assert "teacher_label_distribution" in summary
    assert summary["teacher_label_distribution"] == {"a": 1, "b": 1}


def test_compute_summary_counts_errors() -> None:
    results = [
        (0, _make_labeled(error="unknown_intent"), 0),
        (1, _make_labeled(), 0),
    ]
    summary = compute_summary(
        results, {0: "x"}, "val", "m", 1.0, "r", "3", "fp"
    )
    assert summary["errors"] == 1
    assert summary["error_rate"] == 0.5


# ---------- Predictions DF ----------


def test_predictions_dataframe_has_split_and_prompt_columns() -> None:
    results = [
        (0, _make_labeled(intent_id=0, name="a"), 0),  # correct
        (1, _make_labeled(intent_id=1, name="b"), 0),  # incorrect
    ]
    df = build_predictions_dataframe(results, {0: "a", 1: "b"}, "train", "3", "fp")
    assert (df["split"] == "train").all()
    assert (df["prompt_version"] == "3").all()
    assert (df["prompt_fingerprint"] == "fp").all()
    assert df["correct"].tolist() == [True, False]
    assert df["gold_label_name"].tolist() == ["a", "a"]


# ---------- Prompt snapshot ----------


def test_prompt_snapshot_has_required_keys() -> None:
    # Use real Banking77 class names since prompt_fingerprint looks them up
    # in BANKING77_DESCRIPTIONS by default.
    snapshot = build_prompt_snapshot(["card_arrival", "lost_or_stolen_card"], "tax_block")
    assert set(snapshot.keys()) >= {
        "prompt_version",
        "prompt_fingerprint",
        "model",
        "temperature",
        "instruction_header",
        "instruction_footer",
        "taxonomy_block",
        "response_schema",
    }
    assert snapshot["taxonomy_block"] == "tax_block"


# ---------- CLI ----------


def test_parse_args_requires_split() -> None:
    with pytest.raises(SystemExit):
        _parse_args([])


def test_parse_args_rejects_unknown_split() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--split", "bogus"])


def test_parse_args_defaults() -> None:
    args = _parse_args(["--split", "val"])
    assert args.split == "val"
    assert args.concurrency == 5
    assert args.dry_run is False
    assert args.yes is False
    assert args.no_upload is False
    assert args.max_cost_usd is None  # filled in by main() from DEFAULT_COST_CAPS


def test_parse_args_custom() -> None:
    args = _parse_args(
        ["--split", "train", "--n", "50", "--max-cost-usd", "0.5", "--yes", "--no-upload"]
    )
    assert args.split == "train"
    assert args.n == 50
    assert args.max_cost_usd == 0.5
    assert args.yes is True
    assert args.no_upload is True


def test_default_cost_caps_cover_all_splits() -> None:
    assert set(DEFAULT_COST_CAPS) == set(VALID_SPLITS)
    for cap in DEFAULT_COST_CAPS.values():
        assert cap > 0
