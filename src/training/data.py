"""Training-data loader for the student.

Reads teacher-labeled parquets from a URI (GCS or local) and exposes the
texts and labels in a shape ready for downstream training and calibration.

Val rows carry two label columns side by side:
    - teacher label  -> training target / early-stopping signal
    - ground truth   -> confidence-calibration target

These must stay separate downstream: calibrating routing confidence against
the teacher would only measure agreement with the teacher, not real-world
correctness. Train rows carry only the teacher label.
"""

from dataclasses import dataclass

import pandas as pd
import structlog

from src.data.banking77 import load_banking77

log = structlog.get_logger()


@dataclass(frozen=True)
class TrainingData:
    """Texts and labels for student training.

    `train_labels` and `val_teacher_labels` are teacher predictions; use them
    as training targets and as the early-stopping signal respectively.

    `val_true_labels` is the held-out ground truth. Treat it as calibration
    data only — using it as a training signal collapses the held-out boundary
    that downstream evaluation depends on.
    """

    train_texts: list[str]
    train_labels: list[int]
    val_texts: list[str]
    val_teacher_labels: list[int]
    val_true_labels: list[int]
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]
    prompt_version: str
    prompt_fingerprint: str


def _read_split_parquet(uri: str, expected_split: str) -> pd.DataFrame:
    """Read a labels parquet and fail loudly if it contains the wrong split.

    A parquet's `split` column is the authoritative tag for what it holds.
    The argument acts as a second check: callers say what they expect, the
    file must agree. Prevents accidentally training on data meant for eval.
    """
    df = pd.read_parquet(uri)
    splits_present = sorted(df["split"].unique().tolist())
    if splits_present != [expected_split]:
        raise ValueError(
            f"Expected only split={expected_split!r} rows in {uri}, "
            f"got {splits_present}"
        )
    return df


def _assert_single_fingerprint(df: pd.DataFrame, label: str) -> None:
    fingerprints = df["prompt_fingerprint"].unique()
    if len(fingerprints) != 1:
        raise ValueError(
            f"Mixed prompt fingerprints within {label}: {sorted(fingerprints)}. "
            f"All rows in a labeling run must share one fingerprint."
        )


def load_training_data(
    teacher_train_uri: str,
    teacher_val_uri: str,
) -> TrainingData:
    """Load teacher-labeled train and val parquets into a TrainingData bundle.

    URIs may be local paths or `gs://` (pandas handles both transparently).
    Both inputs must come from the same labeling run — same prompt version
    and content fingerprint — otherwise the function raises ValueError.
    """
    log.info(
        "loading_training_data",
        train_uri=teacher_train_uri,
        val_uri=teacher_val_uri,
    )

    train_df = _read_split_parquet(teacher_train_uri, expected_split="train")
    val_df = _read_split_parquet(teacher_val_uri, expected_split="val")

    _assert_single_fingerprint(train_df, "train")
    _assert_single_fingerprint(val_df, "val")
    train_fp = train_df["prompt_fingerprint"].iloc[0]
    val_fp = val_df["prompt_fingerprint"].iloc[0]
    if train_fp != val_fp:
        raise ValueError(
            f"Prompt fingerprint mismatch: train={train_fp!r}, val={val_fp!r}. "
            f"Train and val must be labeled with the same teacher prompt."
        )

    train_pv = train_df["prompt_version"].iloc[0]
    val_pv = val_df["prompt_version"].iloc[0]
    if train_pv != val_pv:
        raise ValueError(
            f"Prompt version mismatch: train={train_pv!r}, val={val_pv!r}."
        )

    train_before = len(train_df)
    val_before = len(val_df)
    train_df = train_df[train_df["teacher_intent_id"] != -1].reset_index(drop=True)
    val_df = val_df[val_df["teacher_intent_id"] != -1].reset_index(drop=True)
    log.info(
        "dropped_unknown_intent",
        train_dropped=train_before - len(train_df),
        val_dropped=val_before - len(val_df),
    )

    # training/ cannot depend on evaluation/; read label maps from src.data.
    banking77 = load_banking77()
    id_to_label = banking77.id_to_label
    label_to_id = banking77.label_to_id

    log.info(
        "training_data_loaded",
        train_rows=len(train_df),
        val_rows=len(val_df),
        prompt_version=train_pv,
        prompt_fingerprint=train_fp,
    )

    return TrainingData(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["teacher_intent_id"].astype(int).tolist(),
        val_texts=val_df["text"].tolist(),
        val_teacher_labels=val_df["teacher_intent_id"].astype(int).tolist(),
        val_true_labels=val_df["gold_label_id"].astype(int).tolist(),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        prompt_version=str(train_pv),
        prompt_fingerprint=str(train_fp),
    )
