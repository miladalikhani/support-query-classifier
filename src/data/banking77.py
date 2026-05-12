from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_DATASET_ID = "PolyAI/banking77"
NUM_CLASSES = 77
DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 0.1


@dataclass(frozen=True)
class Banking77Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]


def load_banking77(
    seed: int = DEFAULT_SEED,
    val_size: float = DEFAULT_VAL_SIZE,
) -> Banking77Splits:
    """Load Banking77 with a deterministic stratified train/val split.

    The HuggingFace `train` split (10003 rows) is split 90/10 (default) into
    `train` and `val`. The HuggingFace `test` split (3080 rows) is returned
    unchanged. The same `seed` always produces byte-identical splits.
    """
    ds = load_dataset(HF_DATASET_ID)

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas().reset_index(drop=True)

    label_feature = ds["train"].features["label"]
    id_to_label = {i: label_feature.int2str(i) for i in range(label_feature.num_classes)}
    label_to_id = {name: idx for idx, name in id_to_label.items()}

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label"],
    )

    return Banking77Splits(
        train=train_split.reset_index(drop=True),
        val=val_split.reset_index(drop=True),
        test=test_df,
        id_to_label=id_to_label,
        label_to_id=label_to_id,
    )
