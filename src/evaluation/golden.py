import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

GOLDEN_VERSION = "v1"
GOLDEN_ROOT = Path("data") / "golden" / GOLDEN_VERSION
EXAMPLES_PATH = GOLDEN_ROOT / "examples.parquet"
LABELS_PATH = GOLDEN_ROOT / "labels.json"

NUM_CLASSES = 77
EXPECTED_ROWS = 3080


@dataclass(frozen=True)
class GoldenSet:
    examples: pd.DataFrame
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]
    version: str


def load_golden() -> GoldenSet:
    """Load the locked golden test set, materializing from Banking77 on first call."""
    if not EXAMPLES_PATH.exists() or not LABELS_PATH.exists():
        _materialize()

    examples = pd.read_parquet(EXAMPLES_PATH)
    id_to_label = {int(k): v for k, v in json.loads(LABELS_PATH.read_text()).items()}
    label_to_id = {name: idx for idx, name in id_to_label.items()}

    return GoldenSet(
        examples=examples,
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        version=GOLDEN_VERSION,
    )


def _materialize() -> None:
    # Late import so the dependency on the data module is local to materialization.
    from src.data.banking77 import load_banking77

    splits = load_banking77()
    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    splits.test.to_parquet(EXAMPLES_PATH, index=False)
    LABELS_PATH.write_text(json.dumps(splits.id_to_label, indent=2))
