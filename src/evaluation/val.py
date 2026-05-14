"""Val split accessor for the evaluation harness.

The val split lives in the same Banking77 source as the locked golden
test set but is not itself locked. The harness uses it for per-class
threshold fitting and (in the training package) for calibration. It is
surfaced from `src/evaluation/` so the runner doesn't have to import
anything in `src/training/` — the architectural boundary stays clean.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValSet:
    examples: pd.DataFrame  # columns: text, label
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]


def load_val() -> ValSet:
    """Return the Banking77 val split paired with the canonical label maps."""
    from src.data.banking77 import load_banking77

    splits = load_banking77()
    return ValSet(
        examples=splits.val,
        id_to_label=splits.id_to_label,
        label_to_id=splits.label_to_id,
    )
