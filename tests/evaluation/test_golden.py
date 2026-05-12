import ast
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.golden import EXPECTED_ROWS, NUM_CLASSES, GoldenSet, load_golden

REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    return load_golden()


def test_golden_set_has_expected_size(golden: GoldenSet) -> None:
    assert len(golden.examples) == EXPECTED_ROWS


def test_all_77_classes_present(golden: GoldenSet) -> None:
    assert golden.examples["label"].nunique() == NUM_CLASSES


def test_no_nulls_in_text_or_label(golden: GoldenSet) -> None:
    assert golden.examples["text"].notna().all()
    assert golden.examples["label"].notna().all()


def test_label_maps_are_consistent(golden: GoldenSet) -> None:
    assert len(golden.id_to_label) == NUM_CLASSES
    assert len(golden.label_to_id) == NUM_CLASSES
    for label_id, label_name in golden.id_to_label.items():
        assert golden.label_to_id[label_name] == label_id


def test_load_golden_is_bit_stable() -> None:
    a = load_golden()
    b = load_golden()
    pd.testing.assert_frame_equal(a.examples, b.examples)
    assert a.id_to_label == b.id_to_label
    assert a.version == b.version


def test_training_does_not_import_evaluation() -> None:
    """Architectural boundary: src/training/ must not depend on src/evaluation/."""
    training_root = REPO_ROOT / "src" / "training"
    offending = []
    for py_file in training_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src.evaluation"):
                        offending.append(f"{py_file.relative_to(REPO_ROOT)}: import {alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("src.evaluation")
            ):
                offending.append(f"{py_file.relative_to(REPO_ROOT)}: from {node.module} import ...")
    assert not offending, "training/ must not import evaluation/: " + "; ".join(offending)
