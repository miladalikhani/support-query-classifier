import pandas as pd
import pytest

from src.data.banking77 import NUM_CLASSES, Banking77Splits, load_banking77

HF_TRAIN_SIZE = 10003
HF_TEST_SIZE = 3080


@pytest.fixture(scope="module")
def splits() -> Banking77Splits:
    return load_banking77(seed=42)


def test_split_totals(splits: Banking77Splits) -> None:
    assert len(splits.train) + len(splits.val) == HF_TRAIN_SIZE
    assert len(splits.test) == HF_TEST_SIZE


def test_no_overlap_between_splits(splits: Banking77Splits) -> None:
    train_texts = set(splits.train["text"])
    val_texts = set(splits.val["text"])
    test_texts = set(splits.test["text"])
    assert not (train_texts & val_texts)
    assert not (train_texts & test_texts)
    assert not (val_texts & test_texts)


def test_all_classes_present_in_train(splits: Banking77Splits) -> None:
    assert splits.train["label"].nunique() == NUM_CLASSES


def test_label_maps_are_consistent(splits: Banking77Splits) -> None:
    assert len(splits.id_to_label) == NUM_CLASSES
    assert len(splits.label_to_id) == NUM_CLASSES
    for label_id, label_name in splits.id_to_label.items():
        assert splits.label_to_id[label_name] == label_id


def test_no_nulls_in_text_or_label(splits: Banking77Splits) -> None:
    for df in (splits.train, splits.val, splits.test):
        assert df["text"].notna().all()
        assert df["label"].notna().all()


def test_loader_is_deterministic_for_same_seed() -> None:
    a = load_banking77(seed=42)
    b = load_banking77(seed=42)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.val, b.val)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_different_seeds_produce_different_splits() -> None:
    a = load_banking77(seed=42)
    b = load_banking77(seed=7)
    assert not a.train.equals(b.train)


def test_stratified_split_preserves_class_distribution(splits: Banking77Splits) -> None:
    train_dist = splits.train["label"].value_counts(normalize=True).sort_index()
    val_dist = splits.val["label"].value_counts(normalize=True).sort_index()
    max_drift = (train_dist - val_dist).abs().max()
    assert max_drift < 0.02
