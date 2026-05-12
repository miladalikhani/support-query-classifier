import pytest

from src.data.banking77 import load_banking77
from src.labeling.taxonomy import (
    BANKING77_DESCRIPTIONS,
    assert_descriptions_complete,
    format_class_list,
)


@pytest.fixture(scope="module")
def id_to_label() -> dict[int, str]:
    return load_banking77().id_to_label


def test_every_banking77_class_has_a_description(id_to_label: dict[int, str]) -> None:
    assert_descriptions_complete(id_to_label, BANKING77_DESCRIPTIONS)


def test_no_extra_descriptions(id_to_label: dict[int, str]) -> None:
    extras = set(BANKING77_DESCRIPTIONS) - set(id_to_label.values())
    assert extras == set(), f"Unexpected descriptions: {sorted(extras)}"


def test_descriptions_are_non_empty_and_substantive() -> None:
    for name, desc in BANKING77_DESCRIPTIONS.items():
        assert desc.strip(), f"Empty description for class {name}"
        assert len(desc.split()) >= 5, f"Description for {name} is suspiciously short: {desc!r}"


def test_format_class_list_is_deterministic(id_to_label: dict[int, str]) -> None:
    a = format_class_list(id_to_label)
    b = format_class_list(id_to_label)
    assert a == b


def test_format_class_list_contains_every_class_and_description(
    id_to_label: dict[int, str],
) -> None:
    block = format_class_list(id_to_label)
    for name in id_to_label.values():
        assert name in block, f"Class {name} missing from formatted block"
        assert BANKING77_DESCRIPTIONS[name] in block, f"Description for {name} missing"


def test_format_class_list_has_one_line_per_class(id_to_label: dict[int, str]) -> None:
    block = format_class_list(id_to_label)
    assert block.count("\n") == len(id_to_label) - 1


def test_assert_descriptions_complete_raises_on_missing() -> None:
    fake_id_to_label = {0: "totally_made_up_class", 1: "activate_my_card"}
    with pytest.raises(ValueError, match="Missing descriptions for"):
        assert_descriptions_complete(fake_id_to_label, BANKING77_DESCRIPTIONS)
