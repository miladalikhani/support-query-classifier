import pytest

from src.data.banking77 import load_banking77
from src.pii import RedactionResult, RegexRedactor


@pytest.fixture
def redactor() -> RegexRedactor:
    return RegexRedactor(salt=b"test-salt")


# ---------- Email ----------


def test_redacts_simple_email(redactor: RegexRedactor) -> None:
    result = redactor.redact("Email me at user@example.com please")
    assert result.redacted_text == "Email me at [EMAIL] please"
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.kind == "email"
    assert span.start == 12
    assert span.end == 28


def test_redacts_complex_email(redactor: RegexRedactor) -> None:
    result = redactor.redact("a.b+c@sub.example.co.uk is mine")
    assert "[EMAIL]" in result.redacted_text
    assert result.spans[0].kind == "email"


def test_does_not_match_naked_at_username(redactor: RegexRedactor) -> None:
    result = redactor.redact("DM @johnsmith for help")
    assert result.redacted_text == "DM @johnsmith for help"
    assert result.spans == ()


# ---------- Phone ----------


@pytest.mark.parametrize(
    "text",
    [
        "Call +44 20 7946 0958 today",
        "Reach me at (415) 555-2671",
        "phone: 415-555-2671",
        "Try +1 415 555 2671 anytime",
    ],
)
def test_redacts_phone_in_various_formats(redactor: RegexRedactor, text: str) -> None:
    result = redactor.redact(text)
    assert "[PHONE]" in result.redacted_text
    assert any(s.kind == "phone" for s in result.spans)


def test_does_not_match_iso_date_as_phone(redactor: RegexRedactor) -> None:
    result = redactor.redact("Effective date: 2024-05-12")
    assert result.redacted_text == "Effective date: 2024-05-12"
    assert result.spans == ()


# ---------- Card (Luhn) ----------


def test_redacts_luhn_valid_card(redactor: RegexRedactor) -> None:
    # 4111111111111111 is a well-known Luhn-valid Visa test number
    result = redactor.redact("My card is 4111111111111111 expires soon")
    assert "[CARD]" in result.redacted_text
    assert any(s.kind == "card" for s in result.spans)


def test_redacts_card_with_separators(redactor: RegexRedactor) -> None:
    result = redactor.redact("Card: 4111 1111 1111 1111")
    assert "[CARD]" in result.redacted_text


def test_does_not_match_luhn_invalid_long_digits(redactor: RegexRedactor) -> None:
    # 1234567890123456 is 16 digits but fails Luhn
    result = redactor.redact("Order ID: 1234567890123456")
    assert "[CARD]" not in result.redacted_text


# ---------- Account ----------


def test_redacts_account_with_context_word(redactor: RegexRedactor) -> None:
    result = redactor.redact("account 12345678 needs review")
    assert result.redacted_text == "account [ACCT] needs review"
    assert any(s.kind == "account" for s in result.spans)


def test_redacts_account_with_alternate_context(redactor: RegexRedactor) -> None:
    for text in [
        "a/c 12345678",
        "acct 12345678",
        "acct. 12345678",
        "Account No. 12345678",
        "ACCOUNT NUMBER 12345678",
    ]:
        result = redactor.redact(text)
        assert "[ACCT]" in result.redacted_text, f"failed for: {text!r}"


def test_does_not_match_bare_long_digits_as_account(redactor: RegexRedactor) -> None:
    result = redactor.redact("Order reference 12345678 is open")
    assert "[ACCT]" not in result.redacted_text


# ---------- Multiple PII in one message ----------


def test_redacts_multiple_pii_in_one_message(redactor: RegexRedactor) -> None:
    text = "Email user@example.com or call 415-555-2671 about account 12345678"
    result = redactor.redact(text)
    kinds = {s.kind for s in result.spans}
    assert {"email", "phone", "account"} <= kinds
    assert "[EMAIL]" in result.redacted_text
    assert "[PHONE]" in result.redacted_text
    assert "[ACCT]" in result.redacted_text


def test_spans_are_returned_in_text_order(redactor: RegexRedactor) -> None:
    text = "Email user@example.com or call 415-555-2671"
    result = redactor.redact(text)
    starts = [s.start for s in result.spans]
    assert starts == sorted(starts)


# ---------- Hashing ----------


def test_hash_is_deterministic_for_same_salt() -> None:
    a = RegexRedactor(salt=b"salt1")
    b = RegexRedactor(salt=b"salt1")
    h1 = a.redact("hi user@example.com").spans[0].original_hash
    h2 = b.redact("hi user@example.com").spans[0].original_hash
    assert h1 == h2


def test_hash_differs_with_different_salt() -> None:
    a = RegexRedactor(salt=b"salt1")
    b = RegexRedactor(salt=b"salt2")
    h1 = a.redact("hi user@example.com").spans[0].original_hash
    h2 = b.redact("hi user@example.com").spans[0].original_hash
    assert h1 != h2


# ---------- Smoke test against real Banking77 messages ----------


def test_runs_on_banking77_sample_without_error(redactor: RegexRedactor) -> None:
    splits = load_banking77()
    for text in splits.train["text"].head(50):
        result = redactor.redact(text)
        assert isinstance(result, RedactionResult)
        assert isinstance(result.redacted_text, str)
