import hashlib
import re

from src.pii.interface import RedactionResult, RedactionSpan

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

PHONE_INTL_RE = re.compile(r"\+\d{1,3}(?:[\s\-.]?\d{1,4}){2,5}")
PHONE_US_PARENS_RE = re.compile(r"\(\d{3}\)\s*\d{3}[\s\-.]?\d{4}")
PHONE_US_SEP_RE = re.compile(r"\b\d{3}[\s\-.]\d{3}[\s\-.]\d{4}\b")
PHONE_PATTERNS = (PHONE_INTL_RE, PHONE_US_PARENS_RE, PHONE_US_SEP_RE)

CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

ACCOUNT_RE = re.compile(
    r"(?:account\s+(?:no\.?|number|#)?|a/c|acct\.?)\s*[#:.]?\s*(\d{8,12})\b",
    re.IGNORECASE,
)

TOKENS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "card": "[CARD]",
    "account": "[ACCT]",
}

_PHONE_MIN_DIGITS = 7
_PHONE_MAX_DIGITS = 15

Candidate = tuple[int, int, str, str]  # (start, end, kind, original_substring)


def _luhn_valid(digits: list[int]) -> bool:
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _phone_digit_count_ok(raw: str) -> bool:
    n = sum(1 for c in raw if c.isdigit())
    return _PHONE_MIN_DIGITS <= n <= _PHONE_MAX_DIGITS


def _overlaps(a: Candidate, b: Candidate) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


class RegexRedactor:
    """Regex-based PII redactor for email, phone, card (Luhn-validated), and
    account numbers in context. Designed to satisfy the `Redactor` Protocol so
    it can be swapped with a Cloud DLP implementation in production.
    """

    def __init__(self, salt: bytes) -> None:
        self._salt = salt

    def redact(self, text: str) -> RedactionResult:
        candidates: list[Candidate] = []

        for m in EMAIL_RE.finditer(text):
            candidates.append((m.start(), m.end(), "email", m.group(0)))

        for pattern in PHONE_PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group(0)
                if _phone_digit_count_ok(raw):
                    candidates.append((m.start(), m.end(), "phone", raw))

        for m in CARD_RE.finditer(text):
            raw = m.group(0)
            digits = [int(c) for c in raw if c.isdigit()]
            if _luhn_valid(digits):
                candidates.append((m.start(), m.end(), "card", raw))

        for m in ACCOUNT_RE.finditer(text):
            candidates.append((m.start(1), m.end(1), "account", m.group(1)))

        # Resolve overlaps: longest match wins; ties broken by earlier start.
        candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
        accepted: list[Candidate] = []
        for c in candidates:
            if any(_overlaps(c, a) for a in accepted):
                continue
            accepted.append(c)
        accepted.sort(key=lambda c: c[0])

        parts: list[str] = []
        spans: list[RedactionSpan] = []
        cursor = 0
        for start, end, kind, original in accepted:
            parts.append(text[cursor:start])
            parts.append(TOKENS[kind])
            spans.append(
                RedactionSpan(
                    start=start,
                    end=end,
                    kind=kind,
                    original_hash=self._hash(original),
                )
            )
            cursor = end
        parts.append(text[cursor:])

        return RedactionResult(redacted_text="".join(parts), spans=tuple(spans))

    def _hash(self, value: str) -> str:
        h = hashlib.sha256()
        h.update(self._salt)
        h.update(value.encode("utf-8"))
        return h.hexdigest()
