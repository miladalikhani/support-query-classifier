from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RedactionSpan:
    """One PII match in the original text.

    `start`/`end` are character offsets into the original string. `kind` is a
    short tag like 'email', 'phone', 'card', 'account'. `original_hash` is a
    salted sha256 of the original substring — the raw value is never
    persisted alongside it.
    """

    start: int
    end: int
    kind: str
    original_hash: str


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    spans: tuple[RedactionSpan, ...]


@runtime_checkable
class Redactor(Protocol):
    """Anything that can take a string and return a redacted version of it."""

    def redact(self, text: str) -> RedactionResult: ...
