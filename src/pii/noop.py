from src.pii.interface import RedactionResult


class NoopRedactor:
    """Identity redactor: returns the input unchanged with no spans.

    Useful as a safe default when redaction must be wired up but not yet
    active, and as a baseline for the regex redactor's tests.
    """

    def redact(self, text: str) -> RedactionResult:
        return RedactionResult(redacted_text=text, spans=())
