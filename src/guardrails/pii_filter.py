"""Deterministic PII detection & masking — the layer that must NEVER be down.

Regex-based (emails, phones, SSNs, credit cards with Luhn check, API keys,
IPs) so it costs zero tokens and zero latency budget. LLM-based rails (NeMo)
sit ON TOP of this, never instead of it.
"""
import re
from dataclasses import dataclass

# Order matters: most-specific first so e.g. the phone regex can't consume
# the digit tail of an API key before the key pattern runs.
_PATTERNS: dict[str, re.Pattern] = {
    "API_KEY": re.compile(r"\b(sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(
        r"(?<![A-Za-z0-9._-])(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?![A-Za-z0-9])"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


@dataclass
class PIIResult:
    clean_text: str
    found: dict[str, int]        # kind -> count

    @property
    def has_pii(self) -> bool:
        return bool(self.found)


def scrub(text: str) -> PIIResult:
    found: dict[str, int] = {}

    def _mask(kind: str):
        def repl(m: re.Match) -> str:
            if kind == "CREDIT_CARD":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 16 and _luhn_ok(digits)):
                    return m.group()          # not a real card → leave alone
            found[kind] = found.get(kind, 0) + 1
            return f"[{kind}_REDACTED]"
        return repl

    for kind, pattern in _PATTERNS.items():
        text = pattern.sub(_mask(kind), text)
    return PIIResult(clean_text=text, found=found)
