"""Fast heuristic jailbreak / prompt-injection screen.

Pattern layer = milliseconds, catches the loud attacks. For semantic attacks,
enable the NeMo Guardrails self-check rail (USE_NEMO=true) or put Llama Guard
in front — see nemo_config/ and the GUIDE.
"""
import re
from dataclasses import dataclass

# Flexible qualifiers: "ignore all previous instructions", "ignore the rules",
# "disregard any prior system prompts" etc. all resolve to the same intent.
_JAILBREAK_PATTERNS = [
    (r"(ignore|disregard|forget)\s+(?:(?:all|any|the|your|my)\s+)*"
     r"(?:previous|prior|above|earlier|preceding|system)?\s*"
     r"(instructions?|rules?|prompts?|guidelines?|restrictions?|policies)"),

    (r"(bypass|override|disable|turn\s+off)\s+(?:(?:all|any|the|your)\s+)*"
     r"(?:safety|content|security)?\s*(filters?|guardrails?|restrictions?|safeguards?)"),

    (r"(reveal|print|show|repeat|output|leak)\s+(?:(?:me|us|your|the)\s+)*"
     r"(?:(?:hidden|initial|original|system|secret)\s+)+(prompt|instructions?|message)"),

    r"\bDAN\b.{0,40}(mode|jailbreak)",
    r"\bjailbreak\b",
    (r"you(?:'re| are)\s+(?:now\s+)?(free|unfiltered|unrestricted|"
     r"without (?:any )?restrictions)"),
    (r"pretend (?:that )?(?:you|to)\s+(?:have no|are not bound by|don'?t have)\s+"
     r"(?:any\s+)?(rules|guidelines|filters|restrictions)"),
    r"act as if you (?:have|had) no (restrictions|filters|guidelines|rules)",
    r"do anything now",
    r"developer mode",
    r"\bsudo mode\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _JAILBREAK_PATTERNS]


@dataclass
class JailbreakResult:
    blocked: bool
    matched: list[str]


def check(text: str) -> JailbreakResult:
    hits = [p.pattern for p in _COMPILED if p.search(text)]
    return JailbreakResult(blocked=bool(hits), matched=hits)
