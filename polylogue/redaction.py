from __future__ import annotations

import re
from typing import List, Pattern, Tuple


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-[A-Za-z0-9]{20,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")
_GITHUB_PAT_RE = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")
_SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
_AWS_ACCESS_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_GOOGLE_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_\.]{20,}\b")


_REDACTIONS: List[Tuple[Pattern[str], str]] = [
    (_EMAIL_RE, "[redacted-email]"),
    (_ANTHROPIC_KEY_RE, "[redacted-key]"),
    (_OPENAI_KEY_RE, "[redacted-key]"),
    (_GITHUB_PAT_RE, "[redacted-token]"),
    (_GITHUB_TOKEN_RE, "[redacted-token]"),
    (_SLACK_TOKEN_RE, "[redacted-token]"),
    (_AWS_ACCESS_RE, "[redacted-aws-access]"),
    (_GOOGLE_KEY_RE, "[redacted-key]"),
    (_BEARER_RE, "Bearer [redacted-token]"),
]


def sanitize_text(text: str) -> str:
    """Mask common secrets (emails, API keys, tokens) in freeform text."""
    redacted = text
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


__all__ = ["sanitize_text"]

