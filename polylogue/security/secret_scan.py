"""Candidate-only secret detection for captured content (polylogue-27m).

The detector finds spans that *look like* credentials in session text and
records them as non-injectable ``AssertionKind.SECRET_CANDIDATE`` assertions.
It never returns, stores, or logs the matched literal -- callers only ever
see a SHA-256 fingerprint (for idempotent re-detection), a byte length, a
pattern id, and span offsets into the *source* text. This is deliberately a
triage aid pointing an operator at content worth reviewing for excision, not
a leak-prevention boundary in its own right -- see the "Raw artifacts are not
content-redacted" decision in ``docs/security.md``, which this detector does
not change: it surfaces candidates, it does not gate reads.

``devtools test -k secret_candidate`` is the coverage anchor cited by
``docs/plans/security-privacy-coverage.yaml``'s
``security.captured-content-secret-detection`` gap.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility

# ---------------------------------------------------------------------------
# Pattern rules
# ---------------------------------------------------------------------------
#
# Each rule is (pattern_id, compiled regex, capture_group, apply_entropy_filter).
# ``capture_group`` selects which regex group is the candidate secret span
# (0 = the whole match). Rules are ordered tightest/most-specific first so
# that overlap de-duplication in ``scan_text_for_secret_candidates`` prefers
# the more informative pattern id when two rules could both match the same
# span. Only the free-form "generic credential assignment" rule is entropy
# filtered -- the named-format rules (AKIA-prefixed, gh*_, sk-ant-, JWTs,
# PEM headers, ...) are already narrow enough that entropy filtering would
# only create false negatives.

_PATTERN_RULES: tuple[tuple[str, re.Pattern[str], int, bool], ...] = (
    ("private-key-block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), 0, False),
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0, False),
    (
        "aws-secret-access-key",
        re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
        1,
        False,
    ),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"), 0, False),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,48}\b"), 0, False),
    ("anthropic-api-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,120}\b"), 0, False),
    ("openai-api-key", re.compile(r"\bsk-[A-Za-z0-9]{20,120}\b"), 0, False),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        0,
        False,
    ),
    (
        "generic-credential-assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|access[_-]?key)"
            r"\s*[:=]\s*['\"]?([A-Za-z0-9+/_=-]{20,200})['\"]?"
        ),
        1,
        True,
    ),
)

# Below this Shannon-entropy threshold (bits/char), a "generic assignment"
# match reads like an English word or a placeholder ("password=changeme"),
# not a real credential. Named-format rules above are not entropy filtered.
_MIN_ENTROPY_BITS_PER_CHAR = 3.0


@dataclass(frozen=True, slots=True)
class SecretCandidateSpan:
    """One detected candidate span. Never carries the matched literal."""

    pattern_id: str
    start: int
    end: int
    length: int
    fingerprint: str
    entropy_bits_per_char: float


def _shannon_entropy_bits_per_char(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def scan_text_for_secret_candidates(text: str) -> list[SecretCandidateSpan]:
    """Scan ``text`` for credential-shaped spans.

    Candidate-only: this function never returns, logs, or persists the
    matched substring, only derived, one-way metadata. Overlapping matches
    from different rules keep whichever rule (in ``_PATTERN_RULES`` order)
    claimed the span first.
    """
    hits: list[SecretCandidateSpan] = []
    claimed: list[tuple[int, int]] = []
    for pattern_id, pattern, group_index, apply_entropy_filter in _PATTERN_RULES:
        for match in pattern.finditer(text):
            try:
                start, end = match.span(group_index)
            except IndexError:
                continue
            if start == end:
                continue
            if any(start < claimed_end and end > claimed_start for claimed_start, claimed_end in claimed):
                continue
            matched_text = text[start:end]
            entropy = _shannon_entropy_bits_per_char(matched_text)
            if apply_entropy_filter and entropy < _MIN_ENTROPY_BITS_PER_CHAR:
                continue
            fingerprint = hashlib.sha256(matched_text.encode("utf-8", errors="surrogatepass")).hexdigest()
            hits.append(
                SecretCandidateSpan(
                    pattern_id=pattern_id,
                    start=start,
                    end=end,
                    length=end - start,
                    fingerprint=fingerprint,
                    entropy_bits_per_char=round(entropy, 3),
                )
            )
            claimed.append((start, end))
    hits.sort(key=lambda span: span.start)
    return hits


def secret_candidate_assertion_id(target_ref: str, span: SecretCandidateSpan) -> str:
    """Deterministic assertion id: re-scanning identical content is idempotent."""
    digest = hashlib.sha256()
    for part in (target_ref, span.pattern_id, span.fingerprint):
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"assertion-{AssertionKind.SECRET_CANDIDATE}:{digest.hexdigest()}"


def record_secret_candidates(
    conn: sqlite3.Connection,
    *,
    target_ref: str,
    spans: Sequence[SecretCandidateSpan],
    scope_ref: str = "insight:secret-scan@v1",
    now_ms: int,
) -> list[str]:
    """Write each span as a non-injectable ``SECRET_CANDIDATE`` assertion.

    Uses the shared ``upsert_assertion`` write chokepoint with
    ``author_kind="detector"``, which coerces the row to
    ``status=CANDIDATE`` with ``{"inject": false, "promotion_required": true}``
    regardless of what is passed here -- an automated detector cannot
    self-promote a candidate to authoritative/injectable (mirrors the
    existing pathology/transform-candidate writers). Returns the written
    assertion ids.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    written: list[str] = []
    for span in spans:
        assertion_id = secret_candidate_assertion_id(target_ref, span)
        upsert_assertion(
            conn,
            assertion_id=assertion_id,
            scope_ref=scope_ref,
            target_ref=target_ref,
            key=f"secret-candidate/{span.pattern_id}/{span.fingerprint[:16]}",
            kind=AssertionKind.SECRET_CANDIDATE,
            value={
                "pattern_id": span.pattern_id,
                "fingerprint_sha256": span.fingerprint,
                "length": span.length,
                "span": [span.start, span.end],
                "entropy_bits_per_char": span.entropy_bits_per_char,
            },
            author_ref=scope_ref,
            author_kind="detector",
            status=AssertionStatus.CANDIDATE,
            visibility=AssertionVisibility.PRIVATE,
            context_policy={"inject": False, "promotion_required": True},
            now_ms=now_ms,
        )
        written.append(assertion_id)
    return written


__all__ = [
    "SecretCandidateSpan",
    "record_secret_candidates",
    "scan_text_for_secret_candidates",
    "secret_candidate_assertion_id",
]
