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

The production caller is ``scan_session_for_secret_candidates`` below, wired
to the CLI as ``polylogue ops scan-secrets --session <id>``
(``polylogue/cli/commands/scan_secrets.py``). Without a caller reading real
captured content and writing through ``record_secret_candidates``, the
regex/entropy rules and the non-injectable write path exist but never run
against an operator's actual archive (polylogue-27m fix round).
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.storage.sqlite.connection_profile import DB_TIMEOUT, READ_DB_TIMEOUT

# One-shot tier connections here mirror the pattern in security/excision.py:
# a direct connect (not open_connection/open_readonly_connection, so no
# sibling-tier attach) with the shared busy_timeout applied explicitly.
_READ_BUSY_TIMEOUT_MS = READ_DB_TIMEOUT * 1000
_WRITE_BUSY_TIMEOUT_MS = DB_TIMEOUT * 1000

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


@dataclass(frozen=True, slots=True)
class SecretScanResult:
    """Outcome of scanning one session's captured content for secret candidates.

    Never carries any matched literal -- only counts and the written
    assertion ids (which themselves are derived, one-way identifiers; see
    :func:`secret_candidate_assertion_id`).
    """

    session_id: str
    found: bool
    blocks_scanned: int = 0
    candidates_found: int = 0
    written_assertion_ids: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "found": self.found,
            "blocks_scanned": self.blocks_scanned,
            "candidates_found": self.candidates_found,
            "written_assertion_ids": list(self.written_assertion_ids),
        }


def scan_session_for_secret_candidates(
    archive_root: Path,
    session_id: str,
    *,
    now_ms: int | None = None,
) -> SecretScanResult:
    """Scan every block of ``session_id`` for credential-shaped spans.

    Reads each block's ``text`` and ``tool_input`` (tool-call arguments,
    where secrets often show up as ``key=value`` pairs or env assignments)
    from ``index.db``, runs them through
    :func:`scan_text_for_secret_candidates`, and persists any hits as
    non-injectable ``SECRET_CANDIDATE`` assertions in ``user.db`` via
    :func:`record_secret_candidates` -- keyed ``block:<block_id>`` so a
    later ``polylogue ops excise`` on that session/message/block clears the
    corresponding candidate (see ``ExcisionTarget``/``_target_refs``).

    This is the production entrypoint for the module-level scanner: it is
    what turns "the regex rules and write path exist" into "an operator
    running this against their archive actually gets a finding". Returns
    ``found=False`` (no mutation) when the session does not exist.
    """
    timestamp = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)

    index_db = archive_root / "index.db"
    if not index_db.exists():
        return SecretScanResult(session_id=session_id, found=False)

    conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    conn.execute(f"PRAGMA busy_timeout = {_READ_BUSY_TIMEOUT_MS}")
    try:
        session_row = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if session_row is None:
            return SecretScanResult(session_id=session_id, found=False)
        rows = conn.execute(
            "SELECT block_id, COALESCE(text, ''), COALESCE(tool_input, '') FROM blocks "
            "WHERE session_id = ? ORDER BY message_id, position",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    user_db = archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    user_conn = sqlite3.connect(user_db)
    user_conn.execute(f"PRAGMA busy_timeout = {_WRITE_BUSY_TIMEOUT_MS}")
    written: list[str] = []
    try:
        with user_conn:
            for block_id, text, tool_input in rows:
                combined = f"{text} {tool_input}".strip()
                if not combined:
                    continue
                spans = scan_text_for_secret_candidates(combined)
                if not spans:
                    continue
                written.extend(
                    record_secret_candidates(
                        user_conn,
                        target_ref=f"block:{block_id}",
                        spans=spans,
                        now_ms=timestamp,
                    )
                )
    finally:
        user_conn.close()

    return SecretScanResult(
        session_id=session_id,
        found=True,
        blocks_scanned=len(rows),
        candidates_found=len(written),
        written_assertion_ids=tuple(written),
    )


__all__ = [
    "SecretCandidateSpan",
    "SecretScanResult",
    "record_secret_candidates",
    "scan_session_for_secret_candidates",
    "scan_text_for_secret_candidates",
    "secret_candidate_assertion_id",
]
