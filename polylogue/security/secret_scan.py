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

``scan_archive_for_secret_candidates`` is the archive-wide sibling
(polylogue-layg.1): a single-session scan requires an operator to already
know a session id, so nothing discovers candidates archive-wide without one.
It is a bounded, resumable sweep over sessions the ops-tier
``secret_scan_status`` table has not yet covered at the current
``SECRET_SCAN_VERSION`` -- each page scans up to ``max_sessions`` pending
sessions, commits their findings and coverage rows together, and reports how
much work remains. Killing the process mid-sweep loses nothing: only
already-committed sessions are marked covered, so resuming re-derives the
still-pending set from the same table and never re-scans (or duplicates
findings for) a session already covered at the current version. Bumping
``SECRET_SCAN_VERSION`` (e.g. adding a pattern rule) invalidates every
existing coverage row, scheduling an intentional full rescan.
``polylogue ops scan-secrets --all`` (CLI) and
``polylogue.daemon.secret_scan_sweep`` (bounded daemon catch-up) are its two
production callers.
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
from polylogue.logging import get_logger
from polylogue.storage.sqlite.connection_profile import DB_TIMEOUT, READ_DB_TIMEOUT

logger = get_logger(__name__)

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


# ---------------------------------------------------------------------------
# Bulk/archive-wide scan (polylogue-layg.1)
# ---------------------------------------------------------------------------
#
# ``scan_session_for_secret_candidates`` above requires an operator to
# already know a session id. Nothing archive-wide called it, so an operator
# with no prior signal had no way to discover candidates at all. The bulk
# scanner below covers that gap as a bounded, checkpointed sweep: it selects
# sessions the ops-tier ``secret_scan_status`` table has not yet covered at
# the current scanner version, scans each one with the same rules and write
# chokepoint as the single-session path, and commits per-session coverage
# rows in the same transaction as the findings they cover -- so a kill mid
# sweep can never leave a session "covered" without its candidates durably
# recorded, or vice versa.

#: Bump when ``_PATTERN_RULES`` changes in a way that could surface new
#: candidates in previously-scanned content (new rule, widened regex, changed
#: entropy threshold). Every existing ``secret_scan_status`` row is written at
#: the version current when it was scanned, so a bump makes every prior row
#: stale and schedules an intentional rescan on the next sweep -- mirrors
#: ``EmbeddingRecipe``'s model/dimension versioning for the embed backlog.
SECRET_SCAN_VERSION = 1

#: Default bounded page size for one bulk-scan call (CLI ``--limit`` default,
#: daemon sweep window). A large archive is covered incrementally across
#: repeated calls rather than one unbounded scan.
DEFAULT_SECRET_SCAN_PAGE_SIZE = 200


@dataclass(frozen=True, slots=True)
class BulkSecretScanResult:
    """Outcome of one bounded archive-wide sweep.

    Never carries any matched literal or session content -- only counts, the
    scanned session ids (for progress reporting), and whether more pending
    work remains after this page.
    """

    sessions_scanned: int = 0
    blocks_scanned: int = 0
    candidates_found: int = 0
    errors: int = 0
    scanned_session_ids: tuple[str, ...] = field(default_factory=tuple)
    remaining_pending: int = 0

    @property
    def more_pending(self) -> bool:
        return self.remaining_pending > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "sessions_scanned": self.sessions_scanned,
            "blocks_scanned": self.blocks_scanned,
            "candidates_found": self.candidates_found,
            "errors": self.errors,
            "remaining_pending": self.remaining_pending,
            "more_pending": self.more_pending,
        }


def _attached_secret_scan_status_table_exists(conn: sqlite3.Connection, *, schema: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'table' AND name = 'secret_scan_status' LIMIT 1"
    ).fetchone()
    return row is not None


def select_pending_secret_scan_session_ids(
    index_db: Path,
    ops_db: Path,
    *,
    scanner_version: int = SECRET_SCAN_VERSION,
    limit: int = DEFAULT_SECRET_SCAN_PAGE_SIZE,
    origin: str | None = None,
    since_ms: int | None = None,
) -> list[str]:
    """Return up to ``limit`` session ids not yet covered at ``scanner_version``.

    A session is pending when it has no ``secret_scan_status`` row, or one
    recorded at an older ``scanner_version``. Ordered by ``session_id`` for a
    stable, deterministic sweep across repeated bounded calls: two calls with
    no coverage writes between them return the same page, and a call after a
    coverage-writing call naturally advances past it -- the coverage table
    itself is the resumable cursor, so no separate cursor row is needed.
    """
    if not index_db.exists():
        return []
    conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    conn.execute(f"PRAGMA busy_timeout = {_READ_BUSY_TIMEOUT_MS}")
    try:
        if ops_db.exists():
            conn.execute("ATTACH DATABASE ? AS ops", (str(ops_db),))
            has_status_table = _attached_secret_scan_status_table_exists(conn, schema="ops")
        else:
            has_status_table = False
        clauses = ["1 = 1"]
        params: list[object] = []
        if origin is not None:
            clauses.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            clauses.append("s.created_at_ms >= ?")
            params.append(since_ms)
        where_sql = " AND ".join(clauses)
        if has_status_table:
            query = f"""
                SELECT s.session_id
                FROM sessions AS s
                LEFT JOIN ops.secret_scan_status AS st ON st.session_id = s.session_id
                WHERE {where_sql}
                  AND (st.session_id IS NULL OR st.scanner_version < ?)
                ORDER BY s.session_id
                LIMIT ?
            """
            params.extend([scanner_version, limit])
        else:
            query = f"""
                SELECT s.session_id
                FROM sessions AS s
                WHERE {where_sql}
                ORDER BY s.session_id
                LIMIT ?
            """
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


def count_pending_secret_scan_sessions(
    index_db: Path,
    ops_db: Path,
    *,
    scanner_version: int = SECRET_SCAN_VERSION,
    origin: str | None = None,
    since_ms: int | None = None,
) -> int:
    """Count sessions not yet covered at ``scanner_version`` (status reporting)."""
    if not index_db.exists():
        return 0
    conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    conn.execute(f"PRAGMA busy_timeout = {_READ_BUSY_TIMEOUT_MS}")
    try:
        if ops_db.exists():
            conn.execute("ATTACH DATABASE ? AS ops", (str(ops_db),))
            has_status_table = _attached_secret_scan_status_table_exists(conn, schema="ops")
        else:
            has_status_table = False
        clauses = ["1 = 1"]
        params: list[object] = []
        if origin is not None:
            clauses.append("s.origin = ?")
            params.append(origin)
        if since_ms is not None:
            clauses.append("s.created_at_ms >= ?")
            params.append(since_ms)
        where_sql = " AND ".join(clauses)
        if has_status_table:
            query = f"""
                SELECT COUNT(*)
                FROM sessions AS s
                LEFT JOIN ops.secret_scan_status AS st ON st.session_id = s.session_id
                WHERE {where_sql}
                  AND (st.session_id IS NULL OR st.scanner_version < ?)
            """
            params.append(scanner_version)
        else:
            query = f"SELECT COUNT(*) FROM sessions AS s WHERE {where_sql}"
        row = conn.execute(query, params).fetchone()
        return int(row[0] or 0) if row is not None else 0
    finally:
        conn.close()


def _record_secret_scan_status(
    ops_conn: sqlite3.Connection,
    *,
    session_id: str,
    scanner_version: int,
    now_ms: int,
    blocks_scanned: int,
    candidates_found: int,
) -> None:
    ops_conn.execute(
        """
        INSERT INTO secret_scan_status
            (session_id, scanner_version, scanned_at_ms, blocks_scanned, candidates_found)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            scanner_version = excluded.scanner_version,
            scanned_at_ms = excluded.scanned_at_ms,
            blocks_scanned = excluded.blocks_scanned,
            candidates_found = excluded.candidates_found
        """,
        (session_id, scanner_version, now_ms, blocks_scanned, candidates_found),
    )


def scan_archive_for_secret_candidates(
    archive_root: Path,
    *,
    max_sessions: int = DEFAULT_SECRET_SCAN_PAGE_SIZE,
    scanner_version: int = SECRET_SCAN_VERSION,
    now_ms: int | None = None,
    origin: str | None = None,
    since_ms: int | None = None,
) -> BulkSecretScanResult:
    """Scan up to ``max_sessions`` not-yet-covered sessions for secret candidates.

    One bounded page of the archive-wide sweep (polylogue-layg.1): selects
    pending sessions via :func:`select_pending_secret_scan_session_ids`, scans
    each with the same rules and non-injectable write chokepoint as
    :func:`scan_session_for_secret_candidates`, and commits each session's
    findings together with its ``secret_scan_status`` coverage row in one
    ``user.db``/``ops.db`` transaction pair -- so an interrupted sweep never
    leaves a session marked covered without its candidates recorded, and a
    resumed sweep (a fresh call with no ``max_sessions`` change) naturally
    re-derives the still-pending set from the coverage table rather than
    starting over or duplicating findings. Callers loop on
    ``result.more_pending`` for a full-archive drain, or call once for a
    single bounded catch-up window (the daemon sweep shape).
    """
    timestamp = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)

    index_db = archive_root / "index.db"
    if not index_db.exists():
        return BulkSecretScanResult()

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db = archive_root / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    user_db = archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)

    pending_ids = select_pending_secret_scan_session_ids(
        index_db,
        ops_db,
        scanner_version=scanner_version,
        limit=max_sessions,
        origin=origin,
        since_ms=since_ms,
    )
    if not pending_ids:
        return BulkSecretScanResult(remaining_pending=0)

    index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    index_conn.execute(f"PRAGMA busy_timeout = {_READ_BUSY_TIMEOUT_MS}")
    user_conn = sqlite3.connect(user_db)
    user_conn.execute(f"PRAGMA busy_timeout = {_WRITE_BUSY_TIMEOUT_MS}")
    ops_conn = sqlite3.connect(ops_db)
    ops_conn.execute(f"PRAGMA busy_timeout = {_WRITE_BUSY_TIMEOUT_MS}")

    sessions_scanned = 0
    blocks_scanned_total = 0
    candidates_found_total = 0
    errors = 0
    scanned_ids: list[str] = []
    try:
        for session_id in pending_ids:
            try:
                rows = index_conn.execute(
                    "SELECT block_id, COALESCE(text, ''), COALESCE(tool_input, '') FROM blocks "
                    "WHERE session_id = ? ORDER BY message_id, position",
                    (session_id,),
                ).fetchall()
                written: list[str] = []
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
                with ops_conn:
                    _record_secret_scan_status(
                        ops_conn,
                        session_id=session_id,
                        scanner_version=scanner_version,
                        now_ms=timestamp,
                        blocks_scanned=len(rows),
                        candidates_found=len(written),
                    )
            except sqlite3.Error:
                logger.warning("secret scan sweep: session %r failed", session_id, exc_info=True)
                errors += 1
                continue
            sessions_scanned += 1
            blocks_scanned_total += len(rows)
            candidates_found_total += len(written)
            scanned_ids.append(session_id)
    finally:
        index_conn.close()
        user_conn.close()
        ops_conn.close()

    remaining = count_pending_secret_scan_sessions(
        index_db,
        ops_db,
        scanner_version=scanner_version,
        origin=origin,
        since_ms=since_ms,
    )
    return BulkSecretScanResult(
        sessions_scanned=sessions_scanned,
        blocks_scanned=blocks_scanned_total,
        candidates_found=candidates_found_total,
        errors=errors,
        scanned_session_ids=tuple(scanned_ids),
        remaining_pending=remaining,
    )


def scan_path_for_secret_candidates(path: Path, *, max_bytes: int = 20_000_000) -> list[SecretCandidateSpan]:
    """Scan a file already written to disk for credential-shaped spans.

    Companion to :func:`scan_text_for_secret_candidates` for render/export
    writers that stream content directly to a file handle rather than
    building it in memory first (e.g. ``read --view transcript --to file``'s
    non-lineage fast path, ``polylogue/cli/read_views/streaming_markdown.py``)
    -- those still need a scan chokepoint after the fact
    (polylogue-t9xd). Silently returns no findings for anything above
    ``max_bytes`` or unreadable as UTF-8: scanning multi-GB exports
    byte-for-byte would defeat the point of a streaming writer.
    """
    try:
        if path.stat().st_size > max_bytes:
            return []
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text_for_secret_candidates(text)


def describe_secret_candidate_spans(spans: Sequence[SecretCandidateSpan]) -> str:
    """One-line, no-literal summary of scan hits, safe for warning/log output.

    Shared by every caller that needs to *tell a human* a scan fired without
    reproducing this module's never-log-the-literal invariant per call site
    (the staged-content pre-commit gate and the render/export delivery path,
    polylogue-t9xd).
    """
    pattern_ids = sorted({span.pattern_id for span in spans})
    return f"{len(spans)} candidate span(s) ({', '.join(pattern_ids)})"


__all__ = [
    "BulkSecretScanResult",
    "DEFAULT_SECRET_SCAN_PAGE_SIZE",
    "SECRET_SCAN_VERSION",
    "SecretCandidateSpan",
    "SecretScanResult",
    "count_pending_secret_scan_sessions",
    "describe_secret_candidate_spans",
    "record_secret_candidates",
    "scan_archive_for_secret_candidates",
    "scan_path_for_secret_candidates",
    "scan_session_for_secret_candidates",
    "scan_text_for_secret_candidates",
    "select_pending_secret_scan_session_ids",
    "secret_candidate_assertion_id",
]
