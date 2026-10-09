"""Small helpers for live batch ingestion."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import re
import sqlite3
import time
from collections.abc import Awaitable, Buffer, Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, cast

import ijson

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import RetainedReplayOutcome
    from polylogue.sources.source_staging import SourceInputBinding
    from polylogue.sources.sqlite_inspection import SQLiteClassification

from polylogue.archive.artifact_taxonomy import (
    classify_artifact_path,
    strong_path_classification,
)
from polylogue.archive.raw_payload.decode import (
    jsonl_session_artifact,
)
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.core.json import loads as json_loads
from polylogue.core.raw_failure_evidence import PartialAdmission, RetainedRawDecodeRefusalError
from polylogue.core.sources import origin_from_provider
from polylogue.sources.acquisition_boundary import refuse_declared_foreign, refuse_foreign_path
from polylogue.sources.dispatch import (
    ForeignOriginContentError,
    detect_provider_from_raw_stream_evidence,
    is_jsonl_source_path,
)
from polylogue.sources.parsers import antigravity, codex_state, hermes_state, hermes_verification
from polylogue.sources.sqlite_snapshot import is_sqlite_path

_FULL_PARSE_PROGRESS_MAX_BYTES = 64 * 1024 * 1024
_FULL_PARSE_PROGRESS_MAX_FILES = 64
_MAX_APPEND_PLAN_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_APPEND_PLAN_GROUP_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_APPEND_PLAN_GROUP_FILES = 64
_BROWSER_CAPTURE_PREFIX_PROBE_BYTES = 1 * 1024 * 1024
_BROWSER_CAPTURE_PROVIDER_RE = re.compile(rb'"provider"\s*:\s*"([^"\\]{1,80})"')
_CURSOR_HASH_AUTHORITY_PREFIX = "sha256-prefix-v1"
_CLAUDE_FRONTIER_PREFIX = "claude-semantic-v1"


def _sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def encode_cursor_hash_authority(prefix_hash: str, tail_hash: str, *, ctime_ns: int) -> str:
    """Bind a complete accepted-prefix digest to its bounded tail digest."""
    normalized_prefix = prefix_hash.lower()
    normalized_tail = tail_hash.lower()
    if not _sha256_hex(normalized_prefix) or not _sha256_hex(normalized_tail):
        raise ValueError("cursor hash authority requires SHA-256 hex digests")
    if ctime_ns < 0:
        raise ValueError("cursor hash authority requires a non-negative ctime")
    return f"{_CURSOR_HASH_AUTHORITY_PREFIX}:{normalized_prefix}:{normalized_tail}:{ctime_ns}"


def cursor_prefix_hash(authority: str | None) -> str | None:
    if authority is None:
        return None
    parts = authority.split(":")
    if (
        len(parts) != 4
        or parts[0] != _CURSOR_HASH_AUTHORITY_PREFIX
        or not _sha256_hex(parts[1])
        or not _sha256_hex(parts[2])
        or not parts[3].isdigit()
    ):
        return None
    return parts[1]


def cursor_ctime_ns(authority: str | None) -> int | None:
    if cursor_prefix_hash(authority) is None:
        return None
    assert authority is not None
    return int(authority.rsplit(":", 1)[1])


def cursor_tail_hash(authority: str | None) -> str | None:
    """Return the bounded tail digest embedded in cursor authority."""
    if cursor_prefix_hash(authority) is None:
        return None
    assert authority is not None
    return authority.split(":")[2]


@dataclass(frozen=True, slots=True)
class ClaudeSemanticFrontier:
    """Accepted Claude Code observation: replaceable header plus stable body."""

    header_sha256: str
    body_sha256: str
    body_bytes: int


def encode_claude_semantic_frontier_digests(*, header_sha256: str, body_sha256: str, body_bytes: int) -> str:
    """Encode a Claude frontier from already-streamed semantic evidence."""
    if not (_sha256_hex(header_sha256) and _sha256_hex(body_sha256) and body_bytes >= 0):
        raise ValueError("invalid Claude semantic frontier evidence")
    return f"{_CLAUDE_FRONTIER_PREFIX}:{header_sha256}:{body_sha256}:{body_bytes}"


def decode_claude_semantic_frontier(value: str | None) -> ClaudeSemanticFrontier | None:
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) != 4 or parts[0] != _CLAUDE_FRONTIER_PREFIX:
        return None
    if not (_sha256_hex(parts[1]) and _sha256_hex(parts[2]) and parts[3].isdigit()):
        return None
    return ClaudeSemanticFrontier(parts[1], parts[2], int(parts[3]))


def claude_semantic_frontier_for_prefix(
    path: Path,
    end_offset: int,
    *,
    expected_stable_body_sha256: str | None = None,
    expected_stable_body_bytes: int | None = None,
) -> str | None:
    """Encode a Claude frontier ending at one accepted complete-record boundary."""
    frontier, _bytes_read = claude_semantic_frontier_for_prefix_with_bytes(
        path,
        end_offset,
        expected_stable_body_sha256=expected_stable_body_sha256,
        expected_stable_body_bytes=expected_stable_body_bytes,
    )
    return frontier


def claude_semantic_frontier_for_prefix_with_bytes(
    path: Path,
    end_offset: int,
    *,
    expected_stable_body_sha256: str | None = None,
    expected_stable_body_bytes: int | None = None,
) -> tuple[str | None, int]:
    """Return a Claude frontier and every byte consumed while proving it."""
    if (expected_stable_body_sha256 is None) != (expected_stable_body_bytes is None):
        raise ValueError("Claude stable-body proof requires both digest and byte boundary")
    if expected_stable_body_bytes is not None and expected_stable_body_bytes < 0:
        return None, 0
    bytes_read = 0
    try:
        with path.open("rb") as handle:
            header = handle.readline()
            bytes_read += len(header)
            if not header.endswith(b"\n") or len(header) > end_offset:
                return None, bytes_read
            json_loads(header)
            body_bytes = end_offset - len(header)
            body_hasher = hashlib.sha256()
            stable_body_hasher = hashlib.sha256()
            stable_body_remaining = expected_stable_body_bytes
            while body_bytes:
                line = handle.readline()
                bytes_read += len(line)
                if not line or len(line) > body_bytes or not line.endswith(b"\n"):
                    return None, bytes_read
                body_hasher.update(line)
                if stable_body_remaining:
                    if len(line) > stable_body_remaining:
                        return None, bytes_read
                    stable_body_hasher.update(line)
                    stable_body_remaining -= len(line)
                body_bytes -= len(line)
                if line.strip():
                    json_loads(line)
    except (OSError, UnicodeDecodeError, ValueError):
        return None, bytes_read
    if stable_body_remaining not in (None, 0):
        return None, bytes_read
    if expected_stable_body_sha256 is not None and stable_body_hasher.hexdigest() != expected_stable_body_sha256:
        return None, bytes_read
    return (
        encode_claude_semantic_frontier_digests(
            header_sha256=hashlib.sha256(header).hexdigest(),
            body_sha256=body_hasher.hexdigest(),
            body_bytes=end_offset - len(header),
        ),
        bytes_read,
    )


def _archive_blob_exists(archive_root: Path, blob_hash_hex: str) -> bool:
    """Return whether a content-addressed archive blob is present on disk."""
    normalized = blob_hash_hex.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        return False
    return (archive_root / "blob" / normalized[:2] / normalized[2:]).is_file()


class _FullIngestHeartbeat(Protocol):
    def __call__(
        self,
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None: ...


class _AttemptProgressEmitter(Protocol):
    def __call__(
        self,
        phase: str,
        *,
        current_path_override: Path | None = None,
        payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _AppendPlan:
    path: Path
    source_name: str
    start_offset: int
    last_complete_newline: int
    stat_size: int
    st_dev: int
    st_ino: int
    mtime_ns: int
    payload: bytes
    payload_hash: str
    cursor_fingerprint: str | None
    bytes_read: int
    # The physical coordinate frozen from the opened file; every append raw
    # records it.
    canonical_source_path: str
    # The profile identity captured with that coordinate (Hermes profiles);
    # the cursor written after publication carries this captured authority.
    captured_profile_key: str | None
    # Historical fixture/replay callers can preserve a source ordering index;
    # live watcher plans retain the legacy sentinel when no index is known.
    source_index: int = -1
    accepted_tail_hash: str | None = None
    ctime_ns: int | None = None
    accepted_prefix_hash: str | None = None
    authority_bytes_read: int = 0
    # The resolved logical session identity used to bind this append and as a
    # parser fallback when its own record stream cannot self-describe it.
    native_id_hint: str | None = None
    # Acquisition identity is deliberately separate from logical identity.
    # Codex append rows introduced this sidecar together with literal delta
    # bytes. Claude append rows predate it with native_id=NULL, so retaining
    # NULL keeps deterministic raw IDs stable across upgrades and retries.
    acquisition_native_id_hint: str | None = None
    accepted_claude_body_sha256: str | None = None
    accepted_claude_body_bytes: int | None = None
    accepted_claude_header_sha256: str | None = None
    accepted_claude_publication_body_sha256: str | None = None
    parser_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class _AppendResult:
    succeeded: list[_AppendPlan]
    failed: list[_AppendPlan]
    deferred: list[_AppendPlan] = field(default_factory=list)
    worker_count: int = 0
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    # Real session identity for each succeeded append plan (polylogue-20d.13):
    # the append route only ever grows a file whose session already exists
    # (a cursor-tracked prior observation), so every entry here is an
    # existing-session touch, never a newly created session.
    session_ids_by_path: dict[Path, str] = field(default_factory=dict)


class LiveRetainedRunner(Protocol):
    """The retained owner's acquired-raw publication, with its terminal refusals."""

    def __call__(
        self,
        raw_ids: Sequence[str],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
    ) -> Awaitable[RetainedReplayOutcome]: ...


class _DeferredAppend:
    pass


_DEFER_APPEND = _DeferredAppend()


class _ArchiveWriteCounts(Protocol):
    """What one full-ingest archive write reports to its path result."""

    @property
    def session_ids(self) -> Sequence[str]: ...

    @property
    def session_count(self) -> int: ...

    @property
    def message_count(self) -> int: ...

    @property
    def stage_timings_s(self) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class _FullIngestResult:
    succeeded: list[Path]
    failed: list[Path]
    source_payload_read_bytes: int
    acquired_raw_ids: tuple[str, ...] = ()
    # Accepted raw bytes awaiting worker completion or capacity. The cursor
    # schedules a full retry without consuming its finite failure budget.
    preparation_deferred: list[Path] = field(default_factory=list)
    # Source bytes could not yet be read. Retry without quarantining valid
    # input, independently of accepted-Raw worker preparation.
    source_read_deferred: list[Path] = field(default_factory=list)
    # Durably acquired source observations whose index authority is still
    # pending. They must wake the raw owner even with zero session writes.
    raw_deferred: list[Path] = field(default_factory=list)
    #: Planned paths this pass deliberately admitted nothing for, each with
    #: the typed reason. A planned path must land in exactly one of
    #: succeeded, failed, preparation_deferred, source_read_deferred, or here: one that lands in none is
    #: indistinguishable from an idle source (polylogue-6q16u).
    excluded: dict[Path, str] = field(default_factory=dict)
    #: Admitted paths whose provider is the source fallback only because
    #: detection crashed, with the failure. A shape fallback is not listed.
    detection_fallbacks: dict[Path, str] = field(default_factory=dict)
    #: Succeeded paths whose every record settled to a terminal outcome with
    #: nothing admissible (no session, or corrupt input), with the reason.
    #: The cursor advances like any success, so identical bytes are not
    #: re-parsed, but the intake outcome is an exclusion, never an admission
    #: (xf8qp).
    settled_exclusions: dict[Path, str] = field(default_factory=dict)
    #: Succeeded paths admitted only in part (a stable capture with a
    #: truncated final record), with what was left out (xf8qp).
    partial_admissions: dict[Path, PartialAdmission] = field(default_factory=dict)
    raw_fingerprints: dict[Path, str] = field(default_factory=dict)
    raw_byte_sizes: dict[Path, int] = field(default_factory=dict)
    raw_frontier_sizes: dict[Path, int] = field(default_factory=dict)
    raw_source_names: dict[Path, str] = field(default_factory=dict)
    raw_source_revisions: dict[Path, str] = field(default_factory=dict)
    raw_source_fingerprints: dict[Path, str] = field(default_factory=dict)
    captured_content_hashes: dict[Path, str] = field(default_factory=dict)
    captured_canonical_source_paths: dict[Path, str] = field(default_factory=dict)
    captured_profile_keys: dict[Path, str] = field(default_factory=dict)
    captured_file_observations: dict[Path, tuple[int, int, int, int, int]] = field(default_factory=dict)
    #: Wall-clock ns taken just before each captured observation's ``stat``.
    captured_observation_times_ns: dict[Path, int] = field(default_factory=dict)
    worker_count: int = 0
    ingested_session_count: int = 0
    ingested_message_count: int = 0
    changed_session_count: int = 0
    excised_skips: int = 0
    excised_paths: tuple[Path, ...] = ()
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    # Real session ids materialized by this full-ingest group (polylogue-20d.13),
    # threaded from the archive write's ``session_ids`` so callers can emit
    # identity-scoped SSE events instead of an unscoped aggregate.
    changed_session_ids: tuple[str, ...] = ()
    # polylogue-11cg9: True when a declared ``max_pass_seconds`` budget cut
    # this group short of its full input. Paths left out of both
    # ``succeeded`` and ``failed`` in that case were never attempted this
    # pass -- they remain ordinary backlog for the caller's next tick.
    time_budget_exceeded: bool = False
    #: Planned paths held back for publication order: each shares a
    #: canonical session with a path this group published, so it was not
    #: attempted here and publishes in the next group.
    ordering_held: list[Path] = field(default_factory=list)


def _full_ingest_result_from_summary(
    *,
    succeeded: list[Path],
    failed: list[Path],
    preparation_deferred: list[Path] | None = None,
    source_read_deferred: list[Path] | None = None,
    raw_deferred: list[Path] | None = None,
    source_payload_read_bytes: int,
    excluded: dict[Path, str] | None = None,
    detection_fallbacks: dict[Path, str] | None = None,
    settled_exclusions: dict[Path, str] | None = None,
    partial_admissions: dict[Path, PartialAdmission] | None = None,
    raw_fingerprints: dict[Path, str],
    raw_byte_sizes: dict[Path, int],
    raw_frontier_sizes: dict[Path, int] | None = None,
    raw_source_names: dict[Path, str] | None = None,
    raw_source_revisions: dict[Path, str] | None = None,
    raw_source_fingerprints: dict[Path, str] | None = None,
    captured_content_hashes: dict[Path, str] | None = None,
    captured_canonical_source_paths: dict[Path, str] | None = None,
    captured_profile_keys: dict[Path, str] | None = None,
    captured_file_observations: dict[Path, tuple[int, int, int, int, int]] | None = None,
    captured_observation_times_ns: dict[Path, int] | None = None,
    archive_write: _ArchiveWriteCounts | None,
    excised_skips: int = 0,
    excised_paths: tuple[Path, ...] = (),
    time_budget_exceeded: bool = False,
) -> _FullIngestResult:
    return _FullIngestResult(
        succeeded=succeeded,
        failed=failed,
        preparation_deferred=list(preparation_deferred or ()),
        source_read_deferred=list(source_read_deferred or ()),
        raw_deferred=list(raw_deferred or ()),
        source_payload_read_bytes=source_payload_read_bytes,
        excluded=dict(excluded or {}),
        detection_fallbacks=dict(detection_fallbacks or {}),
        settled_exclusions=dict(settled_exclusions or {}),
        partial_admissions=dict(partial_admissions or {}),
        raw_fingerprints=raw_fingerprints,
        raw_byte_sizes=raw_byte_sizes,
        raw_frontier_sizes=raw_frontier_sizes or {},
        raw_source_names=raw_source_names or {},
        raw_source_revisions=raw_source_revisions or {},
        raw_source_fingerprints=raw_source_fingerprints or {},
        captured_content_hashes=captured_content_hashes or {},
        captured_canonical_source_paths=captured_canonical_source_paths or {},
        captured_profile_keys=captured_profile_keys or {},
        captured_file_observations=captured_file_observations or {},
        captured_observation_times_ns=captured_observation_times_ns or {},
        worker_count=0 if archive_write is None else 1,
        ingested_session_count=0 if archive_write is None else archive_write.session_count,
        ingested_message_count=0 if archive_write is None else archive_write.message_count,
        changed_session_count=0 if archive_write is None else len(archive_write.session_ids),
        excised_skips=excised_skips,
        excised_paths=excised_paths,
        changed_session_ids=() if archive_write is None else tuple(archive_write.session_ids),
        stage_timings_s={} if archive_write is None else dict(archive_write.stage_timings_s),
        time_budget_exceeded=time_budget_exceeded,
    )


_FINGERPRINT_STREAM_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True, slots=True)
class JsonlBoundary:
    """The proven record prefix of one acquired JSONL byte sequence."""

    prefix_size: int
    record_count: int
    incomplete_tail: bool
    malformed_record: bool = False


_JSONL_BLANK_LINE_RE = re.compile(rb"(?:\A|\n)(?:[ \t\r]*\n|[ \t\r]+\Z)")


def _jsonl_record_count(prefix: bytes) -> int:
    """Count non-blank JSONL records without penalising the ordinary path.

    JSONL producers normally emit one record per physical line, so the C-level
    ``bytes.count`` fast path is sufficient.  Preserve the older treatment of
    blank lines when one is actually present, though: those lines are not
    records and must not change admission accounting.
    """
    if not prefix:
        return 0
    if _JSONL_BLANK_LINE_RE.search(prefix) is None:
        return prefix.count(b"\n") + (0 if prefix.endswith(b"\n") else 1)
    records = 0
    start = 0
    while start < len(prefix):
        end = prefix.find(b"\n", start)
        line = prefix[start:] if end < 0 else prefix[start:end]
        if line.strip():
            records += 1
        if end < 0:
            break
        start = end + 1
    return records


def _jsonl_tail_candidate(payload: bytes) -> tuple[int, bytes] | None:
    """Return the last non-blank physical line by walking only the tail."""
    end = len(payload)
    while end:
        start = payload.rfind(b"\n", 0, end) + 1
        candidate = payload[start:end].strip()
        if candidate:
            return start, candidate
        if start == 0:
            break
        # Exclude the preceding delimiter and inspect the physical line before
        # it.  This skips any number of terminal blank lines without scanning
        # earlier JSON records.
        end = start - 1
    return None


def jsonl_complete_prefix(payload: bytes) -> JsonlBoundary:
    """Find the maximal newline-terminated, syntactically valid JSON prefix.

    The boundary is decided from the tail because acquisition already retains
    the full payload and parsing remains responsible for validating earlier
    records.  Looking at every byte here made every full JSONL acquisition pay
    a second Python-level parse before its normal parser ran.

    A physical newline cannot occur inside valid JSON string data, so the
    final line is sufficient to distinguish a complete tail from an append in
    progress.  ``bytes.rfind`` and ``bytes.count`` run in C; only that final
    candidate is decoded here.
    """
    if not payload:
        return JsonlBoundary(0, 0, False)

    final_newline = payload.rfind(b"\n")
    complete_end = len(payload) if payload.endswith(b"\n") else final_newline + 1
    tail = _jsonl_tail_candidate(payload)
    if tail is None:
        return JsonlBoundary(complete_end, _jsonl_record_count(payload[:complete_end]), complete_end != len(payload))

    candidate_start, candidate = tail
    unterminated_tail = final_newline < 0 or candidate_start == final_newline + 1
    if not _valid_jsonl_tail(io.BytesIO(candidate), 0, len(candidate)):
        # An unterminated final line is an append in progress, not a malformed
        # record: the producer has not written its delimiter yet. Reporting it
        # as ``malformed_record`` suppressed ``complete_prefix_size`` in
        # ``batch.py``, which then classified an ordinary mid-write snapshot as
        # ``TERMINAL_CORRUPT_INPUT`` and withheld the already-complete records
        # before it. Only a newline-terminated line that does not decode is a
        # completed record the producer got wrong.
        return JsonlBoundary(
            candidate_start,
            _jsonl_record_count(payload[:candidate_start]),
            True,
            not unterminated_tail,
        )

    # The tail candidate itself is unterminated only when it follows the final
    # physical newline (or when the payload has none).  A valid such candidate
    # completes the whole payload; otherwise preserve the original cursor
    # boundary before an unfinished whitespace-only tail.
    if unterminated_tail:
        return JsonlBoundary(len(payload), _jsonl_record_count(payload), False)
    return JsonlBoundary(complete_end, _jsonl_record_count(payload[:complete_end]), complete_end != len(payload))


@dataclass(frozen=True, slots=True)
class JsonlFrontier:
    """The proven record frontier of a JSONL file, decided from its tail.

    The same ``prefix_size``/``incomplete_tail``/``malformed_record`` a
    :class:`JsonlBoundary` carries, without a record count: counting needs
    every byte, and no caller of the file route reads it.
    """

    prefix_size: int
    incomplete_tail: bool
    malformed_record: bool = False


#: The bytes ``bytes.strip()`` removes, so a line is blank exactly when the
#: bytes route would find it blank.
_JSONL_STRIP_BYTES = b" \t\n\r\x0b\x0c"
_JSONL_TAIL_READ_BYTES = 1 << 20


def jsonl_prefix_record_count(handle: IO[bytes], prefix_size: int, *, stop: Callable[[], bool] | None = None) -> int:
    """Count the non-blank JSONL records in the first ``prefix_size`` bytes of ``handle``.

    Streams in bounded chunks, so memory does not grow with the prefix. Only
    a partial admission reads it (the frontier itself is decided from the
    tail), so ordinary passes never pay for the count.
    """
    records = 0
    remaining = prefix_size
    line_has_content = False
    while remaining > 0:
        if stop is not None and stop():
            from polylogue.core.compute import DaemonOperationCancelled

            raise DaemonOperationCancelled("partial JSONL prefix record count")
        chunk = handle.read(min(_JSONL_TAIL_READ_BYTES, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        *complete_lines, open_line = chunk.split(b"\n")
        for line in complete_lines:
            if line_has_content or line.strip(_JSONL_STRIP_BYTES):
                records += 1
            line_has_content = False
        line_has_content = line_has_content or bool(open_line.strip(_JSONL_STRIP_BYTES))
    return records + (1 if line_has_content else 0)


def jsonl_complete_prefix_path(path: Path) -> JsonlFrontier:
    """Find the same JSONL frontier as :func:`jsonl_complete_prefix` from a file's tail."""
    with path.open("rb") as handle:
        return jsonl_frontier_of_handle(handle, path.stat().st_size)


def jsonl_frontier_of_handle(
    handle: IO[bytes], size: int, *, check_stop: Callable[[], None] | None = None
) -> JsonlFrontier:
    """The JSONL frontier of the first ``size`` bytes of a seekable handle, read from the tail.

    A physical newline cannot occur inside a valid JSON string, so the last
    non-blank line decides the frontier; it is located by reading backwards
    from the end and is the only record whose complete grammar is validated. Earlier bytes are never
    read -- the whole-file line walk this replaces read a 440 MB rollout in
    full, under the writer hold, to count records no caller used. This is
    the one file-side owner of the rule; :func:`jsonl_complete_prefix` is
    its in-memory twin, and the two are held equal by a differential test.
    """
    last_newline = _last_newline_before(handle, size, check_stop=check_stop)
    complete_end = last_newline + 1
    last_content = _last_content_byte_before(handle, size, check_stop=check_stop)
    if last_content < 0:
        return JsonlFrontier(complete_end, complete_end != size)
    candidate_start = _last_newline_before(handle, last_content, check_stop=check_stop) + 1
    candidate_end = _next_newline_at_or_after(handle, last_content + 1, size, check_stop=check_stop)
    candidate_terminated = candidate_end < size
    if not _valid_jsonl_tail(handle, candidate_start, last_content + 1, check_stop=check_stop):
        return JsonlFrontier(candidate_start, True, candidate_terminated)
    if candidate_terminated:
        return JsonlFrontier(complete_end, complete_end != size)
    return JsonlFrontier(size, False)


def _last_newline_before(handle: IO[bytes], end: int, *, check_stop: Callable[[], None] | None = None) -> int:
    """Offset of the last ``\\n`` before ``end``, or ``-1``."""
    while end > 0:
        if check_stop is not None:
            check_stop()
        start = max(0, end - _JSONL_TAIL_READ_BYTES)
        handle.seek(start)
        index = handle.read(end - start).rfind(b"\n")
        if index >= 0:
            return start + index
        end = start
    return -1


def _last_content_byte_before(handle: IO[bytes], end: int, *, check_stop: Callable[[], None] | None = None) -> int:
    """Offset of the last byte before ``end`` that ``bytes.strip`` keeps, or ``-1``."""
    while end > 0:
        if check_stop is not None:
            check_stop()
        start = max(0, end - _JSONL_TAIL_READ_BYTES)
        handle.seek(start)
        kept = handle.read(end - start).rstrip(_JSONL_STRIP_BYTES)
        if kept:
            return start + len(kept) - 1
        end = start
    return -1


def _next_newline_at_or_after(
    handle: IO[bytes], start: int, size: int, *, check_stop: Callable[[], None] | None = None
) -> int:
    """Offset of the first ``\\n`` at or after ``start``, or ``size``."""
    offset = start
    while offset < size:
        if check_stop is not None:
            check_stop()
        handle.seek(offset)
        window = handle.read(min(_JSONL_TAIL_READ_BYTES, size - offset))
        if not window:
            break
        index = window.find(b"\n")
        if index >= 0:
            return offset + index
        offset += len(window)
    return size


class _JsonlParsePrefix(io.RawIOBase):
    """Read the parser-accepted prefix without taking the opened input's custody."""

    def __init__(self, handle: IO[bytes], prefix_size: int) -> None:
        super().__init__()
        self._handle = handle
        self._size = prefix_size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return min(self._size, max(0, self._handle.tell()))

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_CUR:
            offset += self.tell()
        elif whence == io.SEEK_END:
            offset += self._size
        elif whence != io.SEEK_SET:
            raise ValueError("invalid JSONL prefix seek origin")
        return self._handle.seek(min(self._size, max(0, offset)))

    def read(self, size: int = -1) -> bytes:
        remaining = self._size - self.tell()
        return self._handle.read(remaining if size < 0 else min(size, remaining))

    def readinto(self, buffer: Buffer) -> int:
        view = memoryview(buffer).cast("B")
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)


@contextmanager
def jsonl_parse_input_of_handle(
    handle: IO[bytes], *, check_stop: Callable[[], None] | None = None
) -> Iterator[IO[bytes]]:
    """Expose the existing accepted-prefix law and restore the caller's offset.

    A malformed finished record remains in the view. Only the boundary owner
    can exclude an incomplete tail; neither the detector nor this adapter
    chooses a byte budget. Closing the view never closes its input.
    """
    original = handle.tell()
    view: _JsonlParsePrefix | None = None
    try:
        prefix_size = jsonl_parse_prefix_size_of_handle(handle, check_stop=check_stop)
        if prefix_size is None:
            yield handle
        else:
            view = _JsonlParsePrefix(handle, prefix_size)
            yield cast(IO[bytes], view)
    finally:
        if view is not None:
            view.close()
        handle.seek(original)


def _valid_jsonl_tail(handle: IO[bytes], start: int, end: int, *, check_stop: Callable[[], None] | None = None) -> bool:
    """Validate complete tail grammar without interpreting a record's identity."""
    from ijson.backends import python as exact_backend

    from polylogue.core.json_envelope import _PrefixStringReader

    # Match the bytes.strip() law of the in-memory frontier before decoding.
    while start < end:
        if check_stop is not None:
            check_stop()
        handle.seek(start)
        chunk = handle.read(min(_JSONL_TAIL_READ_BYTES, end - start))
        leading = len(chunk) - len(chunk.lstrip(_JSONL_STRIP_BYTES))
        start += leading
        if leading != len(chunk):
            break
    handle.seek(start)
    view = _JsonlParsePrefix(handle, end)
    first = view.read(4)
    view.seek(start)
    text = io.TextIOWrapper(io.BufferedReader(view), encoding=json.detect_encoding(first), errors="surrogatepass")

    callback_failure: BaseException | None = None

    class Utf8Input:
        def read(self, size: int = -1) -> bytes:
            if size == 0:
                return b""
            nonlocal callback_failure
            if check_stop is not None:
                try:
                    check_stop()
                except BaseException as exc:
                    callback_failure = exc
                    raise
            return text.read(16384).encode("utf-8", "surrogatepass")

    try:
        events = iter(exact_backend.basic_parse(_PrefixStringReader(Utf8Input(), syntax_only=True), use_float=True))
        if next(events, None) is None:
            return False
        for _event in events:
            pass
        return True
    except (UnicodeError, ijson.JSONError, ValueError):
        if callback_failure is not None:
            raise callback_failure from None
        return False
    finally:
        text.close()


def jsonl_parse_prefix_size(boundary: JsonlBoundary | JsonlFrontier, size: int) -> int | None:
    """The complete-record prefix a strict JSONL parse reads, or ``None`` for all of it.

    Only an unterminated tail -- an append in progress -- is left out, even
    when it is the whole payload (a capture taken before its first record
    finished). A newline-terminated final line that does not decode is a
    finished record, so the whole payload is parsed and the strict decoder
    refuses it. Live preparation and retained replay both read through this
    rule, so they parse the same records of the same bytes; whether an
    incomplete capture is a deferral or corrupt input is live intake's
    decision, recorded on the raw as failure evidence.
    """
    return boundary.prefix_size if 0 <= boundary.prefix_size < size and not boundary.malformed_record else None


def jsonl_parse_prefix_size_of_handle(handle: IO[bytes], *, check_stop: Callable[[], None] | None = None) -> int | None:
    """:func:`jsonl_parse_prefix_size` of a whole seekable handle, reading only its tail.

    The frontier comes from :func:`jsonl_frontier_of_handle`, the same
    tail-first rule every file route uses. The handle is left at offset 0
    for the decoder.
    """
    size = handle.seek(0, 2)
    frontier = jsonl_frontier_of_handle(handle, size, check_stop=check_stop)
    handle.seek(0)
    return jsonl_parse_prefix_size(frontier, size)


def fingerprint_file(path: Path, *, chunk_size: int = _FINGERPRINT_STREAM_CHUNK) -> tuple[str, int]:
    """Return (sha256, last_complete_newline_offset) by streaming the file.

    Streams the whole file once at ``chunk_size`` granularity rather than
    loading the entire payload into memory. The previous implementation read
    the whole file via ``Path.read_bytes()``, which produced a memory peak
    proportional to file size — a 1 GiB JSONL session held ~1 GiB resident
    just to compute its fingerprint after a successful full-ingest cursor
    write. The streaming version keeps the working set bounded by
    ``chunk_size`` independent of file size and is identical in output for
    files of any size.
    """
    hasher = hashlib.sha256()
    last_complete_newline = 0
    offset = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            newline_at = chunk.rfind(b"\n")
            if newline_at >= 0:
                last_complete_newline = offset + newline_at + 1
            offset += len(chunk)
    return hasher.hexdigest(), last_complete_newline


def sha256_range_from_path(
    path: Path,
    *,
    start_offset: int,
    end_offset: int,
    chunk_size: int = _FINGERPRINT_STREAM_CHUNK,
) -> tuple[str, int]:
    """Hash one exact byte range, rejecting a short read."""
    if start_offset < 0 or end_offset < start_offset:
        raise ValueError("invalid source byte range")
    hasher = hashlib.sha256()
    remaining = end_offset - start_offset
    bytes_read = 0
    with path.open("rb") as handle:
        handle.seek(start_offset)
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise EOFError(f"source ended before byte offset {end_offset}")
            hasher.update(chunk)
            bytes_read += len(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest(), bytes_read


def file_prefix_sha256(path: Path, end_offset: int, *, chunk_size: int = 1 << 20) -> str | None:
    """Digest ``path``'s leading ``end_offset`` bytes, or None if unreadable.

    Claude frontier authority is composed from the file that is on disk right
    now. Comparing this digest against the hash of the bytes actually retained
    for that prefix is what distinguishes "the retained prefix is still
    there, and the frontier describes it" from a same-length rewrite, which
    every offset- and size-based check accepts.
    """
    if end_offset < 0:
        return None
    digest = hashlib.sha256()
    remaining = end_offset
    try:
        with path.open("rb") as handle:
            while remaining:
                chunk = handle.read(min(chunk_size, remaining))
                if not chunk:
                    return None
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def tail_hash_from_path(path: Path, byte_size: int, *, chunk_size: int = 64 * 1024) -> tuple[str, int]:
    """Return a bounded hash of the recorded file tail."""
    if byte_size <= 0:
        return hashlib.sha256(b"").hexdigest(), 0
    start = max(0, byte_size - chunk_size)
    with path.open("rb") as handle:
        handle.seek(start)
        chunk = handle.read(byte_size - start)
    return hashlib.sha256(chunk).hexdigest(), len(chunk)


def tail_hash_and_last_complete_newline_from_path(
    path: Path, byte_size: int, *, chunk_size: int = 64 * 1024
) -> tuple[str, int, int]:
    """Return tail hash, last complete newline, and bytes read in one pass."""
    if byte_size <= 0:
        return hashlib.sha256(b"").hexdigest(), 0, 0
    bytes_read = 0
    end = byte_size
    tail_hash: str | None = None
    with path.open("rb") as handle:
        while end > 0:
            start = max(0, end - chunk_size)
            handle.seek(start)
            chunk = handle.read(end - start)
            bytes_read += len(chunk)
            if tail_hash is None:
                tail_hash = hashlib.sha256(chunk).hexdigest()
            newline_at = chunk.rfind(b"\n")
            if newline_at >= 0:
                return tail_hash, start + newline_at + 1, bytes_read
            end = start
    return tail_hash or hashlib.sha256(b"").hexdigest(), 0, bytes_read


def cursor_state_after_full_ingest(
    path: Path, byte_size: int, *, raw_fingerprint: str | None
) -> tuple[str, int, str, int]:
    if raw_fingerprint is None:
        fp, last_nl = fingerprint_file(path)
        tail_hash, _tail_bytes = tail_hash_from_path(path, byte_size)
        if path.suffix.lower() not in {".jsonl", ".ndjson"}:
            last_nl = byte_size
        return fp, last_nl, tail_hash, byte_size
    tail_hash, last_nl, bytes_read = tail_hash_and_last_complete_newline_from_path(path, byte_size)
    if path.suffix.lower() not in {".jsonl", ".ndjson"}:
        last_nl = byte_size
    return raw_fingerprint, last_nl, tail_hash, bytes_read


def last_complete_newline_from_tail(path: Path, byte_size: int, *, chunk_size: int = 64 * 1024) -> tuple[int, int]:
    if byte_size <= 0:
        return 0, 0
    bytes_read = 0
    end = byte_size
    with path.open("rb") as handle:
        while end > 0:
            start = max(0, end - chunk_size)
            handle.seek(start)
            chunk = handle.read(end - start)
            bytes_read += len(chunk)
            newline_at = chunk.rfind(b"\n")
            if newline_at >= 0:
                return start + newline_at + 1, bytes_read
            end = start
    return 0, bytes_read


def _ingest_pass_exhausted(
    *,
    max_pass_seconds: float | None,
    pass_started: float,
    checkpoint: str,
) -> bool:
    """Whether this pass must stop taking new work at ``checkpoint``.

    The caller's ``max_pass_seconds`` schedules the next work item.
    An item already acquired finishes and publishes its cursor before later
    work remains backlog. Writer hold thresholds are telemetry only.
    """
    del checkpoint
    return max_pass_seconds is not None and (time.monotonic() - pass_started) > max_pass_seconds


def _full_parse_progress_groups(paths: list[Path]) -> Iterable[list[Path]]:
    group: list[Path] = []
    group_bytes = 0
    for path in paths:
        byte_size = _path_size(path)
        if group and (
            len(group) >= _FULL_PARSE_PROGRESS_MAX_FILES or group_bytes + byte_size > _FULL_PARSE_PROGRESS_MAX_BYTES
        ):
            yield group
            group = []
            group_bytes = 0
        group.append(path)
        group_bytes += byte_size
    if group:
        yield group


def _append_plan_group_ready(plans: list[_AppendPlan]) -> bool:
    """Return true when pending append plans should be ingested now."""
    if len(plans) >= _MAX_APPEND_PLAN_GROUP_FILES:
        return True
    return sum(plan.bytes_read for plan in plans) >= _MAX_APPEND_PLAN_GROUP_PAYLOAD_BYTES


def _blob_copy_heartbeat(
    heartbeat: _FullIngestHeartbeat | None,
    *,
    path: Path,
    source_payload_read_bytes: int,
) -> Callable[[], None] | None:
    if heartbeat is None:
        return None

    def emit() -> None:
        heartbeat(
            "full_blob_copy",
            current_path=path,
            source_payload_read_bytes=source_payload_read_bytes,
        )

    return emit


def _throttled_phase_heartbeat(
    emit: _AttemptProgressEmitter,
    *,
    interval_s: float = 15.0,
) -> _FullIngestHeartbeat:
    """Throttle durable attempt updates while long file/worker phases run."""
    last_emitted = -interval_s

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        stage_payload: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        nonlocal last_emitted
        now = time.perf_counter()
        if not force and now - last_emitted < interval_s:
            return
        last_emitted = now
        emit(
            phase,
            current_path_override=current_path,
            payload_read_bytes=source_payload_read_bytes,
            stage_payload=stage_payload,
        )

    return heartbeat


def _path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _accumulate_stage_timings(target: dict[str, float], update: dict[str, float]) -> None:
    for stage_name, elapsed in update.items():
        target[stage_name] = target.get(stage_name, 0.0) + float(elapsed)


def _browser_capture_prefix_probe(path: Path) -> tuple[bool, Provider | None]:
    """Detect a browser-capture envelope and its provider for a large file.

    Externally serialized envelopes can place the unbounded
    ``raw_provider_payload`` before ``session.provider``. The prefix probe
    handles envelopes whose provider marker appears near the start. If the
    prefix confirms a capture but cannot identify its provider, a structural
    scan (:func:`_browser_capture_provider_from_path`) finds the marker
    regardless of field order without retaining the complete payload.
    """
    if path.suffix.lower() != ".json":
        return False, None
    try:
        with path.open("rb") as handle:
            prefix = handle.read(_BROWSER_CAPTURE_PREFIX_PROBE_BYTES)
    except OSError:
        return False, None
    if b"polylogue_capture_kind" not in prefix or b"browser_llm_session" not in prefix:
        return False, None
    match = _BROWSER_CAPTURE_PROVIDER_RE.search(prefix)
    if match is not None:
        try:
            provider_token = match.group(1).decode("utf-8")
        except UnicodeDecodeError:
            provider_token = None
        if provider_token is not None:
            provider = Provider.from_string(provider_token)
            if provider is not Provider.UNKNOWN:
                return True, provider
    # The prefix confirmed a browser capture but didn't yield a usable
    # provider marker -- ``session.provider`` may sit past this prefix.
    return True, _browser_capture_provider_from_path(path)


def _browser_capture_provider_from_path(path: Path) -> Provider | None:
    """Stream-parse a browser-capture envelope to find ``session.provider``.

    Memory-bounded regardless of payload size (an ``ijson`` event stream),
    mirroring ``source_acquisition_components._stream_browser_capture_provider``
    -- but reads directly from a filesystem path instead of the blob store,
    since this probe runs before the file has been copied into a blob.
    """
    try:
        with path.open("rb") as handle:
            for element_prefix, event, value in ijson.parse(handle):
                if event == "string" and element_prefix == "session.provider":
                    provider = Provider.from_string(str(value))
                    return provider if provider is not Provider.UNKNOWN else None
    except (OSError, ijson.JSONError):
        return None
    return None


def _detect_provider_from_path(path: Path, fallback_provider: Provider, *, json_document: bool = False) -> Provider:
    return detect_provider_from_path_evidence(path, fallback_provider, json_document=json_document)[0]


def detect_provider_from_path_evidence(
    path: Path,
    fallback_provider: Provider,
    *,
    json_document: bool = False,
    check_stop: Callable[[], None] | None = None,
) -> tuple[Provider, str | None]:
    """Detect a path's provider; the second value names a detection crash.

    ``fallback_provider`` is returned both when no detector claims the input
    (a shape outcome) and when reading or decoding the input failed. The
    second value is ``None`` for the former and describes the failure for the
    latter, so a batch can count payloads whose provider is the fallback only
    because detection crashed (polylogue-fkqxx).
    """
    callback_failure: BaseException | None = None

    def checkpoint() -> None:
        nonlocal callback_failure
        if check_stop is not None:
            try:
                check_stop()
            except BaseException as exc:
                callback_failure = exc
                raise

    checkpoint()
    if fallback_provider is Provider.HERMES and (json_document or path.suffix.lower() == ".json"):
        # Hermes snapshots have a streaming envelope recognizer. Avoid routing
        # them through the generic document sampler, whose fallback builds the
        # complete JSON object merely to confirm the already-declared provider.
        from polylogue.sources.decoder_json import hermes_snapshot_envelope

        try:
            with path.open("rb") as handle:
                if hermes_snapshot_envelope(handle) is not None:
                    return Provider.HERMES, None
        except OSError as exc:
            return fallback_provider, _crash(exc)
    if fallback_provider is Provider.ANTIGRAVITY and antigravity.looks_like_trajectory_db_path(path):
        return Provider.ANTIGRAVITY, None
    if fallback_provider in (Provider.HERMES, Provider.UNKNOWN) and (
        hermes_state.looks_like_state_db_path(path)
        or hermes_verification.looks_like_verification_evidence_db_path(path)
    ):
        return Provider.HERMES, None
    try:
        with path.open("rb") as handle:
            provider, evidence = detect_provider_from_raw_stream_evidence(
                handle,
                path.name,
                fallback_provider,
                truncated_tail_ok=is_jsonl_source_path(str(path)),
                check_stop=checkpoint,
            )
    except OSError as exc:
        if callback_failure is not None:
            raise callback_failure from None
        return fallback_provider, _crash(exc)
    return provider, evidence if evidence.startswith("stream decode error") else None


def _crash(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _jsonl_provider_and_session_artifact(
    path: Path,
    fallback_provider: Provider,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[Provider, bool, str | None]:
    """Classify complete accepted records before session-artifact evaluation."""
    from polylogue.sources.origin_specs import path_declaration_refuses_session

    # A ``raw-only`` declaration is terminal: its bytes are evidence and the
    # record shape cannot decide otherwise (polylogue-ximhz). Checked before
    # any content probe, so a prompt-history log -- whose rows carry the same
    # ``sessionId`` keys a transcript does -- is never session-parsed.
    if path_declaration_refuses_session(fallback_provider, path):
        return fallback_provider, False, None
    provider, detection_failure = detect_provider_from_path_evidence(path, fallback_provider, check_stop=checkpoint)
    if path_declaration_refuses_session(provider, path):
        return provider, False, detection_failure
    artifact = jsonl_session_artifact(path, provider=provider, source_path=path, check_stop=checkpoint)
    if artifact is not None:
        return provider, True, detection_failure
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return provider, path_classification.parse_as_session, detection_failure
    return provider, False, detection_failure


def _parse_path_as_session_artifact(path: Path, *, provider: Provider) -> bool:
    if provider is Provider.ANTIGRAVITY and antigravity.looks_like_trajectory_db_path(path):
        return True
    if provider is Provider.HERMES and (
        hermes_state.looks_like_state_db_path(path)
        or hermes_verification.looks_like_verification_evidence_db_path(path)
    ):
        return True
    if is_jsonl_source_path(str(path)):
        from polylogue.sources.origin_specs import path_declaration_refuses_session

        if path_declaration_refuses_session(provider, path):
            return False
        if jsonl_session_artifact(path, provider=provider) is not None:
            return True
        # A path rule may still rescue content the bounded scan could not
        # inspect, but never content the recognizer positively named a
        # non-session source: a generated extract sits in the provider's own
        # transcript directory, so location cannot outrank its provenance.
        from polylogue.sources.origin_specs import recognize_source_class

        recognition = recognize_source_class(provider, path)
        if recognition is not None and recognition.source_class != "session":
            return False
        path_classification = classify_artifact_path(path, provider=provider)
        return path_classification.parse_as_session if path_classification is not None else False
    path_classification = strong_path_classification(path, provider=provider)
    if path_classification is not None:
        return path_classification.parse_as_session
    # No path declaration establishes non-session material. The existing
    # canonical preparation must inspect the input and publish its typed
    # session/non-session/refusal outcome; a bounded object probe cannot
    # decide this for large valid files or files with arbitrary basenames.
    return True


_RETRYABLE_READ_ERRNOS = frozenset(
    {
        errno.EIO,
        errno.EACCES,
        errno.EPERM,
        errno.ESTALE,
        errno.ETIMEDOUT,
        errno.EAGAIN,
        errno.EBUSY,
        # Process-local and system-wide descriptor exhaustion can clear when
        # concurrent readers close their handles.
        errno.EMFILE,
        errno.ENFILE,
        errno.ENOSPC,
        errno.EDQUOT,
    }
)


class RetryableSourceReadError(RuntimeError):
    """A source read failed for a reason a later pass can clear.

    Raised by :func:`classify_pre_acquisition` so callers handle a retryable
    read as a typed outcome instead of catching raw SQLite or OS errors.
    """

    def __init__(self, path: Path, cause: BaseException) -> None:
        super().__init__(f"{path}: {cause}")
        self.path = path
        self.cause = cause


def retryable_read_fault(exc: BaseException) -> bool:
    """Whether a source read failed for a reason a later read can clear."""
    if isinstance(exc, RetryableSourceReadError):
        return True
    # SQLite snapshot acquisition's live-file adapter preserves the native
    # result code as the explicit cause of its OSError.
    if isinstance(exc, OSError) and isinstance(exc.__cause__, sqlite3.Error):
        return retryable_read_fault(exc.__cause__)
    sqlite_code = getattr(exc, "sqlite_errorcode", None)
    return (isinstance(exc, OSError) and exc.errno in _RETRYABLE_READ_ERRNOS) or (
        isinstance(exc, sqlite3.Error)
        and isinstance(sqlite_code, int)
        and sqlite_code & 0xFF
        in {
            sqlite3.SQLITE_IOERR,
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
            sqlite3.SQLITE_CANTOPEN,
            sqlite3.SQLITE_PERM,
        }
    )


@dataclass(frozen=True, slots=True)
class PreAcquisitionDecision:
    """Whether full intake retains a discovered file, and the sniff it used.

    ``excluded_reason`` is the typed cursor-exclusion reason, or ``None`` when
    intake retains the file's bytes. ``detected_provider`` and
    ``detection_crash`` carry the content sniff this decision already paid
    for, so the retaining branch does not sniff again.
    """

    excluded_reason: str | None
    detected_provider: Provider | None = None
    detection_crash: str | None = None
    #: The exclusion is a foreign-origin refusal, recorded as refused.
    refused: bool = False


def foreign_origin_exclusion(exc: ForeignOriginContentError) -> str:
    """The typed reason intake records for a refused foreign-origin file.

    Intake's cursor and the production baseline's exclusion carry the same
    reason, so a file intake refuses is never a revision the baseline demands.
    """
    return f"{exc.code}: {exc}"


def classify_pre_acquisition(
    path: Path,
    *,
    fallback_provider: Provider,
    source_only: bool,
    size_bytes: int,
    checkpoint: Callable[[], None] | None = None,
    source_binding: SourceInputBinding | None = None,
) -> PreAcquisitionDecision:
    """Decide whether full intake excludes ``path`` before retaining any bytes.

    This is the one authority for pre-acquisition exclusion. The full-ingest
    batch applies it to every file it acquires, and the cold-build production
    baseline applies it to every file discovery accepts, so a revision the
    baseline requires is always one intake retains. The branch order mirrors
    the batch's acquisition branches: an earlier retaining branch wins over a
    later exclusion rule.

    SQLite structural claims come from one proved connection to the accepted
    input. Its captured semantic coordinate owns declaration rules. Retryable
    read faults propagate before exclusion; invalid database bytes retain the
    ordinary unsupported classification.
    """
    sqlite_classification = None
    if is_sqlite_path(path):
        from polylogue.sources.source_staging import bind_source_input
        from polylogue.sources.sqlite_inspection import SQLiteClassification, classify_sqlite_source

        if source_binding is None:
            try:
                with bind_source_input(path) as binding:
                    return classify_pre_acquisition(
                        binding.source_path,
                        fallback_provider=fallback_provider,
                        source_only=source_only,
                        size_bytes=size_bytes,
                        checkpoint=checkpoint,
                        source_binding=binding,
                    )
            except (OSError, sqlite3.Error) as exc:
                if retryable_read_fault(exc):
                    raise RetryableSourceReadError(path, exc) from exc
                raise
        path = source_binding.source_path
        try:
            refuse_declared_foreign(path.name, fallback_provider)
        except ForeignOriginContentError as exc:
            return PreAcquisitionDecision(foreign_origin_exclusion(exc), refused=True)
        if checkpoint is not None:
            checkpoint()
        try:
            sqlite_classification = classify_sqlite_source(source_binding.source, source_binding=source_binding)
        except (OSError, sqlite3.Error) as exc:
            if retryable_read_fault(exc):
                raise RetryableSourceReadError(path, exc) from exc
            if not isinstance(exc, sqlite3.DatabaseError) or getattr(exc, "sqlite_errorcode", None) not in {
                None,
                sqlite3.SQLITE_NOTADB,
            }:
                raise
            sqlite_classification = SQLiteClassification(False, False, False, "unknown")
    try:
        decision = _classify_pre_acquisition(
            path,
            fallback_provider=fallback_provider,
            source_only=source_only,
            size_bytes=size_bytes,
            checkpoint=checkpoint,
            sqlite_classification=sqlite_classification,
        )
    except (OSError, sqlite3.Error) as exc:
        if retryable_read_fault(exc):
            raise RetryableSourceReadError(path, exc) from exc
        raise
    return decision


def classify_pre_writer_admissions(
    paths: Sequence[Path],
    *,
    fallback_provider: Provider,
    checkpoint: Callable[[], None] | None = None,
) -> dict[Path, PreAcquisitionDecision | Exception]:
    """Decide source-only intake admission for ``paths`` before the writer.

    A JSONL decision streams the file's records, so it is taken here, off the
    Source writer, and acquisition reads the result. A retryable read fault
    is returned for the caller to defer; a vanished file is returned as a
    failure. Any other fault
    propagates.
    """
    admissions: dict[Path, PreAcquisitionDecision | Exception] = {}
    for path in paths:
        if checkpoint is not None:
            checkpoint()
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            admissions[path] = exc
            continue
        try:
            admissions[path] = classify_pre_acquisition(
                path,
                fallback_provider=fallback_provider,
                source_only=True,
                size_bytes=size_bytes,
                checkpoint=checkpoint,
            )
        except RetryableSourceReadError as exc:
            admissions[path] = exc
    return admissions


def _classify_pre_acquisition(
    path: Path,
    *,
    fallback_provider: Provider,
    source_only: bool,
    size_bytes: int,
    checkpoint: Callable[[], None] | None,
    sqlite_classification: SQLiteClassification | None,
) -> PreAcquisitionDecision:
    from polylogue.sources.origin_specs import (
        artifact_rule_for_path,
        database_capability_for_provider,
        pre_acquisition_path_exclusion,
        recognize_source_class,
    )

    if path.suffix.lower() == ".zip":
        # ZIP members are admitted or excluded one by one by the member walk.
        return PreAcquisitionDecision(None)
    try:
        # A declared database of another origin is foreign by its name.
        refuse_declared_foreign(path.name, fallback_provider)
    except ForeignOriginContentError as exc:
        return PreAcquisitionDecision(foreign_origin_exclusion(exc), refused=True)
    if (
        fallback_provider is Provider.ANTIGRAVITY
        and path.suffix.lower() == ".pb"
        and antigravity.classify_source_path(path).role is antigravity.AntigravitySourceRole.CONVERSATION_PROTOBUF
    ):
        # Converted as one cohort through the vendor language server.
        return PreAcquisitionDecision(None)
    hermes_capability = database_capability_for_provider(Provider.HERMES)
    hermes_member = hermes_capability.member(path.name) if hermes_capability is not None else None
    hermes_owned_sqlite_name = (
        source_only
        and fallback_provider is Provider.HERMES
        and hermes_member is not None
        and hermes_member.disposition != "out-of-scope"
    )
    if (
        fallback_provider in {Provider.ANTIGRAVITY, Provider.UNKNOWN}
        and sqlite_classification is not None
        and sqlite_classification.antigravity
    ):
        # A schema-verified trajectory store is admitted by its schema, also
        # when staged in the import inbox, whose source carries no provider.
        return PreAcquisitionDecision(None)
    source_class = recognize_source_class(
        fallback_provider, path, source_only=source_only, sqlite_classification=sqlite_classification
    )
    if source_class is not None and source_class.source_class == "unsupported" and not hermes_owned_sqlite_name:
        # Unknown/config/cache material under a broad root is a typed
        # non-session observation -- unless the boundary finds another
        # origin's records in it, which is a refusal. A read fault raises.
        try:
            if sqlite_classification is None:
                refuse_foreign_path(path, fallback_provider)
        except ForeignOriginContentError as exc:
            return PreAcquisitionDecision(foreign_origin_exclusion(exc), refused=True)
        return PreAcquisitionDecision("unsupported source class")
    hermes_signature = sqlite_classification is not None and (
        sqlite_classification.hermes_state or sqlite_classification.hermes_verification
    )
    if hermes_owned_sqlite_name or (not source_only and hermes_signature):
        return PreAcquisitionDecision(None)
    if source_only and fallback_provider is Provider.HERMES and hermes_signature:
        # An undeclared name (a copied ``backup.db``) is a Hermes database by
        # its schema signature; acquisition retains its logical export.
        return PreAcquisitionDecision(None, Provider.HERMES)
    codex_capability = database_capability_for_provider(Provider.CODEX)
    codex_member = codex_capability.member(path.name) if codex_capability is not None else None
    if (
        codex_member is not None
        and codex_member.disposition != "out-of-scope"
        and (
            (source_only and fallback_provider is Provider.CODEX)
            or (sqlite_classification is not None and sqlite_classification.codex_kind in codex_state.IN_SCOPE_KINDS)
        )
    ):
        return PreAcquisitionDecision(None)
    if codex_member is not None:
        return PreAcquisitionDecision("declared out-of-scope or structurally unverified state database")
    origin_artifact_rule = artifact_rule_for_path(fallback_provider, str(path))
    jsonl = is_jsonl_source_path(str(path))
    if sqlite_classification is None:
        exclusion = pre_acquisition_path_exclusion(fallback_provider, path)
        if exclusion is not None:
            return PreAcquisitionDecision(exclusion)
    if source_only and not jsonl:
        return PreAcquisitionDecision(None)
    if sqlite_classification is not None:
        return PreAcquisitionDecision("path rule refuses session parsing", fallback_provider)
    if origin_artifact_rule is not None and origin_artifact_rule.parse_policy != "session":
        return PreAcquisitionDecision(None, fallback_provider)
    if jsonl:
        provider, parse_as_session, crash = (
            _jsonl_provider_and_session_artifact(path, fallback_provider)
            if checkpoint is None
            else _jsonl_provider_and_session_artifact(path, fallback_provider, checkpoint=checkpoint)
        )
        # An unknown JSONL cannot be safely excluded from acquire: the strict
        # parse route persists typed terminal evidence for empty and
        # malformed exports. Known-provider sidecars are excluded here, in
        # source-only intake as in the baseline, because their record-stream
        # classification is already authoritative; retaining one would settle
        # a sidecar as corrupt session input.
        if not parse_as_session and provider is not Provider.UNKNOWN:
            return PreAcquisitionDecision("declared artifact rule: not parsed as a session", provider, crash)
        return PreAcquisitionDecision(None, provider, crash)
    if path.suffix.lower() == ".json":
        return PreAcquisitionDecision(None, fallback_provider)
    provider, crash = detect_provider_from_path_evidence(path, fallback_provider)
    if not _parse_path_as_session_artifact(path, provider=provider):
        return PreAcquisitionDecision("path rule refuses session parsing", provider, crash)
    return PreAcquisitionDecision(None, provider, crash)


def hook_carrier_logical_source_key(*, provider: Provider, source_path: str) -> str:
    """Return the physical-carrier revision key under its acquisition origin."""
    return f"{origin_from_provider(provider).value}:{source_path}"


def bind_hook_carrier_baseline_revision(
    archive: Any,
    raw_id: str,
    *,
    provider: Provider,
    source_path: str,
    source_revision: str,
) -> str:
    """Bind an initially acquired carrier to its physical append chain."""
    logical_source_key = hook_carrier_logical_source_key(provider=provider, source_path=source_path)
    archive.bind_raw_revision(
        raw_id,
        RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=RawRevisionKind.FULL,
            source_revision=source_revision,
            acquisition_generation=0,
            authority=RawRevisionAuthority.ASSERTED,
        ),
    )
    return logical_source_key
