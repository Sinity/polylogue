"""Small helpers for live batch ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import ijson

from polylogue.archive.artifact_taxonomy import (
    classify_artifact,
    classify_artifact_path,
    strong_path_classification,
)
from polylogue.archive.raw_payload.decode import (
    JSONL_RECORD_INSPECTION_BYTES,
    _sample_jsonl_payload_with_detail,
    jsonl_session_artifact,
)
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDecodeError, JSONValue
from polylogue.core.json import loads as json_loads
from polylogue.pipeline.services.process_pool import select_ingest_worker_count
from polylogue.sources.dispatch import _detect_provider_from_raw_bytes, detect_provider, is_jsonl_source_path
from polylogue.sources.parsers import hermes_state, hermes_verification
from polylogue.storage.runtime import RawSessionRecord

_LARGE_FULL_PARSE_PROGRESS_BYTES = 64 * 1024 * 1024
_SMALL_FULL_PARSE_PROGRESS_MAX_BYTES = 64 * 1024 * 1024
_SMALL_FULL_PARSE_PROGRESS_MAX_FILES = 64
_STREAMING_FULL_INGEST_BYTES = 8 * 1024 * 1024
_MAX_APPEND_PLAN_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_APPEND_PLAN_GROUP_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_APPEND_PLAN_GROUP_FILES = 64
_DEFAULT_LIVE_FULL_INGEST_WORKERS = 1
_BROWSER_CAPTURE_PREFIX_PROBE_BYTES = 1 * 1024 * 1024
_BROWSER_CAPTURE_PROVIDER_RE = re.compile(rb'"provider"\s*:\s*"([^"\\]{1,80})"')
_CURSOR_HASH_AUTHORITY_PREFIX = "sha256-prefix-v1"
_CLAUDE_FRONTIER_PREFIX = "claude-semantic-v1"


def codex_append_payload(
    payload: bytes,
    *,
    identity: str,
    legacy_header: bool,
) -> bytes:
    """Encode the payload shape used by Codex append acquisition.

    New appends store the literal file delta. Rows written before the header
    retirement stored a compact synthetic ``session_meta`` record followed by
    that same delta. Keeping both shapes in one encoder lets replay reproduce
    historical rows without creating a second serialization rule.
    """
    if not legacy_header:
        return payload
    session_meta = json.dumps(
        {"type": "session_meta", "payload": {"id": identity}},
        separators=(",", ":"),
    ).encode()
    return session_meta + b"\n" + payload


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


def encode_claude_semantic_frontier(*, header: bytes, body: bytes) -> str:
    """Encode the evidence needed to resume after a mutable-header rewrite."""
    return encode_claude_semantic_frontier_digests(
        header_sha256=hashlib.sha256(header).hexdigest(),
        body_sha256=hashlib.sha256(body).hexdigest(),
        body_bytes=len(body),
    )


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


def claude_semantic_frontier_from_path(path: Path) -> tuple[str, int, int] | None:
    """Return frontier authority, body start, and complete body end.

    A first record and every body record must be complete. A partial first
    record is therefore deferred instead of being interpreted as an empty body.
    """
    try:
        end_offset = path.stat().st_size
    except OSError:
        return None
    frontier = claude_semantic_frontier_for_prefix(path, end_offset)
    if frontier is None:
        return None
    try:
        with path.open("rb") as handle:
            header = handle.readline()
    except OSError:
        return None
    return frontier, len(header), end_offset


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


class _DeferredAppend:
    pass


_DEFER_APPEND = _DeferredAppend()


@dataclass(frozen=True, slots=True)
class _FullIngestResult:
    succeeded: list[Path]
    failed: list[Path]
    source_payload_read_bytes: int
    raw_fingerprints: dict[Path, str] = field(default_factory=dict)
    raw_byte_sizes: dict[Path, int] = field(default_factory=dict)
    raw_frontier_sizes: dict[Path, int] = field(default_factory=dict)
    raw_source_names: dict[Path, str] = field(default_factory=dict)
    raw_source_revisions: dict[Path, str] = field(default_factory=dict)
    raw_source_fingerprints: dict[Path, str] = field(default_factory=dict)
    captured_content_hashes: dict[Path, str] = field(default_factory=dict)
    captured_file_observations: dict[Path, tuple[int, int, int, int, int]] = field(default_factory=dict)
    worker_count: int = 0
    ingested_session_count: int = 0
    ingested_message_count: int = 0
    changed_session_count: int = 0
    wal_bytes_before_checkpoint: int = 0
    wal_bytes_after_checkpoint: int = 0
    wal_checkpointed_pages: int = 0
    wal_busy_pages: int = 0
    wal_checkpoint_elapsed_s: float = 0.0
    wal_checkpoint_mode: str = "none"
    wal_checkpoint_error: str | None = None
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    # Real session ids materialized by this full-ingest group (polylogue-20d.13),
    # threaded from ``_IngestBatchSummary.changed_session_ids`` so callers can
    # emit identity-scoped SSE events instead of an unscoped aggregate.
    changed_session_ids: tuple[str, ...] = ()
    # polylogue-11cg9: True when a declared ``max_pass_seconds`` budget cut
    # this group short of its full input. Paths left out of both
    # ``succeeded`` and ``failed`` in that case were never attempted this
    # pass -- they remain ordinary backlog for the caller's next tick.
    time_budget_exceeded: bool = False


def _full_ingest_result_from_summary(
    *,
    succeeded: list[Path],
    failed: list[Path],
    source_payload_read_bytes: int,
    raw_fingerprints: dict[Path, str],
    raw_byte_sizes: dict[Path, int],
    raw_frontier_sizes: dict[Path, int] | None = None,
    raw_source_names: dict[Path, str] | None = None,
    raw_source_revisions: dict[Path, str] | None = None,
    raw_source_fingerprints: dict[Path, str] | None = None,
    captured_content_hashes: dict[Path, str] | None = None,
    captured_file_observations: dict[Path, tuple[int, int, int, int, int]] | None = None,
    summary: object | None,
    time_budget_exceeded: bool = False,
) -> _FullIngestResult:
    error = getattr(summary, "wal_checkpoint_error", None) if summary is not None else None
    return _FullIngestResult(
        succeeded=succeeded,
        failed=failed,
        source_payload_read_bytes=source_payload_read_bytes,
        raw_fingerprints=raw_fingerprints,
        raw_byte_sizes=raw_byte_sizes,
        raw_frontier_sizes=raw_frontier_sizes or {},
        raw_source_names=raw_source_names or {},
        raw_source_revisions=raw_source_revisions or {},
        raw_source_fingerprints=raw_source_fingerprints or {},
        captured_content_hashes=captured_content_hashes or {},
        captured_file_observations=captured_file_observations or {},
        worker_count=int(getattr(summary, "worker_count", 0)) if summary is not None else 0,
        ingested_session_count=int(getattr(summary, "total_convos", 0)) if summary is not None else 0,
        ingested_message_count=int(getattr(summary, "total_msgs", 0)) if summary is not None else 0,
        changed_session_count=len(getattr(summary, "changed_session_ids", ())) if summary is not None else 0,
        changed_session_ids=tuple(getattr(summary, "changed_session_ids", ()) or ()) if summary is not None else (),
        wal_bytes_before_checkpoint=int(getattr(summary, "wal_bytes_before_checkpoint", 0))
        if summary is not None
        else 0,
        wal_bytes_after_checkpoint=int(getattr(summary, "wal_bytes_after_checkpoint", 0)) if summary is not None else 0,
        wal_checkpointed_pages=int(getattr(summary, "wal_checkpointed_pages", 0)) if summary is not None else 0,
        wal_busy_pages=int(getattr(summary, "wal_busy_pages", 0)) if summary is not None else 0,
        wal_checkpoint_elapsed_s=float(getattr(summary, "wal_checkpoint_elapsed_s", 0.0))
        if summary is not None
        else 0.0,
        wal_checkpoint_mode=str(getattr(summary, "wal_checkpoint_mode", "none")) if summary is not None else "none",
        wal_checkpoint_error=str(error) if error is not None else None,
        stage_timings_s=dict(getattr(summary, "stage_timings_s", {})) if summary is not None else {},
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


def jsonl_complete_prefix(payload: bytes) -> JsonlBoundary:
    """Find the maximal newline-terminated, syntactically valid JSON prefix.

    The lexical scan keeps only the current record in memory and treats
    escaped quotes, backslashes, braces, and newlines inside strings as data.
    The caller retains the original full bytes separately.
    """
    record_start = 0
    depth = 0
    in_string = False
    escaped = False
    complete_end = 0
    records = 0
    for offset, byte in enumerate(payload):
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):  # { [
            depth += 1
        elif byte in (0x7D, 0x5D):  # } ]
            depth = max(0, depth - 1)
        elif byte == 0x0A and depth == 0:
            line = payload[record_start : offset + 1].strip()
            record_start = offset + 1
            if not line:
                complete_end = offset + 1
                continue
            try:
                json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return JsonlBoundary(complete_end, records, True, True)
            complete_end = offset + 1
            records += 1
    trailing = payload[record_start:].strip()
    if trailing and depth == 0 and not in_string:
        try:
            json.loads(trailing)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JsonlBoundary(complete_end, records, True, True)
        else:
            return JsonlBoundary(len(payload), records + 1, False)
    return JsonlBoundary(complete_end, records, complete_end != len(payload))


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


def _full_parse_progress_groups(paths: list[Path]) -> Iterable[list[Path]]:
    small_paths: list[Path] = []
    small_bytes = 0
    for path in paths:
        byte_size = _path_size(path)
        if byte_size < _LARGE_FULL_PARSE_PROGRESS_BYTES:
            if small_paths and (
                len(small_paths) >= _SMALL_FULL_PARSE_PROGRESS_MAX_FILES
                or small_bytes + byte_size > _SMALL_FULL_PARSE_PROGRESS_MAX_BYTES
            ):
                yield small_paths
                small_paths = []
                small_bytes = 0
            small_paths.append(path)
            small_bytes += byte_size
            continue
        if small_paths:
            yield small_paths
            small_paths = []
            small_bytes = 0
        yield [path]
    if small_paths:
        yield small_paths


def _append_plan_group_ready(plans: list[_AppendPlan]) -> bool:
    """Return true when pending append plans should be ingested now."""
    if len(plans) >= _MAX_APPEND_PLAN_GROUP_FILES:
        return True
    return sum(plan.bytes_read for plan in plans) >= _MAX_APPEND_PLAN_GROUP_PAYLOAD_BYTES


def _full_ingest_worker_count(records: list[RawSessionRecord]) -> int:
    """Return the worker count for daemon live full-ingest batches."""
    return select_ingest_worker_count(records, _live_full_ingest_worker_limit())


def _live_full_ingest_worker_limit() -> int:
    """Resolve the daemon live full-ingest worker cap via the layered config."""
    from polylogue.config import load_polylogue_config

    try:
        return load_polylogue_config().live_full_ingest_workers
    except ValueError:
        return _DEFAULT_LIVE_FULL_INGEST_WORKERS


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

    The receiver serializes captures with ``sort_keys=True``
    (``browser_capture/receiver.py``), so the envelope's ``raw_provider_payload``
    field (an unbounded copy of the provider's own wire payload) sorts
    alphabetically *before* ``session`` and therefore before
    ``session.provider``. Once ``raw_provider_payload`` alone exceeds
    ``_BROWSER_CAPTURE_PREFIX_PROBE_BYTES``, the provider marker never appears
    in the leading prefix at all -- a >8MiB capture with a big enough leading
    payload was permanently stamped ``unknown-export`` regardless of how many
    times it was re-captured (polylogue-mvq8). The prefix regex below still
    short-circuits the common case (small ``raw_provider_payload``, provider
    marker within the first MiB); only when that is inconclusive but the
    envelope is confirmed to be a browser capture does this fall back to a
    memory-bounded structural scan (:func:`_browser_capture_provider_from_path`)
    that finds ``session.provider`` regardless of where it falls in the
    payload.
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


def _jsonl_sample_from_path(path: Path, *, max_records: int = 32) -> list[JSONValue]:
    try:
        records, _malformed_lines, _malformed_detail = _sample_jsonl_payload_with_detail(
            path,
            max_samples=max_records,
            scan_full=False,
            max_record_bytes=JSONL_RECORD_INSPECTION_BYTES,
        )
    except ValueError:
        return []
    return records


def _detect_provider_from_path_sample(path: Path, fallback_provider: Provider) -> Provider:
    if hermes_state.looks_like_state_db_path(path) or hermes_verification.looks_like_verification_evidence_db_path(
        path
    ):
        return Provider.HERMES
    if is_jsonl_source_path(str(path)):
        records = _jsonl_sample_from_path(path)
        if records:
            return detect_provider(records) or fallback_provider
        return fallback_provider
    if _path_size(path) > _STREAMING_FULL_INGEST_BYTES:
        browser_capture, provider = _browser_capture_prefix_probe(path)
        return provider or fallback_provider if browser_capture else fallback_provider
    try:
        payload = path.read_bytes()
    except OSError:
        return fallback_provider
    return _detect_provider_from_raw_bytes(payload, path.name, fallback_provider)


def _jsonl_provider_and_session_artifact(
    path: Path,
    fallback_provider: Provider,
) -> tuple[Provider, bool]:
    records = _jsonl_sample_from_path(path)
    provider = (detect_provider(records) if records else None) or fallback_provider
    if jsonl_session_artifact(path, provider=provider) is not None:
        return provider, True
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return provider, path_classification.parse_as_session
    return provider, False


def _parse_path_as_session_artifact(path: Path, *, provider: Provider) -> bool:
    if provider is Provider.HERMES and (
        hermes_state.looks_like_state_db_path(path)
        or hermes_verification.looks_like_verification_evidence_db_path(path)
    ):
        return True
    if is_jsonl_source_path(str(path)):
        if jsonl_session_artifact(path, provider=provider) is not None:
            return True
        path_classification = classify_artifact_path(path, provider=provider)
        return path_classification.parse_as_session if path_classification is not None else False
    path_classification = strong_path_classification(path, provider=provider)
    if path_classification is not None:
        return path_classification.parse_as_session
    if _path_size(path) > _STREAMING_FULL_INGEST_BYTES:
        browser_capture, _browser_provider = _browser_capture_prefix_probe(path)
        if browser_capture:
            return True
        return _large_non_jsonl_path_can_stream(path, provider=provider)
    try:
        document = json_loads(path.read_bytes())
    except JSONDecodeError:
        return False
    return classify_artifact(document, provider=provider, source_path=path).parse_as_session


def _large_non_jsonl_path_can_stream(path: Path, *, provider: Provider) -> bool:
    if path.suffix.lower() != ".json":
        return False
    return provider in {
        Provider.CHATGPT,
        Provider.CLAUDE_AI,
        Provider.DRIVE,
        Provider.GEMINI,
    }


def _parse_payload_as_session_artifact(path: Path, *, provider: Provider, payload: bytes) -> bool:
    if provider is Provider.HERMES and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        # polylogue-hbtj2: this used to be a bare extension match, which
        # would accept ANY ".db"/".sqlite"/".sqlite3" file under a
        # Hermes-tagged source as session-parseable without ever checking
        # its bytes -- exactly the "detection-boundary strictness" bug the
        # audit found (miscaptured SQLite databases opportunistically
        # treated as sessions). Detection must be by content: only a
        # payload whose schema genuinely matches Hermes's state.db /
        # verification_evidence.db shape (verified via a real, read-only
        # ``sqlite3`` connection in ``looks_like_state_db_path`` /
        # ``looks_like_verification_evidence_db_path``) is session-eligible;
        # every other SQLite-shaped file under a Hermes source is refused.
        return hermes_state.looks_like_state_db_path(
            path
        ) or hermes_verification.looks_like_verification_evidence_db_path(path)
    if is_jsonl_source_path(str(path)):
        if jsonl_session_artifact(payload, provider=provider) is not None:
            return True
        path_classification = classify_artifact_path(path, provider=provider)
        return path_classification.parse_as_session if path_classification is not None else False
    path_classification = strong_path_classification(path, provider=provider)
    if path_classification is not None:
        return path_classification.parse_as_session
    try:
        document = json_loads(payload)
    except JSONDecodeError:
        return False
    return classify_artifact(document, provider=provider, source_path=path).parse_as_session
