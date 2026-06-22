"""Small helpers for live batch ingestion."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

import orjson

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.pipeline.services.ingest_batch._core import _select_ingest_worker_count
from polylogue.sources.dispatch import _detect_provider_from_raw_bytes, detect_provider
from polylogue.storage.runtime import RawSessionRecord

_LARGE_FULL_PARSE_PROGRESS_BYTES = 64 * 1024 * 1024
_SMALL_FULL_PARSE_PROGRESS_MAX_BYTES = 64 * 1024 * 1024
_SMALL_FULL_PARSE_PROGRESS_MAX_FILES = 64
_STREAMING_FULL_INGEST_BYTES = 8 * 1024 * 1024
_MAX_APPEND_PLAN_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_APPEND_PLAN_GROUP_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_APPEND_PLAN_GROUP_FILES = 64
_DEFAULT_LIVE_FULL_INGEST_WORKERS = 1


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


@dataclass(frozen=True, slots=True)
class _AppendResult:
    succeeded: list[_AppendPlan]
    failed: list[_AppendPlan]
    worker_count: int = 0


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
    raw_source_names: dict[Path, str] = field(default_factory=dict)
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


def _full_ingest_result_from_summary(
    *,
    succeeded: list[Path],
    failed: list[Path],
    source_payload_read_bytes: int,
    raw_fingerprints: dict[Path, str],
    raw_byte_sizes: dict[Path, int],
    raw_source_names: dict[Path, str] | None = None,
    summary: object | None,
) -> _FullIngestResult:
    error = getattr(summary, "wal_checkpoint_error", None) if summary is not None else None
    return _FullIngestResult(
        succeeded=succeeded,
        failed=failed,
        source_payload_read_bytes=source_payload_read_bytes,
        raw_fingerprints=raw_fingerprints,
        raw_byte_sizes=raw_byte_sizes,
        raw_source_names=raw_source_names or {},
        worker_count=int(getattr(summary, "worker_count", 0)) if summary is not None else 0,
        ingested_session_count=int(getattr(summary, "total_convos", 0)) if summary is not None else 0,
        ingested_message_count=int(getattr(summary, "total_msgs", 0)) if summary is not None else 0,
        changed_session_count=len(getattr(summary, "changed_session_ids", ())) if summary is not None else 0,
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
    )


_FINGERPRINT_STREAM_CHUNK = 1 << 20  # 1 MiB


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
    return _select_ingest_worker_count(records, _live_full_ingest_worker_limit())


def _live_full_ingest_worker_limit() -> int:
    """Resolve the daemon live full-ingest worker cap from the environment."""
    raw_value = os.environ.get("POLYLOGUE_LIVE_FULL_INGEST_WORKERS")
    if raw_value is None or raw_value.strip() == "":
        return _DEFAULT_LIVE_FULL_INGEST_WORKERS
    try:
        return max(1, int(raw_value))
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


def _jsonl_sample_from_path(path: Path, *, max_records: int = 32) -> list[JSONValue]:
    records: list[JSONValue] = []
    with path.open("rb") as handle:
        for line in handle:
            if len(records) >= max_records:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(cast(JSONValue, orjson.loads(raw)))
            except orjson.JSONDecodeError:
                continue
    return records


def _detect_provider_from_path_sample(path: Path, fallback_provider: Provider) -> Provider:
    if path.suffix.lower() == ".jsonl":
        records = _jsonl_sample_from_path(path)
        if records:
            return detect_provider(records) or fallback_provider
        return fallback_provider
    if _path_size(path) > _STREAMING_FULL_INGEST_BYTES:
        return fallback_provider
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
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return provider, path_classification.parse_as_session
    if not records:
        return provider, False
    return provider, classify_artifact(records, provider=provider, source_path=path).parse_as_session


def _parse_path_as_session_artifact(path: Path, *, provider: Provider) -> bool:
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return path_classification.parse_as_session
    if path.suffix.lower() == ".jsonl":
        records = _jsonl_sample_from_path(path)
        if not records:
            return False
        return classify_artifact(records, provider=provider, source_path=path).parse_as_session
    if _path_size(path) > _STREAMING_FULL_INGEST_BYTES:
        return _large_non_jsonl_path_can_stream(path, provider=provider)
    try:
        document = cast(JSONValue, orjson.loads(path.read_bytes()))
    except orjson.JSONDecodeError:
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
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return path_classification.parse_as_session
    if path.suffix.lower() == ".jsonl":
        records: list[JSONValue] = []
        for line in BytesIO(payload):
            if len(records) >= 32:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(cast(JSONValue, orjson.loads(raw)))
            except orjson.JSONDecodeError:
                continue
        if not records:
            return False
        return classify_artifact(records, provider=provider, source_path=path).parse_as_session
    try:
        document = cast(JSONValue, orjson.loads(payload))
    except orjson.JSONDecodeError:
        return False
    return classify_artifact(document, provider=provider, source_path=path).parse_as_session
