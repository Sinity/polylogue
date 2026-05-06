"""Small helpers for live batch ingestion."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

import orjson

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.core.json import JSONValue
from polylogue.pipeline.services.ingest_batch._core import _select_ingest_worker_count
from polylogue.sources.dispatch import _detect_provider_from_raw_bytes, detect_provider
from polylogue.storage.runtime import RawConversationRecord
from polylogue.types import Provider

_LARGE_FULL_PARSE_PROGRESS_BYTES = 64 * 1024 * 1024
_SMALL_FULL_PARSE_PROGRESS_MAX_BYTES = 256 * 1024 * 1024
_SMALL_FULL_PARSE_PROGRESS_MAX_FILES = 256
_STREAMING_FULL_INGEST_BYTES = 8 * 1024 * 1024


class _FullIngestHeartbeat(Protocol):
    def __call__(
        self,
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        force: bool = False,
    ) -> None: ...


class _AttemptProgressEmitter(Protocol):
    def __call__(
        self,
        phase: str,
        *,
        current_path_override: Path | None = None,
        payload_read_bytes: int | None = None,
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


def fingerprint_file(path: Path) -> tuple[str, int]:
    content = path.read_bytes()
    newline_at = content.rfind(b"\n")
    last_complete_newline = 0 if newline_at < 0 else newline_at + 1
    return hashlib.sha256(content).hexdigest(), last_complete_newline


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


def _full_ingest_worker_count(records: list[RawConversationRecord]) -> int:
    return _select_ingest_worker_count(records, None)


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
    try:
        payload = path.read_bytes()
    except OSError:
        return fallback_provider
    return _detect_provider_from_raw_bytes(payload, path.name, fallback_provider)


def _jsonl_provider_and_conversation_artifact(
    path: Path,
    fallback_provider: Provider,
) -> tuple[Provider, bool]:
    records = _jsonl_sample_from_path(path)
    provider = (detect_provider(records) if records else None) or fallback_provider
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return provider, path_classification.parse_as_conversation
    if not records:
        return provider, False
    return provider, classify_artifact(records, provider=provider, source_path=path).parse_as_conversation


def _parse_path_as_conversation_artifact(path: Path, *, provider: Provider) -> bool:
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return path_classification.parse_as_conversation
    if path.suffix.lower() == ".jsonl":
        records = _jsonl_sample_from_path(path)
        if not records:
            return False
        return classify_artifact(records, provider=provider, source_path=path).parse_as_conversation
    try:
        document = cast(JSONValue, orjson.loads(path.read_bytes()))
    except orjson.JSONDecodeError:
        return False
    return classify_artifact(document, provider=provider, source_path=path).parse_as_conversation


def _parse_payload_as_conversation_artifact(path: Path, *, provider: Provider, payload: bytes) -> bool:
    path_classification = classify_artifact_path(path, provider=provider)
    if path_classification is not None:
        return path_classification.parse_as_conversation
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
        return classify_artifact(records, provider=provider, source_path=path).parse_as_conversation
    try:
        document = cast(JSONValue, orjson.loads(payload))
    except orjson.JSONDecodeError:
        return False
    return classify_artifact(document, provider=provider, source_path=path).parse_as_conversation
