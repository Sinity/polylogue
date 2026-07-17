"""Typed local-file and ZIP acquisition components."""

from __future__ import annotations

import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TypeAlias

import ijson

from polylogue.archive.artifact_taxonomy import classify_artifact
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, is_json_value, normalize_json_decimal
from polylogue.core.json import dumps_bytes as json_dumps_bytes
from polylogue.core.metrics import read_current_rss_mb, read_peak_rss_self_mb
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from . import decoders as _decoders
from .decoders import _zip_entry_provider_hint
from .dispatch import GROUP_PROVIDERS, _detect_provider_from_raw_bytes, detect_provider
from .parsers.base import RawSessionData
from .sqlite_snapshot import is_sqlite_path, original_sqlite_source_path, snapshot_sqlite_to_blob

_DETECTION_PREFIX_SIZE = 8192  # 8 KB — enough for provider detection
_HEARTBEAT_INTERVAL_S = 5.0

AcquisitionObservation: TypeAlias = JSONDocument
ObservationCallback: TypeAlias = Callable[[AcquisitionObservation], None]
StatusCallback: TypeAlias = Callable[[str], None]
CursorState: TypeAlias = CursorStatePayload


@dataclass(frozen=True, slots=True)
class SourceReadContext:
    """Common acquisition dependencies for one local source artifact."""

    source: Source
    path: Path
    file_mtime: str | None
    provider_hint: Provider
    blob_store: BlobStore
    observation_callback: ObservationCallback | None = None
    status_callback: StatusCallback | None = None


@dataclass(frozen=True, slots=True)
class ZipEntryReadContext:
    """Acquisition dependencies for one ZIP member."""

    source: Source
    zip_path: Path
    entry: zipfile.ZipInfo
    file_mtime: str | None
    provider_hint: Provider
    blob_store: BlobStore
    observation_callback: ObservationCallback | None = None
    status_callback: StatusCallback | None = None

    @property
    def source_path(self) -> str:
        return f"{self.zip_path}:{self.entry.filename}"


@dataclass(frozen=True, slots=True)
class DetectedEntryPayload:
    """JSON stream payload paired with provider detection timing."""

    provider: Provider
    payload: JSONValue
    detect_provider_ms: float


@dataclass(frozen=True, slots=True)
class SerializedSplitPayload:
    """A serialized split session payload ready for blob persistence."""

    provider: Provider
    payload_bytes: bytes
    source_index: int


@dataclass(slots=True)
class SplitPayloadBuffer:
    """Buffer ZIP payloads until an entry proves it contains multiple sessions."""

    _pending: list[tuple[Provider, bytes]] = field(default_factory=list)
    _next_source_index: int = 0
    did_split: bool = False

    @property
    def pending_index(self) -> int:
        return self._next_source_index + len(self._pending)

    def add(self, provider: Provider, payload_bytes: bytes) -> tuple[SerializedSplitPayload, ...]:
        if self.did_split:
            payload = SerializedSplitPayload(
                provider=provider,
                payload_bytes=payload_bytes,
                source_index=self._next_source_index,
            )
            self._next_source_index += 1
            return (payload,)

        self._pending.append((provider, payload_bytes))
        if len(self._pending) < 2:
            return ()

        self.did_split = True
        emitted = tuple(
            SerializedSplitPayload(
                provider=pending_provider,
                payload_bytes=pending_payload_bytes,
                source_index=index,
            )
            for index, (pending_provider, pending_payload_bytes) in enumerate(
                self._pending,
                start=self._next_source_index,
            )
        )
        self._next_source_index += len(emitted)
        self._pending.clear()
        return emitted


def _artifact_payload(value: object) -> JSONValue:
    normalized = normalize_json_decimal(value)
    return normalized if is_json_value(normalized) else {}


def _heartbeat_label(source_path: str) -> str:
    base_path, separator, zip_entry = source_path.partition(":")
    base_name = Path(base_path).name if base_path else source_path
    return f"{base_name}:{zip_entry}" if separator else base_name


def make_status_heartbeat(
    status_callback: StatusCallback | None,
    *,
    source_name: str,
    source_path: str,
) -> Callable[[], None] | None:
    if status_callback is None:
        return None

    label = _heartbeat_label(source_path)
    last_emit = 0.0

    def emit() -> None:
        nonlocal last_emit
        now = time.monotonic()
        if last_emit and now - last_emit < _HEARTBEAT_INTERVAL_S:
            return
        last_emit = now
        status_callback(f"Scanning [{source_name}] reading {label}")

    return emit


def observe_acquisition(
    observation_callback: ObservationCallback | None,
    *,
    phase: str,
    source_path: str,
    provider_hint: Provider,
    blob_size: int,
    source_index: int | None = None,
    **extra: object,
) -> None:
    if observation_callback is None:
        return
    current_rss_mb = read_current_rss_mb()
    peak_rss_self_mb = read_peak_rss_self_mb()
    if current_rss_mb is None and peak_rss_self_mb is None:
        return
    payload: AcquisitionObservation = {
        "phase": phase,
        "source_path": source_path,
        "provider_hint": str(provider_hint),
        "blob_size": blob_size,
        "blob_mb": round(blob_size / (1024 * 1024), 3),
        "source_index": source_index,
        "current_rss_mb": current_rss_mb,
        "peak_rss_self_mb": peak_rss_self_mb,
    }
    for key, value in extra.items():
        if is_json_value(value):
            payload[key] = value
    observation_callback(payload)


def stream_fileobj_to_blob(
    blob_store: BlobStore,
    handle: IO[bytes],
    *,
    status_callback: StatusCallback | None,
    source_name: str,
    source_path: str,
) -> tuple[str, int]:
    heartbeat = make_status_heartbeat(
        status_callback,
        source_name=source_name,
        source_path=source_path,
    )
    if heartbeat is not None:
        heartbeat()
    return blob_store.write_from_fileobj(handle, heartbeat=heartbeat)


def stream_path_to_blob(
    blob_store: BlobStore,
    path: Path,
    *,
    status_callback: StatusCallback | None,
    source_name: str,
) -> tuple[str, int]:
    heartbeat = make_status_heartbeat(
        status_callback,
        source_name=source_name,
        source_path=str(path),
    )
    if heartbeat is not None:
        heartbeat()
    return blob_store.write_from_path(path, heartbeat=heartbeat)


def raw_data_record(
    *,
    source_path: str,
    file_mtime: str | None,
    provider_hint: Provider,
    blob_hash: str,
    blob_size: int,
    source_index: int | None = None,
    blob_publication_receipt_id: str | None = None,
) -> RawSessionData:
    return RawSessionData(
        raw_bytes=b"",
        source_path=source_path,
        source_index=source_index,
        file_mtime=file_mtime,
        provider_hint=provider_hint,
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_publication_receipt_id=blob_publication_receipt_id,
    )


def iter_entry_payloads(
    handle: IO[bytes],
    *,
    stream_name: str,
    provider_hint: Provider,
) -> Iterable[DetectedEntryPayload]:
    """Yield payloads from a streamed JSON/JSONL document with provider hints."""
    current_provider = provider_hint
    last_detected_provider: Provider | None = None
    provider_locked = False
    for payload in _decoders._iter_json_stream(handle, stream_name):
        normalized_payload = _artifact_payload(payload)
        if provider_locked:
            yield DetectedEntryPayload(current_provider, normalized_payload, 0.0)
            continue

        detect_start = time.perf_counter()
        detected_provider = detect_provider(normalized_payload)
        detect_provider_ms = (time.perf_counter() - detect_start) * 1000.0
        provider = detected_provider or current_provider
        if detected_provider is not None and detected_provider is not Provider.UNKNOWN:
            current_provider = detected_provider
            if detected_provider == last_detected_provider:
                provider_locked = True
            else:
                last_detected_provider = detected_provider
        yield DetectedEntryPayload(provider, normalized_payload, detect_provider_ms)


def make_split_entry_raw_data(
    *,
    blob_store: BlobStore,
    split_payload: SerializedSplitPayload,
    source_path: str,
    file_mtime: str | None,
) -> RawSessionData:
    """Persist a split payload to the blob store and return raw metadata."""
    blob_hash, blob_size = blob_store.write_from_bytes(split_payload.payload_bytes)
    from polylogue.storage.blob_publication import publication_receipt_id

    return raw_data_record(
        source_path=source_path,
        file_mtime=file_mtime,
        provider_hint=split_payload.provider,
        blob_hash=blob_hash,
        blob_size=blob_size,
        source_index=split_payload.source_index,
        blob_publication_receipt_id=publication_receipt_id(blob_store, blob_hash),
    )


def read_plain_source_file(context: SourceReadContext) -> RawSessionData:
    """Stream one non-ZIP source file into the blob store."""
    sqlite_path = is_sqlite_path(context.path)
    original_source_path = original_sqlite_source_path(context.path) if sqlite_path else None
    if (context.provider_hint is Provider.HERMES or original_source_path is not None) and sqlite_path:
        heartbeat = make_status_heartbeat(
            context.status_callback,
            source_name=context.source.name,
            source_path=str(context.path),
        )
        snapshot = snapshot_sqlite_to_blob(context.path, context.blob_store, heartbeat=heartbeat)
        blob_hash, blob_size = snapshot.blob_hash, snapshot.blob_size
        publication_id = snapshot.blob_publication_receipt_id
        detected_provider = Provider.HERMES
    else:
        blob_hash, blob_size = stream_path_to_blob(
            context.blob_store,
            context.path,
            status_callback=context.status_callback,
            source_name=context.source.name,
        )
        prefix = context.blob_store.read_prefix(blob_hash, _DETECTION_PREFIX_SIZE)
        detected_provider = _detect_provider_from_raw_bytes(
            prefix,
            context.path.name,
            context.provider_hint,
            truncated_tail_ok=blob_size > len(prefix),
        )
        if detected_provider is Provider.UNKNOWN and context.source.name == "browser-capture":
            detected_provider = _stream_browser_capture_provider(context.blob_store, blob_hash)
        from polylogue.storage.blob_publication import publication_receipt_id

        publication_id = publication_receipt_id(context.blob_store, blob_hash)
    observe_acquisition(
        context.observation_callback,
        phase="source-file-streamed",
        source_path=str(context.path),
        provider_hint=detected_provider,
        blob_size=blob_size,
        blob_publication_receipt_id=publication_id,
    )
    return raw_data_record(
        source_path=str(original_source_path or context.path),
        file_mtime=context.file_mtime,
        provider_hint=detected_provider,
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_publication_receipt_id=publication_id,
    )


def _stream_browser_capture_provider(blob_store: BlobStore, blob_hash: str) -> Provider:
    """Read the typed envelope provider without materializing a large capture.

    Native browser captures can place a multi-megabyte ``raw_provider_payload``
    before ``session.provider``.  Prefix detection therefore cannot prove the
    provider even though the complete retained artifact can.  Parse the scalar
    event stream until both the envelope kind and nested provider are known;
    ijson keeps this bounded regardless of the native payload size.
    """
    capture_kind: str | None = None
    provider: Provider | None = None
    try:
        with blob_store.open(blob_hash) as handle:
            for prefix, event, value in ijson.parse(handle):
                if event != "string":
                    continue
                if prefix == "polylogue_capture_kind":
                    capture_kind = str(value)
                elif prefix == "session.provider":
                    provider = Provider.from_string(str(value))
                if capture_kind is not None and provider is not None:
                    break
    except ijson.JSONError:
        return Provider.UNKNOWN
    if capture_kind != "browser_llm_session" or provider is None or provider is Provider.UNKNOWN:
        return Provider.UNKNOWN
    return provider


def _stream_preserved_zip_entry(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
    *,
    provider_hint: Provider,
) -> RawSessionData:
    with zf.open(context.entry.filename) as handle:
        blob_hash, blob_size = stream_fileobj_to_blob(
            context.blob_store,
            handle,
            status_callback=context.status_callback,
            source_name=context.source.name,
            source_path=context.source_path,
        )
    from polylogue.storage.blob_publication import publication_receipt_id

    publication_id = publication_receipt_id(context.blob_store, blob_hash)
    observe_acquisition(
        context.observation_callback,
        phase="zip-entry-streamed",
        source_path=context.source_path,
        provider_hint=provider_hint,
        blob_size=blob_size,
        blob_publication_receipt_id=publication_id,
    )
    return raw_data_record(
        source_path=context.source_path,
        file_mtime=context.file_mtime,
        provider_hint=provider_hint,
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_publication_receipt_id=publication_id,
    )


def iter_zip_entry_raw_data(
    zf: zipfile.ZipFile,
    context: ZipEntryReadContext,
) -> Iterable[RawSessionData]:
    """Yield raw records for one ZIP entry, splitting multi-session payloads."""
    entry_provider_hint = _zip_entry_provider_hint(context.entry.filename, context.provider_hint)
    if entry_provider_hint in GROUP_PROVIDERS:
        yield _stream_preserved_zip_entry(zf, context, provider_hint=entry_provider_hint)
        return

    detected_provider = entry_provider_hint
    split_buffer = SplitPayloadBuffer()
    with zf.open(context.entry.filename) as handle:
        for detected in iter_entry_payloads(
            handle,
            stream_name=context.entry.filename,
            provider_hint=entry_provider_hint,
        ):
            detected_provider = detected.provider
            if detected.provider in GROUP_PROVIDERS:
                break
            classify_start = time.perf_counter()
            artifact = classify_artifact(
                detected.payload,
                provider=detected.provider,
                source_path=context.source_path,
            )
            classify_ms = (time.perf_counter() - classify_start) * 1000.0
            if not artifact.parse_as_session:
                continue
            pending_index = split_buffer.pending_index
            serialize_start = time.perf_counter()
            payload_bytes = json_dumps_bytes(detected.payload)
            serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
            observe_acquisition(
                context.observation_callback,
                phase="zip-entry-split-payload-serialized",
                source_path=context.source_path,
                provider_hint=detected.provider,
                blob_size=len(payload_bytes),
                source_index=pending_index,
                artifact_kind=str(artifact.kind),
                detect_provider_ms=round(detected.detect_provider_ms, 3),
                classify_ms=round(classify_ms, 3),
                serialize_ms=round(serialize_ms, 3),
            )
            for split_payload in split_buffer.add(detected.provider, payload_bytes):
                yield make_split_entry_raw_data(
                    blob_store=context.blob_store,
                    split_payload=split_payload,
                    source_path=context.source_path,
                    file_mtime=context.file_mtime,
                )

    if split_buffer.did_split:
        return

    # Preserve original ZIP entry bytes when the entry is grouped,
    # non-session metadata, or a single session document.
    yield _stream_preserved_zip_entry(zf, context, provider_hint=detected_provider)


__all__ = [
    "AcquisitionObservation",
    "CursorState",
    "DetectedEntryPayload",
    "ObservationCallback",
    "SerializedSplitPayload",
    "SourceReadContext",
    "SplitPayloadBuffer",
    "StatusCallback",
    "ZipEntryReadContext",
    "iter_entry_payloads",
    "iter_zip_entry_raw_data",
    "make_status_heartbeat",
    "observe_acquisition",
    "raw_data_record",
    "read_plain_source_file",
    "stream_fileobj_to_blob",
    "stream_path_to_blob",
]
