"""Raw source acquisition iterators over traversal and provider detection helpers."""

from __future__ import annotations

import time
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, BinaryIO, TypeAlias, cast

from polylogue.config import Source
from polylogue.lib.artifact_taxonomy import classify_artifact
from polylogue.lib.json import dumps_bytes as json_dumps_bytes
from polylogue.lib.metrics import read_current_rss_mb, read_peak_rss_self_mb
from polylogue.logging import get_logger
from polylogue.storage.blob_store import BlobStore, get_blob_store
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.types import Provider

from . import cursor as _cursor
from . import decoders as _decoders
from .cursor import _log_source_iteration_summary, _record_cursor_failure
from .decoders import _zip_entry_provider_hint, _ZipEntryValidator
from .dispatch import GROUP_PROVIDERS, _detect_provider_from_raw_bytes, detect_provider
from .parsers.base import RawConversationData
from .source_walk import _setup_source_walk

logger = get_logger(__name__)
_cursor.logger = logger
_decoders.logger = logger

_DETECTION_PREFIX_SIZE = 8192  # 8 KB — enough for provider detection
_HEARTBEAT_INTERVAL_S = 5.0
AcquisitionObservation: TypeAlias = dict[str, object]
ObservationCallback: TypeAlias = Callable[[AcquisitionObservation], None]
StatusCallback: TypeAlias = Callable[[str], None]
CursorState: TypeAlias = CursorStatePayload
DetectedEntryPayload: TypeAlias = tuple[Provider, object, float]


def _heartbeat_label(source_path: str) -> str:
    base_path, separator, zip_entry = source_path.partition(":")
    base_name = Path(base_path).name if base_path else source_path
    return f"{base_name}:{zip_entry}" if separator else base_name


def _make_status_heartbeat(
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


def _observe_acquisition(
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
        **extra,
    }
    observation_callback(payload)


def _stream_fileobj_to_blob(
    blob_store: BlobStore,
    handle: BinaryIO,
    *,
    status_callback: StatusCallback | None,
    source_name: str,
    source_path: str,
) -> tuple[str, int]:
    heartbeat = _make_status_heartbeat(
        status_callback,
        source_name=source_name,
        source_path=source_path,
    )
    if heartbeat is not None:
        heartbeat()
    return blob_store.write_from_fileobj(handle, heartbeat=heartbeat)


def _stream_path_to_blob(
    blob_store: BlobStore,
    path: Path,
    *,
    status_callback: StatusCallback | None,
    source_name: str,
) -> tuple[str, int]:
    heartbeat = _make_status_heartbeat(
        status_callback,
        source_name=source_name,
        source_path=str(path),
    )
    if heartbeat is not None:
        heartbeat()
    return blob_store.write_from_path(path, heartbeat=heartbeat)


def _raw_data_record(
    *,
    source_path: str,
    file_mtime: str | None,
    provider_hint: Provider,
    blob_hash: str,
    blob_size: int,
    source_index: int | None = None,
) -> RawConversationData:
    return RawConversationData(
        raw_bytes=b"",
        source_path=source_path,
        source_index=source_index,
        file_mtime=file_mtime,
        provider_hint=provider_hint,
        blob_hash=blob_hash,
        blob_size=blob_size,
    )


def _iter_entry_payloads(
    handle: BinaryIO | IO[bytes],
    *,
    stream_name: str,
    provider_hint: Provider,
) -> Iterable[DetectedEntryPayload]:
    """Yield payloads from a streamed JSON/JSONL document with provider hints."""
    current_provider = provider_hint
    last_detected_provider: Provider | None = None
    provider_locked = False
    for payload in _decoders._iter_json_stream(handle, stream_name):
        if provider_locked:
            provider = current_provider
            detect_provider_ms = 0.0
        else:
            detect_start = time.perf_counter()
            detected_provider = detect_provider(payload)
            detect_provider_ms = (time.perf_counter() - detect_start) * 1000.0
            provider = detected_provider or current_provider
            if detected_provider is not None and detected_provider is not Provider.UNKNOWN:
                current_provider = detected_provider
                if detected_provider == last_detected_provider:
                    provider_locked = True
                else:
                    last_detected_provider = detected_provider
        yield (provider, payload, detect_provider_ms)


def _make_split_entry_raw_data(
    *,
    blob_store: BlobStore,
    payload_bytes: bytes,
    source_path: str,
    source_index: int,
    file_mtime: str | None,
    provider_hint: Provider,
) -> RawConversationData:
    """Persist a split payload to the blob store and return raw metadata."""
    blob_hash, blob_size = blob_store.write_from_bytes(payload_bytes)
    return _raw_data_record(
        source_path=source_path,
        file_mtime=file_mtime,
        provider_hint=provider_hint,
        blob_hash=blob_hash,
        blob_size=blob_size,
        source_index=source_index,
    )


def iter_source_raw_data(
    source: Source,
    *,
    cursor_state: CursorState | None = None,
    known_mtimes: dict[str, str] | None = None,
    observation_callback: ObservationCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> Iterable[RawConversationData]:
    """Iterate raw source payloads without parsing provider payload semantics.

    For non-ZIP files, uses the blob store for streaming hash — the file is
    never loaded fully into Python memory. Only a small prefix is read for
    provider detection.
    """
    if not source.path:
        return

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=True,
        known_mtimes=known_mtimes,
        discover_sidecars=False,
    )
    if walk is None:
        return

    blob_store = get_blob_store()
    failed_count = 0
    empty_artifact_count = 0
    for path, file_mtime in walk.paths_to_process:
        try:
            provider_hint = Provider.from_string(source.name)
            if path.stat().st_size == 0:
                empty_artifact_count += 1
                logger.debug("Skipping empty source file: %s", path)
                _record_cursor_failure(cursor_state, str(path), "empty file")
                continue

            if path.suffix.lower() == ".zip":
                validator = _ZipEntryValidator(
                    provider_hint,
                    cursor_state=cursor_state,
                    zip_path=path,
                    conversation_only=False,
                )
                with zipfile.ZipFile(path) as zf:
                    for info in validator.filter_entries(zf.infolist()):
                        entry_path = f"{path}:{info.filename}"
                        if info.file_size == 0:
                            empty_artifact_count += 1
                            logger.debug("Skipping empty source entry: %s", entry_path)
                            _record_cursor_failure(cursor_state, entry_path, "empty file")
                            continue
                        entry_provider_hint = _zip_entry_provider_hint(info.filename, provider_hint)
                        if entry_provider_hint in GROUP_PROVIDERS:
                            with zf.open(info.filename) as handle:
                                blob_hash, blob_size = _stream_fileobj_to_blob(
                                    blob_store,
                                    cast(BinaryIO, handle),
                                    status_callback=status_callback,
                                    source_name=source.name,
                                    source_path=entry_path,
                                )
                            _observe_acquisition(
                                observation_callback,
                                phase="zip-entry-streamed",
                                source_path=entry_path,
                                provider_hint=entry_provider_hint,
                                blob_size=blob_size,
                            )
                            yield _raw_data_record(
                                source_path=entry_path,
                                file_mtime=file_mtime,
                                provider_hint=entry_provider_hint,
                                blob_hash=blob_hash,
                                blob_size=blob_size,
                            )
                            continue

                        detected_provider = entry_provider_hint
                        pending_split_payloads: list[tuple[Provider, bytes]] = []
                        split_source_index = 0
                        did_split = False

                        with zf.open(info.filename) as handle:
                            for payload_provider, payload, detect_provider_ms in _iter_entry_payloads(
                                cast(BinaryIO, handle),
                                stream_name=info.filename,
                                provider_hint=entry_provider_hint,
                            ):
                                detected_provider = payload_provider
                                if payload_provider in GROUP_PROVIDERS:
                                    break
                                classify_start = time.perf_counter()
                                artifact = classify_artifact(
                                    payload,
                                    provider=payload_provider,
                                    source_path=entry_path,
                                )
                                classify_ms = (time.perf_counter() - classify_start) * 1000.0
                                if not artifact.parse_as_conversation:
                                    continue
                                pending_index = split_source_index + len(pending_split_payloads)
                                serialize_start = time.perf_counter()
                                payload_bytes = json_dumps_bytes(payload)
                                serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
                                _observe_acquisition(
                                    observation_callback,
                                    phase="zip-entry-split-payload-serialized",
                                    source_path=entry_path,
                                    provider_hint=payload_provider,
                                    blob_size=len(payload_bytes),
                                    source_index=pending_index,
                                    artifact_kind=str(artifact.kind),
                                    detect_provider_ms=round(detect_provider_ms, 3),
                                    classify_ms=round(classify_ms, 3),
                                    serialize_ms=round(serialize_ms, 3),
                                )
                                if did_split:
                                    yield _make_split_entry_raw_data(
                                        blob_store=blob_store,
                                        payload_bytes=payload_bytes,
                                        source_path=entry_path,
                                        source_index=split_source_index,
                                        file_mtime=file_mtime,
                                        provider_hint=payload_provider,
                                    )
                                    split_source_index += 1
                                    continue
                                pending_split_payloads.append((payload_provider, payload_bytes))
                                if len(pending_split_payloads) < 2:
                                    continue
                                did_split = True
                                for buffered_provider, buffered_payload_bytes in pending_split_payloads:
                                    yield _make_split_entry_raw_data(
                                        blob_store=blob_store,
                                        payload_bytes=buffered_payload_bytes,
                                        source_path=entry_path,
                                        source_index=split_source_index,
                                        file_mtime=file_mtime,
                                        provider_hint=buffered_provider,
                                    )
                                    split_source_index += 1
                                pending_split_payloads.clear()

                        if did_split:
                            continue

                        # Preserve original ZIP entry bytes when the entry is
                        # grouped, non-conversation metadata, or a single
                        # conversation document.
                        with zf.open(info.filename) as handle:
                            blob_hash, blob_size = _stream_fileobj_to_blob(
                                blob_store,
                                cast(BinaryIO, handle),
                                status_callback=status_callback,
                                source_name=source.name,
                                source_path=entry_path,
                            )
                        _observe_acquisition(
                            observation_callback,
                            phase="zip-entry-streamed",
                            source_path=entry_path,
                            provider_hint=detected_provider,
                            blob_size=blob_size,
                        )
                        yield _raw_data_record(
                            source_path=entry_path,
                            file_mtime=file_mtime,
                            provider_hint=detected_provider,
                            blob_hash=blob_hash,
                            blob_size=blob_size,
                        )
            else:
                # Stream-hash the file to blob store — never loads full
                # content into Python memory. A 1.5 GB file is hashed and
                # copied in 128 KB chunks.
                blob_hash, blob_size = _stream_path_to_blob(
                    blob_store,
                    path,
                    status_callback=status_callback,
                    source_name=source.name,
                )
                # Read a small prefix for provider detection only.
                prefix = blob_store.read_prefix(blob_hash, _DETECTION_PREFIX_SIZE)
                detected_provider = _detect_provider_from_raw_bytes(
                    prefix,
                    path.name,
                    provider_hint,
                )
                _observe_acquisition(
                    observation_callback,
                    phase="source-file-streamed",
                    source_path=str(path),
                    provider_hint=detected_provider,
                    blob_size=blob_size,
                )
                yield _raw_data_record(
                    source_path=str(path),
                    file_mtime=file_mtime,
                    provider_hint=detected_provider,
                    blob_hash=blob_hash,
                    blob_size=blob_size,
                )
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(
                cursor_state,
                str(path),
                f"File not found (may have been deleted): {exc}",
            )
        except (UnicodeDecodeError, zipfile.BadZipFile, OSError) as exc:
            failed_count += 1
            logger.warning("Failed to read %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))
        except Exception as exc:
            failed_count += 1
            logger.error("Unexpected error reading %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))

    _log_source_iteration_summary(
        source_name=source.name,
        total_paths=len(walk.paths),
        skipped_mtime=walk.skipped_mtime,
        failed_count=failed_count,
        failure_kind="read",
    )
    if empty_artifact_count > 0:
        logger.warning(
            "Skipped %d empty artifacts from source %r. Run with --verbose for details.",
            empty_artifact_count,
            source.name,
        )


__all__ = ["iter_source_raw_data"]
