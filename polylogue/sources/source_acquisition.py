"""Raw source acquisition iterators over traversal and provider detection helpers."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from typing import IO, TypeAlias

from polylogue.config import Source
from polylogue.core.json import JSONValue
from polylogue.logging import get_logger
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.types import Provider

from . import cursor as _cursor
from . import decoders as _decoders
from .cursor import _log_source_iteration_summary, _record_cursor_failure
from .decoders import _ZipEntryValidator
from .parsers.base import RawConversationData
from .source_acquisition_components import (
    ObservationCallback,
    SourceReadContext,
    StatusCallback,
    ZipEntryReadContext,
    iter_entry_payloads,
    iter_zip_entry_raw_data,
    read_plain_source_file,
)
from .source_walk import _setup_source_walk

logger = get_logger(__name__)
_cursor.logger = logger
_decoders.logger = logger

CursorState: TypeAlias = CursorStatePayload
DetectedEntryPayload: TypeAlias = tuple[Provider, JSONValue, float]


def _iter_entry_payloads(
    handle: IO[bytes],
    *,
    stream_name: str,
    provider_hint: Provider,
) -> Iterable[DetectedEntryPayload]:
    """Compatibility wrapper for source law tests around entry payload detection."""
    for detected in iter_entry_payloads(
        handle,
        stream_name=stream_name,
        provider_hint=provider_hint,
    ):
        yield (detected.provider, detected.payload, detected.detect_provider_ms)


def iter_source_raw_data(
    source: Source,
    *,
    cursor_state: CursorState | None = None,
    known_mtimes: dict[str, str] | None = None,
    known_cursors: dict[str, dict[str, object]] | None = None,
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
        known_cursors=known_cursors,
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
                        yield from iter_zip_entry_raw_data(
                            zf,
                            ZipEntryReadContext(
                                source=source,
                                zip_path=path,
                                entry=info,
                                file_mtime=file_mtime,
                                provider_hint=provider_hint,
                                blob_store=blob_store,
                                observation_callback=observation_callback,
                                status_callback=status_callback,
                            ),
                        )
            else:
                yield read_plain_source_file(
                    SourceReadContext(
                        source=source,
                        path=path,
                        file_mtime=file_mtime,
                        provider_hint=provider_hint,
                        blob_store=blob_store,
                        observation_callback=observation_callback,
                        status_callback=status_callback,
                    )
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
