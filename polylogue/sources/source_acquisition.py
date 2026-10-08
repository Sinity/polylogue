"""Raw source acquisition iterators over traversal and provider detection helpers."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO, TypeAlias

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.logging import WARNING, emit, get_logger
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from . import cursor as _cursor
from . import decoders as _decoders
from .acquisition_boundary import open_bound_container
from .cursor import _log_source_iteration_summary, _record_cursor_failure
from .dispatch import ForeignOriginContentError
from .retained_acquisition import SourceInputRecord, iter_captured_zip_input
from .source_acquisition_components import (
    ObservationCallback,
    SourceReadContext,
    StatusCallback,
    iter_entry_payloads,
    read_plain_source_file,
)
from .source_root_admission import refuse_non_capture_source_root
from .source_staging import bind_source_input
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
    """Adapter for source law tests around entry payload detection."""
    for detected in iter_entry_payloads(
        handle,
        stream_name=stream_name,
        provider_hint=provider_hint,
    ):
        yield (detected.provider, detected.payload, detected.detect_provider_ms)


def iter_source_acquisition_records(
    source: Source,
    *,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
    cursor_state: CursorState | None = None,
    known_mtimes: dict[str, str] | None = None,
    known_cursors: dict[str, dict[str, object]] | None = None,
    observation_callback: ObservationCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> Iterable[SourceInputRecord]:
    """Iterate raw source payloads without parsing provider payload semantics.

    For non-ZIP files, uses the blob store for streaming hash — the file is
    never loaded fully into Python memory. Only a small prefix is read for
    provider detection.
    """
    if not source.path:
        return

    if blob_store is None and blob_root is None:
        from polylogue.paths import blob_store_root

        blob_root = blob_store_root()
    if blob_store is None:
        assert blob_root is not None
        blob_store = BlobStore(blob_root)
    # Ahead of the walk: a foreign archive root is refused whether or not the
    # walk finds work in it, so the refusal does not depend on cursor state.
    refuse_non_capture_source_root(source.path, destination=blob_store.root.parent)

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=True,
        known_mtimes=known_mtimes,
        known_cursors=known_cursors,
        discover_sidecars=False,
        blob_store=blob_store,
    )
    if walk is None:
        return

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
                with (
                    bind_source_input(path) as captured,
                    open_bound_container(blob_store, captured) as physical,
                ):
                    for record in iter_captured_zip_input(
                        capture=physical,
                        source_path=str(captured.source_path),
                        source_name=source.name,
                        blob_store=blob_store,
                        file_mtime=file_mtime,
                        observation_callback=observation_callback,
                        status_callback=status_callback,
                    ):
                        if record.member_disposition == "refused":
                            failed_count += 1
                            member_path = f"{physical.captured_identity.semantic_source_path}:{record.member_name}"
                            _record_cursor_failure(cursor_state, member_path, record.diagnostic or "member refused")
                            if record.member_refusal_code == ForeignOriginContentError.code:
                                emit(
                                    "sources.acquisition.foreign_origin_refused",
                                    level=WARNING,
                                    outcome="refused",
                                    source_path=member_path,
                                    reason=record.diagnostic or "member refused",
                                )
                            else:
                                emit(
                                    "sources.zip.member_identity_refused",
                                    level=WARNING,
                                    outcome="refused",
                                    entry=member_path,
                                    reason=record.diagnostic or "member refused",
                                )
                        yield record
            else:
                data = read_plain_source_file(
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
                yield SourceInputRecord('["physical-file-v1",0]', data)
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(
                cursor_state,
                str(path),
                f"File not found (may have been deleted): {exc}",
            )
        except ForeignOriginContentError as exc:
            failed_count += 1
            emit(
                "sources.acquisition.foreign_origin_refused",
                level=WARNING,
                outcome="refused",
                source_path=str(path),
                reason=f"{exc.code}: {exc}",
            )
            _record_cursor_failure(cursor_state, str(path), f"{exc.code}: {exc}")
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


__all__ = ["iter_source_acquisition_records"]
