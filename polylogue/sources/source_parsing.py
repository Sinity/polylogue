"""Source parsing iterators over traversal and parser-emission helpers."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable

from polylogue.config import Source
from polylogue.lib.artifact_taxonomy import classify_artifact_path
from polylogue.logging import get_logger
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.state_views import CursorStatePayload
from polylogue.types import Provider

from . import cursor as _cursor
from . import decoders as _decoders
from .cursor import _log_source_iteration_summary, _ParseContext, _record_cursor_failure
from .decoders import _process_zip
from .dispatch import GROUP_PROVIDERS as _GROUP_PROVIDERS
from .emitter import _ConversationEmitter
from .parsers.base import ParsedConversation, RawConversationData
from .source_walk import _setup_source_walk

logger = get_logger(__name__)
_cursor.logger = logger
_decoders.logger = logger


def iter_source_conversations(
    source: Source,
    *,
    cursor_state: CursorStatePayload | None = None,
) -> Iterable[ParsedConversation]:
    """Iterate parsed conversations from one configured source."""
    for _raw, conversation in iter_source_conversations_with_raw(
        source,
        cursor_state=cursor_state,
        capture_raw=False,
    ):
        yield conversation


def iter_source_conversations_with_raw(
    source: Source,
    *,
    cursor_state: CursorStatePayload | None = None,
    capture_raw: bool = True,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
    """Iterate parsed conversations with optional raw byte capture."""
    if not source.path:
        return

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=capture_raw,
        known_mtimes=known_mtimes,
        discover_sidecars=True,
    )
    if walk is None:
        return

    failed_count = 0
    for path, file_mtime in walk.paths_to_process:
        try:
            path_classification = classify_artifact_path(path, provider=source.name)
            if path_classification is not None and not path_classification.parse_as_conversation:
                continue
            provider_hint = Provider.from_string(source.name)
            should_group = provider_hint in _GROUP_PROVIDERS

            if path.suffix.lower() == ".zip":
                yield from _process_zip(
                    path,
                    provider_hint=provider_hint,
                    should_group=should_group,
                    file_mtime=file_mtime,
                    capture_raw=capture_raw,
                    cursor_state=cursor_state,
                )
            else:
                ctx = _ParseContext(
                    provider_hint=provider_hint,
                    should_group=should_group,
                    source_path_str=str(path),
                    fallback_id=path.stem,
                    file_mtime=file_mtime,
                    capture_raw=capture_raw,
                    sidecar_data=walk.sidecar_data,
                )
                emitter = _ConversationEmitter(ctx)

                if capture_raw and should_group:
                    blob_hash, blob_size = get_blob_store().write_from_path(path)
                    raw_data = RawConversationData(
                        raw_bytes=b"",
                        source_path=str(path),
                        source_index=None,
                        file_mtime=file_mtime,
                        provider_hint=provider_hint,
                        blob_hash=blob_hash,
                        blob_size=blob_size,
                    )
                    with path.open("rb") as handle:
                        yield from emitter.emit(
                            handle,
                            path.name,
                            precomputed_raw=raw_data,
                        )
                else:
                    with path.open("rb") as handle:
                        yield from emitter.emit(handle, path.name)
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(
                cursor_state,
                str(path),
                f"File not found (may have been deleted): {exc}",
            )
        except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            failed_count += 1
            logger.warning("Failed to parse %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))
        except Exception as exc:
            failed_count += 1
            logger.error("Unexpected error processing %s: %s", path, exc)
            _record_cursor_failure(cursor_state, str(path), str(exc))

    _log_source_iteration_summary(
        source_name=source.name,
        total_paths=len(walk.paths),
        skipped_mtime=walk.skipped_mtime,
        failed_count=failed_count,
        failure_kind="parse/read",
    )


__all__ = [
    "iter_source_conversations",
    "iter_source_conversations_with_raw",
]
