"""Source iteration facade over traversal/raw-acquisition helpers."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from io import BytesIO
from typing import Any

from polylogue.config import Source
from polylogue.lib.artifact_taxonomy import classify_artifact_path
from polylogue.logging import get_logger
from polylogue.types import Provider

from . import cursor as _cursor
from . import decoders as _decoders
from .cursor import (
    _initialize_cursor_state,
    _log_source_iteration_summary,
    _ParseContext,
    _record_cursor_failure,
    _select_paths_for_processing,
)
from .decoders import (
    _process_zip,
    _zip_entry_provider_hint,
    _ZipEntryValidator,
)
from .dispatch import GROUP_PROVIDERS as _GROUP_PROVIDERS
from .dispatch import _detect_provider_from_raw_bytes
from .emitter import _ConversationEmitter
from .parsers.base import (
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    RawConversationData,
)
from .source_walk import (
    _build_session_indices,
    _has_supported_extension,
    _resolve_source_paths,
    _setup_source_walk,
    _walk_source_paths,
)

logger = get_logger(__name__)
_cursor.logger = logger
_decoders.logger = logger
MAX_COMPRESSION_RATIO = _decoders.MAX_COMPRESSION_RATIO
MAX_UNCOMPRESSED_SIZE = _decoders.MAX_UNCOMPRESSED_SIZE
ijson = _decoders.ijson
_decode_json_bytes = _decoders._decode_json_bytes
_iter_json_stream = _decoders._iter_json_stream
_get_file_mtime = _cursor._get_file_mtime


def iter_source_conversations(
    source: Source, *, cursor_state: dict[str, Any] | None = None
) -> Iterable[ParsedConversation]:
    """Iterate over conversations from a source.

    Delegates to iter_source_conversations_with_raw with capture_raw=False
    and strips the raw data from the yielded tuples.
    """
    for _raw, conv in iter_source_conversations_with_raw(source, cursor_state=cursor_state, capture_raw=False):
        yield conv


def iter_source_conversations_with_raw(
    source: Source,
    *,
    cursor_state: dict[str, Any] | None = None,
    capture_raw: bool = True,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
    """Iterate over conversations with optional raw byte capture.

    This is the raw-capturing version of iter_source_conversations(). It yields
    tuples of (raw_data, parsed_conversation) where raw_data contains the original
    JSON bytes that produced the conversation.

    For JSONL files where one file = one conversation (claude-code, codex), the
    entire file is captured. For bundle files (chatgpt conversations.json), each
    conversation dict is re-serialized to capture individual raw bytes.

    Args:
        source: Source configuration to iterate
        cursor_state: Optional state dict for tracking progress
        capture_raw: Whether to capture raw bytes (True by default)
        known_mtimes: Optional dict of {source_path: file_mtime} from previous runs.
            Files whose current mtime matches the known mtime are skipped entirely,
            replacing a full read+SHA256 with a single stat() call.

    Yields:
        Tuples of (RawConversationData | None, ParsedConversation)
    """
    if not source.path:
        return

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=capture_raw,
        known_mtimes=known_mtimes,
        build_session_indices=True,
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
                    session_index=walk.session_indices.get(path.parent, {}),
                )
                emitter = _ConversationEmitter(ctx)

                if capture_raw and should_group:
                    raw_bytes = path.read_bytes()
                    yield from emitter.emit(
                        BytesIO(raw_bytes), path.name, pre_read_bytes=raw_bytes,
                    )
                else:
                    with path.open("rb") as handle:
                        yield from emitter.emit(handle, path.name)
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(cursor_state, str(path), f"File not found (may have been deleted): {exc}")
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


def iter_source_raw_data(
    source: Source,
    *,
    cursor_state: dict[str, Any] | None = None,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[RawConversationData]:
    """Iterate raw source payloads without parsing provider payload semantics.

    This iterator is intended for acquisition-stage storage only. It yields one
    RawConversationData item per file (or per ZIP JSON entry) and performs no
    provider parser dispatching.

    Args:
        source: Source configuration to iterate
        cursor_state: Optional state dict for tracking progress
        known_mtimes: Optional dict of {source_path: file_mtime} from previous runs.
            Files whose current mtime matches the known mtime are skipped entirely.

    Yields:
        RawConversationData blobs suitable for raw_conversations storage.
    """
    if not source.path:
        return

    walk = _setup_source_walk(
        source,
        cursor_state=cursor_state,
        include_mtime=True,
        known_mtimes=known_mtimes,
        build_session_indices=False,
    )
    if walk is None:
        return

    failed_count = 0
    for path, file_mtime in walk.paths_to_process:
        try:
            provider_hint = Provider.from_string(source.name)

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
                        entry_provider_hint = _zip_entry_provider_hint(info.filename, provider_hint)
                        with zf.open(info.filename) as handle:
                            raw_bytes = handle.read()
                        entry_provider_hint = _detect_provider_from_raw_bytes(
                            raw_bytes,
                            info.filename,
                            entry_provider_hint,
                        )
                        yield RawConversationData(
                            raw_bytes=raw_bytes,
                            source_path=entry_path,
                            source_index=None,
                            file_mtime=file_mtime,
                            provider_hint=entry_provider_hint,
                        )
            else:
                raw_bytes = path.read_bytes()
                detected_provider = _detect_provider_from_raw_bytes(
                    raw_bytes,
                    path.name,
                    provider_hint,
                )
                yield RawConversationData(
                    raw_bytes=raw_bytes,
                    source_path=str(path),
                    source_index=None,
                    file_mtime=file_mtime,
                    provider_hint=detected_provider,
                )
        except FileNotFoundError as exc:
            failed_count += 1
            logger.warning("File disappeared during processing (TOCTOU race): %s", path)
            _record_cursor_failure(cursor_state, str(path), f"File not found (may have been deleted): {exc}")
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


__all__ = [
    "ParsedConversation",
    "ParsedMessage",
    "ParsedAttachment",
    "RawConversationData",
    "_ConversationEmitter",
    "_ParseContext",
    "_ZipEntryValidator",
    "_build_session_indices",
    "_decode_json_bytes",
    "_get_file_mtime",
    "_has_supported_extension",
    "_initialize_cursor_state",
    "_iter_json_stream",
    "_log_source_iteration_summary",
    "_record_cursor_failure",
    "_resolve_source_paths",
    "_select_paths_for_processing",
    "_setup_source_walk",
    "_walk_source_paths",
    "_zip_entry_provider_hint",
    "iter_source_conversations",
    "iter_source_conversations_with_raw",
    "iter_source_raw_data",
]
