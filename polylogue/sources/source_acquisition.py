"""Raw source acquisition iterators over traversal and provider detection helpers."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from io import BytesIO
from typing import Any

from polylogue.config import Source
from polylogue.lib.artifact_taxonomy import classify_artifact
from polylogue.lib.json import dumps as json_dumps
from polylogue.logging import get_logger
from polylogue.storage.blob_store import get_blob_store
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


def _iter_conversation_payloads(
    raw_bytes: bytes,
    *,
    stream_name: str,
    source_path: str,
    provider_hint: Provider,
) -> Iterable[tuple[Provider, Any]]:
    """Yield conversation-bearing payloads from a raw JSON/JSONL document."""
    for payload in _decoders._iter_json_stream(BytesIO(raw_bytes), stream_name):
        provider = detect_provider(payload) or provider_hint
        artifact = classify_artifact(
            payload,
            provider=provider,
            source_path=source_path,
        )
        if artifact.parse_as_conversation:
            yield (provider, payload)


def _should_split_entry_payloads(
    raw_bytes: bytes,
    *,
    stream_name: str,
    source_path: str,
    provider_hint: Provider,
) -> bool:
    """Return whether the entry contains multiple non-grouped conversations."""
    if provider_hint in GROUP_PROVIDERS:
        return False

    for conversation_count, (_provider, _payload) in enumerate(
        _iter_conversation_payloads(
            raw_bytes,
            stream_name=stream_name,
            source_path=source_path,
            provider_hint=provider_hint,
        ),
        start=1,
    ):
        if conversation_count >= 2:
            return True
    return False


def _iter_split_entry_raw_data(
    raw_bytes: bytes,
    *,
    stream_name: str,
    source_path: str,
    file_mtime: str | None,
    provider_hint: Provider,
) -> Iterable[RawConversationData]:
    """Yield canonical per-conversation raw payloads for a bundle entry."""
    for source_index, (provider, payload) in enumerate(
        _iter_conversation_payloads(
            raw_bytes,
            stream_name=stream_name,
            source_path=source_path,
            provider_hint=provider_hint,
        )
    ):
        yield RawConversationData(
            raw_bytes=json_dumps(payload).encode("utf-8"),
            source_path=source_path,
            source_index=source_index,
            file_mtime=file_mtime,
            provider_hint=provider,
        )


def iter_source_raw_data(
    source: Source,
    *,
    cursor_state: dict[str, Any] | None = None,
    known_mtimes: dict[str, str] | None = None,
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
        build_session_indices=False,
    )
    if walk is None:
        return

    blob_store = get_blob_store()
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
                        if _should_split_entry_payloads(
                            raw_bytes,
                            stream_name=info.filename,
                            source_path=entry_path,
                            provider_hint=entry_provider_hint,
                        ):
                            yield from _iter_split_entry_raw_data(
                                raw_bytes,
                                stream_name=info.filename,
                                source_path=entry_path,
                                file_mtime=file_mtime,
                                provider_hint=entry_provider_hint,
                            )
                        else:
                            # Preserve original ZIP entry bytes when the entry
                            # is a grouped stream or a single-document payload.
                            blob_hash, blob_size = blob_store.write_from_bytes(raw_bytes)
                            yield RawConversationData(
                                raw_bytes=b"",
                                source_path=entry_path,
                                source_index=None,
                                file_mtime=file_mtime,
                                provider_hint=entry_provider_hint,
                                blob_hash=blob_hash,
                                blob_size=blob_size,
                            )
                        del raw_bytes
            else:
                # Stream-hash the file to blob store — never loads full
                # content into Python memory. A 1.5 GB file is hashed and
                # copied in 128 KB chunks.
                blob_hash, blob_size = blob_store.write_from_path(path)
                # Read a small prefix for provider detection only.
                prefix = blob_store.read_prefix(blob_hash, _DETECTION_PREFIX_SIZE)
                detected_provider = _detect_provider_from_raw_bytes(
                    prefix,
                    path.name,
                    provider_hint,
                )
                yield RawConversationData(
                    raw_bytes=b"",
                    source_path=str(path),
                    source_index=None,
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


__all__ = ["iter_source_raw_data"]
