"""ZIP validation and extraction helpers for source ingestion."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

from polylogue.archive.artifact_taxonomy import classify_artifact_path
from polylogue.logging import get_logger
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.types import Provider

from .cursor import _record_cursor_failure
from .parsers.base import ParsedConversation, RawConversationData

logger = get_logger(__name__)

MAX_COMPRESSION_RATIO = 1000
MAX_UNCOMPRESSED_SIZE = 10 * 1024 * 1024 * 1024


class ZipEntryValidator:
    """Validate ZIP entries for security and relevance."""

    __slots__ = ("_provider_hint", "_cursor_state", "_zip_path", "_conversation_only")

    def __init__(
        self,
        provider_hint: str | Provider,
        *,
        cursor_state: CursorStatePayload | None,
        zip_path: Path,
        conversation_only: bool = False,
    ) -> None:
        self._provider_hint = Provider.from_string(provider_hint)
        self._cursor_state = cursor_state
        self._zip_path = zip_path
        self._conversation_only = conversation_only

    def filter_entries(self, entries: list[zipfile.ZipInfo]) -> Iterable[zipfile.ZipInfo]:
        """Yield safe, relevant entries and record failures in cursor state."""
        for info in entries:
            if info.is_dir():
                continue
            name = info.filename
            lower_name = name.lower()

            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    logger.warning(
                        "Skipping suspicious file %s in %s: compression ratio %.1f exceeds limit",
                        name,
                        self._zip_path,
                        ratio,
                    )
                    _record_cursor_failure(
                        self._cursor_state,
                        f"{self._zip_path}:{name}",
                        f"Suspicious compression ratio: {ratio:.1f}",
                    )
                    continue

            if info.file_size > MAX_UNCOMPRESSED_SIZE:
                logger.warning(
                    "Skipping oversized file %s in %s: %d bytes exceeds limit",
                    name,
                    self._zip_path,
                    info.file_size,
                )
                _record_cursor_failure(
                    self._cursor_state,
                    f"{self._zip_path}:{name}",
                    f"File size {info.file_size} exceeds limit",
                )
                continue

            if lower_name.endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")):
                if self._conversation_only:
                    path_classification = classify_artifact_path(
                        f"{self._zip_path}:{name}",
                        provider=self._provider_hint,
                    )
                    if path_classification is not None and not path_classification.parse_as_conversation:
                        continue
                yield info


def zip_entry_provider_hint(entry_name: str, fallback_provider: str | Provider) -> Provider:
    del entry_name
    return Provider.from_string(fallback_provider)


def process_zip(
    zip_path: Path,
    *,
    provider_hint: Provider,
    should_group: bool,
    file_mtime: str | None,
    capture_raw: bool,
    cursor_state: CursorStatePayload | None,
) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
    """Process a ZIP file, yielding conversations from its entries."""
    del should_group

    from .cursor import _ParseContext
    from .dispatch import GROUP_PROVIDERS
    from .emitter import _ConversationEmitter

    validator = ZipEntryValidator(
        provider_hint,
        cursor_state=cursor_state,
        zip_path=zip_path,
        conversation_only=True,
    )

    with zipfile.ZipFile(zip_path) as zf:
        for info in validator.filter_entries(zf.infolist()):
            name = info.filename
            entry_provider_hint = zip_entry_provider_hint(name, provider_hint)
            entry_should_group = entry_provider_hint in GROUP_PROVIDERS
            ctx = _ParseContext(
                provider_hint=entry_provider_hint,
                should_group=entry_should_group,
                source_path_str=f"{zip_path}:{name}",
                fallback_id=zip_path.stem,
                file_mtime=file_mtime,
                capture_raw=capture_raw,
                sidecar_data={},
            )
            emitter = _ConversationEmitter(ctx)
            precomputed_raw: RawConversationData | None = None
            if capture_raw and entry_should_group:
                with zf.open(name) as handle:
                    blob_hash, blob_size = get_blob_store().write_from_fileobj(handle)
                precomputed_raw = RawConversationData(
                    raw_bytes=b"",
                    source_path=f"{zip_path}:{name}",
                    source_index=None,
                    file_mtime=file_mtime,
                    provider_hint=entry_provider_hint,
                    blob_hash=blob_hash,
                    blob_size=blob_size,
                )
            with zf.open(name) as handle:
                yield from emitter.emit(handle, name, precomputed_raw=precomputed_raw)


__all__ = [
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "ZipEntryValidator",
    "process_zip",
    "zip_entry_provider_hint",
]
