"""Byte decoding, JSON streaming, and ZIP processing utilities."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any, BinaryIO

import ijson

from polylogue.logging import get_logger
from polylogue.types import Provider

from .cursor import _record_cursor_failure
from .parsers.base import ParsedConversation, RawConversationData

logger = get_logger(__name__)


_ENCODING_GUESSES: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "utf-32",
    "utf-32-le",
    "utf-32-be",
)

# ZIP bomb protection constants
MAX_COMPRESSION_RATIO = 1000  # 1000x — JSON/JSONL compresses extremely well (100-500x typical)
MAX_UNCOMPRESSED_SIZE = 10 * 1024 * 1024 * 1024  # 10GB — multi-year chat archives can be large


def _decode_json_bytes(blob: bytes) -> str | None:
    """Decode a JSON payload from bytes, trying multiple encodings."""

    for encoding in _ENCODING_GUESSES:
        try:
            decoded = blob.decode(encoding)
        except UnicodeError:
            continue
        cleaned = decoded.replace("\x00", "").lstrip("\ufeff")
        if cleaned:
            return cleaned
    try:
        decoded = blob.decode("utf-8", errors="ignore").replace("\x00", "")
        return decoded if decoded else None
    except (AttributeError, UnicodeDecodeError):
        logger.debug("Failed to coerce JSON bytes after fallbacks.")
        return None


def _iter_json_stream(handle: BinaryIO | IO[bytes], path_name: str, unpack_lists: bool = True) -> Iterable[Any]:
    if path_name.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson")):
        error_count = 0
        # One-ahead buffer: process `pending` only when we know it's NOT the last line.
        # Truncated trailing lines (from in-progress session files) are debug-logged only.
        pending: bytes | str | None = None

        def _yield_pending(raw_pending: bytes | str, *, is_last: bool) -> Iterable[Any]:
            nonlocal error_count
            if isinstance(raw_pending, bytes):
                decoded: str | None = _decode_json_bytes(raw_pending)
                if not decoded:
                    if is_last:
                        logger.debug("Skipping undecodable trailing line from %s", path_name)
                    else:
                        logger.warning("Skipping undecodable line from %s", path_name)
                    return
            else:
                decoded = raw_pending
            try:
                yield json.loads(decoded)
            except json.JSONDecodeError as exc:
                if is_last:
                    logger.debug("Skipping truncated trailing line in %s: %s", path_name, exc)
                else:
                    error_count += 1
                    if error_count <= 3:
                        logger.warning("Skipping invalid JSON line in %s: %s", path_name, exc)
                    elif error_count == 4:
                        logger.warning("Skipping further invalid JSON lines in %s...", path_name)

        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            if pending is not None:
                yield from _yield_pending(pending, is_last=False)
            pending = raw
        if pending is not None:
            yield from _yield_pending(pending, is_last=True)
        if error_count > 3:
            logger.warning("Skipped %d invalid JSON lines in %s", error_count, path_name)
        return

    if unpack_lists:
        # Strategy 1: Try streaming root list
        try:
            found_any = False
            for item in ijson.items(handle, "item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson.common.JSONError:
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates
        except Exception as exc:
            logger.debug("Strategy 1 (ijson items) failed for %s: %s", path_name, exc)
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates

        handle.seek(0)
        # Strategy 2: Try streaming conversations list
        try:
            found_any = False
            for item in ijson.items(handle, "conversations.item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson.common.JSONError:
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates
        except Exception as exc:
            logger.debug("Strategy 2 (ijson conversations.item) failed for %s: %s", path_name, exc)
            if found_any:
                return  # Already yielded items — don't retry to avoid duplicates

        handle.seek(0)
    # Strategy 3: Load full object (fallback for single dicts or unknown structures)
    # Let JSONDecodeError propagate so outer handler can track failed files
    data = json.load(handle)
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        if unpack_lists:
            yield from data
        else:
            yield data


class _ZipEntryValidator:
    """Validate ZIP entries for security and relevance."""

    __slots__ = ("_provider_hint", "_cursor_state", "_zip_path")

    def __init__(
        self,
        provider_hint: str,
        *,
        cursor_state: dict[str, Any] | None,
        zip_path: Path,
    ) -> None:
        self._provider_hint = provider_hint
        self._cursor_state = cursor_state
        self._zip_path = zip_path

    def filter_entries(self, entries: list[zipfile.ZipInfo]) -> Iterable[zipfile.ZipInfo]:
        """Yield safe, relevant entries.  Record failures in cursor_state."""
        for info in entries:
            if info.is_dir():
                continue
            name = info.filename
            lower_name = name.lower()

            # Filter Claude AI ZIP: only process conversations.json
            if self._provider_hint in ("claude", "claude-ai"):
                basename = lower_name.split("/")[-1]
                if basename not in ("conversations.json",):
                    continue

            # ZIP bomb protection: compression ratio
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

            # ZIP bomb protection: uncompressed size
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

            # Only yield entries with supported JSON extensions
            if lower_name.endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")):
                yield info


def _zip_entry_provider_hint(entry_name: str, fallback_provider: str | Provider) -> Provider:
    from .source import detect_provider
    return detect_provider(None, Path(entry_name)) or Provider.from_string(fallback_provider)


def _process_zip(
    zip_path: Path,
    *,
    provider_hint: str,
    should_group: bool,
    file_mtime: str | None,
    capture_raw: bool,
    cursor_state: dict[str, Any] | None,
) -> Iterable[tuple[RawConversationData | None, ParsedConversation]]:
    """Process a ZIP file, yielding conversations from its entries."""
    from .emitter import _ConversationEmitter, _ParseContext
    from .source import _GROUP_PROVIDERS

    validator = _ZipEntryValidator(provider_hint, cursor_state=cursor_state, zip_path=zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for info in validator.filter_entries(zf.infolist()):
            name = info.filename
            entry_provider_hint = _zip_entry_provider_hint(name, provider_hint)
            entry_should_group = entry_provider_hint in _GROUP_PROVIDERS
            ctx = _ParseContext(
                provider_hint=entry_provider_hint,
                should_group=entry_should_group,
                source_path_str=f"{zip_path}:{name}",
                fallback_id=zip_path.stem,
                file_mtime=file_mtime,
                capture_raw=capture_raw,
                session_index={},  # ZIP files don't have session indices
                detect_path=Path(name),
            )
            emitter = _ConversationEmitter(ctx)
            with zf.open(name) as handle:
                yield from emitter.emit(handle, name)


__all__ = [
    "_ENCODING_GUESSES",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "_decode_json_bytes",
    "_iter_json_stream",
    "_ZipEntryValidator",
    "_zip_entry_provider_hint",
    "_process_zip",
]
