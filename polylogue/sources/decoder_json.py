"""Byte decoding and streamed JSON extraction helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import ijson

from polylogue.logging import get_logger

logger = get_logger(__name__)

ENCODING_GUESSES: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "utf-32",
    "utf-32-le",
    "utf-32-be",
)


def decode_json_bytes_with(logger_obj: Any, blob: bytes) -> str | None:
    """Decode a JSON payload from bytes, trying multiple encodings."""
    for encoding in ENCODING_GUESSES:
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
        logger_obj.debug("Failed to coerce JSON bytes after fallbacks.")
        return None


def decode_json_bytes(blob: bytes) -> str | None:
    return decode_json_bytes_with(logger, blob)


def iter_json_stream_with(
    logger_obj: Any,
    ijson_module: Any,
    handle: BinaryIO | IO[bytes],
    path_name: str,
    unpack_lists: bool = True,
) -> Iterable[Any]:
    if path_name.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson")):
        error_count = 0
        pending: bytes | str | None = None

        def yield_pending(raw_pending: bytes | str, *, is_last: bool) -> Iterable[Any]:
            nonlocal error_count
            if isinstance(raw_pending, bytes):
                decoded = decode_json_bytes_with(logger_obj, raw_pending)
                if not decoded:
                    if is_last:
                        logger_obj.debug("Skipping undecodable trailing line from %s", path_name)
                    else:
                        logger_obj.warning("Skipping undecodable line from %s", path_name)
                    return
            else:
                decoded = raw_pending
            try:
                yield json.loads(decoded)
            except json.JSONDecodeError as exc:
                if is_last:
                    logger_obj.debug("Skipping truncated trailing line in %s: %s", path_name, exc)
                else:
                    error_count += 1
                    if error_count <= 3:
                        logger_obj.warning("Skipping invalid JSON line in %s: %s", path_name, exc)
                    elif error_count == 4:
                        logger_obj.warning("Skipping further invalid JSON lines in %s...", path_name)

        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            if pending is not None:
                yield from yield_pending(pending, is_last=False)
            pending = raw
        if pending is not None:
            yield from yield_pending(pending, is_last=True)
        if error_count > 3:
            logger_obj.warning("Skipped %d invalid JSON lines in %s", error_count, path_name)
        return

    if unpack_lists:
        try:
            found_any = False
            for item in ijson_module.items(handle, "item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson_module.common.JSONError:
            if found_any:
                return
        except Exception as exc:
            logger_obj.debug("Strategy 1 (ijson items) failed for %s: %s", path_name, exc)
            if found_any:
                return

        handle.seek(0)
        try:
            found_any = False
            for item in ijson_module.items(handle, "conversations.item"):
                found_any = True
                yield item
            if found_any:
                return
        except ijson_module.common.JSONError:
            if found_any:
                return
        except Exception as exc:
            logger_obj.debug("Strategy 2 (ijson conversations.item) failed for %s: %s", path_name, exc)
            if found_any:
                return

        handle.seek(0)

    data = json.load(handle)
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        if unpack_lists:
            yield from data
        else:
            yield data


def iter_json_stream(handle: BinaryIO | IO[bytes], path_name: str, unpack_lists: bool = True) -> Iterable[Any]:
    yield from iter_json_stream_with(logger, ijson, handle, path_name, unpack_lists)


__all__ = [
    "ENCODING_GUESSES",
    "decode_json_bytes",
    "decode_json_bytes_with",
    "iter_json_stream",
    "iter_json_stream_with",
]
