"""Byte decoding and streamed JSON extraction helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import IO, BinaryIO, Protocol, TypeAlias, cast

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

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = dict[str, "JsonValue"] | list["JsonValue"] | JsonScalar
JsonReadable: TypeAlias = BinaryIO | IO[bytes]


class LoggerLike(Protocol):
    def debug(self, message: str, *args: object) -> object: ...

    def warning(self, message: str, *args: object) -> object: ...


class IjsonCommonLike(Protocol):
    JSONError: type[Exception]


class IjsonModuleLike(Protocol):
    common: IjsonCommonLike

    def items(self, handle: JsonReadable, prefix: str) -> Iterable[JsonValue]: ...


def decode_json_bytes_with(logger_obj: LoggerLike, blob: bytes) -> str | None:
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


def _yield_jsonl_pending(
    logger_obj: LoggerLike,
    raw_pending: bytes | str,
    *,
    is_last: bool,
    path_name: str,
) -> tuple[list[JsonValue], int]:
    if isinstance(raw_pending, bytes):
        decoded = decode_json_bytes_with(logger_obj, raw_pending)
        if not decoded:
            if is_last:
                logger_obj.debug("Skipping undecodable trailing line from %s", path_name)
                return ([], 0)
            else:
                return ([], 1)
    else:
        decoded = raw_pending

    try:
        return ([cast(JsonValue, json.loads(decoded))], 0)
    except json.JSONDecodeError as exc:
        if is_last:
            logger_obj.debug("Skipping truncated trailing line in %s: %s", path_name, exc)
            return ([], 0)
        return ([], 1)


def _iter_jsonl_stream(
    logger_obj: LoggerLike,
    handle: JsonReadable,
    path_name: str,
) -> Iterable[JsonValue]:
    error_count = 0
    pending: bytes | str | None = None

    for line in handle:
        raw = line.strip()
        if not raw:
            continue
        if pending is not None:
            records, new_errors = _yield_jsonl_pending(
                logger_obj,
                pending,
                is_last=False,
                path_name=path_name,
            )
            error_count += new_errors
            if new_errors:
                if error_count <= 3:
                    logger_obj.warning("Skipping invalid JSON line in %s", path_name)
                elif error_count == 4:
                    logger_obj.warning("Skipping further invalid JSON lines in %s...", path_name)
            yield from records
        pending = raw

    if pending is not None:
        records, _new_errors = _yield_jsonl_pending(
            logger_obj,
            pending,
            is_last=True,
            path_name=path_name,
        )
        yield from records

    if error_count > 3:
        logger_obj.warning("Skipped %d invalid JSON lines in %s", error_count, path_name)


def _stream_prefixed_items(
    logger_obj: LoggerLike,
    ijson_module: IjsonModuleLike,
    handle: JsonReadable,
    path_name: str,
    prefix: str,
    *,
    strategy_name: str,
) -> tuple[bool, list[JsonValue]]:
    found_any = False
    records: list[JsonValue] = []
    try:
        for item in ijson_module.items(handle, prefix):
            found_any = True
            records.append(item)
        return (found_any, records)
    except ijson_module.common.JSONError:
        return (found_any, records)
    except Exception as exc:
        logger_obj.debug("Strategy %s failed for %s: %s", strategy_name, path_name, exc)
        return (found_any, records)


def iter_json_stream_with(
    logger_obj: LoggerLike,
    ijson_module: IjsonModuleLike,
    handle: JsonReadable,
    path_name: str,
    unpack_lists: bool = True,
) -> Iterable[JsonValue]:
    if path_name.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson")):
        yield from _iter_jsonl_stream(logger_obj, handle, path_name)
        return

    if unpack_lists:
        found_any, records = _stream_prefixed_items(
            logger_obj,
            ijson_module,
            handle,
            path_name,
            "item",
            strategy_name="1 (ijson items)",
        )
        if found_any:
            yield from records
            return

        handle.seek(0)
        found_any, records = _stream_prefixed_items(
            logger_obj,
            ijson_module,
            handle,
            path_name,
            "conversations.item",
            strategy_name="2 (ijson conversations.item)",
        )
        if found_any:
            yield from records
            return

        handle.seek(0)

    data = cast(JsonValue, json.load(handle))
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        if unpack_lists:
            yield from data
        else:
            yield data


def iter_json_stream(handle: JsonReadable, path_name: str, unpack_lists: bool = True) -> Iterable[JsonValue]:
    yield from iter_json_stream_with(logger, ijson, handle, path_name, unpack_lists)


__all__ = [
    "ENCODING_GUESSES",
    "IjsonModuleLike",
    "JsonReadable",
    "JsonValue",
    "LoggerLike",
    "decode_json_bytes",
    "decode_json_bytes_with",
    "iter_json_stream",
    "iter_json_stream_with",
]
