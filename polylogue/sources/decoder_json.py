"""Byte decoding and streamed JSON extraction helpers."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterable
from typing import IO, Protocol, TypeAlias, TypeGuard

import ijson
import orjson

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
JsonReadable: TypeAlias = IO[bytes]


class LoggerLike(Protocol):
    def debug(self, message: str, *args: object) -> object: ...

    def warning(self, message: str, *args: object) -> object: ...


class IjsonCommonLike(Protocol):
    JSONError: type[Exception]


class IjsonModuleLike(Protocol):
    common: IjsonCommonLike

    def items(self, handle: JsonReadable, prefix: str) -> Iterable[JsonValue]: ...


class PartialJsonStreamError(ValueError):
    """A prefixed JSON stream was truncated/corrupted partway through decoding.

    Surfaced instead of silently returning the records accumulated before the
    corruption. ``recovered`` is the number of items successfully yielded before
    the failure; ``offset`` is the byte offset of the failure when ``ijson``
    reports one (``None`` otherwise).
    """

    def __init__(
        self,
        path_name: str,
        *,
        recovered: int,
        offset: int | None,
        cause: BaseException,
    ) -> None:
        self.path_name = path_name
        self.recovered = recovered
        self.offset = offset
        self.cause = cause
        location = f" at byte offset {offset}" if offset is not None else ""
        super().__init__(
            f"partial JSON stream decode of {path_name}: corruption{location} after {recovered} record(s): {cause}"
        )


def _is_json_value(value: object) -> TypeGuard[JsonValue]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


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
    try:
        parsed = orjson.loads(raw_pending)
    except orjson.JSONDecodeError:
        parsed = None
    else:
        if _is_json_value(parsed):
            return ([parsed], 0)
        logger_obj.debug("Skipping non-JSON-compatible decoded line from %s", path_name)
        return ([], 0)

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
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        if is_last:
            logger_obj.debug("Skipping truncated trailing line in %s: %s", path_name, exc)
            return ([], 0)
        return ([], 1)
    if _is_json_value(parsed):
        return ([parsed], 0)
    logger_obj.debug("Skipping non-JSON-compatible decoded line from %s", path_name)
    return ([], 0)


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
    except ijson_module.common.JSONError as exc:
        if found_any:
            # Mid-stream corruption: the array/object was valid for the first
            # ``len(records)`` items then broke. Returning the partial set here
            # silently truncates the session set, so surface a typed error
            # instead. A JSONError with zero items found is a normal
            # "wrong prefix, try the next strategy" signal and is swallowed.
            offset = _json_error_offset(exc)
            logger_obj.warning(
                "Partial JSON stream decode of %s (strategy %s): corruption after %d record(s)%s",
                path_name,
                strategy_name,
                len(records),
                f" at byte offset {offset}" if offset is not None else "",
            )
            raise PartialJsonStreamError(
                path_name,
                recovered=len(records),
                offset=offset,
                cause=exc,
            ) from exc
        return (found_any, records)
    except Exception as exc:
        logger_obj.debug("Strategy %s failed for %s: %s", strategy_name, path_name, exc)
        return (found_any, records)


def _json_error_offset(exc: BaseException) -> int | None:
    """Extract a byte/char offset from an ijson JSONError when available."""
    for attr in ("pos", "offset"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    # ijson messages often embed the byte position, e.g. "... at 1234".
    match = re.search(r"at (\d+)", str(exc))
    if match:
        return int(match.group(1))
    return None


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

    # The ijson multi-strategy parse below rewinds via ``handle.seek(0)``. A
    # ZIP-entry stream (zipfile.ZipExtFile) is not seekable, so materialize it
    # into a seekable buffer once before the seeking strategies run.
    seekable = getattr(handle, "seekable", None)
    if callable(seekable) and not seekable():
        handle = io.BytesIO(handle.read())

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
            "sessions.item",
            strategy_name="2 (ijson sessions.item)",
        )
        if found_any:
            yield from records
            return

        handle.seek(0)

    data = json.load(handle)
    if not _is_json_value(data):
        raise ValueError(f"decoded payload from {path_name} does not satisfy the JsonValue contract")
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
    "PartialJsonStreamError",
    "decode_json_bytes",
    "decode_json_bytes_with",
    "iter_json_stream",
    "iter_json_stream_with",
]
