"""Byte decoding and streamed JSON extraction helpers."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import IO, Protocol, TypeAlias, TypeGuard, cast

import ijson

from polylogue.core.json import JSONDecodeError
from polylogue.core.json import loads as json_loads
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


def normalize_ijson_stdlib_numbers(value: object) -> object:
    """Match ``json.load`` numbers while retaining only one decoded record."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = normalize_ijson_stdlib_numbers(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            value[key] = normalize_ijson_stdlib_numbers(item)
    return value


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


class JsonlDecodeError(ValueError):
    """A complete JSONL record could not be decoded.

    Tolerant JSONL consumers may continue after malformed records, but callers
    that need fail-closed evidence can opt into raising this typed signal after
    the stream has been consumed.
    """

    def __init__(self, path_name: str, *, line_number: int, cause: BaseException) -> None:
        self.path_name = path_name
        self.line_number = line_number
        self.cause = cause
        super().__init__(f"JSONL decode failed for {path_name} at line {line_number}: {cause}")


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
    line_number: int,
) -> tuple[list[JsonValue], int, int | None]:
    try:
        parsed = json_loads(raw_pending)
    except JSONDecodeError:
        parsed = None
    else:
        # Every backend the facade selects decodes into JSON's own vocabulary
        # and nothing else, so a successful decode is a ``JsonValue`` by
        # construction. Re-walking each record to rediscover that is the
        # dominant cost of the JSONL decode boundary, not the C decode itself.
        return ([parsed], 0, None)

    if isinstance(raw_pending, bytes):
        decoded = decode_json_bytes_with(logger_obj, raw_pending)
        if not decoded:
            if is_last:
                logger_obj.debug("Skipping undecodable trailing line from %s", path_name)
                return ([], 1, line_number)
            else:
                return ([], 1, line_number)
    else:
        decoded = raw_pending

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        if is_last:
            logger_obj.debug("Skipping truncated trailing line in %s: %s", path_name, exc)
            return ([], 1, line_number)
        return ([], 1, line_number)
    return ([cast(JsonValue, parsed)], 0, None)


def _iter_jsonl_stream(
    logger_obj: LoggerLike,
    handle: JsonReadable,
    path_name: str,
    *,
    fail_on_decode_error: bool = False,
) -> Iterable[JsonValue]:
    error_count = 0
    pending: bytes | str | None = None
    physical_line_number = 0
    pending_line_number: int | None = None
    first_decode_error_line: int | None = None

    for line in handle:
        physical_line_number += 1
        raw = line.strip()
        if not raw:
            continue
        if pending is not None:
            records, new_errors, error_line = _yield_jsonl_pending(
                logger_obj,
                pending,
                is_last=False,
                path_name=path_name,
                line_number=pending_line_number or physical_line_number,
            )
            if first_decode_error_line is None and error_line is not None:
                first_decode_error_line = error_line
            error_count += new_errors
            if new_errors:
                if error_count <= 3:
                    logger_obj.warning("Skipping invalid JSON line in %s", path_name)
                elif error_count == 4:
                    logger_obj.warning("Skipping further invalid JSON lines in %s...", path_name)
            yield from records
        pending = raw
        pending_line_number = physical_line_number

    if pending is not None:
        records, new_errors, error_line = _yield_jsonl_pending(
            logger_obj,
            pending,
            is_last=True,
            path_name=path_name,
            line_number=pending_line_number or physical_line_number,
        )
        if first_decode_error_line is None and error_line is not None:
            first_decode_error_line = error_line
        error_count += new_errors
        yield from records

    if fail_on_decode_error and error_count:
        raise JsonlDecodeError(
            path_name,
            line_number=first_decode_error_line or physical_line_number,
            cause=ValueError("malformed JSONL record"),
        )

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
        if found_any:
            # Same failure, and therefore the same handling as the JSONError
            # branch above: records were already recovered, so returning the
            # partial set silently truncates the session set. Only the
            # exception type differs (an OS read fault, a decoder assertion, a
            # backend-specific error), and the type does not change what was
            # lost.
            logger_obj.warning(
                "Partial JSON stream decode of %s (strategy %s): %s after %d record(s)",
                path_name,
                strategy_name,
                type(exc).__name__,
                len(records),
            )
            raise PartialJsonStreamError(
                path_name,
                recovered=len(records),
                offset=_json_error_offset(exc),
                cause=exc,
            ) from exc
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
    fail_on_decode_error: bool = False,
) -> Iterable[JsonValue]:
    normalized_path = path_name.lower()
    if normalized_path.endswith((".jsonl", ".jsonl.txt", ".ndjson")) or any(
        marker in normalized_path for marker in (".jsonl.", ".ndjson.")
    ):
        yield from _iter_jsonl_stream(
            logger_obj,
            handle,
            path_name,
            fail_on_decode_error=fail_on_decode_error,
        )
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


def iter_json_stream(
    handle: JsonReadable,
    path_name: str,
    unpack_lists: bool = True,
    *,
    fail_on_decode_error: bool = False,
) -> Iterable[JsonValue]:
    yield from iter_json_stream_with(
        logger,
        ijson,
        handle,
        path_name,
        unpack_lists,
        fail_on_decode_error=fail_on_decode_error,
    )


def json_record_container(handle: JsonReadable) -> str | None:
    """Identify an array whose members can be decoded one at a time.

    The caller owns a seekable retained file. This probe builds no JSON
    objects and restores the position for a second, record-producing pass.
    """
    try:
        events = ijson.parse(handle)
        first = next(events, None)
        if first is not None and first[1] == "start_array":
            return "item"
        if first is not None and first[1] == "start_map":
            for prefix, event, value in events:
                if prefix == "" and event == "map_key" and value == "sessions":
                    next_event = next(events, None)
                    if next_event is not None and next_event[:2] == ("sessions", "start_array"):
                        return "sessions.item"
                if prefix == "" and event == "end_map":
                    break
    except ijson.common.JSONError:
        return None
    finally:
        handle.seek(0)
    return None


def generic_message_object_envelope(handle: JsonReadable) -> dict[str, JsonValue] | None:
    """Read a simple object envelope while leaving its message array on disk.

    Other nested root fields may carry provider semantics, so those documents
    stay on their provider parser route. The event pass also validates the
    complete JSON before a scratch artifact can be published.
    """
    scalar_fields = {
        "id",
        "title",
        "name",
        "created_at",
        "create_time",
        "created",
        "createdAt",
        "updated_at",
        "update_time",
        "updated",
        "updatedAt",
        "modified",
    }
    envelope: dict[str, JsonValue] = {}
    current_key: str | None = None
    message_arrays = 0
    try:
        events = ijson.parse(handle)
        if next(events, None) != ("", "start_map", None):
            return None
        for prefix, event, value in events:
            if prefix == "" and event == "map_key":
                current_key = str(value)
                if current_key == "messages":
                    message_arrays += 1
                elif current_key not in scalar_fields:
                    return None
                continue
            if prefix == "" and event == "end_map":
                current_key = None
                continue
            if current_key is None or prefix != current_key:
                continue
            if current_key == "messages" and event == "start_array":
                continue
            if event in {"start_array", "start_map"}:
                return None
            if event in {"string", "number", "boolean", "null"}:
                if current_key == "messages":
                    return None
                envelope[current_key] = cast(JsonValue, normalize_ijson_stdlib_numbers(value))
    except ijson.common.JSONError:
        return None
    finally:
        handle.seek(0)
    return envelope if message_arrays == 1 else None


def grok_export_item_count(handle: JsonReadable, *, on_item: Callable[[int, bool], None] | None = None) -> int | None:
    """Validate a Grok object and report each member's shape without decoding it."""
    count = 0
    keys = 0
    arrays = 0
    member_keys: set[str] = set()
    member_is_map = False
    member_conversation = False
    member_responses = False
    valid_members = 0
    root_beads_keys: set[str] = set()

    def finish_member() -> None:
        nonlocal valid_members
        valid = member_is_map and member_conversation and member_responses
        valid_members += int(valid)
        if on_item is not None:
            on_item(count - 1, valid)

    try:
        events = ijson.parse(handle)
        if next(events, None) != ("", "start_map", None):
            return None
        for prefix, event, value in events:
            if (
                event == "string"
                and prefix.rsplit(".", 1)[-1] in {"type", "content_type", "kind", "record_type"}
                and isinstance(value, str)
                and (
                    value.startswith(("future_", "unknown_", "unsupported_"))
                    or value in {"future", "unknown", "unsupported"}
                )
            ):
                # The ordinary admission wrapper emits an evidence event for
                # future wire types; retain that exact path for such exports.
                return None
            if prefix == "" and event == "map_key":
                if value in {"sessions", "polylogue_capture_kind"}:
                    # Dispatch gives these envelopes precedence over Grok.
                    return None
                if value in {"id", "kind", "created_at", "issue_id", "extra"}:
                    root_beads_keys.add(value)
                if value == "conversations":
                    keys += 1
            elif prefix == "conversations" and event == "start_array":
                arrays += 1
            elif prefix == "conversations.item" and event in {
                "start_map",
                "start_array",
                "string",
                "number",
                "boolean",
                "null",
            }:
                count += 1
                member_keys.clear()
                member_is_map = event == "start_map"
                member_conversation = False
                member_responses = False
                if event not in {"start_map", "start_array"}:
                    finish_member()
            elif prefix == "conversations.item" and event == "map_key" and value in {"conversation", "responses"}:
                if value in member_keys:
                    return None
                member_keys.add(value)
            elif prefix == "conversations.item.conversation" and event == "start_map":
                member_conversation = True
            elif prefix == "conversations.item.responses" and event == "start_array":
                member_responses = True
            elif prefix == "conversations.item" and event in {"end_map", "end_array"}:
                finish_member()
    except ijson.common.JSONError:
        return None
    finally:
        handle.seek(0)
    # The retained artifact taxonomy checks Beads interactions before Grok.
    # Leave this ambiguous wrapper on that exact classification path.
    if len(root_beads_keys) == 5:
        return None
    return count if keys == arrays == 1 and (count == 0 or valid_members > 0) else None


def _json_subtree(events: Iterable[tuple[str, str, object]], event: str, value: object) -> object:
    if event not in {"start_map", "start_array"}:
        return normalize_ijson_stdlib_numbers(value)
    builder = ijson.common.ObjectBuilder()
    builder.event(event, value)
    depth = 1
    for _prefix, child_event, child_value in events:
        builder.event(child_event, child_value)
        if child_event in {"start_map", "start_array"}:
            depth += 1
        elif child_event in {"end_map", "end_array"}:
            depth -= 1
            if depth == 0:
                return normalize_ijson_stdlib_numbers(builder.value)
    raise ValueError("incomplete Grok JSON member")


def _skip_json_subtree(events: Iterable[tuple[str, str, object]], event: str) -> None:
    if event not in {"start_map", "start_array"}:
        return
    depth = 1
    for _prefix, child_event, _value in events:
        if child_event in {"start_map", "start_array"}:
            depth += 1
        elif child_event in {"end_map", "end_array"}:
            depth -= 1
            if depth == 0:
                return
    raise ValueError("incomplete Grok JSON member")


def _grok_conversation_fields(events: Iterable[tuple[str, str, object]]) -> dict[str, object]:
    """Read only metadata consumed by the Grok parser, skipping other fields."""
    fields: dict[str, object] = {}
    for prefix, event, value in events:
        if prefix == "conversations.item.conversation" and event == "end_map":
            return fields
        if prefix == "conversations.item.conversation" and event == "map_key" and value in {"title", "create_time"}:
            fields.pop(value, None)
        if prefix == "conversations.item.conversation.title" and event in {"string", "number", "boolean", "null"}:
            fields["title"] = normalize_ijson_stdlib_numbers(value)
        elif prefix == "conversations.item.conversation.create_time" and event in {
            "start_map",
            "string",
            "number",
            "boolean",
            "null",
        }:
            fields["create_time"] = _json_subtree(events, event, value)
    raise ValueError("incomplete Grok conversation metadata")


def iter_grok_export_events(
    handle: JsonReadable, *, include_item: Callable[[int], bool] | None = None
) -> Iterable[tuple[str, object | None]]:
    """Yield conversation boundaries, metadata and individual responses."""
    events = iter(ijson.parse(handle))
    member_index = -1
    included = True
    for prefix, event, value in events:
        if prefix == "conversations.item" and event in {
            "start_map",
            "start_array",
            "string",
            "number",
            "boolean",
            "null",
        }:
            member_index += 1
            included = include_item(member_index) if include_item is not None else True
            yield "begin", None
            if event not in {"start_map", "start_array"}:
                yield "end", None
        elif prefix == "conversations.item" and event in {"end_map", "end_array"}:
            yield "end", None
        elif prefix == "conversations.item.conversation" and event == "start_map":
            if included:
                yield "conversation", _grok_conversation_fields(events)
            else:
                _skip_json_subtree(events, event)
        elif prefix == "conversations.item.responses" and event == "start_array":
            yield "responses", None
        elif prefix == "conversations.item.responses.item" and event in {
            "start_map",
            "start_array",
            "string",
            "number",
            "boolean",
            "null",
        }:
            if included:
                yield "response", _json_subtree(events, event, value)
            else:
                _skip_json_subtree(events, event)


def iter_json_container_records(handle: JsonReadable, prefix: str) -> Iterable[JsonValue]:
    """Yield complete array members; a corrupt suffix raises after its prefix."""
    yield from ijson.items(handle, prefix)


__all__ = [
    "ENCODING_GUESSES",
    "IjsonModuleLike",
    "JsonlDecodeError",
    "JsonReadable",
    "JsonValue",
    "LoggerLike",
    "PartialJsonStreamError",
    "decode_json_bytes",
    "decode_json_bytes_with",
    "iter_json_stream",
    "iter_json_stream_with",
    "iter_json_container_records",
    "json_record_container",
]
