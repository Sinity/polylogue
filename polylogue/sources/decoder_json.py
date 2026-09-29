"""Byte decoding and streamed JSON extraction helpers."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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


def claude_design_object_envelope(handle: JsonReadable) -> dict[str, JsonValue] | None:
    """Prove a single Design chat and retain only fields used outside messages.

    Future-shaped records stay on the ordinary admission route so its typed
    unknown event and accounting remain authoritative.
    """
    scalar_fields = {"uuid", "id", "title", "name", "is_temporary", "created_at", "updated_at"}
    # These can make the retained artifact taxonomy choose a non-session
    # document before it considers conversational message evidence.
    taxonomy_markers = {"event_type", "session_id", "timestamp", "provider", "kind", "issue_id", "extra"}
    envelope: dict[str, JsonValue] = {}
    current_key: str | None = None
    message_arrays = 0
    has_project = False
    chat_messages_array = False
    future_shape = False
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
                future_shape = True
            if prefix == "" and event == "map_key":
                current_key = str(value)
                if current_key in taxonomy_markers:
                    return None
                if current_key == "messages":
                    message_arrays += 1
                elif current_key == "project":
                    has_project = True
                continue
            if prefix == "chat_messages" and event == "start_array":
                chat_messages_array = True
            if current_key is None or prefix != current_key:
                continue
            if current_key == "messages":
                if event not in {"start_array", "end_array"}:
                    return None
            elif current_key in scalar_fields:
                if event in {"start_array", "start_map"}:
                    return None
                if event in {"string", "number", "boolean", "null"}:
                    envelope[current_key] = cast(JsonValue, normalize_ijson_stdlib_numbers(value))
    except ijson.common.JSONError:
        return None
    finally:
        handle.seek(0)
    if message_arrays != 1 or not has_project or chat_messages_array or future_shape:
        return None
    envelope["project"] = None
    return envelope


@dataclass(slots=True)
class _FutureTypeFrame:
    """The first future type in one JSON container, in parser traversal order."""

    kind: str
    key: str | None = None
    own_types: dict[str, str | None] = field(default_factory=dict)
    first_child: str | None = None
    # Map children keyed by their owning key: a repeated key replaces the
    # earlier value's candidate in place, matching the decoder's
    # last-value-wins dict (first-key position, last value).
    child_types: dict[str, str | None] = field(default_factory=dict)

    def selected(self) -> str | None:
        if self.kind == "map":
            for key in ("type", "content_type", "kind", "record_type"):
                if value := self.own_types.get(key):
                    return value
            for candidate in self.child_types.values():
                if candidate is not None:
                    return candidate
            return None
        return self.first_child


def _future_wire_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith(("future_", "unknown_", "unsupported_")) or value in {"future", "unknown", "unsupported"}:
        return value
    return None


class _FirstFutureType:
    """Select a document's future wire type in ``_unknown_wire_type`` order.

    Parser admission records the first such type of one outer record. This
    follows the same traversal over parse events, after the root map opens.
    """

    _TYPE_KEYS = frozenset({"type", "content_type", "kind", "record_type"})

    def __init__(self) -> None:
        self._frames = [_FutureTypeFrame("map")]
        self.value: str | None = None

    def observe(self, event: str, value: object) -> None:
        frame = self._frames[-1] if self._frames else None
        if frame is None:
            return
        if event == "map_key":
            frame.key = str(value)
        elif event in {"start_map", "start_array"}:
            if frame.kind == "map" and frame.key in self._TYPE_KEYS:
                frame.own_types[frame.key or ""] = None
            self._frames.append(_FutureTypeFrame("map" if event == "start_map" else "array"))
        elif event in {"end_map", "end_array"}:
            selected = self._frames.pop().selected()
            if self._frames:
                parent = self._frames[-1]
                if parent.kind == "map":
                    parent.child_types[parent.key or ""] = selected
                elif selected is not None and parent.first_child is None:
                    parent.first_child = selected
            else:
                self.value = selected
        elif frame.kind == "map":
            if frame.key in self._TYPE_KEYS:
                frame.own_types[frame.key or ""] = _future_wire_type(value) if event == "string" else None
            frame.child_types[frame.key or ""] = None


def _root_envelope_without(
    handle: JsonReadable,
    streamed: frozenset[str],
    rerouted_root_keys: frozenset[str],
) -> tuple[dict[str, JsonValue], dict[str, int]] | None:
    """Build a root object without the arrays that a caller streams separately.

    Returns the remaining document, how many arrays each ``streamed`` path
    held, and, under ``__admission_future_type``, the first future wire type
    parser admission would report. A root key in ``rerouted_root_keys``, a
    streamed path holding a non-array, or invalid JSON refuses the document.
    The pass reads the complete input, so a truncated suffix refuses it too.
    """
    builder = ijson.common.ObjectBuilder()
    future_type = _FirstFutureType()
    arrays = dict.fromkeys(streamed, 0)
    # A repeated key on the way to a streamed array would make the decoder
    # keep only its last value, while the stream would already have taken the
    # overwritten array; such documents stay on the collecting route.
    guarded = {".".join(path.split(".")[: depth + 1]) for path in streamed for depth in range(path.count(".") + 1)}
    seen_guarded: set[str] = set()
    skipped: str | None = None
    expect_array: str | None = None
    try:
        events = ijson.parse(handle)
        first = next(events, None)
        if first != ("", "start_map", None):
            return None
        builder.event("start_map", None)
        for prefix, event, value in events:
            future_type.observe(event, value)
            if expect_array is not None:
                if event != "start_array":
                    return None
                skipped, expect_array = expect_array, None
                continue
            if skipped is not None:
                if prefix == skipped and event == "end_array":
                    skipped = None
                continue
            if event == "map_key":
                path = f"{prefix}.{value}" if prefix else str(value)
                if not prefix and path in rerouted_root_keys:
                    return None
                if path in guarded:
                    if path in seen_guarded:
                        return None
                    seen_guarded.add(path)
                if path in arrays:
                    arrays[path] += 1
                    expect_array = path
                    continue
            builder.event(event, value)
    except ijson.common.JSONError:
        return None
    finally:
        handle.seek(0)
    envelope = normalize_ijson_stdlib_numbers(builder.value)
    if not isinstance(envelope, dict):
        return None
    if future_type.value is not None:
        envelope["__admission_future_type"] = future_type.value
    return cast(dict[str, JsonValue], envelope), arrays


def claude_ai_object_envelope(handle: JsonReadable) -> dict[str, JsonValue] | None:
    """Prove one claude.ai conversation and keep every root field but its messages.

    Shapes the object parser routes elsewhere (account memories, projects,
    browser captures, ``sessions`` wrappers) stay on that route.
    """
    result = _root_envelope_without(
        handle,
        frozenset({"chat_messages"}),
        frozenset({"sessions", "account_uuid", "docs", "polylogue_capture_kind"}),
    )
    if result is None or result[1]["chat_messages"] != 1:
        return None
    return result[0]


def drive_chunked_prompt_envelope(handle: JsonReadable) -> tuple[dict[str, JsonValue], str] | None:
    """Prove one AI Studio chunked prompt and name the chunk array it streams.

    Returns every document field except that array, and its ijson prefix.
    The parser reads ``chunkedPrompt.chunks`` whenever ``chunkedPrompt`` is a
    non-empty object, and the root ``chunks`` otherwise. Records the Drive
    lowering sends to another parser (Gemini CLI checkpoints, message
    objects, ChatGPT fragments, lone chunks, session wrappers, browser
    captures) stay on that route, as does a document holding both arrays.
    """
    result = _root_envelope_without(
        handle,
        frozenset({"chunkedPrompt.chunks", "chunks"}),
        frozenset({"sessions", "messages", "mapping", "sessionId", "role", "author", "polylogue_capture_kind"}),
    )
    if result is None:
        return None
    envelope, arrays = result
    prompt = envelope.get("chunkedPrompt")
    if isinstance(prompt, dict) and (prompt or arrays["chunkedPrompt.chunks"]):
        if arrays["chunkedPrompt.chunks"] != 1 or arrays["chunks"]:
            return None
        return envelope, "chunkedPrompt.chunks"
    if arrays["chunks"] != 1 or arrays["chunkedPrompt.chunks"]:
        return None
    return envelope, "chunks"


def hermes_snapshot_envelope(handle: JsonReadable) -> dict[str, JsonValue] | None:
    """Validate a Hermes snapshot while leaving its messages outside the envelope.

    Only fields read by the snapshot parser are retained. Taxonomy-sensitive
    root fields keep their type, presence, and transcript-suffix evidence in
    a bounded witness rather than their possibly giant string values. A second
    pass reads the tool catalog, one semantic event in the existing contract.
    """
    scalar_fields = {
        "session_id",
        "model",
        "system_prompt",
        "session_start",
        "last_updated",
        "base_url",
        "platform",
        "message_count",
        "polylogue_artifact",
        "state_db_path",
        "verification_db_path",
        "schema_version",
    }
    provenance_keys = {"file", "source_file", "source_path", "transcript", "session_file"}
    content_keys = {"content", "text", "message_text", "body"}
    native_markers = {"uuid", "sessionId", "parentUuid", "message", "payload", "cwd", "version"}
    envelope: dict[str, JsonValue] = {}
    current_key: str | None = None
    frames = [_FutureTypeFrame("map")]
    taxonomy_witness: dict[str, JsonValue] = {}
    message_arrays = 0
    tool_fields = 0
    first_future_type: str | None = None
    try:
        events = ijson.parse(handle)
        if next(events, None) != ("", "start_map", None):
            return None
        for prefix, event, value in events:
            if prefix == "" and event == "map_key":
                current_key = str(value)
                if current_key == "messages":
                    message_arrays += 1
                elif current_key == "tools":
                    tool_fields += 1
                if current_key in provenance_keys | content_keys | native_markers:
                    taxonomy_witness[current_key] = None
                if current_key == "steps":
                    envelope.pop("steps", None)
            if event == "map_key":
                frames[-1].key = str(value)
                continue
            if current_key == "messages" and prefix == "messages" and event not in {"start_array", "end_array"}:
                return None
            if event in {"start_array", "start_map"}:
                if len(frames) == 1 and current_key in scalar_fields:
                    envelope.pop(current_key, None)
                if len(frames) == 1 and current_key == "steps" and event == "start_array":
                    envelope["steps"] = []
                if frames[-1].kind == "map" and frames[-1].key in {"type", "content_type", "kind", "record_type"}:
                    frames[-1].own_types[frames[-1].key or ""] = None
                frames.append(_FutureTypeFrame("map" if event == "start_map" else "array"))
                continue
            if event in {"end_array", "end_map"}:
                selected = frames.pop().selected()
                if frames:
                    if selected is not None and frames[-1].first_child is None:
                        frames[-1].first_child = selected
                else:
                    first_future_type = selected
                continue
            if frames[-1].kind == "map" and frames[-1].key in {"type", "content_type", "kind", "record_type"}:
                frames[-1].own_types[frames[-1].key or ""] = _future_wire_type(value) if event == "string" else None
            if len(frames) == 1 and prefix == current_key:
                if current_key in scalar_fields:
                    envelope[current_key] = cast(JsonValue, normalize_ijson_stdlib_numbers(value))
                if current_key in provenance_keys and isinstance(value, str):
                    taxonomy_witness[current_key] = (
                        "source.json"
                        if value.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson", ".json"))
                        else "source"
                    )
                if current_key in content_keys and isinstance(value, str):
                    taxonomy_witness[current_key] = "copied" if value else ""
    except ijson.common.JSONError:
        return None
    finally:
        handle.seek(0)
    if message_arrays != 1 or tool_fields > 1:
        return None
    if not isinstance(envelope.get("session_id"), str) or not any(
        key in envelope for key in ("session_start", "last_updated", "platform")
    ):
        return None
    if tool_fields:
        tools = next(ijson.items(handle, "tools"), None)
        if isinstance(tools, list):
            envelope["tools"] = cast(JsonValue, normalize_ijson_stdlib_numbers(tools))
        handle.seek(0)
    if first_future_type is not None:
        envelope["__admission_future_type"] = first_future_type
    if taxonomy_witness:
        envelope["__taxonomy_witness"] = taxonomy_witness
    return envelope


def grok_export_item_count(
    handle: JsonReadable,
    *,
    on_item: Callable[[int, bool, str | None], None] | None = None,
    on_positive_marker: Callable[[bool], None] | None = None,
) -> int | None:
    """Validate a Grok object and report each member's shape without decoding it.

    ``on_item`` also receives the member's first future wire type, which the
    per-conversation parser admission reports; the collecting lowering admits
    conversation members only, so a future type elsewhere carries no event.
    """
    count = 0
    keys = 0
    arrays = 0
    member_keys: set[str] = set()
    member_is_map = False
    member_conversation = False
    member_responses = False
    member_future_type: _FirstFutureType | None = None
    valid_members = 0
    taxonomy_keys: set[str] = set()
    taxonomy_values: dict[str, bool] = {}
    root_field_count = 0
    messages_array = False
    messages_items = 0
    messages_positive = False
    recordish_keys = {"record_type", "sessionId", "parentUuid", "message", "payload", "tool_name", "tool_input"}
    envelope_keys = {"uuid", "sessionId", "parentUuid", "message", "payload", "cwd", "version"}
    provenance_keys = {"file", "source_file", "source_path", "transcript", "session_file"}
    content_keys = {"content", "text", "message_text", "body"}
    required_string_keys = {"id", "kind", "created_at", "issue_id", "event_type", "session_id", "timestamp"}
    taxonomy_fields = (
        recordish_keys
        | envelope_keys
        | provenance_keys
        | content_keys
        | {
            "id",
            "kind",
            "created_at",
            "issue_id",
            "extra",
            "event_type",
            "session_id",
            "timestamp",
            "provider",
            "type",
            "role",
            "mapping",
            "chat_messages",
            "chunkedPrompt",
            "chunks",
            "source",
            "cascadeId",
            "markdown",
            "session",
            "parent",
            "child",
            "conversation",
        }
    )

    def finish_member() -> None:
        nonlocal valid_members, member_future_type
        valid = member_is_map and member_conversation and member_responses
        valid_members += int(valid)
        future_type = member_future_type.value if member_future_type is not None else None
        member_future_type = None
        if on_item is not None:
            on_item(count - 1, valid, future_type)

    try:
        events = ijson.parse(handle)
        if next(events, None) != ("", "start_map", None):
            return None
        for prefix, event, value in events:
            if member_future_type is not None:
                member_future_type.observe(event, value)
            if prefix == "" and event == "map_key":
                root_field_count = min(root_field_count + 1, 17)
                if value in {"sessions", "polylogue_capture_kind"}:
                    return None
                if value == "conversations":
                    keys += 1
                elif value == "messages":
                    messages_array = False
                    messages_items = 0
                    messages_positive = False
                elif value in taxonomy_fields:
                    taxonomy_keys.add(value)
                    taxonomy_values[value] = False
            elif prefix in taxonomy_keys and event == "string":
                if prefix == "provider":
                    taxonomy_values[prefix] = value in {"claude-code", "codex"}
                elif prefix in provenance_keys:
                    taxonomy_values[prefix] = value.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson", ".json"))
                elif prefix in content_keys:
                    taxonomy_values[prefix] = bool(value)
                elif prefix == "source":
                    taxonomy_values[prefix] = value == "antigravity_language_server"
                elif prefix in {"cascadeId", "markdown"} or prefix in required_string_keys:
                    taxonomy_values[prefix] = True
            elif prefix in taxonomy_keys and event == "start_map":
                if prefix in {"mapping", "chunkedPrompt", "extra"}:
                    taxonomy_values[prefix] = True
            elif prefix in taxonomy_keys and event == "start_array":
                if prefix in {"chat_messages", "chunks"}:
                    taxonomy_values[prefix] = True
            elif prefix == "messages" and event == "start_array":
                messages_array = True
            elif prefix == "messages.item" and event in {
                "start_map",
                "start_array",
                "string",
                "number",
                "boolean",
                "null",
            }:
                messages_items += 1
            elif prefix == "messages.item" and event == "map_key" and messages_items <= 12:
                messages_positive = messages_positive or value in {"role", "content", "text", "parts", "author"}
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
                member_future_type = _FirstFutureType() if member_is_map else None
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
    if all(taxonomy_values.get(key, False) for key in ("event_type", "session_id", "timestamp", "provider")):
        return None
    if (
        not taxonomy_keys.intersection(envelope_keys)
        and any(taxonomy_values.get(key, False) for key in provenance_keys)
        and any(taxonomy_values.get(key, False) for key in content_keys)
    ):
        return None
    has_envelope = bool(taxonomy_keys.intersection(envelope_keys))
    relationship_index = {"session", "parent", "child", "type", "timestamp"} <= taxonomy_keys or {
        "conversation",
        "parent",
        "child",
        "type",
        "timestamp",
    } <= taxonomy_keys
    record_marker = bool(
        taxonomy_keys.intersection(recordish_keys)
        or ("type" in taxonomy_keys and has_envelope)
        or ("role" in taxonomy_keys and taxonomy_keys.intersection({"content", "text"}) and root_field_count <= 16)
    ) and not (relationship_index and not has_envelope)
    session_marker = (
        any(taxonomy_values.get(key, False) for key in ("mapping", "chat_messages", "chunkedPrompt", "chunks"))
        or (messages_array and messages_positive)
        or all(taxonomy_values.get(key, False) for key in ("source", "cascadeId", "markdown"))
    )
    beads_overlap = all(taxonomy_values.get(key, False) for key in ("id", "kind", "created_at", "issue_id", "extra"))
    if on_positive_marker is not None:
        on_positive_marker(bool(record_marker or session_marker) and not beads_overlap)
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
