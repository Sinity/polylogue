"""Disk-backed parsed messages for worker preparation and sealed publication."""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import sys
import threading
import uuid
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import (
    Callable,
    Container,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set,
)
from contextlib import closing, contextmanager
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO, TypeVar, cast, overload
from urllib.parse import quote

import ijson
from pydantic import BaseModel

from polylogue.core.enums import Origin
from polylogue.core.hashing import hash_text
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.message_native_identity import (
    message_native_key,
    native_id_from_key,
    source_native_id_from_json,
    source_native_id_json,
)
from polylogue.core.sql_settlement import current_native_sql_lifetimes
from polylogue.core.work_progress import advance_work_progress
from polylogue.sources import value_bounds
from polylogue.sources.decoder_json import _json_subtree, normalize_ijson_stdlib_numbers
from polylogue.sources.live.tool_result_sidecars import (
    SidecarDebt,
    SidecarMatch,
)
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSessionEvent
from polylogue.sources.parsers.base_models import SINK_JSON_CONTEXT
from polylogue.sources.parsers.claude.common import _ClaudeMessageEvidence
from polylogue.sources.pickle_spool import PickleSpool
from polylogue.sources.sidecar_evidence import RetainedSidecarScope
from polylogue.sources.streamed_event_payload import (
    StreamedJsonArray,
    ensure_streamed_json_array_table,
)
from polylogue.sources.streamed_event_payload import (
    _open_reader as _open_streamed_array_reader,
)
from polylogue.sources.value_bounds import require_storable_string
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

# The occurrence a parent id names (see the active-branch meaning below):
# the nearest earlier occurrence, else the last one.
_EARLIER_PARENT_OCCURRENCE_SQL = (
    "SELECT message_ordinal, parent_id FROM prepared_message INDEXED BY prepared_message_provider "
    "WHERE session_ordinal = ? AND provider_id = ? AND message_ordinal < ? ORDER BY message_ordinal DESC LIMIT 1"
)
_LAST_PARENT_OCCURRENCE_SQL = (
    "SELECT message_ordinal, parent_id FROM prepared_message INDEXED BY prepared_message_provider "
    "WHERE session_ordinal = ? AND provider_id = ? ORDER BY message_ordinal DESC LIMIT 1"
)


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
# A \\u escape of a surrogate, preceded by an even run of backslashes (a real
# escape, not the literal text of one).
_ESCAPED_SURROGATE = re.compile(r"(?<!\\)(?:\\\\)*\\u[dD][89a-fA-F][0-9a-fA-F]{2}")


def _text_json(value: object) -> str:
    """JSON text SQLite can store: a lone surrogate stays a ``\\uXXXX`` escape.

    Provider JSON admits lone surrogate escapes and decoding keeps them as
    code points, which are not valid UTF-8. Escaping them keeps the value
    exact through a round trip instead of failing the insert.
    """
    encoded = json.dumps(value, ensure_ascii=False)
    if encoded.isascii() or _LONE_SURROGATE.search(encoded) is None:
        # No substitution, so no second full-size copy of a large row.
        return encoded
    return _LONE_SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", encoded)


_ModelT = TypeVar("_ModelT", ParsedMessage, ParsedSessionEvent)


def _from_text_json(model: type[_ModelT], encoded: str) -> _ModelT:
    """Decode sink JSON; an escaped lone surrogate needs the stdlib decoder.

    pydantic's JSON parser rejects a surrogate escape, and the stdlib parser
    reads it exactly, so such a row is parsed once by the stdlib and validated
    in Python mode: no marked copy, dump or restore pass. The fields rendered
    differently in JSON mode (a paste digest's hex) read the
    :data:`SINK_JSON_CONTEXT` flag and parse as JSON mode would.
    """
    if model is ParsedMessage:
        payload = json.loads(encoded)
        for field in ("provider_message_id", "parent_message_provider_id"):
            if field in payload:
                payload[field] = source_native_id_from_json(payload[field])
        payload["provider_message_id"] = payload.get("provider_message_id") or ""
        return model.model_validate(payload, context=SINK_JSON_CONTEXT)
    if not _may_hold_escaped_surrogate(encoded) or _ESCAPED_SURROGATE.search(encoded) is None:
        return model.model_validate_json(encoded)
    return model.model_validate(json.loads(encoded), context=SINK_JSON_CONTEXT)


def _may_hold_escaped_surrogate(encoded: str) -> bool:
    """Whether ``encoded`` contains the ``\\u`` + ``d``/``D`` an escaped surrogate starts with.

    Every ``_ESCAPED_SURROGATE`` match contains one of these two substrings,
    so their absence -- a C substring scan -- decides the common case. The
    regex alone, with its backslash-run prefix, tried a match at every
    offset of every decoded row: 40 s of a 440 MB rollout's preparation.
    """
    return "\\ud" in encoded or "\\uD" in encoded


def _read_uri(path: Path) -> str:
    return f"file:{quote(str(path))}?mode=ro"


def _write_row(conn: sqlite3.Connection, sql: str, parameters: tuple[object, ...], *, kind: str) -> None:
    """Write one scratch row, typing SQLite's refusal of an oversized row.

    Each serialized value is bounded on its own, but SQLite applies the same
    length limit to the complete encoded row, so values just under the bound
    can still combine past it with the row's other columns.
    """
    try:
        conn.execute(sql, parameters)
    except sqlite3.DataError as exc:
        if "too big" not in str(exc):
            raise
        observed = sum(
            len(value.encode("utf-8", "surrogatepass")) if isinstance(value, str) else len(value)
            for value in parameters
            if isinstance(value, (str, bytes))
        )
        raise value_bounds.ValueBoundRefusedError(kind, observed, value_bounds.MAX_STORABLE_VALUE_BYTES) from exc


#: Byte budget for decoded sessions kept across passes, counted in the
#: memory their decoded messages hold (:func:`_decoded_size`), not in their
#: sealed JSON bytes: a decoded message holds several times its JSON. A
#: session larger than half of it is never retained and keeps streaming from
#: disk, so whale memory stays bounded.
DECODED_SESSION_BUDGET_BYTES = 64 * 1024 * 1024


def _decoded_size(root: object) -> int:
    """Bytes one decoded message holds: the deep ``sys.getsizeof`` of its object graph.

    The shared singletons a decode never allocates (``None``, booleans, enum
    members) and the interned field names of a model's attribute dictionary
    are not charged.
    """
    total = 0
    stack = [root]
    while stack:
        value = stack.pop()
        if value is None or isinstance(value, (bool, Enum)):
            continue
        total += sys.getsizeof(value)
        if isinstance(value, BaseModel):
            state = value.__dict__
            total += sys.getsizeof(state) + sys.getsizeof(value.__pydantic_fields_set__)
            stack.extend(state.values())
            if value.__pydantic_extra__:
                stack.append(value.__pydantic_extra__)
            if value.__pydantic_private__:
                stack.append(value.__pydantic_private__)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                stack.append(key)
                stack.append(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
        elif is_dataclass(value):
            if hasattr(value, "__dict__"):
                total += sys.getsizeof(value.__dict__)
            stack.extend(getattr(value, field.name) for field in fields(value))
    return total


_DecodedKey = tuple[str, int, int, int, int, int, int]


class _DecodedSessions:
    """A small process-wide LRU of fully decoded sealed sessions.

    Publishing one session walks its messages about twenty times (content
    identities, timestamps, messages, blocks, file edits, events, links,
    paste spans, ...). Each walk re-read the sealed carrier and re-ran pydantic
    validation of every message: on the fresh-build benchmark that was 45% of
    the ingest writer's CPU. A sealed carrier is immutable, so its first
    complete walk is retained for the following ones.

    Retained messages are shared between walks. The writer treats parsed
    messages as values -- it derives rows and ``model_copy`` for changes --
    and never assigns to one in place.
    """

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = budget_bytes
        self._entries: OrderedDict[_DecodedKey, tuple[tuple[ParsedMessage, ...], int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: _DecodedKey) -> tuple[ParsedMessage, ...] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: _DecodedKey, messages: tuple[ParsedMessage, ...], size: int) -> None:
        with self._lock:
            if key in self._entries or size > self.budget_bytes // 2:
                return
            self._entries[key] = (messages, size)
            self._bytes += size
            while self._bytes > self.budget_bytes and self._entries:
                _key, (_messages, evicted) = self._entries.popitem(last=False)
                self._bytes -= evicted

    def discard_path(self, path: str) -> None:
        with self._lock:
            for key in [key for key in self._entries if key[0] == path]:
                self._bytes -= self._entries.pop(key)[1]

    def discard_under(self, directory: str) -> None:
        prefix = directory.rstrip(os.sep) + os.sep
        with self._lock:
            for key in [key for key in self._entries if key[0].startswith(prefix)]:
                self._bytes -= self._entries.pop(key)[1]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


_DECODED_SESSIONS = _DecodedSessions(DECODED_SESSION_BUDGET_BYTES)

#: Oversized sealed sessions whose decoded walk is kept as a pickle spool.
#: Only a speed tier: an evicted session's next walk decodes its carrier again.
DECODED_SPOOL_SLOTS = 4


class _DecodedSpools:
    """Disk-backed decoded walks of sealed sessions too large for the LRU.

    A whale session is walked as many times as a small one, but keeping it
    decoded in memory would make memory proportional to it, so each walk
    used to re-run pydantic JSON validation over every message. The first
    complete walk now also spools the decoded messages; later walks unpickle
    them, several times cheaper, with memory still bounded by one message.
    """

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self._entries: OrderedDict[_DecodedKey, PickleSpool[ParsedMessage]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: _DecodedKey) -> PickleSpool[ParsedMessage] | None:
        with self._lock:
            spool = self._entries.get(key)
            if spool is not None:
                self._entries.move_to_end(key)
            return spool

    def put(self, key: _DecodedKey, spool: PickleSpool[ParsedMessage]) -> None:
        # Dropped spools are released by their last holder, so a replay in
        # flight keeps reading one that was evicted or discarded meanwhile.
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = spool
            while len(self._entries) > self.slots:
                self._entries.popitem(last=False)

    def discard_path(self, path: str) -> None:
        with self._lock:
            for key in [key for key in self._entries if key[0] == path]:
                del self._entries[key]

    def discard_under(self, directory: str) -> None:
        prefix = directory.rstrip(os.sep) + os.sep
        with self._lock:
            for key in [key for key in self._entries if key[0].startswith(prefix)]:
                del self._entries[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_DECODED_SPOOLS = _DecodedSpools(DECODED_SPOOL_SLOTS)


class _HeldWalk:
    """The decoded walk of one unsealed session held immutable by its producer."""

    __slots__ = ("spool", "building", "depth")

    def __init__(self) -> None:
        self.spool: PickleSpool[ParsedMessage] | None = None
        self.building = False
        self.depth = 1


#: Unsealed sessions whose producer declared them immutable for a window
#: (:meth:`SqliteMessageSink.held_walks`), by (writer connection, ordinal).
_HELD_WALKS: dict[tuple[int, int], _HeldWalk] = {}
_HELD_WALKS_LOCK = threading.Lock()


def discard_decoded_sessions(path: Path) -> None:
    """Release retained decodes of one sealed carrier before it is removed."""
    _DECODED_SESSIONS.discard_path(str(path))
    _DECODED_SPOOLS.discard_path(str(path))


def discard_decoded_sessions_under(directory: Path) -> None:
    """Release retained decodes of every carrier in a scratch tree being removed."""
    _DECODED_SESSIONS.discard_under(str(directory))
    _DECODED_SPOOLS.discard_under(str(directory))


def _message_json(value: ParsedMessage) -> str:
    payload = value.model_dump(mode="json")
    # Pydantic excludes the derived carrier from Source content dumps. The
    # preparation sink must retain it across outcome association and replay.
    for block, block_payload in zip(value.blocks, payload["blocks"], strict=True):
        if block.source_content_identity is not None:
            block_payload["source_content_identity"] = block.source_content_identity
    payload["provider_message_id"] = source_native_id_json(value.provider_message_id)
    payload["parent_message_provider_id"] = source_native_id_json(value.parent_message_provider_id)
    payload["parent_message_position"] = value.parent_message_position
    payload["owner_coordinate"] = asdict(value.owner_coordinate) if value.owner_coordinate is not None else None
    if value.active_leaf_fallback:
        payload["active_leaf_fallback"] = True
    # Each serialized record is one SQLite cell: individually storable
    # values can still combine into an unstorable row.
    return require_storable_string(_text_json(payload), kind="serialized message")


def _event_json(value: ParsedSessionEvent) -> str:
    streamed_arrays = {key: item.array_id for key, item in value.payload.items() if isinstance(item, StreamedJsonArray)}
    ordinary_payload = {key: item for key, item in value.payload.items() if key not in streamed_arrays}
    payload = value.model_dump(mode="json", exclude={"payload"})
    payload["payload"] = ordinary_payload
    payload["source_message_provider_id"] = source_native_id_json(value.source_message_provider_id)
    payload["boundary_message_position"] = value.boundary_message_position
    payload["owner_coordinate"] = asdict(value.owner_coordinate) if value.owner_coordinate is not None else None
    if streamed_arrays:
        payload = {
            "$polylogue_prepared_event": 1,
            "event": payload,
            "streamed_arrays": streamed_arrays,
        }
    return require_storable_string(_text_json(payload), kind="serialized event")


def _event_from_json(encoded: str, path: Path, connection: sqlite3.Connection | None = None) -> ParsedSessionEvent:
    """Restore explicitly tagged streamed payload arrays from the prepared store."""
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or payload.get("$polylogue_prepared_event") != 1:
        payload["source_message_provider_id"] = source_native_id_from_json(payload.get("source_message_provider_id"))
        return ParsedSessionEvent.model_validate(payload)
    event = payload.get("event")
    arrays = payload.get("streamed_arrays")
    if not isinstance(event, dict) or not isinstance(arrays, dict):
        raise ValueError("invalid streamed prepared event envelope")
    owner = connection
    if owner is None:
        with _open_streamed_array_reader(path) as reader:
            _restore_streamed_arrays(event, arrays, path, reader, marker_connection=None)
    else:
        _restore_streamed_arrays(event, arrays, path, owner, marker_connection=owner)
    event["source_message_provider_id"] = source_native_id_from_json(event.get("source_message_provider_id"))
    return ParsedSessionEvent.model_validate(event)


def _restore_streamed_arrays(
    event: dict[str, object],
    arrays: dict[str, object],
    path: Path,
    connection: sqlite3.Connection,
    *,
    marker_connection: sqlite3.Connection | None,
) -> None:
    event_payload = event.get("payload")
    if not isinstance(event_payload, dict):
        raise ValueError("streamed prepared event payload is not an object")
    for key, raw_array_id in arrays.items():
        if not isinstance(raw_array_id, str):
            raise ValueError("streamed prepared event array reference is malformed")
        row = connection.execute(
            "SELECT item_count FROM prepared_streamed_json_array WHERE array_id = ?", (raw_array_id,)
        ).fetchone()
        if row is None:
            raise ValueError("streamed prepared event array disappeared")
        event_payload[key] = StreamedJsonArray(marker_connection, path, raw_array_id, int(row[0]))


def _attachment_json(value: ParsedAttachment) -> str:
    payload = value.model_dump(mode="json")
    payload["message_provider_id"] = source_native_id_json(value.message_provider_id)
    payload["message_position"] = value.message_position
    payload["message_variant_index"] = value.message_variant_index
    payload["owner_coordinate"] = asdict(value.owner_coordinate) if value.owner_coordinate is not None else None
    payload["precomputed_blob"] = value.precomputed_blob
    payload["_prepared_inline_bytes"] = (
        base64.b64encode(value.inline_bytes).decode("ascii") if value.inline_bytes is not None else None
    )
    return require_storable_string(_text_json(payload), kind="serialized attachment")


def _attachment_from_json(encoded: str) -> ParsedAttachment:
    payload = json.loads(encoded)
    payload["message_provider_id"] = source_native_id_from_json(payload.get("message_provider_id"))
    inline = payload.pop("_prepared_inline_bytes", None)
    if inline is not None:
        payload["inline_bytes"] = base64.b64decode(inline, validate=True)
    return ParsedAttachment.model_validate(payload)


def _decode_attachment(encoded: str, path: Path, session_ordinal: int, attachment_ordinal: int) -> ParsedAttachment:
    attachment = _attachment_from_json(encoded)
    # Acquisition lookup belongs to this physical carrier row. A cohort
    # copy captures its own claim at this row; former spool paths are not
    # publication locators for the new artifact.
    return attachment.model_copy(update={"prepared_carrier_key": (str(path), session_ordinal, attachment_ordinal)})


# The envelope's pointer line, and the bare path as it also appears inside the
# retained head/tail excerpt. Both spellings resolve to the same basename.
_POINTER_RE = re.compile(r"tool-outputs/[^\s\"'\\,)]+")

# Gemini CLI's masking envelope. Either marker alone identifies a truncated
# inline rendering: the wrapper tag is absent on some tools that emit only the
# "Output too large" preamble.
_MASK_RE = re.compile(
    r"<tool_output_masked>|Output too large\. Showing first [\d,]+ and last [\d,]+ characters",
)


_ADVERTISED_EXCERPT_RE = re.compile(r"Showing first ([\d,]+) and last ([\d,]+) characters")


def _advertised_output_length(text: str) -> int:
    """The minimum length of the full output a masking envelope describes.

    The envelope excerpts the first N and last M characters of a longer
    output, so the full output holds at least N + M. Without the counts only
    a non-empty file is evidence.
    """
    match = _ADVERTISED_EXCERPT_RE.search(text)
    if match is None:
        return 1
    first, last = (group.replace(",", "") for group in match.groups())
    # A comma-only or absurdly long count quantifies nothing; the envelope
    # still marks the output as masked.
    if not (first.isdigit() and last.isdigit()) or len(first) > 18 or len(last) > 18:
        return 1
    return int(first) + int(last)


def is_masked_tool_output(text: str | None) -> bool:
    """True when ``text`` is Gemini CLI's truncated rendering of a larger output."""
    return bool(text) and _MASK_RE.search(text or "") is not None


def _tool_call_texts(tool_record: JSONDocument) -> list[str]:
    """Every string a tool call could carry a sidecar pointer in."""
    texts: list[str] = []
    results = tool_record.get("result")
    for result_item in results if isinstance(results, list) else []:
        response = json_document(json_document(result_item).get("functionResponse")).get("response")
        texts.extend(value for value in json_document(response).values() if isinstance(value, str))
    display = tool_record.get("resultDisplay")
    if isinstance(display, str):
        texts.append(display)
    elif display is not None:
        texts.append(json.dumps(display))
    return texts


class GeminiToolOutputIndex:
    """Disk-backed owner, pointer, and replacement state for one checkpoint."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.executescript("""
            CREATE TABLE gemini_tool_owner (
                tool_id TEXT PRIMARY KEY, first_ordinal INTEGER NOT NULL,
                inline_len INTEGER NOT NULL, masked INTEGER NOT NULL, complete_len INTEGER NOT NULL
            );
            CREATE TABLE gemini_tool_pointer (filename TEXT PRIMARY KEY, tool_id TEXT NOT NULL);
            CREATE TABLE gemini_tool_present (filename TEXT PRIMARY KEY);
            CREATE TABLE gemini_tool_matched (tool_id TEXT PRIMARY KEY);
            CREATE TABLE gemini_tool_debt (
                ordinal INTEGER PRIMARY KEY, filename TEXT NOT NULL, byte_size INTEGER NOT NULL,
                reason TEXT NOT NULL, file_mtime_ms INTEGER
            );
            CREATE TABLE gemini_tool_replacement (tool_id TEXT PRIMARY KEY, full_text TEXT NOT NULL);
        """)
        self._tool_ordinal = 0

    def observe(self, message: object) -> None:
        raw_calls = json_document(message).get("toolCalls")
        for item in raw_calls if isinstance(raw_calls, list) else []:
            tool_record = json_document(item)
            tool_id = tool_record.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                continue
            inline = ""
            masked = False
            complete_len = 0
            results = tool_record.get("result")
            for result_item in results if isinstance(results, list) else []:
                response = json_document(json_document(result_item).get("functionResponse")).get("response")
                output = json_document(response).get("output")
                if isinstance(output, str):
                    inline = output if len(output) > len(inline) else inline
                    if is_masked_tool_output(output):
                        masked = True
                        complete_len = max(complete_len, _advertised_output_length(output))
            self.conn.execute(
                "INSERT INTO gemini_tool_owner VALUES (?, ?, ?, ?, ?) ON CONFLICT(tool_id) "
                "DO UPDATE SET inline_len = excluded.inline_len, masked = excluded.masked, "
                "complete_len = excluded.complete_len",
                (tool_id, self._tool_ordinal, len(inline), int(masked), complete_len),
            )
            self._tool_ordinal += 1
            for text in _tool_call_texts(tool_record):
                for pointer in _POINTER_RE.findall(text):
                    self.conn.execute(
                        "INSERT OR IGNORE INTO gemini_tool_pointer VALUES (?, ?)",
                        (os.path.basename(pointer), tool_id),
                    )

    def _owner_for_stem(self, stem: str) -> str | None:
        exact = self.conn.execute("SELECT tool_id FROM gemini_tool_owner WHERE tool_id = ?", (stem,)).fetchone()
        if exact is not None:
            return str(exact[0])
        row = self.conn.execute(
            "SELECT tool_id FROM gemini_tool_owner WHERE instr(?, '_' || tool_id || '_') > 0 "
            "OR substr(?, -length(tool_id) - 1) = '_' || tool_id "
            "OR substr(?, 1, length(tool_id) + 1) = tool_id || '_' "
            "ORDER BY length(tool_id) DESC, first_ordinal LIMIT 1",
            (stem, stem, stem),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def _pointer_owner(self, filename: str, stem: str) -> str | None:
        row = self.conn.execute(
            "SELECT tool_id FROM gemini_tool_pointer WHERE filename IN (?, ?) "
            "ORDER BY CASE filename WHEN ? THEN 0 ELSE 1 END LIMIT 1",
            (filename, stem, filename),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def join(self, scope: RetainedSidecarScope) -> Iterator[SidecarMatch | SidecarDebt]:
        """Yield ordered matches, then ordered debt, keeping only one file in memory."""
        if not scope.available:
            return
        debt_ordinal = 0
        for entry in sorted(scope.files, key=lambda candidate: candidate.filename):
            self.conn.execute("INSERT OR IGNORE INTO gemini_tool_present VALUES (?)", (entry.filename,))
            stem = entry.filename.rsplit(".", 1)[0]
            tool_id = self._owner_for_stem(stem) or self._pointer_owner(entry.filename, stem)
            owner = (
                self.conn.execute(
                    "SELECT inline_len, masked, complete_len FROM gemini_tool_owner WHERE tool_id = ?", (tool_id,)
                ).fetchone()
                if tool_id is not None
                else None
            )
            reason = None
            full_text = ""
            if owner is None:
                reason = "no_owning_tool_call"
            elif entry.byte_size > value_bounds.MAX_STORABLE_VALUE_BYTES:
                # The one physical limit: a value SQLite cannot store in a
                # cell is refused typed, never truncated. Files are joined one
                # at a time, and the replaced block carries the full text
                # anyway, so no machine-dependent memory share applies.
                reason = value_bounds.VALUE_BOUND_REFUSED
            else:
                try:
                    full_text = value_bounds.require_storable_string(entry.read_text(), kind="gemini tool sidecar")
                except OSError as exc:
                    reason = f"read_error:{type(exc).__name__}"
                except value_bounds.ValueBoundRefusedError:
                    # Replacement characters for invalid UTF-8 can expand a
                    # file under the byte limit past it once decoded.
                    reason = value_bounds.VALUE_BOUND_REFUSED
                else:
                    if bool(owner[1]) and len(full_text) < int(owner[2]):
                        # A file still being written can read as an empty or
                        # partial prefix. The envelope advertises how much it
                        # excerpts ("first N and last M characters"); a
                        # sidecar shorter than that is not the full output.
                        reason = "sidecar_less_complete_than_inline"
                        full_text = ""
            if reason is not None:
                self.conn.execute(
                    "INSERT INTO gemini_tool_debt VALUES (?, ?, ?, ?, ?)",
                    (debt_ordinal, entry.filename, entry.byte_size, reason, entry.file_mtime_ms),
                )
                debt_ordinal += 1
                continue
            assert tool_id is not None and owner is not None
            was_truncated = bool(owner[1]) or len(full_text) > int(owner[0])
            self.conn.execute("INSERT OR IGNORE INTO gemini_tool_matched VALUES (?)", (tool_id,))
            if was_truncated:
                self.conn.execute(
                    "INSERT INTO gemini_tool_replacement VALUES (?, ?) "
                    "ON CONFLICT(tool_id) DO UPDATE SET full_text = excluded.full_text",
                    (tool_id, full_text),
                )
            yield SidecarMatch(
                tool_use_id=tool_id,
                filename=entry.filename,
                byte_size=entry.byte_size,
                content_hash=hash_text(full_text),
                was_truncated=was_truncated,
                # The validated copy was hashed and stored in the disk-backed
                # replacement table above. Keep only a re-open handle here;
                # a match must not retain one full output per sidecar.
                read_text=entry.read_text,
                file_mtime_ms=entry.file_mtime_ms,
            )
        for filename, tool_id in self.conn.execute(
            "SELECT filename, tool_id FROM gemini_tool_pointer ORDER BY filename"
        ):
            present = self.conn.execute("SELECT 1 FROM gemini_tool_present WHERE filename = ?", (filename,)).fetchone()
            matched = self.conn.execute("SELECT 1 FROM gemini_tool_matched WHERE tool_id = ?", (tool_id,)).fetchone()
            if present is None and matched is None:
                self.conn.execute(
                    "INSERT INTO gemini_tool_debt VALUES (?, ?, 0, 'expected_sidecar_not_retained', NULL)",
                    (debt_ordinal, filename),
                )
                debt_ordinal += 1
        for filename, byte_size, reason, file_mtime_ms in self.conn.execute(
            "SELECT filename, byte_size, reason, file_mtime_ms FROM gemini_tool_debt ORDER BY ordinal"
        ):
            yield SidecarDebt(filename, byte_size, reason, file_mtime_ms)

    def replacement_for(self, tool_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT full_text FROM gemini_tool_replacement WHERE tool_id = ?", (tool_id,)
        ).fetchone()
        return str(row[0]) if row is not None else None

    def close(self) -> None:
        for name in ("owner", "pointer", "present", "matched", "debt", "replacement"):
            self.conn.execute(f"DROP TABLE gemini_tool_{name}")


# The active-branch meaning a prepared session is lowered with.
#
# One meaning, two physical forms: ``normalize_active_branch`` lowers a
# resident message list, and ``SqliteMessageSink.normalize_active_path`` lowers
# a disk-backed session with the same rules, so either preparation route
# publishes the same leaf and path values.
#
# - **Leaf.** A session whose producer marked exactly one occurrence as the
#   active leaf keeps that occurrence. Otherwise the last message becomes the
#   leaf as a storage default, and it carries ``active_leaf_fallback`` so a
#   later pass never reads its own default as producer evidence.
# - **Path.** Only a producer leaf implies an active path. The walk starts at
#   the leaf occurrence itself and follows each occurrence's own parent id.
#   A parent id names the nearest earlier occurrence with that provider id
#   (a transcript records a parent before its child); with no earlier one it
#   names the last occurrence. Only the occurrences on that chain are marked:
#   another occurrence repeating a chain member's provider id is a different
#   message and keeps its own value. A cycle ends the walk.
# - **Idempotence.** Lowering an already lowered session changes nothing.


def producer_leaf_position(messages: Sequence[ParsedMessage]) -> int | None:
    """The producer-marked leaf occurrence, or ``None`` when the storage default applies."""
    marked = [position for position, message in enumerate(messages) if message.is_active_leaf]
    if len(marked) != 1 or messages[marked[0]].active_leaf_fallback:
        return None
    return marked[0]


def parent_occurrence(
    occurrences: Sequence[int],
    child_position: int,
) -> int | None:
    """The occurrence a child's parent id names, from that id's ascending positions."""
    if not occurrences:
        return None
    earlier = bisect_left(occurrences, child_position)
    return occurrences[earlier - 1] if earlier else occurrences[-1]


def normalize_active_branch(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    """Settle leaf and active-path values for one resident session."""
    if not messages:
        return messages
    leaf = producer_leaf_position(messages)
    if leaf is None:
        last = len(messages) - 1
        return [
            message.model_copy(update={"is_active_leaf": position == last, "active_leaf_fallback": position == last})
            if bool(message.is_active_leaf) != (position == last) or message.active_leaf_fallback != (position == last)
            else message
            for position, message in enumerate(messages)
        ]
    if not messages[leaf].provider_message_id:
        return messages
    positions_by_id: dict[str, list[int]] = {}
    for position, message in enumerate(messages):
        if message.provider_message_id:
            positions_by_id.setdefault(message.provider_message_id, []).append(position)
    chain: set[int] = set()
    cursor: int | None = leaf
    while cursor is not None and cursor not in chain:
        chain.add(cursor)
        parent_id = messages[cursor].parent_message_provider_id
        cursor = parent_occurrence(positions_by_id.get(parent_id, ()), cursor) if parent_id else None
    return [
        message.model_copy(update={"is_active_path": True})
        if position in chain and message.is_active_path is not True
        else message
        for position, message in enumerate(messages)
    ]


class SqliteMessageSink(MutableSequence[ParsedMessage]):
    """One session's messages, in stable ordinal order without a resident list."""

    def __init__(
        self,
        path: Path,
        session_ordinal: int,
        *,
        writer: sqlite3.Connection | None = None,
        count: int = 0,
        store: SqliteMessageStore | None = None,
    ) -> None:
        self.path = path
        self.session_ordinal = session_ordinal
        self._writer = writer
        self._count = count
        self._store = store

    def __len__(self) -> int:
        return self._count

    def _ordinal(self, index: int) -> int:
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        return ordinal

    @overload
    def __getitem__(self, index: int) -> ParsedMessage: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedMessage]: ...

    def __getitem__(self, index: int | slice) -> ParsedMessage | list[ParsedMessage]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = self._ordinal(index)
        if self._writer is not None:
            row = self._writer.execute(
                "SELECT message_json FROM prepared_message WHERE session_ordinal = ? AND message_ordinal = ?",
                (self.session_ordinal, ordinal),
            ).fetchone()
        else:
            key = self._decoded_key()
            decoded = _DECODED_SESSIONS.get(key) if key is not None else None
            if decoded is not None:
                return decoded[ordinal]
            with _prepared_reader(self.path) as conn:
                row = conn.execute(
                    "SELECT message_json FROM prepared_message WHERE session_ordinal = ? AND message_ordinal = ?",
                    (self.session_ordinal, ordinal),
                ).fetchone()
        if row is None:
            raise ValueError("prepared message row disappeared")
        return _from_text_json(ParsedMessage, row[0])

    @overload
    def __setitem__(self, index: int, value: ParsedMessage) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ParsedMessage]) -> None: ...

    def __setitem__(self, index: int | slice, value: ParsedMessage | Iterable[ParsedMessage]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared messages are immutable")
        if isinstance(index, slice):
            raise TypeError("prepared message sink does not support slice replacement")
        if not isinstance(value, ParsedMessage):
            raise TypeError("prepared message replacement must be a ParsedMessage")
        self._require_not_held()
        ordinal = self._ordinal(index)
        self._writer.execute(
            "UPDATE prepared_message SET message_json = ?, provider_id = ?, parent_id = ?, active_leaf = ? "
            "WHERE session_ordinal = ? AND message_ordinal = ?",
            (
                _message_json(value),
                message_native_key(value.provider_message_id),
                message_native_key(value.parent_message_provider_id),
                int(bool(value.is_active_leaf)),
                self.session_ordinal,
                ordinal,
            ),
        )

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("prepared messages cannot be deleted")

    def insert(self, index: int, value: ParsedMessage) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared messages are immutable")
        if index != self._count:
            raise TypeError("prepared messages can only be appended")
        self._require_not_held()
        advance_work_progress(messages=1)
        _write_row(
            self._writer,
            "INSERT INTO prepared_message VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.session_ordinal,
                self._count,
                _message_json(value),
                message_native_key(value.provider_message_id),
                message_native_key(value.parent_message_provider_id),
                int(bool(value.is_active_leaf)),
            ),
            kind="prepared message row",
        )
        self._count += 1

    def __iter__(self) -> Iterator[ParsedMessage]:
        yield from self.iter_from(0)

    def _held_key(self) -> tuple[int, int] | None:
        return None if self._writer is None else (id(self._writer), self.session_ordinal)

    def _require_not_held(self) -> None:
        key = self._held_key()
        if key is not None and key in _HELD_WALKS:
            raise RuntimeError("prepared messages are held immutable while their walks are retained")

    @contextmanager
    def held_walks(self) -> Iterator[None]:
        """Declare this unsealed session immutable and reuse its decoded walk meanwhile.

        Preparing one session walks its messages several times (content
        identities, owners, rows, blocks, tool outcomes), and each walk re-ran
        pydantic validation of every stored message. Inside this window the
        first complete walk is spooled and later walks replay the spool, with
        memory still bounded by one message; each replay yields fresh
        objects, as a decode does. Any edit of the session inside the window
        raises instead of being silently missed by a replay. A sealed sink is
        immutable already and keeps its own decoded cache.
        """
        key = self._held_key()
        if key is None:
            yield
            return
        with _HELD_WALKS_LOCK:
            held = _HELD_WALKS.get(key)
            if held is None:
                _HELD_WALKS[key] = _HeldWalk()
            else:
                held.depth += 1
        try:
            yield
        finally:
            with _HELD_WALKS_LOCK:
                held = _HELD_WALKS[key]
                held.depth -= 1
                if not held.depth:
                    del _HELD_WALKS[key]
                    if held.spool is not None:
                        held.spool.close()

    def _iter_writer_rows(self, start: int) -> Iterator[ParsedMessage]:
        assert self._writer is not None
        cursor = self._writer.execute(
            "SELECT message_json FROM prepared_message WHERE session_ordinal = ? "
            "AND message_ordinal >= ? ORDER BY message_ordinal",
            (self.session_ordinal, start),
        )
        for row in cursor:
            yield _from_text_json(ParsedMessage, row[0])

    def _iter_held(self, held: _HeldWalk, start: int) -> Iterator[ParsedMessage]:
        spool = held.spool
        if spool is not None and len(spool) == self._count:
            yield from spool.iter_from(start)
            return
        with _HELD_WALKS_LOCK:
            build = start == 0 and spool is None and not held.building
            if build:
                held.building = True
        if not build:
            yield from self._iter_writer_rows(start)
            return
        building = PickleSpool[ParsedMessage](indexed=True)
        kept = False
        try:
            for message in self._iter_writer_rows(0):
                building.append(message)
                yield message
            if len(building) == self._count:
                held.spool = building
                kept = True
        finally:
            held.building = False
            if not kept:
                building.close()

    def provider_message_ids(self, *, include_none: bool) -> Set[str | None]:
        """A disk-backed set for membership comparison of large sessions."""
        return SqliteProviderMessageIds(self, include_none=include_none)

    def occurred_at_bounds(self) -> tuple[int | None, int | None]:
        """The ``occurred_at_ms`` extrema of the stored rows, without decoding a message.

        Read from the rows as they are now, so an unsealed sink edited after
        an earlier call reports its current timeline.
        """
        sql = (
            "SELECT MIN(json_extract(message_json, '$.occurred_at_ms')), "
            "MAX(json_extract(message_json, '$.occurred_at_ms')) "
            "FROM prepared_message WHERE session_ordinal = ?"
        )
        if self._writer is not None:
            row = self._writer.execute(sql, (self.session_ordinal,)).fetchone()
        else:
            with _prepared_reader(self.path) as conn:
                row = conn.execute(sql, (self.session_ordinal,)).fetchone()
        low, high = row if row is not None else (None, None)
        return (int(low) if low is not None else None, int(high) if high is not None else None)

    def iter_from(self, start: int) -> Iterator[ParsedMessage]:
        """Stream a suffix without decoding or scanning its inherited prefix."""
        for message in self._iter_from(start):
            advance_work_progress(messages=1)
            yield message

    def _iter_from(self, start: int) -> Iterator[ParsedMessage]:
        if start < 0 or start > self._count:
            raise IndexError(start)
        if self._writer is not None:
            held = _HELD_WALKS.get((id(self._writer), self.session_ordinal))
            yield from (self._iter_held(held, start) if held is not None else self._iter_writer_rows(start))
            return
        key = self._decoded_key()
        decoded = _DECODED_SESSIONS.get(key) if key is not None else None
        if decoded is not None:
            yield from decoded[start:]
            return
        spooled = _DECODED_SPOOLS.get(key) if key is not None else None
        if spooled is not None:
            yield from spooled.iter_from(start)
            return
        retained: list[ParsedMessage] | None = [] if key is not None and start == 0 else None
        spool: PickleSpool[ParsedMessage] | None = None
        retained_bytes = 0
        for row in _prepared_ordinal_rows(
            self.path,
            table="prepared_message",
            ordinal="message_ordinal",
            columns="message_json",
            session=self.session_ordinal,
            start=start,
        ):
            message = _from_text_json(ParsedMessage, cast(str, row[0]))
            if retained is not None:
                retained_bytes += _decoded_size(message)
                if retained_bytes > _DECODED_SESSIONS.budget_bytes // 2:
                    # Too large to keep decoded in memory: the rest of
                    # this walk goes to a spool the next walks replay.
                    spool = PickleSpool[ParsedMessage](indexed=True)
                    for earlier in retained:
                        spool.append(earlier)
                    retained = None
                else:
                    retained.append(message)
            if spool is not None:
                spool.append(message)
            yield message
        # Only a walk that reached the end holds the whole session.
        # An empty session costs nothing to decode and would occupy an LRU
        # entry the byte budget never charges for.
        if key is not None and retained and len(retained) == self._count:
            _DECODED_SESSIONS.put(key, tuple(retained), retained_bytes)
        elif key is not None and spool is not None and len(spool) == self._count:
            _DECODED_SPOOLS.put(key, spool)

    def _decoded_key(self) -> _DecodedKey | None:
        """Identify this sealed session's bytes, or ``None`` when unreadable."""
        try:
            stat = os.stat(self.path)
        except OSError:
            return None
        return (
            str(self.path),
            self.session_ordinal,
            stat.st_dev,
            stat.st_ino,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
        )

    def normalize_active_path(self) -> SqliteMessageSink:
        """Lower leaf and path values as ``normalize_active_branch`` does, without a message list."""
        if self._writer is None:
            raise ValueError("active-path normalization requires the original mutable scratch operand")
        if not self._count:
            return self
        leaves = self._writer.execute(
            "SELECT message_ordinal, provider_id, parent_id, message_json FROM prepared_message "
            "WHERE session_ordinal = ? AND active_leaf = 1 ORDER BY message_ordinal LIMIT 2",
            (self.session_ordinal,),
        ).fetchall()
        if len(leaves) != 1 or _from_text_json(ParsedMessage, leaves[0][3]).active_leaf_fallback:
            self._settle_fallback_leaf()
            return self
        leaf_ordinal, leaf_id, leaf_parent, _leaf_json = leaves[0]
        if not leaf_id:
            return self
        self._writer.execute("DROP TABLE IF EXISTS temp.prepared_active_path")
        self._writer.execute("CREATE TEMP TABLE prepared_active_path (message_ordinal INTEGER PRIMARY KEY)")
        # The walk starts at the leaf occurrence itself and follows each
        # occurrence's own parent id: another occurrence repeating a provider
        # id may name a different parent.
        self._writer.execute("INSERT INTO prepared_active_path VALUES (?)", (leaf_ordinal,))
        child_ordinal: int = leaf_ordinal
        parent_id: str | None = leaf_parent
        while parent_id:
            parent = (
                self._writer.execute(
                    _EARLIER_PARENT_OCCURRENCE_SQL, (self.session_ordinal, parent_id, child_ordinal)
                ).fetchone()
                or self._writer.execute(_LAST_PARENT_OCCURRENCE_SQL, (self.session_ordinal, parent_id)).fetchone()
            )
            if parent is None:
                break
            result = self._writer.execute("INSERT OR IGNORE INTO prepared_active_path VALUES (?)", (parent[0],))
            if result.rowcount == 0:
                break
            child_ordinal, parent_id = parent
        ordinal = -1
        while True:
            row = self._writer.execute(
                "SELECT message_ordinal FROM temp.prepared_active_path WHERE message_ordinal > ? "
                "ORDER BY message_ordinal LIMIT 1",
                (ordinal,),
            ).fetchone()
            if row is None:
                break
            ordinal = row[0]
            message = self[ordinal]
            if message.is_active_path is not True:
                self[ordinal] = message.model_copy(update={"is_active_path": True})
        self._writer.execute("DROP TABLE temp.prepared_active_path")
        return self

    def _settle_fallback_leaf(self) -> None:
        """Make the last message the storage-default leaf, marked as not producer evidence."""
        assert self._writer is not None
        last = self._count - 1
        ordinal = -1
        while True:
            row = self._writer.execute(
                "SELECT message_ordinal FROM prepared_message WHERE session_ordinal = ? AND message_ordinal > ? "
                "AND (active_leaf = 1 OR message_ordinal = ?) ORDER BY message_ordinal LIMIT 1",
                (self.session_ordinal, ordinal, last),
            ).fetchone()
            if row is None:
                return
            ordinal = row[0]
            expected = ordinal == last
            message = self[ordinal]
            if bool(message.is_active_leaf) != expected or message.active_leaf_fallback != expected:
                self[ordinal] = message.model_copy(
                    update={"is_active_leaf": expected, "active_leaf_fallback": expected}
                )

    @contextmanager
    def atomic_edit(self) -> Iterator[None]:
        if self._writer is None:
            raise TypeError("sealed prepared messages are immutable")
        name = f"prepared_edit_{uuid.uuid4().hex}"
        count_before = self._count
        self._writer.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self._writer.execute(f"ROLLBACK TO {name}")
            self._writer.execute(f"RELEASE {name}")
            self._count = count_before
            raise
        else:
            self._writer.execute(f"RELEASE {name}")

    def normalized_messages(self, events: Sequence[ParsedSessionEvent], *, origin: Origin) -> SqliteMessageSink:
        """Borrow this artifact's separately retained canonical writer operand."""
        if self._writer is None:
            with _prepared_reader(self.path) as connection:
                row = connection.execute(
                    "SELECT normalized_ordinal FROM prepared_message_normalization WHERE original_ordinal=?",
                    (self.session_ordinal,),
                ).fetchone()
            if row is None:
                raise ValueError("sealed parser messages lack their canonical normalized operand")
            return SqliteMessageSink(self.path, int(row[0]), count=self._count)
        row = self._writer.execute(
            "SELECT normalized_ordinal FROM prepared_message_normalization WHERE original_ordinal=?",
            (self.session_ordinal,),
        ).fetchone()
        if row is not None:
            return SqliteMessageSink(self.path, int(row[0]), writer=self._writer, count=self._count, store=self._store)
        if self._store is None or self._store.conn is not self._writer:
            raise ValueError("message normalization requires its original scratch store")
        from polylogue.sources.tool_outcomes import derive_tool_outcomes

        normalized = self._store.new_sink()
        # One pass: each original message is decoded once and appended in its
        # normalized form. Active-path normalization only sets leaf flags,
        # which tool-outcome normalization neither reads nor writes.
        derive_tool_outcomes(self, events, origin=origin, into=normalized)
        normalized.normalize_active_path()
        self._writer.executemany(
            "INSERT INTO prepared_message_normalization VALUES (?,?)",
            (
                (self.session_ordinal, normalized.session_ordinal),
                (normalized.session_ordinal, normalized.session_ordinal),
            ),
        )
        return normalized


class SqliteProviderMessageIds(Set[str | None]):
    """Native IDs backed by the prepared-message provider index."""

    def __init__(self, messages: SqliteMessageSink, *, include_none: bool) -> None:
        self.messages = messages
        self.include_none = include_none

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self.messages._writer is not None:
            yield self.messages._writer
        else:
            with _prepared_reader(self.messages.path) as connection:
                yield connection

    def _where(self, alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return f"{prefix}session_ordinal = ?" + ("" if self.include_none else f" AND {prefix}provider_id IS NOT NULL")

    def __contains__(self, value: object) -> bool:
        if value is None and not self.include_none:
            return False
        if value is not None and not isinstance(value, str):
            return False
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM prepared_message WHERE session_ordinal = ? AND provider_id IS ? LIMIT 1",
                (self.messages.session_ordinal, message_native_key(value)),
            ).fetchone()
            return row is not None

    def __iter__(self) -> Iterator[str | None]:
        if self.messages._writer is not None:
            for (provider_id,) in self.messages._writer.execute(
                f"SELECT DISTINCT provider_id FROM prepared_message WHERE {self._where()} ORDER BY provider_id",
                (self.messages.session_ordinal,),
            ):
                yield native_id_from_key(provider_id) if provider_id is not None else None
            return
        if self.include_none:
            with _prepared_reader(self.messages.path) as connection:
                has_none = (
                    connection.execute(
                        "SELECT 1 FROM prepared_message WHERE session_ordinal = ? AND provider_id IS NULL LIMIT 1",
                        (self.messages.session_ordinal,),
                    ).fetchone()
                    is not None
                )
            if has_none:
                yield None
        after: str | None = None
        while True:
            with _prepared_reader(self.messages.path) as connection:
                rows = connection.execute(
                    "SELECT DISTINCT provider_id FROM prepared_message WHERE session_ordinal = ? "
                    "AND provider_id IS NOT NULL AND (? IS NULL OR provider_id > ?) ORDER BY provider_id LIMIT 512",
                    (self.messages.session_ordinal, after, after),
                ).fetchall()
            if not rows:
                return
            after = str(rows[-1][0])
            for (provider_id,) in rows:
                yield native_id_from_key(str(provider_id))

    def __len__(self) -> int:
        with self._connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT provider_id FROM prepared_message WHERE {self._where()})",
                (self.messages.session_ordinal,),
            ).fetchone()
            return int(row[0])

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        if isinstance(other, SqliteProviderMessageIds):
            with _prepared_reader(self.messages.path) as conn:
                other_table = "prepared_message"
                if other.messages.path != self.messages.path:
                    conn.execute("ATTACH DATABASE ? AS other_prepared", (_read_uri(other.messages.path),))
                    other_table = "other_prepared.prepared_message"
                left = self._where()
                right = other._where()
                row = conn.execute(
                    f"SELECT 1 FROM (SELECT provider_id FROM prepared_message WHERE {left} "
                    f"EXCEPT SELECT provider_id FROM {other_table} WHERE {right}) LIMIT 1",
                    (self.messages.session_ordinal, other.messages.session_ordinal),
                ).fetchone()
                return row is None
        return all(value in other for value in self)


class SqliteAttachmentSink(MutableSequence[ParsedAttachment]):
    """Ordered attachments retained in the same sealed carrier as messages."""

    def __init__(
        self,
        path: Path,
        session_ordinal: int,
        *,
        writer: sqlite3.Connection | None = None,
        count: int = 0,
    ) -> None:
        self.path = path
        self.session_ordinal = session_ordinal
        self._writer = writer
        self._count = count

    def __len__(self) -> int:
        return self._count

    def _ordinal(self, index: int) -> int:
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        return ordinal

    @overload
    def __getitem__(self, index: int) -> ParsedAttachment: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedAttachment]: ...

    def __getitem__(self, index: int | slice) -> ParsedAttachment | list[ParsedAttachment]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = self._ordinal(index)
        if self._writer is not None:
            row = self._writer.execute(
                "SELECT attachment_json FROM prepared_attachment WHERE session_ordinal = ? AND attachment_ordinal = ?",
                (self.session_ordinal, ordinal),
            ).fetchone()
        else:
            with _prepared_reader(self.path) as conn:
                row = conn.execute(
                    "SELECT attachment_json FROM prepared_attachment WHERE session_ordinal = ? AND attachment_ordinal = ?",
                    (self.session_ordinal, ordinal),
                ).fetchone()
        if row is None:
            raise ValueError("prepared attachment row disappeared")
        return _decode_attachment(row[0], self.path, self.session_ordinal, ordinal)

    @overload
    def __setitem__(self, index: int, value: ParsedAttachment) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ParsedAttachment]) -> None: ...

    def __setitem__(self, index: int | slice, value: ParsedAttachment | Iterable[ParsedAttachment]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared attachments are immutable")
        if isinstance(index, slice) or not isinstance(value, ParsedAttachment):
            raise TypeError("prepared attachment replacement needs one attachment")
        _write_row(
            self._writer,
            "UPDATE prepared_attachment SET attachment_json = ? WHERE session_ordinal = ? AND attachment_ordinal = ?",
            (_attachment_json(value), self.session_ordinal, self._ordinal(index)),
            kind="prepared attachment row",
        )

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("prepared attachments cannot be deleted")

    def insert(self, index: int, value: ParsedAttachment) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared attachments are immutable")
        if index != self._count:
            raise TypeError("prepared attachments can only be appended")
        _write_row(
            self._writer,
            "INSERT INTO prepared_attachment VALUES (?, ?, ?)",
            (self.session_ordinal, self._count, _attachment_json(value)),
            kind="prepared attachment row",
        )
        self._count += 1

    @contextmanager
    def original_items_for_rewrite(self) -> Iterator[Iterator[ParsedAttachment]]:
        """Keep exact original carrier rows while an expansion overwrites slots."""
        if self._writer is None:
            raise TypeError("sealed prepared attachments cannot be rewritten")
        conn = self._writer
        table = "attachment_rewrite_" + uuid.uuid4().hex
        cursor = None
        try:
            conn.execute(
                f"CREATE TEMP TABLE {table} AS SELECT attachment_ordinal, attachment_json "
                "FROM prepared_attachment WHERE session_ordinal = ? ORDER BY attachment_ordinal",
                (self.session_ordinal,),
            )
            cursor = conn.execute(
                f"SELECT attachment_ordinal, attachment_json FROM {table} ORDER BY attachment_ordinal"
            )

            def items() -> Iterator[ParsedAttachment]:
                while page := cursor.fetchmany(256):
                    for ordinal, encoded in page:
                        yield _decode_attachment(encoded, self.path, self.session_ordinal, ordinal)

            yield items()
        finally:
            if cursor is not None:
                cursor.close()
            conn.execute(f"DROP TABLE IF EXISTS {table}")

    def __iter__(self) -> Iterator[ParsedAttachment]:
        if self._writer is not None:
            rows = self._writer.execute(
                "SELECT attachment_ordinal, attachment_json FROM prepared_attachment WHERE session_ordinal = ? ORDER BY attachment_ordinal",
                (self.session_ordinal,),
            )
            for ordinal, encoded in rows:
                yield _decode_attachment(encoded, self.path, self.session_ordinal, ordinal)
            return
        for ordinal, encoded in _prepared_ordinal_rows(
            self.path,
            table="prepared_attachment",
            ordinal="attachment_ordinal",
            columns="attachment_ordinal, attachment_json",
            session=self.session_ordinal,
        ):
            yield _decode_attachment(cast(str, encoded), self.path, self.session_ordinal, cast(int, ordinal))


class SqliteSessionEventSink(MutableSequence[ParsedSessionEvent]):
    """One session's semantic events with disk-backed deterministic ordering."""

    def __init__(
        self,
        path: Path,
        session_ordinal: int,
        *,
        writer: sqlite3.Connection | None = None,
        count: int = 0,
    ) -> None:
        self.path = path
        self.session_ordinal = session_ordinal
        self._writer = writer
        self._count = count

    def __len__(self) -> int:
        return self._count

    def _ordinal(self, index: int) -> int:
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        return ordinal

    @overload
    def __getitem__(self, index: int) -> ParsedSessionEvent: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedSessionEvent]: ...

    def __getitem__(self, index: int | slice) -> ParsedSessionEvent | list[ParsedSessionEvent]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = self._ordinal(index)
        conn = self._writer
        if conn is None:
            with _prepared_reader(self.path) as reader:
                row = reader.execute(
                    "SELECT event_json FROM prepared_event WHERE session_ordinal = ? AND event_ordinal = ?",
                    (self.session_ordinal, ordinal),
                ).fetchone()
        else:
            row = conn.execute(
                "SELECT event_json FROM prepared_event WHERE session_ordinal = ? AND event_ordinal = ?",
                (self.session_ordinal, ordinal),
            ).fetchone()
        if row is None:
            raise ValueError("prepared event row disappeared")
        return _event_from_json(row[0], self.path, self._writer)

    @overload
    def __setitem__(self, index: int, value: ParsedSessionEvent) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ParsedSessionEvent]) -> None: ...

    def __setitem__(self, index: int | slice, value: ParsedSessionEvent | Iterable[ParsedSessionEvent]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared events are immutable")
        if isinstance(index, slice) or not isinstance(value, ParsedSessionEvent):
            raise TypeError("prepared event replacement needs one event")
        ordinal = self._ordinal(index)
        self._writer.execute(
            "UPDATE prepared_event SET timestamp = ?, event_type = ?, event_json = ? WHERE session_ordinal = ? AND event_ordinal = ?",
            (value.timestamp, value.event_type, _event_json(value), self.session_ordinal, ordinal),
        )

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("prepared events cannot be deleted")

    def insert(self, index: int, value: ParsedSessionEvent) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared events are immutable")
        if index < 0:
            index = max(0, self._count + index)
        index = min(index, self._count)
        if index < self._count:
            # A two-phase ordinal shift avoids transient primary-key
            # collisions and keeps Codex's late compaction event insertion
            # on disk rather than copying the complete event list.
            self._writer.execute(
                "UPDATE prepared_event SET event_ordinal = -1 - event_ordinal "
                "WHERE session_ordinal = ? AND event_ordinal >= ?",
                (self.session_ordinal, index),
            )
            self._writer.execute(
                "UPDATE prepared_event SET event_ordinal = -event_ordinal "
                "WHERE session_ordinal = ? AND event_ordinal < 0",
                (self.session_ordinal,),
            )
        _write_row(
            self._writer,
            "INSERT INTO prepared_event (session_ordinal, event_ordinal, timestamp, event_type, event_json) VALUES (?, ?, ?, ?, ?)",
            (self.session_ordinal, index, value.timestamp, value.event_type, _event_json(value)),
            kind="prepared event row",
        )
        self._count += 1

    def insert_sorted(self, insertions: Iterable[tuple[int, ParsedSessionEvent]]) -> None:
        """Insert many events at original indices with one renumbering pass.

        ``insertions`` are ordered by index; each index refers to the sequence
        before any of them is inserted, and events sharing an index keep their
        order. Equivalent to calling :meth:`insert` for each at
        ``index + <events already inserted>``, without shifting the tail once
        per event.
        """
        if self._writer is None:
            raise TypeError("sealed prepared events are immutable")
        writer = self._writer
        writer.execute("DROP TABLE IF EXISTS temp.prepared_event_insert")
        writer.execute(
            "CREATE TEMP TABLE prepared_event_insert (seq INTEGER PRIMARY KEY, idx INTEGER NOT NULL, "
            "timestamp TEXT, event_type TEXT NOT NULL, event_json TEXT NOT NULL)"
        )
        added = 0
        previous = -1
        for index, value in insertions:
            index = min(max(index, 0), self._count)
            if index < previous:
                raise ValueError("insertions must be ordered by index")
            previous = index
            writer.execute(
                "INSERT INTO temp.prepared_event_insert VALUES (?, ?, ?, ?, ?)",
                (added, index, value.timestamp, value.event_type, _event_json(value)),
            )
            added += 1
        if added:
            # Cumulative insertions per distinct index: an existing event at
            # ordinal ``o`` moves by the count at the largest index <= ``o``,
            # found by one primary-key seek rather than a scan of every
            # insertion before it.
            writer.execute("DROP TABLE IF EXISTS temp.prepared_event_shift")
            writer.execute("CREATE TEMP TABLE prepared_event_shift (idx INTEGER PRIMARY KEY, cum INTEGER NOT NULL)")
            writer.execute(
                "INSERT INTO temp.prepared_event_shift "
                "SELECT idx, SUM(COUNT(*)) OVER (ORDER BY idx) FROM temp.prepared_event_insert GROUP BY idx"
            )
            writer.execute(
                "UPDATE prepared_event SET event_ordinal = -1 - (event_ordinal + "
                "(SELECT s.cum FROM temp.prepared_event_shift AS s WHERE s.idx <= prepared_event.event_ordinal "
                "ORDER BY s.idx DESC LIMIT 1)) "
                "WHERE session_ordinal = ? AND event_ordinal >= (SELECT MIN(idx) FROM temp.prepared_event_shift)",
                (self.session_ordinal,),
            )
            writer.execute("DROP TABLE temp.prepared_event_shift")
            writer.execute(
                "UPDATE prepared_event SET event_ordinal = -1 - event_ordinal "
                "WHERE session_ordinal = ? AND event_ordinal < 0",
                (self.session_ordinal,),
            )
            writer.execute(
                "INSERT INTO prepared_event (session_ordinal, event_ordinal, timestamp, event_type, event_json) "
                "SELECT ?, idx + seq, timestamp, event_type, event_json FROM temp.prepared_event_insert ORDER BY seq",
                (self.session_ordinal,),
            )
            self._count += added
        writer.execute("DROP TABLE temp.prepared_event_insert")

    def __iter__(self) -> Iterator[ParsedSessionEvent]:
        if self._writer is not None:
            for row in self._writer.execute(
                "SELECT event_json FROM prepared_event WHERE session_ordinal = ? ORDER BY event_ordinal",
                (self.session_ordinal,),
            ):
                yield _event_from_json(row[0], self.path, self._writer)
            return
        for (encoded,) in _prepared_ordinal_rows(
            self.path,
            table="prepared_event",
            ordinal="event_ordinal",
            columns="event_json",
            session=self.session_ordinal,
        ):
            yield _event_from_json(cast(str, encoded), self.path)

    def iter_ordered(self, type_order_tier: Mapping[str, int]) -> Iterator[ParsedSessionEvent]:
        clauses = " ".join("WHEN ? THEN ?" for _ in type_order_tier)
        tier_sql = "CASE event_type " + clauses + " ELSE 0 END" if clauses else "0"
        parameters: tuple[object, ...] = tuple(item for pair in type_order_tier.items() for item in pair)
        base_sql = (
            "WITH ordered AS (SELECT event_json, COALESCE(timestamp, '') AS stamp, "
            + tier_sql
            + " AS tier, event_ordinal FROM prepared_event WHERE session_ordinal = ?) "
            "SELECT event_json, stamp, tier, event_ordinal FROM ordered"
        )
        if self._writer is not None:
            for row in self._writer.execute(
                base_sql + " ORDER BY stamp, tier, event_ordinal", (*parameters, self.session_ordinal)
            ):
                yield _event_from_json(row[0], self.path, self._writer)
            return
        after: tuple[str, int, int] | None = None
        while True:
            predicate = " WHERE (stamp, tier, event_ordinal) > (?, ?, ?)" if after is not None else ""
            with _prepared_reader(self.path) as connection:
                rows = connection.execute(
                    base_sql + predicate + " ORDER BY stamp, tier, event_ordinal LIMIT 512",
                    (*parameters, self.session_ordinal, *(after or ())),
                ).fetchall()
            if not rows:
                return
            after = (str(rows[-1][1]), int(rows[-1][2]), int(rows[-1][3]))
            for row in rows:
                yield _event_from_json(row[0], self.path, self._writer)

    def sort_in_place(self, type_order_tier: Mapping[str, int]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared events are immutable")
        self._writer.execute(
            "UPDATE prepared_event SET sort_tier = 0 WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        for event_type, tier in type_order_tier.items():
            self._writer.execute(
                "UPDATE prepared_event SET sort_tier = ? WHERE session_ordinal = ? AND event_type = ?",
                (tier, self.session_ordinal, event_type),
            )
        # Reorder the core prefix once. Later sidecar events append after it,
        # even when their timestamps are earlier than the core's last event.
        self._writer.execute("DROP TABLE IF EXISTS temp.prepared_event_order")
        self._writer.execute(
            "CREATE TEMP TABLE prepared_event_order AS "
            "SELECT session_ordinal, event_ordinal AS old_ordinal, "
            "ROW_NUMBER() OVER (PARTITION BY session_ordinal "
            "ORDER BY COALESCE(timestamp, ''), sort_tier, event_ordinal) - 1 AS new_ordinal "
            "FROM prepared_event WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        self._writer.execute(
            "UPDATE prepared_event SET event_ordinal = -1 - ("
            "SELECT new_ordinal FROM prepared_event_order AS o "
            "WHERE o.session_ordinal = prepared_event.session_ordinal "
            "AND o.old_ordinal = prepared_event.event_ordinal) "
            "WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        self._writer.execute(
            "UPDATE prepared_event SET event_ordinal = -1 - event_ordinal WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        self._writer.execute("DROP TABLE temp.prepared_event_order")


@contextmanager
def _prepared_reader(path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect_measured(_read_uri(path), uri=True)
    owner = NativeSQLCustodyOwner(connection, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        yield owner.require_connection()
    except BaseException as primary:
        from polylogue.storage.sqlite.connection_profile import _close_failed_native_construction

        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


#: Serialized payload characters one page of prepared rows holds at most
#: (beyond its first row): a memory bound on paging, never on what is read.
_PAGE_PAYLOAD_CHARS = 1024 * 1024


def _prepared_ordinal_rows(
    path: Path, *, table: str, ordinal: str, columns: str, session: int | None, start: int = 0
) -> Iterator[tuple[object, ...]]:
    """Read indexed immutable pages and close SQL before yielding any row.

    A page ends at 512 rows or once its rows' last column (the serialized
    payload) reaches :data:`_PAGE_PAYLOAD_CHARS`, so a page of large messages
    holds no more than one of small ones. Each row is released as it is
    yielded, so the next page is never read while the previous one is held.
    """
    after = start - 1
    while True:
        rows: list[tuple[object, ...]] = []
        held = 0
        with (
            _prepared_reader(path) as connection,
            closing(
                connection.execute(
                    f"SELECT {ordinal}, {columns} FROM {table} WHERE "
                    + ("session_ordinal = ? AND " if session is not None else "")
                    + f"{ordinal} > ? ORDER BY {ordinal} LIMIT 512",
                    (session, after) if session is not None else (after,),
                )
            ) as cursor,
        ):
            while held < _PAGE_PAYLOAD_CHARS and (fetched := cursor.fetchone()) is not None:
                rows.append(tuple(fetched))
                payload = fetched[-1]
                held += len(payload) if isinstance(payload, (str, bytes)) else 0
        if not rows:
            return
        after = int(cast(int, rows[-1][0]))
        rows.reverse()
        while rows:
            row = rows.pop()
            yield tuple(row[1:])
            del row


class SqliteMessageStore:
    """Own the unsealed scratch transaction until its producer has finished."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = connect_measured(path)
        self._sql_owner = NativeSQLCustodyOwner(
            self.conn, lifetime_dependencies=(*current_native_sql_lifetimes(), self)
        )
        try:
            self.conn.execute("PRAGMA journal_mode = DELETE")
            # Scratch for one process: admission trusts the in-memory content
            # seal of the closed file, never a store that outlived a crash, so
            # syncing each commit to stable storage buys nothing.
            self.conn.execute("PRAGMA synchronous = OFF")
            # The schema is created inside the store's one transaction: as separate
            # autocommit statements each CREATE paid its own journal and fsync, per
            # prepared artifact, before any row was spooled.
            self.conn.execute("PRAGMA temp_store = FILE")
            self.conn.execute("BEGIN IMMEDIATE")
            ensure_streamed_json_array_table(self.conn)
            self.conn.execute(
                "CREATE TABLE prepared_message (session_ordinal INTEGER NOT NULL, message_ordinal INTEGER NOT NULL, message_json TEXT NOT NULL, provider_id TEXT, parent_id TEXT, active_leaf INTEGER NOT NULL, PRIMARY KEY (session_ordinal, message_ordinal)) WITHOUT ROWID"
            )
            self.conn.execute(
                "CREATE INDEX prepared_message_provider ON prepared_message(session_ordinal, provider_id, message_ordinal)"
            )
            self.conn.execute(
                "CREATE TABLE prepared_event (session_ordinal INTEGER NOT NULL, event_ordinal INTEGER NOT NULL, timestamp TEXT, event_type TEXT NOT NULL, event_json TEXT NOT NULL, sort_tier INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (session_ordinal, event_ordinal)) WITHOUT ROWID"
            )
            self.conn.execute(
                "CREATE TABLE prepared_attachment (session_ordinal INTEGER NOT NULL, attachment_ordinal INTEGER NOT NULL, attachment_json TEXT NOT NULL, PRIMARY KEY (session_ordinal, attachment_ordinal)) WITHOUT ROWID"
            )
            self.conn.execute(
                "CREATE TABLE prepared_message_normalization (original_ordinal INTEGER PRIMARY KEY, normalized_ordinal INTEGER NOT NULL)"
            )
            self._next_session_ordinal = 0
            self._next_event_ordinal = 0
            self._next_attachment_ordinal = 0
        except BaseException as primary:
            from polylogue.storage.sqlite.connection_profile import _close_failed_native_construction

            _close_failed_native_construction(self._sql_owner, primary)
            raise

    def new_sink(self) -> SqliteMessageSink:
        sink = SqliteMessageSink(self.path, self._next_session_ordinal, writer=self.conn, store=self)
        self._next_session_ordinal += 1
        return sink

    def new_event_sink(self) -> SqliteSessionEventSink:
        sink = SqliteSessionEventSink(self.path, self._next_event_ordinal, writer=self.conn)
        self._next_event_ordinal += 1
        return sink

    def new_attachment_sink(self) -> SqliteAttachmentSink:
        sink = SqliteAttachmentSink(self.path, self._next_attachment_ordinal, writer=self.conn)
        self._next_attachment_ordinal += 1
        return sink

    def close(self) -> None:
        self._sql_owner.close()


class ClaudeChatEvidence:
    """Claude chat records in scratch, rebuilt into evidence when emitted.

    Only the raw record is stored: its evidence is a pure function of the
    record, its array index and its evidence key.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.execute("CREATE TABLE claude_evidence (original_index INTEGER PRIMARY KEY, raw_json TEXT NOT NULL)")

    def put(self, evidence: _ClaudeMessageEvidence) -> None:
        self._conn.execute(
            "INSERT INTO claude_evidence VALUES (?, ?)", (evidence.original_index, json.dumps(dict(evidence.raw)))
        )

    def raw(self, original_index: int) -> dict[str, object]:
        row = self._conn.execute(
            "SELECT raw_json FROM claude_evidence WHERE original_index = ?", (original_index,)
        ).fetchone()
        if row is None:
            raise KeyError(original_index)
        raw = json.loads(row[0])
        assert isinstance(raw, dict)
        return raw

    def get(
        self,
        original_index: int,
        rebuild: Callable[[dict[str, object]], _ClaudeMessageEvidence],
    ) -> _ClaudeMessageEvidence:
        return rebuild(self.raw(original_index))

    def close(self) -> None:
        self._conn.execute("DROP TABLE claude_evidence")


class ClaudeAttachmentScratch:
    """Merged Claude attachment rows in scratch, in first-seen order."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.execute(
            "CREATE TABLE claude_attachment (ordinal INTEGER PRIMARY KEY, attachment_id TEXT NOT NULL UNIQUE, "
            "name TEXT, mime_type TEXT, attachment_json TEXT NOT NULL)"
        )

    def get(self, provider_attachment_id: str) -> ParsedAttachment | None:
        row = self._conn.execute(
            "SELECT attachment_json FROM claude_attachment WHERE attachment_id = ?", (provider_attachment_id,)
        ).fetchone()
        return _attachment_from_json(row[0]) if row is not None else None

    def put(self, attachment: ParsedAttachment) -> None:
        self._conn.execute(
            "INSERT INTO claude_attachment (attachment_id, name, mime_type, attachment_json) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(attachment_id) DO UPDATE SET name = excluded.name, mime_type = excluded.mime_type, "
            "attachment_json = excluded.attachment_json",
            (attachment.provider_attachment_id, attachment.name, attachment.mime_type, _attachment_json(attachment)),
        )

    def descriptor_owners(self) -> Callable[[str, str | None], str | None]:
        conn = self._conn
        conn.execute("DROP TABLE IF EXISTS claude_attachment_owner")
        conn.execute(
            "CREATE TABLE claude_attachment_owner AS SELECT attachment_id, name, mime_type FROM claude_attachment"
        )
        conn.execute("CREATE INDEX claude_attachment_owner_descriptor ON claude_attachment_owner(name, mime_type)")

        def owner(name: str, mime_type: str | None) -> str | None:
            rows = conn.execute(
                "SELECT attachment_id FROM claude_attachment_owner WHERE name = ? AND mime_type IS ? LIMIT 2",
                (name, mime_type),
            ).fetchall()
            return str(rows[0][0]) if len(rows) == 1 else None

        return owner

    def __iter__(self) -> Iterator[ParsedAttachment]:
        last = -1
        while True:
            row = self._conn.execute(
                "SELECT ordinal, attachment_json FROM claude_attachment WHERE ordinal > ? ORDER BY ordinal LIMIT 1",
                (last,),
            ).fetchone()
            if row is None:
                return
            last = row[0]
            yield _attachment_from_json(row[1])

    def close(self) -> None:
        self._conn.execute("DROP TABLE claude_attachment")
        self._conn.execute("DROP TABLE IF EXISTS claude_attachment_owner")


class ChatGPTNodeMapping(Mapping[str, object]):
    """Keep node bytes, insertion order, and duplicate-key resolution in scratch."""

    def __init__(self, conn: sqlite3.Connection, *, progress: Callable[[], None] | None = None) -> None:
        self.conn = conn
        self._progress = progress
        conn.execute(
            "CREATE TABLE chatgpt_node (node_key TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, "
            "node_json TEXT NOT NULL, child_ordinal INTEGER, parent_key TEXT)"
        )
        conn.execute("CREATE INDEX chatgpt_node_parent ON chatgpt_node(parent_key, ordinal)")
        conn.execute(
            "CREATE TABLE chatgpt_child (node_ordinal INTEGER NOT NULL, item_ordinal INTEGER NOT NULL, "
            "child_json TEXT NOT NULL, child_key TEXT, PRIMARY KEY (node_ordinal, item_ordinal)) WITHOUT ROWID"
        )
        conn.execute("CREATE INDEX chatgpt_child_key ON chatgpt_child(node_ordinal, child_key, item_ordinal)")
        conn.execute(
            "CREATE TABLE chatgpt_sibling (node_key TEXT PRIMARY KEY, sibling_ordinal INTEGER NOT NULL) WITHOUT ROWID"
        )
        # Sibling ordinals depend on every node's final parent, so they are
        # numbered in one pass after the last ``put`` rather than per node.
        self._siblings_current = False

    def put(self, key: str, node: object, ordinal: int) -> None:
        encoded = require_storable_string(_text_json(node), kind="serialized mapping node")
        # ``_sibling_ordinals``' grouping: only mapping nodes count, and a
        # missing or empty parent groups under the root key "".
        parent = node.get("parent") if isinstance(node, dict) else None
        parent_key = (parent if isinstance(parent, str) and parent else "") if isinstance(node, dict) else None
        self._siblings_current = False
        previous = self.conn.execute("SELECT child_ordinal FROM chatgpt_node WHERE node_key = ?", (key,)).fetchone()
        if previous is not None and previous[0] is not None:
            self.conn.execute("DELETE FROM chatgpt_child WHERE node_ordinal = ?", (previous[0],))
        self.conn.execute(
            "INSERT INTO chatgpt_node VALUES (?, ?, ?, NULL, ?) "
            "ON CONFLICT(node_key) DO UPDATE SET node_json = excluded.node_json, child_ordinal = NULL, "
            "parent_key = excluded.parent_key",
            (key, ordinal, encoded, parent_key),
        )

    def put_child(self, node_ordinal: int, item_ordinal: int, child: object) -> None:
        self.conn.execute(
            "INSERT INTO chatgpt_child VALUES (?, ?, ?, ?)",
            (
                node_ordinal,
                item_ordinal,
                _text_json(child),
                child if isinstance(child, str) else None,
            ),
        )

    def mark_children(self, key: str, ordinal: int) -> None:
        self.conn.execute("UPDATE chatgpt_node SET child_ordinal = ? WHERE node_key = ?", (ordinal, key))

    def shallow_node(self, key: str) -> object:
        if self._progress is not None:
            self._progress()
        row = self.conn.execute("SELECT node_json FROM chatgpt_node WHERE node_key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def children_are_strings(self, key: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM chatgpt_child WHERE node_ordinal = "
            "(SELECT child_ordinal FROM chatgpt_node WHERE node_key = ?) AND child_key IS NULL LIMIT 1",
            (key,),
        ).fetchone()
        return row is None

    def children_are_all_strings(self) -> bool:
        return self.conn.execute("SELECT 1 FROM chatgpt_child WHERE child_key IS NULL LIMIT 1").fetchone() is None

    def shallow_view(self) -> _ShallowChatGPTMapping:
        """Expose node shapes to the canonical parser without rebuilding child arrays."""
        return _ShallowChatGPTMapping(self)

    def sibling_ordinal(self, key: str) -> int:
        """Arrival ordinal of ``key`` among nodes naming the same parent.

        The scratch form of ``chatgpt._sibling_ordinals``: mapping order is
        the export's record order, answered by index instead of a dict over
        every node. All ordinals are numbered by one windowed scan the first
        time one is asked for after a ``put``.
        """
        if not self._siblings_current:
            self.conn.execute("DELETE FROM chatgpt_sibling")
            self.conn.execute(
                "INSERT INTO chatgpt_sibling SELECT node_key, "
                "ROW_NUMBER() OVER (PARTITION BY parent_key ORDER BY ordinal) - 1 "
                "FROM chatgpt_node WHERE parent_key IS NOT NULL"
            )
            self._siblings_current = True
        row = self.conn.execute("SELECT sibling_ordinal FROM chatgpt_sibling WHERE node_key = ?", (key,)).fetchone()
        return 0 if row is None else int(row[0])

    def declared_child_position(self, parent_key: str, child_id: object) -> int | None:
        """First index of ``child_id`` in the parent's spilled ``children`` array.

        Only a dict node spills its array; any other parent, or a parent
        whose ``children`` member is absent or not an array, lists nothing.
        """
        if not isinstance(child_id, str):
            return None
        row = self.conn.execute(
            "SELECT node_json, child_ordinal FROM chatgpt_node WHERE node_key = ?", (parent_key,)
        ).fetchone()
        if row is None or row[1] is None or not str(row[0]).startswith("{"):
            return None
        found = self.conn.execute(
            "SELECT MIN(item_ordinal) FROM chatgpt_child WHERE node_ordinal = ? AND child_key = ?",
            (row[1], child_id),
        ).fetchone()
        return int(found[0]) if found is not None and found[0] is not None else None

    def iter_children(self, key: str) -> Iterator[str]:
        for (child,) in self.conn.execute(
            "SELECT child_key FROM chatgpt_child WHERE node_ordinal = "
            "(SELECT child_ordinal FROM chatgpt_node WHERE node_key = ?) ORDER BY item_ordinal",
            (key,),
        ):
            if child is not None:
                yield child

    def __getitem__(self, key: str) -> object:
        row = self.conn.execute(
            "SELECT node_json, child_ordinal FROM chatgpt_node WHERE node_key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        node = json.loads(row[0])
        if row[1] is not None and isinstance(node, dict):
            node["children"] = [
                json.loads(child_json)
                for (child_json,) in self.conn.execute(
                    "SELECT child_json FROM chatgpt_child WHERE node_ordinal = ? ORDER BY item_ordinal", (row[1],)
                )
            ]
        return node

    def __iter__(self) -> Iterator[str]:
        for (key,) in self.conn.execute("SELECT node_key FROM chatgpt_node ORDER BY ordinal"):
            if self._progress is not None:
                self._progress()
            yield key

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chatgpt_node").fetchone()[0])

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, str)
            and self.conn.execute("SELECT 1 FROM chatgpt_node WHERE node_key = ?", (key,)).fetchone() is not None
        )


class _ShallowChatGPTMapping(Mapping[str, object]):
    """Nodes without their spilled ``children`` arrays, which stay in scratch.

    The parser reads an array only for declared sibling order, answered here
    by :meth:`declared_child_position`.
    """

    def __init__(self, mapping: ChatGPTNodeMapping) -> None:
        self.mapping = mapping

    def __getitem__(self, key: str) -> object:
        return self.mapping.shallow_node(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def declared_child_position(self, parent_key: str, child_id: object) -> int | None:
        return self.mapping.declared_child_position(parent_key, child_id)

    def sibling_ordinal(self, key: str) -> int:
        return self.mapping.sibling_ordinal(key)


class _ScratchChatGPTEntries:
    """Normalized ChatGPT messages held in scratch until final ordering.

    Implements ``chatgpt.MessageEntries``: the ordering, parent and timing
    rules are the parser's own, answered by index lookups instead of a list.
    """

    _ORDER = "(timestamp IS NULL), timestamp, idx"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute(
            "CREATE TABLE chatgpt_entry (node_key TEXT PRIMARY KEY, idx INTEGER NOT NULL, timestamp REAL, "
            "position INTEGER NOT NULL, provider_id TEXT, message_json TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX chatgpt_entry_provider ON chatgpt_entry(provider_id)")
        conn.execute(f"CREATE INDEX chatgpt_entry_order ON chatgpt_entry({self._ORDER})")

    def add(self, timestamp: float | None, idx: int, node_id: str, message: ParsedMessage) -> None:
        _write_row(
            self.conn,
            "INSERT INTO chatgpt_entry VALUES (?, ?, ?, ?, ?, ?)",
            (
                message_native_key(node_id),
                idx,
                timestamp,
                message.position,
                message_native_key(message.provider_message_id),
                _message_json(message),
            ),
            kind="normalized message row",
        )

    def ordered(self) -> Iterator[ParsedMessage]:
        # A separate cursor: the consumer writes other scratch tables while
        # this one is being stepped.
        cursor = self.conn.execute(f"SELECT message_json FROM chatgpt_entry ORDER BY {self._ORDER}")
        try:
            for (encoded,) in cursor:
                # Written by ``_message_json``: a lone surrogate is an escape
                # pydantic's parser refuses, so decode through the sink's own.
                yield _from_text_json(ParsedMessage, encoded)
        finally:
            cursor.close()

    def provider_for_node(self, node_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT provider_id FROM chatgpt_entry WHERE node_key = ?", (message_native_key(node_id),)
        ).fetchone()
        return native_id_from_key(str(row[0])) if row is not None and row[0] is not None else None

    def position_for_node(self, node_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT position FROM chatgpt_entry WHERE node_key = ?", (message_native_key(node_id),)
        ).fetchone()
        return int(row[0]) if row is not None else None

    def emitted_provider_ids(self) -> Container[str]:
        return _ScratchProviderIds(self.conn)

    def last_emitted_among(self, provider_ids: frozenset[str]) -> str | None:
        best: tuple[int, float, int, str] | None = None
        ordered_ids = sorted(key for value in provider_ids if (key := message_native_key(value)) is not None)
        for start in range(0, len(ordered_ids), 500):
            chunk = ordered_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            row = self.conn.execute(
                f"SELECT timestamp IS NULL, COALESCE(timestamp, 0.0), idx, provider_id FROM chatgpt_entry "
                f"WHERE provider_id IN ({placeholders}) "
                "ORDER BY timestamp IS NULL DESC, timestamp DESC, idx DESC LIMIT 1",
                chunk,
            ).fetchone()
            if row is not None:
                candidate = (int(row[0]), float(row[1]), int(row[2]), str(row[3]))
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        return native_id_from_key(best[3]) if best is not None else None


class _ScratchProviderIds(Container[str]):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __contains__(self, value: object) -> bool:
        return (
            isinstance(value, str)
            and self.conn.execute(
                "SELECT 1 FROM chatgpt_entry WHERE provider_id = ? LIMIT 1", (message_native_key(value),)
            ).fetchone()
            is not None
        )


class ScratchSessionSpill:
    """Scratch-backed collections for one session parsed with ``spill=``."""

    def __init__(self, store: SqliteMessageStore) -> None:
        self.store = store
        self._record_origins: _ScratchStringMap | None = None
        self._attachment_origins: _ScratchStringMap | None = None

    def entries(self) -> _ScratchChatGPTEntries:
        return _ScratchChatGPTEntries(self.store.conn)

    def messages(self) -> SqliteMessageSink:
        return self.store.new_sink()

    def attachments(self) -> SqliteAttachmentSink:
        return self.store.new_attachment_sink()

    def events(self) -> SqliteSessionEventSink:
        return self.store.new_event_sink()

    def seen_set(self) -> _ScratchStringSet:
        return _ScratchStringSet(self.store.conn)

    def string_map(self) -> _ScratchStringMap:
        return _ScratchStringMap(self.store.conn)

    def connection(self) -> sqlite3.Connection:
        return self.store.conn

    def set_record_origin(self, position: int, original_key: str) -> None:
        """Retain private raw occurrence evidence, never a provider identity."""
        if self._record_origins is None:
            self._record_origins = _ScratchStringMap(self.store.conn)
        self._record_origins[str(position)] = json.dumps(original_key, ensure_ascii=True)

    def set_attachment_record_origin(self, ordinal: int, raw_position: int) -> None:
        """Keep capture asset custody separate from message identity lowering."""
        if self._attachment_origins is None:
            self._attachment_origins = _ScratchStringMap(self.store.conn)
        self._attachment_origins[str(ordinal)] = str(raw_position)

    def attachment_record_origin(self, ordinal: int) -> int:
        if self._attachment_origins is None:
            raise KeyError(ordinal)
        return int(self._attachment_origins[str(ordinal)])

    def record_origin(self, position: int) -> str:
        if self._record_origins is None:
            raise KeyError(position)
        value = json.loads(self._record_origins[str(position)])
        if not isinstance(value, str):
            raise ValueError("stored native raw occurrence is invalid")
        return value


class _ScratchStringSet(MutableSet[str]):
    """A string set kept in scratch: one table per preparation, one id per set."""

    _next_id = 0

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scratch_string_set (set_id INTEGER NOT NULL, value TEXT NOT NULL, "
            "PRIMARY KEY (set_id, value)) WITHOUT ROWID"
        )
        type(self)._next_id += 1
        self.set_id = type(self)._next_id

    def __contains__(self, value: object) -> bool:
        return (
            isinstance(value, str)
            and self.conn.execute(
                "SELECT 1 FROM scratch_string_set WHERE set_id = ? AND value = ?", (self.set_id, value)
            ).fetchone()
            is not None
        )

    def __iter__(self) -> Iterator[str]:
        cursor = self.conn.execute(
            "SELECT value FROM scratch_string_set WHERE set_id = ? ORDER BY value", (self.set_id,)
        )
        try:
            for (value,) in cursor:
                yield str(value)
        finally:
            cursor.close()

    def __len__(self) -> int:
        return int(
            self.conn.execute("SELECT COUNT(*) FROM scratch_string_set WHERE set_id = ?", (self.set_id,)).fetchone()[0]
        )

    def add(self, value: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO scratch_string_set VALUES (?, ?)", (self.set_id, value))

    def discard(self, value: str) -> None:
        self.conn.execute("DELETE FROM scratch_string_set WHERE set_id = ? AND value = ?", (self.set_id, value))


class GenerationTimings:
    """One authoritative timing per generation, selected in SQLite.

    Every per-node fact the selection reads (branch roots, the best
    candidate per branch, related and legacy-duration message ids) and the
    owner each timing resolves to live in tables on ``conn``: the scratch
    database on the preparation route, an in-memory database otherwise. A
    parse therefore holds none of them in process memory.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        for table in ("gt_candidate", "gt_related", "gt_legacy", "gt_owner"):
            conn.execute(f"DROP TABLE IF EXISTS temp.{table}")
        conn.execute(
            "CREATE TEMP TABLE gt_candidate (branch_key TEXT PRIMARY KEY, first_ordinal INTEGER NOT NULL, "
            "s1 INTEGER NOT NULL, s2 INTEGER NOT NULL, s3 INTEGER NOT NULL, s4 INTEGER NOT NULL, s5 TEXT NOT NULL, "
            "elapsed_ms INTEGER NOT NULL, timing_json TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX temp.gt_candidate_order ON gt_candidate(first_ordinal)")
        conn.execute(
            "CREATE TEMP TABLE gt_related (branch_key TEXT NOT NULL, message_id TEXT NOT NULL, "
            "PRIMARY KEY (branch_key, message_id)) WITHOUT ROWID"
        )
        conn.execute(
            "CREATE TEMP TABLE gt_legacy (branch_key TEXT NOT NULL, message_id TEXT NOT NULL, "
            "duration_ms INTEGER NOT NULL, PRIMARY KEY (branch_key, message_id)) WITHOUT ROWID"
        )
        conn.execute("CREATE INDEX temp.gt_legacy_message ON gt_legacy(message_id)")
        conn.execute(
            "CREATE TEMP TABLE gt_owner (ordinal INTEGER PRIMARY KEY, owner TEXT NOT NULL, elapsed_ms INTEGER)"
        )
        conn.execute("CREATE INDEX temp.gt_owner_owner ON gt_owner(owner, ordinal)")
        self.branch_memo: MutableMapping[str, str] = _ScratchStringMap(conn)
        self._ordinal = 0

    def add_related(self, branch_key: str, message_id: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO gt_related VALUES (?, ?)", (branch_key, message_id))

    def set_legacy_duration(self, branch_key: str, message_id: str, duration_ms: int) -> None:
        self.conn.execute(
            "INSERT INTO gt_legacy VALUES (?, ?, ?) ON CONFLICT(branch_key, message_id) "
            "DO UPDATE SET duration_ms = excluded.duration_ms",
            (branch_key, message_id, duration_ms),
        )

    def offer(
        self, branch_key: str, score: tuple[int, int, int, int, str], elapsed_ms: int, timing: Mapping[str, object]
    ) -> None:
        """Keep ``timing`` for its branch when it outranks the kept one.

        A tie keeps the earlier candidate, exactly as ``max`` over the
        branch's candidates in arrival order did.
        """
        self.conn.execute(
            "INSERT INTO gt_candidate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(branch_key) DO UPDATE SET "
            "s1 = excluded.s1, s2 = excluded.s2, s3 = excluded.s3, s4 = excluded.s4, s5 = excluded.s5, "
            "elapsed_ms = excluded.elapsed_ms, timing_json = excluded.timing_json "
            "WHERE (excluded.s1, excluded.s2, excluded.s3, excluded.s4, excluded.s5) "
            "> (gt_candidate.s1, gt_candidate.s2, gt_candidate.s3, gt_candidate.s4, gt_candidate.s5)",
            (
                branch_key,
                self._ordinal,
                *score,
                elapsed_ms,
                json.dumps(dict(timing), sort_keys=True),
            ),
        )
        self._ordinal += 1

    def selected(self) -> Iterator[tuple[str, dict[str, object]]]:
        """Each branch's timing, in the order its first candidate arrived."""
        cursor = self.conn.execute("SELECT branch_key, timing_json FROM gt_candidate ORDER BY first_ordinal")
        try:
            for branch_key, timing_json in cursor:
                yield str(branch_key), json.loads(timing_json)
        finally:
            cursor.close()

    def related(self, branch_key: str) -> frozenset[str]:
        return frozenset(
            str(row[0])
            for row in self.conn.execute("SELECT message_id FROM gt_related WHERE branch_key = ?", (branch_key,))
        )

    def resolve(self, owner: str, elapsed_duration_ms: int) -> None:
        """Record the message a selected timing is finally anchored to."""
        self.conn.execute("INSERT INTO gt_owner (owner, elapsed_ms) VALUES (?, ?)", (owner, elapsed_duration_ms))

    def resolved_duration_ms(self, message_id: str) -> int | None:
        """The duration of the last timing anchored to ``message_id``."""
        row = self.conn.execute(
            "SELECT elapsed_ms FROM gt_owner WHERE owner = ? ORDER BY ordinal DESC LIMIT 1", (message_id,)
        ).fetchone()
        return int(row[0]) if row is not None else None

    def repeats_selected_duration(self, message_id: str) -> bool:
        """Whether ``message_id``'s legacy duration copies its branch's selected timing."""
        return (
            self.conn.execute(
                "SELECT 1 FROM gt_legacy l JOIN gt_candidate c ON c.branch_key = l.branch_key "
                "WHERE l.message_id = ? AND l.duration_ms = c.elapsed_ms LIMIT 1",
                (message_id,),
            ).fetchone()
            is not None
        )


class _ScratchStringMap(MutableMapping[str, str]):
    """A string-to-string map kept in scratch, one id per map."""

    _next_id = 0

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scratch_string_map (map_id INTEGER NOT NULL, key TEXT NOT NULL, "
            "value TEXT NOT NULL, PRIMARY KEY (map_id, key)) WITHOUT ROWID"
        )
        type(self)._next_id += 1
        self.map_id = type(self)._next_id

    def __getitem__(self, key: str) -> str:
        row = self.conn.execute(
            "SELECT value FROM scratch_string_map WHERE map_id = ? AND key = ?", (self.map_id, key)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __setitem__(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO scratch_string_map VALUES (?, ?, ?) ON CONFLICT(map_id, key) DO UPDATE SET value = excluded.value",
            (self.map_id, key, value),
        )

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        self.conn.execute("DELETE FROM scratch_string_map WHERE map_id = ? AND key = ?", (self.map_id, key))

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, str)
            and self.conn.execute(
                "SELECT 1 FROM scratch_string_map WHERE map_id = ? AND key = ?", (self.map_id, key)
            ).fetchone()
            is not None
        )

    def __iter__(self) -> Iterator[str]:
        cursor = self.conn.execute("SELECT key FROM scratch_string_map WHERE map_id = ? ORDER BY key", (self.map_id,))
        try:
            for (key,) in cursor:
                yield str(key)
        finally:
            cursor.close()

    def __len__(self) -> int:
        return int(
            self.conn.execute("SELECT COUNT(*) FROM scratch_string_map WHERE map_id = ?", (self.map_id,)).fetchone()[0]
        )


def read_chatgpt_mapping_object(
    handle: BinaryIO,
    conn: sqlite3.Connection,
    *,
    require_source_header: bool = True,
    progress: Callable[[], None] | None = None,
) -> tuple[dict[str, object], ChatGPTNodeMapping] | None:
    """Extract mapping nodes without collecting the session in memory.

    Source detection requires its native header witness. A receiver with an
    authenticated declared provider instead delegates header/identity validity
    to the ordinary canonical parser over the extracted original mapping.
    """
    events = iter(ijson.parse(handle))
    mapping = ChatGPTNodeMapping(conn, progress=progress)
    if next(events, None) != ("", "start_map", None):
        return None
    envelope: dict[str, object] = {}
    mapping_count = 0
    for prefix, event, value in events:
        if prefix != "" or event != "map_key":
            continue
        key = require_storable_string(str(value), kind="object key")
        next_event = next(events, None)
        if next_event is None:
            raise ValueError("incomplete ChatGPT object")
        child_prefix, child_event, child_value = next_event
        if key != "mapping" or child_event != "start_map":
            envelope[key] = normalize_ijson_stdlib_numbers(_json_subtree(events, child_event, child_value))
            continue
        mapping_count += 1
        if mapping_count != 1:
            return None
        ordinal = 0
        for node_prefix, node_event, node_value in events:
            if node_prefix == "mapping" and node_event == "end_map":
                break
            if node_prefix != "mapping" or node_event != "map_key":
                raise ValueError("invalid ChatGPT mapping structure")
            node_key = require_storable_string(str(node_value), kind="mapping key")
            node_start = next(events, None)
            if node_start is None:
                raise ValueError("incomplete ChatGPT mapping node")
            node: object
            if node_start[1] == "start_map":
                node, has_children = _read_chatgpt_node(events, mapping, ordinal, node_start[0])
            else:
                node = normalize_ijson_stdlib_numbers(_json_subtree(events, node_start[1], node_start[2]))
                has_children = False
            mapping.put(node_key, node, ordinal)
            if progress is not None:
                progress()
            if has_children:
                mapping.mark_children(node_key, ordinal)
            ordinal += 1  # noqa: SIM113  (nested value events are not node ordinals)
        else:
            raise ValueError("incomplete ChatGPT mapping")
    if (
        mapping_count != 1
        or not mapping
        or require_source_header
        and (
            not isinstance(envelope.get("current_node"), str)
            or not isinstance(envelope.get("create_time"), (int, float))
            or not isinstance(envelope.get("conversation_id"), str)
            and not isinstance(envelope.get("id"), str)
        )
    ):
        return None
    envelope["mapping"] = mapping
    return envelope, mapping


def _read_chatgpt_node(
    events: Iterator[tuple[str, str, object]], mapping: ChatGPTNodeMapping, ordinal: int, node_prefix: str
) -> tuple[dict[str, object], bool]:
    """Spill the direct children array while decoding the rest of one node."""
    node: dict[str, object] = {}
    has_children = False
    for prefix, event, value in events:
        if prefix == node_prefix and event == "end_map":
            return node, has_children
        if prefix != node_prefix or event != "map_key":
            raise ValueError("invalid ChatGPT mapping node")
        key = require_storable_string(str(value), kind="object key")
        start = next(events, None)
        if start is None:
            raise ValueError("incomplete ChatGPT mapping node")
        if key != "children" or start[1] != "start_array":
            node[key] = normalize_ijson_stdlib_numbers(_json_subtree(events, start[1], start[2]))
            if key == "children":
                mapping.conn.execute("DELETE FROM chatgpt_child WHERE node_ordinal = ?", (ordinal,))
                has_children = False
            continue
        # A duplicate JSON member follows ordinary last-value-wins behavior.
        # Its previous items are scratch only and can be removed immediately.
        mapping.conn.execute("DELETE FROM chatgpt_child WHERE node_ordinal = ?", (ordinal,))
        node[key] = []
        has_children = True
        child_ordinal = 0
        for child_prefix, child_event, child_value in events:
            if child_prefix == f"{node_prefix}.children" and child_event == "end_array":
                break
            if child_prefix != f"{node_prefix}.children.item":
                raise ValueError("invalid ChatGPT children array")
            child = normalize_ijson_stdlib_numbers(_json_subtree(events, child_event, child_value))
            mapping.put_child(ordinal, child_ordinal, child)
            child_ordinal += 1  # noqa: SIM113  (the array end event is not a child)
        else:
            raise ValueError("incomplete ChatGPT children array")
    raise ValueError("incomplete ChatGPT mapping node")


__all__ = [
    "GeminiToolOutputIndex",
    "SqliteMessageSink",
    "SqliteMessageStore",
    "SqliteAttachmentSink",
    "SqliteSessionEventSink",
    "ChatGPTNodeMapping",
    "ScratchSessionSpill",
    "normalize_active_branch",
    "read_chatgpt_mapping_object",
]
