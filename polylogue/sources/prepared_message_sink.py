"""Disk-backed parsed messages for worker preparation and sealed publication."""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Container, Iterable, Iterator, Mapping, MutableSequence, MutableSet, Set
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, overload
from urllib.parse import quote

import ijson

from polylogue.core.hashing import hash_text
from polylogue.core.json import JSONDocument, json_document
from polylogue.sources import value_bounds
from polylogue.sources.decoder_json import _json_subtree, normalize_ijson_stdlib_numbers
from polylogue.sources.live.tool_result_sidecars import (
    SidecarDebt,
    SidecarMatch,
)
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSessionEvent
from polylogue.sources.sidecar_evidence import RetainedSidecarScope
from polylogue.sources.value_bounds import require_storable_string

_ACTIVE_PARENT_LOOKUP_SQL = (
    "SELECT parent_id FROM prepared_message INDEXED BY prepared_message_provider "
    "WHERE session_ordinal = ? AND provider_id = ? ORDER BY message_ordinal DESC LIMIT 1"
)


def _read_uri(path: Path) -> str:
    return f"file:{quote(str(path))}?mode=ro"


def _message_json(value: ParsedMessage) -> str:
    payload = value.model_dump(mode="json")
    payload["parent_message_position"] = value.parent_message_position
    payload["owner_coordinate"] = asdict(value.owner_coordinate) if value.owner_coordinate is not None else None
    # Each serialized record is one SQLite cell: individually storable
    # values can still combine into an unstorable row.
    return require_storable_string(json.dumps(payload, ensure_ascii=False), kind="serialized message")


def _event_json(value: ParsedSessionEvent) -> str:
    payload = value.model_dump(mode="json")
    payload["boundary_message_position"] = value.boundary_message_position
    return require_storable_string(json.dumps(payload, ensure_ascii=False), kind="serialized event")


def _attachment_json(value: ParsedAttachment) -> str:
    payload = value.model_dump(mode="json")
    payload["message_position"] = value.message_position
    payload["message_variant_index"] = value.message_variant_index
    payload["owner_coordinate"] = asdict(value.owner_coordinate) if value.owner_coordinate is not None else None
    payload["precomputed_blob"] = value.precomputed_blob
    payload["_prepared_inline_bytes"] = (
        base64.b64encode(value.inline_bytes).decode("ascii") if value.inline_bytes is not None else None
    )
    return require_storable_string(json.dumps(payload, ensure_ascii=False), kind="serialized attachment")


def _decode_attachment(encoded: str, path: Path, session_ordinal: int, attachment_ordinal: int) -> ParsedAttachment:
    payload = json.loads(encoded)
    inline = payload.pop("_prepared_inline_bytes", None)
    if inline is not None:
        payload["inline_bytes"] = base64.b64decode(inline, validate=True)
    return ParsedAttachment.model_validate(payload).model_copy(
        update={"prepared_carrier_key": (str(path), session_ordinal, attachment_ordinal)}
    )


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
                # The only limit on a sidecar is what one SQLite cell holds.
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
                full_text=full_text,
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


class SqliteMessageSink(MutableSequence[ParsedMessage]):
    """One session's messages, in stable ordinal order without a resident list."""

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
            with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
                row = conn.execute(
                    "SELECT message_json FROM prepared_message WHERE session_ordinal = ? AND message_ordinal = ?",
                    (self.session_ordinal, ordinal),
                ).fetchone()
        if row is None:
            raise ValueError("prepared message row disappeared")
        return ParsedMessage.model_validate_json(row[0])

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
        ordinal = self._ordinal(index)
        self._writer.execute(
            "UPDATE prepared_message SET message_json = ?, provider_id = ?, parent_id = ?, active_leaf = ? "
            "WHERE session_ordinal = ? AND message_ordinal = ?",
            (
                _message_json(value),
                value.provider_message_id,
                value.parent_message_provider_id,
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
        self._writer.execute(
            "INSERT INTO prepared_message VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.session_ordinal,
                self._count,
                _message_json(value),
                value.provider_message_id,
                value.parent_message_provider_id,
                int(bool(value.is_active_leaf)),
            ),
        )
        self._count += 1

    def __iter__(self) -> Iterator[ParsedMessage]:
        yield from self.iter_from(0)

    def provider_message_ids(self, *, include_none: bool) -> Set[str | None]:
        """A disk-backed set for membership comparison of large sessions."""
        return SqliteProviderMessageIds(self, include_none=include_none)

    def iter_from(self, start: int) -> Iterator[ParsedMessage]:
        """Stream a suffix without decoding or scanning its inherited prefix."""
        if start < 0 or start > self._count:
            raise IndexError(start)
        if self._writer is not None:
            cursor = self._writer.execute(
                "SELECT message_json FROM prepared_message WHERE session_ordinal = ? "
                "AND message_ordinal >= ? ORDER BY message_ordinal",
                (self.session_ordinal, start),
            )
            for row in cursor:
                yield ParsedMessage.model_validate_json(row[0])
            return
        with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
            cursor = conn.execute(
                "SELECT message_json FROM prepared_message WHERE session_ordinal = ? "
                "AND message_ordinal >= ? ORDER BY message_ordinal",
                (self.session_ordinal, start),
            )
            for row in cursor:
                yield ParsedMessage.model_validate_json(row[0])

    def normalize_active_path(self) -> SqliteMessageSink:
        """Apply the writer's leaf/path normalization without a message list."""
        if self._writer is None:
            # Publication artifacts are immutable. The worker has already
            # normalized them before sealing.
            return self
        leaf_count = self._writer.execute(
            "SELECT COUNT(*) FROM prepared_message WHERE session_ordinal = ? AND active_leaf = 1",
            (self.session_ordinal,),
        ).fetchone()[0]
        if not self._count:
            return self
        if leaf_count != 1:
            for ordinal, active in self._writer.execute(
                "SELECT message_ordinal, active_leaf FROM prepared_message WHERE session_ordinal = ?",
                (self.session_ordinal,),
            ):
                expected = ordinal == self._count - 1
                if bool(active) != expected:
                    message = self[ordinal]
                    self[ordinal] = message.model_copy(update={"is_active_leaf": expected})
            return self
        leaf = self._writer.execute(
            "SELECT provider_id FROM prepared_message WHERE session_ordinal = ? AND active_leaf = 1",
            (self.session_ordinal,),
        ).fetchone()[0]
        if not leaf:
            return self
        self._writer.execute("DROP TABLE IF EXISTS temp.prepared_active_path")
        self._writer.execute("CREATE TEMP TABLE prepared_active_path (provider_id TEXT PRIMARY KEY)")
        cursor: str | None = leaf
        while cursor:
            result = self._writer.execute("INSERT OR IGNORE INTO prepared_active_path VALUES (?)", (cursor,))
            if result.rowcount == 0:
                break
            parent = self._writer.execute(
                _ACTIVE_PARENT_LOOKUP_SQL,
                (self.session_ordinal, cursor),
            ).fetchone()
            cursor = parent[0] if parent is not None else None
        for (ordinal,) in self._writer.execute(
            "SELECT message_ordinal FROM prepared_message WHERE session_ordinal = ? "
            "AND provider_id IN (SELECT provider_id FROM prepared_active_path)",
            (self.session_ordinal,),
        ):
            message = self[ordinal]
            self[ordinal] = message.model_copy(update={"is_active_path": True})
        self._writer.execute("DROP TABLE temp.prepared_active_path")
        return self

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


class SqliteProviderMessageIds(Set[str | None]):
    """Native IDs backed by the prepared-message provider index."""

    def __init__(self, messages: SqliteMessageSink, *, include_none: bool) -> None:
        self.messages = messages
        self.include_none = include_none

    def _connection(self) -> sqlite3.Connection:
        return self.messages._writer or sqlite3.connect(_read_uri(self.messages.path), uri=True)

    def _where(self, alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return f"{prefix}session_ordinal = ?" + ("" if self.include_none else f" AND {prefix}provider_id IS NOT NULL")

    def __contains__(self, value: object) -> bool:
        if value is None and not self.include_none:
            return False
        if value is not None and not isinstance(value, str):
            return False
        conn = self._connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM prepared_message WHERE session_ordinal = ? AND provider_id IS ? LIMIT 1",
                (self.messages.session_ordinal, value),
            ).fetchone()
            return row is not None
        finally:
            if conn is not self.messages._writer:
                conn.close()

    def __iter__(self) -> Iterator[str | None]:
        conn = self._connection()
        try:
            for (provider_id,) in conn.execute(
                f"SELECT DISTINCT provider_id FROM prepared_message WHERE {self._where()} ORDER BY provider_id",
                (self.messages.session_ordinal,),
            ):
                yield provider_id
        finally:
            if conn is not self.messages._writer:
                conn.close()

    def __len__(self) -> int:
        conn = self._connection()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT provider_id FROM prepared_message WHERE {self._where()})",
                (self.messages.session_ordinal,),
            ).fetchone()
            return int(row[0])
        finally:
            if conn is not self.messages._writer:
                conn.close()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        if isinstance(other, SqliteProviderMessageIds):
            conn = sqlite3.connect(_read_uri(self.messages.path), uri=True)
            try:
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
            finally:
                conn.close()
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
            with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
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
        self._writer.execute(
            "UPDATE prepared_attachment SET attachment_json = ? WHERE session_ordinal = ? AND attachment_ordinal = ?",
            (_attachment_json(value), self.session_ordinal, self._ordinal(index)),
        )

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("prepared attachments cannot be deleted")

    def insert(self, index: int, value: ParsedAttachment) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared attachments are immutable")
        if index != self._count:
            raise TypeError("prepared attachments can only be appended")
        self._writer.execute(
            "INSERT INTO prepared_attachment VALUES (?, ?, ?)",
            (self.session_ordinal, self._count, _attachment_json(value)),
        )
        self._count += 1

    def __iter__(self) -> Iterator[ParsedAttachment]:
        if self._writer is not None:
            rows = self._writer.execute(
                "SELECT attachment_ordinal, attachment_json FROM prepared_attachment WHERE session_ordinal = ? ORDER BY attachment_ordinal",
                (self.session_ordinal,),
            )
            for ordinal, encoded in rows:
                yield _decode_attachment(encoded, self.path, self.session_ordinal, ordinal)
            return
        with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
            rows = conn.execute(
                "SELECT attachment_ordinal, attachment_json FROM prepared_attachment WHERE session_ordinal = ? ORDER BY attachment_ordinal",
                (self.session_ordinal,),
            )
            for ordinal, encoded in rows:
                yield _decode_attachment(encoded, self.path, self.session_ordinal, ordinal)


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
            with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as reader:
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
        return ParsedSessionEvent.model_validate_json(row[0])

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
        self._writer.execute(
            "INSERT INTO prepared_event (session_ordinal, event_ordinal, timestamp, event_type, event_json) VALUES (?, ?, ?, ?, ?)",
            (self.session_ordinal, index, value.timestamp, value.event_type, _event_json(value)),
        )
        self._count += 1

    def __iter__(self) -> Iterator[ParsedSessionEvent]:
        yield from self._iter_query("ORDER BY event_ordinal")

    def _iter_query(self, order_sql: str, parameters: tuple[object, ...] = ()) -> Iterator[ParsedSessionEvent]:
        sql = "SELECT event_json FROM prepared_event WHERE session_ordinal = ? " + order_sql
        if self._writer is not None:
            cursor = self._writer.execute(sql, (self.session_ordinal, *parameters))
            for row in cursor:
                yield ParsedSessionEvent.model_validate_json(row[0])
            return
        with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
            for row in conn.execute(sql, (self.session_ordinal, *parameters)):
                yield ParsedSessionEvent.model_validate_json(row[0])

    def iter_ordered(self, type_order_tier: Mapping[str, int]) -> Iterator[ParsedSessionEvent]:
        clauses = " ".join("WHEN ? THEN ?" for _ in type_order_tier)
        order = "ORDER BY COALESCE(timestamp, ''), CASE event_type " + clauses + " ELSE 0 END, event_ordinal"
        parameters: tuple[object, ...] = tuple(item for pair in type_order_tier.items() for item in pair)
        yield from self._iter_query(order, parameters)

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


class SqliteMessageStore:
    """Own the unsealed scratch transaction until its producer has finished."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode = DELETE")
        # The schema is created inside the store's one transaction: as separate
        # autocommit statements each CREATE paid its own journal and fsync, per
        # prepared artifact, before any row was spooled.
        self.conn.execute("BEGIN IMMEDIATE")
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
        self._next_session_ordinal = 0
        self._next_event_ordinal = 0
        self._next_attachment_ordinal = 0

    def new_sink(self) -> SqliteMessageSink:
        sink = SqliteMessageSink(self.path, self._next_session_ordinal, writer=self.conn)
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
        self.conn.close()


class ChatGPTNodeMapping(Mapping[str, object]):
    """Keep node bytes, insertion order, and duplicate-key resolution in scratch."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
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
        encoded = require_storable_string(json.dumps(node, ensure_ascii=False), kind="serialized mapping node")
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
                json.dumps(child, ensure_ascii=False),
                child if isinstance(child, str) else None,
            ),
        )

    def mark_children(self, key: str, ordinal: int) -> None:
        self.conn.execute("UPDATE chatgpt_node SET child_ordinal = ? WHERE node_key = ?", (ordinal, key))

    def shallow_node(self, key: str) -> object:
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
            "position INTEGER NOT NULL, provider_id TEXT NOT NULL, message_json TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX chatgpt_entry_provider ON chatgpt_entry(provider_id)")
        conn.execute(f"CREATE INDEX chatgpt_entry_order ON chatgpt_entry({self._ORDER})")

    def add(self, timestamp: float | None, idx: int, node_id: str, message: ParsedMessage) -> None:
        self.conn.execute(
            "INSERT INTO chatgpt_entry VALUES (?, ?, ?, ?, ?, ?)",
            (node_id, idx, timestamp, message.position, message.provider_message_id, _message_json(message)),
        )

    def ordered(self) -> Iterator[ParsedMessage]:
        # A separate cursor: the consumer writes other scratch tables while
        # this one is being stepped.
        cursor = self.conn.execute(f"SELECT message_json FROM chatgpt_entry ORDER BY {self._ORDER}")
        try:
            for (encoded,) in cursor:
                yield ParsedMessage.model_validate_json(encoded)
        finally:
            cursor.close()

    def provider_for_node(self, node_id: str) -> str | None:
        row = self.conn.execute("SELECT provider_id FROM chatgpt_entry WHERE node_key = ?", (node_id,)).fetchone()
        return str(row[0]) if row is not None else None

    def position_for_node(self, node_id: str) -> int | None:
        row = self.conn.execute("SELECT position FROM chatgpt_entry WHERE node_key = ?", (node_id,)).fetchone()
        return int(row[0]) if row is not None else None

    def emitted_provider_ids(self) -> Container[str]:
        return _ScratchProviderIds(self.conn)

    def last_emitted_among(self, provider_ids: frozenset[str]) -> str | None:
        best: tuple[int, float, int, str] | None = None
        ordered_ids = sorted(provider_ids)
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
        return best[3] if best is not None else None


class _ScratchProviderIds(Container[str]):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __contains__(self, value: object) -> bool:
        return (
            isinstance(value, str)
            and self.conn.execute("SELECT 1 FROM chatgpt_entry WHERE provider_id = ? LIMIT 1", (value,)).fetchone()
            is not None
        )


class ScratchSessionSpill:
    """Scratch-backed collections for one session parsed with ``spill=``."""

    def __init__(self, store: SqliteMessageStore) -> None:
        self.store = store

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
        for (value,) in self.conn.execute(
            "SELECT value FROM scratch_string_set WHERE set_id = ? ORDER BY value", (self.set_id,)
        ).fetchall():
            yield str(value)

    def __len__(self) -> int:
        return int(
            self.conn.execute("SELECT COUNT(*) FROM scratch_string_set WHERE set_id = ?", (self.set_id,)).fetchone()[0]
        )

    def add(self, value: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO scratch_string_set VALUES (?, ?)", (self.set_id, value))

    def discard(self, value: str) -> None:
        self.conn.execute("DELETE FROM scratch_string_set WHERE set_id = ? AND value = ?", (self.set_id, value))


def read_chatgpt_mapping_object(
    handle: BinaryIO, conn: sqlite3.Connection
) -> tuple[dict[str, object], ChatGPTNodeMapping] | None:
    """Consume a complete native object while writing each mapping node immediately."""
    events = iter(ijson.parse(handle))
    mapping = ChatGPTNodeMapping(conn)
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
            if has_children:
                mapping.mark_children(node_key, ordinal)
            ordinal += 1  # noqa: SIM113  (nested value events are not node ordinals)
        else:
            raise ValueError("incomplete ChatGPT mapping")
    if (
        mapping_count != 1
        or not mapping
        or not isinstance(envelope.get("current_node"), str)
        or not isinstance(envelope.get("create_time"), (int, float))
        or not isinstance(envelope.get("conversation_id"), str)
        and not isinstance(envelope.get("id"), str)
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
    "read_chatgpt_mapping_object",
]
