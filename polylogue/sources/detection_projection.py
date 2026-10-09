"""Complete event projections declared by the provider detector owners.

A projection describes exactly the fields and container folds its predicate
uses. Unselected containers are consumed and validated, without building their
values. Array witnesses preserve the declared first/all/any predicate; they
are not a prefix sample. Scalar decoding retains one value at a time.
"""

from __future__ import annotations

import codecs
import io
import sqlite3
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal
from json import JSONDecodeError
from typing import IO, Literal, Protocol, cast

import ijson

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.content_identity import JSON_TEXT_ENCODINGS
from polylogue.core.json import JSONDocument, json_document_or_none
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


@dataclass(frozen=True, slots=True)
class DetectorProjection:
    fields: Mapping[str, DetectorProjection | None] | None = None
    item: DetectorProjection | None = None
    array_fold: Literal["type", "first", "any", "all"] = "type"
    array_predicate: Callable[[object], bool] | None = None
    mapping_predicate: Callable[[object], bool] | None = None
    mapping_witness: object = None
    mapping_key_predicate: Callable[[str], bool] | None = None
    preserve_mapping_size: bool = False
    capture_metadata_values: bool = False


_UNREAD_RECORD = object()
_RECORD_IS_SELF = object()


class DetectionReadMapping(dict[str, object]):
    """One detector projection with a lazily validated JSON record view.

    Detector predicates only read projections. Repeated predicates and a
    dynamic resolver therefore share this conversion within the record's
    lifetime, without trusting arbitrary projection witnesses as JSON.
    """

    def __init__(self, fields: dict[str, object]) -> None:
        super().__init__(fields)
        self._json_record: object = _UNREAD_RECORD

    def json_record(self) -> JSONDocument | None:
        if self._json_record is _UNREAD_RECORD:
            record = json_document_or_none(self)
            # Avoid a self-reference cycle on every accepted record.
            self._json_record = _RECORD_IS_SELF if record is self else record
        if self._json_record is _RECORD_IS_SELF:
            return cast(JSONDocument, self)
        return cast(JSONDocument | None, self._json_record)


class _ProjectedMapping(DetectionReadMapping):
    """Selected predicate fields carrying the original unique-key count."""

    def __init__(
        self, fields: dict[str, object], original_size: int, metadata_values_scalarish: bool | None = None
    ) -> None:
        super().__init__(fields)
        self.original_size = original_size
        self.metadata_values_scalarish = metadata_values_scalarish

    def __len__(self) -> int:
        return self.original_size


def _detection_read_view(value: object) -> object:
    """Attach conversion reuse only to the root records a detector will read."""
    if isinstance(value, dict) and not isinstance(value, DetectionReadMapping):
        return DetectionReadMapping(value)
    if isinstance(value, list):
        return [
            DetectionReadMapping(item)
            if isinstance(item, dict) and not isinstance(item, DetectionReadMapping)
            else item
            for item in value
        ]
    return value


class _ByteReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _ObservedLine:
    def __init__(self, source: _ByteReader, check_stop: Callable[[], None] | None = None) -> None:
        self.source = source
        self.check_stop = check_stop
        self.nonblank = False
        self.callback_failure: BaseException | None = None

    def read(self, size: int = -1) -> bytes:
        check_compute_cancelled()
        if self.check_stop is not None:
            try:
                self.check_stop()
            except BaseException as exc:
                self.callback_failure = exc
                raise
        data = self.source.read(size)
        self.nonblank |= bool(data.strip(b" \t\r\n"))
        return data

    def drain(self) -> None:
        while self.read(1024 * 1024):
            pass


class _DetectionText(io.RawIOBase):
    def __init__(self, handle: _ByteReader, encoding: str, check_stop: Callable[[], None] | None = None) -> None:
        self.handle = handle
        self.check_stop = check_stop
        self.callback_failure: BaseException | None = None
        self.decoder = codecs.getincrementaldecoder(encoding)(errors="surrogatepass")
        self.pending = bytearray()
        self.ended = False
        self.started = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        check_compute_cancelled()
        if self.check_stop is not None:
            try:
                self.check_stop()
            except BaseException as exc:
                self.callback_failure = exc
                raise
        view = memoryview(buffer)  # type: ignore[arg-type]
        while not self.pending and not self.ended:
            check_compute_cancelled()
            chunk = self.handle.read(1024 * 1024)
            self.ended = not chunk
            text = self.decoder.decode(chunk, final=self.ended)
            if not self.started:
                text = text.lstrip("\ufeff")
                self.started = bool(text)
            # JSON accepts escaped lone surrogates, while the event decoder's
            # UTF-8 reader rejects their directly encoded provider spelling.
            # Escaping only those code units preserves both lone units and
            # adjacent CESU-8 pairs without changing ordinary string content.
            self.pending.extend(text.encode("utf-8", "backslashreplace"))
        count = min(len(view), len(self.pending))
        view[:count] = self.pending[:count]
        del self.pending[:count]
        return count


def _skip(events: Iterator[tuple[str, object]], event: str) -> object:
    if event not in {"start_map", "start_array"}:
        return None
    depth = 1
    while depth:
        following, _value = next(events)
        if following in {"start_map", "start_array"}:
            depth += 1
        elif following in {"end_map", "end_array"}:
            depth -= 1
    return {} if event == "start_map" else []


def _project(
    events: Iterator[tuple[str, object]],
    event: str,
    value: object,
    rule: DetectorProjection | None,
    stack: ExitStack,
) -> object:
    return _detection_read_view(
        _project_value(
            events,
            event,
            value,
            rule,
            stack,
            scalarish_depth=-1 if rule is not None and rule.capture_metadata_values else None,
        )[0]
    )


def _consume_scalarish(events: Iterator[tuple[str, object]], event: str, depth: int) -> bool:
    """Fold the taxonomy's scalarish predicate without retaining unknown values."""
    if event not in {"start_map", "start_array"}:
        return True
    if depth >= 2:
        _skip(events, event)
        return False
    if event == "start_array":
        count = 0
        accepted = True
        while True:
            following, _value = next(events)
            if following == "end_array":
                return count <= 32 and accepted
            count += 1
            accepted = _consume_scalarish(events, following, depth + 1) and accepted
    with scratch_connection_context(prefix="polylogue-taxonomy-values-", filename="keys.db") as keys:
        keys.execute("PRAGMA journal_mode=DELETE")
        keys.execute("PRAGMA temp_store=FILE")
        keys.execute("BEGIN")
        keys.execute("CREATE TABLE keys(name BLOB PRIMARY KEY, accepted INTEGER NOT NULL) WITHOUT ROWID")
        while True:
            following, key = next(events)
            if following == "end_map":
                count, refused = keys.execute("SELECT COUNT(*), COALESCE(SUM(accepted=0),0) FROM keys").fetchone()
                return count <= 8 and not refused
            if following != "map_key" or not isinstance(key, str):
                raise ijson.JSONError("invalid taxonomy object event")
            following, _value = next(events)
            accepted = _consume_scalarish(events, following, depth + 1)
            keys.execute(
                "INSERT INTO keys VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET accepted=excluded.accepted",
                (key.encode("utf-8", "surrogatepass"), int(accepted)),
            )


def _project_value(
    events: Iterator[tuple[str, object]],
    event: str,
    value: object,
    rule: DetectorProjection | None,
    stack: ExitStack,
    *,
    scalarish_depth: int | None,
) -> tuple[object, bool]:
    if rule is None:
        if scalarish_depth is not None:
            shape: dict[str, object] | list[object] | None = (
                {} if event == "start_map" else [] if event == "start_array" else None
            )
            return shape, _consume_scalarish(events, event, scalarish_depth)
        return _skip(events, event), True
    if event == "start_map":
        with ExitStack() as mapping_stack:
            fields: dict[str, object] = {}
            matching_key = False
            database: sqlite3.Connection | None = None
            if rule.mapping_predicate is not None or rule.preserve_mapping_size or scalarish_depth is not None:
                database = mapping_stack.enter_context(
                    scratch_connection_context(prefix="polylogue-detector-", filename="keys.db")
                )
                # Duplicate keys update this complete key set. Keep their
                # rollback journal on the same private disk as the rows.
                database.execute("PRAGMA journal_mode=DELETE")
                database.execute("PRAGMA temp_store=FILE")
                database.execute("BEGIN")
                database.execute(
                    "CREATE TABLE keys (name BLOB PRIMARY KEY, accepted INTEGER NOT NULL, scalarish INTEGER) WITHOUT ROWID"
                )
            while True:
                event, key = next(events)
                if event == "end_map":
                    break
                if event != "map_key" or not isinstance(key, str):
                    raise ValueError("invalid detector object event")
                event, value = next(events)
                child_depth = None if scalarish_depth is None or scalarish_depth >= 2 else scalarish_depth + 1
                child_rule = rule.item if rule.mapping_predicate is not None else (rule.fields or {}).get(key)
                item, scalarish = _project_value(events, event, value, child_rule, stack, scalarish_depth=child_depth)
                if rule.mapping_predicate is not None:
                    assert database is not None
                    assert rule.mapping_predicate is not None
                    database.execute(
                        "INSERT INTO keys(name,accepted) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET accepted=excluded.accepted",
                        (key.encode("utf-8", "surrogatepass"), int(rule.mapping_predicate(item))),
                    )
                elif rule.mapping_key_predicate is not None:
                    matching_key |= rule.mapping_key_predicate(key)
                elif rule.fields is not None and key in rule.fields:
                    fields[key] = item
                if (rule.preserve_mapping_size or scalarish_depth is not None) and rule.mapping_predicate is None:
                    assert database is not None
                    database.execute(
                        "INSERT INTO keys(name,accepted) VALUES (?, 1) ON CONFLICT(name) DO NOTHING",
                        (key.encode("utf-8", "surrogatepass"),),
                    )
                if scalarish_depth is not None:
                    assert database is not None
                    database.execute(
                        "UPDATE keys SET scalarish=? WHERE name=?",
                        (int(scalarish), key.encode("utf-8", "surrogatepass")),
                    )
            scalarish = True
            metadata_values = None
            if scalarish_depth is not None:
                assert database is not None
                count, refused = database.execute("SELECT COUNT(*), COALESCE(SUM(scalarish=0),0) FROM keys").fetchone()
                metadata_values = not refused
                scalarish = scalarish_depth < 2 and count <= 8 and metadata_values
            if database is not None and rule.mapping_predicate is not None:
                count, refused = database.execute("SELECT COUNT(*), SUM(accepted=0) FROM keys").fetchone()
                return ({} if not count else {"node": None if refused else rule.mapping_witness}), scalarish
            if rule.mapping_key_predicate is not None:
                return (rule.mapping_witness if matching_key else {}), scalarish
            if rule.preserve_mapping_size:
                assert database is not None
                count = int(database.execute("SELECT COUNT(*) FROM keys").fetchone()[0])
                return _ProjectedMapping(fields, count, metadata_values), scalarish
            return fields, scalarish
    if event == "start_array":
        first: object = None
        witness: object = None
        count = 0
        matched = False
        scalarish_items = True
        while True:
            event, value = next(events)
            if event == "end_array":
                break
            item, child_scalarish = _project_value(
                events,
                event,
                value,
                rule.item,
                stack,
                scalarish_depth=None if scalarish_depth is None or scalarish_depth >= 2 else scalarish_depth + 1,
            )
            scalarish_items &= child_scalarish
            if count == 0:
                first = item
            count += 1
            if rule.array_predicate is not None:
                accepts = rule.array_predicate(item)
                if (rule.array_fold == "any" and accepts) or (rule.array_fold == "all" and not accepts):
                    witness = item
                    matched = True
        scalarish = scalarish_depth is None or (scalarish_depth < 2 and count <= 32 and scalarish_items)
        if not count or rule.array_fold == "type":
            return [], scalarish
        if rule.array_fold == "first":
            return ([first] if count == 1 else [first, None]), scalarish
        return [witness if matched else first], scalarish
    if isinstance(value, Decimal):
        return float(value), True
    return value, True


def _document_projection(
    handle: _ByteReader,
    rule: DetectorProjection,
    stream_predicate: Callable[[object], bool] | None,
) -> object:
    from ijson.backends import python as exact_backend

    from polylogue.core.json_envelope import LexemeAlignedReader, _PrefixStringReader

    with ExitStack() as stack:
        events = iter(exact_backend.basic_parse(LexemeAlignedReader(_PrefixStringReader(handle, scalar_values=True))))
        event, value = next(events)
        root_rule = (
            DetectorProjection(
                item=rule, array_fold="first" if stream_predicate is None else "any", array_predicate=stream_predicate
            )
            if event == "start_array"
            else rule
        )
        result = _project(events, event, value, root_rule, stack)
        # Consume EOF even when the predicate already has a positive witness.
        # Extra values and malformed suffixes cannot prove healthy admission.
        for _event in events:
            raise ijson.JSONError("trailing JSON detection value")
        return result


def project_detection_input(
    handle: IO[bytes],
    rule: DetectorProjection,
    *,
    stream_predicate: Callable[[object], bool] | None = None,
    check_stop: Callable[[], None] | None = None,
) -> tuple[Literal["record", "sequence"], object]:
    """Project complete JSON input, keeping duplicate-key last-value semantics.

    A complete document is tested before the physical JSONL alternative, as
    the object detector does. Each JSONL line must contain exactly one value;
    adjacent values or a malformed late line cannot establish provider proof.
    """
    from polylogue.core.json_envelope import _LineSource

    start = handle.tell()
    for encoding in JSON_TEXT_ENCODINGS:
        handle.seek(start)
        source = _DetectionText(handle, encoding, check_stop)
        reader = io.BufferedReader(source)
        syntax_error: Exception | None = None
        try:
            try:
                payload = _document_projection(reader, rule, stream_predicate)
                return ("sequence" if isinstance(payload, list) else "record"), payload
            except (ijson.JSONError, JSONDecodeError, StopIteration) as exc:
                if source.callback_failure is not None:
                    raise source.callback_failure from None
                syntax_error = exc
                while reader.read(1024 * 1024):
                    pass
        except UnicodeError:
            if source.callback_failure is not None:
                raise source.callback_failure from None
            continue
        finally:
            reader.close()
        # This encoding decoded the whole source. Syntax does not authorize
        # switching to another encoding and finding unrelated text in it.
        handle.seek(start)
        source = _DetectionText(handle, encoding, check_stop)
        reader = io.BufferedReader(source)
        try:
            lines = _LineSource(reader)
            count = 0
            first: object = None
            witness: object = None
            matched = False
            while (line := lines.next_line()) is not None:
                observed = _ObservedLine(line)
                try:
                    item = _document_projection(observed, rule, None)
                except (StopIteration, ijson.JSONError, JSONDecodeError):
                    if source.callback_failure is not None:
                        raise source.callback_failure from None
                    observed.drain()
                    if not observed.nonblank:
                        continue
                    raise
                observed.drain()
                if count == 0:
                    first = item
                count += 1
                if stream_predicate is not None and stream_predicate(item):
                    witness, matched = item, True
            if not count:
                raise ijson.JSONError("empty JSON detection input") from syntax_error
            if stream_predicate is not None:
                return "sequence", [witness if matched else first]
            return "sequence", [first] if count == 1 else [first, None]
        finally:
            reader.close()
    raise ijson.JSONError("unsupported JSON text encoding")


def project_detection_value(value: object, rule: DetectorProjection) -> object:
    """Project one decoded JSON value exactly as the event projection would.

    A decoded record already holds its complete value, so only the declared
    fields and folds are visited; unselected subtrees are never walked. A root
    array takes the same first-item wrapper a physical JSONL line receives.
    """
    root_rule = DetectorProjection(item=rule, array_fold="first") if isinstance(value, list) else rule
    return project_detection_root(value, root_rule)


def project_detection_root(value: object, root_rule: DetectorProjection) -> object:
    """Project a decoded value under an already chosen root rule, as :func:`_project` does."""
    return _detection_read_view(
        _project_object(value, root_rule, scalarish_depth=-1 if root_rule.capture_metadata_values else None)[0]
    )


def _object_scalarish(value: object, depth: int) -> bool:
    """:func:`_consume_scalarish` over a decoded value."""
    if isinstance(value, list):
        return depth < 2 and len(value) <= 32 and all(_object_scalarish(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return depth < 2 and len(value) <= 8 and all(_object_scalarish(item, depth + 1) for item in value.values())
    return True


def _project_object(
    value: object, rule: DetectorProjection | None, *, scalarish_depth: int | None
) -> tuple[object, bool]:
    """:func:`_project_value` over a decoded value; keys are already unique (last value wins)."""
    if rule is None:
        shape: dict[str, object] | list[object] | None = (
            {} if isinstance(value, dict) else [] if isinstance(value, list) else None
        )
        return shape, True if scalarish_depth is None else _object_scalarish(value, scalarish_depth)
    child_depth = None if scalarish_depth is None or scalarish_depth >= 2 else scalarish_depth + 1
    if isinstance(value, dict):
        fields: dict[str, object] = {}
        matching_key = False
        refused = False
        values_scalarish = True
        # Decoded values are already structurally validated. Without a
        # whole-mapping predicate or metadata fold, ignored values contribute
        # nothing; visit only the declared fields (including explicit nulls).
        selected_fields_only = (
            scalarish_depth is None and rule.mapping_predicate is None and rule.mapping_key_predicate is None
        )
        entries = (
            ((key, value[key]) for key in rule.fields or {} if key in value) if selected_fields_only else value.items()
        )
        for key, child in entries:
            child_rule = rule.item if rule.mapping_predicate is not None else (rule.fields or {}).get(key)
            item, scalarish = _project_object(child, child_rule, scalarish_depth=child_depth)
            values_scalarish &= scalarish
            if rule.mapping_predicate is not None:
                refused |= not rule.mapping_predicate(item)
            elif rule.mapping_key_predicate is not None:
                matching_key |= rule.mapping_key_predicate(key)
            elif rule.fields is not None and key in rule.fields:
                fields[key] = item
        count = len(value)
        mapping_scalarish = True
        metadata_values = None
        if scalarish_depth is not None:
            metadata_values = values_scalarish
            mapping_scalarish = scalarish_depth < 2 and count <= 8 and metadata_values
        if rule.mapping_predicate is not None:
            return ({} if not count else {"node": None if refused else rule.mapping_witness}), mapping_scalarish
        if rule.mapping_key_predicate is not None:
            return (rule.mapping_witness if matching_key else {}), mapping_scalarish
        if rule.preserve_mapping_size:
            return _ProjectedMapping(fields, count, metadata_values), mapping_scalarish
        return fields, mapping_scalarish
    if isinstance(value, list):
        first: object = None
        witness: object = None
        matched = False
        items_scalarish = True
        for index, child in enumerate(value):
            item, scalarish = _project_object(child, rule.item, scalarish_depth=child_depth)
            items_scalarish &= scalarish
            if index == 0:
                first = item
            if rule.array_predicate is not None:
                accepts = rule.array_predicate(item)
                if (rule.array_fold == "any" and accepts) or (rule.array_fold == "all" and not accepts):
                    witness = item
                    matched = True
        count = len(value)
        array_scalarish = scalarish_depth is None or (scalarish_depth < 2 and count <= 32 and items_scalarish)
        if not count or rule.array_fold == "type":
            return [], array_scalarish
        if rule.array_fold == "first":
            return ([first] if count == 1 else [first, None]), array_scalarish
        return [witness if matched else first], array_scalarish
    if isinstance(value, Decimal):
        return float(value), True
    return value, True


class _CheckedInput(io.RawIOBase):
    """A borrowed byte input whose every read first honours cancellation."""

    def __init__(self, handle: IO[bytes], check_stop: Callable[[], None] | None) -> None:
        self.handle = handle
        self.check_stop = check_stop

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        check_compute_cancelled()
        if self.check_stop is not None:
            self.check_stop()
        view = memoryview(buffer)  # type: ignore[arg-type]
        data = self.handle.read(len(view))
        view[: len(data)] = data
        return len(data)

    def close(self) -> None:
        # Closing this adapter never closes the caller's input.
        super().close()


def iter_decoded_jsonl_records(
    handle: IO[bytes],
    *,
    check_stop: Callable[[], None] | None = None,
    on_decode_failure: Callable[[Exception], None] | None = None,
) -> Generator[object, None, None]:
    """Decode each physical JSONL line as exactly one strict JSON value.

    One line is held at a time, up to SQLite's value length (the physical
    bound on a stored record; a longer line is read past and refused by
    name), and decoded by the C JSON backend, with directly encoded provider
    surrogates decoded exactly. A record stream is never parsed as one
    document. A supplied failure observer permits candidacy from healthy
    records while retaining decode-loss evidence; without one, the first
    refused record raises. The caller owns the original handle.
    """
    import json

    from polylogue.core.json import JSONDecodeError as FacadeDecodeError
    from polylogue.core.json import decode_provider_utf8
    from polylogue.core.json import loads as json_loads
    from polylogue.core.json_envelope import OversizedRecord, bounded_lines

    def refuse(_constant: str) -> object:
        raise ValueError("non-finite JSON constant")

    def refused(line_number: int, cause: BaseException | None) -> None:
        failure = ijson.JSONError(f"malformed JSONL record at line {line_number}")
        failure.__cause__ = cause
        if on_decode_failure is None:
            raise failure
        on_decode_failure(failure)

    with io.BufferedReader(_CheckedInput(handle, check_stop), buffer_size=1024 * 1024) as lines:
        for line_number, line in enumerate(bounded_lines(lines), start=1):
            if isinstance(line, OversizedRecord):
                refused(line_number, None)
                continue
            assert isinstance(line, bytes)
            raw = line.strip(b" \t\r\n")
            if raw.startswith(codecs.BOM_UTF8):
                raw = raw[len(codecs.BOM_UTF8) :].lstrip(b" \t\r\n")
            if not raw:
                continue
            try:
                value = json_loads(raw)
            except FacadeDecodeError as exc:
                try:
                    value = json.loads(decode_provider_utf8(raw), parse_constant=refuse)
                except (UnicodeDecodeError, ValueError):
                    refused(line_number, exc)
                    continue
            if check_stop is not None:
                check_stop()
            yield value


def iter_projected_jsonl_records(
    handle: IO[bytes],
    rule: DetectorProjection,
    *,
    check_stop: Callable[[], None] | None = None,
    on_decode_failure: Callable[[Exception], None] | None = None,
) -> Generator[object, None, None]:
    """Project each strictly decoded JSONL record in one pass over the input."""
    records = iter_decoded_jsonl_records(handle, check_stop=check_stop, on_decode_failure=on_decode_failure)
    try:
        for record in records:
            yield project_detection_value(record, rule)
    finally:
        records.close()


def iter_projected_document_records(
    handle: IO[bytes],
    rule: DetectorProjection,
    *,
    encoding: str = "utf-8",
    check_stop: Callable[[], None] | None = None,
    on_root: Callable[[Literal["record", "sequence"]], None] | None = None,
) -> Generator[object, None, None]:
    """Project a complete document's root records without retaining its array.

    Yielded records are provisional until normal exhaustion validates the
    closing delimiter and EOF. The caller owns the original input handle.
    """
    from ijson.backends import python as exact_backend

    from polylogue.core.json_envelope import LexemeAlignedReader, _PrefixStringReader

    source = _DetectionText(handle, encoding, check_stop)
    with io.BufferedReader(source) as reader, ExitStack() as stack:
        events = iter(exact_backend.basic_parse(LexemeAlignedReader(_PrefixStringReader(reader, scalar_values=True))))
        try:
            event, value = next(events)
            if on_root is not None:
                on_root("sequence" if event == "start_array" else "record")
            if event == "start_array":
                while True:
                    event, value = next(events)
                    if event == "end_array":
                        break
                    yield _project(events, event, value, rule, stack)
            else:
                yield _project(events, event, value, rule, stack)
            for _event in events:
                raise ijson.JSONError("trailing JSON candidacy value")
        except StopIteration as exc:
            raise ijson.JSONError("incomplete JSON candidacy document") from exc
        except (UnicodeError, ijson.JSONError, JSONDecodeError):
            if source.callback_failure is not None:
                raise source.callback_failure from None
            raise
