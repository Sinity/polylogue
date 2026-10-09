"""Complete event projections declared by the provider detector owners.

A projection describes exactly the fields and container folds its predicate
uses. Unselected containers are consumed and validated, without building their
values. Array witnesses preserve the declared first/all/any predicate; they
are not a prefix sample. Scalar decoding retains one value at a time.
"""

from __future__ import annotations

import codecs
import io
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack, closing
from dataclasses import dataclass
from decimal import Decimal
from json import JSONDecodeError
from typing import IO, TYPE_CHECKING, Literal, Protocol, cast

import ijson

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.content_identity import JSON_TEXT_ENCODINGS
from polylogue.core.json import JSONDocument, json_document_or_none
from polylogue.schemas.observation_spill import SpilledKey, StreamedJSONDocument, _literal_key

if TYPE_CHECKING:
    from polylogue.schemas.observation_spill import _ScalarTokenStore


@dataclass(frozen=True, slots=True)
class DetectorProjection:
    fields: Mapping[str, DetectorProjection | None] | None = None
    item: DetectorProjection | None = None
    array_fold: Literal["type", "first", "any", "all"] = "type"
    array_predicate: Callable[[object], bool] | None = None
    mapping_predicate: Callable[[object], bool] | None = None
    mapping_witness: object = None
    mapping_key_prefix: str | None = None
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
        self.original_size = len(fields)
        self.metadata_values_scalarish = getattr(fields, "metadata_values_scalarish", None)
        self._json_record: object = _UNREAD_RECORD

    def __len__(self) -> int:
        return self.original_size

    def json_record(self) -> JSONDocument | None:
        if self._json_record is _UNREAD_RECORD:
            record = json_document_or_none(self)
            # Avoid a self-reference cycle on every accepted record.
            self._json_record = _RECORD_IS_SELF if record is self else record
        if self._json_record is _RECORD_IS_SELF:
            return cast(JSONDocument, self)
        return cast(JSONDocument | None, self._json_record)


class _ProjectedMapping(dict[str, object]):
    """Selected predicate fields carrying the original unique-key count."""

    def __init__(
        self, fields: dict[str, object], original_size: int, metadata_values_scalarish: bool | None = None
    ) -> None:
        super().__init__(fields)
        self.original_size = original_size
        self.metadata_values_scalarish = metadata_values_scalarish

    def __len__(self) -> int:
        return self.original_size


def detection_read_view(value: object) -> object:
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
    return _project_value(
        events,
        event,
        value,
        rule,
        stack,
        scalarish_depth=-1 if rule is not None and rule.capture_metadata_values else None,
    )[0]


class _ProjectionKeys:
    """Exact duplicate folds over the scalar owner, with bounded key references."""

    def __init__(self, stack: ExitStack) -> None:
        self.stack = stack
        self.owner: _ScalarTokenStore | None = None
        self.scope = 0

    def observe(self, key: str | SpilledKey, *, accepted: bool = True, scalarish: bool = True) -> None:
        from polylogue.schemas.observation_spill import _ScalarTokenStore

        if self.owner is None:
            if isinstance(key, SpilledKey):
                if key._owner is None:
                    raise ValueError("detector event key has no scalar owner")
                self.owner = key._owner
            else:
                document = StreamedJSONDocument(None)
                self.stack.enter_context(document)
                self.owner = _ScalarTokenStore(document.connection)
            connection = self.owner.connection
            if not self.owner.projection_scope:
                connection.execute(
                    "CREATE TABLE projection_keys(scope INTEGER,token INTEGER,digest BLOB,"
                    "accepted INTEGER NOT NULL,scalarish INTEGER NOT NULL,PRIMARY KEY(scope,token)) WITHOUT ROWID"
                )
                connection.execute("CREATE INDEX projection_key_digest ON projection_keys(scope,digest)")
            self.owner.projection_scope += 1
            self.scope = self.owner.projection_scope
            self.stack.callback(connection.execute, "DELETE FROM projection_keys WHERE scope=?", (self.scope,))
        connection = self.owner.connection
        reference = _literal_key(connection, key) if isinstance(key, str) else key
        if reference.connection is not connection:
            raise ValueError("detector event keys have different owners")
        canonical = reference.token
        digest = reference.digest
        with closing(
            connection.execute("SELECT token FROM projection_keys WHERE scope=? AND digest=?", (self.scope, digest))
        ) as rows:
            for (token,) in rows:
                if reference.compare(SpilledKey(connection, token)) == 0:
                    canonical = token
                    break
        connection.execute(
            "INSERT INTO projection_keys VALUES (?,?,?,?,?) ON CONFLICT(scope,token) DO UPDATE SET "
            "accepted=excluded.accepted,scalarish=excluded.scalarish",
            (self.scope, canonical, digest, int(accepted), int(scalarish)),
        )

    def totals(self) -> tuple[int, int, int]:
        if self.owner is None:
            return 0, 0, 0
        row = self.owner.connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(accepted=0),0),COALESCE(SUM(scalarish=0),0) "
            "FROM projection_keys WHERE scope=?",
            (self.scope,),
        ).fetchone()
        return int(row[0]), int(row[1]), int(row[2])


def _selected_key(key: str | SpilledKey, fields: Mapping[str, object]) -> str | None:
    if not fields:
        return None
    if isinstance(key, str):
        return key if key in fields else None
    small = key.small_name
    if small is not None:
        return small if small in fields else None
    return next((name for name in fields if key.matches(name)), None)


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
    with ExitStack() as stack:
        keys = _ProjectionKeys(stack)
        while True:
            following, key = next(events)
            if following == "end_map":
                count, refused, _ = keys.totals()
                return count <= 8 and not refused
            if following != "map_key" or not isinstance(key, str | SpilledKey):
                raise ijson.JSONError("invalid taxonomy object event")
            following, _value = next(events)
            accepted = _consume_scalarish(events, following, depth + 1)
            keys.observe(key, accepted=accepted)


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
            keys = (
                _ProjectionKeys(mapping_stack)
                if (rule.mapping_predicate is not None or rule.preserve_mapping_size or scalarish_depth is not None)
                else None
            )
            while True:
                event, key = next(events)
                if event == "end_map":
                    break
                if event != "map_key" or not isinstance(key, str | SpilledKey):
                    raise ValueError("invalid detector object event")
                event, value = next(events)
                child_depth = None if scalarish_depth is None or scalarish_depth >= 2 else scalarish_depth + 1
                name = _selected_key(key, rule.fields or {})
                child_rule = (
                    rule.item
                    if rule.mapping_predicate is not None
                    else (rule.fields or {}).get(name)
                    if name is not None
                    else None
                )
                item, scalarish = _project_value(events, event, value, child_rule, stack, scalarish_depth=child_depth)
                accepted = rule.mapping_predicate(item) if rule.mapping_predicate is not None else True
                if keys is not None:
                    keys.observe(key, accepted=accepted, scalarish=scalarish)
                if rule.mapping_predicate is None:
                    if rule.mapping_key_prefix is not None:
                        matching_key |= key.startswith(rule.mapping_key_prefix)
                    elif name is not None:
                        fields[name] = item
            count, refused, scalarish_refused = keys.totals() if keys is not None else (0, 0, 0)
            scalarish = True
            metadata_values = None
            if scalarish_depth is not None:
                metadata_values = not scalarish_refused
                scalarish = scalarish_depth < 2 and count <= 8 and metadata_values
            if rule.mapping_predicate is not None:
                return ({} if not count else {"node": None if refused else rule.mapping_witness}), scalarish
            if rule.mapping_key_prefix is not None:
                return (rule.mapping_witness if matching_key else {}), scalarish
            if rule.preserve_mapping_size:
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
    from polylogue.schemas.observation_spill import _ScalarTokenReference

    if isinstance(value, _ScalarTokenReference):
        value = value.read()
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
    return _project_object(value, root_rule, scalarish_depth=-1 if root_rule.capture_metadata_values else None)[0]


def _object_scalarish(value: object, depth: int) -> bool:
    """:func:`_consume_scalarish` over a decoded value."""
    if isinstance(value, list):
        children = value.structure_values() if hasattr(value, "structure_values") else value
        return depth < 2 and len(value) <= 32 and all(_object_scalarish(item, depth + 1) for item in children)
    if isinstance(value, dict):
        children = (
            (child for _key, child in value.structure_key_items())
            if hasattr(value, "structure_key_items")
            else value.values()
        )
        return depth < 2 and len(value) <= 8 and all(_object_scalarish(item, depth + 1) for item in children)
    return True


def _project_object(
    value: object, rule: DetectorProjection | None, *, scalarish_depth: int | None
) -> tuple[object, bool]:
    """:func:`_project_value` over a decoded value; keys are already unique (last value wins)."""
    from polylogue.schemas.observation_spill import SpilledObject

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
            scalarish_depth is None and rule.mapping_predicate is None and rule.mapping_key_prefix is None
        )
        structural_entries = not selected_fields_only and isinstance(value, SpilledObject)
        entries = (
            ((key, value[key]) for key in rule.fields or {} if key in value)
            if selected_fields_only
            else value.structure_key_items()
            if isinstance(value, SpilledObject) and structural_entries
            else value.items()
        )
        for key, child in entries:
            name = _selected_key(key, rule.fields or {})
            child_rule = (
                rule.item
                if rule.mapping_predicate is not None
                else (rule.fields or {}).get(name)
                if name is not None
                else None
            )
            if isinstance(value, SpilledObject) and isinstance(key, SpilledKey) and child_rule is not None:
                child = value.value_for_key(key)
            item, scalarish = _project_object(child, child_rule, scalarish_depth=child_depth)
            values_scalarish &= scalarish
            if rule.mapping_predicate is not None:
                refused |= not rule.mapping_predicate(item)
            elif rule.mapping_key_prefix is not None:
                matching_key |= key.startswith(rule.mapping_key_prefix)
            elif name is not None:
                fields[name] = item
        count = len(value)
        mapping_scalarish = True
        metadata_values = None
        if scalarish_depth is not None:
            metadata_values = values_scalarish
            mapping_scalarish = scalarish_depth < 2 and count <= 8 and metadata_values
        if rule.mapping_predicate is not None:
            return ({} if not count else {"node": None if refused else rule.mapping_witness}), mapping_scalarish
        if rule.mapping_key_prefix is not None:
            return (rule.mapping_witness if matching_key else {}), mapping_scalarish
        if rule.preserve_mapping_size:
            return _ProjectedMapping(fields, count, metadata_values), mapping_scalarish
        return fields, mapping_scalarish
    if isinstance(value, list):
        # Decoded arrays are already structurally validated. These folds
        # discard their tail values unless metadata or a predicate needs them.
        if scalarish_depth is None and rule.array_predicate is None:
            if not value or rule.array_fold == "type":
                return [], True
            if rule.array_fold == "first":
                projected_first, _scalarish = _project_object(value[0], rule.item, scalarish_depth=None)
                return ([projected_first] if len(value) == 1 else [projected_first, None]), True
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
    """Decode complete physical records, retaining exact large values on disk.

    Records are exposed only after syntax succeeds. A later failure denies
    complete provider proof after earlier records were observed. The caller
    owns the input handle; this generator owns its lazy record tree.
    """
    from contextlib import closing

    from polylogue.sources.decoder_json import DecodedRecordSequence, JsonlDecodeError, _retain_jsonl_records, logger

    def refused(cause: Exception) -> None:
        line_number = cause.line_number if isinstance(cause, JsonlDecodeError) else 0
        failure = ijson.JSONError(f"malformed JSONL record at line {line_number}")
        failure.__cause__ = cause
        if on_decode_failure is None:
            raise failure
        on_decode_failure(failure)

    with (
        closing(DecodedRecordSequence(())) as tape,
        closing(
            _retain_jsonl_records(
                tape,
                logger,
                _CheckedInput(handle, check_stop),
                "recognition.jsonl",
                fail_on_decode_error=False,
                repair=False,
                on_decode_failure=refused,
            )
        ) as records,
    ):
        for value in records:
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
