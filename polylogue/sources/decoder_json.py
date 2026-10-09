"""Byte decoding and streamed JSON extraction helpers."""

from __future__ import annotations

import codecs
import json
import re
import sys
import tempfile
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import IO, TYPE_CHECKING, Never, Protocol, SupportsIndex, TypeAlias, TypeGuard, TypeVar, cast, overload

import ijson

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.content_identity import JSON_TEXT_ENCODINGS
from polylogue.core.json import (
    JSONDecodeError,
    _decode_integer,
    _ValidatedJSONContainer,
    decode_provider_utf8,
    normalize_json_decimal,
)
from polylogue.core.json import loads as json_loads
from polylogue.core.json_envelope import JSONL_MEMORY_BUFFER_BYTES, OversizedRecord, bounded_lines
from polylogue.logging import get_logger
from polylogue.sources import value_bounds
from polylogue.sources.pickle_spool import PickleSpool

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.schemas.observation_spill import StreamedJSONDocument


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = dict[str, "JsonValue"] | list["JsonValue"] | JsonScalar
JsonReadable: TypeAlias = IO[bytes]


class JsonlReadable(Protocol):
    """A borrowed physical-record input needs only bounded byte reads."""

    def read(self, size: int = -1) -> bytes: ...


def normalize_ijson_stdlib_numbers(value: object) -> object:
    """Match ``json.load`` numbers while retaining only one decoded record.

    Every object-streaming route passes each decoded member through here, so
    this is also where one member's scalars meet SQLite's storable-value
    limit (``polylogue.sources.value_bounds``).
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        value_bounds.require_storable_string(value)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = normalize_ijson_stdlib_numbers(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            value_bounds.require_storable_string(key, kind="object key")
            value[key] = normalize_ijson_stdlib_numbers(item)
    return value


@dataclass(frozen=True)
class _TreeRecordRef:
    node_id: int


class DecodedRecordSequence(list[JsonValue], _ValidatedJSONContainer):
    """Creator-local, repeatable decoded records with disk-backed ordinal lookup.

    Only values produced by the decoder enter this private tape. It carries
    no acquisition evidence and must close before its preparation returns.
    """

    def __init__(self, records: Iterable[JsonValue]) -> None:
        list.__init__(self)
        self._spool: PickleSpool[JsonValue | _TreeRecordRef] = PickleSpool(indexed=True)
        self._tree: StreamedJSONDocument | None = None
        self._closed = False
        try:
            iterator = iter(records)
            try:
                for value in iterator:
                    check_compute_cancelled()
                    normalized = normalize_json_decimal(value)
                    if not _is_json_value(normalized):
                        raise ValueError("decoded record does not satisfy the JsonValue contract")
                    self._spool.append(normalized)
                    del normalized, value
            finally:
                primary = sys.exception()
                close = getattr(iterator, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException as cleanup:
                        if primary is not None:
                            raise BaseExceptionGroup(
                                "decoded input and iterator close failed", [primary, cleanup]
                            ) from None
                        raise
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "decoded record retention and physical close failed", [primary, cleanup]
                ) from None
            raise

    @classmethod
    def from_jsonl(
        cls,
        handle: JsonlReadable | Iterable[bytes],
        path_name: str,
        *,
        logger_obj: LoggerLike | None = None,
        fail_on_decode_error: bool = False,
        repair: bool = True,
        on_decode_failure: Callable[[Exception], None] | None = None,
        sample_records: int | None = None,
    ) -> DecodedRecordSequence:
        """Retain physical records through parser completion and cohort replay.

        The small-record buffer changes storage strategy only. Larger records
        retain exact scalar chunks and lazy containers under this tape's owner.
        """
        tape = cls(())
        try:
            with closing(
                _retain_jsonl_records(
                    tape,
                    logger_obj or logger,
                    handle,
                    path_name,
                    fail_on_decode_error=fail_on_decode_error,
                    repair=repair,
                    on_decode_failure=on_decode_failure,
                    expose_records=False,
                )
            ) as retained:
                for _record in retained:
                    if sample_records is not None and len(tape) >= sample_records:
                        break
        except BaseException as primary:
            try:
                tape.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup("JSONL retention and physical close failed", [primary, cleanup]) from None
            raise
        return tape

    @classmethod
    def from_raw_document(cls, path: Path) -> DecodedRecordSequence:
        """Own a completed raw JSON root with the facade's exact attempt order."""
        tape = cls(())
        failure: Exception | None = None
        try:
            with path.open("rb") as source:
                detected_encoding = json.detect_encoding(source.read(4))
            attempts = (
                ("utf-8-sig", True, False),
                (detected_encoding, False, False),
                ("utf-8-sig", True, True),
            )
            for encoding, provider_utf8, allow_nonfinite in attempts:
                try:
                    tape._append_document(
                        path,
                        allow_nonfinite=allow_nonfinite,
                        strip_bom=False,
                        provider_utf8=provider_utf8,
                        text_encoding=encoding,
                    )
                except (ijson.JSONError, UnicodeError, ValueError) as error:
                    if failure is None:
                        failure = error
                else:
                    return tape
            assert failure is not None
            raise JSONDecodeError(str(failure)) from failure
        except BaseException:
            tape.close()
            raise

    @classmethod
    def from_archive_jsonl(
        cls, handle: JsonlReadable, *, textual: bool = False, dict_only: bool = False
    ) -> tuple[DecodedRecordSequence, int, str | None]:
        """Own retained raw records with the archival provider-text policy."""
        from polylogue.core.json_envelope import _LineSource

        tape = cls(())
        malformed = 0
        detail = None
        first_decoded = True
        try:
            with tempfile.TemporaryDirectory(prefix="polylogue-raw-jsonl-") as directory:
                raw_path = Path(directory) / "line.json"
                converted = Path(directory) / "text.json"
                source = _LineSource(handle)
                line_number = 0
                while (line := source.next_line()) is not None:
                    line_number += 1
                    raw = _capture_jsonl_line(iter(partial(line.read, 64 * 1024), b""), raw_path, whitespace=b"")
                    try:
                        if raw is not None:
                            text = raw.decode("utf-8", "surrogatepass") if textual else decode_provider_utf8(raw)
                            if first_decoded:
                                text = text.lstrip("\ufeff")
                            first_decoded = False
                            text = text.strip()
                            if not text:
                                continue
                            value = cast(JsonValue, json.loads(text, parse_int=_decode_integer))
                            if not dict_only or isinstance(value, dict):
                                tape._spool.append(value)
                        else:
                            nonempty = _convert_archive_text(
                                raw_path, converted, textual=textual, first_decoded=first_decoded
                            )
                            first_decoded = False
                            if not nonempty:
                                continue
                            tape._append_document(
                                converted,
                                allow_nonfinite=True,
                                strip_bom=False,
                                provider_utf8=False,
                                dict_only=dict_only,
                                text_encoding="utf-8",
                            )
                    except (json.JSONDecodeError, ijson.JSONError, UnicodeError, ValueError) as error:
                        malformed += 1
                        if detail is None:
                            reason = error.reason if isinstance(error, UnicodeDecodeError) else str(error)
                            detail = f"line {line_number}: {reason}"
            if not tape:
                raise ValueError("No valid JSONL records found")
            return tape, malformed, detail
        except BaseException:
            tape.close()
            raise

    def _append_document(
        self,
        path: Path,
        *,
        allow_nonfinite: bool,
        strip_bom: bool,
        provider_utf8: bool = True,
        dict_only: bool = False,
        text_encoding: str | None = None,
    ) -> None:
        if self._tree is None:
            from polylogue.schemas.observation_spill import StreamedJSONDocument

            self._tree = StreamedJSONDocument(None)
            self._tree.__enter__()
        node = self._tree.append_document(
            path,
            allow_nonfinite=allow_nonfinite,
            strip_bom=strip_bom,
            provider_utf8=provider_utf8,
            text_encoding=text_encoding,
        )
        if (
            not dict_only
            or self._tree.connection.execute("SELECT kind FROM json_nodes WHERE id=?", (node,)).fetchone()[0]
            == "object"
        ):
            self._spool.append(_TreeRecordRef(node))

    def _value(self, retained: JsonValue | _TreeRecordRef) -> JsonValue:
        if isinstance(retained, _TreeRecordRef):
            from polylogue.schemas.observation_spill import _load_node

            assert self._tree is not None
            return _load_node(self._tree.connection, retained.node_id)
        return retained

    def __len__(self) -> int:
        self._require_open()
        return len(self._spool)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("decoded record sequence is closed")

    @overload
    def __getitem__(self, ordinal: SupportsIndex) -> JsonValue: ...

    @overload
    def __getitem__(self, ordinal: slice) -> list[JsonValue]: ...

    def __getitem__(self, ordinal: SupportsIndex | slice) -> JsonValue | list[JsonValue]:
        self._require_open()
        check_compute_cancelled()
        if isinstance(ordinal, slice):
            return [self[index] for index in range(*ordinal.indices(len(self)))]
        ordinal = ordinal.__index__()
        if ordinal < 0:
            ordinal += len(self)
        if not 0 <= ordinal < len(self):
            raise IndexError(ordinal)
        return self._value(next(self._spool.iter_from(ordinal)))

    def __iter__(self) -> Iterator[JsonValue]:
        self._require_open()
        for value in self._spool:
            self._require_open()
            check_compute_cancelled()
            yield self._value(value)

    def structure_values(self) -> Iterator[JsonValue]:
        """Traverse root kinds while leaving unselected scalar tokens on disk."""
        from polylogue.schemas.observation_spill import _load_structure_node

        self._require_open()
        for value in self._spool:
            self._require_open()
            check_compute_cancelled()
            if isinstance(value, _TreeRecordRef):
                assert self._tree is not None
                yield _load_structure_node(self._tree.connection, value.node_id)
            else:
                yield value

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, (list, DecodedRecordSequence))
            and len(self) == len(other)
            and all(left == right for left, right in zip(self, other, strict=True))
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __contains__(self, value: object) -> bool:
        return any(item == value for item in self)

    def __bool__(self) -> bool:
        return bool(len(self))

    def _immutable(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("decoded record sequence is read-only")

    append = clear = extend = insert = pop = remove = reverse = sort = _immutable
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._spool.close()
        finally:
            if self._tree is not None:
                self._tree.__exit__(None, None, None)
        self._closed = True


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
    for encoding in JSON_TEXT_ENCODINGS:
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
        # Provider bytes with directly encoded surrogates (lone or CESU-8
        # pairs) decode exactly; only other bytes fall to the lenient guess.
        try:
            provider_text: str | None = decode_provider_utf8(raw_pending)
        except UnicodeDecodeError:
            provider_text = None
        if provider_text is not None:
            try:
                return ([cast(JsonValue, json.loads(provider_text, parse_int=_decode_integer))], 0, None)
            except json.JSONDecodeError:
                pass
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
        parsed = json.loads(decoded, parse_int=_decode_integer)
    except json.JSONDecodeError as exc:
        if is_last:
            logger_obj.debug("Skipping truncated trailing line in %s: %s", path_name, exc)
            return ([], 1, line_number)
        return ([], 1, line_number)
    return ([cast(JsonValue, parsed)], 0, None)


# A buffering choice only: records beyond it continue through exact disk
# transport. It is never an input support, storage, or schema limit.
_JSONL_MEMORY_BYTES = JSONL_MEMORY_BUFFER_BYTES


def _capture_jsonl_line(chunks: Iterable[bytes], path: Path, *, whitespace: bytes | None = None) -> bytes | None:
    memory: bytearray | None = bytearray()
    output = None
    started = False
    written = last_content = 0
    try:
        for chunk in chunks:
            check_compute_cancelled()
            if not started:
                chunk = chunk.lstrip(whitespace)
                started = bool(chunk)
            if not chunk:
                continue
            if memory is not None and len(memory) + len(chunk) <= _JSONL_MEMORY_BYTES:
                memory.extend(chunk)
                continue
            if output is None:
                output = path.open("wb")
                assert memory is not None
                output.write(memory)
                written = len(memory)
                last_content = len(memory.rstrip(whitespace))
                memory = None
            output.write(chunk)
            content_length = len(chunk.rstrip(whitespace))
            if content_length:
                last_content = written + content_length
            written += len(chunk)
        if memory is not None:
            return bytes(memory).strip(whitespace)
        assert output is not None
        output.truncate(last_content)
        return None
    finally:
        if output is not None:
            output.close()


def _convert_jsonl_text(
    source: Path,
    target: Path,
    encoding: str,
    *,
    errors: str = "strict",
    clean: bool = False,
    strip_bom: bool = False,
    provider: bool = False,
) -> bool:
    """Choose a text policy only after decoding the complete physical line."""
    from polylogue.schemas.observation_spill import _ExactJSONText

    nonempty = False
    started = not strip_bom
    with source.open("rb") as input_stream, target.open("wb") as output:
        if provider:
            reader = _ExactJSONText(input_stream, "utf-8-sig", strip_bom=False)
            while chunk := reader.read(64 * 1024):
                check_compute_cancelled()
                nonempty = True
                output.write(chunk)
            return nonempty
        decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
        while True:
            check_compute_cancelled()
            chunk = input_stream.read(64 * 1024)
            value = decoder.decode(chunk, final=not chunk)
            if clean:
                value = value.replace("\x00", "")
            if not started:
                value = value.lstrip("\ufeff")
                started = bool(value)
            if value:
                nonempty = True
                output.write(value.encode("utf-8", "surrogatepass"))
            if not chunk:
                return nonempty


def _convert_archive_text(source: Path, target: Path, *, textual: bool, first_decoded: bool) -> bool:
    """Finish decoding before choosing first-line BOM policy; trim Unicode whitespace on disk."""
    from polylogue.schemas.observation_spill import _ExactJSONText

    encoding = "utf-8" if textual else "utf-8-sig"
    started = False
    bom_started = not first_decoded
    written = last_content = 0
    with source.open("rb") as input_stream, target.open("wb") as output:
        reader = _ExactJSONText(input_stream, encoding, strip_bom=False, provider_utf8=not textual)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogatepass")
        while True:
            check_compute_cancelled()
            chunk = reader.read(64 * 1024)
            value = decoder.decode(chunk, final=not chunk)
            if not bom_started:
                value = value.lstrip("\ufeff")
                bom_started = bool(value)
            if not started:
                value = value.lstrip()
                started = bool(value)
            encoded = value.encode("utf-8", "surrogatepass")
            output.write(encoded)
            content = value.rstrip().encode("utf-8", "surrogatepass")
            if content:
                last_content = written + len(content)
            written += len(encoded)
            if not chunk:
                output.truncate(last_content)
                return bool(last_content)


def _append_large_jsonl_record(tape: DecodedRecordSequence, source: Path, converted: Path) -> bool:
    failures = (ijson.JSONError, ValueError, UnicodeError)
    try:
        tape._append_document(source, allow_nonfinite=False, strip_bom=False)
        return True
    except failures:
        pass
    try:
        _convert_jsonl_text(source, converted, "utf-8", provider=True)
        tape._append_document(converted, allow_nonfinite=True, strip_bom=False)
        return True
    except failures:
        pass
    # The legacy repair decoder chooses the first nonempty text that fully
    # decodes. A syntax failure after that choice never tries another codec.
    for encoding in JSON_TEXT_ENCODINGS:
        try:
            nonempty = _convert_jsonl_text(source, converted, encoding, clean=True, strip_bom=True)
        except UnicodeError:
            continue
        if nonempty:
            try:
                tape._append_document(converted, allow_nonfinite=True, strip_bom=False)
                return True
            except failures:
                return False
    _convert_jsonl_text(source, converted, "utf-8", errors="ignore", clean=True)
    try:
        tape._append_document(converted, allow_nonfinite=True, strip_bom=False)
        return True
    except failures:
        return False


def _retain_jsonl_records(
    tape: DecodedRecordSequence,
    logger_obj: LoggerLike,
    handle: JsonlReadable | Iterable[bytes],
    path_name: str,
    *,
    fail_on_decode_error: bool,
    repair: bool,
    on_decode_failure: Callable[[Exception], None] | None,
    expose_records: bool = True,
) -> Generator[JsonValue, None, None]:
    from polylogue.core.json_envelope import _LineSource

    def line_chunks(physical_line: bytes) -> Iterator[bytes]:
        for start in range(0, len(physical_line), 64 * 1024):
            yield physical_line[start : start + 64 * 1024]

    def physical_lines() -> Generator[Iterable[bytes], None, None]:
        if callable(getattr(handle, "read", None)):
            source = _LineSource(cast(JsonlReadable, handle))
            while (line := source.next_line()) is not None:
                yield iter(partial(line.read, 64 * 1024), b"")
        else:
            iterator = iter(cast(Iterable[bytes], handle))
            try:
                for physical_line in iterator:
                    yield line_chunks(physical_line)
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    errors = 0
    first_error = None
    line_number = 0
    with tempfile.TemporaryDirectory(prefix="polylogue-jsonl-record-") as directory:
        raw_path = Path(directory) / "line.json"
        converted = Path(directory) / "text.json"
        with closing(physical_lines()) as lines:
            for line_number, chunks in enumerate(lines, start=1):
                raw = _capture_jsonl_line(chunks, raw_path, whitespace=None if repair else b" \t\r\n")
                if raw == b"":
                    continue
                previous_records = len(tape)
                records: list[JsonValue] = []
                if raw is not None:
                    if repair:
                        records, failed, _error_line = _yield_jsonl_pending(
                            logger_obj, raw, is_last=False, path_name=path_name, line_number=line_number
                        )
                    else:
                        if raw.startswith(codecs.BOM_UTF8):
                            raw = raw[len(codecs.BOM_UTF8) :].lstrip(b" \t\r\n")
                        if not raw:
                            continue
                        try:
                            records = [json_loads(raw)]
                            failed = 0
                        except JSONDecodeError:
                            try:

                                def refuse_constant(_value: str) -> object:
                                    raise ValueError("non-finite JSON constant")

                                records = [
                                    json.loads(
                                        decode_provider_utf8(raw),
                                        parse_constant=refuse_constant,
                                        parse_int=_decode_integer,
                                    )
                                ]
                                failed = 0
                            except (UnicodeError, ValueError):
                                records, failed = [], 1
                    for record in records:
                        tape._spool.append(record)
                else:
                    if repair:
                        failed = int(not _append_large_jsonl_record(tape, raw_path, converted))
                    else:
                        try:
                            tape._append_document(raw_path, allow_nonfinite=False, strip_bom=False)
                            failed = 0
                        except (ijson.JSONError, ValueError, UnicodeError):
                            try:
                                _convert_jsonl_text(raw_path, converted, "utf-8", provider=True)
                                with converted.open("rb") as converted_stream:
                                    nonblank = any(
                                        chunk.strip(b" \t\r\n")
                                        for chunk in iter(lambda: converted_stream.read(64 * 1024), b"")
                                    )
                                if nonblank:
                                    tape._append_document(converted, allow_nonfinite=False, strip_bom=False)
                                failed = 0
                            except (ijson.JSONError, ValueError, UnicodeError):
                                failed = 1
                if failed:
                    errors += failed
                    if first_error is None:
                        first_error = line_number
                    if on_decode_failure is not None:
                        on_decode_failure(
                            JsonlDecodeError(
                                path_name, line_number=line_number, cause=ValueError("malformed JSONL record")
                            )
                        )
                    if errors <= 3:
                        logger_obj.warning("Skipping invalid JSON line in %s", path_name)
                    elif errors == 4:
                        logger_obj.warning("Skipping further invalid JSON lines in %s...", path_name)
                if not expose_records:
                    yield None
                elif raw is not None:
                    yield from records
                else:
                    for record_index in range(previous_records, len(tape)):
                        yield tape[record_index]
    if fail_on_decode_error and errors:
        raise JsonlDecodeError(
            path_name, line_number=first_error or line_number, cause=ValueError("malformed JSONL record")
        )
    if errors > 3:
        logger_obj.warning("Skipped %d invalid JSON lines in %s", errors, path_name)


def _iter_jsonl_stream(
    logger_obj: LoggerLike,
    handle: JsonReadable | Iterable[bytes],
    path_name: str,
    *,
    fail_on_decode_error: bool = False,
) -> Iterable[JsonValue]:
    error_count = 0
    pending: bytes | str | None = None
    physical_line_number = 0
    pending_line_number: int | None = None
    first_decode_error_line: int | None = None

    for line in bounded_lines(handle):
        physical_line_number += 1
        raw = None if isinstance(line, OversizedRecord) else line.strip()
        if raw is not None and not raw:
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
            pending = None
        if isinstance(line, OversizedRecord):
            # Refused by name at the record bound, never allocated.
            error_count += 1
            if first_decode_error_line is None:
                first_decode_error_line = physical_line_number
            logger_obj.warning(
                "Skipping JSONL record of %d bytes at line %d in %s: beyond the record bound",
                line.size,
                physical_line_number,
                path_name,
            )
            continue
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


class _StdlibJsonRecordReader:
    """Borrow a JSON file and decode one selected value with stdlib semantics."""

    def __init__(self, handle: JsonReadable) -> None:
        self._handle = handle
        first = handle.read(4)
        self._text_decoder = codecs.getincrementaldecoder(json.detect_encoding(first))(errors="surrogatepass")
        self._buffer = self._text_decoder.decode(first)
        self._eof = False
        self._decoder = json.JSONDecoder()

    def _fill(self) -> bool:
        if self._eof:
            return False
        check_compute_cancelled()
        chunk = self._handle.read(64 * 1024)
        self._eof = not chunk
        self._buffer += self._text_decoder.decode(chunk, final=self._eof)
        return bool(chunk)

    def peek(self) -> str:
        while True:
            self._buffer = self._buffer.lstrip(" \t\r\n")
            if self._buffer or self._eof:
                return self._buffer[:1]
            self._fill()

    def expect(self, token: str) -> None:
        if self.peek() != token:
            raise json.JSONDecodeError(f"expected {token}", self._buffer, 0)
        self._buffer = self._buffer[1:]

    def value(self) -> JsonValue:
        self.peek()
        while True:
            check_compute_cancelled()
            try:
                value, end = self._decoder.raw_decode(self._buffer)
            except json.JSONDecodeError:
                if self._eof:
                    raise
                self._fill()
                continue
            if end == len(self._buffer) and not self._eof:
                self._fill()
                continue
            if end < len(self._buffer) and self._buffer[end] not in " \t\r\n,:]}":
                raise json.JSONDecodeError("invalid value boundary", self._buffer, end)
            self._buffer = self._buffer[end:]
            if not _is_json_value(value):
                raise json.JSONDecodeError("decoded value does not satisfy the JsonValue contract", self._buffer, 0)
            return value

    def array(self) -> Iterator[JsonValue]:
        self.expect("[")
        if self.peek() == "]":
            self.expect("]")
            return
        while True:
            yield self.value()
            if self.peek() == "]":
                self.expect("]")
                return
            self.expect(",")

    def discard(self) -> None:
        """Validate unused structure without assembling its arrays or objects."""
        token = self.peek()
        if token == "[":
            self.expect("[")
            if self.peek() == "]":
                self.expect("]")
                return
            while True:
                self.discard()
                if self.peek() == "]":
                    self.expect("]")
                    return
                self.expect(",")
        if token == "{":
            self.expect("{")
            if self.peek() == "}":
                self.expect("}")
                return
            while True:
                if self.peek() != '"' or not isinstance(self.value(), str):
                    raise json.JSONDecodeError("object key must be a string", self._buffer, 0)
                self.expect(":")
                self.discard()
                if self.peek() == "}":
                    self.expect("}")
                    return
                self.expect(",")
        self.value()

    def finish(self) -> None:
        if self.peek():
            raise json.JSONDecodeError("extra data", self._buffer, 0)


def _stdlib_prefixed_items(handle: JsonReadable, prefix: str) -> PickleSpool[JsonValue] | None:
    """Retain selected records with the original stdlib decoder's value law."""
    handle.seek(0)
    selected: PickleSpool[JsonValue] | None = None
    try:
        reader = _StdlibJsonRecordReader(handle)
        if reader.peek() == "[":
            if prefix == "item":
                selected = PickleSpool()
                for value in reader.array():
                    selected.append(value)
            else:
                reader.discard()
        elif reader.peek() == "{":
            reader.expect("{")
            if reader.peek() != "}":
                while True:
                    if reader.peek() != '"':
                        raise json.JSONDecodeError("object key must be a string", reader._buffer, 0)
                    key = reader.value()
                    reader.expect(":")
                    if prefix == "sessions.item" and key == "sessions":
                        if selected is not None:
                            selected.close()
                            selected = None
                        if reader.peek() == "[":
                            selected = PickleSpool()
                            for value in reader.array():
                                selected.append(value)
                        else:
                            reader.discard()
                    else:
                        reader.discard()
                    if reader.peek() == "}":
                        break
                    reader.expect(",")
            reader.expect("}")
        else:
            reader.discard()
        reader.finish()
        return selected
    except (json.JSONDecodeError, UnicodeDecodeError):
        if selected is not None:
            selected.close()
        return None
    except BaseException:
        if selected is not None:
            selected.close()
        raise


def _stream_prefixed_items(
    logger_obj: LoggerLike,
    ijson_module: IjsonModuleLike,
    handle: JsonReadable,
    path_name: str,
    prefix: str,
    *,
    strategy_name: str,
) -> tuple[bool, PickleSpool[JsonValue] | None]:
    records: PickleSpool[JsonValue] = PickleSpool()
    transferred = False
    try:

        def candidates() -> Generator[JsonValue, None, None]:
            yield from ijson_module.items(handle, prefix)

        decode_failure: Exception | None = None
        with closing(candidates()) as items:
            while True:
                check_compute_cancelled()
                try:
                    item = next(items)
                except StopIteration:
                    break
                except DaemonOperationCancelled:
                    raise
                except (ijson_module.common.JSONError, UnicodeError, ValueError) as failure:
                    decode_failure = failure
                    break
                records.append(item)
                del item
        if decode_failure is not None:
            if len(records):
                recovered = _stdlib_prefixed_items(handle, prefix)
                if recovered is not None:
                    transferred = True
                    try:
                        records.close()
                    except BaseException as recovery_failure:
                        try:
                            recovered.close()
                        except BaseException as cleanup:
                            raise BaseExceptionGroup(
                                "record recovery and spool close failed", [recovery_failure, cleanup]
                            ) from None
                        raise
                    return True, recovered
                logger_obj.warning(
                    "Partial JSON stream decode of %s (strategy %s): %s after %d record(s)",
                    path_name,
                    strategy_name,
                    type(decode_failure).__name__,
                    len(records),
                )
                raise PartialJsonStreamError(
                    path_name,
                    recovered=len(records),
                    offset=_json_error_offset(decode_failure),
                    cause=decode_failure,
                ) from decode_failure
            if not isinstance(decode_failure, ijson_module.common.JSONError):
                logger_obj.debug("Strategy %s failed for %s: %s", strategy_name, path_name, decode_failure)
            return False, None
        if not len(records):
            return False, None
        transferred = True
        return True, records
    finally:
        if not transferred:
            primary = sys.exception()
            try:
                records.close()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("record decoding and spool close failed", [primary, cleanup]) from None
                raise


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
    handle: JsonReadable | Iterable[bytes],
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

    if not callable(getattr(handle, "read", None)):
        raise TypeError("JSON document strategies require a readable byte stream")
    # JSONL callers may supply physical lines; document strategies require IO.
    handle = cast(JsonReadable, handle)

    # Rewinding strategies borrow one disk-backed copy for a nonseekable
    # original. Chunking bounds transport memory, not valid input size.
    seekable = getattr(handle, "seekable", None)
    if callable(seekable) and not seekable():
        with tempfile.TemporaryFile(prefix="polylogue-json-input-") as spool:
            while True:
                check_compute_cancelled()
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                spool.write(chunk)
            spool.seek(0)
            yield from iter_json_stream_with(
                logger_obj,
                ijson_module,
                spool,
                path_name,
                unpack_lists,
                fail_on_decode_error=fail_on_decode_error,
            )
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
            assert records is not None
            with closing(records):
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
            assert records is not None
            with closing(records):
                yield from records
            return

        handle.seek(0)

    reader = _StdlibJsonRecordReader(handle)
    data = reader.value()
    reader.finish()
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

    def adopt_child(self, selected: str | None) -> None:
        """Record a closed child container's candidate under this frame."""
        if self.kind == "map":
            self.child_types[self.key or ""] = selected
        elif selected is not None and self.first_child is None:
            self.first_child = selected


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
                self._frames[-1].adopt_child(selected)
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
    *,
    optional: frozenset[str] = frozenset(),
) -> tuple[dict[str, JsonValue], dict[str, int]] | None:
    """Build a root object without the arrays that a caller streams separately.

    Returns the remaining document, how many arrays each ``streamed`` path
    held, and, under ``__admission_future_type``, the first future wire type
    parser admission would report. A root key in ``rerouted_root_keys``, a
    streamed path holding a non-array (unless it is ``optional``, whose
    non-array value stays in the document), or invalid JSON refuses the
    document. The pass reads the complete input, so a truncated suffix
    refuses it too.
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
    expect_key: object = None
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
                    if expect_array not in optional:
                        return None
                    arrays[expect_array] -= 1
                    expect_array = None
                    builder.event("map_key", expect_key)
                    builder.event(event, value)
                    continue
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
                    expect_key = value
                    continue
            builder.event(event, value)
    except ijson.common.JSONError:
        return None
    finally:
        handle.seek(0)
    envelope = normalize_ijson_stdlib_numbers(builder.value)
    if not isinstance(envelope, dict):
        return None
    if "__admission_future_type" in envelope:
        # The streamed parsers read this key as probe metadata; a source
        # document that carries it stays on the object parser.
        return None
    if future_type.value is not None:
        envelope["__admission_future_type"] = future_type.value
    return cast(dict[str, JsonValue], envelope), arrays


#: Conversation-level attachment arrays, read after the messages in this order.
CLAUDE_AI_ATTACHMENT_ARRAYS = ("attachments", "files")


def claude_ai_object_envelope(handle: JsonReadable) -> tuple[dict[str, JsonValue], tuple[str, ...]] | None:
    """Prove one claude.ai conversation and keep every root field but its arrays.

    Returns the envelope without ``chat_messages`` and the root attachment
    arrays it names, each streamed separately; a non-array ``attachments``
    or ``files`` value stays in the envelope. Shapes the object parser
    routes elsewhere (account memories, projects, browser captures,
    ``sessions`` wrappers) stay on that route.
    """
    result = _root_envelope_without(
        handle,
        frozenset({"chat_messages", *CLAUDE_AI_ATTACHMENT_ARRAYS}),
        frozenset({"sessions", "account_uuid", "docs", "polylogue_capture_kind"}),
        optional=frozenset(CLAUDE_AI_ATTACHMENT_ARRAYS),
    )
    if result is None or result[1]["chat_messages"] != 1:
        return None
    envelope, arrays = result
    return envelope, tuple(key for key in CLAUDE_AI_ATTACHMENT_ARRAYS if arrays[key])


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
                    # Detectors may test presence rather than scalar type.
                    # Keep a type witness without materializing the container.
                    envelope[current_key] = {} if event == "start_map" else []
                if len(frames) == 1 and current_key == "steps" and event == "start_array":
                    envelope["steps"] = []
                if frames[-1].kind == "map" and frames[-1].key in {"type", "content_type", "kind", "record_type"}:
                    frames[-1].own_types[frames[-1].key or ""] = None
                frames.append(_FutureTypeFrame("map" if event == "start_map" else "array"))
                continue
            if event in {"end_array", "end_map"}:
                selected = frames.pop().selected()
                if frames:
                    frames[-1].adopt_child(selected)
                else:
                    first_future_type = selected
                continue
            if frames[-1].kind == "map":
                if frames[-1].key in {"type", "content_type", "kind", "record_type"}:
                    frames[-1].own_types[frames[-1].key or ""] = _future_wire_type(value) if event == "string" else None
                frames[-1].child_types[frames[-1].key or ""] = None
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
    detect: bool = True,
) -> int | None:
    """Validate a Grok object and report each member's shape without decoding it.

    ``on_item`` also receives the member's first future wire type, which the
    per-conversation parser admission reports; the collecting lowering admits
    conversation members only, so a future type elsewhere carries no event.

    With ``detect``, a document whose root also carries a hook event's or a
    transcript extract's markers is refused, so provider detection weighs
    it whole. A route whose provider is already Grok passes ``detect=False``:
    the Grok lowering reads only ``conversations`` whatever else the root holds.
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
    envelope_keys = {"uuid", "sessionId", "parentUuid", "message", "payload", "cwd", "version"}
    provenance_keys = {"file", "source_file", "source_path", "transcript", "session_file"}
    content_keys = {"content", "text", "message_text", "body"}
    hook_keys = {"event_type", "session_id", "timestamp", "provider"}
    taxonomy_fields = envelope_keys | provenance_keys | content_keys | hook_keys
    taxonomy_keys: set[str] = set()
    taxonomy_values: dict[str, bool] = {}

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
                if value in {"sessions", "polylogue_capture_kind"}:
                    return None
                if isinstance(value, str) and value.startswith("conversations."):
                    # A dotted root key would pose as a member path below.
                    return None
                if value == "conversations":
                    keys += 1
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
                elif prefix in hook_keys:
                    taxonomy_values[prefix] = True
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
    if detect and all(taxonomy_values.get(key, False) for key in hook_keys):
        return None
    if (
        detect
        and not taxonomy_keys.intersection(envelope_keys)
        and any(taxonomy_values.get(key, False) for key in provenance_keys)
        and any(taxonomy_values.get(key, False) for key in content_keys)
    ):
        return None
    return count if keys == arrays == 1 and (count == 0 or valid_members > 0) else None


def grok_taxonomy_witness(handle: JsonReadable) -> JsonValue:
    """The Grok document as artifact taxonomy weighs it, within a bounded size.

    Every root field is kept, each container cut to its first 64 entries;
    ``conversations`` keeps its first 64 members, each member's fields cut
    the same way, so every response kept is whole. Numbers read as
    ``json.load`` reads them, since the collecting route classifies that
    decode. The caller has proved the document with ``grok_export_item_count``.
    """
    events = iter(ijson.parse(handle))
    try:
        _prefix, event, _value = _next_event(events)
        if event != "start_map":
            raise ValueError("JSON document is not an object")
        witness: dict[str, object] = {}
        while True:
            _prefix, event, key = _next_event(events)
            if event == "end_map":
                return cast(JsonValue, witness)
            _prefix, event, value = _next_event(events)
            witness[str(key)] = _stdlib_floats(
                _witness_subtree(events, event, value, levels=3 if key == "conversations" else 1)
            )
    finally:
        handle.seek(0)


def _stdlib_floats(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_stdlib_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _stdlib_floats(item) for key, item in value.items()}
    return value


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


_CONTAINER_START = frozenset({"start_map", "start_array"})
_CONTAINER_END = frozenset({"end_map", "end_array"})


def _next_event(events: Iterator[tuple[str, str, object]]) -> tuple[str, str, object]:
    try:
        return next(events)
    except StopIteration:
        raise ValueError("incomplete JSON member") from None


def _enter_root_array(events: Iterator[tuple[str, str, object]], key: str) -> None:
    """Advance past the ``start_array`` of the root field ``key``.

    Walks the root object's structure, so a differently named root key that
    merely contains dots cannot be mistaken for a path into ``key``.
    """
    _prefix, event, _value = _next_event(events)
    if event != "start_map":
        raise ValueError("JSON document is not an object")
    while True:
        _prefix, event, name = _next_event(events)
        if event == "end_map":
            raise ValueError(f"JSON document has no {key} array")
        _prefix, event, value = _next_event(events)
        if name == key:
            break
        _skip_json_subtree(events, event)
    if event != "start_array":
        raise ValueError(f"JSON document {key} is not an array")


def spill_member_arrays(
    handle: JsonReadable,
    container: str,
    nested: str,
    *,
    on_member: Callable[[int, JsonValue, int | None], None],
    on_nested_item: Callable[[int, int, JsonValue], None],
) -> bool:
    """Split each member of the root array ``container`` from its ``nested`` list.

    Each ``nested`` item goes to ``on_nested_item(member, ordinal, item)``
    before ``on_member(member, fields, count)`` receives the member without
    that list and the list's length. A member whose ``nested`` value is not
    a list keeps it in ``fields`` with a ``None`` count; a non-object member
    arrives as itself (an array member as ``[]``), since the parser reads
    neither. Only one nested item is decoded at a time.

    Returns ``False`` when a member repeats the ``nested`` key: the decoder
    keeps only its last value, while this pass would already have spilled
    the earlier one. The caller has proved ``container`` is the one root
    array.
    """
    events = iter(ijson.parse(handle))
    _enter_root_array(events, container)
    index = -1
    while True:
        _prefix, event, value = _next_event(events)
        if event == "end_array":
            return True
        index += 1
        if event == "start_array":
            _skip_json_subtree(events, event)
            on_member(index, [], None)
            continue
        if event != "start_map":
            on_member(index, cast(JsonValue, normalize_ijson_stdlib_numbers(value)), None)
            continue
        builder = ijson.common.ObjectBuilder()
        builder.event("start_map", None)
        count: int | None = None
        seen_nested = False
        depth = 1
        while True:
            _prefix, event, value = _next_event(events)
            if depth == 1 and event == "map_key" and value == nested:
                if seen_nested:
                    return False
                seen_nested = True
                _prefix, event, value = _next_event(events)
                if event == "start_array":
                    count = 0
                    while True:
                        _prefix, event, value = _next_event(events)
                        if event == "end_array":
                            break
                        item = _json_subtree(events, event, value)
                        on_nested_item(index, count, cast(JsonValue, item))
                        count += 1
                    continue
                builder.event("map_key", nested)
                builder.event(event, value)
                nested_depth = 1 if event in _CONTAINER_START else 0
                while nested_depth:
                    _prefix, event, value = _next_event(events)
                    builder.event(event, value)
                    if event in _CONTAINER_START:
                        nested_depth += 1
                    elif event in _CONTAINER_END:
                        nested_depth -= 1
                continue
            builder.event(event, value)
            if event in _CONTAINER_START:
                depth += 1
            elif event in _CONTAINER_END:
                depth -= 1
                if depth == 0:
                    break
        on_member(index, cast(JsonValue, normalize_ijson_stdlib_numbers(builder.value)), count)


_OTLP_SCOPE_FIELDS = ("scopeSpans", "instrumentationLibrarySpans")


def _walk_otlp_scopes(
    events: Iterator[tuple[str, str, object]],
    resource: int,
    field_name: str,
    *,
    on_scope: Callable[[int, str, int, dict[str, object]], None],
    on_span: Callable[[int, str, int, int, dict[str, object]], None],
) -> bool:
    """Walk one scope array whose ``start_array`` has been read."""
    scope = -1
    while True:
        _prefix, event, value = _next_event(events)
        if event == "end_array":
            return True
        scope += 1
        if event != "start_map":
            _skip_json_subtree(events, event)
            continue
        schema_fields: dict[str, object] = {}
        seen_spans = False
        while True:
            _prefix, event, key = _next_event(events)
            if event == "end_map":
                break
            _prefix, event, value = _next_event(events)
            if key == "spans":
                if seen_spans:
                    return False
                seen_spans = True
                if event != "start_array":
                    _skip_json_subtree(events, event)
                    continue
                span = -1
                while True:
                    _prefix, event, value = _next_event(events)
                    if event == "end_array":
                        break
                    span += 1
                    if event == "start_map":
                        decoded = _json_subtree(events, event, value)
                        assert isinstance(decoded, dict)
                        on_span(resource, field_name, scope, span, decoded)
                    else:
                        _skip_json_subtree(events, event)
            elif key in {"schemaUrl", "schema_url"}:
                schema_fields[str(key)] = _json_subtree(events, event, value)
            else:
                _skip_json_subtree(events, event)
        on_scope(resource, field_name, scope, schema_fields)


def spill_otlp_spans(
    handle: JsonReadable,
    root_key: str,
    *,
    on_resource: Callable[[int, dict[str, object], str | None], None],
    on_scope: Callable[[int, str, int, dict[str, object]], None],
    on_span: Callable[[int, str, int, int, dict[str, object]], None],
) -> bool:
    """Walk an OTLP-JSON export one span at a time.

    ``root_key`` names the proved root ``resourceSpans`` array. For each
    object entry, ``on_span`` receives every object span of both scope arrays
    as ``(resource, scope field, scope, span, span)``; ``on_scope`` receives
    each object scope's last ``schemaUrl``/``schema_url`` values;
    ``on_resource`` receives the entry's last ``resource`` value and the scope
    field the parser reads (``scopeSpans`` when present, else
    ``instrumentationLibrarySpans``), or ``None`` when that value is not an
    array. Non-object entries, scopes and spans carry no span and are skipped.

    The walk follows the document's structure rather than ijson's dotted
    prefixes, so a key containing a dot cannot pose as a nested path. Returns
    ``False`` when a resource repeats a scope field or a scope repeats
    ``spans``, since the decoder would keep only the last value.
    """
    events = iter(ijson.parse(handle))
    _enter_root_array(events, root_key)
    resource = -1
    while True:
        _prefix, event, value = _next_event(events)
        if event == "end_array":
            return True
        resource += 1
        if event != "start_map":
            _skip_json_subtree(events, event)
            continue
        fields: dict[str, object] = {}
        scope_arrays: dict[str, bool] = {}
        while True:
            _prefix, event, key = _next_event(events)
            if event == "end_map":
                break
            _prefix, event, value = _next_event(events)
            if key in _OTLP_SCOPE_FIELDS:
                field_name = str(key)
                if field_name in scope_arrays:
                    return False
                scope_arrays[field_name] = event == "start_array"
                if event != "start_array":
                    _skip_json_subtree(events, event)
                elif not _walk_otlp_scopes(events, resource, field_name, on_scope=on_scope, on_span=on_span):
                    return False
            elif key == "resource":
                fields["resource"] = _json_subtree(events, event, value)
            else:
                _skip_json_subtree(events, event)
        selected = next((name for name in _OTLP_SCOPE_FIELDS if name in scope_arrays), None)
        on_resource(resource, fields, selected if selected is not None and scope_arrays[selected] else None)


_Member = TypeVar("_Member")

#: Entries kept from each container field of a taxonomy witness member.
_WITNESS_WIDTH = 64


def _record_container_members(
    events: Iterator[tuple[str, str, object]],
    container: str,
    on_member: Callable[[str, object], _Member],
) -> Generator[_Member, None, bool]:
    """Hand each member of the root array or root ``sessions`` array to ``on_member``.

    ``on_member(event, value)`` receives a member's first event, consumes
    the rest of that member from ``events``, and its result is yielded. The
    walk follows the document's structure, so a dotted key cannot pose as
    the container path. Returns ``False`` for a root object that repeats
    ``sessions`` (the decoder keeps only its last value, while this walk
    would already have handed over the earlier one) or has none. The whole
    input is read, so trailing garbage raises.
    """
    _prefix, event, _value = _next_event(events)
    if container == "item":
        if event != "start_array":
            raise ValueError("JSON document is not an array")
        while True:
            _prefix, event, value = _next_event(events)
            if event == "end_array":
                break
            yield on_member(event, value)
    else:
        if event != "start_map":
            raise ValueError("JSON document is not an object")
        seen_sessions = False
        while True:
            _prefix, event, name = _next_event(events)
            if event == "end_map":
                break
            _prefix, event, value = _next_event(events)
            if name != "sessions":
                _skip_json_subtree(events, event)
                continue
            if seen_sessions or event != "start_array":
                return False
            seen_sessions = True
            while True:
                _prefix, event, value = _next_event(events)
                if event == "end_array":
                    break
                yield on_member(event, value)
        if not seen_sessions:
            return False
    for _event in events:
        pass
    return True


def _witness_subtree(
    events: Iterator[tuple[str, str, object]], event: str, value: object, *, levels: int = 1
) -> object:
    """Decode one value with its containers cut to their first 64 entries, ``levels`` deep.

    Entries past the cut are skipped unread; below ``levels`` every kept
    entry is whole. A repeated key keeps its first position and last value,
    as the decoder's dict does.
    """
    if event not in _CONTAINER_START:
        return value
    result: dict[str, object] | list[object] = {} if event == "start_map" else []
    entries = 0
    while True:
        _prefix, child_event, child_value = _next_event(events)
        if child_event in _CONTAINER_END:
            return result
        key: str | None = None
        if isinstance(result, dict):
            key = str(child_value)
            _prefix, child_event, child_value = _next_event(events)
        if entries >= _WITNESS_WIDTH:
            _skip_json_subtree(events, child_event)
        else:
            if levels > 1:
                entry = _witness_subtree(events, child_event, child_value, levels=levels - 1)
            else:
                builder = ijson.common.ObjectBuilder()
                _build_subtree(events, builder, child_event, child_value)
                entry = builder.value
            if isinstance(result, dict):
                assert key is not None
                result[key] = entry
            else:
                result.append(entry)
        entries += 1


def _build_subtree(
    events: Iterator[tuple[str, str, object]], builder: ijson.common.ObjectBuilder, event: str, value: object
) -> None:
    builder.event(event, value)
    depth = 1 if event in _CONTAINER_START else 0
    while depth:
        _prefix, event, value = _next_event(events)
        builder.event(event, value)
        if event in _CONTAINER_START:
            depth += 1
        elif event in _CONTAINER_END:
            depth -= 1


def _shape_value(event: str, value: object) -> JsonValue:
    if event == "start_map":
        return {}
    if event == "start_array":
        return []
    return cast(JsonValue, value)


def scan_container_members(
    handle: JsonReadable,
    container: str,
    *,
    shape_keys: frozenset[str],
    witnesses: int,
    on_member: Callable[[int, JsonValue, JsonValue | None], None],
) -> int | None:
    """Walk a record container once without decoding any whole member.

    ``container`` is ``json_record_container``'s answer. For each member,
    ``on_member(index, shape, witness)`` receives its ``shape`` -- the
    member's ``shape_keys`` fields, each a scalar or an empty container of
    its type, the last repeated key winning as in the decoder; a non-object
    member is reduced the same way -- and, for the first ``witnesses``
    members, a taxonomy witness: the member with each container field cut to
    its first 64 entries, every kept entry whole. Numbers keep ijson's
    ``Decimal``, as the container's decoded members do.

    Returns the member count, or ``None`` when a repeated root ``sessions``
    key leaves the decoder a different container than a stream would read.
    Invalid JSON anywhere, including a truncated suffix, raises.
    """
    events = iter(ijson.parse(handle))
    index = -1

    def member(event: str, value: object) -> None:
        nonlocal index
        index += 1
        witness: JsonValue | None = None
        if event != "start_map":
            if index < witnesses:
                witness = cast(JsonValue, _witness_subtree(events, event, value))
            else:
                _skip_json_subtree(events, event)
            on_member(index, _shape_value(event, value), witness)
            return
        shape: dict[str, JsonValue] = {}
        fields: dict[str, object] | None = {} if index < witnesses else None
        while True:
            _prefix, key_event, key = _next_event(events)
            if key_event == "end_map":
                break
            _prefix, field_event, field_value = _next_event(events)
            if key in shape_keys:
                shape[str(key)] = _shape_value(field_event, field_value)
            if fields is not None:
                fields[str(key)] = _witness_subtree(events, field_event, field_value)
            else:
                _skip_json_subtree(events, field_event)
        on_member(index, shape, cast(JsonValue, fields) if fields is not None else None)

    walk = _record_container_members(events, container, member)
    try:
        while True:
            next(walk)
    except StopIteration as stop:
        if not stop.value:
            return None
    finally:
        handle.seek(0)
    return index + 1


def _json_text(value: object) -> bytes:
    if isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            return json.dumps(value).encode("ascii")
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, Decimal) and value == value.to_integral_value():
        # The bundle lowering reads an integral number as an ``int``
        # (``normalize_json_decimal``); written as one, it reads back as one.
        return str(int(value)).encode("ascii")
    return str(value).encode("ascii")


def iter_container_member_files(
    handle: JsonReadable, container: str, member_path: Path
) -> Generator[int | None, None, None]:
    """Write each object member of a record container to ``member_path`` in turn.

    Yields the member's index after its complete JSON text is on disk, or
    ``None`` for a member that is not an object: no bundle lowering reads
    one. Only one scalar is resident at a time, so a member of any size is
    handed to the single-object streaming routes. The caller reads the file
    before resuming the iterator, which overwrites it.
    """
    events = iter(ijson.parse(handle))
    index = -1

    def member(event: str, value: object) -> int | None:
        nonlocal index
        index += 1
        if event != "start_map":
            _skip_json_subtree(events, event)
            return None
        with member_path.open("wb") as output:
            write = output.write
            # One entry per open container: whether it already holds a value.
            filled: list[bool] = []
            while True:
                if event in _CONTAINER_START:
                    if filled and filled[-1]:
                        write(b",")
                    if filled:
                        filled[-1] = True
                    write(b"{" if event == "start_map" else b"[")
                    filled.append(False)
                elif event in _CONTAINER_END:
                    filled.pop()
                    write(b"}" if event == "end_map" else b"]")
                    if not filled:
                        break
                elif event == "map_key":
                    if filled[-1]:
                        write(b",")
                    write(_json_text(value))
                    write(b":")
                    # The key's value supplies no separator of its own.
                    filled[-1] = False
                else:
                    if filled[-1]:
                        write(b",")
                    filled[-1] = True
                    write(_json_text(value))
                _prefix, event, value = _next_event(events)
        return index

    if not (yield from _record_container_members(events, container, member)):
        raise ValueError("JSON record container changed during preparation")


def iter_root_array_items(handle: JsonReadable, key: str) -> Iterator[JsonValue]:
    """Yield each item of the root object's ``key`` array, one decoded at a time.

    The walk follows the document's structure and stops at the first
    ``key``; callers have proved it is the only one.
    """
    events = iter(ijson.parse(handle))
    _enter_root_array(events, key)
    while True:
        _prefix, event, value = _next_event(events)
        if event == "end_array":
            return
        yield cast(JsonValue, _json_subtree(events, event, value))


def iter_json_container_records(handle: JsonReadable, prefix: str) -> Iterable[JsonValue]:
    """Yield complete array members; a corrupt suffix raises after its prefix.

    Members keep ijson's ``Decimal`` numbers for the bundle parsers. The
    storable-value limit is not applied to decoded members: a field the
    parser ignores never reaches storage, and every value that does is
    bounded, typed, where it is written (``prepared_message_sink._write_row``).
    """
    yield from ijson.items(handle, prefix)


__all__ = [
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
    "grok_taxonomy_witness",
    "iter_container_member_files",
    "iter_json_container_records",
    "iter_root_array_items",
    "json_record_container",
    "scan_container_members",
    "spill_member_arrays",
    "spill_otlp_spans",
]
