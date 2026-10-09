"""Bounded lexical JSON transport and physical-line grammar helpers."""

from __future__ import annotations

import codecs
import re
from collections.abc import Callable
from typing import IO, Protocol

#: Characters of a top-level string an envelope keeps. The source-class
#: signatures read only the type and short leading text of root fields.
ENVELOPE_TEXT_PREFIX_CHARS = 4096

#: Raw bytes of any string token passed to the tokenizer. Twelve raw bytes
#: can encode one character (a surrogate-pair escape), so this keeps at least
#: :data:`ENVELOPE_TEXT_PREFIX_CHARS` characters of every string.
_STRING_PREFIX_BYTES = 12 * ENVELOPE_TEXT_PREFIX_CHARS

_READ_BYTES = 1024 * 1024

# Strategy-only buffering window shared by physical-record consumers. A
# longer record continues through disk transport without an input refusal.
JSONL_MEMORY_BUFFER_BYTES = 64 * 1024

_ESCAPE_TOKEN = re.compile(rb'\\(?:u([0-9a-fA-F]{4})|["\\/bfnrt])')

#: Tokens of a skipped string suffix worth examining: complete valid escapes
#: pass; a lone backslash (an invalid or not-yet-complete escape) or a raw
#: control character is not valid JSON string content.
_SKIPPED_TOKEN = re.compile(rb'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])|\\|[\x00-\x1f]')

#: Injected into the tokenizer's view of a string whose skipped suffix is not
#: valid JSON, so the tokenizer rejects the document as the full decoder does.
_INVALID_ESCAPE = b"\\q"

#: A run of bytes that can belong to a JSON number token outside strings.
_NUMBER_RUN = re.compile(rb"[-+0-9.eE]+")
_EXACT_NUMBER_RUN = re.compile(rb"-Infinity|Infinity|NaN|[-0-9][-+0-9.eE]*")
_INVALID_STRUCTURE_BYTE = re.compile(rb"[^\x09\x0a\x0d\x20-\x7e]")

#: Passed in place of an over-long integer token so the tokenizer rejects it
#: as malformed instead of converting it.
_INVALID_NUMBER_END = b"x"

#: The non-finite constants the production decoder (stdlib ``json``) accepts
#: and the tokenizer rejects. Each reaches the tokenizer as a float
#: placeholder padded with spaces, so a constant glued to another token stays
#: as malformed as the decoder finds it.
_NON_FINITE = re.compile(rb"-Infinity|Infinity|NaN")
_NON_FINITE_PLACEHOLDER = b" 0.0 "

#: A leading part of a non-finite constant at the end of a chunk, held back
#: until the next chunk completes or ends it.
_NON_FINITE_PARTIAL = re.compile(rb"(?:-|-?I|-?In|-?Inf|-?Infi|-?Infin|-?Infini|-?Infinit|N|Na)\Z")

#: Marks the end of a number token's integer part.
_NUMBER_NON_INTEGER = re.compile(rb"[.eE]")

#: Raw bytes kept by grammar-only transport before using a typed stand-in.
#: Selected projection readers use exact owned scalar sinks instead.
_NUMBER_VIEW_BYTES = 8192

#: Significant exponent digits of a number token the tokenizer sees exactly.
#: ``Decimal`` refuses an exponent beyond about 10**18 that the JSON decoder
#: accepts as an overflowing float; a token whose exponent has more
#: significant digits than this is passed as a placeholder, so the exponent
#: of any token passed exactly (at most :data:`_NUMBER_VIEW_BYTES` digits of
#: significand) stays inside ``Decimal``'s range.
_NUMBER_EXPONENT_DIGITS = 17

#: A UTF-16 surrogate code unit encoded directly as three UTF-8 bytes. The
#: production decoder keeps these (``surrogatepass``) in historical provider
#: data; the tokenizer rejects them, so string content reaches it with each
#: one replaced by U+FFFD, which has the same encoded length.
_SURROGATE_BYTES = re.compile(rb"\xed[\xa0-\xbf][\x80-\xbf]")
_SURROGATE_STAND_IN = "\ufffd".encode()

#: RFC 8259 number grammar as a byte-driven automaton, so a token too long to
#: pass exactly is still checked in full before it is replaced by a
#: placeholder. States: 0 start, 1 sign, 2 leading zero, 3 integer digits,
#: 4 point, 5 fraction digits, 6 exponent mark, 7 exponent sign, 8 exponent
#: digits; -1 is invalid.
_NUMBER_ACCEPTING = frozenset({2, 3, 5, 8})
_DIGITS = frozenset(b"0123456789")


def _number_step(state: int, byte: int) -> int:
    if state == 0:
        return 1 if byte == 0x2D else 2 if byte == 0x30 else 3 if byte in _DIGITS else -1
    if state == 1:
        return 2 if byte == 0x30 else 3 if byte in _DIGITS else -1
    if state in (2, 3):
        if byte in _DIGITS:
            return 3 if state == 3 else -1
        return 4 if byte == 0x2E else 6 if byte in (0x65, 0x45) else -1
    if state in (4, 5):
        if byte in _DIGITS:
            return 5
        return 6 if state == 5 and byte in (0x65, 0x45) else -1
    if state == 6:
        return 7 if byte in (0x2B, 0x2D) else 8 if byte in _DIGITS else -1
    if state in (7, 8):
        return 8 if byte in _DIGITS else -1
    return -1


class _EscapeSavings:
    """Raw bytes a string token's escapes save when decoded to UTF-8.

    The raw length of a token minus this is the UTF-8 length of the decoded
    value: ``\\n`` decodes to one byte, ``\\u00e9`` to two, and an adjacent
    escaped surrogate pair to one four-byte character.
    """

    def __init__(self) -> None:
        self.saved = 0
        self._high_surrogate = False

    def escape(self, token: bytes, *, adjacent: bool) -> None:
        if token[1:2] != b"u":
            self.saved += 1
            self._high_surrogate = False
            return
        unit = int(token[2:6], 16)
        if 0xDC00 <= unit <= 0xDFFF and adjacent and self._high_surrogate:
            # The pair decodes to four bytes; its high half was counted as three.
            self.saved += 6 - 1
            self._high_surrogate = False
            return
        self.saved += 6 - (1 if unit < 0x80 else 2 if unit < 0x800 else 3)
        self._high_surrogate = 0xD800 <= unit <= 0xDBFF


class _Readable(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


def _prefix_cut(content: bytes) -> int:
    """Largest cut of raw string content that splits no escape, pair or UTF-8 sequence."""
    tokens = list(_ESCAPE_TOKEN.finditer(content))
    last_end = tokens[-1].end() if tokens else 0
    cut = len(content)
    dangling = content.find(b"\\", last_end)
    if dangling >= 0:
        cut = dangling
    for token in reversed(tokens):
        if token.end() < cut:
            break
        unit = token.group(1)
        if token.end() == cut and unit is not None and 0xD800 <= int(unit, 16) <= 0xDBFF:
            cut = token.start()
            break
    index = cut - 1
    while index >= 0 and cut - index <= 4 and 0x80 <= content[index] <= 0xBF:
        index -= 1
    if index >= 0:
        lead = content[index]
        width = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
        if index + width > cut:
            cut = index
    return cut


class _PrefixStringReader:
    """Pass a JSON byte stream through, keeping only a prefix of each string token.

    Quote parity decides string boundaries. A string longer than
    :data:`_STRING_PREFIX_BYTES` raw bytes reaches the tokenizer cut at a
    boundary that splits no escape or character; the rest is skipped unread
    by the tokenizer. The root fields a signature reads keep their full
    leading text. ``syntax_only`` validates grammar without retaining token
    truncation metadata or decoding large integer values; it cannot supply
    record fields or identity evidence. Optional scalar sinks retain exact
    token chunks by ordinal while the tokenizer receives a bounded view;
    callers must restore semantic values from that owned transport.
    """

    def __init__(
        self,
        source: _Readable,
        *,
        scalar_values: bool = False,
        syntax_only: bool = False,
        string_sink: Callable[[int, bytes, bool], None] | None = None,
        number_sink: Callable[[int, bytes, bool], None] | None = None,
        allow_nonfinite: bool = True,
    ) -> None:
        self._source = source
        self._string_sink = string_sink
        self._number_sink = number_sink
        self._allow_nonfinite = allow_nonfinite
        self._number_ordinal = 0
        self._scalar_values = scalar_values
        self._syntax_only = syntax_only
        self._in_string = False
        self._backslashes = 0
        self._string = bytearray()
        self._skipping = False
        self._eof = False
        #: Ordinal of the current string token, counted from 1 in stream order;
        #: the tokenizer reports keys and string values in the same order.
        self._ordinal = 0
        #: Decoded UTF-8 length of each string token passed on as a prefix only.
        self.truncated: dict[int, int] = {}
        self._skip_savings = _EscapeSavings()
        #: Whether the last suffix bytes examined ended with a complete escape.
        self._skip_escape_at_end = False
        #: Ordinals of string tokens whose passed content had a surrogate
        #: code unit replaced by a stand-in.
        self.substituted: set[int] = set()
        self._skipped_bytes = 0
        self._skip_carry = b""
        self._skip_invalid = False
        #: State of the number token being passed outside strings, which may
        #: span chunks.
        self._number_open = False
        self._number_view = bytearray()
        self._number_long = False
        self._number_is_integer = True
        self._number_in_integer_part = True
        self._skip_decoder: codecs.IncrementalDecoder | None = None
        #: Structure bytes held back at a chunk end (see ``_NON_FINITE_PARTIAL``).
        self._structure_carry = b""

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            # The tokenizer probes with an empty read to learn the stream type.
            return b""
        out = bytearray()
        while not out and not self._eof:
            chunk = self._source.read(_READ_BYTES)
            if not chunk:
                out += self.finish()
                break
            out += self.feed(chunk)
        return bytes(out)

    def feed(self, chunk: bytes) -> bytes:
        """Feed a push reader through the same bounded lexical transport."""
        if self._eof:
            raise RuntimeError("JSON lexical transport is finished")
        out = bytearray()
        self._consume(chunk, out)
        return bytes(out)

    def finish(self) -> bytes:
        """Flush lexical EOF once, preserving incomplete-token failures."""
        if self._eof:
            return b""
        self._eof = True
        out = bytearray()
        if self._in_string and not self._skipping:
            out += self._string
        else:
            if self._structure_carry:
                self._pass_structure(b"", out)
            if self._number_open:
                self._end_number(out)
        return bytes(out)

    def _pass_structure(self, segment: bytes, out: bytearray, *, chunk_end: bool = False) -> None:
        """Pass bytes outside any string through, number tokens bounded.

        Each number token is held back until it ends. A short token reaches
        the tokenizer exactly. An integer longer than Python's conversion
        limit, which the JSON decoder refuses with ``ValueError``, reaches it
        as a malformed token, so the record is rejected as the decoder rejects
        it and the C tokenizer never converts the value. Any other token
        longer than :data:`_NUMBER_VIEW_BYTES`, or with an exponent beyond
        ``Decimal``'s range, reaches it as a placeholder of the same JSON
        type: an envelope is a view of presence and type, and a signature
        never reads a number's value, so a number of any length costs bounded
        memory. ``NaN`` and the infinities, which the decoder accepts, reach
        it as a float placeholder.
        """
        segment = self._structure_carry + segment
        self._structure_carry = b""
        # The Python event lexer accepts Unicode whitespace; JSON permits
        # only space, tab, CR and LF outside strings. Any non-ASCII byte here
        # is likewise invalid JSON syntax, including split UTF-8 whitespace.
        segment = _INVALID_STRUCTURE_BYTE.sub(b"!", segment)
        if chunk_end and (partial := _NON_FINITE_PARTIAL.search(segment)) is not None:
            self._structure_carry = segment[partial.start() :]
            segment = segment[: partial.start()]
        # Detector projections use numeric type, truth, and finite literal
        # equality. A non-finite placeholder must remain truthy and cannot
        # become a finite discriminator such as schema_version=1.
        if self._number_sink is None:
            segment = _NON_FINITE.sub(b" 1e999 " if self._scalar_values else _NON_FINITE_PLACEHOLDER, segment)
        position = 0
        number_runs = _EXACT_NUMBER_RUN if self._number_sink is not None else _NUMBER_RUN
        runs = number_runs.finditer(segment)
        if self._number_sink is not None and self._number_open and (continuation := _NUMBER_RUN.match(segment)):
            from itertools import chain

            runs = chain((continuation,), number_runs.finditer(segment, continuation.end()))
        for run in runs:
            if self._number_open and run.start() > 0:
                self._end_number(out)
            out += segment[position : run.start()]
            if not self._number_open:
                self._start_number()
            self._extend_number(run.group())
            position = run.end()
            if run.end() < len(segment):
                self._end_number(out)
        if position < len(segment):
            if self._number_open:
                self._end_number(out)
            out += segment[position:]

    def _start_number(self) -> None:
        self._number_ordinal += 1
        self._number_open = True
        self._number_view = bytearray()
        self._number_long = False
        self._number_is_integer = True
        self._number_in_integer_part = True
        self._number_exponent_digits = 0
        self._number_state = 0

    def _extend_number(self, token: bytes) -> None:
        if self._number_sink is not None:
            self._number_sink(self._number_ordinal, token, False)
        state = self._number_state
        if state != -1:
            for byte in token:
                state = _number_step(state, byte)
                if state == -1:
                    break
                if state == 8 and (byte != 0x30 or self._number_exponent_digits):
                    self._number_exponent_digits += 1
            self._number_state = state
        if self._number_in_integer_part:
            mark = _NUMBER_NON_INTEGER.search(token)
            if mark is not None:
                self._number_in_integer_part = False
                self._number_is_integer = False
        if not self._number_long:
            self._number_view += token
            if len(self._number_view) > _NUMBER_VIEW_BYTES and (
                not self._scalar_values or self._number_sink is not None
            ):
                self._number_long = True
                self._number_view = bytearray()

    def _end_number(self, out: bytearray) -> None:
        self._number_open = False
        oversized = self._number_long or self._number_exponent_digits > _NUMBER_EXPONENT_DIGITS
        if self._number_sink is not None:
            self._number_sink(self._number_ordinal, b"", True)
            nonfinite = bytes(self._number_view) in {b"NaN", b"Infinity", b"-Infinity"}
            if nonfinite:
                out += b"0.0" if self._allow_nonfinite else _INVALID_NUMBER_END
            elif self._number_state not in _NUMBER_ACCEPTING:
                out += _INVALID_NUMBER_END
            else:
                out += b"0" if self._number_is_integer else b"0.0"
            self._number_view = bytearray()
            return
        if self._scalar_values:
            token = bytes(self._number_view)
            if not self._number_is_integer and self._number_state in _NUMBER_ACCEPTING:
                # Decimal's exponent range is smaller than json.loads' float
                # range. Normalize one complete float token before handing it
                # to the event tokenizer, preserving zero and overflow.
                number = float(token)
                token = (
                    b"-1e999"
                    if number == float("-inf")
                    else b"1e999"
                    if number == float("inf")
                    else repr(number).encode("ascii")
                )
            out += token
        elif self._syntax_only:
            out += b"0" if self._number_state in _NUMBER_ACCEPTING else _INVALID_NUMBER_END
        elif oversized and self._number_state not in _NUMBER_ACCEPTING:
            # A malformed long token stays malformed: the placeholder must not
            # turn a document the decoder rejects into one it would accept.
            out += _INVALID_NUMBER_END
        elif oversized:
            out += b"0" if self._number_is_integer else b"0.0"
        else:
            out += self._number_view
        self._number_view = bytearray()

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def _consume(self, data: bytes, out: bytearray) -> None:
        position = 0
        while position < len(data):
            if not self._in_string:
                quote = data.find(b'"', position)
                if quote < 0:
                    self._pass_structure(data[position:], out, chunk_end=True)
                    return
                self._pass_structure(data[position : quote + 1], out)
                self._in_string = True
                self._backslashes = 0
                self._string = bytearray()
                self._skipping = False
                self._ordinal += 1
                position = quote + 1
                continue
            end = self._string_end(data, position)
            piece = data[position:end] if end >= 0 else data[position:]
            if self._string_sink is not None:
                self._string_sink(self._ordinal, piece, end >= 0)
            if self._skipping:
                self._skipped_bytes += len(piece)
                self._validate_skipped(piece, out)
            else:
                self._string += piece
                if len(self._string) > _STRING_PREFIX_BYTES and (
                    not self._scalar_values or self._string_sink is not None
                ):
                    cut = _prefix_cut(bytes(self._string[:_STRING_PREFIX_BYTES]))
                    self._emit_string(bytes(self._string[:cut]), out)
                    self._skipped_bytes = len(self._string)
                    self._skip_savings = _EscapeSavings()
                    last_end = -1
                    for token in _ESCAPE_TOKEN.finditer(bytes(self._string[:cut])):
                        self._skip_savings.escape(token.group(), adjacent=token.start() == last_end)
                        last_end = token.end()
                    # The cut never separates the halves of a surrogate pair.
                    self._skip_escape_at_end = False
                    self._skip_carry = b""
                    self._skip_invalid = False
                    self._skip_decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogatepass")
                    self._validate_skipped(bytes(self._string[cut:]), out)
                    self._string = bytearray()
                    self._skipping = True
            if end < 0:
                return
            if self._skipping:
                if not self._skip_invalid and self._skip_carry:
                    # An escape left incomplete by the closing quote.
                    self._skip_invalid = True
                    out += _INVALID_ESCAPE
                if not self._skip_invalid and self._skip_decoder is not None:
                    try:
                        self._skip_decoder.decode(b"", final=True)
                    except UnicodeDecodeError:
                        # A UTF-8 sequence left incomplete by the closing quote.
                        self._skip_invalid = True
                        out += _INVALID_ESCAPE
                if not self._syntax_only and self._string_sink is None:
                    self.truncated[self._ordinal] = self._skipped_bytes - self._skip_savings.saved
            else:
                self._emit_string(bytes(self._string), out)
            out += b'"'
            self._in_string = False
            self._string = bytearray()
            position = end + 1

    def _emit_string(self, content: bytes, out: bytearray) -> None:
        if self._scalar_values and self._string_sink is None:
            out += content
            return
        replaced, count = _SURROGATE_BYTES.subn(_SURROGATE_STAND_IN, content)
        if count and not self._syntax_only and self._string_sink is None:
            self.substituted.add(self._ordinal)
        out += replaced

    def _validate_skipped(self, piece: bytes, out: bytearray) -> None:
        """Check that a skipped string suffix is valid JSON string content.

        The tokenizer never sees the suffix, so an invalid UTF-8 sequence
        (a directly encoded surrogate code unit is valid, as the production
        decoder's ``surrogatepass`` retry keeps it), invalid escape or raw
        control character there is injected as an invalid escape: the
        document is then rejected exactly as the full decoder rejects it.
        Valid escapes are counted toward the decoded length.
        """
        if self._skip_invalid:
            return
        if self._skip_decoder is not None:
            try:
                self._skip_decoder.decode(piece)
            except UnicodeDecodeError:
                self._skip_invalid = True
                out += _INVALID_ESCAPE
                return
        buffer = self._skip_carry + piece
        self._skip_carry = b""
        last_end = 0 if self._skip_escape_at_end else -1
        for match in _SKIPPED_TOKEN.finditer(buffer):
            token = match.group()
            if len(token) > 1:
                self._skip_savings.escape(token, adjacent=match.start() == last_end)
                last_end = match.end()
                continue
            if token == b"\\" and len(buffer) - match.start() < 6:
                # Possibly an escape the next chunk completes.
                self._skip_carry = buffer[match.start() :]
                self._skip_escape_at_end = match.start() == last_end
                return
            self._skip_invalid = True
            out += _INVALID_ESCAPE
            return
        self._skip_escape_at_end = last_end == len(buffer)

    def _string_end(self, data: bytes, start: int) -> int:
        """Index of the closing quote of the open string in ``data``, or -1."""
        search = start
        while (quote := data.find(b'"', search)) >= 0:
            run = 0
            index = quote - 1
            while index >= start and data[index] == 0x5C:
                run += 1
                index -= 1
            if index < start:
                run += self._backslashes
            if run % 2 == 0:
                return quote
            search = quote + 1
        segment = data[start:]
        tail = len(segment) - len(segment.rstrip(b"\\"))
        self._backslashes = tail + (self._backslashes if tail == len(segment) else 0)
        return -1


class _LineSource:
    """Physical lines of a byte stream, each readable as its own stream."""

    def __init__(self, handle: _Readable) -> None:
        self._handle = handle
        self._buffer = b""
        self._eof = False

    def next_line(self) -> _Line | None:
        if not self._buffer and not self._eof:
            self._fill()
        if not self._buffer and self._eof:
            return None
        return _Line(self)

    def strip_leading_byte_order_marks(self) -> None:
        while True:
            while len(self._buffer) < len(codecs.BOM_UTF8) and not self._eof:
                self._fill()
            if not self._buffer.startswith(codecs.BOM_UTF8):
                return
            self._buffer = self._buffer[len(codecs.BOM_UTF8) :]

    def _fill(self) -> None:
        chunk = self._handle.read(_READ_BYTES)
        if chunk:
            self._buffer += chunk
        else:
            self._eof = True


class _Line:
    """One physical line; reads stop at its newline.

    ``size`` counts the line's bytes read so far and ``decodable`` says
    whether they are provider UTF-8 (``decode_provider_utf8``'s rule), as the
    JSONL decoder judges a line before reading it as JSON.
    """

    def __init__(self, source: _LineSource) -> None:
        self._source = source
        self._done = False
        self.size = 0
        self.decodable = True
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogatepass")

    def read(self, size: int = -1) -> bytes:
        if self._done or size == 0:
            return b""
        source = self._source
        if not source._buffer and not source._eof:
            source._fill()
        if not source._buffer:
            self._finish(b"")
            return b""
        newline = source._buffer.find(b"\n")
        if newline >= 0:
            data, source._buffer = source._buffer[:newline], source._buffer[newline + 1 :]
            self._finish(data)
            return data
        data, source._buffer = source._buffer, b""
        self._observe(data, final=False)
        return data

    def _finish(self, data: bytes) -> None:
        self._done = True
        self._observe(data, final=True)

    def _observe(self, data: bytes, *, final: bool) -> None:
        self.size += len(data)
        if self.decodable:
            try:
                self._decoder.decode(data, final=final)
            except UnicodeDecodeError:
                self.decodable = False

    def drain(self) -> None:
        while self.read(_READ_BYTES):
            pass


def jsonl_has_record_successor(handle: IO[bytes], *, check_stop: Callable[[], None] | None = None) -> bool:
    """Prove a complete first physical value has later nonblank line bytes.

    This chooses a grammar attempt, never admission. Syntax-only tokenization
    retains no record values; a single or multiline document stays on its
    existing document route. The borrowed stream's position is restored.
    """
    import ijson
    from ijson.backends import python as exact_backend

    from polylogue.core.compute_cancel import check_compute_cancelled

    position = handle.tell()
    callback_failure: BaseException | None = None

    class ObservedLine:
        def __init__(self, line: _Line) -> None:
            self.line = line
            self.nonblank = False

        def read(self, size: int = -1) -> bytes:
            nonlocal callback_failure
            check_compute_cancelled()
            if check_stop is not None:
                try:
                    check_stop()
                except BaseException as exc:
                    callback_failure = exc
                    raise
            data = self.line.read(size)
            self.nonblank |= bool(data.strip(b" \t\r\n"))
            return data

    try:
        lines = _LineSource(handle)
        while True:
            lines.strip_leading_byte_order_marks()
            line = lines.next_line()
            if line is None:
                return False
            observed = ObservedLine(line)
            try:
                events = iter(
                    exact_backend.basic_parse(
                        LexemeAlignedReader(_PrefixStringReader(observed, syntax_only=True)), use_float=True
                    )
                )
                first = next(events, None)
                for _event in events:
                    pass
            except (UnicodeError, ijson.JSONError, ValueError):
                if callback_failure is not None:
                    raise callback_failure from None
                if observed.nonblank:
                    return False
                continue
            if first is not None:
                break
        if not line.decodable:
            return False
        while (line := lines.next_line()) is not None:
            observed = ObservedLine(line)
            while observed.read(_READ_BYTES):
                if observed.nonblank:
                    return True
        return False
    finally:
        handle.seek(position)


#: Set in an envelope whose object had root keys outside the declared fields,
#: so a caller can still tell an empty object from one with other content,
#: without the envelope holding those keys.
UNDECLARED_FIELDS = "\x00undeclared"


_LEXEME_END_BYTES = b"[]{}, \t\r\n"


class LexemeAlignedReader:
    """Pass JSON bytes through in chunks that each end where a lexeme has ended.

    ``ijson.backends.python`` is the exact tokenizer (it keeps lone surrogate
    escapes that the C tokenizer refuses), but its lexer appends every chunk
    to its buffer while a lexeme spans a chunk boundary and replaces the
    buffer only once a chunk is consumed with no lexeme left over. Fed
    arbitrary chunks, that buffer keeps the whole consumed document. Each
    chunk this reader returns ends right after a closing quote, a structural
    character or whitespace outside any string, so the lexer finishes every
    chunk and starts the next with a fresh buffer. Only the chunking changes:
    the bytes, and so the lexemes and events, are exactly the source's.

    A source run with no lexeme end at all (one token longer than
    ``flush_bytes``, which a valid JSON document passed through
    :class:`_PrefixStringReader` cannot contain) is passed through as it is.
    """

    def __init__(self, source: _Readable, *, flush_bytes: int = 4 * _READ_BYTES) -> None:
        self._source = source
        self._flush_bytes = flush_bytes
        self._pending = bytearray()
        #: Bytes of ``_pending`` whose string state is known.
        self._scanned = 0
        self._in_string = False
        self._eof = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            # The tokenizer probes with an empty read to learn the stream type.
            return b""
        while True:
            cut = self._scan()
            if not cut and self._eof:
                cut = len(self._pending)
            elif not cut and len(self._pending) >= self._flush_bytes:
                # Everything whose string state is known; an escape whose
                # escaped byte has not arrived stays behind.
                cut = self._scanned or len(self._pending)
            if cut or self._eof:
                chunk = bytes(self._pending[:cut])
                del self._pending[:cut]
                self._scanned = max(0, self._scanned - cut)
                return chunk
            data = self._source.read(_READ_BYTES)
            if data:
                self._pending += data
            else:
                self._eof = True

    def _scan(self) -> int:
        """Advance string state over unscanned bytes; return the last lexeme end, or 0."""
        data = self._pending
        size = len(data)
        position = self._scanned
        in_string = self._in_string
        cut = 0
        # The last run of bytes outside any string; only its lexeme ends can
        # follow the last closing quote.
        span = (position, position) if not in_string else (0, 0)
        while position < size:
            if in_string:
                quote = data.find(b'"', position)
                limit = size if quote < 0 else quote
                while (escape := data.find(b"\\", position, limit)) >= 0:
                    if escape + 1 >= size:
                        # The escaped byte has not arrived yet.
                        self._scanned = escape
                        self._in_string = True
                        return max(cut, self._span_cut(span))
                    position = escape + 2
                    if position > limit:
                        # That escape was the quote's: the string goes on.
                        quote = data.find(b'"', position)
                        limit = size if quote < 0 else quote
                if quote < 0:
                    position = size
                    break
                in_string = False
                position = quote + 1
                cut = position
                span = (position, position)
                continue
            quote = data.find(b'"', position)
            end = size if quote < 0 else quote
            span = (position, end)
            if quote < 0:
                position = size
                break
            in_string = True
            position = quote + 1
        self._scanned = position
        self._in_string = in_string
        return max(cut, self._span_cut(span))

    def _span_cut(self, span: tuple[int, int]) -> int:
        start, end = span
        if start >= end:
            return 0
        return max(self._pending.rfind(byte, start, end) for byte in _LEXEME_END_SINGLE_BYTES) + 1


_LEXEME_END_SINGLE_BYTES = tuple(bytes((byte,)) for byte in _LEXEME_END_BYTES)


__all__ = [
    "ENVELOPE_TEXT_PREFIX_CHARS",
    "UNDECLARED_FIELDS",
    "LexemeAlignedReader",
]
