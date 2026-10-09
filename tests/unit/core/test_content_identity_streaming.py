"""Member content identity has one definition for every payload size.

The byte route streams the document (``stream_payload_content_identity``);
the value route walks an already-decoded value
(``structural_content_identity``). Both must name the same content with the
same digest, and no size may switch the definition to a byte digest.
"""

from __future__ import annotations

import io
import json
import unicodedata
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.core.content_identity import (
    _STREAM_READ_BYTES,
    payload_content_identity,
    stream_payload_content_identity,
    structural_content_identity,
)
from polylogue.core.json import loads


def _decoded_identity(payload: bytes) -> str:
    """The identity through the in-memory decoder, or the byte digest if it refuses."""
    try:
        value = loads(payload)
    except Exception:
        return sha256(payload).hexdigest()
    try:
        return structural_content_identity(value)
    except (TypeError, ValueError):
        return sha256(payload).hexdigest()


_json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**30), max_value=10**30)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(alphabet=st.characters(codec="utf-8"), max_size=12),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(alphabet="abéé", max_size=3), children, max_size=5)
    ),
    max_leaves=25,
)


@settings(max_examples=300, deadline=None)
@given(value=_json_values, ensure_ascii=st.booleans(), compact=st.booleans())
def test_streamed_identity_equals_decoded_identity(value: object, ensure_ascii: bool, compact: bool) -> None:
    """Anti-vacuity: any divergence between the stream encoder and the value
    encoder (object entry order, number or text form) fails a generated case."""
    separators = (",", ":") if compact else (", ", ": ")
    payload = json.dumps(value, ensure_ascii=ensure_ascii, separators=separators).encode()
    assert payload_content_identity(payload) == _decoded_identity(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}',
        b"\xef\xbb\xbf" + b'{"a":1}',
        b'"\\ud800"',
        b'"\\ud83d\\ude00"',
        b'"\\\\ud83d\\ude00"',
        b'"\\ude00\\ud83d"',
        b'["\\ud83d", "\\ude00"]',
        b'{"k\\ud800":1}',
        b'{"\\u00e9":1,"e\\u0301":2}',
        b"1e300",
        b"1e400",
        b"18446744073709551616",
        b"-0.0",
        b'{"a":1} {"b":1}',
        b"",
        b"not json",
        b'"\xff"',
    ],
)
def test_edge_cases_match_the_decoder(payload: bytes) -> None:
    """Duplicate keys, byte-order marks, surrogate escapes, number forms and
    non-JSON bytes all resolve exactly as the in-memory decoder resolves them."""
    assert payload_content_identity(payload) == _decoded_identity(payload)


def test_a_lone_surrogate_escape_is_not_confused_with_a_question_mark() -> None:
    """Anti-vacuity: drop the surrogate scan and the C tokenizer decodes the
    lone escape to ``?``, giving two different documents one identity."""
    assert payload_content_identity(b'"\\ud800"') != payload_content_identity(b'"?"')


def test_identity_does_not_switch_method_above_any_size() -> None:
    """A document larger than several read windows keeps its structural
    identity: re-serializing it changes the bytes but not the identity.

    Anti-vacuity: reinstate a size ceiling that substitutes the byte digest and
    the two serializations get different identities.
    """
    value = [{"id": index, "text": "\U0001f600" * 4000, "meta": {"b": 1, "a": [1.5, None]}} for index in range(120)]
    compact = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()
    spaced = json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True).encode()
    assert len(compact) > 3 * _STREAM_READ_BYTES

    streamed_compact = stream_payload_content_identity(io.BytesIO(compact))
    streamed_spaced = stream_payload_content_identity(io.BytesIO(spaced))

    assert streamed_compact == streamed_spaced == structural_content_identity(value)
    assert streamed_compact != sha256(compact).hexdigest()


def test_surrogate_escape_split_across_read_windows_is_seen() -> None:
    """An escape pair and a lone escape straddling a window boundary resolve
    like the decoder does."""
    prefix = b'["' + b"x" * (_STREAM_READ_BYTES - 5)
    paired = prefix + b'\\ud83d\\ude00"]'
    lone = prefix + b'\\ud83dx"]'
    assert payload_content_identity(paired) == _decoded_identity(paired)
    assert payload_content_identity(lone) == _decoded_identity(lone)


_SPILL_CASES = [
    json.dumps({"text": "a" * 300, "k": 1}).encode(),
    json.dumps({"text": 'line\n"q"\\ \t' * 40}).encode(),
    json.dumps({"t": "\U0001f600" * 60}, ensure_ascii=True).encode(),
    json.dumps({"t": "é" * 150}, ensure_ascii=False).encode(),
    json.dumps({"t": "é가" * 60}, ensure_ascii=True).encode(),
    json.dumps({"k" * 300: 1, "a": 2}).encode(),
    json.dumps(["x" * 300, "y", "z" * 300]).encode(),
    b'{"t":"' + b"a" * 300 + b'\\ud800b"}',
    b'["' + b"\\\\" * 200 + b'"]',
    b'["' + b"a" * 300 + b'\\x"]',
    b'["' + b"a" * 300,
    b'{"a":1e400,"a":1}',
    b'{"a":1,"a":1e400}',
    b"[1e9999999999999999999]",
]


@pytest.mark.parametrize("window", [7, 64, 4096])
@pytest.mark.parametrize("payload", _SPILL_CASES)
def test_spilled_strings_and_small_windows_match_the_decoder(
    payload: bytes, window: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strings above the spill bound and every window boundary give the decoder's identity.

    Tiny read windows and a tiny spill bound put escapes, surrogate pairs,
    combining sequences and multi-byte UTF-8 on every boundary.

    Anti-vacuity: split a surrogate pair, a UTF-8 sequence or a decoded
    composition across spill windows and a case diverges from the decoder.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 64)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", window)
    assert payload_content_identity(payload) == _decoded_identity(payload)


def test_a_duplicate_key_drops_a_discarded_non_finite_member() -> None:
    """Last-key-wins decides before a non-finite member forfeits structure.

    Anti-vacuity: fail on the first non-finite number and ``{"a":1e400,"a":1}``
    gets its byte digest instead of the identity of ``{"a":1}``.
    """
    assert payload_content_identity(b'{"a":1e400,"a":1}') == payload_content_identity(b'{"a":1}')
    assert payload_content_identity(b'{"a":1,"a":1e400}') == sha256(b'{"a":1,"a":1e400}').hexdigest()


def test_a_single_huge_string_is_never_held_whole(tmp_path: Path) -> None:
    """Anti-vacuity: hand the whole string to the tokenizer and the traced peak
    exceeds the string's own size."""
    import tracemalloc

    path = tmp_path / "member.json"
    size = 64 * 1024 * 1024
    with path.open("wb") as handle:
        handle.write(b'{"t":"')
        for _ in range(size // (1024 * 1024)):
            handle.write(b"a" * (1024 * 1024))
        handle.write(b'"}')
    tracemalloc.start()
    try:
        with path.open("rb") as handle:
            stream_payload_content_identity(handle)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < size // 2


def test_a_long_backslash_run_scans_in_linear_time() -> None:
    """Anti-vacuity: a non-possessive backslash run retries from every position,
    and a 1 MiB run takes minutes instead of well under the bound."""
    import time

    payload = b'["' + b"\\\\" * (512 * 1024) + b'"]'
    started = time.perf_counter()
    payload_content_identity(payload)
    assert time.perf_counter() - started < 20


def test_a_large_object_orders_members_through_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: order or de-duplicate spilled members differently from the
    in-memory path and the identity diverges from the decoder."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_ENTRY_MEMORY_BYTES", 4 * (content_identity._ENTRY_OVERHEAD_BYTES + 4))
    value = {f"k{index % 7}́" if index % 3 else f"z{index}": index for index in range(40)}
    payload = json.dumps(value).encode() + b""
    duplicated = b'{"b":1,"a":2,"c":3,"d":4,"e":5,"a":9,"f":1e400,"f":6}'
    assert payload_content_identity(payload) == _decoded_identity(payload)
    assert payload_content_identity(duplicated) == _decoded_identity(duplicated)


def test_an_invalid_escape_is_rejected_before_the_rest_of_a_spilled_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: carry the invalid escape to end of input and the whole
    suffix is held in memory; here it only has to resolve like the decoder."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 64)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 128)
    payload = b'["' + b"a" * 100 + b"\\x" + b"b" * 5000 + b'"]'
    assert payload_content_identity(payload) == sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("payload", "token"),
    [
        (b'{"n": ' + b"1" * 64 + b"}", "number token"),
        (b'{"' + b"k" * 200 + b'": 1}', "object key"),
    ],
)
def test_a_token_beyond_the_physical_value_limit_is_refused_by_name(
    payload: bytes, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: fall back to a byte digest or hash the token anyway and
    no ``ContentIdentityRefusal`` is raised."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    with pytest.raises(content_identity.ContentIdentityRefusal) as refusal:
        payload_content_identity(payload)
    assert refusal.value.token == token


@pytest.mark.parametrize(
    "payload",
    [
        b"x" * 200,
        b'{"n": ' + b"1" * 64 + b"x}",
        b'{"n": 1.' + b"2" * 64 + b"e}",
        b'{"' + b"k" * 200 + b'": 1',
    ],
)
def test_an_overlong_token_in_a_non_json_member_takes_the_byte_identity(
    payload: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal waits until the member is known to be JSON.

    Anti-vacuity: raise at the overlong token instead of after the parse and
    these opaque members are refused instead of receiving their byte digest.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    assert payload_content_identity(payload) == sha256(payload).hexdigest()


def test_an_overlong_float_in_valid_json_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: forward only a prefix that is not a complete number and the
    valid member is mistaken for non-JSON and gets a byte digest."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    with pytest.raises(content_identity.ContentIdentityRefusal) as refusal:
        payload_content_identity(b'{"n": 1.' + b"2" * 64 + b"e5}")
    assert refusal.value.token == "number token"


@pytest.mark.parametrize(
    "payload",
    [
        # 12 combining marks: 24 UTF-8 bytes, but 48 at four bytes per character.
        b'["a' + "́".encode() * 12 + b'"]',
        # A key spelled with escapes: 60 raw bytes that decode to 10.
        b'{"' + b"\\u0061" * 10 + b'": 1}',
    ],
)
def test_a_value_is_measured_by_its_decoded_utf8_size(payload: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: measure four bytes per character or the escaped spelling and
    these storable values are refused instead of matching the decoder."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    assert payload_content_identity(payload) == _decoded_identity(payload)


@pytest.mark.parametrize("bom", [True, False])
@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
def test_a_wide_member_is_read_as_the_source_decoder_reads_it(encoding: str, bom: bool) -> None:
    """A UTF-16/32 member, with or without a byte-order mark, shares the
    identity of its UTF-8 serialization exactly when the record parser's
    ``json.load`` reads it as that value; otherwise it keeps its byte identity.

    Anti-vacuity: read only BOM-bearing members (or only UTF-8) and the
    BOM-less UTF-16-LE and UTF-32-LE members, which the decoder reads, fall
    back to their byte digests.
    """
    value = {"title": "caf\u00e9 \U0001f600", "n": [1, 2.5, None, True]}
    text = json.dumps(value, ensure_ascii=False)
    payload = ("\ufeff".encode(encoding) if bom else b"") + text.encode(encoding)
    try:
        decoder_reads_value = json.loads(payload) == value
    except ValueError:
        decoder_reads_value = False

    expected = payload_content_identity(text.encode()) if decoder_reads_value else sha256(payload).hexdigest()
    assert payload_content_identity(payload) == expected
    if encoding == "utf-16-le":
        assert decoder_reads_value


def test_a_raw_nul_keeps_the_byte_identity_the_record_parser_implies() -> None:
    """Anti-vacuity: drop NULs before tokenizing and the malformed member
    shares the clean document's identity although ``json.load`` rejects it."""
    payload = b'{"a":\x00 1}'
    with pytest.raises(ValueError):
        json.loads(payload)
    assert payload_content_identity(payload) == sha256(payload).hexdigest()


def test_deeply_nested_arrays_stream_in_constant_memory() -> None:
    """A run of directly nested arrays shares one frame.

    Anti-vacuity: allocate a frame per ``[`` and a two-million-deep member
    holds hundreds of megabytes of frames, far past the bound asserted here.
    The bound's headroom is the tokenizer's own per-window event batch, which
    does not grow with depth.
    """
    import tracemalloc

    depth = 2_000_000
    compact = b"[" * depth + b"1" + b"]" * depth
    spaced = b"[ " * depth + b"1" + b" ]" * depth

    tracemalloc.start()
    try:
        identity = stream_payload_content_identity(io.BytesIO(compact))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 128 * 1024 * 1024
    assert identity == stream_payload_content_identity(io.BytesIO(spaced))
    assert identity != sha256(compact).hexdigest()


def test_mixed_nesting_matches_the_decoded_identity() -> None:
    """Array runs broken by objects, and objects nested directly in objects,
    hash exactly as the decoded value does."""
    payload = b'[[1,[2]],[[{"a":[[3]],"b":{"c":{"d":[]}},"e":{}}]],[[[]]]]'
    assert stream_payload_content_identity(io.BytesIO(payload)) == structural_content_identity(loads(payload))


def test_a_key_too_long_for_a_scratch_row_keeps_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spilled object's key that fits one value but not a row beside the
    key hash and digest is spooled to its own scratch file and merged in order.

    Anti-vacuity: store every spilled key in the scratch table and the row
    for the long key exceeds the (pinned) SQLite length limit, raising
    ``sqlite3.DataError`` instead of returning the identity.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 300)
    monkeypatch.setattr(content_identity, "_ENTRY_MEMORY_BYTES", 4 * (content_identity._ENTRY_OVERHEAD_BYTES + 4))
    value: dict[str, object] = {f"k{index}": index for index in range(12)}
    value["m" + "x" * 279] = "long"
    payload = json.dumps(value).encode()

    assert payload_content_identity(payload) == structural_content_identity(value)


def test_a_spilled_key_of_control_characters_is_never_re_escaped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long key reaches the hasher decoded, so control characters and lone
    surrogates keep their exact identity.

    Anti-vacuity: hand the key to the tokenizer as text with its control
    characters raw and the tokenizer rejects it, giving the byte digest.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 64)
    key = "\x01" * 300 + "\ud800" + "x"
    value = {key: 1, "b": [key]}
    payload = json.dumps(value).encode()

    assert payload_content_identity(payload) == structural_content_identity(value)


def test_nesting_past_the_open_container_bound_takes_the_byte_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: drop the bound and the deep member gets a structural
    identity, holding a frame per open object."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_MAX_OPEN_CONTAINERS", 8)
    deep = b'{"a":' * 20 + b"1" + b"}" * 20
    shallow = b'{"a":' * 5 + b"[[[[[[[[[[1]]]]]]]]]]" + b"}" * 5

    assert payload_content_identity(deep) == sha256(deep).hexdigest()
    assert payload_content_identity(shallow) == structural_content_identity(loads(shallow))


def test_member_memory_is_one_budget_across_nested_and_long_keyed_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Few members with long keys, and many small objects nested in one
    another, both spill once the document's shared budget is spent.

    Anti-vacuity: budget members per object by count and neither document
    ever reaches the scratch table, so the spill counter stays zero.
    """
    from polylogue.core import content_identity

    spilled: list[int] = []
    original = content_identity._Entries._spill

    def counting_spill(self: object) -> None:
        spilled.append(1)
        original(self)  # type: ignore[arg-type]

    monkeypatch.setattr(content_identity._Entries, "_spill", counting_spill)
    monkeypatch.setattr(content_identity, "_ENTRY_MEMORY_BYTES", 4096)
    # Four 1200-character keys cost more than the 4096-byte budget together.
    long_keys = {f"k{index}" * 600: index for index in range(4)}
    # Each open object already holds "b" when its nested value starts, so the
    # 40 open objects spend one budget together.
    nested: object = 1
    for index in range(40):
        nested = {"b": index, "a": nested}

    for value in (long_keys, nested):
        spilled.clear()
        payload = json.dumps(value).encode()
        assert payload_content_identity(payload) == structural_content_identity(value)
        assert spilled


def test_bytes_no_encoding_decodes_keep_their_byte_identity() -> None:
    """The decoder's lossy last resort would read these as ``{}``; the
    identity must not let them share the clean document's."""
    payload = b"\xef\xbb\xbf{}\xff\xff"
    assert payload_content_identity(payload) == sha256(payload).hexdigest()
    assert payload_content_identity(payload) != payload_content_identity(b"{}")


def test_a_duplicate_key_drops_a_discarded_refused_member(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused value is scoped to its member, so last-key-wins can replace it.

    Anti-vacuity: keep the refusal document-wide and ``{"a":<overlong>,"a":1}``
    is refused instead of receiving the identity of ``{"a":1}``; drop the
    refusal on replacement in the other order and an overlong surviving value
    is hashed.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    number = b"1." + b"2" * 64
    expected = payload_content_identity(b'{"a":1}')
    for overlong in (number,):
        assert payload_content_identity(b'{"a":' + overlong + b',"a":1}') == expected
        assert payload_content_identity(b'{"b":[{"a":' + overlong + b',"a":1}]}') == payload_content_identity(
            b'{"b":[{"a":1}]}'
        )
        with pytest.raises(content_identity.ContentIdentityRefusal):
            payload_content_identity(b'{"a":1,"a":' + overlong + b"}")
        with pytest.raises(content_identity.ContentIdentityRefusal):
            payload_content_identity(b'{"a":[1,' + overlong + b'],"b":2}')


def test_a_duplicate_key_drops_an_object_holding_a_refused_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An overlong key refuses the object that holds it, not the document.

    Anti-vacuity: keep key refusals document-wide and ``{"a":{<overlong key>:1},"a":1}``
    is refused instead of receiving the identity of ``{"a":1}``.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    inner = b'{"' + b"k" * 64 + b'":1}'
    assert payload_content_identity(b'{"a":' + inner + b',"a":1}') == payload_content_identity(b'{"a":1}')
    for survivor in (inner, b'{"a":1,"a":' + inner + b"}"):
        with pytest.raises(content_identity.ContentIdentityRefusal) as refusal:
            payload_content_identity(survivor)
        assert refusal.value.token == "object key"


def test_the_refusal_raised_is_the_one_that_survived(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: raise the first-seen value refusal and a document whose
    discarded value was refused reports that token instead of the surviving key."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    number = b"1." + b"2" * 64
    document = b'{"a":' + number + b',"a":1,"' + b"k" * 64 + b'":0}'
    with pytest.raises(content_identity.ContentIdentityRefusal) as refusal:
        payload_content_identity(document)
    assert refusal.value.token == "object key"


def test_keys_too_long_for_a_row_are_spooled_not_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: keep such keys in the object's memory map and none are
    spooled, so a member with many of them holds them all at once."""
    from polylogue.core import content_identity

    spooled: list[int] = []
    real_of = content_identity._SpooledKey.of

    def counting(normalized: bytes, checkpoint: Any = None) -> content_identity._SpooledKey:
        spooled.append(len(normalized))
        return real_of(normalized, checkpoint)

    monkeypatch.setattr(content_identity._SpooledKey, "of", counting)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 300)
    monkeypatch.setattr(content_identity, "_ENTRY_MEMORY_BYTES", 4 * (content_identity._ENTRY_OVERHEAD_BYTES + 4))
    value: dict[str, object] = {f"k{index}": index for index in range(12)}
    for letter in "abcdef":
        value[letter + "x" * 279] = letter
    value["b" + "x" * 279] = "replaced"
    assert payload_content_identity(json.dumps(value).encode()) == structural_content_identity(value)
    assert len(spooled) >= 6


def test_exact_key_bytes_obey_the_physical_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    decomposed = "e\u0301" * 12
    with pytest.raises(content_identity.ContentIdentityRefusal, match="object key"):
        payload_content_identity(json.dumps({decomposed: 1}, ensure_ascii=False).encode())
    composed = "é" * 12
    assert payload_content_identity(json.dumps({composed: 1}).encode()) == structural_content_identity({composed: 1})


def test_spooled_exact_keys_preserve_their_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 300)
    monkeypatch.setattr(content_identity, "_ENTRY_MEMORY_BYTES", 4 * (content_identity._ENTRY_OVERHEAD_BYTES + 4))
    stem = "x" * 150
    composed, decomposed = stem + "é", stem + "é"
    filler = {f"k{index}": index for index in range(12)}
    first = {**filler, composed: "one", decomposed: "two"}
    second = {**filler, composed: "two", decomposed: "one"}
    for value in (first, second):
        assert payload_content_identity(json.dumps(value, ensure_ascii=False).encode()) == structural_content_identity(
            value
        )
    assert structural_content_identity(first) != structural_content_identity(second)
    assert structural_content_identity(dict(reversed(list(first.items())))) == structural_content_identity(first)


def test_an_encoded_lone_surrogate_shares_the_identity_of_its_escape() -> None:
    """``json.loads`` decodes bytes with ``surrogatepass``, so an encoded lone
    surrogate is the same value as its ``\\ud800`` escape.

    Anti-vacuity: decode the member strictly and the encoded form falls back
    to its byte digest while the escaped form keeps its structural identity.
    """
    encoded = b'{"x":"a\xed\xa0\x80b"}'
    assert json.loads(encoded) == json.loads(b'{"x":"a\\ud800b"}')
    assert payload_content_identity(encoded) == payload_content_identity(b'{"x":"a\\ud800b"}')
    assert payload_content_identity(encoded) == structural_content_identity({"x": "a\ud800b"})


@pytest.mark.parametrize(
    "payload",
    [
        # An encoded surrogate pair decodes to two characters no escape spells.
        b'["\xed\xa0\xbd\xed\xb8\x80"]',
        # After an escape of the other half, re-spelling it would pair them.
        b'["\\ud83d\xed\xb8\x80"]',
        # After a backslash, the escape it would become changes the value.
        b'["\\\xed\xa0\x80"]',
    ],
)
def test_an_encoded_surrogate_that_cannot_be_re_spelled_keeps_the_byte_identity(payload: bytes) -> None:
    """Anti-vacuity: re-spell these as escapes and they take the identity of a
    different value than the decoder reads."""
    assert payload_content_identity(payload) == sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "token",
    [
        b"0." + b"1" * 300,
        b"-1234567890123456789012345678901234567890.5e-3",
        b"0." + b"0" * 300 + b"5e290",
        b"1e" + b"0" * 300 + b"5",
        b"1" + b"0" * 400 + b".5",
        b"1." + b"0" * 300 + b"e999",
        b"1" + b"0" * 350 + b".5",
    ],
)
def test_a_long_number_is_canonicalized_as_it_streams(token: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    """A number longer than the hold bound never reaches the tokenizer whole.

    Anti-vacuity: forward the run until the physical value limit and the
    tokenizer receives the whole token; canonicalize it wrongly and the
    identity differs from the decoder's float.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_HOLD_NUMBER_BYTES", 32)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 16)
    payload = b'{"n": [' + token + b"]}"
    assert payload_content_identity(payload) == _decoded_identity(payload)
    spills = content_identity._SpilledStrings()
    reader = content_identity._TokenReader(io.BytesIO(payload), spills, scan=False)
    handed_on = b"".join(iter(lambda: reader.read(16), b""))
    assert max(len(run) for run in content_identity._BARE_TOKEN.findall(handed_on)) <= 32


def test_a_long_integer_is_exact_without_runtime_digit_limit_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from decimal import Decimal

    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_HOLD_NUMBER_BYTES", 32)
    digits = b"9" * 5000
    payload = b"[-" + digits + b"]"
    previous = sys.get_int_max_str_digits()
    assert payload_content_identity(payload) == structural_content_identity([-int(Decimal(digits.decode()))])
    assert sys.get_int_max_str_digits() == previous


def test_a_long_key_is_never_held_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key decoded past the spill bound keeps only its hash and exact-byte scratch.

    Anti-vacuity: join the decoded windows into one string and the key comes
    back whole; compare long keys by anything but the decoded text and the
    escaped repeat does not replace the first member.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    key = "\u1100\u1161\u11a8" * 12
    escaped = "".join(f"\\u{ord(char):04x}" for char in key)
    payload = f'{{"{key}": 1, "a": 2, "{escaped}": 3}}'.encode()
    assert payload_content_identity(payload) == structural_content_identity({key: 3, "a": 2})

    spills = content_identity._SpilledStrings()
    reader = content_identity._TokenReader(io.BytesIO(b""), spills, scan=False)
    spill = io.BytesIO(escaped.encode())
    decoded = reader._decode_key(spill)
    assert isinstance(decoded, content_identity._LongKey)
    decoded.close()


def test_the_identity_pass_calls_the_checkpoint_per_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: call the checkpoint only while the caller spools the member
    and a cancellation cannot stop the identity scan."""
    from polylogue.core import content_identity

    class StopError(Exception):
        pass

    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise StopError

    with pytest.raises(StopError):
        stream_payload_content_identity(io.BytesIO(json.dumps({"a": "x" * 64}).encode()), checkpoint=checkpoint)


def test_valid_large_integer_duplicate_settles_before_physical_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid replaced integer does not poison the final exact JSON value."""
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    payload = b'{"a": ' + b"9" * 5000 + b', "a": 1}'
    assert payload_content_identity(payload) == payload_content_identity(b'{"a": 1}')
    monkeypatch.setattr(content_identity, "_HOLD_NUMBER_BYTES", 64)
    assert payload_content_identity(payload) == payload_content_identity(b'{"a": 1}')


def test_an_encoded_surrogate_after_an_escaped_backslash_keeps_its_identity() -> None:
    """Backslash parity decides whether a raw surrogate can be re-spelled.

    Anti-vacuity: refuse every raw surrogate after a backslash and the
    escaped-backslash form falls back to its byte digest while the escaped
    surrogate form of the same value keeps its structural identity.
    """
    encoded = b'{"x":"\\\\\xed\xa0\x80"}'
    escaped = b'{"x":"\\\\\\ud800"}'
    assert json.loads(encoded) == json.loads(escaped)
    assert payload_content_identity(encoded) == payload_content_identity(escaped) == _decoded_identity(escaped)


def test_long_combining_runs_preserve_exact_decoded_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "_SPILL_STRING_BYTES", 16)
    monkeypatch.setattr(content_identity, "_STREAM_READ_BYTES", 8)
    for value in ("a" + "́" * 300 + "̧" * 50 + "b", "̈́" * 200, "é" + "̧" * 90):
        payload = json.dumps([value], ensure_ascii=False).encode()
        assert payload_content_identity(payload) == _decoded_identity(payload)
        assert structural_content_identity([value]) != structural_content_identity(
            [unicodedata.normalize("NFC", value)]
        )


def test_spooled_keys_check_the_checkpoint_per_window() -> None:
    """Anti-vacuity: reread a spooled key without the checkpoint and a
    cancellation waits for every window of every long key."""
    from polylogue.core import content_identity

    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1

    key = content_identity._SpooledKey.of(b"k" * 10, checkpoint)
    assert b"".join(key.chunks()) == b"k" * 10
    key.close()
    assert calls >= 1


def test_a_key_past_a_small_physical_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runtime whose value limit is below the spill bound still refuses a key past it.

    Anti-vacuity: spill only at ``_SPILL_STRING_BYTES`` and this key reaches
    the tokenizer whole, receiving a structural identity.
    """
    from polylogue.core import content_identity

    monkeypatch.setattr(content_identity, "physical_value_limit", lambda: 32)
    with pytest.raises(content_identity.ContentIdentityRefusal) as refusal:
        payload_content_identity(json.dumps({"k" * 40: 1}).encode())
    assert refusal.value.token == "object key"


@pytest.mark.parametrize("digits", [4301, 65537])
def test_valid_large_integer_stream_identity_matches_exact_selected_value(digits: int) -> None:
    from decimal import Decimal

    from polylogue.core.content_identity import structural_content_identity

    value = int(Decimal("9" * digits))
    payload = b'{"a":' + b"9" * digits + b"}"
    assert payload_content_identity(payload) == structural_content_identity({"a": value})
