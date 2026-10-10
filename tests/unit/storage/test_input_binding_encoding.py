"""Exact scalar framing survives fused allocation and fixed-prefix reuse."""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from polylogue.storage.derived.session.input_binding import encode_input_binding_row


def _declared_frame(values: Sequence[object]) -> bytes:
    """Independent framing specification: count, then typed byte-length values."""
    frames = [str(len(values)).encode("ascii") + b":"]
    for value in values:
        if value is None:
            frames.append(b"N")
            continue
        if isinstance(value, str):
            kind, encoded = b"T", value.encode("utf-8")
        elif isinstance(value, bytes):
            kind, encoded = b"B", value
        elif isinstance(value, int):
            kind, encoded = b"I", str(value).encode("ascii")
        elif isinstance(value, float):
            kind, encoded = b"R", value.hex().encode("ascii")
        else:
            raise TypeError(f"unsupported SQLite binding value: {type(value).__name__}")
        frames.append(kind + str(len(encoded)).encode("ascii") + b":" + encoded)
    return b"".join(frames)


def test_binding_scalar_framing_preserves_null_types_unicode_and_binary() -> None:
    assert encode_input_binding_row([None, "", b"", 1, True, False, "é", b"\x00\xff"]) == (
        b"8:NT0:B0:I1:1I4:TrueI5:FalseT2:\xc3\xa9B2:\x00\xff"
    )
    assert encode_input_binding_row(["a", "bc"]) != encode_input_binding_row(["ab", "c"])
    assert encode_input_binding_row([1]) != encode_input_binding_row(["1"])
    assert encode_input_binding_row([None]) != encode_input_binding_row([""])


def test_binding_scalar_framing_matches_declared_format_across_lookup_boundaries() -> None:
    values: list[object] = [
        None,
        "",
        b"",
        True,
        False,
        -1,
        0,
        1,
        4095,
        4096,
        -(2**63),
        2**63 - 1,
        10**100,
        -(10**100),
        0.0,
        -0.0,
        1.25,
        float("inf"),
        float("-inf"),
        float("nan"),
        "N:T1:1\x00",
        "Żółć e\u0301 🙂",
        bytes(range(256)),
    ]
    for size in (1, 1023, 1024, 1025, 4096):
        values.extend(("x" * size, b"x" * size))
    values.extend(("é" * 511 + "a", "é" * 512, "é" * 512 + "a"))
    for count in (0, 1, 127, 128, 129, 4096):
        row = [values[index % len(values)] for index in range(count)]
        assert encode_input_binding_row(row) == _declared_frame(row)
    randomizer = random.Random(41)
    for _ in range(300):
        row = [randomizer.choice(values) for _ in range(randomizer.randrange(160))]
        assert encode_input_binding_row(row) == _declared_frame(row)


def test_small_integer_frames_do_not_replace_subclass_spelling() -> None:
    class SpelledInteger(int):
        def __str__(self) -> str:
            return "seven"

    assert encode_input_binding_row([SpelledInteger(7)]) == b"1:I5:seven"


@pytest.mark.parametrize("value", [object(), bytearray(b"x"), memoryview(b"x"), [], {}])
def test_unsupported_binding_values_keep_the_declared_type_error(value: object) -> None:
    with pytest.raises(TypeError, match=f"^unsupported SQLite binding value: {type(value).__name__}$"):
        encode_input_binding_row([value])


def test_binding_text_encoding_keeps_strict_unicode_failure() -> None:
    with pytest.raises(UnicodeEncodeError):
        encode_input_binding_row(["\ud800"])
