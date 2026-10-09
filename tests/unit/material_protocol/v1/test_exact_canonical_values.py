"""Opaque material JSON values and keys retain their exact Unicode spelling."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1 import MaterialValueError
from polylogue.material_protocol.v1.canonical import canonical_bytes, canonical_line, parse_json_value


@pytest.mark.parametrize(
    "payload",
    [
        {"é": 1, "e\u0301": 2},
        {"outer": {"é": 1, "e\u0301": 2}},
        [{"outer": {"é": "first", "e\u0301": "second"}}],
    ],
)
def test_distinct_operational_keys_and_values_round_trip(payload: JSONValue) -> None:
    assert parse_json_value(canonical_bytes(payload)) == payload
    assert parse_json_value(canonical_line(payload)) == payload


def test_nonfinite_and_nonstring_key_values_have_typed_refusals() -> None:
    with pytest.raises(MaterialValueError):
        canonical_bytes({1: "value"})  # type: ignore[dict-item]
    with pytest.raises(MaterialValueError):
        canonical_bytes({"value": float("inf")})


def test_actual_producer_helper_emits_shared_numeric_unicode_boundary_vector() -> None:
    expected = (Path(__file__).parents[3] / "fixtures/material_protocol/v1/canonical_boundaries.json").read_bytes()
    payload: JSONValue = {
        "decimal_float": 0.00001,
        "large_float": 1e30,
        "large_integer": 9007199254740993,
        "nested": {"cafe\u0301": ["e\u0301", "日本語"]},
        "small_float": 1e-9,
        "u64_max": 18446744073709551615,
    }
    assert canonical_line(payload) == expected
