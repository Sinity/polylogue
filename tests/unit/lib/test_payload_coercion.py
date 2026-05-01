from __future__ import annotations

from datetime import date, datetime

import pytest

from polylogue.core.payload_coercion import (
    coerce_float,
    coerce_int,
    int_pair,
    is_payload_mapping,
    mapping_or_empty,
    mapping_sequence,
    optional_date,
    optional_datetime,
    optional_string,
    string_int_mapping,
    string_sequence,
)


def test_mapping_helpers_accept_string_keyed_mappings_and_iterables() -> None:
    mapping = {"name": "value"}

    assert is_payload_mapping(mapping) is True
    assert is_payload_mapping({1: "value"}) is False
    assert mapping_or_empty(mapping) == mapping
    assert mapping_or_empty("not-a-mapping") == {}
    assert mapping_sequence(mapping) == (mapping,)
    assert mapping_sequence([mapping, {"count": 2}, {1: "ignored"}, "ignored"]) == (
        {"name": "value"},
        {"count": 2},
    )


def test_string_sequence_and_optional_string_handle_scalars_and_mappings() -> None:
    assert string_sequence(None) == ()
    assert string_sequence("alpha") == ("alpha",)
    assert string_sequence({"name": "value"}) == ()
    assert string_sequence(["alpha", 2, True]) == ("alpha", "2", "True")
    assert optional_string(None) is None
    assert optional_string("ready") == "ready"
    assert optional_string(42) == "42"


def test_optional_datetime_and_date_support_instances_and_iso_strings() -> None:
    dt = datetime(2026, 4, 23, 12, 30, 0)
    day = date(2026, 4, 23)

    assert optional_datetime(dt) is dt
    assert optional_datetime("2026-04-23T12:30:00") == dt
    assert optional_date(dt) == day
    assert optional_date(day) == day
    assert optional_date("2026-04-23") == day


def test_numeric_coercion_handles_defaults_booleans_and_stringish_inputs() -> None:
    assert coerce_int(None, default=7) == 7
    assert coerce_int(True) == 1
    assert coerce_int(3.9) == 3
    assert coerce_int(b"8") == 8
    assert coerce_float(None, default=1.5) == 1.5
    assert coerce_float(False) == 0.0
    assert coerce_float(4) == 4.0
    assert coerce_float("3.25") == 3.25


def test_string_int_mapping_and_int_pair_cover_mapping_scalar_and_empty_inputs() -> None:
    assert string_int_mapping({"ok": "2", "other": 1.9, "flag": True}) == {"ok": 2, "other": 1, "flag": 1}
    assert int_pair(None, default=(4, 5)) == (4, 5)
    assert int_pair({"x": 1}, default=(4, 5)) == (4, 5)
    assert int_pair("9", default=(4, 5)) == (9, 5)
    assert int_pair([], default=(4, 5)) == (4, 5)
    assert int_pair(["6", 7, 8], default=(4, 5)) == (6, 7)


def test_invalid_temporal_strings_raise_value_errors() -> None:
    with pytest.raises(ValueError):
        optional_datetime("not-a-datetime")

    with pytest.raises(ValueError):
        optional_date("not-a-date")
