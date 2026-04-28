from __future__ import annotations

from dataclasses import dataclass

import pytest

from polylogue.products.authored_payloads import (
    PayloadDict,
    canonical_payload_json,
    merge_unique_string_tuples,
    payload_bool,
    payload_count,
    payload_count_mapping,
    payload_float,
    payload_int,
    payload_items,
    payload_mapping,
    payload_optional_string,
    payload_records,
    payload_string,
    payload_string_tuple,
    require_payload_mapping,
)


@dataclass(frozen=True)
class _Record:
    name: str
    count: int

    def to_payload(self) -> PayloadDict:
        return {"name": self.name, "count": self.count}


def test_require_payload_mapping_returns_string_keyed_mapping() -> None:
    payload = require_payload_mapping({"passed": 2}, context="qa.summary")

    assert payload == {"passed": 2}


def test_require_payload_mapping_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="qa.summary"):
        require_payload_mapping({1: "bad"}, context="qa.summary")


def test_payload_count_mapping_coerces_count_values() -> None:
    counts = payload_count_mapping({"passed": "2", "failed": 1.0}, context="qa.summary")

    assert counts == {"passed": 2, "failed": 1}


def test_payload_scalar_and_sequence_helpers_preserve_contracts() -> None:
    assert payload_items(["a", "b"]) == ("a", "b")
    assert payload_items("not-a-sequence") == ()
    assert payload_string(None, default="fallback") == "fallback"
    assert payload_string(9) == "9"
    assert payload_optional_string("ready") == "ready"
    assert payload_optional_string("") is None
    assert payload_string_tuple(["a", "b"]) == ("a", "b")
    assert payload_string_tuple(["a", 2]) == ()


def test_payload_numeric_and_bool_helpers_coerce_supported_values() -> None:
    assert payload_int(True, "count") == 1
    assert payload_count("3", "count") == 3
    assert payload_float("1.5", "cost") == 1.5
    assert payload_bool("yes") is True
    assert payload_bool("off") is False
    assert payload_bool(None, default=True) is True


def test_payload_mapping_and_tuple_merge_cover_rejections() -> None:
    assert payload_mapping({"ok": 1}) == {"ok": 1}
    assert payload_mapping({1: "bad"}) is None
    assert merge_unique_string_tuples(("a", "", "b"), ("b", "c"), skip_empty=True) == ("a", "b", "c")


def test_payload_type_errors_are_actionable() -> None:
    with pytest.raises(TypeError, match="count must be present"):
        payload_count(None, "count")

    with pytest.raises(TypeError, match="cost must be a float-compatible value"):
        payload_float(object(), "cost")

    with pytest.raises(TypeError, match="Expected a boolean-compatible payload value"):
        payload_bool(object())


def test_payload_records_serializes_typed_records() -> None:
    assert payload_records((_Record("one", 1), _Record("two", 2))) == [
        {"name": "one", "count": 1},
        {"name": "two", "count": 2},
    ]


def test_canonical_payload_json_sorts_authored_payload_keys() -> None:
    assert canonical_payload_json({"z": 1, "a": 2}) == '{"a":2,"z":1}'
