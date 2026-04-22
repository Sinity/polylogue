from __future__ import annotations

from dataclasses import dataclass

import pytest

from polylogue.authored_payloads import (
    PayloadDict,
    canonical_payload_json,
    payload_count_mapping,
    payload_records,
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


def test_payload_records_serializes_typed_records() -> None:
    assert payload_records((_Record("one", 1), _Record("two", 2))) == [
        {"name": "one", "count": 1},
        {"name": "two", "count": 2},
    ]


def test_canonical_payload_json_sorts_authored_payload_keys() -> None:
    assert canonical_payload_json({"z": 1, "a": 2}) == '{"a":2,"z":1}'
