"""Strict parity with the original JSON-mode result contract."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from polylogue.operations import daemon_protocol as protocol


class Choice(StrEnum):
    FIRST = "first"


class WireForms(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    coordinates: tuple[int, ...]
    names: list[str]
    choice: Choice
    observed: datetime
    count: int
    complete: bool
    metadata: dict[str, object]


def _payload() -> dict[str, object]:
    return {
        "coordinates": [1, 2],
        "names": ["λ"],
        "choice": "first",
        "observed": "2026-01-01T00:00:00Z",
        "count": 1,
        "complete": True,
        "metadata": {"one": [1, "two", None]},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("coordinates", [1, 2]),
        ("coordinates", (1, 2)),
        ("coordinates", [True]),
        ("coordinates", ["1"]),
        ("names", ("λ",)),
        ("names", [1]),
        ("choice", "first"),
        ("choice", "invalid"),
        ("choice", 1),
        ("observed", "2026-01-01T00:00:00+00:00"),
        ("observed", 1),
        ("observed", "invalid"),
        ("count", True),
        ("count", "1"),
        ("complete", 1),
        ("metadata", {1: "integer", None: "null"}),
        ("metadata", {"bad": float("nan")}),
        ("metadata", {("bad",): "key"}),
        ("metadata", {"bad": object()}),
    ],
)
def test_native_wire_forms_match_original_json_validation(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    try:
        original = WireForms.model_validate_json(json.dumps(payload, allow_nan=False), strict=True)
    except (TypeError, ValueError):
        with pytest.raises((TypeError, ValueError)):
            protocol._check_result_wire(payload, native_result=True)
            protocol._validate_json_result_model(WireForms, payload)
    else:
        protocol._check_result_wire(payload, native_result=True)
        actual = protocol._json_result_validator(WireForms).validate_python(payload)
        assert actual.model_dump() == original.model_dump()


def test_decoded_peer_does_not_gain_native_tuple_or_key_coercion() -> None:
    for value in ({"array": (1,)}, {1: "key"}):
        with pytest.raises((TypeError, ValueError)):
            protocol._check_result_wire(value, native_result=False)


def test_cycle_refuses_but_repeated_shared_value_is_legal() -> None:
    shared: list[Any] = ["same"]
    protocol._check_result_wire([shared, shared], native_result=True)
    shared.append(shared)
    with pytest.raises(ValueError, match="circular"):
        protocol._check_result_wire(shared, native_result=True)


def test_large_result_validation_does_not_encode_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def encode(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("whole result encoding")

    monkeypatch.setattr(protocol, "json", SimpleNamespace(dumps=encode))
    protocol.validate_operation_result(
        "read.dialogue", {"view": "dialogue", "payload": {"text": "λ" * (4 * 1024 * 1024)}}
    )
