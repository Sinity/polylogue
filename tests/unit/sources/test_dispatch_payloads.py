"""Provider-dispatch payload normalization regressions."""

from __future__ import annotations

from decimal import Decimal

from polylogue.sources.dispatch import _payload_sequence


def test_payload_sequence_normalizes_streaming_decimals() -> None:
    payload = [{"whole": Decimal("2"), "fraction": Decimal("2.5"), "items": [Decimal("3")]}]

    normalized = _payload_sequence(payload)
    assert normalized == [{"whole": 2, "fraction": 2.5, "items": [3]}]
    first = normalized[0]
    assert isinstance(first, dict)
    items = first["items"]
    assert isinstance(items, list)
    assert isinstance(first["whole"], int)
    assert isinstance(first["fraction"], float)
    assert isinstance(items[0], int)
