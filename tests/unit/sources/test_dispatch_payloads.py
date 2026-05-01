"""Provider-dispatch payload normalization regressions."""

from __future__ import annotations

from decimal import Decimal

from polylogue.sources.dispatch import _payload_sequence


def test_payload_sequence_normalizes_streaming_decimals() -> None:
    payload = [{"whole": Decimal("2"), "fraction": Decimal("2.5"), "items": [Decimal("3")]}]

    assert _payload_sequence(payload) == [{"whole": 2, "fraction": 2.5, "items": [3]}]
