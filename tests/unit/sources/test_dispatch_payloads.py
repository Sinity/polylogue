"""Provider-dispatch payload normalization regressions."""

from __future__ import annotations

from decimal import Decimal

from polylogue.sources.dispatch import _payload_record, _payload_sequence, parse_payload
from polylogue.types import Provider


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


def test_payload_record_normalizes_streaming_decimals_for_chatgpt_parse() -> None:
    payload = {
        "id": "chatgpt-decimal",
        "title": "Decimal timestamp",
        "create_time": Decimal("1704995846.046526"),
        "mapping": {
            "root": {
                "id": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
                "children": [],
            }
        },
    }

    normalized = _payload_record(payload)
    assert normalized is not None
    assert normalized["create_time"] == 1704995846.046526

    sessions = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "chatgpt-decimal"
