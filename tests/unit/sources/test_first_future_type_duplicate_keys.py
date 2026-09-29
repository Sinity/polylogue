"""The streamed future-type probe agrees with the parser on duplicate keys."""

from __future__ import annotations

import io
import json

import ijson
import pytest

from polylogue.sources.decoder_json import _FirstFutureType
from polylogue.sources.parsers.base_support import _unknown_wire_type


def _streamed(document: str) -> str | None:
    probe = _FirstFutureType()
    events = ijson.parse(io.BytesIO(document.encode()))
    assert next(events) == ("", "start_map", None)
    for _prefix, event, value in events:
        probe.observe(event, value)
    return probe.value


@pytest.mark.parametrize(
    "document",
    [
        # A later value replaces the subtree that carried the future type.
        '{"metadata": {"kind": "future_x"}, "metadata": {}}',
        '{"metadata": {"kind": "future_x"}, "metadata": 1}',
        # The replacement keeps the first key's position, so its type wins over later keys.
        '{"a": {}, "b": {"kind": "future_b"}, "a": {"kind": "future_a"}}',
        '{"type": "future_x", "type": "message"}',
        '{"items": [{"kind": "ok"}, {"record_type": "unknown_y"}]}',
    ],
)
def test_streamed_probe_follows_last_value_wins(document: str) -> None:
    """Anti-vacuity: keeping the first value's candidate reports future_x / future_b."""
    assert _streamed(document) == _unknown_wire_type(json.loads(document))
