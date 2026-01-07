from __future__ import annotations

from decimal import Decimal

from polylogue.core import json


def test_dumps_handles_decimal():
    payload = {"value": Decimal("1.25")}

    output = json.dumps(payload)

    data = json.loads(output)
    assert data["value"] == 1.25


def test_dumps_accepts_none_option():
    payload = {"value": 123}

    default_output = json.dumps(payload)
    explicit_output = json.dumps(payload, option=None)

    assert explicit_output == default_output
