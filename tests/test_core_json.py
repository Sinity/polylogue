from __future__ import annotations

from decimal import Decimal

from polylogue.core import json


def test_dumps_handles_decimal():
    payload = {"value": Decimal("1.25")}

    output = json.dumps(payload)

    data = json.loads(output)
    assert data["value"] == 1.25
