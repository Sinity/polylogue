"""Request-local sizing for the existing compact JSON list wire format."""

from __future__ import annotations

import json
from typing import Any


class CompactJSONPage:
    """Append rows without serializing the accumulated prefix repeatedly.

    Brackets cost two bytes and each subsequent row adds one comma. The row
    encoder is the existing compact ``json.dumps`` policy, not a general JSON
    identity or streaming codec. Owners still check their complete envelope
    and signed-cursor sizes after constructing the page.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.items: list[dict[str, Any]] = []
        self.encoded_bytes = 2

    def try_append(self, row: dict[str, Any]) -> bool:
        encoded = json.dumps(row, separators=(",", ":")).encode()
        required = self.encoded_bytes + len(encoded) + bool(self.items)
        if required > self.max_bytes:
            return False
        self.items.append(row)
        self.encoded_bytes = required
        return True
