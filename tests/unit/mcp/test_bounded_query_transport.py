from __future__ import annotations

import json

from pydantic import BaseModel

from polylogue.mcp.server_support import _json_payload, _response_context


class _LargeItem(BaseModel):
    text: str


class _LargePage(BaseModel):
    items: tuple[_LargeItem, ...]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


def test_oversized_item_envelope_returns_useful_advancing_page() -> None:
    payload = _LargePage(
        items=tuple(_LargeItem(text=f"item-{index}-" + "x" * 4000) for index in range(20)),
        total=20,
        limit=20,
        offset=0,
        next_offset=20,
    )

    with _response_context("query_units", {"limit": 20, "offset": 0, "expression": "actions"}):
        result = _json_payload(payload)

    body = json.loads(result)
    assert body["status"] == "response_budget_exceeded"
    assert body["returned_items"] > 0
    assert body["page"]["items"]
    assert body["continuation"]["arguments"]["offset"] == body["returned_items"]
    assert len(result.encode("utf-8")) <= 25_000
