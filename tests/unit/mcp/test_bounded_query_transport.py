from __future__ import annotations

import json

from pydantic import BaseModel

from polylogue.mcp.server_support import _json_payload, _response_context
from polylogue.surfaces.outcome import decide_outcome
from polylogue.surfaces.payloads import QueryUnitEnvelope, QueryUnitProjectedRowPayload


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

    with _response_context("read", {"limit": 20, "offset": 0, "ref": "session:demo"}):
        result = _json_payload(payload)

    body = json.loads(result)
    assert body["status"] == "response_budget_exceeded"
    assert body["returned_items"] > 0
    assert body["page"]["items"]
    assert body["continuation"]["arguments"]["offset"] == body["returned_items"]
    assert len(result.encode("utf-8")) <= 25_000


def test_oversized_projected_item_envelope_returns_useful_advancing_page() -> None:
    """The MCP budgeter pages selected rows even though full ``items`` is empty.

    Anti-vacuity: prioritize the empty ``items`` field over ``projected_items``
    and the over-budget envelope has no page or returned rows.
    """
    payload = QueryUnitEnvelope(
        unit="message",
        query="messages where role:user | select text",
        items=(),
        projected_items=tuple(
            QueryUnitProjectedRowPayload(root={"text": f"item-{index}-" + "x" * 4000}) for index in range(20)
        ),
        total=20,
        limit=20,
        offset=0,
        next_offset=None,
        outcome=decide_outcome(matched=20),
    )

    with _response_context("query", {"limit": 20, "offset": 0}):
        result = _json_payload(payload)

    body = json.loads(result)
    assert body["status"] == "response_budget_exceeded"
    assert body["returned_items"] > 0
    assert body["page"]["items"] == []
    assert body["page"]["projected_items"]
    assert len(result.encode("utf-8")) <= 25_000
