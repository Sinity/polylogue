"""Opaque v2 producer positions; v1 is deliberately retired."""

from __future__ import annotations

import base64
import json

import pytest

from polylogue.archive.query.search_contract import SearchExecution
from polylogue.archive.query.search_cursor import (
    InvalidSearchCursorError,
    decode_search_cursor,
    encode_search_cursor,
    position,
)
from polylogue.surfaces.payloads import build_search_envelope


def test_position_roundtrip_retains_exact_numeric_key() -> None:
    anchor = position("session", "neutral", (("12345678901234567890", True, True), ("neutral", False, False)))
    token = encode_search_cursor(anchor, lane="hybrid", request_identity="selection")
    cursor = decode_search_cursor(token)
    assert cursor.position == anchor
    assert cursor.query_hash == "selection"
    assert cursor.v == 2


@pytest.mark.parametrize("body", [{"v": 1, "r": 1, "c": "a", "l": "dialogue", "o": True}, [], {"v": 2}])
def test_predecessor_and_incomplete_tokens_refused(body: object) -> None:
    token = base64.urlsafe_b64encode(json.dumps(body).encode()).decode()
    with pytest.raises(InvalidSearchCursorError):
        decode_search_cursor(token)


@pytest.mark.parametrize("token", ["", "%%%%", "bm90LWpzb24", "☃"])
def test_malformed_token_refused(token: str) -> None:
    with pytest.raises(InvalidSearchCursorError):
        decode_search_cursor(token)


def test_renderer_does_not_invent_continuation_without_producer_evidence() -> None:
    envelope = build_search_envelope(
        (),
        total=100,
        limit=1,
        offset=0,
        query="neutral",
        retrieval_lane="dialogue",
        execution=SearchExecution(("text",), ("text",), has_more=False),
    )
    assert envelope.next_cursor is None
    assert envelope.next_offset is None


@pytest.mark.parametrize(
    "value,numeric", [(float("nan"), False), (float("inf"), False), ("NaN", True), ("not-a-number", True)]
)
def test_invalid_numeric_key_refused_during_decode(value: object, numeric: bool) -> None:
    body = {
        "v": 2,
        "lane": "dialogue",
        "query_hash": "selection",
        "position": {"grain": "block", "identity": "neutral", "key": [{"value": value, "numeric_text": numeric}]},
    }
    with pytest.raises(InvalidSearchCursorError):
        decode_search_cursor(base64.urlsafe_b64encode(json.dumps(body).encode()).decode())


def test_missing_version_refused() -> None:
    body = {
        "lane": "dialogue",
        "query_hash": "selection",
        "position": {"grain": "block", "identity": "neutral", "key": [{"value": 1}]},
    }
    with pytest.raises(InvalidSearchCursorError):
        decode_search_cursor(base64.urlsafe_b64encode(json.dumps(body).encode()).decode())


def test_mixed_comparator_types_are_typed_refusal() -> None:
    with pytest.raises(InvalidSearchCursorError):
        position("block", "a", ((1, False, False),)).after(position("block", "b", (("one", False, False),)))


def test_cursor_refusal_uses_daemon_400_boundary() -> None:
    from polylogue.core.errors import PolylogueError

    error = InvalidSearchCursorError("invalid cursor")
    assert isinstance(error, PolylogueError)
    assert error.http_status_code == 400


def test_declarative_identity_binds_predicate_and_ignores_only_page_state() -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.query.predicate import QueryFieldPredicate
    from polylogue.archive.query.search_cursor import plan_cursor_identity

    plan = SessionQueryPlan(
        since=datetime(2026, 1, 1, tzinfo=UTC), boolean_predicate=QueryFieldPredicate("title", ("a",))
    )
    identity = plan_cursor_identity(plan)
    assert identity == plan_cursor_identity(replace(plan, limit=3, offset=10, cursor="opaque"))
    assert identity != plan_cursor_identity(replace(plan, boolean_predicate=QueryFieldPredicate("title", ("b",))))
    assert identity != plan_cursor_identity(replace(plan, has_branches=True))
