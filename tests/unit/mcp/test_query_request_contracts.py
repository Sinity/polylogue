"""Contract: MCPSessionQueryRequest and its payload builders stay correct.

Audit #1282 (R3) originally covered this via the auto-generated inputSchema
of three separately-registered MCP tools (search/list_sessions/facets), each
built with a dynamic ``__signature__`` derived from the
:class:`MCPSessionQueryRequest` dataclass (``session_query_request_signature``
in ``polylogue/mcp/query_contracts.py``). polylogue-t46.8 retired that
per-tool registration pattern: ``MCPSessionQueryRequest`` is still the
canonical request shape, but it now feeds a single hand-written subset of
parameters on ``query()``'s own signature
(``polylogue/mcp/server_cutover.py``'s ``_query_sessions``/
``query(projection="sessions", ...)``) rather than an auto-derived schema for
a dedicated tool per operation. The schema-derivation tests this file used to
have no longer apply -- there is no dynamic signature to drift-check. What
remains here is the dataclass-level and payload-builder-level coverage, which
never depended on tool registration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.archive.query.fields import mcp_query_field_names
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.mcp.archive_support import archive_query_filters, archive_search_payload, archive_session_list_payload
from polylogue.mcp.query_contracts import MCPSessionQueryRequest
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary


def _clamp_limit(value: int | object) -> int:
    return value if isinstance(value, int) and value > 0 else 10


def test_mcp_query_request_matches_canonical_query_field_registry() -> None:
    """MCP request fields stay aligned with query fields marked MCP-capable."""
    from dataclasses import fields

    dataclass_field_names = {field.name for field in fields(MCPSessionQueryRequest)}
    assert mcp_query_field_names() <= dataclass_field_names


def test_archive_query_filters_forward_max_words() -> None:
    spec = MCPSessionQueryRequest(max_words=12).build_spec(_clamp_limit)

    assert archive_query_filters(spec)["max_words"] == 12


def test_list_sessions_routes_near_session_to_query_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_archive_search_hits(
        plan: SessionQueryPlan,
        **kwargs: object,
    ) -> tuple[list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]], str]:
        observed["similar_session_id"] = plan.similar_session_id
        observed["archive_root"] = kwargs["archive_root"]
        return [], "semantic"

    monkeypatch.setattr("polylogue.archive.query.archive_execution.archive_search_hits", fake_archive_search_hits)
    archive = MagicMock()
    archive.archive_root = Path("/archive")
    spec = MCPSessionQueryRequest(similar_session_id="seed-session", limit=5).build_spec(_clamp_limit)

    payload = archive_session_list_payload(archive, spec, archive_root=Path("/archive"))

    assert observed == {"similar_session_id": "seed-session", "archive_root": Path("/archive")}
    assert payload.items == ()
    assert payload.total is None


def test_list_sessions_stops_after_requested_distinct_page() -> None:
    summaries = {
        "codex-session:first": ArchiveSessionSummary(
            session_id="codex-session:first",
            native_id="first",
            origin="codex-session",
            title="First",
            created_at=None,
            updated_at=None,
            message_count=1,
            word_count=10,
            tags=(),
        ),
        "codex-session:second": ArchiveSessionSummary(
            session_id="codex-session:second",
            native_id="second",
            origin="codex-session",
            title="Second",
            created_at=None,
            updated_at=None,
            message_count=1,
            word_count=10,
            tags=(),
        ),
    }

    def hit(session_id: str) -> ArchiveSessionSearchHit:
        return ArchiveSessionSearchHit(
            rank=1,
            session_id=session_id,
            block_id=f"{session_id}:block",
            message_id=f"{session_id}:message",
            origin="codex-session",
            title=summaries[session_id].title,
            snippet="needle",
        )

    archive = MagicMock()
    archive.count_search_sessions.return_value = 3
    archive.search_summaries.side_effect = [[hit("codex-session:first"), *[hit("codex-session:second")] * 249]]
    archive.read_summary.side_effect = summaries.__getitem__
    spec = MCPSessionQueryRequest(contains="needle", limit=2).build_spec(_clamp_limit)

    payload = archive_session_list_payload(archive, spec)

    assert [item.id for item in payload.items] == ["codex-session:first", "codex-session:second"]
    assert payload.total == 3
    assert payload.next_offset == 2
    assert [item.match_count for item in payload.items] == [1, 249]
    assert all(item.match_count_is_exact is False for item in payload.items)
    assert archive.search_summaries.call_count == 1


def test_archive_search_routes_near_session_to_query_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = ArchiveSessionSummary(
        session_id="codex-session:near-result",
        native_id="near-result",
        origin="codex-session",
        title="Nearby",
        created_at=None,
        updated_at=None,
        message_count=1,
        word_count=10,
        tags=(),
    )
    hit = ArchiveSessionSearchHit(
        rank=1,
        session_id=summary.session_id,
        block_id="block-1",
        message_id="message-1",
        origin=summary.origin,
        title=summary.title,
        snippet="nearby match",
    )

    def fake_archive_search_hits(
        plan: SessionQueryPlan,
        **kwargs: object,
    ) -> tuple[list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]], str]:
        assert plan.similar_session_id == "seed-session"
        assert kwargs["archive_root"] == Path("/archive")
        return [(hit, summary)], "semantic"

    monkeypatch.setattr("polylogue.archive.query.archive_execution.archive_search_hits", fake_archive_search_hits)
    archive = MagicMock()
    archive.archive_root = Path("/archive")
    archive.read_summary.return_value = summary
    spec = MCPSessionQueryRequest(similar_session_id="seed-session", limit=5).build_spec(_clamp_limit)

    payload = archive_search_payload(
        archive,
        spec,
        query="",
        limit=5,
        offset=0,
        retrieval_lane="semantic",
        sort=None,
        archive_root=Path("/archive"),
    )

    assert payload.retrieval_lane == "semantic"
    assert [hit.session.id for hit in payload.hits] == ["codex-session:near-result"]
