# mypy: disable-error-code="assignment,comparison-overlap,arg-type,override"

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from polylogue.api.insights import PolylogueInsightsMixin
from polylogue.api.sync import SyncPolylogue
from polylogue.api.sync.insights import SyncInsightQueriesMixin
from polylogue.api.sync.sessions import SyncSessionQueriesMixin


class _FilterStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def origin(self, value: object) -> _FilterStub:
        self.calls.append(("origin", value))
        return self

    def since(self, value: object) -> _FilterStub:
        self.calls.append(("since", value))
        return self

    def until(self, value: object) -> _FilterStub:
        self.calls.append(("until", value))
        return self

    def limit(self, value: object) -> _FilterStub:
        self.calls.append(("limit", value))
        return self

    def list_summaries(self) -> str:
        self.calls.append(("list_summaries", None))
        return "summaries-coro"


class _SyncHarness(SyncSessionQueriesMixin, SyncInsightQueriesMixin):
    pass


def test_sync_session_queries_forward_through_sync_bridge() -> None:
    filter_stub = _FilterStub()
    facade = SimpleNamespace(
        get_session=lambda session_id: ("get_session", session_id),
        get_sessions=lambda session_ids: ("get_sessions", tuple(session_ids)),
        get_messages_paginated=lambda session_id, **kwargs: ("messages-page", session_id, kwargs),
        bulk_get_messages=lambda session_ids, **kwargs: ("messages-bulk", tuple(session_ids), kwargs),
        list_sessions=lambda **kwargs: ("list_sessions", kwargs),
        query_sessions=lambda **kwargs: ("query_sessions", kwargs),
        count_sessions=lambda **kwargs: ("count_sessions", kwargs),
        filter=lambda: filter_stub,
        get_session_summary=lambda session_id: ("summary", session_id),
        get_session_stats=lambda session_id: ("stats", session_id),
        get_session_tree=lambda session_id: ("tree", session_id),
        list_tags=lambda **kwargs: ("tags", kwargs),
        search=lambda query, **kwargs: ("search", query, kwargs),
        stats=lambda: "stats-coro",
        health_check=lambda: "health-coro",
        neighbor_candidates=lambda **kwargs: ("neighbors", kwargs),
    )
    archive = _SyncHarness()
    archive._facade = facade

    with patch("polylogue.api.sync.sessions.run_coroutine_sync", side_effect=lambda coro: coro) as mock_run:
        assert archive.get_session("conv-1") == ("get_session", "conv-1")
        assert archive.get_sessions(["a", "b"]) == ("get_sessions", ("a", "b"))
        assert archive.get_messages_paginated("conv-1", limit=20, offset=5) == (
            "messages-page",
            "conv-1",
            {
                "message_role": (),
                "message_type": None,
                "limit": 20,
                "offset": 5,
                "content_projection": None,
            },
        )
        assert archive.bulk_get_messages(["a", "b"], since="2026-01-01", until="2026-01-02") == (
            "messages-bulk",
            ("a", "b"),
            {"since": "2026-01-01", "until": "2026-01-02", "message_role": (), "content_projection": None},
        )
        assert archive.list_sessions(origin="claude-code-session", limit=3) == (
            "list_sessions",
            {"origin": "claude-code-session", "limit": 3},
        )
        assert archive.query_sessions(origin="claude-code-session", limit=3, has_tool_use=True) == (
            "query_sessions",
            {
                "origin": "claude-code-session",
                "tag": None,
                "since": None,
                "until": None,
                "sort": None,
                "limit": 3,
                "offset": 0,
                "has_tool_use": True,
                "has_thinking": False,
                "has_paste": False,
                "typed_only": False,
                "min_messages": None,
                "max_messages": None,
                "min_words": None,
            },
        )
        assert archive.count_sessions(origin="claude-code-session", since="2026-01-01") == (
            "count_sessions",
            {"origin": "claude-code-session", "since": "2026-01-01", "until": None},
        )
        assert (
            archive.list_summaries(
                origin="claude-code-session",
                since="2026-01-01",
                until="2026-01-31",
                limit=5,
            )
            == "summaries-coro"
        )
        assert archive.get_session_summary("conv-1") == ("summary", "conv-1")
        assert archive.get_session_stats("conv-1") == ("stats", "conv-1")
        assert archive.get_session_tree("conv-1") == ("tree", "conv-1")
        assert archive.list_tags(origin="claude-code-session") == ("tags", {"origin": "claude-code-session"})
        assert archive.search("query", limit=7, source="inbox", since="2026-01-01") == (
            "search",
            "query",
            {"limit": 7, "source": "inbox", "since": "2026-01-01"},
        )
        assert archive.stats() == "stats-coro"
        assert archive.health_check() == "health-coro"
        assert archive.neighbor_candidates(session_id="conv-1", limit=4) == (
            "neighbors",
            {
                "session_id": "conv-1",
                "query": None,
                "provider": None,
                "limit": 4,
                "window_hours": 24,
            },
        )

    assert filter_stub.calls == [
        ("origin", "claude-code-session"),
        ("since", "2026-01-01"),
        ("until", "2026-01-31"),
        ("limit", 5),
        ("list_summaries", None),
    ]
    assert mock_run.call_count == 16


def test_sync_product_queries_forward_through_sync_bridge() -> None:
    facade = SimpleNamespace(
        get_session_insight_status=lambda: "status-coro",
        get_session_profile_insight=lambda session_id, **kwargs: ("profile", session_id, kwargs),
        list_session_profile_insights=lambda query=None: ("profiles", query),
        list_session_tag_rollup_insights=lambda query=None: ("tags", query),
        get_session_work_event_insights=lambda session_id: ("events", session_id),
        list_session_work_event_insights=lambda query=None: ("events-list", query),
        get_session_phase_insights=lambda session_id: ("phases", session_id),
        list_session_phase_insights=lambda query=None: ("phases-list", query),
        get_thread_insight=lambda thread_id: ("thread", thread_id),
        list_thread_insights=lambda query=None: ("threads", query),
        list_archive_coverage_insights=lambda query=None: ("coverage", query),
        list_tool_usage_insights=lambda query=None: ("tool-usage", query),
        list_session_cost_insights=lambda query=None: ("session-costs", query),
        list_cost_rollup_insights=lambda query=None: ("cost-rollups", query),
        list_archive_debt_insights=lambda query=None: ("debt", query),
        insight_readiness_report=lambda query=None: ("readiness", query),
    )
    archive = _SyncHarness()
    archive._facade = facade

    with patch("polylogue.api.sync.insights.run_coroutine_sync", side_effect=lambda coro: coro) as mock_run:
        assert archive.get_session_insight_status() == "status-coro"
        assert archive.get_session_profile_insight("conv-1", tier="evidence") == (
            "profile",
            "conv-1",
            {"tier": "evidence"},
        )
        assert archive.list_session_profile_insights("query") == ("profiles", "query")
        assert archive.list_session_tag_rollup_insights("query") == ("tags", "query")
        assert archive.get_session_work_event_insights("conv-1") == ("events", "conv-1")
        assert archive.list_session_work_event_insights("query") == ("events-list", "query")
        assert archive.get_session_phase_insights("conv-1") == ("phases", "conv-1")
        assert archive.list_session_phase_insights("query") == ("phases-list", "query")
        assert archive.get_thread_insight("thread-1") == ("thread", "thread-1")
        assert archive.list_thread_insights("query") == ("threads", "query")
        assert archive.list_archive_coverage_insights("query") == ("coverage", "query")
        assert archive.list_tool_usage_insights("query") == ("tool-usage", "query")
        assert archive.list_session_cost_insights("query") == ("session-costs", "query")
        assert archive.list_cost_rollup_insights("query") == ("cost-rollups", "query")
        assert archive.list_archive_debt_insights("query") == ("debt", "query")
        assert archive.insight_readiness_report("query") == ("readiness", "query")

    assert mock_run.call_count == 16


def test_sync_polylogue_wraps_async_facade_and_context_manager() -> None:
    facade = MagicMock()
    facade.close.return_value = "close-coro"
    facade.filter.return_value = "filter-object"

    with (
        patch("polylogue.api.Polylogue", return_value=facade) as mock_facade_class,
        patch("polylogue.api.sync._run", return_value=None) as mock_run,
    ):
        archive = SyncPolylogue(archive_root="archive-root", db_path="db.sqlite")
        assert archive.filter() == "filter-object"
        assert "SyncPolylogue(facade=" in repr(archive)
        assert archive.__enter__() is archive
        archive.close()
        archive.__exit__(None, None, None)

    mock_facade_class.assert_called_once_with(archive_root="archive-root", db_path="db.sqlite")
    assert [call.args[0] for call in mock_run.call_args_list] == [facade.close.return_value, facade.close.return_value]


@pytest.mark.asyncio
async def test_polylogue_products_mixin_forwards_all_product_calls(tmp_path: Path) -> None:
    """The insight mixin forwards every product call to ArchiveStore.

    The mixin reads insights directly from ``ArchiveStore`` (opened from
    ``self.config``), not from an ``operations`` surface. This pins that every
    public product method opens the archive and delegates to its
    same-named substrate method, returning the substrate result unchanged.
    """
    from polylogue.config import Config

    archive = MagicMock()
    archive.list_session_tag_rollup_insights.return_value = []
    archive.stats_by.return_value = {}
    archive.get_session_work_event_insights.return_value = ["events"]
    archive.list_session_work_event_insights.return_value = ["events-list"]
    archive.get_session_phase_insights.return_value = ["phases"]
    archive.list_session_phase_insights.return_value = ["phases-list"]
    archive.get_thread_insight.return_value = "thread"
    archive.list_thread_insights.return_value = ["threads"]
    archive.list_archive_coverage_insights.return_value = ["coverage"]
    archive.list_tool_usage_insights.return_value = ["tool-usage"]
    archive.list_session_cost_insights.return_value = []
    archive.list_cost_rollup_insights.return_value = []
    archive.list_archive_debt_insights.return_value = ["debt"]

    open_existing = MagicMock()
    open_existing.return_value.__enter__.return_value = archive
    open_existing.return_value.__exit__.return_value = False

    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[], db_path=tmp_path / "index.db")

    class _Harness(PolylogueInsightsMixin):
        def __init__(self, config: Config) -> None:
            self._config = config

        @property
        def config(self) -> Config:
            return self._config

    harness = _Harness(config)

    with patch("polylogue.api.insights.ArchiveStore.open_existing", open_existing):
        # tag-rollup merges synthesized provider rollups + sorts; cost enriches.
        # These post-process (not pure forwarders) — assert delegation + empty post-process.
        assert await harness.list_session_tag_rollup_insights() == []
        assert await harness.get_session_work_event_insights("conv-1") == ["events"]
        assert await harness.list_session_work_event_insights() == ["events-list"]
        assert await harness.get_session_phase_insights("conv-1") == ["phases"]
        assert await harness.list_session_phase_insights() == ["phases-list"]
        assert await harness.get_thread_insight("thread-1") == "thread"
        assert await harness.list_thread_insights() == ["threads"]
        assert await harness.list_archive_coverage_insights() == ["coverage"]
        tool_usage = await harness.list_tool_usage_insights()
        assert len(tool_usage) == 1
        assert tool_usage[0].insight_kind == "tool_usage"
        assert await harness.list_session_cost_insights() == []
        assert await harness.list_cost_rollup_insights() == []
        assert await harness.list_archive_debt_insights() == ["debt"]

    archive.list_session_tag_rollup_insights.assert_called_once()
    archive.get_session_work_event_insights.assert_called_once_with("conv-1")
    archive.list_session_work_event_insights.assert_called_once()
    archive.get_session_phase_insights.assert_called_once_with("conv-1")
    archive.list_session_phase_insights.assert_called_once()
    archive.get_thread_insight.assert_called_once_with("thread-1")
    archive.list_thread_insights.assert_called_once()
    archive.list_archive_coverage_insights.assert_called_once()
    archive.list_tool_usage_insights.assert_not_called()
    archive.list_session_cost_insights.assert_called()  # also called by cost-rollup derivation
    archive.list_archive_debt_insights.assert_called_once()
