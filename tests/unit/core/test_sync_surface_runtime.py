# mypy: disable-error-code="assignment,comparison-overlap,arg-type,override"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.facade_products import PolylogueProductsMixin
from polylogue.sync import SyncPolylogue
from polylogue.sync_conversation_queries import SyncConversationQueriesMixin
from polylogue.sync_product_queries import SyncProductQueriesMixin


class _FilterStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def provider(self, value: object) -> _FilterStub:
        self.calls.append(("provider", value))
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


class _SyncHarness(SyncConversationQueriesMixin, SyncProductQueriesMixin):
    pass


def test_sync_conversation_queries_forward_through_sync_bridge() -> None:
    filter_stub = _FilterStub()
    facade = SimpleNamespace(
        get_conversation=lambda conversation_id: ("get_conversation", conversation_id),
        get_conversations=lambda conversation_ids: ("get_conversations", tuple(conversation_ids)),
        list_conversations=lambda **kwargs: ("list_conversations", kwargs),
        filter=lambda: filter_stub,
        search=lambda query, **kwargs: ("search", query, kwargs),
        stats=lambda: "stats-coro",
    )
    archive = _SyncHarness()
    archive._facade = facade

    with patch("polylogue.sync_conversation_queries.run_coroutine_sync", side_effect=lambda coro: coro) as mock_run:
        assert archive.get_conversation("conv-1") == ("get_conversation", "conv-1")
        assert archive.get_conversations(["a", "b"]) == ("get_conversations", ("a", "b"))
        assert archive.list_conversations(provider="claude-code", limit=3) == (
            "list_conversations",
            {"provider": "claude-code", "limit": 3},
        )
        assert (
            archive.list_summaries(
                provider="claude-code",
                since="2026-01-01",
                until="2026-01-31",
                limit=5,
            )
            == "summaries-coro"
        )
        assert archive.search("query", limit=7, source="inbox", since="2026-01-01") == (
            "search",
            "query",
            {"limit": 7, "source": "inbox", "since": "2026-01-01"},
        )
        assert archive.stats() == "stats-coro"

    assert filter_stub.calls == [
        ("provider", "claude-code"),
        ("since", "2026-01-01"),
        ("until", "2026-01-31"),
        ("limit", 5),
        ("list_summaries", None),
    ]
    assert mock_run.call_count == 6


def test_sync_product_queries_forward_through_sync_bridge() -> None:
    facade = SimpleNamespace(
        get_session_product_status=lambda: "status-coro",
        get_session_profile_product=lambda conversation_id, **kwargs: ("profile", conversation_id, kwargs),
        list_session_profile_products=lambda query=None: ("profiles", query),
        get_session_enrichment_product=lambda conversation_id: ("enrichment", conversation_id),
        list_session_enrichment_products=lambda query=None: ("enrichments", query),
        list_session_tag_rollup_products=lambda query=None: ("tags", query),
        get_session_work_event_products=lambda conversation_id: ("events", conversation_id),
        list_session_work_event_products=lambda query=None: ("events-list", query),
        get_session_phase_products=lambda conversation_id: ("phases", conversation_id),
        list_session_phase_products=lambda query=None: ("phases-list", query),
        get_work_thread_product=lambda thread_id: ("thread", thread_id),
        list_work_thread_products=lambda query=None: ("threads", query),
        list_day_session_summary_products=lambda query=None: ("days", query),
        list_week_session_summary_products=lambda query=None: ("weeks", query),
        list_provider_analytics_products=lambda query=None: ("analytics", query),
        list_archive_debt_products=lambda query=None: ("debt", query),
    )
    archive = _SyncHarness()
    archive._facade = facade

    with patch("polylogue.sync_product_queries.run_coroutine_sync", side_effect=lambda coro: coro) as mock_run:
        assert archive.get_session_product_status() == "status-coro"
        assert archive.get_session_profile_product("conv-1", tier="evidence") == (
            "profile",
            "conv-1",
            {"tier": "evidence"},
        )
        assert archive.list_session_profile_products("query") == ("profiles", "query")
        assert archive.get_session_enrichment_product("conv-1") == ("enrichment", "conv-1")
        assert archive.list_session_enrichment_products("query") == ("enrichments", "query")
        assert archive.list_session_tag_rollup_products("query") == ("tags", "query")
        assert archive.get_session_work_event_products("conv-1") == ("events", "conv-1")
        assert archive.list_session_work_event_products("query") == ("events-list", "query")
        assert archive.get_session_phase_products("conv-1") == ("phases", "conv-1")
        assert archive.list_session_phase_products("query") == ("phases-list", "query")
        assert archive.get_work_thread_product("thread-1") == ("thread", "thread-1")
        assert archive.list_work_thread_products("query") == ("threads", "query")
        assert archive.list_day_session_summary_products("query") == ("days", "query")
        assert archive.list_week_session_summary_products("query") == ("weeks", "query")
        assert archive.list_provider_analytics_products("query") == ("analytics", "query")
        assert archive.list_archive_debt_products("query") == ("debt", "query")

    assert mock_run.call_count == 16


def test_sync_polylogue_wraps_async_facade_and_context_manager() -> None:
    facade = MagicMock()
    facade.close.return_value = "close-coro"
    facade.filter.return_value = "filter-object"

    with (
        patch("polylogue.facade.Polylogue", return_value=facade) as mock_facade_class,
        patch("polylogue.sync._run", return_value=None) as mock_run,
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
async def test_polylogue_products_mixin_forwards_all_product_calls() -> None:
    operations = SimpleNamespace(
        list_session_tag_rollup_products=AsyncMock(return_value=["tags"]),
        get_session_work_event_products=AsyncMock(return_value=["events"]),
        list_session_work_event_products=AsyncMock(return_value=["events-list"]),
        get_session_phase_products=AsyncMock(return_value=["phases"]),
        list_session_phase_products=AsyncMock(return_value=["phases-list"]),
        get_work_thread_product=AsyncMock(return_value="thread"),
        list_work_thread_products=AsyncMock(return_value=["threads"]),
        list_day_session_summary_products=AsyncMock(return_value=["days"]),
        list_week_session_summary_products=AsyncMock(return_value=["weeks"]),
        list_provider_analytics_products=AsyncMock(return_value=["analytics"]),
        list_archive_debt_products=AsyncMock(return_value=["debt"]),
    )

    class _Harness(PolylogueProductsMixin):
        def __init__(self, operations: object) -> None:
            self._operations = operations

        @property
        def operations(self) -> object:
            return self._operations

    archive = _Harness(operations)

    assert await archive.list_session_tag_rollup_products("query") == ["tags"]
    assert await archive.get_session_work_event_products("conv-1") == ["events"]
    assert await archive.list_session_work_event_products("query") == ["events-list"]
    assert await archive.get_session_phase_products("conv-1") == ["phases"]
    assert await archive.list_session_phase_products("query") == ["phases-list"]
    assert await archive.get_work_thread_product("thread-1") == "thread"
    assert await archive.list_work_thread_products("query") == ["threads"]
    assert await archive.list_day_session_summary_products("query") == ["days"]
    assert await archive.list_week_session_summary_products("query") == ["weeks"]
    assert await archive.list_provider_analytics_products("query") == ["analytics"]
    assert await archive.list_archive_debt_products("query") == ["debt"]

    operations.list_session_tag_rollup_products.assert_awaited_once_with("query")
    operations.get_session_work_event_products.assert_awaited_once_with("conv-1")
    operations.list_session_work_event_products.assert_awaited_once_with("query")
    operations.get_session_phase_products.assert_awaited_once_with("conv-1")
    operations.list_session_phase_products.assert_awaited_once_with("query")
    operations.get_work_thread_product.assert_awaited_once_with("thread-1")
    operations.list_work_thread_products.assert_awaited_once_with("query")
    operations.list_day_session_summary_products.assert_awaited_once_with("query")
    operations.list_week_session_summary_products.assert_awaited_once_with("query")
    operations.list_provider_analytics_products.assert_awaited_once_with("query")
    operations.list_archive_debt_products.assert_awaited_once_with("query")
