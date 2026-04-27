# mypy: disable-error-code="assignment,comparison-overlap,arg-type"

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.storage.products.product_read_support import hydrate_mapping, hydrate_optional, hydrate_sequence
from polylogue.storage.repository.product.profile_reads import RepositoryProductProfileReadMixin
from polylogue.storage.repository.product.summary_reads import RepositoryProductSummaryReadMixin
from polylogue.storage.repository.product.thread_reads import RepositoryProductThreadReadMixin
from polylogue.storage.repository.product.timeline_reads import RepositoryProductTimelineReadMixin
from polylogue.storage.repository.raw.repository_raw import RepositoryRawMixin


def test_product_read_support_hydrates_optional_sequence_and_mapping() -> None:
    assert hydrate_optional(None, lambda record: f"hydrated:{record}") is None
    assert hydrate_optional("record", lambda record: f"hydrated:{record}") == "hydrated:record"
    assert hydrate_sequence(["a", "b"], lambda record: record.upper()) == ["A", "B"]
    assert hydrate_mapping({"a": "one", "b": "two"}, lambda record: record.upper()) == {"a": "ONE", "b": "TWO"}


@pytest.mark.asyncio
async def test_repository_product_profile_reads_build_typed_queries() -> None:
    queries = SimpleNamespace(
        get_session_profile=AsyncMock(return_value="record"),
        get_session_profiles_batch=AsyncMock(return_value={"conv-1": "record-a"}),
        _list_session_profiles_query=AsyncMock(return_value=["record-a", "record-b"]),
    )

    class _Repo(RepositoryProductProfileReadMixin):
        def __init__(self, queries: object) -> None:
            self.queries = queries

    repo = _Repo(queries)

    with patch(
        "polylogue.storage.repository.product.profile_reads.hydrate_session_profile",
        side_effect=lambda record: f"profile:{record}",
    ):
        assert await repo.get_session_profile_record("conv-1") == "record"
        assert await repo.get_session_profile("conv-1") == "profile:record"
        assert await repo.get_session_profiles_batch(["conv-1"]) == {"conv-1": "profile:record-a"}
        assert await repo.list_session_profiles(
            provider="claude-code",
            since="2026-01-01",
            until="2026-01-02",
            first_message_since="2026-01-01T00:00:00Z",
            first_message_until="2026-01-02T00:00:00Z",
            session_date_since="2026-01-01",
            session_date_until="2026-01-02",
            min_wallclock_seconds=300,
            max_wallclock_seconds=900,
            sort="wallclock",
            tier="evidence",
            limit=5,
            offset=2,
            query="refactor",
        ) == ["profile:record-a", "profile:record-b"]
        assert await repo.list_session_profile_records(query="refactor") == ["record-a", "record-b"]
        assert await repo.get_session_enrichment_record("conv-1") == "record"
        assert await repo.list_session_enrichment_records(query="enrichment") == ["record-a", "record-b"]

    list_query = queries._list_session_profiles_query.await_args_list[0].args[0]
    assert list_query.provider == "claude-code"
    assert list_query.first_message_since == "2026-01-01T00:00:00Z"
    assert list_query.session_date_until == "2026-01-02"
    assert list_query.min_wallclock_seconds == 300
    assert list_query.max_wallclock_seconds == 900
    assert list_query.sort == "wallclock"
    assert list_query.tier == "evidence"
    assert list_query.limit == 5
    assert list_query.offset == 2
    assert list_query.query == "refactor"

    enrichment_query = queries._list_session_profiles_query.await_args_list[-1].args[0]
    assert enrichment_query.tier == "enrichment"
    assert enrichment_query.query == "enrichment"


@pytest.mark.asyncio
async def test_repository_product_thread_and_timeline_reads_build_typed_queries() -> None:
    queries = SimpleNamespace(
        get_work_thread=AsyncMock(return_value="thread-record"),
        _list_work_threads_query=AsyncMock(return_value=["thread-record"]),
        get_session_work_events=AsyncMock(return_value=["event-record"]),
        get_session_phases=AsyncMock(return_value=["phase-record"]),
        _list_session_work_events_query=AsyncMock(return_value=["event-record"]),
        _list_session_phases_query=AsyncMock(return_value=["phase-record"]),
    )

    class _Repo(RepositoryProductThreadReadMixin, RepositoryProductTimelineReadMixin):
        def __init__(self, queries: object) -> None:
            self.queries = queries

    repo = _Repo(queries)

    with (
        patch(
            "polylogue.storage.repository.product.thread_reads.hydrate_work_thread",
            side_effect=lambda record: f"thread:{record}",
        ),
        patch(
            "polylogue.storage.repository.product.timeline_reads.hydrate_work_event",
            side_effect=lambda record: f"event:{record}",
        ),
        patch(
            "polylogue.storage.repository.product.timeline_reads.hydrate_session_phase",
            side_effect=lambda record: f"phase:{record}",
        ),
    ):
        assert await repo.get_work_thread_record("thread-1") == "thread-record"
        assert await repo.get_work_thread("thread-1") == "thread:thread-record"
        assert await repo.list_work_threads(
            since="2026-01-01", until="2026-01-02", limit=3, offset=1, query="repo"
        ) == ["thread:thread-record"]
        assert await repo.list_work_thread_records(query="repo") == ["thread-record"]

        assert await repo.get_session_work_event_records("conv-1") == ["event-record"]
        assert await repo.get_session_phase_records("conv-1") == ["phase-record"]
        assert await repo.get_session_work_events("conv-1") == ["event:event-record"]
        assert await repo.get_session_phases("conv-1") == ["phase:phase-record"]
        assert await repo.list_session_work_events(
            conversation_id="conv-1",
            provider="claude-code",
            since="2026-01-01",
            until="2026-01-02",
            kind="implementation",
            limit=4,
            offset=2,
            query="editor",
        ) == ["event:event-record"]
        assert await repo.list_session_work_event_records(query="editor") == ["event-record"]
        assert await repo.list_session_phases(
            conversation_id="conv-1",
            provider="claude-code",
            since="2026-01-01",
            until="2026-01-02",
            kind="planning",
            limit=2,
            offset=1,
        ) == ["phase:phase-record"]
        assert await repo.list_session_phase_records(kind="planning") == ["phase-record"]

    work_thread_query = queries._list_work_threads_query.await_args_list[0].args[0]
    assert work_thread_query.since == "2026-01-01"
    assert work_thread_query.offset == 1
    assert work_thread_query.query == "repo"

    timeline_query = queries._list_session_work_events_query.await_args_list[0].args[0]
    assert timeline_query.conversation_id == "conv-1"
    assert timeline_query.provider == "claude-code"
    assert timeline_query.kind == "implementation"
    assert timeline_query.query == "editor"


@pytest.mark.asyncio
async def test_repository_product_summary_reads_build_typed_queries() -> None:
    queries = SimpleNamespace(
        _list_session_tag_rollup_rows_query=AsyncMock(return_value=["tag-row"]),
        _list_day_session_summaries_query=AsyncMock(return_value=["day-row"]),
    )

    class _Repo(RepositoryProductSummaryReadMixin):
        def __init__(self, queries: object) -> None:
            self.queries = queries

    repo = _Repo(queries)

    assert await repo.list_session_tag_rollup_records(
        provider="claude-code",
        since="2026-01-01",
        until="2026-01-02",
        query="tag",
    ) == ["tag-row"]
    assert await repo.list_day_session_summary_records(
        provider="claude-code",
        since="2026-01-01",
        until="2026-01-02",
    ) == ["day-row"]

    tag_query = queries._list_session_tag_rollup_rows_query.await_args.args[0]
    day_query = queries._list_day_session_summaries_query.await_args.args[0]
    assert tag_query.provider == "claude-code"
    assert tag_query.query == "tag"
    assert day_query.provider == "claude-code"
    assert day_query.until == "2026-01-02"


class _ConnectionContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    async def __aenter__(self) -> object:
        return self._conn

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _Backend:
    def __init__(self, conn: object) -> None:
        self._conn = conn
        self.transaction_depth = 7

    def connection(self) -> _ConnectionContext:
        return _ConnectionContext(self._conn)


async def _aiter(items: list[object]) -> AsyncIterator[object]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_repository_raw_forwards_query_and_mutation_calls() -> None:
    conn = object()

    class _Repo(RepositoryRawMixin):
        def __init__(self, backend: object) -> None:
            self._backend = backend

    repo = _Repo(_Backend(conn))

    with (
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.save_raw_conversation",
            new=AsyncMock(return_value=True),
        ) as mock_save_raw,
        patch(
            "polylogue.storage.repository.raw.repository_raw.artifacts_q.save_artifact_observation",
            new=AsyncMock(return_value=True),
        ) as mock_save_artifact,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_conversation",
            new=AsyncMock(return_value="raw-record"),
        ) as mock_get_raw,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.apply_raw_state_update", new=AsyncMock()
        ) as mock_update_state,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.mark_raw_parsed", new=AsyncMock()
        ) as mock_mark_parsed,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.mark_raw_validated", new=AsyncMock()
        ) as mock_mark_validated,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_known_source_mtimes",
            new=AsyncMock(return_value={"inbox": "1"}),
        ) as mock_mtimes,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.reset_parse_status",
            new=AsyncMock(return_value=3),
        ) as mock_reset_parse,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.reset_validation_status",
            new=AsyncMock(return_value=4),
        ) as mock_reset_validation,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_conversations_batch",
            new=AsyncMock(return_value=["a"]),
        ) as mock_batch,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_blob_sizes",
            new=AsyncMock(return_value=[("a", 12)]),
        ) as mock_blob_sizes,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_conversation_states",
            new=AsyncMock(return_value={"a": "state"}),
        ) as mock_states,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_conversation_count",
            new=AsyncMock(return_value=9),
        ) as mock_count,
    ):
        assert await repo.save_raw_conversation("record") is True
        assert await repo.save_artifact_observation("artifact") is True
        assert await repo.get_raw_conversation("raw-1") == "raw-record"
        await repo.update_raw_state("raw-1", state="state-update")
        await repo.mark_raw_parsed("raw-1", error="boom", payload_provider="chatgpt")
        await repo.mark_raw_validated(
            "raw-1",
            status="error",
            error="boom",
            drift_count=2,
            provider="chatgpt",
            mode="strict",
            payload_provider="chatgpt",
        )
        assert await repo.get_known_source_mtimes() == {"inbox": "1"}
        assert await repo.reset_parse_status(provider="chatgpt", source_names=["inbox"]) == 3
        assert await repo.reset_validation_status(provider="chatgpt", source_names=["inbox"]) == 4
        assert await repo.get_raw_conversations_batch(["raw-1"]) == ["a"]
        assert await repo.get_raw_blob_sizes(["raw-1"]) == [("a", 12)]
        assert await repo.get_raw_conversation_states(["raw-1"]) == {"a": "state"}
        assert await repo.get_raw_conversation_count("chatgpt") == 9

    mock_save_raw.assert_awaited_once_with(conn, "record", 7)
    mock_save_artifact.assert_awaited_once_with(conn, "artifact", 7)
    mock_get_raw.assert_awaited_once_with(conn, "raw-1")
    mock_update_state.assert_awaited_once_with(conn, "raw-1", state="state-update", transaction_depth=7)
    mock_mark_parsed.assert_awaited_once_with(
        conn,
        "raw-1",
        error="boom",
        payload_provider="chatgpt",
        transaction_depth=7,
    )
    mock_mark_validated.assert_awaited_once_with(
        conn,
        "raw-1",
        status="error",
        error="boom",
        drift_count=2,
        provider="chatgpt",
        mode="strict",
        payload_provider="chatgpt",
        transaction_depth=7,
    )
    mock_mtimes.assert_awaited_once_with(conn)
    mock_reset_parse.assert_awaited_once_with(conn, provider="chatgpt", source_names=["inbox"], transaction_depth=7)
    mock_reset_validation.assert_awaited_once_with(
        conn,
        provider="chatgpt",
        source_names=["inbox"],
        transaction_depth=7,
    )
    mock_batch.assert_awaited_once_with(conn, ["raw-1"])
    mock_blob_sizes.assert_awaited_once_with(conn, ["raw-1"])
    mock_states.assert_awaited_once_with(conn, ["raw-1"])
    mock_count.assert_awaited_once_with(conn, provider="chatgpt")


@pytest.mark.asyncio
async def test_repository_raw_streams_iterators() -> None:
    conn = object()

    class _Repo(RepositoryRawMixin):
        def __init__(self, backend: object) -> None:
            self._backend = backend

    repo = _Repo(_Backend(conn))

    with (
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.iter_raw_conversations",
            return_value=_aiter(["raw-a", "raw-b"]),
        ) as mock_iter_raw,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.iter_raw_headers",
            return_value=_aiter([("raw-a", 1), ("raw-b", 2)]),
        ) as mock_iter_headers,
    ):
        conversations = [record async for record in repo.iter_raw_conversations(provider="chatgpt", limit=2)]
        headers = [
            header
            async for header in repo.iter_raw_headers(
                source_names=["inbox"],
                provider_name="chatgpt",
                require_unparsed=True,
                require_unvalidated=True,
                validation_statuses=["error"],
                page_size=50,
            )
        ]

    assert conversations == ["raw-a", "raw-b"]
    assert headers == [("raw-a", 1), ("raw-b", 2)]
    mock_iter_raw.assert_called_once_with(conn, provider="chatgpt", limit=2)
    mock_iter_headers.assert_called_once_with(
        conn,
        source_names=["inbox"],
        provider_name="chatgpt",
        require_unparsed=True,
        require_unvalidated=True,
        validation_statuses=["error"],
        page_size=50,
    )
