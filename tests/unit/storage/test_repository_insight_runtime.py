# mypy: disable-error-code="assignment,comparison-overlap,arg-type"

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.storage.derived.insight_read_support import hydrate_mapping, hydrate_optional, hydrate_sequence
from polylogue.storage.repository.insight.profile_reads import RepositoryInsightProfileReadMixin
from polylogue.storage.repository.raw.repository_raw import RepositoryRawMixin


def test_insight_read_support_hydrates_optional_sequence_and_mapping() -> None:
    assert hydrate_optional(None, lambda record: f"hydrated:{record}") is None
    assert hydrate_optional("record", lambda record: f"hydrated:{record}") == "hydrated:record"
    assert hydrate_sequence(["a", "b"], lambda record: record.upper()) == ["A", "B"]
    assert hydrate_mapping({"a": "one", "b": "two"}, lambda record: record.upper()) == {"a": "ONE", "b": "TWO"}


@pytest.mark.asyncio
async def test_repository_profile_batch_read_hydrates_records() -> None:
    """``get_session_profiles_batch`` is the one repository-level profile read
    with a production caller (``api/archive.py`` analyzed-session batches).

    Anti-vacuity: dropping the hydration step (returning the raw record map)
    makes this red -- the caller would receive ``SessionProfileRecord`` rows
    where it expects hydrated ``SessionProfile`` objects.
    """
    queries = SimpleNamespace(
        get_session_profiles_batch=AsyncMock(return_value={"conv-1": "record-a"}),
    )

    class _Repo(RepositoryInsightProfileReadMixin):
        def __init__(self, queries: object) -> None:
            self.queries = queries

    repo = _Repo(queries)

    with patch(
        "polylogue.storage.repository.insight.profile_reads.hydrate_session_profile",
        side_effect=lambda record: f"profile:{record}",
    ):
        assert await repo.get_session_profile_records_batch(["conv-1"]) == {"conv-1": "record-a"}
        assert await repo.get_session_profiles_batch(["conv-1"]) == {"conv-1": "profile:record-a"}

    queries.get_session_profiles_batch.assert_awaited_with(["conv-1"])


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

    def read_connection(self) -> _ConnectionContext:
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
            "polylogue.storage.repository.raw.repository_raw.raw_queries.save_raw_session",
            new=AsyncMock(return_value=True),
        ) as mock_save_raw,
        patch(
            "polylogue.storage.repository.raw.repository_raw.artifacts_q.save_artifact_observation",
            new=AsyncMock(return_value=True),
        ) as mock_save_artifact,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_session",
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
            "polylogue.storage.repository.raw.repository_raw.cursor_queries.get_known_source_cursors",
            new=AsyncMock(return_value={"inbox": {"st_dev": 1, "st_ino": 2, "st_size": 3, "mtime_ns": 4}}),
        ) as mock_cursors,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.reset_parse_status",
            new=AsyncMock(return_value=3),
        ) as mock_reset_parse,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.reset_validation_status",
            new=AsyncMock(return_value=4),
        ) as mock_reset_validation,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_sessions_batch",
            new=AsyncMock(return_value=["a"]),
        ) as mock_batch,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_blob_sizes",
            new=AsyncMock(return_value=[("a", 12)]),
        ) as mock_blob_sizes,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_session_states",
            new=AsyncMock(return_value={"a": "state"}),
        ) as mock_states,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.get_raw_session_count",
            new=AsyncMock(return_value=9),
        ) as mock_count,
    ):
        assert await repo.save_raw_session("record") is True
        assert await repo.save_artifact_observation("artifact") is True
        assert await repo.get_raw_session("raw-1") == "raw-record"
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
        assert await repo.get_known_source_cursors() == {
            "inbox": {"st_dev": 1, "st_ino": 2, "st_size": 3, "mtime_ns": 4}
        }
        assert await repo.reset_parse_status(origin="chatgpt", source_names=["inbox"]) == 3
        assert await repo.reset_validation_status(origin="chatgpt", source_names=["inbox"]) == 4
        assert await repo.get_raw_sessions_batch(["raw-1"]) == ["a"]
        assert await repo.get_raw_blob_sizes(["raw-1"]) == [("a", 12)]
        assert await repo.get_raw_session_states(["raw-1"]) == {"a": "state"}
        assert await repo.get_raw_session_count("chatgpt") == 9

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
    mock_cursors.assert_awaited_once_with(conn)
    mock_reset_parse.assert_awaited_once_with(conn, origin="chatgpt", source_names=["inbox"], transaction_depth=7)
    mock_reset_validation.assert_awaited_once_with(
        conn,
        origin="chatgpt",
        source_names=["inbox"],
        transaction_depth=7,
    )
    mock_batch.assert_awaited_once_with(conn, ["raw-1"])
    mock_blob_sizes.assert_awaited_once_with(conn, ["raw-1"])
    mock_states.assert_awaited_once_with(conn, ["raw-1"])
    mock_count.assert_awaited_once_with(conn, origin="chatgpt")


@pytest.mark.asyncio
async def test_repository_raw_streams_iterators() -> None:
    conn = object()

    class _Repo(RepositoryRawMixin):
        def __init__(self, backend: object) -> None:
            self._backend = backend

    repo = _Repo(_Backend(conn))

    with (
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.iter_raw_sessions",
            return_value=_aiter(["raw-a", "raw-b"]),
        ) as mock_iter_raw,
        patch(
            "polylogue.storage.repository.raw.repository_raw.raw_queries.iter_raw_headers",
            return_value=_aiter([("raw-a", 1), ("raw-b", 2)]),
        ) as mock_iter_headers,
    ):
        sessions = [record async for record in repo.iter_raw_sessions(origin="chatgpt", limit=2)]
        headers = [
            header
            async for header in repo.iter_raw_headers(
                source_paths=["inbox"],
                source_name="chatgpt",
                require_unparsed=True,
                require_unvalidated=True,
                validation_statuses=["error"],
                page_size=50,
            )
        ]

    assert sessions == ["raw-a", "raw-b"]
    assert headers == [("raw-a", 1), ("raw-b", 2)]
    mock_iter_raw.assert_called_once_with(conn, origin="chatgpt", limit=2)
    mock_iter_headers.assert_called_once_with(
        conn,
        source_paths=["inbox"],
        source_name="chatgpt",
        require_unparsed=True,
        require_unvalidated=True,
        validation_statuses=["error"],
        page_size=50,
    )
