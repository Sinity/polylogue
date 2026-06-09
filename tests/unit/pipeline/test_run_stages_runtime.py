# mypy: disable-error-code="no-untyped-def,call-arg,arg-type,attr-defined"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from polylogue.config import Source
from polylogue.pipeline import run_stages
from polylogue.pipeline.run_stages import (
    EmbedStageOutcome,
    IndexStageOutcome,
    MaterializeStageOutcome,
)


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb


async def _aiter(*values: str):
    for value in values:
        yield value


def _backend(
    *,
    conn: object | None = None,
    status: object | None = None,
    count: int = 0,
    iter_values: tuple[str, ...] = (),
) -> object:
    connection = conn if conn is not None else SimpleNamespace()
    return SimpleNamespace(
        get_session_insight_status=AsyncMock(return_value=status),
        count_session_ids=AsyncMock(return_value=count),
        iter_session_ids=lambda **kwargs: _aiter(*iter_values),
        connection=lambda: _AsyncContext(connection),
    )


@pytest.mark.asyncio
async def test_execute_schema_generation_stage_counts_successes_and_failures() -> None:
    with patch("polylogue.schemas.operator.schema_inference.generate_all_schemas") as generate_all_schemas:
        generate_all_schemas.return_value = [SimpleNamespace(success=True), SimpleNamespace(success=False)]

        outcome = await run_stages.execute_schema_generation_stage()

    assert outcome.generated == 1
    assert outcome.failed == 1


@pytest.mark.asyncio
async def test_execute_materialize_stage_covers_noop_and_incremental_refresh_paths() -> None:
    empty_outcome = await run_stages.execute_materialize_stage(
        stage="all",
        source_names=None,
        processed_ids=set(),
        backend=_backend(),
    )
    assert empty_outcome == MaterializeStageOutcome(item_count=0, rebuilt=False)

    backend = _backend(status=SimpleNamespace(total_sessions=2, profile_row_count=0))
    progress = MagicMock()
    with patch(
        "polylogue.pipeline.services.ingest_batch.refresh_session_insights_bulk",
        new=AsyncMock(return_value={"mode": "refresh"}),
    ) as refresh:
        outcome = await run_stages.execute_materialize_stage(
            stage="all",
            source_names=None,
            processed_ids={"conv-b", "conv-a"},
            backend=backend,
            progress_callback=progress,
        )

    assert outcome.rebuilt is False
    assert outcome.item_count == 2
    assert outcome.observation == {"mode": "refresh"}
    progress.assert_called_with(0, desc="Materializing: 0/2")
    assert refresh.await_args is not None
    assert refresh.await_args.args[1] == ["conv-a", "conv-b"]
    backend.get_session_insight_status.assert_not_awaited()

    reprocess_backend = _backend(status=SimpleNamespace(total_sessions=2, profile_row_count=0))
    reprocess_progress = MagicMock()
    with patch(
        "polylogue.pipeline.services.ingest_batch.refresh_session_insights_bulk",
        new=AsyncMock(return_value={"mode": "refresh"}),
    ) as refresh:
        reprocess_outcome = await run_stages.execute_materialize_stage(
            stage="reprocess",
            source_names=None,
            processed_ids={"conv-b", "conv-a"},
            backend=reprocess_backend,
            progress_callback=reprocess_progress,
        )

    assert reprocess_outcome.rebuilt is False
    assert reprocess_outcome.item_count == 2
    assert reprocess_outcome.observation == {"mode": "refresh"}
    reprocess_progress.assert_called_with(0, desc="Materializing: 0/2")
    assert refresh.await_args is not None
    assert refresh.await_args.args[1] == ["conv-a", "conv-b"]
    reprocess_backend.get_session_insight_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_materialize_stage_covers_refresh_scoped_and_unscoped_paths() -> None:
    backend = _backend(
        status=SimpleNamespace(total_sessions=3, profile_row_count=2),
    )
    with patch(
        "polylogue.pipeline.services.ingest_batch.refresh_session_insights_bulk",
        new=AsyncMock(return_value={"mode": "refresh"}),
    ) as refresh:
        all_outcome = await run_stages.execute_materialize_stage(
            stage="all",
            source_names=None,
            processed_ids={"conv-1"},
            backend=backend,
        )

    assert all_outcome.rebuilt is False
    assert all_outcome.observation == {"mode": "refresh"}
    assert refresh.await_args is not None
    assert refresh.await_args.args[1] == ["conv-1"]

    zero_scoped = await run_stages.execute_materialize_stage(
        stage="materialize",
        source_names=("drive",),
        processed_ids=set(),
        backend=_backend(count=0),
    )
    assert zero_scoped == MaterializeStageOutcome(item_count=0, rebuilt=False)

    scoped_backend = _backend(count=2, iter_values=("conv-1", "conv-2"))
    with patch(
        "polylogue.pipeline.services.ingest_batch.refresh_session_insights_bulk",
        new=AsyncMock(return_value={"sessions": 2}),
    ) as refresh:
        scoped = await run_stages.execute_materialize_stage(
            stage="materialize",
            source_names=("drive",),
            processed_ids=set(),
            backend=scoped_backend,
        )

    assert scoped.item_count == 2
    assert scoped.observation == {"sessions": 2}
    assert refresh.await_args is not None
    assert refresh.await_args.args[1] == ["conv-1", "conv-2"]

    cursor = SimpleNamespace(fetchone=AsyncMock(return_value=(3,)))
    conn = SimpleNamespace(execute=AsyncMock(return_value=cursor), commit=AsyncMock())
    backend = _backend(conn=conn)
    counts = SimpleNamespace(
        profiles=3,
        work_events=1,
        phases=1,
        threads=1,
        tag_rollups=0,
    )
    with patch(
        "polylogue.storage.insights.session.rebuild.rebuild_session_insights_async", new=AsyncMock(return_value=counts)
    ):
        unscoped = await run_stages.execute_materialize_stage(
            stage="materialize",
            source_names=None,
            processed_ids=set(),
            backend=backend,
        )

    assert unscoped.rebuilt is True
    assert unscoped.item_count == 3
    assert unscoped.observation is not None
    assert unscoped.observation["mode"] == "rebuild"

    other_stage = await run_stages.execute_materialize_stage(
        stage="parse",
        source_names=None,
        processed_ids={"conv-1"},
        backend=_backend(),
    )
    assert other_stage == MaterializeStageOutcome(item_count=0, rebuilt=False)


@pytest.mark.asyncio
async def test_execute_ingest_stage_routes_active_archive_to_native_writer(tmp_path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path
    source_path = tmp_path / "gemini-session.json"
    source_path.write_text(
        """
        {
          "sessionId": "stage-v1-parse",
          "projectHash": "project-hash",
          "startTime": "2026-04-08T20:45:00.000Z",
          "lastUpdated": "2026-04-08T20:47:00.000Z",
          "kind": "chat",
          "summary": "Stage Native Parse",
          "messages": [
            {"id": "u1", "timestamp": "2026-04-08T20:45:01.000Z", "type": "user", "content": ["stage archive needle"]}
          ]
        }
        """,
        encoding="utf-8",
    )
    with ArchiveStore(archive_root):
        pass

    with patch("polylogue.pipeline.services.parsing.ParsingService") as parsing_service:
        result = await run_stages.execute_ingest_stage(
            config=SimpleNamespace(archive_root=archive_root, db_path=tmp_path / "index.db"),
            repository=SimpleNamespace(),
            archive_root=archive_root,
            sources=[Source(name="gemini-cli", path=source_path)],
            stage="all",
        )

    parsing_service.assert_not_called()
    assert result.parse_result.counts["sessions"] == 1
    observation = result.parse_result.batch_observations[-1]
    assert observation["archive_write_mode"] == "archive"
    assert observation["archive_root"] == str(archive_root)
    assert observation["archive_write_targets"] == ["source.db", "index.db"]
    assert observation["archive_source_rows"] == 1
    assert observation["archive_index_rows"] == 1
    with ArchiveStore.open_existing(archive_root) as archive_db:
        assert archive_db.search_blocks("needle")


@pytest.mark.asyncio
async def test_execute_index_stage_covers_parse_index_reprocess_and_error_paths() -> None:
    index_service = SimpleNamespace(
        update_index=AsyncMock(return_value=True),
        rebuild_index=AsyncMock(return_value=True),
        get_index_status=AsyncMock(return_value={"exists": True}),
    )
    with patch("polylogue.pipeline.services.indexing.IndexService", return_value=index_service):
        parse_no_ids = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="parse",
            source_names=None,
            processed_ids=set(),
            backend=_backend(),
        )
        assert parse_no_ids == IndexStageOutcome(indexed=False, item_count=0)

        parse_ids = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="parse",
            source_names=None,
            processed_ids={"conv-1"},
            backend=_backend(),
        )
        assert parse_ids == IndexStageOutcome(indexed=True, item_count=1)

        scoped_backend = _backend(count=2, iter_values=("conv-1", "conv-2"))
        index_ids = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="index",
            source_names=("drive",),
            processed_ids=set(),
            backend=scoped_backend,
        )
        assert index_ids == IndexStageOutcome(indexed=True, item_count=2)

        update_calls_before_explicit_index = index_service.update_index.await_count
        explicit_index_delta = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="index",
            source_names=None,
            processed_ids={"conv-delta"},
            backend=_backend(),
        )
        assert explicit_index_delta == IndexStageOutcome(indexed=True, item_count=1)
        assert index_service.update_index.await_count == update_calls_before_explicit_index + 1

        rebuild_backend = _backend(count=4)
        rebuild = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="index",
            source_names=None,
            processed_ids=set(),
            backend=rebuild_backend,
        )
        assert rebuild == IndexStageOutcome(indexed=True, item_count=4)

        index_service.get_index_status.return_value = {"exists": False}
        all_rebuild = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="all",
            source_names=None,
            processed_ids={"conv-1", "conv-2"},
            backend=_backend(),
        )
        assert all_rebuild == IndexStageOutcome(indexed=True, item_count=2)

        index_service.get_index_status.return_value = {"exists": True}
        update_calls_before_reprocess = index_service.update_index.await_count
        reprocess = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="reprocess",
            source_names=None,
            processed_ids={"conv-9"},
            backend=_backend(),
        )
        assert reprocess == IndexStageOutcome(indexed=True, item_count=0)
        assert index_service.update_index.await_count == update_calls_before_reprocess

        index_service.update_index.side_effect = RuntimeError("bad index")
        error = await run_stages.execute_index_stage(
            config=SimpleNamespace(),
            stage="parse",
            source_names=None,
            processed_ids={"conv-bad"},
            backend=_backend(),
        )
        assert error.indexed is False
        assert error.error == "RuntimeError: bad index"


@pytest.mark.asyncio
async def test_execute_embed_stage_covers_stats_errors_single_and_batch_paths() -> None:
    config = SimpleNamespace()

    stats_payload = {
        "status": "complete",
        "total_sessions": 5,
        "embedded_sessions": 5,
        "pending_sessions": 0,
    }
    with patch(
        "polylogue.storage.embeddings.status_payload.embedding_status_payload", return_value=stats_payload
    ) as show_stats:
        stats = await run_stages.execute_embed_stage(
            config=config, backend=SimpleNamespace(), stats_only=True, json_output=True
        )
    assert stats == EmbedStageOutcome(embedded_count=0, error_count=0, stats_only=True)
    show_stats.assert_called_once()

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(click.Abort):
            await run_stages.execute_embed_stage(config=config, backend=SimpleNamespace())

    with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=True):
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=None):
            with pytest.raises(click.Abort):
                await run_stages.execute_embed_stage(config=config, backend=SimpleNamespace())

    provider = SimpleNamespace(model="voyage-4")
    from polylogue.storage.embeddings.materialization import EmbedSessionOutcome

    embedded_outcome = EmbedSessionOutcome(status="embedded", session_id="conv-1", embedded_message_count=2)
    with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=True):
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=provider):
            with patch(
                "polylogue.storage.embeddings.materialization.embed_session_sync",
                return_value=embedded_outcome,
            ) as embed_one:
                with patch("polylogue.storage.repository.SessionRepository", return_value="repo"):
                    single = await run_stages.execute_embed_stage(
                        config=config,
                        backend=SimpleNamespace(),
                        session_id="conv-1",
                        model="voyage-4-large",
                    )
    assert provider.model == "voyage-4-large"
    assert single == EmbedStageOutcome(embedded_count=1, error_count=0)
    embed_one.assert_called_once()

    provider = SimpleNamespace(model="voyage-4")
    with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=True):
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=provider):
            with patch(
                "polylogue.storage.embeddings.materialization.iter_pending_sessions",
                return_value=[],
            ) as iter_pending:
                with patch("polylogue.storage.repository.SessionRepository", return_value="repo"):
                    batch = await run_stages.execute_embed_stage(
                        config=config,
                        backend=SimpleNamespace(),
                        rebuild=True,
                        limit=3,
                    )
    assert batch == EmbedStageOutcome(embedded_count=0, error_count=0)
    iter_pending.assert_called_once()
    assert iter_pending.call_args.kwargs["rebuild"] is True
    assert iter_pending.call_args.kwargs["max_sessions"] == 3
