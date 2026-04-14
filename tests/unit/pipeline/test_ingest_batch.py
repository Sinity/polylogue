"""Focused tests for sync ingest-batch DB writes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polylogue.pipeline.services.ingest_batch import (
    _build_batch_memory_observation,
    _drain_ready_conversation_entries,
    _failed_raw_state_update,
    _IngestBatchSummary,
    _iter_ingest_results_sync,
    _persist_batch_raw_state_updates,
    _RawIngestOutcome,
    _select_ingest_worker_count,
    _successful_raw_state_update,
    _topo_sort_conversation_entries,
    _unattributed_batch_elapsed_s,
    _write_conversation,
    refresh_session_products_bulk,
)
from polylogue.pipeline.services.ingest_worker import ConversationData, IngestRecordResult
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.state_views import RawConversationStateUpdate
from polylogue.storage.store import _make_ref_id


def _conversation_data(
    conversation_id: str,
    *,
    content_hash: str,
    parent_conversation_id: str | None = None,
    message_tuples: list[tuple] | None = None,
    block_tuples: list[tuple] | None = None,
    stats_tuple: tuple | None = None,
    attachment_tuples: list[tuple] | None = None,
    attachment_ref_tuples: list[tuple] | None = None,
) -> ConversationData:
    conversation_tuple = (
        conversation_id,
        "codex",
        conversation_id.split(":", 1)[-1],
        "Conversation",
        "2026-04-02T00:00:00Z",
        "2026-04-02T00:00:00Z",
        0.0,
        content_hash,
        None,
        "{}",
        1,
        parent_conversation_id,
        None,
        None,
    )
    return ConversationData(
        conversation_id=conversation_id,
        content_hash=content_hash,
        provider_name="codex",
        conversation_tuple=conversation_tuple,
        message_tuples=list(message_tuples or []),
        block_tuples=list(block_tuples or []),
        stats_tuple=stats_tuple or (),
        attachment_tuples=list(attachment_tuples or []),
        attachment_ref_tuples=list(attachment_ref_tuples or []),
    )


def _message_tuple(
    message_id: str,
    conversation_id: str,
    *,
    role: str,
    text: str,
    content_hash: str,
    sort_key: float,
) -> tuple:
    return (
        message_id,
        conversation_id,
        message_id,
        role,
        text,
        sort_key,
        content_hash,
        1,
        None,
        0,
        "codex",
        len(text.split()),
        0,
        0,
    )


def _block_tuple(
    *,
    block_id: str,
    message_id: str,
    conversation_id: str,
    block_index: int,
    text: str,
) -> tuple:
    return (
        block_id,
        message_id,
        conversation_id,
        block_index,
        "text",
        text,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _attachment_tuple(attachment_id: str, *, mime_type: str = "image/png") -> tuple:
    return (
        attachment_id,
        mime_type,
        1024,
        None,
        0,
        None,
    )


def _attachment_ref_tuple(
    attachment_id: str,
    conversation_id: str,
    message_id: str,
) -> tuple:
    return (
        _make_ref_id(attachment_id, conversation_id, message_id),
        attachment_id,
        conversation_id,
        message_id,
        None,
    )


def test_topo_sort_conversation_entries_orders_parent_before_child() -> None:
    parent = _conversation_data("codex:parent", content_hash="hash-parent")
    child = _conversation_data(
        "codex:child",
        content_hash="hash-child",
        parent_conversation_id="codex:parent",
    )

    ordered = _topo_sort_conversation_entries(
        [
            ("raw-child", child),
            ("raw-parent", parent),
        ]
    )

    assert [entry[1].conversation_id for entry in ordered] == [
        "codex:parent",
        "codex:child",
    ]


def test_write_conversation_clears_missing_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        child = _conversation_data(
            "codex:child",
            content_hash="hash-child",
            parent_conversation_id="codex:missing-parent",
        )

        _write_conversation(conn, child)
        conn.commit()

        row = conn.execute(
            "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
            ("codex:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_conversation_id"] is None


def test_write_conversation_preserves_existing_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        parent = _conversation_data("codex:parent", content_hash="hash-parent")
        child = _conversation_data(
            "codex:child",
            content_hash="hash-child",
            parent_conversation_id="codex:parent",
        )

        _write_conversation(conn, parent)
        _write_conversation(conn, child)
        conn.commit()

        row = conn.execute(
            "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
            ("codex:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_conversation_id"] == "codex:parent"


def test_write_conversation_replaces_runtime_rows_on_content_change(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        v1 = _conversation_data(
            "codex:replace",
            content_hash="hash-v1",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:replace",
                    role="user",
                    text="first",
                    content_hash="msg-v1-1",
                    sort_key=1.0,
                ),
                _message_tuple(
                    "msg-2",
                    "codex:replace",
                    role="assistant",
                    text="second",
                    content_hash="msg-v1-2",
                    sort_key=2.0,
                ),
            ],
            block_tuples=[
                _block_tuple(
                    block_id="blk-msg-1-0",
                    message_id="msg-1",
                    conversation_id="codex:replace",
                    block_index=0,
                    text="alpha",
                ),
                _block_tuple(
                    block_id="blk-msg-1-1",
                    message_id="msg-1",
                    conversation_id="codex:replace",
                    block_index=1,
                    text="beta",
                ),
            ],
            stats_tuple=("codex:replace", "codex", 2, 2, 0, 0),
            attachment_tuples=[
                _attachment_tuple("att-1"),
                _attachment_tuple("att-2", mime_type="image/jpeg"),
            ],
            attachment_ref_tuples=[
                _attachment_ref_tuple("att-1", "codex:replace", "msg-1"),
                _attachment_ref_tuple("att-2", "codex:replace", "msg-2"),
            ],
        )
        changed, counts = _write_conversation(conn, v1)
        assert changed is True
        assert counts["messages"] == 2

        v2 = _conversation_data(
            "codex:replace",
            content_hash="hash-v2",
            message_tuples=[
                _message_tuple(
                    "msg-1",
                    "codex:replace",
                    role="user",
                    text="first updated",
                    content_hash="msg-v2-1",
                    sort_key=1.0,
                )
            ],
            block_tuples=[
                _block_tuple(
                    block_id="blk-msg-1-0-v2",
                    message_id="msg-1",
                    conversation_id="codex:replace",
                    block_index=0,
                    text="alpha updated",
                )
            ],
            stats_tuple=("codex:replace", "codex", 1, 2, 0, 0),
            attachment_tuples=[_attachment_tuple("att-1")],
            attachment_ref_tuples=[_attachment_ref_tuple("att-1", "codex:replace", "msg-1")],
        )
        changed, counts = _write_conversation(conn, v2)
        assert changed is True
        assert counts["messages"] == 1
        conn.commit()

        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                ("codex:replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM content_blocks WHERE conversation_id = ?",
                ("codex:replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM attachment_refs WHERE conversation_id = ?",
                ("codex:replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT message_id FROM attachment_refs WHERE conversation_id = ? AND attachment_id = ?",
                ("codex:replace", "att-1"),
            ).fetchone()[0]
            == "msg-1"
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM attachments WHERE attachment_id = ?",
                ("att-2",),
            ).fetchone()[0]
            == 0
        )
        stats_row = conn.execute(
            "SELECT message_count FROM conversation_stats WHERE conversation_id = ?",
            ("codex:replace",),
        ).fetchone()
        assert stats_row is not None
        assert stats_row[0] == 1


def test_iter_ingest_results_sync_runs_inline_for_single_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_records = [
        SimpleNamespace(raw_id="raw-1"),
        SimpleNamespace(raw_id="raw-2"),
    ]
    seen: list[str] = []

    def fake_ingest_record(
        raw_record,
        archive_root_str: str,
        validation_mode: str = "strict",
        measure_ingest_result_size: bool = False,
        *,
        blob_root_str: str | None = None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        seen.append(raw_record.raw_id)
        return IngestRecordResult(raw_id=raw_record.raw_id)

    def fail_process_pool_executor(*, max_workers: int):
        raise AssertionError(f"process pool should not be used for single-worker batches: {max_workers}")

    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch.ingest_record", fake_ingest_record)
    monkeypatch.setattr(
        "polylogue.pipeline.services.ingest_batch.process_pool_executor",
        fail_process_pool_executor,
    )

    results = list(
        _iter_ingest_results_sync(
            raw_records,
            archive_root_str="/tmp/archive",
            blob_root_str="/tmp/blob-store",
            validation_mode="strict",
            worker_count=1,
            measure_ingest_result_size=False,
        )
    )

    assert seen == ["raw-1", "raw-2"]
    assert [result.raw_id for result in results] == ["raw-1", "raw-2"]


def test_select_ingest_worker_count_throttles_large_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch.os.cpu_count", lambda: 16)
    raw_records = [SimpleNamespace(blob_size=60 * 1024 * 1024) for _ in range(2)]

    worker_count = _select_ingest_worker_count(raw_records, None)

    assert worker_count == 2


def test_select_ingest_worker_count_keeps_parallelism_for_small_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.ingest_batch.os.cpu_count", lambda: 16)
    raw_records = [SimpleNamespace(blob_size=4 * 1024 * 1024) for _ in range(6)]

    worker_count = _select_ingest_worker_count(raw_records, None)

    assert worker_count == 6


def test_drain_ready_conversation_entries_preserves_late_parent_fk(tmp_path: Path) -> None:
    with open_connection(tmp_path / "ingest.db") as conn:
        parent = _conversation_data("codex:parent", content_hash="hash-parent")
        child = _conversation_data(
            "codex:child",
            content_hash="hash-child",
            parent_conversation_id="codex:parent",
        )

        summary = _IngestBatchSummary()
        materialized_ids: set[str] = set()
        pending_by_parent: dict[str, list[tuple[str, ConversationData]]] = {}

        _drain_ready_conversation_entries(
            conn,
            [("raw-child", child)],
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
        )
        assert list(pending_by_parent) == ["codex:parent"]

        _drain_ready_conversation_entries(
            conn,
            [("raw-parent", parent)],
            summary=summary,
            materialized_ids=materialized_ids,
            pending_by_parent=pending_by_parent,
        )
        conn.commit()

        row = conn.execute(
            "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
            ("codex:child",),
        ).fetchone()
        assert row is not None
        assert row["parent_conversation_id"] == "codex:parent"


@pytest.mark.asyncio
async def test_refresh_session_products_bulk_dedupes_related_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = SimpleNamespace(commit=AsyncMock())

    @asynccontextmanager
    async def _connection():
        yield fake_conn

    fake_backend = SimpleNamespace(connection=_connection)

    async def _fake_apply(conn, conversation_ids: list[str], *, transaction_depth: int):
        del conn, transaction_depth
        assert conversation_ids == ["conv-1", "conv-2", "conv-3"]
        return SimpleNamespace(
            counts={
                "profiles": 3,
                "work_events": 0,
                "phases": 0,
                "threads": 0,
                "tag_rollups": 0,
                "day_summaries": 0,
            },
            affected_groups={
                ("chatgpt", "2026-04-02"),
                ("chatgpt", "2026-04-03"),
            },
            thread_root_ids={"root-a", "root-b"},
            chunk_observations=[
                {
                    "conversation_count": 3,
                    "hydrated_count": 3,
                    "profiles_written": 3,
                    "work_events_written": 0,
                    "phases_written": 0,
                    "load_ms": 12.5,
                    "hydrate_ms": 3.1,
                    "build_ms": 9.9,
                    "write_ms": 7.7,
                    "total_ms": 33.2,
                },
            ],
        )

    refresh_thread_roots = AsyncMock(return_value=2)
    refresh_aggregates = AsyncMock()

    monkeypatch.setattr(
        "polylogue.storage.session_product_refresh._apply_session_product_conversation_updates_async",
        _fake_apply,
    )
    monkeypatch.setattr(
        "polylogue.storage.session_product_refresh._refresh_thread_roots_async",
        refresh_thread_roots,
    )
    monkeypatch.setattr(
        "polylogue.storage.session_product_refresh.refresh_async_provider_day_aggregates",
        refresh_aggregates,
    )

    observation = await refresh_session_products_bulk(
        fake_backend,
        ["conv-1", "conv-2", "conv-3"],
    )

    refresh_thread_roots.assert_awaited_once()
    thread_args = refresh_thread_roots.await_args
    assert thread_args.args[0] is fake_conn
    assert thread_args.args[1] == ["root-a", "root-b"]
    assert thread_args.kwargs["transaction_depth"] == 1
    refresh_aggregates.assert_awaited_once()
    aggregate_args = refresh_aggregates.await_args
    assert aggregate_args.args[0] is fake_conn
    assert aggregate_args.args[1] == {
        ("chatgpt", "2026-04-02"),
        ("chatgpt", "2026-04-03"),
    }
    assert aggregate_args.kwargs["transaction_depth"] == 1
    fake_conn.commit.assert_awaited_once()
    assert observation is not None
    assert observation["conversations"] == 3
    assert observation["unique_thread_roots"] == 2
    assert observation["unique_provider_days"] == 2
    assert float(observation["elapsed_ms"]) >= 0.0
    assert float(observation["update_ms"]) >= 0.0
    assert float(observation["thread_refresh_ms"]) >= 0.0
    assert float(observation["aggregate_refresh_ms"]) >= 0.0
    assert observation["update_chunk_count"] == 1
    assert observation["update_slow_chunk_count"] == 0
    assert observation["update_max_chunk_ms"] == 33.2
    assert observation["update_max_chunk_load_ms"] == 12.5
    assert observation["update_max_chunk_hydrate_ms"] == 3.1
    assert observation["update_max_chunk_build_ms"] == 9.9
    assert observation["update_max_chunk_write_ms"] == 7.7


def test_successful_raw_state_update_combines_parse_and_validation_fields() -> None:
    outcome = _RawIngestOutcome(
        raw_id="raw-1",
        payload_provider="chatgpt",
        validation_status="passed",
        validation_error=None,
        parse_error=None,
        error=None,
        had_conversations=True,
    )

    state = _successful_raw_state_update(
        outcome=outcome,
        parsed_at="2026-04-02T00:00:00Z",
        validation_mode="strict",
    )

    assert state == RawConversationStateUpdate(
        parsed_at="2026-04-02T00:00:00Z",
        parse_error=None,
        payload_provider="chatgpt",
        validation_status="passed",
        validation_error=None,
        validation_mode="strict",
    )


def test_failed_raw_state_update_combines_parse_and_validation_fields() -> None:
    outcome = _RawIngestOutcome(
        raw_id="raw-1",
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        parse_error="parse failed",
        error="parse failed",
        had_conversations=False,
    )

    state = _failed_raw_state_update(
        outcome=outcome,
        error="parse failed",
        validation_mode="strict",
    )

    assert state == RawConversationStateUpdate(
        parse_error="parse failed",
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        validation_mode="strict",
    )


def test_failed_raw_state_update_keeps_validation_only_failure_out_of_parse_error() -> None:
    outcome = _RawIngestOutcome(
        raw_id="raw-1",
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        parse_error=None,
        error="schema mismatch",
        had_conversations=False,
    )

    state = _failed_raw_state_update(
        outcome=outcome,
        error="schema mismatch",
        validation_mode="strict",
    )

    assert state == RawConversationStateUpdate(
        parse_error=None,
        payload_provider="chatgpt",
        validation_status="failed",
        validation_error="schema mismatch",
        validation_mode="strict",
    )


def test_unattributed_batch_elapsed_subtracts_setup_and_teardown() -> None:
    summary = _IngestBatchSummary(
        setup_elapsed_s=0.12,
        result_wait_s=0.8,
        drain_elapsed_s=0.2,
        flush_elapsed_s=0.05,
        commit_elapsed_s=0.04,
        teardown_elapsed_s=0.31,
    )

    residual = _unattributed_batch_elapsed_s(
        elapsed_s=1.7,
        batch_summary=summary,
        raw_state_update_elapsed_s=0.08,
    )

    assert residual == pytest.approx(0.10)


def test_build_batch_memory_observation_separates_lifetime_peak_from_batch_growth() -> None:
    observation = _build_batch_memory_observation(
        rss_start_mb=512.0,
        rss_end_mb=544.5,
        peak_rss_self_start_mb=768.0,
        peak_rss_self_end_mb=1024.0,
        peak_rss_children_mb=64.0,
        max_current_rss_mb=812.2,
    )

    assert observation == {
        "rss_start_mb": 512.0,
        "rss_end_mb": 544.5,
        "rss_delta_mb": 32.5,
        "process_peak_rss_self_mb": 1024.0,
        "peak_rss_growth_mb": 256.0,
        "peak_rss_children_mb": 64.0,
        "max_current_rss_mb": 812.2,
    }


@pytest.mark.asyncio
async def test_persist_batch_raw_state_updates_uses_one_typed_update_per_raw() -> None:
    update_raw_state = AsyncMock()
    repository = SimpleNamespace(update_raw_state=update_raw_state)
    service = SimpleNamespace(repository=repository)

    @asynccontextmanager
    async def _bulk_connection():
        yield

    backend = SimpleNamespace(bulk_connection=_bulk_connection)
    outcomes = {
        "raw-success": _RawIngestOutcome(
            raw_id="raw-success",
            payload_provider="chatgpt",
            validation_status="passed",
            validation_error=None,
            parse_error=None,
            error=None,
            had_conversations=True,
        ),
        "raw-failed": _RawIngestOutcome(
            raw_id="raw-failed",
            payload_provider="codex",
            validation_status="failed",
            validation_error="bad schema",
            parse_error="parse failed",
            error="parse failed",
            had_conversations=False,
        ),
    }

    elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=outcomes,
        succeeded_raw_ids={"raw-success"},
        skipped_raw_ids=set(),
        failed_raw_ids={"raw-failed": "parse failed"},
        validation_mode="strict",
    )

    assert elapsed_s >= 0.0
    assert update_raw_state.await_count == 2
    success_call, failed_call = update_raw_state.await_args_list
    assert success_call.args == ("raw-success",)
    assert success_call.kwargs["state"].validation_status == "passed"
    assert success_call.kwargs["state"].parsed_at is not None
    assert failed_call.args == ("raw-failed",)
    assert failed_call.kwargs["state"].parse_error == "parse failed"
    assert failed_call.kwargs["state"].validation_error == "bad schema"


@pytest.mark.asyncio
async def test_persist_batch_raw_state_updates_preserves_validation_only_failure_without_quarantine() -> None:
    update_raw_state = AsyncMock()
    repository = SimpleNamespace(update_raw_state=update_raw_state)
    service = SimpleNamespace(repository=repository)

    @asynccontextmanager
    async def _bulk_connection():
        yield

    backend = SimpleNamespace(bulk_connection=_bulk_connection)
    outcomes = {
        "raw-schema-invalid": _RawIngestOutcome(
            raw_id="raw-schema-invalid",
            payload_provider="chatgpt",
            validation_status="failed",
            validation_error="bad schema",
            parse_error=None,
            error="bad schema",
            had_conversations=False,
        ),
    }

    elapsed_s = await _persist_batch_raw_state_updates(
        service,
        backend,
        outcomes=outcomes,
        succeeded_raw_ids=set(),
        skipped_raw_ids=set(),
        failed_raw_ids={"raw-schema-invalid": "bad schema"},
        validation_mode="strict",
    )

    assert elapsed_s >= 0.0
    update_raw_state.assert_awaited_once()
    state = update_raw_state.await_args.kwargs["state"]
    assert state.parse_error is None
    assert state.validation_error == "bad schema"
    assert state.validation_status == "failed"
