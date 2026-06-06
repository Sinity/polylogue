from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.storage.insights.session.aggregates import refresh_async_provider_day_aggregates
from polylogue.storage.insights.session.rebuild import (
    _SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS,
    _SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS,
    load_sync_batch,
    rebuild_session_insights_sync,
)
from polylogue.storage.insights.session.refresh import (
    _apply_session_insight_session_updates_async,
    _refresh_thread_roots_async,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import make_message, make_session, store_records


def _chunk_metric(chunk_observation: object, key: str) -> float:
    value = getattr(chunk_observation, key)
    assert isinstance(value, int | float)
    return float(value)


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_batches_hydrated_sessions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-refresh", title="Refresh Test"),
            messages=[
                make_message("conv-refresh:msg-1", "conv-refresh", text="Need help with batching"),
                make_message(
                    "conv-refresh:msg-2",
                    "conv-refresh",
                    role="assistant",
                    text="Let's batch the refresh path.",
                ),
            ],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            ["conv-refresh"],
            transaction_depth=1,
            page_size=10,
        )

    assert update.counts.profiles == 1
    assert update.thread_root_ids == {"conv-refresh"}
    assert update.affected_groups
    assert len(update.chunk_observations) == 1
    assert _chunk_metric(update.chunk_observations[0], "load_ms") >= 0.0
    assert _chunk_metric(update.chunk_observations[0], "hydrate_ms") >= 0.0
    assert _chunk_metric(update.chunk_observations[0], "build_ms") >= 0.0
    assert _chunk_metric(update.chunk_observations[0], "write_ms") >= 0.0


@pytest.mark.asyncio
async def test_session_insight_refresh_materializes_logical_session_identity(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "logical-session.db"
    with open_connection(db_path) as conn:
        for session_id, parent_id in (("root", None), ("continuation", "root"), ("fork", "continuation")):
            store_records(
                session=make_session(
                    session_id,
                    source_name="claude-code",
                    title=session_id,
                    parent_session_id=parent_id,
                    updated_at="2026-05-25T10:00:00+00:00",
                ),
                messages=[
                    make_message(
                        f"{session_id}:msg-1",
                        session_id,
                        text=f"{session_id} logical identity test",
                        timestamp="2026-05-25T10:00:00+00:00",
                    )
                ],
                attachments=[],
                conn=conn,
            )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        await _apply_session_insight_session_updates_async(
            conn,
            ["root", "continuation", "fork"],
            transaction_depth=1,
            page_size=10,
        )
        await refresh_async_provider_day_aggregates(
            conn,
            {("claude-code-session", "2026-05-25")},
            transaction_depth=1,
        )
        await conn.commit()

    with open_connection(db_path) as conn:
        profile_rows = conn.execute(
            """
            SELECT session_id, logical_session_id
            FROM session_profiles
            ORDER BY session_id
            """
        ).fetchall()
        tag_rollup = conn.execute(
            """
            SELECT session_count, logical_session_count, logical_session_ids_json
            FROM session_tag_rollups
            WHERE source_name = 'claude-code-session'
              AND bucket_day = '2026-05-25'
              AND tag = 'origin:claude-code-session'
            """
        ).fetchone()

    assert {tuple(row) for row in profile_rows} == {
        ("continuation", "root"),
        ("fork", "root"),
        ("root", "root"),
    }
    assert tag_rollup is not None
    assert int(tag_rollup["session_count"]) == 3
    assert int(tag_rollup["logical_session_count"]) == 1
    assert json.loads(tag_rollup["logical_session_ids_json"]) == ["root"]


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_counts_provider_event_compactions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-provider-events.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-provider-event", source_name="codex", title="Compaction Test"),
            messages=[
                make_message(
                    "conv-provider-event:msg-1",
                    "conv-provider-event",
                    text="Continue after compaction",
                )
            ],
            attachments=[],
            conn=conn,
        )
        conn.execute(
            """
            INSERT INTO provider_events (
                event_id,
                session_id,
                source_name,
                event_index,
                event_type,
                timestamp,
                materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "conv-provider-event:provider-event:000000",
                "conv-provider-event",
                "codex",
                0,
                "compaction",
                "2026-04-01T10:05:00+00:00",
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO provider_event_compactions (event_id, summary)
            VALUES (?, ?)
            """,
            ("conv-provider-event:provider-event:000000", "Earlier context"),
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            ["conv-provider-event"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert update.counts.profiles == 1
    with open_connection(db_path) as conn:
        row = conn.execute(
            "SELECT evidence_payload_json FROM session_profiles WHERE session_id = ?",
            ("conv-provider-event",),
        ).fetchone()

    assert row is not None
    assert json.loads(row["evidence_payload_json"])["compaction_count"] == 1


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_uses_provider_events_for_terminal_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-provider-terminal.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-provider-terminal",
                source_name="codex",
                title="Terminal Test",
                updated_at="2026-04-01T10:05:30+00:00",
            ),
            messages=[
                make_message(
                    "conv-provider-terminal:msg-1",
                    "conv-provider-terminal",
                    text="Run the command",
                )
            ],
            attachments=[],
            conn=conn,
        )
        conn.execute(
            """
            INSERT INTO provider_events (
                event_id,
                session_id,
                source_name,
                event_index,
                event_type,
                timestamp,
                materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "conv-provider-terminal:provider-event:000000",
                "conv-provider-terminal",
                "codex",
                0,
                "function_call",
                "2026-04-01T10:05:00+00:00",
                1,
            ),
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        await _apply_session_insight_session_updates_async(
            conn,
            ["conv-provider-terminal"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    with open_connection(db_path) as conn:
        row = conn.execute(
            "SELECT evidence_payload_json FROM session_profiles WHERE session_id = ?",
            ("conv-provider-terminal",),
        ).fetchone()
        latency_row = conn.execute(
            "SELECT stuck_tool_count FROM session_latency_profiles WHERE session_id = ?",
            ("conv-provider-terminal",),
        ).fetchone()

    assert row is not None
    assert json.loads(row["evidence_payload_json"])["terminal_state"] == "tool_left"
    assert latency_row is not None
    assert latency_row["stuck_tool_count"] == 0


def test_targeted_session_insight_rebuild_refreshes_only_affected_groups_and_roots(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-sync-targeted.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-chatgpt-a",
                source_name="chatgpt",
                title="ChatGPT A",
                created_at="2026-04-02T10:00:00+00:00",
                updated_at="2026-04-02T10:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-a:msg-1",
                    "conv-chatgpt-a",
                    text="ChatGPT A message",
                    timestamp="2026-04-02T10:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session(
                "conv-claude-a",
                source_name="claude-ai",
                title="Claude A",
                created_at="2026-04-03T09:00:00+00:00",
                updated_at="2026-04-03T09:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-claude-a:msg-1",
                    "conv-claude-a",
                    text="Claude A message",
                    timestamp="2026-04-03T09:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        rebuild_session_insights_sync(conn)
        conn.execute(
            "UPDATE session_tag_rollups SET search_text = ? WHERE source_name = ?",
            ("sentinel tag untouched", "claude-ai-export"),
        )
        conn.execute(
            "UPDATE work_threads SET search_text = ? WHERE thread_id = ?",
            ("sentinel thread untouched", "conv-claude-a"),
        )
        store_records(
            session=make_session(
                "conv-chatgpt-b",
                source_name="chatgpt",
                title="ChatGPT B",
                created_at="2026-04-02T11:00:00+00:00",
                updated_at="2026-04-02T11:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-b:msg-1",
                    "conv-chatgpt-b",
                    text="ChatGPT B message",
                    timestamp="2026-04-02T11:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        counts = rebuild_session_insights_sync(conn, session_ids=["conv-chatgpt-b"])
        tag_rows = conn.execute(
            """
            SELECT source_name, bucket_day, session_count, search_text
            FROM session_tag_rollups
            WHERE tag = 'origin:' || source_name
            ORDER BY source_name, bucket_day
            """
        ).fetchall()
        claude_thread = conn.execute(
            "SELECT search_text FROM work_threads WHERE thread_id = ?",
            ("conv-claude-a",),
        ).fetchone()

    assert counts.profiles == 1
    assert [(row["source_name"], row["bucket_day"], row["session_count"]) for row in tag_rows] == [
        ("chatgpt-export", "2026-04-02", 2),
        ("claude-ai-export", "2026-04-03", 1),
    ]
    assert [row["search_text"] for row in tag_rows if row["source_name"] == "claude-ai-export"] == [
        "sentinel tag untouched"
    ]
    assert claude_thread is not None
    assert claude_thread["search_text"] == "sentinel thread untouched"


def test_session_insight_rebuild_fills_profile_time_from_session_timestamp(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "profile-time-fallback.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-codex-untimestamped",
                source_name="codex",
                title="Codex untimestamped messages",
                created_at="2026-05-12T09:00:00+00:00",
                updated_at="2026-05-12T09:30:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-codex-untimestamped:msg-1",
                    "conv-codex-untimestamped",
                    text="Start work",
                    sort_key=None,
                ),
                make_message(
                    "conv-codex-untimestamped:msg-2",
                    "conv-codex-untimestamped",
                    role="assistant",
                    text="Finished work",
                    sort_key=None,
                ),
            ],
            attachments=[],
            conn=conn,
        )
        rebuild_session_insights_sync(conn)
        row = conn.execute(
            """
            SELECT first_message_at, last_message_at, canonical_session_date, evidence_payload_json
            FROM session_profiles
            WHERE session_id = ?
            """,
            ("conv-codex-untimestamped",),
        ).fetchone()

    assert row is not None
    assert row["first_message_at"] == "2026-05-12T09:00:00+00:00"
    assert row["last_message_at"] == "2026-05-12T09:30:00+00:00"
    assert row["canonical_session_date"] == "2026-05-12"
    evidence = json.loads(row["evidence_payload_json"])
    assert evidence["timestamp_coverage"] == "none"
    assert evidence["timestamp_source"] == "session_timestamp_fallback"


def test_session_insight_rebuild_materializes_message_token_costs(tmp_path: Path) -> None:
    db_path = tmp_path / "profile-token-costs.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-token-costs",
                source_name="claude-code",
                title="Token costs",
                created_at="2026-05-12T09:00:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-token-costs:msg-1",
                    "conv-token-costs",
                    text="Run the plan",
                    model_name="claude-sonnet-4-5",
                    input_tokens=1_000,
                    output_tokens=500,
                    cache_read_tokens=200,
                    cache_write_tokens=100,
                ),
            ],
            attachments=[],
            conn=conn,
        )
        rebuild_session_insights_sync(conn)
        row = conn.execute(
            """
            SELECT
                total_input_tokens,
                total_output_tokens,
                total_cache_read_tokens,
                total_cache_write_tokens,
                total_cost_usd,
                total_credit_cost,
                cost_provenance,
                per_model_cost_json
            FROM session_profiles
            WHERE session_id = ?
            """,
            ("conv-token-costs",),
        ).fetchone()

    assert row is not None
    assert row["total_input_tokens"] == 1_000
    assert row["total_output_tokens"] == 500
    assert row["total_cache_read_tokens"] == 200
    assert row["total_cache_write_tokens"] == 100
    assert row["total_cost_usd"] > 0
    assert row["total_credit_cost"] > 0
    assert row["cost_provenance"] == "provider_reported"
    per_model = json.loads(row["per_model_cost_json"])
    assert per_model[0]["provider_model_name"] == "claude-sonnet-4-5"


def test_session_insight_rebuild_preserves_session_provider_cost(tmp_path: Path) -> None:
    db_path = tmp_path / "profile-provider-cost.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-provider-cost",
                source_name="claude-code",
                title="Provider cost",
                created_at="2026-05-12T09:00:00+00:00",
                provider_meta={
                    "total_cost_usd": 1.25,
                    "total_duration_ms": 42_000,
                    "models_used": ["claude-opus-4-5"],
                },
            ),
            messages=[
                make_message(
                    "conv-provider-cost:msg-1",
                    "conv-provider-cost",
                    text="Provider reported total",
                ),
            ],
            attachments=[],
            conn=conn,
        )
        rebuild_session_insights_sync(conn)
        row = conn.execute(
            """
            SELECT total_cost_usd, total_duration_ms, cost_provenance, per_model_cost_json
            FROM session_profiles
            WHERE session_id = ?
            """,
            ("conv-provider-cost",),
        ).fetchone()

    assert row is not None
    assert row["total_cost_usd"] == 1.25
    assert row["total_duration_ms"] == 42_000
    assert row["cost_provenance"] == "provider_reported"
    per_model = json.loads(row["per_model_cost_json"])
    assert per_model[0]["provider_model_name"] == "claude-opus-4-5"


def test_session_insight_load_skips_plain_text_blocks(tmp_path: Path) -> None:
    db_path = tmp_path / "refresh-block-filter.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-blocks", source_name="codex", title="Block Filter"),
            messages=[
                make_message(
                    "conv-blocks:msg-1",
                    "conv-blocks",
                    text="Plain text is already stored on the message row.",
                    content_blocks=[{"type": "text", "text": "Plain text is already stored on the message row."}],
                ),
                make_message(
                    "conv-blocks:msg-2",
                    "conv-blocks",
                    role="assistant",
                    text="exec_command",
                    content_blocks=[
                        {
                            "type": "tool_use",
                            "name": "exec_command",
                            "id": "call-1",
                            "input": {"cmd": "git status"},
                        }
                    ],
                ),
            ],
            attachments=[],
            conn=conn,
        )
        batch = load_sync_batch(conn, ["conv-blocks"])

    assert [str(block.message_id) for block in batch.blocks] == ["conv-blocks:msg-2"]


def test_session_insight_load_includes_provider_events_for_profile_classifiers(tmp_path: Path) -> None:
    db_path = tmp_path / "refresh-provider-event-load.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-provider-event-load", source_name="codex", title="Event Load"),
            messages=[
                make_message(
                    "conv-provider-event-load:msg-1",
                    "conv-provider-event-load",
                    text="Run the tool",
                )
            ],
            attachments=[],
            conn=conn,
        )
        conn.execute(
            """
            INSERT INTO provider_events (
                event_id,
                session_id,
                source_name,
                event_index,
                event_type,
                timestamp,
                materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "conv-provider-event-load:provider-event:000000",
                "conv-provider-event-load",
                "codex",
                0,
                "function_call",
                "2026-04-01T10:05:00+00:00",
                1,
            ),
        )
        batch = load_sync_batch(conn, ["conv-provider-event-load"])
        hydrated = batch.provider_events_by_session["conv-provider-event-load"]

    assert [event.event_type for event in hydrated] == ["function_call"]


def test_session_insight_load_bounds_large_text_payloads(tmp_path: Path) -> None:
    db_path = tmp_path / "refresh-text-bound.db"
    large_message = "m" * (_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS + 100)
    large_tool_output = "o" * (_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS + 100)
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-large-text", source_name="codex", title="Large Text"),
            messages=[
                make_message(
                    "conv-large-text:msg-1",
                    "conv-large-text",
                    role="assistant",
                    text=large_message,
                    content_blocks=[
                        {
                            "type": "tool_result",
                            "tool_id": "call-1",
                            "text": large_tool_output,
                        }
                    ],
                )
            ],
            attachments=[],
            conn=conn,
        )
        batch = load_sync_batch(conn, ["conv-large-text"])

    assert batch.messages[0].text == large_message[:_SESSION_INSIGHT_MESSAGE_TEXT_PREVIEW_CHARS]
    assert batch.blocks[0].text == large_tool_output[:_SESSION_INSIGHT_BLOCK_TEXT_PREVIEW_CHARS]


def test_targeted_session_insight_rebuild_splits_large_message_batches(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rebuild-message-budget.db"
    session_ids: list[str] = []
    with open_connection(db_path) as conn:
        for index in range(3):
            session_id = f"conv-rebuild-budget-{index:02d}"
            session_ids.append(session_id)
            store_records(
                session=make_session(session_id, title=f"Rebuild Budget {index:02d}"),
                messages=[
                    make_message(
                        f"{session_id}:msg-1",
                        session_id,
                        text=f"Message for {session_id}",
                    )
                ],
                attachments=[],
                conn=conn,
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO session_stats (
                session_id,
                source_name,
                message_count,
                word_count,
                tool_use_count,
                thinking_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("conv-rebuild-budget-00", "chatgpt", 4_000, 4_000, 0, 0),
                ("conv-rebuild-budget-01", "chatgpt", 4_000, 4_000, 0, 0),
                ("conv-rebuild-budget-02", "chatgpt", 100, 100, 0, 0),
            ],
        )
        conn.commit()

        chunk_profile_counts: list[int] = []

        def record_progress(amount: int, desc: str | None = None) -> None:
            del desc
            chunk_profile_counts.append(amount)

        counts = rebuild_session_insights_sync(
            conn,
            session_ids=session_ids,
            page_size=10,
            progress_callback=record_progress,
        )

    assert counts.profiles == 3
    assert chunk_profile_counts == [1, 2]


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_preserves_thread_roots_for_children(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-thread.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-root", title="Root"),
            messages=[make_message("conv-root:msg-1", "conv-root", text="Root message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session(
                "conv-child",
                title="Child",
                parent_session_id="conv-root",
            ),
            messages=[make_message("conv-child:msg-1", "conv-child", text="Child message")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            ["conv-root", "conv-child"],
            transaction_depth=1,
            page_size=10,
        )

    assert update.counts.profiles == 2
    assert update.thread_root_ids == {"conv-root"}
    assert len(update.chunk_observations) == 1


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_uses_small_default_chunks(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-default-chunks.db"
    session_ids: list[str] = []
    with open_connection(db_path) as conn:
        for index in range(11):
            session_id = f"conv-{index:02d}"
            session_ids.append(session_id)
            store_records(
                session=make_session(session_id, title=f"Session {index:02d}"),
                messages=[
                    make_message(
                        f"{session_id}:msg-1",
                        session_id,
                        text=f"Message for {session_id}",
                    )
                ],
                attachments=[],
                conn=conn,
            )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            session_ids,
            transaction_depth=1,
        )

    assert update.counts.profiles == 11
    assert len(update.chunk_observations) == 2
    assert update.chunk_observations[0].session_count == 10
    assert update.chunk_observations[0].estimated_message_count == 10
    assert update.chunk_observations[1].session_count == 1
    assert update.chunk_observations[1].estimated_message_count == 1


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_splits_large_message_batches(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-message-budget.db"
    session_ids: list[str] = []
    with open_connection(db_path) as conn:
        for index in range(3):
            session_id = f"conv-budget-{index:02d}"
            session_ids.append(session_id)
            store_records(
                session=make_session(session_id, title=f"Budget {index:02d}"),
                messages=[
                    make_message(
                        f"{session_id}:msg-1",
                        session_id,
                        text=f"Message for {session_id}",
                    )
                ],
                attachments=[],
                conn=conn,
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO session_stats (
                session_id,
                source_name,
                message_count,
                word_count,
                tool_use_count,
                thinking_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("conv-budget-00", "chatgpt", 4_000, 4_000, 0, 0),
                ("conv-budget-01", "chatgpt", 4_000, 4_000, 0, 0),
                ("conv-budget-02", "chatgpt", 100, 100, 0, 0),
            ],
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            session_ids,
            transaction_depth=1,
        )

    assert update.counts.profiles == 3
    assert len(update.chunk_observations) == 2
    assert update.chunk_observations[0].session_count == 1
    assert update.chunk_observations[0].estimated_message_count == 4_000
    assert update.chunk_observations[1].session_count == 2
    assert update.chunk_observations[1].estimated_message_count == 4_100


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_clears_deleted_sessions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-delete.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-stale", title="Stale"),
            messages=[make_message("conv-stale:msg-1", "conv-stale", text="Before delete")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        first_update = await _apply_session_insight_session_updates_async(
            conn,
            ["conv-stale"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert first_update.counts.profiles == 1

    with open_connection(db_path) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", ("conv-stale",))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", ("conv-stale",))
        conn.commit()

    async with backend.connection() as conn:
        second_update = await _apply_session_insight_session_updates_async(
            conn,
            ["conv-stale"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert second_update.counts.profiles == 0
    assert len(second_update.chunk_observations) == 1

    with open_connection(db_path) as conn:
        profile_count = conn.execute(
            "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?",
            ("conv-stale",),
        ).fetchone()[0]
        work_event_count = conn.execute(
            "SELECT COUNT(*) FROM session_work_events WHERE session_id = ?",
            ("conv-stale",),
        ).fetchone()[0]
        phase_count = conn.execute(
            "SELECT COUNT(*) FROM session_phases WHERE session_id = ?",
            ("conv-stale",),
        ).fetchone()[0]

    assert profile_count == 0
    assert work_event_count == 0
    assert phase_count == 0


@pytest.mark.asyncio
async def test_refresh_thread_roots_async_batches_root_rebuilds(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-thread-roots.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-root-a", title="Root A"),
            messages=[make_message("conv-root-a:msg-1", "conv-root-a", text="Root A message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session(
                "conv-child-a",
                title="Child A",
                parent_session_id="conv-root-a",
            ),
            messages=[make_message("conv-child-a:msg-1", "conv-child-a", text="Child A message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session("conv-root-b", title="Root B"),
            messages=[make_message("conv-root-b:msg-1", "conv-root-b", text="Root B message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session(
                "conv-child-b",
                title="Child B",
                parent_session_id="conv-root-b",
            ),
            messages=[make_message("conv-child-b:msg-1", "conv-child-b", text="Child B message")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            ["conv-root-a", "conv-child-a", "conv-root-b", "conv-child-b"],
            transaction_depth=1,
            page_size=10,
        )
        refreshed = await _refresh_thread_roots_async(
            conn,
            sorted(update.thread_root_ids),
            transaction_depth=1,
        )
        await conn.commit()

    assert refreshed == 2

    with open_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT thread_id, root_id, session_count
            FROM work_threads
            ORDER BY thread_id
            """
        ).fetchall()

    assert [(row["thread_id"], row["root_id"], row["session_count"]) for row in rows] == [
        ("conv-root-a", "conv-root-a", 2),
        ("conv-root-b", "conv-root-b", 2),
    ]


@pytest.mark.asyncio
async def test_refresh_async_provider_day_aggregates_batches_multiple_groups(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "refresh-provider-day-groups.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-chatgpt-a",
                source_name="chatgpt",
                title="ChatGPT A",
                created_at="2026-04-02T10:00:00+00:00",
                updated_at="2026-04-02T10:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-a:msg-1",
                    "conv-chatgpt-a",
                    text="ChatGPT A message",
                    timestamp="2026-04-02T10:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session(
                "conv-chatgpt-b",
                source_name="chatgpt",
                title="ChatGPT B",
                created_at="2026-04-02T11:00:00+00:00",
                updated_at="2026-04-02T11:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-b:msg-1",
                    "conv-chatgpt-b",
                    text="ChatGPT B message",
                    timestamp="2026-04-02T11:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session(
                "conv-claude-a",
                source_name="claude-ai",
                title="Claude A",
                created_at="2026-04-03T09:00:00+00:00",
                updated_at="2026-04-03T09:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-claude-a:msg-1",
                    "conv-claude-a",
                    text="Claude A message",
                    timestamp="2026-04-03T09:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            ["conv-chatgpt-a", "conv-chatgpt-b", "conv-claude-a"],
            transaction_depth=1,
            page_size=10,
        )
        await refresh_async_provider_day_aggregates(
            conn,
            update.affected_groups,
            transaction_depth=1,
        )
        await conn.commit()

    with open_connection(db_path) as conn:
        tag_rows = conn.execute(
            """
            SELECT source_name, bucket_day, session_count
            FROM session_tag_rollups
            WHERE tag = 'origin:' || source_name
            ORDER BY source_name, bucket_day
            """
        ).fetchall()

    assert [(row["source_name"], row["bucket_day"], row["session_count"]) for row in tag_rows] == [
        ("chatgpt-export", "2026-04-02", 2),
        ("claude-ai-export", "2026-04-03", 1),
    ]
