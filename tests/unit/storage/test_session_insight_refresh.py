from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.storage.insights.session.rebuild as rebuild_mod
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
from polylogue.storage.insights.session.repair_assessment import assess_session_insight_repairs
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import make_message, make_session, store_records


def _current_index_db(tmp_path: Path, name: str) -> Path:
    archive_root = tmp_path / name
    initialize_active_archive_root(archive_root)
    return archive_root / "index.db"


def _sid(native_id: str, origin: str = "unknown-export") -> str:
    return f"{origin}:{native_id}"


def _chunk_metric(chunk_observation: object, key: str) -> float:
    value = getattr(chunk_observation, key)
    assert isinstance(value, int | float)
    return float(value)


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_batches_hydrated_sessions(
    tmp_path: Path,
) -> None:
    db_path = _current_index_db(tmp_path, "refresh")
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
            [_sid("conv-refresh")],
            transaction_depth=1,
            page_size=10,
        )

    assert update.counts.profiles == 1
    assert update.thread_root_ids == {_sid("conv-refresh")}
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
    db_path = _current_index_db(tmp_path, "logical-session")
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
            [
                _sid("root", "claude-code-session"),
                _sid("continuation", "claude-code-session"),
                _sid("fork", "claude-code-session"),
            ],
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
        (_sid("continuation", "claude-code-session"), _sid("root", "claude-code-session")),
        (_sid("fork", "claude-code-session"), _sid("root", "claude-code-session")),
        (_sid("root", "claude-code-session"), _sid("root", "claude-code-session")),
    }
    assert tag_rollup is not None
    assert int(tag_rollup["session_count"]) == 3
    assert int(tag_rollup["logical_session_count"]) == 1
    assert json.loads(tag_rollup["logical_session_ids_json"]) == [_sid("root", "claude-code-session")]


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_counts_session_event_compactions(
    tmp_path: Path,
) -> None:
    db_path = _current_index_db(tmp_path, "refresh-session-events")
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-session-event", source_name="codex", title="Compaction Test"),
            messages=[
                make_message(
                    "conv-session-event:msg-1",
                    "conv-session-event",
                    text="Continue after compaction",
                )
            ],
            attachments=[],
            conn=conn,
        )
        conn.execute(
            """
            INSERT INTO session_events (
                session_id,
                source_message_id,
                position,
                event_type,
                summary,
                occurred_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "codex-session:conv-session-event",
                None,
                0,
                "compaction",
                "Earlier context",
                1775037900000,
            ),
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_insight_session_updates_async(
            conn,
            ["codex-session:conv-session-event"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert update.counts.profiles == 1
    with open_connection(db_path) as conn:
        row = conn.execute(
            "SELECT evidence_payload_json FROM session_profiles WHERE session_id = ?",
            ("codex-session:conv-session-event",),
        ).fetchone()

    assert row is not None
    assert json.loads(row["evidence_payload_json"])["compaction_count"] == 1


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_uses_session_events_for_terminal_state(
    tmp_path: Path,
) -> None:
    db_path = _current_index_db(tmp_path, "refresh-provider-terminal")
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
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "bash",
                            "tool_id": "call-1",
                            "tool_input": {"command": "sleep 30"},
                        }
                    ],
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
            ["codex-session:conv-provider-terminal"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    with open_connection(db_path) as conn:
        row = conn.execute(
            "SELECT terminal_state, evidence_payload_json FROM session_profiles WHERE session_id = ?",
            ("codex-session:conv-provider-terminal",),
        ).fetchone()
        latency_row = conn.execute(
            "SELECT stuck_tool_count FROM session_latency_profiles WHERE session_id = ?",
            ("codex-session:conv-provider-terminal",),
        ).fetchone()

    assert row is not None
    assert row["terminal_state"] == "tool_left"
    assert "terminal_state" not in json.loads(row["evidence_payload_json"])
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
            "UPDATE threads SET search_text = ? WHERE thread_id = ?",
            ("sentinel thread untouched", _sid("conv-claude-a", "claude-ai-export")),
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
        counts = rebuild_session_insights_sync(conn, session_ids=[_sid("conv-chatgpt-b", "chatgpt-export")])
        tag_rows = conn.execute(
            """
            SELECT source_name, bucket_day, session_count, search_text
            FROM session_tag_rollups
            WHERE tag = 'origin:' || source_name
            ORDER BY source_name, bucket_day
            """
        ).fetchall()
        claude_thread = conn.execute(
            "SELECT search_text FROM threads WHERE thread_id = ?",
            (_sid("conv-claude-a", "claude-ai-export"),),
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


def test_targeted_session_insight_rebuild_moves_tag_rollup_between_days(
    tmp_path: Path,
) -> None:
    """Backfilled sessions must refresh the old and new provider-day buckets."""
    db_path = tmp_path / "refresh-sync-backfill-day.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-chatgpt-backfill",
                source_name="chatgpt",
                title="ChatGPT Backfill",
                created_at="2026-04-03T10:00:00+00:00",
                updated_at="2026-04-03T10:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-backfill:msg-1",
                    "conv-chatgpt-backfill",
                    text="Initial later-day message",
                    timestamp="2026-04-03T10:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        rebuild_session_insights_sync(conn)

        store_records(
            session=make_session(
                "conv-chatgpt-backfill",
                source_name="chatgpt",
                title="ChatGPT Backfill",
                created_at="2026-04-01T08:00:00+00:00",
                # The source row is newly acquired even though it contains
                # earlier transcript evidence. If updated_at moved backwards,
                # the archive writer would correctly reject the replace as
                # stale before the insight rebuild contract is exercised.
                updated_at="2026-04-04T08:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-backfill:msg-1",
                    "conv-chatgpt-backfill",
                    text="Backfilled earlier-day message",
                    timestamp="2026-04-01T08:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        counts = rebuild_session_insights_sync(
            conn,
            session_ids=[_sid("conv-chatgpt-backfill", "chatgpt-export")],
        )
        tag_rows = conn.execute(
            """
            SELECT bucket_day, session_count
            FROM session_tag_rollups
            WHERE source_name = 'chatgpt-export'
              AND tag = 'origin:chatgpt-export'
            ORDER BY bucket_day
            """
        ).fetchall()

    assert counts.profiles == 1
    assert [(row["bucket_day"], row["session_count"]) for row in tag_rows] == [
        ("2026-04-01", 1),
    ]


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
            (_sid("conv-codex-untimestamped", "codex-session"),),
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
            (_sid("conv-token-costs", "claude-code-session"),),
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
    # Provider-reported cost is sourced from typed per-message token usage now
    # (#803/#1139): the legacy session-level ``provider_meta`` total/duration
    # passthrough was removed, and ``Session.total_cost_usd`` /
    # ``total_duration_ms`` are hardwired to 0 on the hydrated session. The
    # materialized profile cost therefore comes from the provider-reported
    # message token columns and carries the model in the per-model breakdown.
    db_path = tmp_path / "profile-provider-cost.db"
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-provider-cost",
                source_name="claude-code",
                title="Provider cost",
                created_at="2026-05-12T09:00:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-provider-cost:msg-1",
                    "conv-provider-cost",
                    text="Provider reported total",
                    model_name="claude-opus-4-5",
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
            SELECT total_cost_usd, cost_provenance, per_model_cost_json
            FROM session_profiles
            WHERE session_id = ?
            """,
            (_sid("conv-provider-cost", "claude-code-session"),),
        ).fetchone()

    assert row is not None
    assert row["total_cost_usd"] > 0
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
                    blocks=[{"type": "text", "text": "Plain text is already stored on the message row."}],
                ),
                make_message(
                    "conv-blocks:msg-2",
                    "conv-blocks",
                    role="assistant",
                    text="exec_command",
                    blocks=[
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
        batch = load_sync_batch(conn, [_sid("conv-blocks", "codex-session")])

    assert [str(block.message_id) for block in batch.blocks] == [_sid("conv-blocks", "codex-session") + ":msg-2"]


def test_session_insight_load_includes_compaction_session_events_for_profile_classifiers(tmp_path: Path) -> None:
    db_path = _current_index_db(tmp_path, "refresh-session-event-load")
    with open_connection(db_path) as conn:
        store_records(
            session=make_session("conv-session-event-load", source_name="codex", title="Event Load"),
            messages=[
                make_message(
                    "conv-session-event-load:msg-1",
                    "conv-session-event-load",
                    text="Run the tool",
                )
            ],
            attachments=[],
            conn=conn,
        )
        conn.execute(
            """
            INSERT INTO session_events (
                session_id,
                source_message_id,
                position,
                event_type,
                summary,
                occurred_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "codex-session:conv-session-event-load",
                None,
                0,
                "compaction",
                "Earlier context",
                1775037900000,
            ),
        )
        batch = load_sync_batch(conn, ["codex-session:conv-session-event-load"])
        hydrated = batch.session_events_by_session["codex-session:conv-session-event-load"]

    assert [event.event_type for event in hydrated] == ["compaction"]


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
                    blocks=[
                        {
                            "type": "text",
                            "text": large_message,
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "call-1",
                            "text": large_tool_output,
                        },
                    ],
                )
            ],
            attachments=[],
            conn=conn,
        )
        batch = load_sync_batch(conn, [_sid("conv-large-text", "codex-session")])

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
            session_ids.append(_sid(session_id))
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
            UPDATE sessions
            SET message_count = ?
            WHERE session_id = ?
            """,
            [
                (4_000, _sid("conv-rebuild-budget-00")),
                (4_000, _sid("conv-rebuild-budget-01")),
                (100, _sid("conv-rebuild-budget-02")),
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


def test_large_session_rebuild_uses_bounded_degraded_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "large-session-degraded.db"
    native = "conv-large-bounded"
    session_id = _sid(native, "codex-session")
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(native, source_name="codex", title="Large bounded profile"),
            messages=[
                make_message(f"{native}:msg-1", native, text="first prompt"),
                make_message(f"{native}:msg-2", native, role="assistant", text="answer"),
            ],
            attachments=[],
            conn=conn,
        )
        conn.execute(
            """
            UPDATE sessions
            SET message_count = ?, word_count = ?, tool_use_count = ?, thinking_count = ?
            WHERE session_id = ?
            """,
            (50, 1234, 7, 3, session_id),
        )
        conn.commit()

        monkeypatch.setattr(rebuild_mod, "_SESSION_INSIGHT_DEGRADED_MESSAGE_THRESHOLD", 10)
        counts = rebuild_session_insights_sync(conn, session_ids=[session_id])
        profile = conn.execute(
            "SELECT * FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        work_events = conn.execute(
            "SELECT COUNT(*) FROM session_work_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        markers = {
            str(row["insight_type"]): int(row["materializer_version"])
            for row in conn.execute(
                "SELECT insight_type, materializer_version FROM insight_materialization WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        }

    assert counts.profiles == 1
    assert counts.work_events == 0
    assert counts.phases == 0
    assert profile["workflow_shape"] == "bounded_large_session"
    assert profile["message_count"] == 50
    assert profile["word_count"] == 1234
    assert profile["tool_use_count"] == 7
    assert "large_session_bounded" in profile["inference_payload_json"]
    assert "large_session_bounded" in profile["enrichment_payload_json"]
    assert work_events == 0
    assert markers["session_profile"] == SESSION_INSIGHT_MATERIALIZER_VERSION
    assert markers["latency"] == SESSION_INSIGHT_MATERIALIZER_VERSION
    assert markers["work_events"] == SESSION_INSIGHT_MATERIALIZER_VERSION
    assert markers["phases"] == SESSION_INSIGHT_MATERIALIZER_VERSION
    assert markers["thread"] == SESSION_INSIGHT_MATERIALIZER_VERSION


def test_full_rebuild_commits_incrementally_and_prunes_orphans(tmp_path: Path) -> None:
    """Bounded-WAL full rebuild (#2458) must:

    1. commit each chunk so intermediate committed state is visible from a
       second connection;
    2. keep prior per-session insights visible for not-yet-processed sessions
       during the rebuild (no empty window);
    3. prune per-session insight rows whose session was deleted since the last
       rebuild (orphan cleanup).
    """
    db_path = _current_index_db(tmp_path, "bounded-wal-rebuild")
    live_natives = ["conv-live-0", "conv-live-1", "conv-live-2"]
    orphan_native = "conv-orphan"

    with open_connection(db_path) as conn:
        for native in [*live_natives, orphan_native]:
            store_records(
                session=make_session(native, title=f"v1-{native}"),
                messages=[make_message(f"{native}:msg-1", native, text=f"work {native}")],
                attachments=[],
                conn=conn,
            )
        conn.commit()
        # Baseline rebuild: every session gets a profile row.
        rebuild_session_insights_sync(conn)
        conn.commit()

    live_ids = [_sid(native) for native in live_natives]
    orphan_id = _sid(orphan_native)

    with open_connection(db_path) as conn:
        baseline = {
            str(row["session_id"]) for row in conn.execute("SELECT session_id FROM session_profiles").fetchall()
        }
    assert baseline == {*live_ids, orphan_id}

    # Flip live session titles v1 -> v2 (so the rebuilt profile.title changes).
    with open_connection(db_path) as conn:
        for native in live_natives:
            conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (f"v2-{native}", _sid(native)))
        conn.commit()

    # Delete the orphan's session row WITHOUT FK cascade (a plain sqlite3
    # connection has foreign_keys OFF by default) so its session_profiles row
    # survives as an orphan the rebuild must prune.
    with sqlite3.connect(str(db_path)) as raw:
        raw.execute("DELETE FROM sessions WHERE session_id = ?", (orphan_id,))
        raw.commit()

    with open_connection(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?",
                (orphan_id,),
            ).fetchone()[0]
            == 1
        ), "orphan profile should still be present going into the rebuild"

    committed_v2_counts: list[int] = []
    live_profile_counts: list[int] = []

    def observe(amount: int, desc: str | None = None) -> None:
        del amount
        # Sample only on per-chunk materialize heartbeats, not the orphan-prune
        # or aggregate-clear heartbeats (those descs start with "rebuild:").
        if desc is not None and desc.startswith("rebuild:"):
            return
        with sqlite3.connect(str(db_path)) as probe:
            probe.row_factory = sqlite3.Row
            rows = probe.execute(
                "SELECT session_id, title FROM session_profiles WHERE session_id IN (?, ?, ?)",
                tuple(live_ids),
            ).fetchall()
        # Every live session always has a profile row at every callback (no empty window).
        live_profile_counts.append(len({str(row["session_id"]) for row in rows}))
        # Live profiles whose committed title already flipped to v2.
        committed_v2_counts.append(sum(1 for row in rows if str(row["title"]).startswith("v2-")))

    with open_connection(db_path) as conn:
        counts = rebuild_session_insights_sync(conn, page_size=1, progress_callback=observe)

    # 3 live sessions, page_size=1 -> 3 single-session chunks -> 3 materialize callbacks.
    assert len(committed_v2_counts) == 3
    # No empty window: every live session is present at every callback.
    assert live_profile_counts == [3, 3, 3]
    # Incremental commit: the callback fires before its own chunk commits, so the
    # count of already-committed v2 profiles grows 0 -> 1 -> 2 across chunks. A
    # single end-of-rebuild commit would instead leave this [0, 0, 0].
    assert committed_v2_counts == [0, 1, 2]

    with open_connection(db_path) as conn:
        final = {
            str(row["session_id"]): str(row["title"])
            for row in conn.execute("SELECT session_id, title FROM session_profiles").fetchall()
        }
    # Orphan pruned; all live profiles rebuilt to v2.
    assert set(final) == set(live_ids)
    assert all(title.startswith("v2-") for title in final.values())
    assert counts.profiles == 3


def test_full_rebuild_chunks_by_message_budget_before_page_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded-WAL sync rebuild must split by message budget, not just page size.

    This guards the production failure mode from #2458: a "small" session count
    can still represent a huge message window. The test keeps page_size larger
    than the whole fixture so only the message-budget planner can create more
    than one commit boundary.
    """
    db_path = _current_index_db(tmp_path, "bounded-wal-message-budget")
    natives = ["conv-budget-0", "conv-budget-1", "conv-budget-2"]

    with open_connection(db_path) as conn:
        for native in natives:
            store_records(
                session=make_session(native, title=f"v1-{native}"),
                messages=[make_message(f"{native}:msg-1", native, text=f"work {native}")],
                attachments=[],
                conn=conn,
            )
        conn.commit()
        rebuild_session_insights_sync(conn)
        conn.commit()

    session_ids = [_sid(native) for native in natives]
    with open_connection(db_path) as conn:
        conn.executemany(
            "UPDATE sessions SET title = ?, message_count = ? WHERE session_id = ?",
            [
                ("v2-conv-budget-0", 2, session_ids[0]),
                ("v2-conv-budget-1", 2, session_ids[1]),
                ("v2-conv-budget-2", 1, session_ids[2]),
            ],
        )
        conn.commit()

    monkeypatch.setattr(rebuild_mod, "_SESSION_INSIGHT_REBUILD_MESSAGE_BUDGET", 3)
    committed_v2_counts: list[int] = []
    visible_profile_counts: list[int] = []

    def observe(amount: int, desc: str | None = None) -> None:
        del amount
        if desc is not None and desc.startswith("rebuild:"):
            return
        with sqlite3.connect(str(db_path)) as probe:
            probe.row_factory = sqlite3.Row
            rows = probe.execute(
                "SELECT session_id, title FROM session_profiles WHERE session_id IN (?, ?, ?)",
                tuple(session_ids),
            ).fetchall()
        visible_profile_counts.append(len({str(row["session_id"]) for row in rows}))
        committed_v2_counts.append(sum(1 for row in rows if str(row["title"]).startswith("v2-")))

    with open_connection(db_path) as conn:
        counts = rebuild_session_insights_sync(conn, page_size=50, progress_callback=observe)

    # With page_size=50, fixed-count chunking would produce one callback. The
    # patched message budget (3) forces multiple chunks despite the small
    # session count.
    assert len(committed_v2_counts) == 2
    assert visible_profile_counts == [3, 3]
    # The callback fires before the current chunk commit, so the later callback
    # sees a partial committed state from a prior budget chunk. A single
    # terminal commit would leave this [0, 0].
    assert committed_v2_counts[0] == 0
    assert 0 < committed_v2_counts[1] < len(session_ids)
    assert counts.profiles == 3


def test_full_rebuild_restores_thread_spine_membership_and_markers(tmp_path: Path) -> None:
    """A full canonical rebuild (session_ids=None) must leave the threads spine
    intact: thread_sessions populated, created_at_ms not zeroed, and a 'thread'
    materialization marker stamped for every member — not just the root — so
    readiness row_debt reaches 0 on the daemon/repair convergence path (#1743
    P13). This is the regression that the destructive replace_threads rewrite
    used to leave broken (thread_sessions empty, created_at_ms = 0)."""
    db_path = _current_index_db(tmp_path, "thread-spine-adversarial")
    with open_connection(db_path) as conn:
        store_records(
            session=make_session(
                "conv-root",
                source_name="claude-code",
                title="Root",
                created_at="2026-05-01T10:00:00+00:00",
                updated_at="2026-05-01T10:05:00+00:00",
            ),
            messages=[make_message("conv-root:msg-1", "conv-root", text="Root work")],
            attachments=[],
            conn=conn,
        )
        store_records(
            session=make_session(
                "conv-child",
                source_name="claude-code",
                title="Child",
                parent_session_id="conv-root",
                created_at="2026-05-01T11:00:00+00:00",
                updated_at="2026-05-01T11:05:00+00:00",
            ),
            messages=[make_message("conv-child:msg-1", "conv-child", text="Continuation work")],
            attachments=[],
            conn=conn,
        )
        conn.commit()
        rebuild_session_insights_sync(conn)

    root_id = _sid("conv-root", "claude-code-session")
    child_id = _sid("conv-child", "claude-code-session")
    with open_connection(db_path) as conn:
        thread_markers = {
            str(row["session_id"]): int(row["materializer_version"])
            for row in conn.execute(
                "SELECT session_id, materializer_version FROM insight_materialization WHERE insight_type = 'thread'"
            ).fetchall()
        }
        members = {
            str(row["session_id"])
            for row in conn.execute(
                "SELECT session_id FROM thread_sessions WHERE thread_id = ?",
                (root_id,),
            ).fetchall()
        }
        created_at_ms = conn.execute(
            "SELECT created_at_ms FROM threads WHERE thread_id = ?",
            (root_id,),
        ).fetchone()["created_at_ms"]

    # (a) continuation member carries its own 'thread' marker, not only the root.
    assert thread_markers == {
        root_id: SESSION_INSIGHT_MATERIALIZER_VERSION,
        child_id: SESSION_INSIGHT_MATERIALIZER_VERSION,
    }
    # (b) membership join repopulated and created_at_ms re-derived (not zeroed).
    assert members == {root_id, child_id}
    assert created_at_ms > 0
    # Convergence: a full rebuild leaves zero readiness debt on the same reader
    # the repair path uses (ArchiveStore.session_insight_status).
    with ArchiveStore.open_existing(db_path.parent, read_only=False) as archive:
        status = archive.session_insight_status()
    assert assess_session_insight_repairs(status).row_debt == 0


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_preserves_thread_roots_for_children(
    tmp_path: Path,
) -> None:
    db_path = _current_index_db(tmp_path, "refresh-thread")
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
            [_sid("conv-root"), _sid("conv-child")],
            transaction_depth=1,
            page_size=10,
        )

    assert update.counts.profiles == 2
    assert update.thread_root_ids == {_sid("conv-root")}
    assert len(update.chunk_observations) == 1


@pytest.mark.asyncio
async def test_apply_session_insight_session_updates_async_uses_small_default_chunks(
    tmp_path: Path,
) -> None:
    db_path = _current_index_db(tmp_path, "refresh-default-chunks")
    session_ids: list[str] = []
    with open_connection(db_path) as conn:
        for index in range(11):
            session_id = f"conv-{index:02d}"
            session_ids.append(_sid(session_id))
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
    db_path = _current_index_db(tmp_path, "refresh-message-budget")
    session_ids: list[str] = []
    with open_connection(db_path) as conn:
        for index in range(3):
            session_id = f"conv-budget-{index:02d}"
            session_ids.append(_sid(session_id))
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
            UPDATE sessions
            SET message_count = ?
            WHERE session_id = ?
            """,
            [
                (4_000, _sid("conv-budget-00")),
                (4_000, _sid("conv-budget-01")),
                (100, _sid("conv-budget-02")),
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
    db_path = _current_index_db(tmp_path, "refresh-delete")
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
            [_sid("conv-stale")],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert first_update.counts.profiles == 1

    with open_connection(db_path) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (_sid("conv-stale"),))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (_sid("conv-stale"),))
        conn.commit()

    async with backend.connection() as conn:
        second_update = await _apply_session_insight_session_updates_async(
            conn,
            [_sid("conv-stale")],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert second_update.counts.profiles == 0
    assert len(second_update.chunk_observations) == 1

    with open_connection(db_path) as conn:
        profile_count = conn.execute(
            "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?",
            (_sid("conv-stale"),),
        ).fetchone()[0]
        work_event_count = conn.execute(
            "SELECT COUNT(*) FROM session_work_events WHERE session_id = ?",
            (_sid("conv-stale"),),
        ).fetchone()[0]
        phase_count = conn.execute(
            "SELECT COUNT(*) FROM session_phases WHERE session_id = ?",
            (_sid("conv-stale"),),
        ).fetchone()[0]

    assert profile_count == 0
    assert work_event_count == 0
    assert phase_count == 0


@pytest.mark.asyncio
async def test_refresh_thread_roots_async_batches_root_rebuilds(
    tmp_path: Path,
) -> None:
    db_path = _current_index_db(tmp_path, "refresh-thread-roots")
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
            [_sid("conv-root-a"), _sid("conv-child-a"), _sid("conv-root-b"), _sid("conv-child-b")],
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
            SELECT thread_id, thread_id AS root_id, session_count
            FROM threads
            ORDER BY thread_id
            """
        ).fetchall()

    assert [(row["thread_id"], row["root_id"], row["session_count"]) for row in rows] == [
        (_sid("conv-root-a"), _sid("conv-root-a"), 2),
        (_sid("conv-root-b"), _sid("conv-root-b"), 2),
    ]


@pytest.mark.asyncio
async def test_refresh_async_provider_day_aggregates_batches_multiple_groups(
    tmp_path: Path,
) -> None:
    db_path = _current_index_db(tmp_path, "refresh-provider-day-groups")
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
            [
                _sid("conv-chatgpt-a", "chatgpt-export"),
                _sid("conv-chatgpt-b", "chatgpt-export"),
                _sid("conv-claude-a", "claude-ai-export"),
            ],
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
