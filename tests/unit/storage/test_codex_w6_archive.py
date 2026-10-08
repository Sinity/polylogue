"""Worker-6 regressions through the archive writer, SQL projections and reads."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import aiosqlite
import pytest

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.archive_query_reads import ArchiveAggMetricSpec
from polylogue.storage.sqlite.queries.sessions_reads import get_session, get_sessions_batch
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session
from tests.infra.session_profiles import write_session_profile


def _seed_session(archive: ArchiveStore) -> str:
    return write_index_session(
        archive,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="w6-session",
            messages=[
                ParsedMessage(provider_message_id="user", role=Role.USER, text="start here"),
                ParsedMessage(provider_message_id="assistant", role=Role.ASSISTANT, text="a useful answer"),
            ],
        ),
    )


def _watched_names(archive: ArchiveStore) -> list[str]:
    with closing(sqlite3.connect(archive.user_db_path)) as conn:
        return [str(row[0]) for row in conn.execute("SELECT name FROM query_names WHERE watch = 1 ORDER BY name")]


def test_action_pairs_follow_paired_result_updates_and_deletes(tmp_path: Path) -> None:
    """The native triggers must not copy a stale cached outcome from tool_use."""
    with ArchiveStore(tmp_path) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="w6-actions",
                messages=[
                    ParsedMessage(
                        provider_message_id="message",
                        role=Role.ASSISTANT,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE, tool_name="Bash", tool_id="run", tool_input={"command": "true"}
                            ),
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT, tool_id="run", text="ok", is_error=False, exit_code=0
                            ),
                        ],
                    )
                ],
            ),
        )
        conn = archive._conn
        conn.execute(
            "UPDATE blocks SET tool_outcome = 'error' WHERE session_id = ? AND block_type = 'tool_use'", (session_id,)
        )
        assert (
            conn.execute("SELECT result_state FROM actions WHERE session_id = ?", (session_id,)).fetchone()[0]
            == "outcome_success"
        )
        conn.execute(
            "UPDATE blocks SET tool_outcome = 'unknown', tool_result_outcome_unknown_reason = 'not_reported' "
            "WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        )
        assert conn.execute(
            "SELECT result_state, outcome_unknown_reason FROM actions WHERE session_id = ?", (session_id,)
        ).fetchone()[:] == ("outcome_unknown", "not_reported")
        conn.execute("DELETE FROM blocks WHERE session_id = ? AND block_type = 'tool_result'", (session_id,))
        assert (
            conn.execute("SELECT result_state FROM actions WHERE session_id = ?", (session_id,)).fetchone()[0]
            == "no_result"
        )


async def test_work_event_payload_cannot_override_validated_identity(tmp_path: Path) -> None:
    from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def seed() -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return _seed_session(archive)

    session_id = await run_archive_fixture_write(tmp_path, seed)

    def acquire() -> dict[str, object]:
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.admit_work_event(
                session_id=session_id,
                event_type="tool_run",
                event_id="  event-1  ",
                summary="declared summary",
                payload={"event_id": "", "summary": "payload override", "tool_name": "Bash"},
            )

    admitted = await run_archive_fixture_write(tmp_path, acquire)
    assert admitted["event_id"] == "event-1"
    raw_id = admitted["raw_id"]
    assert isinstance(raw_id, str)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert raw_id in {selected for receipt in receipts for selected in receipt.writer_changed_raw_ids}
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        stored = archive._conn.execute(
            "SELECT payload_json FROM session_events WHERE session_id = ? AND event_type = 'tool_run'",
            (session_id,),
        ).fetchone()
        assert stored is not None
        payload = json.loads(stored[0])
        assert (payload["event_id"], payload["summary"]) == ("event-1", "declared summary")
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        retained = source.execute("SELECT raw_id FROM raw_sessions WHERE raw_id LIKE 'agent-work-event:%'").fetchone()
        assert retained is not None and retained[0] == raw_id


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), {"nested": [float("nan")]}])
def test_metadata_first_write_refuses_nonfinite_json_atomically(tmp_path: Path, value: object) -> None:
    with ArchiveStore(tmp_path) as archive:
        session_id = _seed_session(archive)
        with pytest.raises((TypeError, ValueError)):
            archive.set_user_metadata((session_id,), (("valid", 1), ("invalid", value)))
        assert archive.read_user_metadata(session_id) == {}
        assert archive.set_user_metadata((session_id,), (("invalid", 1.25),)) == 1
        assert archive.read_user_metadata(session_id) == {"invalid": 1.25}


def test_resurrected_view_rename_does_not_clear_another_views_watch(tmp_path: Path) -> None:
    query = json.dumps({"query": "sessions where origin:codex-session"})
    with ArchiveStore(tmp_path) as archive:
        archive.save_view("a", "shared", query, watch=True)
        archive.delete_view("a")
        archive.save_view("b", "shared", query, watch=True)
        archive.save_view("a", "revived", query, watch=True)
        assert _watched_names(archive) == ["revived", "shared"]
        shared = archive.get_view_by_name("shared")
        assert shared is not None
        assert shared["view_id"] == "b"


def test_redeleting_view_does_not_clear_reassigned_name_watch(tmp_path: Path) -> None:
    query = json.dumps({"query": "sessions where origin:codex-session"})
    with ArchiveStore(tmp_path) as archive:
        archive.save_view("a", "shared", query, watch=True)
        archive.delete_view("a")
        archive.save_view("b", "shared", query, watch=True)
        archive.delete_view("a")
        assert _watched_names(archive) == ["shared"]
        shared = archive.get_view_by_name("shared")
        assert shared is not None
        assert shared["view_id"] == "b"


@pytest.mark.parametrize("limit,offset", [(0, 0), (1, 10)])
def test_empty_aggregate_page_preserves_total_group_count(tmp_path: Path, limit: int, offset: int) -> None:
    source = parse_unit_source_expression("messages where role:(user|assistant)")
    assert source is not None
    metrics = (ArchiveAggMetricSpec(label="words", fn="sum", field="word_count"),)
    with ArchiveStore(tmp_path) as archive:
        _seed_session(archive)
        complete = archive.query_unit_agg_metrics("message", source.predicate, group_by=("role",), metrics=metrics)
        assert len(complete.rows) == complete.total_groups == 2
        page = archive.query_unit_agg_metrics(
            "message",
            source.predicate,
            group_by=("role",),
            metrics=metrics,
            limit=limit,
            offset=offset,
        )
        assert page.rows == ()
        assert page.total_groups == 2


@pytest.mark.asyncio
async def test_session_record_reads_preserve_milliseconds(tmp_path: Path) -> None:
    created = "2026-01-02T03:04:05.123+00:00"
    updated = "2026-01-02T03:04:06.987+00:00"

    def seed() -> tuple[str, Path]:
        with ArchiveStore(tmp_path) as archive:
            written = write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="w6-time",
                    created_at=created,
                    updated_at=updated,
                    messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
                ),
            )
            return written, archive.index_db_path

    # The archive writer's synchronous lease must not block this event loop.
    session_id, index_path = run_off_event_loop(seed)
    async with aiosqlite.connect(index_path) as conn:
        conn.row_factory = sqlite3.Row
        one = await get_session(conn, session_id)
        batch = await get_sessions_batch(conn, [session_id])
    assert one is not None
    assert len(batch) == 1
    for record in (one, batch[0]):
        assert record.created_at is not None and record.updated_at is not None
        assert datetime.fromisoformat(record.created_at) == datetime.fromisoformat(created)
        assert datetime.fromisoformat(record.updated_at) == datetime.fromisoformat(updated)


@pytest.mark.parametrize("column,value", [("attachment_id", "replacement"), ("position", 1)])
def test_attachment_reference_identity_and_order_invalidate_profile_binding(
    tmp_path: Path, column: str, value: object
) -> None:
    """A native reference UPDATE invalidates the binding and queues its owner."""
    with ArchiveStore(tmp_path) as archive:
        session_id = _seed_session(archive)
        conn = archive._conn
        message_id = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position LIMIT 1", (session_id,)
        ).fetchone()[0]
        conn.executemany(
            "INSERT INTO attachments(attachment_id, display_name, media_type, byte_count) VALUES (?, ?, 'text/plain', 1)",
            [("original", "a.txt"), ("replacement", "b.txt")],
        )
        conn.execute(
            "INSERT INTO attachment_refs(attachment_id, session_id, message_id, position) VALUES ('original', ?, ?, 0)",
            (session_id, message_id),
        )
        write_session_profile(conn, session_id, input_content_hash="published-binding")
        conn.execute("DELETE FROM session_profile_demand WHERE session_id = ?", (session_id,))
        conn.execute(f"UPDATE attachment_refs SET {column} = ? WHERE session_id = ?", (value, session_id))
        assert (
            conn.execute(
                "SELECT input_content_hash FROM session_profiles WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            is None
        )
        assert (
            conn.execute("SELECT revision FROM session_profile_demand WHERE session_id = ?", (session_id,)).fetchone()[
                0
            ]
            > 0
        )


@pytest.mark.parametrize("outcome,reason,expected", [("unknown", "not_reported", "unknown"), ("error", None, "failed")])
def test_observed_tool_rollup_respects_canonical_outcome_over_exit_code(
    tmp_path: Path,
    outcome: str,
    reason: str | None,
    expected: str,
) -> None:
    with ArchiveStore(tmp_path) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="w6-observed-outcome",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Bash",
                                tool_id="tool",
                                tool_input={"command": "true"},
                            ),
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="tool",
                                text="provider signal",
                                is_error=False,
                                exit_code=0,
                            ),
                        ],
                    )
                ],
            ),
        )
        archive._conn.execute(
            "UPDATE blocks SET tool_outcome = ?, tool_result_outcome_unknown_reason = ? "
            "WHERE session_id = ? AND block_type = 'tool_result'",
            (outcome, reason, session_id),
        )
        rows = archive.list_tool_observed_event_count_rows()
        assert len(rows) == 1
        assert (rows[0]["status"], rows[0]["event_count"]) == (expected, 1)
        # The observed-event relation projects and filters (through its
        # source pushdown) on the same canonical outcome.
        source = parse_unit_source_expression(f"observed-events where kind:tool_finished AND status:{expected}")
        assert source is not None
        counts = archive.query_unit_counts("observed-event", source.predicate, group_by="status")
        assert [(row.group_key, row.count) for row in counts] == [(expected, 1)]
