"""bd polylogue-xo9gq: Claude Code hook ``agent_id``/``agent_type`` is subagent
lineage evidence, resolved onto the tool call it names.

Exercises ``context.claude_agent_dispatch_correlation`` directly against real
``source.db``/``index.db`` connections; the facade-level integration test lives
with the other reconciliation facade tests in
``tests/unit/api/test_facade_contracts.py``.

Anti-vacuity: every resolved assertion here comes from reading ``agent_id`` out
of the hook payload. Drop that read and ``resolved`` is empty in each case --
the transcript-side block rows alone assert nothing about which agent instance
ran the call.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.context.claude_agent_dispatch_correlation import correlate_claude_agent_dispatches
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent, write_source_hook_event
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_HASH = b"x" * 32


def _index_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
    return conn


def _source_conn(archive_root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _init_source_db(archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive_root / "source.db") as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        conn.commit()


def _write_tool_hook_event(
    archive_root: Path,
    *,
    event_id: str,
    event_type: str = "PreToolUse",
    session_native_id: str,
    payload: dict[str, object],
    observed_at_ms: int = 1_000,
) -> None:
    """Write one hook event in the shape ``sources/hooks._persist_record`` stores:
    the full spool envelope, with the harness's own payload nested inside it."""
    envelope: dict[str, object] = {
        "event_id": event_id,
        "event_type": event_type,
        "session_id": session_native_id,
        "timestamp": "2026-07-12T10:00:00Z",
        "provider": "claude-code",
        "payload": payload,
        "observed_at_ms": observed_at_ms,
    }
    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_hook_event(
            conn,
            origin="claude-code-session",
            source_path=f"synthetic:hooks/{event_id}.json",
            payload=json.dumps(envelope).encode("utf-8"),
            acquired_at_ms=observed_at_ms,
            raw_id=f"raw-{event_id}",
            hook_event=ArchiveHookEvent(
                hook_event_id=f"hook:{event_id}",
                origin="claude-code-session",
                source_path=f"synthetic:hooks/{event_id}.json",
                event_type=event_type,
                payload=envelope,
                observed_at_ms=observed_at_ms,
                native_id=f"{session_native_id}:{event_type}:{event_id}",
                session_native_id=session_native_id,
            ),
        )


def _seed_tool_use_block(index_conn: sqlite3.Connection, *, native_id: str, tool_id: str) -> str:
    session_id = f"claude-code-session:{native_id}"
    index_conn.execute(
        "INSERT OR IGNORE INTO sessions (native_id, origin, title, content_hash, message_count) VALUES (?, ?, ?, ?, ?)",
        (native_id, "claude-code-session", "test", _HASH, 1),
    )
    index_conn.execute(
        "INSERT OR IGNORE INTO messages (session_id, native_id, position, role, content_hash) VALUES (?, ?, ?, ?, ?)",
        (session_id, f"msg-{tool_id}", 0, "assistant", _HASH),
    )
    message_id = f"{session_id}:n:msg-{tool_id}"
    index_conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type, tool_name, tool_id) VALUES (?, ?, ?, ?, ?, ?)",
        (message_id, session_id, 0, "tool_use", "Bash", tool_id),
    )
    index_conn.commit()
    return f"{message_id}:0"


def test_agent_id_resolves_to_the_tool_call_it_names(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    _write_tool_hook_event(
        archive_root,
        event_id="e1",
        session_native_id="sess-1",
        payload={
            "session_id": "sess-1",
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_use_id": "toolu_1",
            "agent_id": "alog-consolidator-d4b9429a916062bb",
            "agent_type": "log-consolidator",
        },
    )
    index_conn = _index_conn()
    block_id = _seed_tool_use_block(index_conn, native_id="sess-1", tool_id="toolu_1")

    report = correlate_claude_agent_dispatches(_source_conn(archive_root), index_conn)

    assert report.total_agent_bearing_events == 1
    assert report.distinct_agent_ids == 1
    assert report.unresolved_tool_use_ids == ()
    assert report.contradicted_tool_use_ids == ()
    assert report.resolved_count == 1
    assertion = report.resolved[0]
    assert assertion.agent_id == "alog-consolidator-d4b9429a916062bb"
    assert assertion.agent_type == "log-consolidator"
    assert assertion.tool_use_id == "toolu_1"
    assert assertion.tool_use_block_id == block_id
    assert assertion.hook_session_native_id == "sess-1"
    assert assertion.resolved_session_id == "claude-code-session:sess-1"


def test_camelcase_agent_payload_asserts_the_same_lineage(tmp_path: Path) -> None:
    """bd polylogue-cp806 + xo9gq: the camelCase generation carries the same
    assertion, and reading only ``agent_id``/``tool_use_id`` would drop it."""
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    _write_tool_hook_event(
        archive_root,
        event_id="e1",
        session_native_id="sess-1",
        payload={
            "sessionId": "sess-1",
            "hookEventName": "PreToolUse",
            "toolName": "Bash",
            "toolUseId": "toolu_1",
            "agentId": "agent-instance-1",
            "agentType": "reviewer",
        },
    )
    index_conn = _index_conn()
    _seed_tool_use_block(index_conn, native_id="sess-1", tool_id="toolu_1")

    report = correlate_claude_agent_dispatches(_source_conn(archive_root), index_conn)

    assert report.resolved_count == 1
    assert report.resolved[0].agent_id == "agent-instance-1"
    assert report.resolved[0].agent_type == "reviewer"


def test_a_call_the_archive_has_not_ingested_is_unresolved_not_dropped(tmp_path: Path) -> None:
    """The hook can outrun ingest; "not yet ingested" must stay distinguishable
    from "never read"."""
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    _write_tool_hook_event(
        archive_root,
        event_id="e1",
        session_native_id="sess-1",
        payload={"tool_use_id": "toolu_missing", "agent_id": "agent-instance-1", "agent_type": "reviewer"},
    )

    report = correlate_claude_agent_dispatches(_source_conn(archive_root), _index_conn())

    assert report.total_agent_bearing_events == 1
    assert report.resolved_count == 0
    assert report.unresolved_tool_use_ids == ("toolu_missing",)


def test_two_agents_claiming_one_call_is_a_contradiction_not_a_last_writer_win(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    for event_id, agent_id in (("e1", "agent-a"), ("e2", "agent-b")):
        _write_tool_hook_event(
            archive_root,
            event_id=event_id,
            session_native_id="sess-1",
            payload={"tool_use_id": "toolu_1", "agent_id": agent_id, "agent_type": "reviewer"},
        )
    index_conn = _index_conn()
    _seed_tool_use_block(index_conn, native_id="sess-1", tool_id="toolu_1")

    report = correlate_claude_agent_dispatches(_source_conn(archive_root), index_conn)

    assert report.contradicted_tool_use_ids == ("toolu_1",)
    assert report.resolved_count == 0


def test_pre_and_post_tool_use_of_one_call_assert_it_once(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    for event_id, event_type in (("e1", "PreToolUse"), ("e2", "PostToolUse")):
        _write_tool_hook_event(
            archive_root,
            event_id=event_id,
            event_type=event_type,
            session_native_id="sess-1",
            payload={"tool_use_id": "toolu_1", "agent_id": "agent-a", "agent_type": "reviewer"},
        )
    index_conn = _index_conn()
    _seed_tool_use_block(index_conn, native_id="sess-1", tool_id="toolu_1")

    report = correlate_claude_agent_dispatches(_source_conn(archive_root), index_conn)

    assert report.total_agent_bearing_events == 2
    assert report.resolved_count == 1


def test_a_tool_call_with_no_agent_assertion_produces_none(tmp_path: Path) -> None:
    """Anti-vacuity for the whole module: the block rows alone assert nothing."""
    archive_root = tmp_path / "archive"
    _init_source_db(archive_root)
    _write_tool_hook_event(
        archive_root,
        event_id="e1",
        session_native_id="sess-1",
        payload={"tool_use_id": "toolu_1", "tool_name": "Bash"},
    )
    index_conn = _index_conn()
    _seed_tool_use_block(index_conn, native_id="sess-1", tool_id="toolu_1")

    report = correlate_claude_agent_dispatches(_source_conn(archive_root), index_conn)

    assert report.total_agent_bearing_events == 0
    assert report.resolved_count == 0
    assert report.unresolved_tool_use_ids == ()
