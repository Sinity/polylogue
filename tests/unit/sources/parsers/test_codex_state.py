"""Codex live SQLite state acquisition/parsing (polylogue-0jf4).

Fixtures here are synthesised structure only -- schema shapes observed
against a real ``~/.codex`` install (captured 2026-07-29, five databases,
~706 MB), never real thread ids, titles, cwds, or conversation content. See
``polylogue/sources/parsers/codex_state.py`` for the acquisition/consumption
scope this module implements versus explicitly defers.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from polylogue.sources.parsers.base import ParsedSessionEvent
from polylogue.sources.parsers.codex_state import (
    CODEX_STATE_FIDELITY,
    IN_SCOPE_KINDS,
    classify_codex_sqlite_path,
    is_in_scope_codex_sqlite_path,
    looks_like_state_db_payload,
    marker_payload,
    parse_codex_goals_db,
    parse_codex_memories_db,
    parse_codex_state_db,
    spawn_edges_as_session_events,
)
from polylogue.sources.sqlite_snapshot import codex_state_raw_id, snapshot_sqlite_database, snapshot_sqlite_to_blob
from polylogue.storage.blob_store import BlobStore


def _write_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                rollout_path TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                source TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                cwd TEXT NOT NULL,
                title TEXT NOT NULL,
                sandbox_policy TEXT NOT NULL,
                approval_mode TEXT NOT NULL,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                has_user_event INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0,
                archived_at INTEGER,
                git_sha TEXT,
                git_branch TEXT,
                git_origin_url TEXT,
                cli_version TEXT NOT NULL DEFAULT '',
                first_user_message TEXT NOT NULL DEFAULT '',
                agent_nickname TEXT,
                agent_role TEXT,
                memory_mode TEXT NOT NULL DEFAULT 'enabled',
                model TEXT,
                reasoning_effort TEXT,
                agent_path TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                thread_source TEXT,
                preview TEXT NOT NULL DEFAULT '',
                recency_at INTEGER NOT NULL DEFAULT 0,
                recency_at_ms INTEGER NOT NULL DEFAULT 0,
                history_mode TEXT NOT NULL DEFAULT 'legacy',
                name TEXT
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT NOT NULL,
                child_thread_id TEXT NOT NULL PRIMARY KEY,
                status TEXT NOT NULL
            );
            CREATE TABLE thread_dynamic_tools (
                thread_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                input_schema TEXT NOT NULL,
                PRIMARY KEY(thread_id, position)
            );
            """
        )
        conn.execute(
            "INSERT INTO threads (id, rollout_path, created_at, updated_at, source, model_provider, cwd, title, "
            "sandbox_policy, approval_mode, agent_nickname, agent_role, model, created_at_ms, updated_at_ms) "
            "VALUES (?, ?, ?, ?, 'cli', 'openai', '/work/example', ?, 'workspace-write', 'on-request', "
            "?, ?, 'gpt-5', ?, ?)",
            (
                "0000-thread-parent",
                "/home/example/.codex/sessions/2026/01/01/rollout-parent.jsonl",
                1700000000,
                1700000100,
                "Investigate flaky retry logic",
                None,
                None,
                1700000000000,
                1700000100000,
            ),
        )
        conn.execute(
            "INSERT INTO threads (id, rollout_path, created_at, updated_at, source, model_provider, cwd, title, "
            "sandbox_policy, approval_mode, agent_nickname, agent_role, model, created_at_ms, updated_at_ms) "
            "VALUES (?, ?, ?, ?, 'cli', 'openai', '/work/example', ?, 'workspace-write', 'on-request', "
            "?, ?, 'gpt-5', ?, ?)",
            (
                "0000-thread-child",
                "/home/example/.codex/sessions/2026/01/01/rollout-child.jsonl",
                1700000050,
                1700000090,
                "",
                None,
                "review",
                1700000050000,
                1700000090000,
            ),
        )
        conn.execute(
            "INSERT INTO thread_spawn_edges (parent_thread_id, child_thread_id, status) VALUES (?, ?, ?)",
            ("0000-thread-parent", "0000-thread-child", "closed"),
        )
        conn.commit()


def _write_goals_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE thread_goals (
                thread_id TEXT PRIMARY KEY NOT NULL,
                goal_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                token_budget INTEGER,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                time_used_seconds INTEGER NOT NULL DEFAULT 0,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO thread_goals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "0000-thread-parent",
                "goal-1",
                "Land the retry fix",
                "active",
                100000,
                4200,
                900,
                1700000000000,
                1700000100000,
            ),
        )
        conn.commit()


def _write_memories_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE stage1_outputs (
                thread_id TEXT PRIMARY KEY,
                source_updated_at INTEGER NOT NULL,
                raw_memory TEXT NOT NULL,
                rollout_summary TEXT NOT NULL,
                rollout_slug TEXT,
                generated_at INTEGER NOT NULL,
                usage_count INTEGER,
                last_usage INTEGER,
                selected_for_phase2 INTEGER NOT NULL DEFAULT 0,
                selected_for_phase2_source_updated_at INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO stage1_outputs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "0000-thread-parent",
                1700000100,
                "summary text",
                "rollout summary",
                "retry-fix",
                1700000110,
                3,
                1700000120,
                1,
                None,
            ),
        )
        conn.commit()


def _write_logs_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE logs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, level TEXT NOT NULL, "
            "target TEXT NOT NULL);"
        )
        conn.commit()


def _write_automation_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE automations (id TEXT PRIMARY KEY, name TEXT NOT NULL, prompt TEXT NOT NULL);
            CREATE TABLE automation_runs (thread_id TEXT PRIMARY KEY, automation_id TEXT NOT NULL, status TEXT NOT NULL);
            """
        )
        conn.commit()


# --- classification ---------------------------------------------------


def test_classifies_thread_state_db(tmp_path: Path) -> None:
    path = tmp_path / "state_5.sqlite"
    _write_state_db(path)
    assert classify_codex_sqlite_path(path) == "thread_state"
    assert is_in_scope_codex_sqlite_path(path) is True


def test_classifies_goals_db(tmp_path: Path) -> None:
    path = tmp_path / "goals_1.sqlite"
    _write_goals_db(path)
    assert classify_codex_sqlite_path(path) == "goals"
    assert is_in_scope_codex_sqlite_path(path) is True


def test_classifies_memories_db(tmp_path: Path) -> None:
    path = tmp_path / "memories_1.sqlite"
    _write_memories_db(path)
    assert classify_codex_sqlite_path(path) == "memories"
    assert is_in_scope_codex_sqlite_path(path) is True


def test_classifies_logs_db_out_of_scope(tmp_path: Path) -> None:
    path = tmp_path / "logs_2.sqlite"
    _write_logs_db(path)
    assert classify_codex_sqlite_path(path) == "logs"
    assert is_in_scope_codex_sqlite_path(path) is False


def test_classifies_automation_db_out_of_scope(tmp_path: Path) -> None:
    path = tmp_path / "codex-dev.db"
    _write_automation_db(path)
    assert classify_codex_sqlite_path(path) == "automation"
    assert is_in_scope_codex_sqlite_path(path) is False


def test_classifies_unreadable_path_as_unknown(tmp_path: Path) -> None:
    path = tmp_path / "not-a-database.sqlite"
    path.write_bytes(b"not sqlite bytes at all")
    assert classify_codex_sqlite_path(path) == "unknown"
    assert is_in_scope_codex_sqlite_path(path) is False


def test_fidelity_declaration_covers_every_known_kind() -> None:
    kinds = {classification.kind for classification in CODEX_STATE_FIDELITY}
    assert kinds == {"thread_state", "goals", "memories", "logs", "automation"}
    dispositions = {classification.kind: classification.disposition for classification in CODEX_STATE_FIDELITY}
    assert dispositions["thread_state"] == "acquire"
    assert dispositions["goals"] == "acquire-partial"
    assert dispositions["memories"] == "acquire-partial"
    assert dispositions["logs"] == "out-of-scope"
    assert dispositions["automation"] == "out-of-scope"
    assert {"thread_state", "goals", "memories"} == IN_SCOPE_KINDS


# --- parsing ------------------------------------------------------------


def test_parse_state_db_extracts_titles_and_spawn_edges(tmp_path: Path) -> None:
    path = tmp_path / "state_5.sqlite"
    _write_state_db(path)
    snapshot = parse_codex_state_db(path)
    by_id = {thread.thread_id: thread for thread in snapshot.threads}
    assert by_id["0000-thread-parent"].title == "Investigate flaky retry logic"
    assert by_id["0000-thread-parent"].cwd == "/work/example"
    assert by_id["0000-thread-parent"].model == "gpt-5"
    assert by_id["0000-thread-child"].title == ""
    assert by_id["0000-thread-child"].agent_role == "review"
    assert len(snapshot.spawn_edges) == 1
    edge = snapshot.spawn_edges[0]
    assert edge.parent_thread_id == "0000-thread-parent"
    assert edge.child_thread_id == "0000-thread-child"
    assert edge.status == "closed"


def test_parse_goals_db(tmp_path: Path) -> None:
    path = tmp_path / "goals_1.sqlite"
    _write_goals_db(path)
    goals = parse_codex_goals_db(path)
    assert len(goals) == 1
    assert goals[0].thread_id == "0000-thread-parent"
    assert goals[0].objective == "Land the retry fix"
    assert goals[0].status == "active"
    assert goals[0].token_budget == 100000


def test_parse_memories_db_omits_raw_memory_text(tmp_path: Path) -> None:
    path = tmp_path / "memories_1.sqlite"
    _write_memories_db(path)
    records = parse_codex_memories_db(path)
    assert len(records) == 1
    record = records[0]
    assert record.thread_id == "0000-thread-parent"
    assert record.usage_count == 3
    assert record.has_rollout_slug is True
    assert record.selected_for_phase2 is True
    assert not hasattr(record, "raw_memory")


# --- session_events shape for spawn edges --------------------------------


def test_spawn_edges_as_session_events(tmp_path: Path) -> None:
    path = tmp_path / "state_5.sqlite"
    _write_state_db(path)
    snapshot = parse_codex_state_db(path)
    grouped = spawn_edges_as_session_events(snapshot.spawn_edges)
    assert set(grouped) == {"0000-thread-parent"}
    events = grouped["0000-thread-parent"]
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ParsedSessionEvent)
    assert event.event_type == "codex_thread_spawn_edge"
    assert event.payload == {
        "parent_thread_id": "0000-thread-parent",
        "child_thread_id": "0000-thread-child",
        "status": "closed",
    }


# --- marker payload round trip -------------------------------------------


def test_marker_payload_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state_5.sqlite"
    payload = marker_payload(path, kind="thread_state")
    assert looks_like_state_db_payload(payload) is True
    assert payload["state_db_path"] == str(path)


def test_marker_payload_rejects_out_of_scope_kind(tmp_path: Path) -> None:
    path = tmp_path / "logs_2.sqlite"
    payload = marker_payload(path, kind="logs")
    assert looks_like_state_db_payload(payload) is False


def test_marker_payload_rejects_unrelated_payload() -> None:
    assert looks_like_state_db_payload({"polylogue_artifact": "something_else"}) is False


# --- consistent snapshot of a live-locked database (AC4) -----------------


def test_snapshot_reads_consistent_state_while_writer_holds_a_transaction(tmp_path: Path) -> None:
    """A live Codex holding the WAL file open must never be blocked or observed mid-write."""
    source = tmp_path / "state_5.sqlite"
    _write_state_db(source)
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA journal_mode=WAL")

    writer_ready = threading.Event()
    release_writer = threading.Event()

    def hold_write_transaction() -> None:
        conn = sqlite3.connect(source, timeout=5.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE threads SET title = 'mutated mid-snapshot' WHERE id = '0000-thread-parent'")
            writer_ready.set()
            # Hold the write transaction open (uncommitted) until the snapshot
            # has had a chance to run concurrently.
            release_writer.wait(timeout=5.0)
        finally:
            conn.rollback()
            conn.close()

    writer_thread = threading.Thread(target=hold_write_transaction)
    writer_thread.start()
    try:
        assert writer_ready.wait(timeout=5.0), "writer never reached its held transaction"
        # Give the writer a moment to be mid-transaction before snapshotting.
        time.sleep(0.05)
        destination = tmp_path / "state_5.snapshot.sqlite"
        snapshot_sqlite_database(source, destination)
    finally:
        release_writer.set()
        writer_thread.join(timeout=5.0)

    snapshot = parse_codex_state_db(destination)
    by_id = {thread.thread_id: thread for thread in snapshot.threads}
    # The snapshot must see the last *committed* state, never the writer's
    # uncommitted in-flight mutation.
    assert by_id["0000-thread-parent"].title == "Investigate flaky retry logic"


# --- content-hash idempotency ---------------------------------------------


def test_snapshot_to_blob_is_idempotent_by_content(tmp_path: Path) -> None:
    source = tmp_path / "state_5.sqlite"
    _write_state_db(source)
    blob_store = BlobStore(tmp_path / "blobs")

    first = snapshot_sqlite_to_blob(source, blob_store)
    second = snapshot_sqlite_to_blob(source, blob_store)
    assert first.blob_hash == second.blob_hash

    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE threads SET title = 'a different title' WHERE id = '0000-thread-parent'")
        conn.commit()
    third = snapshot_sqlite_to_blob(source, blob_store)
    assert third.blob_hash != first.blob_hash


def test_codex_state_raw_id_is_stable_and_path_scoped(tmp_path: Path) -> None:
    a = tmp_path / "a" / "state_5.sqlite"
    b = tmp_path / "b" / "state_5.sqlite"
    a.parent.mkdir()
    b.parent.mkdir()
    _write_state_db(a)
    _write_state_db(b)
    blob_store = BlobStore(tmp_path / "blobs")
    snap_a = snapshot_sqlite_to_blob(a, blob_store)
    snap_b = snapshot_sqlite_to_blob(b, blob_store)
    # Byte-identical content -> identical blob hash ...
    assert snap_a.blob_hash == snap_b.blob_hash
    # ... but raw identity stays scoped to the originating path, matching the
    # Hermes profile-raw pattern (two installs never collapse into one raw row).
    assert codex_state_raw_id(a, snap_a.blob_hash) != codex_state_raw_id(b, snap_b.blob_hash)
    assert codex_state_raw_id(a, snap_a.blob_hash) == codex_state_raw_id(a, snap_a.blob_hash)
