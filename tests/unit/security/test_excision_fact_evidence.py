"""Excising a session must reach the fact-tier evidence that names it.

Artifacts admitted with ``parse_policy='fact'`` get their own
``raw_sessions`` row but mint no ``sessions`` row, so ``sessions.raw_id``
never names them and no index relation reaches them. Claude Code's
``~/.claude/todos/<session-uuid>[-agent-<uuid>].json`` plan snapshots are
linked to their session only by the identity in their own filename, which is
why excising a session used to leave the agent's plan text readable under the
excised session id (polylogue-si5kj).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.security.excision import (
    apply_session_excision,
    plan_session_excision,
    resolve_session_excision_target,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ContentExcisedError,
    write_source_raw_session,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_SESSION_UUID = "11111111-2222-3333-4444-555555555555"
_OTHER_UUID = "99999999-8888-7777-6666-555555555555"
_TODO_PAYLOAD = b'[{"content": "delete the incriminating file", "status": "completed", "id": "t1"}]'
_OTHER_PAYLOAD = b'[{"content": "unrelated session plan", "status": "pending", "id": "t9"}]'


def _seed(tmp_path: Path) -> str:
    """One Claude Code session plus its TODO snapshot, and an unrelated one."""
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    blob_store = BlobStore(tmp_path / "blob")
    blob_store.write_from_bytes(_TODO_PAYLOAD)
    blob_store.write_from_bytes(_OTHER_PAYLOAD)

    conn = sqlite3.connect(source_db)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        write_source_raw_session(
            conn,
            origin="claude-code-session",
            source_path=f"/home/op/.claude/todos/{_SESSION_UUID}.json",
            source_index=0,
            payload=_TODO_PAYLOAD,
            acquired_at_ms=1_000,
            native_id=None,
            raw_id="raw-todo",
        )
        write_source_raw_session(
            conn,
            origin="claude-code-session",
            source_path=f"/home/op/.claude/todos/{_OTHER_UUID}.json",
            source_index=0,
            payload=_OTHER_PAYLOAD,
            acquired_at_ms=1_100,
            native_id=None,
            raw_id="raw-todo-other",
        )
        conn.commit()
    finally:
        conn.close()

    index_conn = sqlite3.connect(index_db)
    index_conn.execute("PRAGMA foreign_keys = ON")
    try:
        index_conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES (?, 'claude-code-session', NULL, 'Plan owner', zeroblob(32), 1000, 2000)",
            (_SESSION_UUID,),
        )
        session_id = str(
            index_conn.execute(
                "SELECT session_id FROM sessions WHERE native_id = ?",
                (_SESSION_UUID,),
            ).fetchone()[0]
        )
        index_conn.commit()
    finally:
        index_conn.close()
    return session_id


def _readable_todo_session_ids(tmp_path: Path) -> set[str]:
    """Session ids whose TODO plan text is still readable from retained bytes.

    Reads the retained ``raw_sessions`` rows and their blobs directly, which
    is what "the plan is still readable under this session id" means; no
    read-model module stands between the assertion and the bytes.
    """
    blob_store = BlobStore(tmp_path / "blob")
    readable: set[str] = set()
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute("SELECT source_path, lower(hex(blob_hash)) FROM raw_sessions").fetchall()
    for source_path, blob_hash in rows:
        name = Path(str(source_path)).stem
        if not blob_store.read_all(str(blob_hash)):
            continue
        readable.add(name)
    return readable


def _raw_ids(tmp_path: Path) -> set[str]:
    with sqlite3.connect(tmp_path / "source.db") as conn:
        return {str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions")}


def _excised_hashes(tmp_path: Path) -> set[bytes]:
    with sqlite3.connect(tmp_path / "source.db") as conn:
        return {bytes(row[0]) for row in conn.execute("SELECT removed_hash FROM excised_content")}


def test_excision_removes_the_session_todo_plan_evidence(tmp_path: Path) -> None:
    """TODO plan text must not stay readable under an excised session id.

    Anti-vacuity: removing the ``_session_fact_raw_ids`` call from
    ``resolve_session_excision_target`` leaves ``raw-todo`` in
    ``raw_sessions``, its hash absent from ``excised_content``, and
    ``_readable_todo_session_ids`` still returning the plan text under the
    excised session id. The unrelated session's snapshot is asserted
    untouched, so a resolver that simply deleted every todo row fails too.
    """
    session_id = _seed(tmp_path)

    # Precondition: the plan really is readable before the excision.
    before = _readable_todo_session_ids(tmp_path)
    assert before == {_SESSION_UUID, _OTHER_UUID}

    target = resolve_session_excision_target(tmp_path, session_id)
    assert target.fact_raw_ids == ("raw-todo",)
    assert {raw.raw_id for raw in target.raw_targets} == {"raw-todo"}

    plan = plan_session_excision(tmp_path, session_id)
    assert plan.source_fact_rows == 1
    assert plan.source_raw_rows == 1

    receipt = apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt.found
    assert receipt.counts["source_fact_rows"] == 1

    assert _raw_ids(tmp_path) == {"raw-todo-other"}, "the session's plan snapshot survived its excision"
    assert hashlib.sha256(_TODO_PAYLOAD).digest() in _excised_hashes(tmp_path)

    after = _readable_todo_session_ids(tmp_path)
    assert after == {_OTHER_UUID}, "the excised session's plan is still readable"

    # Non-resurrection: re-acquiring the same snapshot file is refused.
    conn = sqlite3.connect(tmp_path / "source.db")
    try:
        with pytest.raises(ContentExcisedError):
            write_source_raw_session(
                conn,
                origin="claude-code-session",
                source_path=f"/home/op/.claude/todos/{_SESSION_UUID}.json",
                source_index=0,
                payload=_TODO_PAYLOAD,
                acquired_at_ms=5_000,
                native_id=None,
                raw_id="raw-todo-again",
            )
    finally:
        conn.close()


def test_subagent_todo_snapshot_is_excised_with_its_parent_session(tmp_path: Path) -> None:
    """A delegated subagent's plan file belongs to the session in its stem.

    Claude Code names it ``<session-uuid>-agent-<agent-uuid>.json``; the bytes
    are the parent session's own work product and must go with it.

    Anti-vacuity: matching the filename's full stem against the native id
    instead of its session-id group leaves ``raw-todo-agent`` behind.
    """
    session_id = _seed(tmp_path)
    agent_payload = b'[{"content": "subagent plan", "status": "pending", "id": "a1"}]'
    BlobStore(tmp_path / "blob").write_from_bytes(agent_payload)

    conn = sqlite3.connect(tmp_path / "source.db")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        write_source_raw_session(
            conn,
            origin="claude-code-session",
            source_path=f"/home/op/.claude/todos/{_SESSION_UUID}-agent-{_OTHER_UUID}.json",
            source_index=0,
            payload=agent_payload,
            acquired_at_ms=1_200,
            native_id=None,
            raw_id="raw-todo-agent",
        )
        conn.commit()
    finally:
        conn.close()

    target = resolve_session_excision_target(tmp_path, session_id)
    assert set(target.fact_raw_ids) == {"raw-todo", "raw-todo-agent"}

    apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert _raw_ids(tmp_path) == {"raw-todo-other"}
    assert hashlib.sha256(agent_payload).digest() in _excised_hashes(tmp_path)
