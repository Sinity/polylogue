"""Neutral original payload builders shared by retained parser controls."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.sources.sqlite_export import logical_export_bytes
from polylogue.sources.sqlite_snapshot import member_export_scope


def _chatgpt_session(native_id: str, *texts: str) -> dict[str, object]:
    mapping: dict[str, object] = {}
    previous: str | None = None
    for index, text in enumerate(texts):
        node_id = f"{native_id}-node-{index}"
        mapping[node_id] = {
            "id": node_id,
            "parent": previous,
            "children": [],
            "message": {
                "id": f"{native_id}-message-{index}",
                "author": {"role": "user" if index % 2 == 0 else "assistant"},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": 1_700_000_000 + index,
            },
        }
        if previous is not None:
            previous_row = mapping[previous]
            assert isinstance(previous_row, dict)
            previous_row["children"] = [node_id]
        previous = node_id
    return {
        "id": native_id,
        "conversation_id": native_id,
        "title": native_id,
        "create_time": 1_700_000_000,
        "update_time": 1_700_000_000 + len(texts),
        "current_node": previous,
        "mapping": mapping,
    }


def _single_session_state_db_bytes(tmp_path: Path) -> bytes:
    """The retained material for a single-session Hermes ``state.db``.

    Acquisition retains the declared member's canonical logical export, not a
    page image, and both the replay route and the historical-backfill route
    refuse anything else -- so the fixture must be built under the member's
    declared filename and export scope to be replayable at all.
    """
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (19);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, model_config TEXT, parent_session_id TEXT,
                started_at REAL, ended_at REAL, end_reason TEXT, title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
                timestamp REAL NOT NULL, tool_calls TEXT, observed INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES ('root', 'cli', '{}', 1.0, 8.0, 'completed', 'root')"
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, 'root', 'user', 'hi', 2.0)"
        )
    return logical_export_bytes(db_path, scope=member_export_scope(db_path))


def _relationship_index_jsonl_bytes(count: int = 8) -> bytes:
    """Bytes shaped like the real sinex analysis artifact from polylogue-9ykn
    (gvgi): a graph-edge index sitting under a watched Claude Code directory,
    with no session/message envelope at all -- just conversation/parent/
    child/type/timestamp keys, whose ``type`` happens to be a bare
    "assistant"/"user" role word.
    """
    lines = [
        json.dumps(
            {
                "conversation": f"conv-{index}",
                "parent": f"parent-{index}",
                "child": f"child-{index}",
                "type": "assistant" if index % 2 else "user",
                "timestamp": "2026-05-01T00:00:00.000Z",
            }
        )
        for index in range(count)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


_CLAUDE_USER_RECORD = (
    b'{"parentUuid":null,"type":"user","sessionId":"strict-replay","message":{"role":"user","content":"kept"},'
    b'"uuid":"user-1","timestamp":"2025-01-01T00:00:00Z"}\n'
)


_CLAUDE_ASSISTANT_RECORD = (
    b'{"parentUuid":"user-1","type":"assistant","sessionId":"strict-replay","message":{"role":"assistant",'
    b'"content":[{"type":"text","text":"reply"}]},"uuid":"assistant-1","timestamp":"2025-01-01T00:00:01Z"}\n'
)


def _codex_thread_state_snapshot_bytes(tmp_path: Path, title: str) -> bytes:
    """Return the retained material for one Codex state revision: its export."""
    from polylogue.sources.sqlite_export import logical_export_bytes
    from polylogue.sources.sqlite_snapshot import member_export_scope

    state_path = tmp_path / f"{title}" / "state_5.sqlite"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(state_path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY, title TEXT, cwd TEXT, created_at_ms INTEGER,
                updated_at_ms INTEGER, source TEXT, model TEXT, agent_nickname TEXT,
                agent_role TEXT, archived INTEGER
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT, child_thread_id TEXT, status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("codex-state-thread", title, "/work", 1, 1, "cli", None, None, None, 0),
        )
        conn.commit()
    return logical_export_bytes(state_path, scope=member_export_scope(state_path))


def _codex_thread_state_page_image_bytes(tmp_path: Path, title: str) -> bytes:
    """Return the PAGE IMAGE of one Codex state database, not its export.

    Pre-logical-export acquisition retained the raw SQLite file. That material
    is still in the archive, and a cold rebuild must account for it without
    parsing it and without ending the run.
    """
    state_path = tmp_path / f"{title}" / "state_5.sqlite"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(state_path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY, title TEXT, cwd TEXT, created_at_ms INTEGER,
                updated_at_ms INTEGER, source TEXT, model TEXT, agent_nickname TEXT,
                agent_role TEXT, archived INTEGER
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT, child_thread_id TEXT, status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("codex-state-thread", title, "/work", 1, 1, "cli", None, None, None, 0),
        )
        conn.commit()
    return state_path.read_bytes()


def _antigravity_trajectory_db_bytes(tmp_path: Path) -> bytes:
    database = tmp_path / "trajectory.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE trajectory_meta(trajectory_id TEXT, cascade_id TEXT);"
            "CREATE TABLE steps(idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT, trajectory_id TEXT);"
            "CREATE TABLE parent_references(cascade_id TEXT, parent_id TEXT);"
            "CREATE TABLE conversation_summaries(cascade_id TEXT, title TEXT, last_modified_time TEXT);"
            "INSERT INTO trajectory_meta VALUES ('trajectory-1','cascade-1');"
            "INSERT INTO conversation_summaries VALUES ('cascade-1','Complete title','2026-01-01T00:00:00Z');"
            "INSERT INTO steps VALUES (0,'message','v1','{\"role\":\"user\",\"text\":\"retained text\"}','trajectory-1');"
            "INSERT INTO steps VALUES (1,'future_step','v2','{\"opaque\":\"retained evidence\"}','trajectory-1');"
            "INSERT INTO parent_references VALUES ('cascade-1','parent-a');"
            "INSERT INTO parent_references VALUES ('cascade-1','parent-b');"
        )
    return logical_export_bytes(database)
