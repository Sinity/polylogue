"""End-to-end proof that Codex live SQLite state (polylogue-0jf4) is acquired
and its evidence attaches to the EXISTING codex-session row it describes,
never as a session of its own.

Drives the real ``LiveBatchProcessor`` (acquire -> parse -> archive write),
exactly the daemon's own full-ingest path -- not a mock of the join, not a
unit test of ``sources/parsers/codex_state.py`` in isolation (that already
exists in ``tests/unit/sources/parsers/test_codex_state.py``). The production
surface under test is ``sources/live/batch.py``'s acquire-loop branch that
snapshots ``state_5.sqlite`` via the SQLite backup API and the parse-stage
branch that calls ``_write_codex_thread_state_evidence`` -> ``ArchiveStore
.write_hook_event``. Removing either wiring point (or reverting the acquire
loop to a raw ``path.read_bytes()``, or dropping the spawn-edge/title loop in
``_write_codex_thread_state_evidence``) makes the assertions below fail --
this is not a self-validating mock: the join runs against a real acquired
sqlite blob and a real archive.

Hard constraint (operator, 2026-07-29, precedent: polylogue-31r1 hook-event
inflation from 18,391 to 83,286 sessions): thread_spawn_edges/titles must
attach to the EXISTING codex-session row, never mint a session of their own.
``test_codex_state_ingest_leaves_session_count_unchanged`` is the direct
regression test for that constraint.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore

_THREAD_ID = "66c7b83d-1b42-43a5-977c-870299c489a6"
_CHILD_THREAD_ID = "449dd1eb-ea3d-4710-925b-7398a78fe3a7"
_CODEX_SESSION_ID = f"codex-session:{_THREAD_ID}"


def _write_codex_rollout(path: Path) -> None:
    """A minimal, synthetic Codex JSONL rollout -- no real transcript bytes,
    matching this repo's existing ``tests/data/codex_event_stream`` shape."""
    lines = [
        {
            "type": "session_meta",
            "payload": {"id": _THREAD_ID, "timestamp": "2026-07-20T10:00:00Z", "cwd": "/repo"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-user-1",
                "role": "user",
                "timestamp": "2026-07-20T10:00:05Z",
                "content": [{"type": "input_text", "text": "synthetic prompt"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-asst-1",
                "role": "assistant",
                "timestamp": "2026-07-20T10:00:08Z",
                "content": [{"type": "output_text", "text": "synthetic reply"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _write_state_5_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                title TEXT,
                cwd TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                source TEXT,
                model TEXT,
                agent_nickname TEXT,
                agent_role TEXT,
                archived INTEGER
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT,
                child_thread_id TEXT,
                status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO threads (id, title, cwd, created_at_ms, updated_at_ms, source, model, "
            "agent_nickname, agent_role, archived) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_THREAD_ID, "Synthetic curated title", "/repo", 1000, 2000, "cli", "gpt-synthetic", None, None, 0),
        )
        conn.execute(
            "INSERT INTO thread_spawn_edges (parent_thread_id, child_thread_id, status) VALUES (?, ?, ?)",
            (_THREAD_ID, _CHILD_THREAD_ID, "closed"),
        )


def _make_processor(workspace_env: dict[str, Path], root_name: str, db_name: str) -> tuple[Polylogue, Path, Path]:
    codex_root = workspace_env["data_root"] / root_name / "sessions"
    codex_root.mkdir(parents=True)
    codex_state_root = workspace_env["data_root"] / root_name
    db_path = workspace_env["data_root"] / db_name
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    return archive, codex_root, codex_state_root


@pytest.mark.asyncio
async def test_codex_state_ingest_leaves_session_count_unchanged(
    workspace_env: dict[str, Path],
) -> None:
    """AC1 (polylogue-0jf4): thread_spawn_edges/titles attach to the EXISTING
    session; ingesting state_5.sqlite mints ZERO new sessions. Same invariant
    shape as polylogue-rujy's tool-result-sidecar test, same incident
    precedent (polylogue-31r1)."""
    archive, codex_root, codex_state_root = _make_processor(
        workspace_env, "codex-home-unchanged", "codex-state-unchanged.db"
    )
    cursor = CursorStore(workspace_env["data_root"] / "codex-state-unchanged.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, suffixes=(".sqlite", ".db")),
        ),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        rollout_path = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        _write_codex_rollout(rollout_path)
        metrics = await processor.ingest_files([rollout_path], emit_event=False)
        assert metrics.failed_file_count == 0
        assert metrics.ingested_session_count == 1

        before_count = await archive.count_sessions()
        assert before_count == 1

        state_path = codex_state_root / "state_5.sqlite"
        _write_state_5_sqlite(state_path)
        state_metrics = await processor.ingest_files([state_path], emit_event=False)
        assert state_metrics.failed_file_count == 0
        # The state db produces zero NEW sessions -- its evidence attaches to
        # the codex-session row the JSONL rollout already created.
        assert state_metrics.ingested_session_count == 0

        after_count = await archive.count_sessions()
        assert after_count == before_count == 1
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_codex_state_thread_title_and_spawn_edge_attach_to_existing_session(
    workspace_env: dict[str, Path],
) -> None:
    """AC3 (polylogue-0jf4): threads.title and thread_spawn_edges reach the
    archive as typed evidence, joined to the existing session by thread_id.

    Anti-vacuity: this fails if ``_write_codex_thread_state_evidence`` (or
    its call site in ``sources/live/batch.py``) is removed -- the summary
    would then be empty/None rather than carrying the two new event types.
    """
    archive, codex_root, codex_state_root = _make_processor(
        workspace_env, "codex-home-evidence", "codex-state-evidence.db"
    )
    cursor = CursorStore(workspace_env["data_root"] / "codex-state-evidence.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, suffixes=(".sqlite", ".db")),
        ),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        rollout_path = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        _write_codex_rollout(rollout_path)
        metrics = await processor.ingest_files([rollout_path], emit_event=False)
        assert metrics.ingested_session_count == 1

        state_path = codex_state_root / "state_5.sqlite"
        _write_state_5_sqlite(state_path)
        state_metrics = await processor.ingest_files([state_path], emit_event=False)
        assert state_metrics.failed_file_count == 0

        summary = await archive.get_hook_event_summary_for_session(_CODEX_SESSION_ID)
        assert summary is not None
        by_event_type = cast("dict[str, int]", summary["by_event_type"])
        assert by_event_type.get("codex_thread_title") == 1
        assert by_event_type.get("codex_thread_spawn_edge") == 1
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_codex_out_of_scope_state_db_is_excluded_not_read(
    workspace_env: dict[str, Path],
) -> None:
    """logs_2.sqlite/codex-dev.db (CODEX_STATE_FIDELITY: out-of-scope) are
    excluded by filename before any bytes are read -- never acquired, never
    a failed-parse record."""
    archive, codex_root, codex_state_root = _make_processor(
        workspace_env, "codex-home-out-of-scope", "codex-state-out-of-scope.db"
    )
    cursor = CursorStore(workspace_env["data_root"] / "codex-state-out-of-scope.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="codex-state", root=codex_state_root, suffixes=(".sqlite", ".db")),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        logs_path = codex_state_root / "logs_2.sqlite"
        with sqlite3.connect(logs_path) as conn:
            conn.executescript(
                "CREATE TABLE logs (ts INTEGER, level TEXT, target TEXT, module_path TEXT, file TEXT, line INTEGER);"
            )
            conn.execute(
                "INSERT INTO logs (ts, level, target, module_path, file, line) VALUES (1, 'INFO', 't', 'm', 'f', 1)"
            )

        metrics = await processor.ingest_files([logs_path], emit_event=False)
        assert metrics.failed_file_count == 0
        assert metrics.ingested_session_count == 0

        assert await archive.count_sessions() == 0
    finally:
        await archive.close()
