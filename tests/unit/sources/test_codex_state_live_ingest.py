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
from polylogue.storage.blob_store import BlobStore

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


def _write_state_5_sqlite(path: Path, *, wal_mode: bool = False) -> None:
    with sqlite3.connect(path) as conn:
        if wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
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
        conn.commit()
        if wal_mode:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


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
        _write_state_5_sqlite(state_path, wal_mode=True)
        state_metrics = await processor.ingest_files([state_path], emit_event=False)
        assert state_metrics.failed_file_count == 0
        # The state db produces zero NEW sessions -- its evidence attaches to
        # the codex-session row the JSONL rollout already created.
        assert state_metrics.ingested_session_count == 0

        after_count = await archive.count_sessions()
        assert after_count == before_count == 1
        assert BlobStore(workspace_env["archive_root"] / "blob").verify_all().passed
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


def _write_goals_1_sqlite(path: Path) -> None:
    """``goals_1.sqlite`` (CODEX_STATE_FIDELITY: acquire-partial) -- a table
    shape with no threads, so nothing but the raw snapshot itself is evidence."""
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE thread_goals (thread_id TEXT PRIMARY KEY, objective TEXT, updated_at_ms INTEGER);
            CREATE TABLE thread_goal_continuation_deferrals (thread_id TEXT, deferred_until_ms INTEGER);
            """
        )
        conn.execute(
            "INSERT INTO thread_goals (thread_id, objective, updated_at_ms) VALUES (?, ?, ?)",
            (_THREAD_ID, "synthetic objective", 1000),
        )
        conn.commit()


def _cursor_authority_gap_states(archive_root: Path) -> list[str]:
    from polylogue.readiness.capability import raw_frontier_integrity_projection
    from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot

    projection = raw_frontier_integrity_projection(archive_root, raw_materialization_readiness_snapshot(archive_root))
    return [sample.state for sample in projection.cursor_authority_gap_samples]


@pytest.mark.asyncio
async def test_codex_state_snapshot_raw_never_blocks_cursor_authority(
    workspace_env: dict[str, Path],
) -> None:
    """polylogue-6q16u: a fresh root that has acquired ``~/.codex/goals_1.sqlite``
    converges instead of deadlocking on the cursor-authority gate.

    The snapshot raw is non-session evidence with no byte frontier, so its
    terminal source-tier receipt (the ``non_session`` membership census plus a
    finalized parse state) must be written by the same live-ingest pass that
    admits it. Anti-vacuity: drop the terminal receipt from the codex-state
    branch of ``LiveBatchProcessor._ingest_full_records_archive`` and the gate
    reports ``source_raws_without_accepted_head`` for the sqlite path, the
    block reason is non-empty, and the follow-up rollout ingest raises
    ``CursorAuthorityBlockedError`` -- exactly the rehearsal-4 failure.
    """
    from polylogue.readiness.capability import raw_frontier_source_selection_block_reason

    archive, codex_root, codex_state_root = _make_processor(workspace_env, "codex-home-gate", "codex-state-gate.db")
    archive_root = workspace_env["archive_root"]
    # The gate reads the archive's own ops-tier cursors, so the cursor store
    # must be the archive's, not a side database.
    cursor = CursorStore(archive_root / "ops.db")
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
        first_rollout = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        _write_codex_rollout(first_rollout)
        metrics = await processor.ingest_files([first_rollout], emit_event=False)
        assert metrics.ingested_session_count == 1
        assert processor.cursor_authority_block_reason() is None

        goals_path = codex_state_root / "goals_1.sqlite"
        _write_goals_1_sqlite(goals_path)
        state_metrics = await processor.ingest_files([goals_path], emit_event=False)
        assert state_metrics.failed_file_count == 0
        assert state_metrics.ingested_session_count == 0

        assert _cursor_authority_gap_states(archive_root) == []
        assert raw_frontier_source_selection_block_reason(archive_root) is None
        assert processor.cursor_authority_block_reason() is None

        with sqlite3.connect(archive_root / "source.db") as conn:
            rows = conn.execute(
                """
                SELECT c.status, r.parsed_at_ms IS NOT NULL, r.parse_error
                FROM raw_sessions AS r
                LEFT JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
                WHERE r.source_path = ?
                """,
                (str(goals_path),),
            ).fetchall()
        assert rows == [("non_session", 1, None)]

        # The whole point: the next backlog chunk is still admitted.
        second_rollout = codex_root / f"rollout-2026-07-21T10-00-00-{_CHILD_THREAD_ID}.jsonl"
        second_rollout.write_text(
            first_rollout.read_text(encoding="utf-8").replace(_THREAD_ID, _CHILD_THREAD_ID), encoding="utf-8"
        )
        follow_up = await processor.ingest_files([second_rollout], emit_event=False)
        assert follow_up.failed_file_count == 0
        assert follow_up.ingested_session_count == 1
        assert await archive.count_sessions() == 2
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_retained_codex_state_raw_without_receipt_is_resolved_from_the_blob(
    workspace_env: dict[str, Path],
) -> None:
    """A codex-state raw admitted before the terminal receipt existed (the live
    archive's ``goals_1``/``memories_1``/``state_5`` rows) is resolved from its
    immutable blob by ``resolve_retained_codex_state_receipts`` -- the step the
    daemon runs before the raw-materialization source-selection gate.

    Anti-vacuity: with the resolver a no-op, the seeded state keeps reporting
    ``source_raws_without_accepted_head`` and the gate stays blocked.
    """
    from polylogue.readiness.capability import raw_frontier_source_selection_block_reason
    from polylogue.sources.codex_state_evidence import resolve_retained_codex_state_receipts

    archive, codex_root, codex_state_root = _make_processor(workspace_env, "codex-home-legacy", "codex-state-legacy.db")
    archive_root = workspace_env["archive_root"]
    cursor = CursorStore(archive_root / "ops.db")
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
        rollout = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        _write_codex_rollout(rollout)
        await processor.ingest_files([rollout], emit_event=False)
        goals_path = codex_state_root / "goals_1.sqlite"
        _write_goals_1_sqlite(goals_path)
        state_path = codex_state_root / "state_5.sqlite"
        _write_state_5_sqlite(state_path)
        await processor.ingest_files([goals_path, state_path], emit_event=False)
        assert raw_frontier_source_selection_block_reason(archive_root) is None
    finally:
        await archive.close()

    # Seed the pre-receipt shape the live source tier carries: raw admitted,
    # cursor at EOF, no census, never finalized.
    with sqlite3.connect(archive_root / "source.db") as conn:
        raw_ids = [
            str(row[0])
            for row in conn.execute(
                "SELECT raw_id FROM raw_sessions WHERE source_path IN (?, ?)", (str(goals_path), str(state_path))
            )
        ]
        assert len(raw_ids) == 2
        placeholders = ",".join("?" for _ in raw_ids)
        conn.execute(f"DELETE FROM raw_membership_census WHERE raw_id IN ({placeholders})", raw_ids)
        conn.execute(f"DELETE FROM raw_authority_parser_census WHERE raw_id IN ({placeholders})", raw_ids)
        conn.execute(f"UPDATE raw_sessions SET parsed_at_ms = NULL WHERE raw_id IN ({placeholders})", raw_ids)
        conn.commit()
    assert sorted(_cursor_authority_gap_states(archive_root)) == ["source_raws_without_accepted_head"] * 2
    assert raw_frontier_source_selection_block_reason(archive_root) is not None

    assert resolve_retained_codex_state_receipts(archive_root) == 2
    assert _cursor_authority_gap_states(archive_root) == []
    assert raw_frontier_source_selection_block_reason(archive_root) is None
    # Idempotent: a second pass finds nothing left to resolve.
    assert resolve_retained_codex_state_receipts(archive_root) == 0
