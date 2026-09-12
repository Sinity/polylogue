"""End-to-end proof that Codex live SQLite state is acquired as one logical
export and its evidence reaches existing read models, never as a session of
its own or a bespoke Codex durable table.

Drives the real ``LiveBatchProcessor`` (acquire -> parse -> archive write),
exactly the daemon's own full-ingest path -- not a mock of the join, not a
unit test of ``sources/parsers/codex_state.py`` in isolation (that already
exists in ``tests/unit/sources/parsers/test_codex_state.py``). The production
surface under test is ``sources/live/batch.py``'s acquire-loop branch that
exports ``state_5.sqlite`` and the parse-stage branch that calls
``record_codex_state_snapshot_terminal`` ->
``codex_state_projection.apply_retained_state_export``. Removing either wiring
point (or reverting the acquire loop to a raw ``path.read_bytes()``) makes the
assertions below fail -- this is not a self-validating mock: the join runs
against a real acquired export blob and a real archive.

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

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import MaterialObservation

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
async def test_codex_state_thread_title_and_spawn_edge_reach_the_index_tier(
    workspace_env: dict[str, Path],
) -> None:
    """threads.title and thread_spawn_edges reach index.db as derived rows.

    Anti-vacuity: this fails if the projection write (or its call site in
    ``sources/live/batch.py``) is removed -- both tables would then be empty.
    The blob-ref assertion fails if the retired per-row hook minting comes
    back: it wrote one durable ``hook_payload`` blob per thread and per edge.
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

        with sqlite3.connect(workspace_env["archive_root"] / "index.db") as index_conn:
            threads = index_conn.execute(
                "SELECT thread_id, title FROM codex_thread_state ORDER BY thread_id"
            ).fetchall()
            edges = index_conn.execute(
                "SELECT parent_thread_id, child_thread_id, status FROM codex_thread_spawn_edges"
            ).fetchall()
            provenance = index_conn.execute("SELECT raw_id, blob_hash FROM codex_thread_state_provenance").fetchall()
        assert [row[0] for row in threads] == [_THREAD_ID]
        assert threads[0][1]
        assert edges == [(_THREAD_ID, _CHILD_THREAD_ID, "closed")]
        assert len(provenance) == 1

        with sqlite3.connect(workspace_env["archive_root"] / "source.db") as source_conn:
            hook_payload_refs = source_conn.execute(
                "SELECT count(*) FROM blob_refs WHERE ref_type = 'hook_payload'"
            ).fetchone()[0]
            hook_events = source_conn.execute("SELECT count(*) FROM raw_hook_events").fetchone()[0]
        assert hook_payload_refs == 0
        assert hook_events == 0
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


def _write_goals_1_sqlite(
    path: Path,
    *,
    objective: str = "synthetic objective",
    status: str = "active",
    token_budget: int | None = 100_000,
    tokens_used: int = 4_200,
) -> None:
    """Write one current ``goals_1.sqlite`` logical export fixture."""
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE thread_goals (
                thread_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                token_budget INTEGER,
                tokens_used INTEGER NOT NULL,
                time_used_seconds INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE thread_goal_continuation_deferrals (thread_id TEXT, deferred_until_ms INTEGER);
            """
        )
        conn.execute(
            "INSERT INTO thread_goals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_THREAD_ID, "goal-1", objective, status, token_budget, tokens_used, 900, 1_000, 2_000),
        )
        conn.commit()


def _write_memories_1_sqlite(
    path: Path,
    *,
    raw_memory: str = "generated memory text",
    rollout_summary: str = "generated rollout summary",
    usage_count: int = 3,
) -> None:
    """Write one provider-generated ``memories_1.sqlite`` fixture."""
    path.unlink(missing_ok=True)
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
            CREATE TABLE jobs (id TEXT PRIMARY KEY, state TEXT);
            """
        )
        conn.execute(
            "INSERT INTO stage1_outputs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _THREAD_ID,
                1_700_000_100,
                raw_memory,
                rollout_summary,
                "synthetic-rollout",
                1_700_000_110,
                usage_count,
                1_700_000_120,
                1,
                None,
            ),
        )
        conn.commit()


def _material_payloads(archive_root: Path) -> list[tuple[MaterialObservation, dict[str, object]]]:
    """Read retained Codex content through the public material reader."""
    from polylogue.storage.materials import list_materials, read_material

    with sqlite3.connect(archive_root / "source.db") as conn:
        observations = list_materials(conn, evidence_ref=_CODEX_SESSION_ID)
        return [
            (
                observation,
                json.loads(read_material(conn, observation.material_id, blob_store=BlobStore(archive_root / "blob"))),
            )
            for observation in observations
        ]


@pytest.mark.asyncio
async def test_codex_goals_and_memories_survive_as_scoped_public_materials(
    workspace_env: dict[str, Path],
) -> None:
    """Ordinary ingest derives public, provenance-bearing state evidence.

    Anti-vacuity: removing the goals/memories branch from
    ``record_codex_state_snapshot_terminal`` leaves this reader empty. The
    native databases are removed before the final read, so this proves the
    evidence comes from retained exports and generic material blobs.
    """
    from polylogue.storage.materials import list_material_links

    archive, codex_root, codex_state_root = _make_processor(
        workspace_env, "codex-home-materials", "codex-state-materials.db"
    )
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, suffixes=(".sqlite", ".db")),
        ),
        cursor=CursorStore(workspace_env["data_root"] / "codex-state-materials.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    archive_root = workspace_env["archive_root"]
    try:
        rollout_path = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        overlapping_rollout = codex_root / f"rollout-2026-07-20T10-00-01-{_THREAD_ID}.jsonl"
        _write_codex_rollout(rollout_path)
        _write_codex_rollout(overlapping_rollout)
        assert (
            await processor.ingest_files([rollout_path, overlapping_rollout], emit_event=False)
        ).failed_file_count == 0

        goals_path = codex_state_root / "goals_1.sqlite"
        memories_path = codex_state_root / "memories_1.sqlite"
        _write_goals_1_sqlite(goals_path)
        _write_memories_1_sqlite(memories_path)
        state_metrics = await processor.ingest_files([goals_path, memories_path], emit_event=False)
        assert state_metrics.failed_file_count == 0
        assert state_metrics.ingested_session_count == 0

        first = _material_payloads(archive_root)
        assert len(first) == 2
        first_by_generated = {payload["generated"]: (observation, payload) for observation, payload in first}
        goal_observation, goal = first_by_generated[False]
        memory_observation, memory = first_by_generated[True]
        assert goal == {
            "created_at_ms": 1_000,
            "generated": False,
            "goal_id": "goal-1",
            "objective": "synthetic objective",
            "provider": "codex",
            "status": "active",
            "thread_id": _THREAD_ID,
            "time_used_seconds": 900,
            "token_budget": 100_000,
            "tokens_used": 4_200,
            "updated_at_ms": 2_000,
        }
        assert memory["raw_memory"] == "generated memory text"
        assert memory["rollout_summary"] == "generated rollout summary"
        assert memory["usage_count"] == 3
        assert memory["generated"] is True
        encoded_scope = codex_state_root.as_posix().replace("/", "%2F")
        assert f"scope={encoded_scope}" in goal_observation.source_uri
        assert f"/{_THREAD_ID}/goal-1?" in goal_observation.source_uri
        assert f"/{_THREAD_ID}/{_THREAD_ID}?" in memory_observation.source_uri

        with sqlite3.connect(archive_root / "source.db") as conn:
            goal_links = list_material_links(conn, goal_observation.material_id)
            assert {(link.relation, link.authority) for link in goal_links} == {
                ("acquired_from", "provider"),
                ("refers_to", "provider"),
            }
            assert {link.evidence_ref for link in goal_links if link.relation == "refers_to"} == {_CODEX_SESSION_ID}
        with sqlite3.connect(archive_root / "user.db") as conn:
            assert conn.execute("SELECT count(*) FROM assertions").fetchone() == (0,)

        # Replaying identical state and overlapping rollout evidence creates
        # no second value and never sums Codex's source usage counter.
        assert (
            await processor.ingest_files([goals_path, memories_path, overlapping_rollout], emit_event=False)
        ).failed_file_count == 0
        replayed = _material_payloads(archive_root)
        assert len(replayed) == 2
        assert {payload.get("usage_count") for _observation, payload in replayed if payload["generated"]} == {3}

        # Native deletion is only an absence observation. Public material reads
        # remain byte-for-byte available from archive blobs.
        goals_path.unlink()
        memories_path.unlink()
        retained = _material_payloads(archive_root)
        assert [(observation.material_id, payload) for observation, payload in retained] == [
            (observation.material_id, payload) for observation, payload in replayed
        ]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_codex_goal_materials_do_not_cross_supersede_source_roots(
    workspace_env: dict[str, Path],
) -> None:
    """R3: equal Codex coordinates in distinct installs remain distinct."""
    archive_root = workspace_env["archive_root"]
    root_a = workspace_env["data_root"] / "codex-install-a"
    root_b = workspace_env["data_root"] / "codex-install-b"
    sessions_a = root_a / "sessions"
    sessions_a.mkdir(parents=True)
    root_b.mkdir()
    archive = Polylogue(archive_root=archive_root, db_path=workspace_env["data_root"] / "codex-state-scopes.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex-a", root=sessions_a),
            WatchSource(name="codex-state-a", root=root_a, suffixes=(".sqlite", ".db")),
            WatchSource(name="codex-state-b", root=root_b, suffixes=(".sqlite", ".db")),
        ),
        cursor=CursorStore(workspace_env["data_root"] / "codex-state-scopes.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        rollout_path = sessions_a / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        _write_codex_rollout(rollout_path)
        assert (await processor.ingest_files([rollout_path], emit_event=False)).failed_file_count == 0
        goals_a = root_a / "goals_1.sqlite"
        goals_b = root_b / "goals_1.sqlite"
        _write_goals_1_sqlite(goals_a, objective="goal from install A")
        _write_goals_1_sqlite(goals_b, objective="goal from install B")
        assert (await processor.ingest_files([goals_a, goals_b], emit_event=False)).failed_file_count == 0

        materials = _material_payloads(archive_root)
        assert len(materials) == 2
        assert {payload["objective"] for _observation, payload in materials} == {
            "goal from install A",
            "goal from install B",
        }
        assert {observation.source_uri for observation, _payload in materials} == {
            f"codex://state/goal/{_THREAD_ID}/goal-1?scope={root_a.as_posix().replace('/', '%2F')}",
            f"codex://state/goal/{_THREAD_ID}/goal-1?scope={root_b.as_posix().replace('/', '%2F')}",
        }
    finally:
        await archive.close()


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


@pytest.mark.asyncio
async def test_fresh_root_catch_up_completes_every_chunk_with_a_codex_state_snapshot(
    workspace_env: dict[str, Path],
) -> None:
    """polylogue-6q16u: a fresh archive root whose scan contains one
    ``~/.codex/*.sqlite`` finishes the whole catch-up.

    The rehearsal-4 failure was not a single ingest call: the watcher planned
    45,490 files, ingested three chunks, and then every later chunk, the
    hook-spool drain and raw materialization were refused with ``1 cursor/head
    authority row(s) could not be compared``. This drives the real chunked
    catch-up route (``LiveWatcher._catch_up`` -> ``_plan_catch_up`` ->
    coordinated chunk ingest) over enough files for five chunks, with the
    snapshot raw among them.

    Anti-vacuity: drop the terminal source-tier receipt from the codex-state
    branch of ``LiveBatchProcessor._ingest_full_records_archive`` and the
    chunk that admits ``goals_1.sqlite`` leaves an uncomparable cursor row, so
    the following chunks ingest nothing and the session count stops short.
    """
    from polylogue.readiness.capability import raw_frontier_source_selection_block_reason

    archive, codex_root, codex_state_root = _make_processor(workspace_env, "codex-home-catchup", "codex-catchup.db")
    archive_root = workspace_env["archive_root"]
    _write_goals_1_sqlite(codex_state_root / "goals_1.sqlite")
    template = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
    _write_codex_rollout(template)
    rollout_count = 17
    thread_ids = [f"{index:08x}-1b42-43a5-977c-870299c489a6" for index in range(rollout_count)]
    for index, thread_id in enumerate(thread_ids):
        path = codex_root / f"rollout-2026-07-20T10-00-{index:02d}-{thread_id}.jsonl"
        path.write_text(template.read_text(encoding="utf-8").replace(_THREAD_ID, thread_id), encoding="utf-8")
    template.unlink()

    watcher = live_watcher.LiveWatcher(
        archive,
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, suffixes=(".sqlite", ".db")),
        ),
        cursor=CursorStore(archive_root / "ops.db"),
    )
    try:
        candidates = watcher._scan_catch_up_candidates([codex_root, codex_state_root])
        assert len(candidates) == rollout_count + 1
        chunks = watcher._chunk_catch_up_paths(
            tuple(candidate.path for candidate in candidates),
            {candidate.path: candidate for candidate in candidates},
        )
        assert len(chunks) >= 5, "the deadlock only showed after chunk 3"

        await watcher._catch_up([codex_root, codex_state_root])

        assert raw_frontier_source_selection_block_reason(archive_root) is None
        assert _cursor_authority_gap_states(archive_root) == []
        assert watcher._batch_processor.cursor_authority_block_reason() is None
        assert await archive.count_sessions() == rollout_count
    finally:
        watcher.stop()
        await archive.close()
