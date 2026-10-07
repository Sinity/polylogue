"""End-to-end proof that Codex live SQLite state is acquired as one logical
export and its evidence reaches existing read models, never as a session of
its own or a bespoke Codex durable table.

Drives the real ``LiveBatchProcessor`` (acquire -> parse -> archive write),
exactly the daemon's own full-ingest path -- not a mock of the join, not a
unit test of ``sources/parsers/codex_state.py`` in isolation (that already
exists in ``tests/unit/sources/parsers/test_codex_state.py``). The production
surface under test is ``sources/live/batch.py``'s acquire-loop branch that
exports ``state_5.sqlite`` and the supplied retained owner that calls
``prepare_codex_state_source_terminal`` with the same prepared thread projection. Removing either wiring
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

import functools
import json
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import MaterialObservation
from polylogue.storage.sqlite.agent_thread_state import read_provenance, read_spawn_edges, read_thread_titles
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.live_batch import prepared_live_batch_processor
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.raw_owner_routes import LiveOwnerSet, ingest_files_with_owners, live_owner_set


def _lease_writer(root: Path) -> Callable[[str, Callable[[], bool]], bool]:
    def writer(actor: str, work: Callable[[], bool]) -> bool:
        with write_lease(actor, archive_root=root):
            return work()

    return writer


async def _live_failure_details(processor: LiveBatchProcessor, root: Path) -> str:
    from polylogue.operations.user_overlay_reads import readable_required_tier
    from polylogue.storage.io_phase_metrics import connection_cursor
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    def original_error() -> str:
        with (
            readable_required_tier(root / "ops.db", ArchiveTier.OPS) as connection,
            connection_cursor(
                connection,
                "SELECT phase,error_message FROM ingest_attempts "
                "ORDER BY COALESCE(heartbeat_at_ms,finished_at_ms,started_at_ms) DESC,started_at_ms DESC LIMIT 1",
            ) as rows,
        ):
            row = rows.fetchone()
            return repr(tuple(row) if row is not None else None)

    runner = processor._convergence_runner
    assert runner is not None
    detail = await runner("fixture.live.original-error.read", original_error)
    if not isinstance(detail, str):
        raise TypeError("original Live error reader returned a non-text observation")
    return detail


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


def _write_state_5_sqlite(path: Path, *, wal_mode: bool = False, include_agent_role: bool = True) -> None:
    with sqlite3.connect(path) as conn:
        if wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            f"""
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                title TEXT,
                cwd TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                source TEXT,
                model TEXT,
                agent_nickname TEXT,
                {"agent_role TEXT," if include_agent_role else ""}
                archived INTEGER
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT,
                child_thread_id TEXT,
                status TEXT
            );
            """
        )
        role_columns = "agent_nickname, agent_role, " if include_agent_role else "agent_nickname, "
        role_values = "?, ?, " if include_agent_role else "?, "
        role_args = (None, None) if include_agent_role else (None,)
        conn.execute(
            "INSERT INTO threads (id, title, cwd, created_at_ms, updated_at_ms, source, model, "
            f"{role_columns}archived) VALUES (?, ?, ?, ?, ?, ?, ?, {role_values}?)",
            (_THREAD_ID, "Synthetic curated title", "/repo", 1000, 2000, "cli", "gpt-synthetic", *role_args, 0),
        )
        conn.execute(
            "INSERT INTO thread_spawn_edges (parent_thread_id, child_thread_id, status) VALUES (?, ?, ?)",
            (_THREAD_ID, _CHILD_THREAD_ID, "closed"),
        )
        conn.commit()
        if wal_mode:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


async def _inspect_accepted_frontier(archive_root: Path, owners: LiveOwnerSet | None = None) -> None:
    """Run the daemon's accepted-frontier inspection stage after a live pass.

    Source selection stays blocked until the frontier is inspected; the daemon
    runs this inspection as a convergence stage, never inside a live pass.
    """
    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier

    if owners is not None:
        await owners.raw_owner.run_convergence_sync(
            "test.codex-state.frontier",
            inspect_prepared_raw_authority_frontier,
            archive_root,
            input_demand=owners.compute.amend_current_input_demand,
        )
        return
    async with prepared_live_convergence_owner(archive_root) as owner:
        await owner.run_convergence_sync(
            "test.codex-state.frontier",
            inspect_prepared_raw_authority_frontier,
            archive_root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )


def _make_processor(workspace_env: dict[str, Path], root_name: str, db_name: str) -> tuple[Polylogue, Path, Path]:
    codex_root = workspace_env["data_root"] / root_name / "sessions"
    codex_root.mkdir(parents=True)
    codex_state_root = workspace_env["data_root"] / root_name
    db_path = workspace_env["archive_root"] / "index.db"
    run_off_event_loop(lambda: bootstrap_archive_root(workspace_env["archive_root"]))
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
    failures: list[str] = []
    async with prepared_live_batch_processor(
        workspace_env["archive_root"],
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        failure_details=failures,
    ) as processor:
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
            assert state_metrics.failed_file_count == 0, (
                failures,
                await _live_failure_details(processor, workspace_env["archive_root"]),
            )
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
    failures: list[str] = []
    async with prepared_live_batch_processor(
        workspace_env["archive_root"],
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        failure_details=failures,
    ) as processor:
        try:
            rollout_path = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
            _write_codex_rollout(rollout_path)
            metrics = await processor.ingest_files([rollout_path], emit_event=False)
            assert metrics.ingested_session_count == 1

            state_path = codex_state_root / "state_5.sqlite"
            _write_state_5_sqlite(state_path)
            state_metrics = await processor.ingest_files([state_path], emit_event=False)
            assert state_metrics.failed_file_count == 0, (
                failures,
                await _live_failure_details(processor, workspace_env["archive_root"]),
            )

            with sqlite3.connect(workspace_env["archive_root"] / "index.db") as index_conn:
                titles = read_thread_titles(index_conn)
                edges = read_spawn_edges(index_conn)
                provenance = read_provenance(index_conn)
            assert sorted(titles) == [_THREAD_ID]
            assert titles[_THREAD_ID]
            assert edges == {(_THREAD_ID, _CHILD_THREAD_ID): "closed"}
            assert provenance is not None and provenance.raw_id and provenance.blob_hash

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
    cursor = CursorStore(workspace_env["archive_root"] / "ops.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="codex-state", root=codex_state_root, layout=export_drop_layout((".sqlite", ".db"))),),
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

        metrics = await ingest_files_with_owners(processor, [logs_path], emit_event=False)
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
    ``prepare_codex_state_source_terminal`` leaves this reader empty. The
    native databases are removed before the final read, so this proves the
    evidence comes from retained exports and generic material blobs.
    """
    from polylogue.storage.materials import list_material_links

    archive, codex_root, codex_state_root = _make_processor(
        workspace_env, "codex-home-materials", "codex-state-materials.db"
    )
    failures: list[str] = []
    async with prepared_live_batch_processor(
        workspace_env["archive_root"],
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        failure_details=failures,
    ) as processor:
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
            assert state_metrics.failed_file_count == 0, (
                failures,
                await _live_failure_details(processor, workspace_env["archive_root"]),
            )
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
async def test_codex_state_embedded_nul_and_utf8_boundary_reach_complete_materials(
    workspace_env: dict[str, Path],
) -> None:
    """An embedded NUL and a split UTF-8 codepoint cannot hide a text suffix.

    Anti-vacuity: SQLite TEXT length/substr truncates at NUL, so using either
    for chunking loses the suffix while falsely writing a complete terminal.
    """
    from polylogue.storage.materials import list_materials_page, read_material

    archive, codex_root, codex_state_root = _make_processor(workspace_env, "codex-home-nul", "codex-state-nul.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        cursor=CursorStore(workspace_env["archive_root"] / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    archive_root = workspace_env["archive_root"]
    # Ω straddles the reader's 64 KiB byte window; the NUL is after it.
    goal_text = "a" * 65_535 + "Ω\x00goal-tail"
    memory_text = "b" * 65_535 + "Ω\x00memory-tail"
    summary_text = "summary\x00summary-tail"
    try:
        goals_path = codex_state_root / "goals_1.sqlite"
        memories_path = codex_state_root / "memories_1.sqlite"
        _write_goals_1_sqlite(goals_path, objective=goal_text)
        _write_memories_1_sqlite(memories_path, raw_memory=memory_text, rollout_summary=summary_text)
        result = await ingest_files_with_owners(processor, [goals_path, memories_path], emit_event=False)
        assert result.failed_file_count == 0

        with sqlite3.connect(archive_root / "source.db") as conn:
            page = list_materials_page(conn, evidence_ref=_CODEX_SESSION_ID, limit=10)
            assert page.next_cursor is None
            parts = [
                json.loads(read_material(conn, item.material_id, blob_store=BlobStore(archive_root / "blob")))
                for item in page.items
            ]
            terminals = conn.execute(
                """
                SELECT r.source_path, c.status, c.detail, r.parse_error
                FROM raw_sessions AS r
                JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
                WHERE r.source_path IN (?, ?)
                """,
                (str(goals_path), str(memories_path)),
            ).fetchall()

        goal = next(part for part in parts if part.get("goal_id") == "goal-1")
        memory = next(part for part in parts if "raw_memory" in part)
        chunks = {(part["record_type"], part["field"]): part for part in parts if "field" in part}
        assert goal["objective"] + chunks[("goals", "objective")]["text"] == goal_text
        assert memory["raw_memory"] + chunks[("memories", "raw_memory")]["text"] == memory_text
        assert memory["rollout_summary"] == summary_text
        assert goal["text_continuation"]["objective"]["length_chars"] == len(goal_text)
        assert memory["text_continuation"]["raw_memory"]["length_chars"] == len(memory_text)
        assert chunks[("goals", "objective")]["offset_chars"] == 64_000
        assert chunks[("memories", "raw_memory")]["offset_chars"] == 64_000
        assert all(chunk["final"] for chunk in chunks.values())
        assert len(terminals) == 2
        assert all(
            status == "non_session" and "complete" in detail and error is None
            for _path, status, detail, error in terminals
        )
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
    run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
    archive = Polylogue(archive_root=archive_root, db_path=workspace_env["data_root"] / "codex-state-scopes.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=sessions_a),
            WatchSource(name="codex-state", root=root_a, layout=export_drop_layout((".sqlite", ".db"))),
            WatchSource(name="codex-state", root=root_b, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        cursor=CursorStore(workspace_env["archive_root"] / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        rollout_path = sessions_a / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        _write_codex_rollout(rollout_path)
        assert (await ingest_files_with_owners(processor, [rollout_path], emit_event=False)).failed_file_count == 0
        goals_a = root_a / "goals_1.sqlite"
        goals_b = root_b / "goals_1.sqlite"
        _write_goals_1_sqlite(goals_a, objective="goal from install A")
        _write_goals_1_sqlite(goals_b, objective="goal from install B")
        assert (await ingest_files_with_owners(processor, [goals_a, goals_b], emit_event=False)).failed_file_count == 0

        materials = _material_payloads(archive_root)
        assert len(materials) == 2, materials
        assert {payload["objective"] for _observation, payload in materials} == {
            "goal from install A",
            "goal from install B",
        }
        source_uris = {observation.source_uri for observation, _payload in materials}
        assert source_uris == {
            f"codex://state/goals/{_THREAD_ID}/goal-1?scope={root_a.as_posix().replace('/', '%2F')}",
            f"codex://state/goals/{_THREAD_ID}/goal-1?scope={root_b.as_posix().replace('/', '%2F')}",
        }, source_uris
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
            WatchSource(name="codex-state", root=codex_state_root, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        first_rollout = codex_root / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
        _write_codex_rollout(first_rollout)
        metrics = await ingest_files_with_owners(processor, [first_rollout], emit_event=False)
        assert metrics.ingested_session_count == 1
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        assert (reason := processor.cursor_authority_block_reason()) is None, reason

        goals_path = codex_state_root / "goals_1.sqlite"
        _write_goals_1_sqlite(goals_path)
        state_metrics = await ingest_files_with_owners(processor, [goals_path], emit_event=False)
        assert state_metrics.failed_file_count == 0
        assert state_metrics.ingested_session_count == 0

        assert _cursor_authority_gap_states(archive_root) == []
        await _inspect_accepted_frontier(archive_root)
        assert (reason := raw_frontier_source_selection_block_reason(archive_root)) is None, reason
        assert (reason := processor.cursor_authority_block_reason()) is None, reason

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
        follow_up = await ingest_files_with_owners(processor, [second_rollout], emit_event=False)
        assert follow_up.failed_file_count == 0
        assert follow_up.ingested_session_count == 1
        assert await archive.count_sessions() == 2
    finally:
        await archive.close()


def test_historical_codex_page_image_is_not_finalized_as_current_state(
    workspace_env: dict[str, Path],
) -> None:
    """A page image gets its diagnostic receipt without a current state projection."""
    from polylogue.core.enums import Provider
    from tests.infra.retained_replay import replay_retained_components

    archive_root = workspace_env["archive_root"]
    state_path = workspace_env["data_root"] / "state_5.sqlite"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _write_state_5_sqlite(state_path)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=state_path.read_bytes(),
            source_path=str(state_path),
            canonical_source_path=str(state_path),
            acquired_at_ms=1_767_000_000_000,
        )
        archive.commit()

    replay_retained_components(archive_root, selected_raw_ids=[raw_id])
    from polylogue.sources.revision_backfill import LEGACY_PAGE_IMAGE_CENSUS_DETAIL

    with sqlite3.connect(archive_root / "source.db") as conn:
        detail = conn.execute("SELECT detail FROM raw_membership_census WHERE raw_id = ?", (raw_id,)).fetchone()
        assert detail is not None and LEGACY_PAGE_IMAGE_CENSUS_DETAIL in str(detail[0])
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        assert archive.index_connection is not None
        assert read_thread_titles(archive.index_connection, thread_ids=[_THREAD_ID]) == {}


@pytest.mark.asyncio
async def test_schema_drift_candidate_does_not_block_other_retained_state_receipts(
    workspace_env: dict[str, Path],
) -> None:
    """A malformed thread-state schema is isolated while valid candidates finalize.

    Anti-vacuity: coupling preparation to a malformed sibling prevents the
    valid goals snapshot from receiving its terminal receipt.
    """
    from tests.infra.retained_replay import replay_retained_components_async

    archive, codex_root, codex_state_root = _make_processor(
        workspace_env, "codex-home-schema-drift", "codex-state-schema-drift.db"
    )
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=codex_state_root, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        cursor=CursorStore(workspace_env["archive_root"] / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    bad_path = codex_state_root / "state_5.sqlite"
    good_path = codex_state_root / "goals_1.sqlite"
    _write_state_5_sqlite(bad_path, include_agent_role=False)
    _write_goals_1_sqlite(good_path)
    try:
        await ingest_files_with_owners(processor, [bad_path, good_path], emit_event=False)
    finally:
        await archive.close()

    with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
        rows = conn.execute(
            "SELECT raw_id, source_path FROM raw_sessions WHERE source_path IN (?, ?) ORDER BY source_path",
            (str(bad_path), str(good_path)),
        ).fetchall()
        assert len(rows) == 2
        raw_ids = [str(row[0]) for row in rows]
        placeholders = ",".join("?" for _ in raw_ids)
        conn.execute(f"DELETE FROM raw_membership_census WHERE raw_id IN ({placeholders})", raw_ids)
        conn.execute(f"DELETE FROM raw_authority_parser_census WHERE raw_id IN ({placeholders})", raw_ids)
        conn.execute(
            f"UPDATE raw_sessions SET parsed_at_ms = NULL, parse_error = NULL WHERE raw_id IN ({placeholders})", raw_ids
        )
        conn.commit()

    retained_by_path = {str(path): str(raw_id) for raw_id, path in rows}
    await replay_retained_components_async(
        workspace_env["archive_root"],
        selected_raw_ids=[retained_by_path[str(good_path)]],
    )
    with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
        states = dict(
            conn.execute(
                "SELECT r.source_path, c.status FROM raw_sessions AS r "
                "LEFT JOIN raw_membership_census AS c USING (raw_id) WHERE r.source_path IN (?, ?)",
                (str(bad_path), str(good_path)),
            ).fetchall()
        )
    assert states[str(bad_path)] is None
    assert states[str(good_path)] == "non_session"


@pytest.mark.asyncio
async def test_a_fresh_root_admits_every_page_with_a_codex_state_snapshot_among_them(
    workspace_env: dict[str, Path],
) -> None:
    """polylogue-6q16u: a fresh archive root whose walk contains one
    ``~/.codex/*.sqlite`` finishes every page.

    The rehearsal-4 failure was not a single ingest call: the run ingested
    three batches, and then every later batch, plus raw materialization, was
    refused with ``1 cursor/head authority row(s) could not be compared``.
    This drives the real intake route (``FairIntakeDispatcher.run_once`` ->
    ``FileIntakeAdapter.admit_page`` -> ``LiveWatcher._ingest_files``) over
    enough files for five pages, with the snapshot raw among them.

    Anti-vacuity: drop the terminal source-tier receipt from the codex-state
    branch of ``LiveBatchProcessor._ingest_full_records_archive`` and the page
    that admits ``goals_1.sqlite`` leaves an uncomparable cursor row, so the
    following pages ingest nothing and the session count stops short.
    """
    from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
    from polylogue.readiness.capability import raw_frontier_source_selection_block_reason

    archive, codex_root, codex_state_root = _make_processor(workspace_env, "codex-home-catchup", "codex-catchup.db")
    archive_root = workspace_env["archive_root"]
    _write_goals_1_sqlite(codex_state_root / "goals_1.sqlite")
    day = codex_root / "2026" / "07" / "20"
    day.mkdir(parents=True, exist_ok=True)
    template = day / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
    _write_codex_rollout(template)
    rollout_count = 17
    thread_ids = [f"{index:08x}-1b42-43a5-977c-870299c489a6" for index in range(rollout_count)]
    for index, thread_id in enumerate(thread_ids):
        path = day / f"rollout-2026-07-20T10-00-{index:02d}-{thread_id}.jsonl"
        path.write_text(template.read_text(encoding="utf-8").replace(_THREAD_ID, thread_id), encoding="utf-8")
    template.unlink()

    sources = (
        WatchSource(name="codex", root=codex_root),
        WatchSource(name="codex-state", root=codex_state_root),
    )
    async with live_owner_set(archive_root) as owners:
        watcher = live_watcher.LiveWatcher(
            archive,
            sources,
            cursor=CursorStore(archive_root / "ops.db"),
            **owners.watcher_kwargs(),
        )
        try:
            context = DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=sources)
            # Four rows per page over eighteen files: the deadlock only showed
            # after the third page.
            adapters = tuple(FileIntakeAdapter(context, source) for source in sources)
            dispatcher = FairIntakeDispatcher(
                tuple(
                    IntakeClassSpec(name=source.name, adapter=adapter, page_size=4)
                    for source, adapter in zip(sources, adapters, strict=True)
                )
            )
            pages = 0
            for _ in range(12):
                result = await dispatcher.run_once()
                pages += 1
                if result.quiescent and not any(adapter.discovery_pending for adapter in adapters):
                    break
            assert pages >= 5

            await _inspect_accepted_frontier(archive_root, owners)
            assert (reason := raw_frontier_source_selection_block_reason(archive_root)) is None, reason
            assert _cursor_authority_gap_states(archive_root) == []
            assert watcher._batch_processor.cursor_authority_block_reason() is None
            assert await archive.count_sessions() == rollout_count
        finally:
            watcher.stop()
            await archive.close()


def test_codex_state_source_scope_is_lexical_not_process_dependent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install scope depends only on the retained path text, not on the resolving process.

    Anti-vacuity: restoring ``Path(source_path).expanduser().resolve(strict=False)``
    makes the relative path resolve against the process CWD (so the two CWDs
    disagree) and makes the symlinked prefix collapse to its real directory, so
    the daemon and the CLI compute different scopes for one durable
    ``raw_sessions.source_path`` and the state-to-rollout join splits.
    """
    from polylogue.sources.codex_state_projection import codex_state_source_scope

    relative = "codex-install/sessions/2026/01/rollout.jsonl"
    monkeypatch.chdir(tmp_path)
    from_here = codex_state_source_scope(relative)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert codex_state_source_scope(relative) == from_here
    assert from_here == "codex-install"

    real = tmp_path / "real-install"
    (real / "sessions").mkdir(parents=True)
    link = tmp_path / "linked-install"
    link.symlink_to(real, target_is_directory=True)
    assert codex_state_source_scope(str(link / "sessions" / "rollout.jsonl")) == str(link)


@pytest.mark.asyncio
@pytest.mark.parametrize("state_first", [True, False], ids=["state-first", "rollout-first"])
async def test_codex_state_title_does_not_depend_on_admission_order(tmp_path: Path, state_first: bool) -> None:
    """A thread-state export admitted after its rollout still titles it.

    A rollout written before ``state_5.sqlite`` is enriched without the
    projected thread title. Its evidence binding then differs from the
    evidence the archive holds once the state export is projected, so the
    canonical raw-observation convergence re-derives it on the retained route
    and both orders store the same row.

    Anti-vacuity: make ``RawObservationDerivation._enrichment_evidence_moved``
    return ``False`` and the rollout-first order keeps the first-prompt title.
    """
    from tests.infra.raw_owner_routes import converge_pending_raws_async

    install = tmp_path / "codex-home"
    sessions = install / "sessions"
    sessions.mkdir(parents=True)
    rollout_path = sessions / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
    _write_codex_rollout(rollout_path)
    state_path = install / "state_5.sqlite"
    _write_state_5_sqlite(state_path)

    archive_root = tmp_path / "archive"
    run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=sessions),
            WatchSource(name="codex-state", root=install, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        cursor=CursorStore(archive_root / "index.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        for path in (state_path, rollout_path) if state_first else (rollout_path, state_path):
            metrics = await ingest_files_with_owners(processor, [path], emit_event=False)
            assert metrics.failed_file_count == 0
    finally:
        await archive.close()

    async with prepared_live_convergence_owner(archive_root) as owner:
        for _attempt in range(3):
            await converge_pending_raws_async(owner, archive_root, limit=64)

    with sqlite3.connect(archive_root / "index.db") as conn:
        rows = conn.execute("SELECT session_id, title FROM sessions").fetchall()
        bound = conn.execute("SELECT session_id FROM session_enrichment_bindings").fetchall()
    assert rows == [(_CODEX_SESSION_ID, "Synthetic curated title")]
    assert bound == [(_CODEX_SESSION_ID,)]

    # Anti-vacuity for inspection reading thread state: resolving it through
    # the source-tier connection (index only attached as ``index_tier``)
    # yields no state title, so the curated binding never matches and the
    # rollout is re-derived on every sweep.
    from polylogue.operations.raw_observation_derivation import (
        make_raw_observation_derivation,
        raw_observation_frame,
    )

    with sqlite3.connect(archive_root / "source.db") as conn:
        rollout_raws = [
            str(row[0])
            for row in conn.execute("SELECT raw_id FROM raw_sessions WHERE source_path LIKE '%rollout-%.jsonl'")
        ]
    assert rollout_raws

    def inspect(compute_adapter: BoundedComputeAdapter) -> Mapping[str, str]:
        return make_raw_observation_derivation(archive_root, compute_adapter=compute_adapter).inspect(
            raw_observation_frame(archive_root), rollout_raws
        )

    async with prepared_live_convergence_owner(archive_root) as owner:
        statuses = await owner.run_convergence_sync("test.codex-state.inspect", inspect, owner._compute_adapter)
    assert set(statuses.values()) == {"valid"}, statuses


def test_enrichment_evidence_moved_reads_titles_through_a_real_index_connection(tmp_path: Path) -> None:
    """``session_enrichment_evidence_key`` must resolve titles against index.db.

    Anti-vacuity (Codex P1, #5643): ``_enrichment_evidence_moved`` passed the
    source-tier connection (index.db only attached under the ``index_tier``
    alias) as ``index_conn``. ``read_thread_titles`` queries unqualified
    ``work_evidence_*`` tables, which resolve against that connection's own
    ``main`` schema -- source.db, which has no such tables -- so the query
    raised ``sqlite3.OperationalError``, caught and degraded to an empty
    mapping every time. The recomputed evidence key then always looked like
    "no title evidence", identical to the pre-state key, so a transcript
    already carrying a curated title from state evidence could never be
    detected as stale when that evidence later changed.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.revision_backfill import session_enrichment_evidence_key
    from polylogue.storage.sqlite.agent_thread_state import ThreadRecord, write_thread_state_graph
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL

    index_path = tmp_path / "index.db"
    with sqlite3.connect(index_path) as conn:
        conn.executescript(INDEX_DDL)
        assert write_thread_state_graph(
            conn,
            source_scope="/codex-home",
            threads=[ThreadRecord(_THREAD_ID, "Curated title", 2_000)],
            spawn_edges=[],
            raw_id="raw-state-1",
            blob_hash="blob-state-1",
            observed_at_ms=1_000,
            export_order=lambda _raw_id: None,
        )
        conn.commit()

    source_path = "/codex-home/sessions/rollout-2026-07-20T10-00-00-" + _THREAD_ID + ".jsonl"
    key = functools.partial(
        session_enrichment_evidence_key,
        provider=Provider.CODEX,
        source_path=source_path,
        native_id=_THREAD_ID,
        source_conn=None,
        blob_root=None,
    )

    # No index evidence at all -- the "nothing has a title" baseline.
    no_evidence = key(index_conn=None)

    # The exact defect: passing a connection whose *own* main schema is not
    # index.db (source.db here, standing in for the source-tier connection
    # with the index only attached as an alias) must not silently resolve to
    # the same "no title" identity as having no index evidence at all --
    # that identity match is what made a curated-title session's binding
    # look perpetually current even after the title moved.
    with sqlite3.connect(":memory:") as wrong_conn:
        degraded = key(index_conn=wrong_conn)
    assert degraded == no_evidence

    with sqlite3.connect(index_path) as real_conn:
        current = key(index_conn=real_conn)
    assert current is not None
    assert current != no_evidence
    assert current != degraded


def _write_spawn_state(path: Path, *, parent: str) -> None:
    """A synthetic ``state_5.sqlite`` naming ``parent`` as the rollout thread's spawner."""
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY, title TEXT, cwd TEXT, created_at_ms INTEGER, updated_at_ms INTEGER,
                source TEXT, model TEXT, agent_nickname TEXT, agent_role TEXT, archived INTEGER
            );
            CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT, status TEXT);
            """
        )
        conn.execute(
            "INSERT INTO thread_spawn_edges (parent_thread_id, child_thread_id, status) VALUES (?, ?, 'closed')",
            (parent, _THREAD_ID),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("state_first", [True, False])
async def test_a_rollout_takes_its_spawn_parent_from_its_own_install_root(
    workspace_env: dict[str, Path], state_first: bool
) -> None:
    """Two installs' state name different parents for one thread; the rollout's root decides.

    The rollout was acquired under root A, so its session's authoritative
    parent is root A's whether the state exports are ingested before the
    rollout or after it, and root B's export, observed later, never moves it.

    Anti-vacuity: read the projected parent without the rollout's install
    root and root B's later export wins: ``b-parent`` becomes authoritative.
    """
    from polylogue.archive.topology.edge import HOOK_AUTHORITATIVE_LINK_METHOD

    roots = {name: workspace_env["data_root"] / f"codex-{name}" for name in ("a", "b")}
    sources: list[WatchSource] = []
    for root in roots.values():
        (root / "sessions").mkdir(parents=True)
        sources.append(WatchSource(name="codex", root=root / "sessions"))
        sources.append(WatchSource(name="codex-state", root=root, layout=export_drop_layout((".sqlite", ".db"))))
    db_path = workspace_env["archive_root"] / "index.db"
    run_off_event_loop(lambda: bootstrap_archive_root(workspace_env["archive_root"]))
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    processor = LiveBatchProcessor(
        archive,
        tuple(sources),
        cursor=CursorStore(workspace_env["archive_root"] / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    rollout_path = roots["a"] / "sessions" / f"rollout-2026-07-20T10-00-00-{_THREAD_ID}.jsonl"
    _write_codex_rollout(rollout_path)

    async def ingest_states() -> None:
        for name in ("a", "b"):
            state_path = roots[name] / "state_5.sqlite"
            _write_spawn_state(state_path, parent=f"{name}-parent")
            metrics = await ingest_files_with_owners(processor, [state_path], emit_event=False)
            assert metrics.failed_file_count == 0

    try:
        if state_first:
            await ingest_states()
        metrics = await ingest_files_with_owners(processor, [rollout_path], emit_event=False)
        assert metrics.ingested_session_count == 1
        if not state_first:
            await ingest_states()
    finally:
        await archive.close()

    with sqlite3.connect(workspace_env["archive_root"] / "index.db") as index_conn:
        links = dict(
            index_conn.execute(
                "SELECT dst_native_id, method FROM session_links WHERE src_session_id = ?", (_CODEX_SESSION_ID,)
            ).fetchall()
        )
    assert links == {"a-parent": HOOK_AUTHORITATIVE_LINK_METHOD}


@pytest.mark.asyncio
async def test_live_state_publishes_every_captured_page_after_source_changes(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication uses the complete sealed capture, even after the live source changes."""
    archive, codex_root, state_root = _make_processor(workspace_env, "paged-state", "paged-state.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(name="codex", root=codex_root),
            WatchSource(name="codex-state", root=state_root, layout=export_drop_layout((".sqlite", ".db"))),
        ),
        cursor=CursorStore(workspace_env["archive_root"] / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    source = state_root / "state_5.sqlite"
    _write_state_5_sqlite(source)
    expected = {f"synthetic-thread-{i:04d}": f"Captured title {i}" for i in range(1001)}
    with sqlite3.connect(source) as conn:
        conn.executemany(
            "INSERT INTO threads VALUES (?, ?, '/repo', 1000, 2000, 'cli', 'synthetic', NULL, NULL, 0)",
            expected.items(),
        )
        conn.commit()
    expected[_THREAD_ID] = "Synthetic curated title"
    original_publish = processor._ingest_full_paths_prepared

    async def change_live_source_then_publish(*args: object, **kwargs: object) -> object:
        with sqlite3.connect(source) as conn:
            conn.execute("UPDATE threads SET title = 'Later live title'")
            conn.commit()
        return await original_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(processor, "_ingest_full_paths_prepared", change_live_source_then_publish)
    try:
        result = await ingest_files_with_owners(processor, [source], emit_event=False)
        assert result.failed_file_count == 0
        with sqlite3.connect(workspace_env["archive_root"] / "index.db") as conn:
            assert read_thread_titles(conn) == expected
            assert read_spawn_edges(conn) == {(_THREAD_ID, _CHILD_THREAD_ID): "closed"}
        with sqlite3.connect(source) as conn:
            assert conn.execute("SELECT DISTINCT title FROM threads").fetchall() == [("Later live title",)]
    finally:
        await archive.close()
