from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

import polylogue.daemon.convergence_stages as stages
from polylogue.daemon.convergence_stages import (
    make_default_convergence_stages,
    make_embed_stage,
    make_fts_stage,
    make_insights_stage,
)
from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
from polylogue.storage.insights.session.runtime import SessionInsightCounts
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.frozen_clock import FrozenClock
from tests.infra.storage_records import make_message, make_session, store_records


def _seed_raw_source_session(conn: sqlite3.Connection, *, session_id: str, source_path: Path) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id,
            source_name,
            source_path,
            blob_size,
            acquired_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            f"raw-{session_id}",
            "codex",
            str(source_path),
            source_path.stat().st_size,
            "2026-05-24T01:00:00+00:00",
        ),
    )
    store_records(
        session=make_session(
            session_id,
            source_name="codex",
            title=session_id,
            created_at="2026-05-24T01:00:00+00:00",
            updated_at="2026-05-24T01:00:00+00:00",
            raw_id=f"raw-{session_id}",
        ),
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


def _truncate(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def _seed_minimal_archive(db_path: Path, source_path: Path, *, session_id: str = "codex-session:s1") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}\n", encoding="utf-8")
    with sqlite3.connect(db_path.with_name("source.db")) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, source_path) VALUES (?, ?)",
            ("raw-s1", str(source_path)),
        )
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                raw_id TEXT,
                title TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0,
                tool_use_count INTEGER NOT NULL DEFAULT 0,
                paste_count INTEGER NOT NULL DEFAULT 0,
                sort_key_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                variant_index INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE blocks (
                block_id TEXT,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                text TEXT
            );
            CREATE VIRTUAL TABLE blocks_fts USING fts5(
                block_id UNINDEXED,
                message_id UNINDEXED,
                session_id UNINDEXED,
                block_type UNINDEXED,
                text,
                content='blocks',
                content_rowid='rowid'
            );
            CREATE TABLE insight_materialization (
                insight_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                materializer_version INTEGER NOT NULL,
                materialized_at_ms INTEGER NOT NULL,
                source_updated_at_ms INTEGER,
                source_sort_key_ms INTEGER,
                input_high_water_mark_ms INTEGER,
                input_row_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(insight_type, session_id)
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                substantive_count INTEGER NOT NULL DEFAULT 0,
                attachment_count INTEGER NOT NULL DEFAULT 0,
                work_event_count INTEGER NOT NULL DEFAULT 0,
                phase_count INTEGER NOT NULL DEFAULT 0,
                search_text TEXT NOT NULL DEFAULT '',
                provenance_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sessions(
                session_id, raw_id, title, message_count, user_message_count,
                assistant_message_count, tool_use_count, paste_count, sort_key_ms, updated_at_ms
            ) VALUES (?, 'raw-s1', 'Native session', 1, 1, 0, 0, 0, 1770000000000, 1770000000000)
            """,
            (session_id,),
        )
        conn.execute(
            "INSERT INTO messages(message_id, session_id, position) VALUES (?, ?, 0)",
            (f"{session_id}:m1", session_id),
        )
        conn.execute(
            """
            INSERT INTO blocks(block_id, message_id, session_id, position, block_type, text)
            VALUES (?, ?, ?, 0, 'text', 'archive searchable block')
            """,
            (f"{session_id}:m1:0", f"{session_id}:m1", session_id),
        )
        conn.commit()


def test_fts_stage_repairs_archive_when_db_anchor_exists(tmp_path: Path) -> None:
    archive_db = tmp_path / "index.db"
    (tmp_path / "index.db").touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    stage = make_fts_stage(tmp_path / "index.db")

    assert stage.check(source_path) is True
    assert stage.execute(source_path) is True
    with sqlite3.connect(archive_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM blocks_fts_docsize").fetchone()[0] == 1


def test_insights_stage_materializes_archive_profiles_from_archive_tiers(tmp_path: Path) -> None:
    archive_db = tmp_path / "index.db"
    source_path = tmp_path / "codex.jsonl"
    session_id = "codex-session:s1"
    _seed_minimal_archive(archive_db, source_path, session_id=session_id)

    stage = make_insights_stage(tmp_path / "index.db")

    assert stage.check_sessions is not None
    assert stage.execute_sessions is not None
    assert stage.check_sessions([session_id]) == {session_id}
    assert stage.execute_sessions([session_id]) is True
    assert stage.check_sessions([session_id]) == set()
    with sqlite3.connect(archive_db) as conn:
        profile = conn.execute("SELECT session_id, substantive_count FROM session_profiles").fetchone()
        materialization = conn.execute(
            """
            SELECT insight_type, session_id, materializer_version, source_sort_key_ms
            FROM insight_materialization
            """
        ).fetchone()
    assert profile == (session_id, 1)
    assert materialization == ("session_profile", session_id, SESSION_INSIGHT_MATERIALIZER_VERSION, 1770000000000)


def test_insights_stage_rebuilds_sync_against_configured_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    opened_paths: list[Path] = []
    rebuilt = False

    class FakeConnection:
        def execute(self, sql: str, params: tuple[str, ...] = ()) -> object:
            if "sqlite_master" in sql:
                return _FakeCursor([(1,)])
            if "raw_sessions" in sql:
                return _FakeCursor([(params[0], "conv-1")])
            if "session_profiles" in sql:
                return _FakeCursor([])
            raise AssertionError(f"unexpected SQL: {sql} {params}")

        def commit(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeCursor:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def fetchone(self) -> object | None:
            return self._rows[0] if self._rows else None

        def fetchall(self) -> list[object]:
            return self._rows

    @contextmanager
    def fake_open_connection(path: Path) -> Iterator[FakeConnection]:
        opened_paths.append(path)
        yield FakeConnection()

    def fake_rebuild(
        conn: FakeConnection,
        *,
        session_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        nonlocal rebuilt
        del conn
        rebuilt = True
        assert session_ids == ["conv-1"]
        assert page_size == 10
        return SessionInsightCounts(
            profiles=1,
            work_events=2,
            phases=3,
            threads=4,
            tag_rollups=5,
        )

    def fail_if_used(coro: object) -> object:
        raise AssertionError("insights stage should not open an asyncio runner")

    monkeypatch.setattr(asyncio, "run", fail_if_used)
    monkeypatch.setattr("polylogue.daemon.convergence_stages._active_archive_index_path", lambda _db_path: None)
    monkeypatch.setattr(
        "polylogue.daemon.convergence_stages._session_ids_for_source_path", lambda _conn, _path: ["conv-1"]
    )
    monkeypatch.setattr("polylogue.daemon.convergence_stages._hot_insight_session_ids", lambda _conn, _ids: set())
    monkeypatch.setattr("polylogue.storage.sqlite.connection.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)

    assert make_insights_stage(db_path).execute(tmp_path / "source.jsonl") is True
    assert opened_paths == [db_path]
    assert rebuilt is True


def test_fts_stage_repairs_only_missing_action_index_when_messages_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    repaired_messages: list[list[str]] = []
    inserted_actions: list[list[str]] = []
    rebuilt = False
    committed = False

    class FakeConnection:
        def close(self) -> None:
            pass

        def commit(self) -> None:
            nonlocal committed
            committed = True

    def fake_open_connection(path: Path, *, timeout: float) -> FakeConnection:
        assert path == db_path
        assert timeout == 30.0
        return FakeConnection()

    def fake_repair_messages(conn: FakeConnection, session_ids: list[str]) -> None:
        repaired_messages.append(session_ids)

    def fake_insert_missing_actions(conn: FakeConnection, session_ids: list[str]) -> None:
        inserted_actions.append(session_ids)

    def fake_rebuild(conn: FakeConnection) -> None:
        nonlocal rebuilt
        rebuilt = True

    needs_calls: list[list[str]] = []
    marked_ready: list[FakeConnection] = []

    def fake_repair_needs(_conn: FakeConnection, session_ids: list[str]) -> stages._FtsRepairNeeds:
        needs_calls.append(session_ids)
        return stages._FtsRepairNeeds(actions=len(needs_calls) == 1)

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync", fake_repair_messages)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.insert_missing_action_fts_index_sync",
        fake_insert_missing_actions,
    )
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", fake_rebuild)
    monkeypatch.setattr(
        stages,
        "_session_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )
    monkeypatch.setattr(stages, "_fts_repair_needs_for_sessions", fake_repair_needs)
    monkeypatch.setattr(stages, "_action_events_exist_for_sessions", lambda _conn, _ids: False)
    monkeypatch.setattr(stages, "_mark_message_fts_ready_after_targeted_repair", lambda conn: marked_ready.append(conn))

    stage = make_fts_stage(db_path)
    assert stage.execute_many is not None
    assert stage.execute_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is True
    assert repaired_messages == []
    assert inserted_actions == [["conv-a", "conv-b"]]
    assert needs_calls == [["conv-a", "conv-b"], ["conv-a", "conv-b"]]
    assert len(marked_ready) == 1
    assert committed is True
    assert rebuilt is False


def test_action_event_probe_uses_indexed_base_table() -> None:
    queries: list[tuple[str, tuple[str, ...]]] = []

    class FakeConnection:
        def execute(self, sql: str, params: tuple[str, ...] = ()) -> object:
            queries.append((sql, params))
            if "sqlite_master" in sql:
                return _FakeCursor([("action_events",)])
            if "action_events" in sql:
                return _FakeCursor([(1,)])
            raise AssertionError(f"unexpected SQL: {sql}")

    class _FakeCursor:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchone(self) -> tuple[object, ...] | None:
            return self._rows[0] if self._rows else None

    conn = cast(sqlite3.Connection, FakeConnection())
    assert stages._action_events_exist_for_sessions(conn, ["conv-a", "conv-b"]) is True
    probe_sql = queries[-1][0]
    assert "FROM action_events\n" in probe_sql
    assert "action_events_fts" not in probe_sql
    assert queries[-1][1] == ("conv-a", "conv-b")


def test_fts_repair_needs_probe_uses_docsize_shadow_tables() -> None:
    queries: list[tuple[str, tuple[str, ...]]] = []
    existing_tables = {"messages_fts_docsize", "action_events", "action_events_fts", "action_events_fts_docsize"}

    class FakeConnection:
        def execute(self, sql: str, params: tuple[str, ...] = ()) -> object:
            queries.append((sql, params))
            if "sqlite_master" in sql:
                return _FakeCursor([(1,)] if params and params[0] in existing_tables else [])
            if "messages AS m" in sql:
                return _FakeCursor([(0,)])
            if "action_events AS ae" in sql:
                return _FakeCursor([(0,)])
            raise AssertionError(f"unexpected SQL: {sql}")

    class _FakeCursor:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchone(self) -> tuple[object, ...] | None:
            return self._rows[0] if self._rows else None

    conn = cast(sqlite3.Connection, FakeConnection())

    assert stages._fts_repair_needs_for_sessions(conn, ["conv-a"]) == stages._FtsRepairNeeds()

    probe_sql = "\n".join(sql for sql, _params in queries)
    assert "LEFT JOIN messages_fts_docsize" in probe_sql
    assert "LEFT JOIN action_events_fts_docsize" in probe_sql
    assert "LEFT JOIN messages_fts AS" not in probe_sql
    assert "LEFT JOIN action_events_fts AS" not in probe_sql


def test_fts_repair_needs_ignores_empty_text_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    with open_connection(db_path) as conn:
        restore_fts_triggers_sync(conn)
        store_records(
            session=make_session("conv-empty-text", source_name="codex"),
            messages=[make_message("msg-empty-text", "conv-empty-text", text="")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

        assert stages._fts_repair_needs_for_sessions(conn, ["conv-empty-text"]) == stages._FtsRepairNeeds()


def test_targeted_fts_ready_marker_preserves_ledger_counts(tmp_path: Path) -> None:
    from polylogue.storage.fts.freshness import record_fts_surface_state_sync

    db_path = tmp_path / "archive.sqlite"
    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions(session_id, source_name, provider_session_id, version) VALUES(?,?,?,1)",
            ("conv-ledger", "codex", "provider-conv"),
        )
        conn.execute(
            "INSERT INTO messages(message_id, session_id, role, text, source_name, version) VALUES(?,?,?,?,?,1)",
            ("msg-ledger", "conv-ledger", "user", "indexed text", "codex"),
        )
        record_fts_surface_state_sync(
            conn,
            surface="messages_fts",
            state="stale",
            source_rows=100,
            indexed_rows=99,
            missing_rows=1,
            detail="pre-existing exact snapshot",
        )
        stages._mark_message_fts_ready_after_targeted_repair(conn)
        row = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows, detail
            FROM fts_freshness_state
            WHERE surface='messages_fts'
            """,
        ).fetchone()

    assert tuple(row) == (
        "ready",
        100,
        99,
        0,
        0,
        0,
        "targeted changed-session repair complete",
    )


def test_targeted_fts_ready_marker_handles_legacy_freshness_table(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions(session_id, source_name, provider_session_id, version) VALUES(?,?,?,1)",
            ("conv-legacy-ledger", "codex", "provider-conv"),
        )
        conn.execute(
            "INSERT INTO messages(message_id, session_id, role, text, source_name, version) VALUES(?,?,?,?,?,1)",
            ("msg-legacy-ledger", "conv-legacy-ledger", "user", "indexed text", "codex"),
        )
        conn.execute("DROP TABLE IF EXISTS fts_freshness_state")
        conn.execute(
            """
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL
            )
            """
        )
        conn.execute("INSERT INTO fts_freshness_state VALUES ('messages_fts', 'stale', '2026-05-24T00:00:00+00:00')")

        stages._mark_message_fts_ready_after_targeted_repair(conn)
        row = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows, detail
            FROM fts_freshness_state
            WHERE surface='messages_fts'
            """,
        ).fetchone()

    assert tuple(row) == (
        "ready",
        0,
        0,
        0,
        0,
        0,
        "targeted changed-session repair complete",
    )


def test_default_convergence_stages_always_register_embed_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.delenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", raising=False)

    stage_names = [stage.name for stage in make_default_convergence_stages(tmp_path / "archive.sqlite")]

    assert stage_names == ["fts", "embed", "insights"]


def test_embed_stage_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.delenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", raising=False)
    db_path = tmp_path / "archive.sqlite"
    db_path.touch()

    stage = make_embed_stage(db_path)

    assert stage.check(tmp_path / "source.jsonl") is False
    assert stage.execute(tmp_path / "source.jsonl") is True


def test_insights_stage_batches_sync_rebuild_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    rebuild_calls: list[tuple[list[str], int]] = []

    class FakeConnection:
        def commit(self) -> None:
            pass

        def close(self) -> None:
            pass

    @contextmanager
    def fake_open_connection(path: Path) -> Iterator[FakeConnection]:
        assert path == db_path
        yield FakeConnection()

    def fake_rebuild(
        conn: FakeConnection,
        *,
        session_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn
        rebuild_calls.append((session_ids, page_size))
        return SessionInsightCounts(profiles=2, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.sqlite.connection.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(
        stages,
        "_session_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )
    monkeypatch.setattr(stages, "_hot_insight_session_ids", lambda _conn, _ids: set())

    stage = make_insights_stage(db_path)
    assert stage.execute_many is not None
    assert stage.execute_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is True
    assert rebuild_calls == [(["conv-a", "conv-b"], 10)]


def test_insights_stage_defers_hot_large_session_debt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    source_path = tmp_path / "active-codex.jsonl"
    _truncate(source_path, stages._HOT_INSIGHT_SOURCE_BYTES + 1)
    with open_connection(db_path) as conn:
        _seed_raw_source_session(conn, session_id="conv-hot", source_path=source_path)
        conn.commit()

    def fail_rebuild(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("hot active sources should wait for convergence debt retry")

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fail_rebuild)

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(["conv-hot"]) is False


def test_session_ids_missing_profiles_includes_stale(tmp_path: Path) -> None:
    """#1620: the path-fallback debt loop must surface stale profiles, not just missing ones.

    Before the fix, sessions whose JSONL had gone quiet but whose
    ``sessions.sort_key`` drifted from the materialized
    ``source_sort_key`` were never picked up by the daemon's debt loop —
    ``remaining=0`` was reported indefinitely.
    """
    db_path = tmp_path / "missing_profiles.sqlite"
    cutoff_safe_sort_key = 1.0  # well below now - HOT_SOURCE_GRACE_SECONDS
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER,
                source_sort_key REAL,
                source_updated_at TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (session_id, sort_key) VALUES ('conv-missing', ?)",
            (cutoff_safe_sort_key,),
        )
        conn.execute(
            "INSERT INTO sessions (session_id, sort_key) VALUES ('conv-stale', ?)",
            (cutoff_safe_sort_key,),
        )
        conn.execute(
            "INSERT INTO sessions (session_id, sort_key) VALUES ('conv-fresh', ?)",
            (cutoff_safe_sort_key,),
        )
        conn.execute(
            """
            INSERT INTO session_profiles (session_id, materializer_version, source_sort_key)
            VALUES ('conv-stale', ?, ?)
            """,
            (SESSION_INSIGHT_MATERIALIZER_VERSION, cutoff_safe_sort_key + 1000.0),
        )
        conn.execute(
            """
            INSERT INTO session_profiles (session_id, materializer_version, source_sort_key)
            VALUES ('conv-fresh', ?, ?)
            """,
            (SESSION_INSIGHT_MATERIALIZER_VERSION, cutoff_safe_sort_key),
        )
        conn.commit()

        ids = stages._session_ids_missing_profiles(conn)

    assert set(ids) == {"conv-missing", "conv-stale"}


def test_insights_staleness_uses_sort_key_not_timestamp_text(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER,
                source_sort_key REAL,
                source_updated_at TEXT
            );
            INSERT INTO sessions (session_id, sort_key, updated_at)
            VALUES ('conv-current', 1779606000.0, '2026-05-24T07:00:00+00:00');
            """
        )
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id,
                materializer_version,
                source_sort_key,
                source_updated_at
            ) VALUES (
                'conv-current',
                ?,
                1779606000.0,
                '2026-05-24T07:00:00Z'
            );
            """,
            (SESSION_INSIGHT_MATERIALIZER_VERSION,),
        )

        stale = stages._stale_session_profile_ids(conn, ["conv-current"])

    assert stale == []


def test_insights_session_rebuild_returns_false_when_still_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    source_path = tmp_path / "quiet-codex.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    with open_connection(db_path) as conn:
        _seed_raw_source_session(conn, session_id="conv-stale", source_path=source_path)
        conn.commit()

    def no_op_rebuild(
        conn: sqlite3.Connection,
        *,
        session_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn, session_ids, page_size
        return SessionInsightCounts(profiles=0, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", no_op_rebuild)

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(["conv-stale"]) is False


def test_insights_stage_rebuilds_large_session_after_quiet_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    source_path = tmp_path / "quiet-codex.jsonl"
    _truncate(source_path, stages._HOT_INSIGHT_SOURCE_BYTES + 1)
    quiet_mtime = frozen_clock.now().timestamp() - stages._HOT_INSIGHT_QUIET_SECONDS - 5
    os.utime(source_path, (quiet_mtime, quiet_mtime))
    rebuild_calls: list[tuple[list[str], int]] = []
    with open_connection(db_path) as conn:
        _seed_raw_source_session(conn, session_id="conv-quiet", source_path=source_path)
        conn.commit()

    def fake_rebuild(
        conn: sqlite3.Connection,
        *,
        session_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn
        rebuild_calls.append((session_ids, page_size))
        return SessionInsightCounts(profiles=1, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(stages, "_stale_session_profile_ids", lambda _conn, _ids: [])

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(["conv-quiet"]) is True
    assert rebuild_calls == [(["conv-quiet"], 10)]


def test_insights_stage_rebuilds_small_active_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    source_path = tmp_path / "small-active-codex.jsonl"
    _truncate(source_path, 1024)
    rebuild_calls: list[tuple[list[str], int]] = []
    with open_connection(db_path) as conn:
        _seed_raw_source_session(conn, session_id="conv-small", source_path=source_path)
        conn.commit()

    def fake_rebuild(
        conn: sqlite3.Connection,
        *,
        session_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn
        rebuild_calls.append((session_ids, page_size))
        return SessionInsightCounts(profiles=1, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(stages, "_stale_session_profile_ids", lambda _conn, _ids: [])

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(["conv-small"]) is True
    assert rebuild_calls == [(["conv-small"], 10)]


def test_insights_stage_scopes_session_debt_to_stale_profiles(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    sessions = {
        "conv-fresh": "2026-05-24T01:00:00+00:00",
        "conv-missing-profile": "2026-05-24T01:01:00+00:00",
        "conv-stale-source": "2026-05-24T01:02:00+00:00",
        "conv-stale-version": "2026-05-24T01:03:00+00:00",
    }
    with open_connection(db_path) as conn:
        for session_id, updated_at in sessions.items():
            store_records(
                session=make_session(
                    session_id,
                    source_name="codex",
                    title=session_id,
                    created_at=updated_at,
                    updated_at=updated_at,
                ),
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
        rebuild_session_insights_sync(
            conn,
            session_ids=list(sessions),
            page_size=10,
        )
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", ("conv-missing-profile",))
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            ("2026-05-24T02:02:00+00:00", "conv-stale-source"),
        )
        conn.execute(
            "UPDATE session_profiles SET materializer_version = ? WHERE session_id = ?",
            (SESSION_INSIGHT_MATERIALIZER_VERSION - 1, "conv-stale-version"),
        )
        conn.commit()

    stage = make_insights_stage(db_path)
    assert stage.check_sessions is not None
    assert stage.check_sessions(
        [
            "conv-fresh",
            "conv-missing-profile",
            "conv-stale-source",
            "conv-stale-version",
            "conv-unknown",
        ]
    ) == {
        "conv-missing-profile",
        "conv-stale-source",
        "conv-stale-version",
    }


def test_embedding_config_enabled_with_key() -> None:
    """Embedding is enabled when config has both enabled flag and API key."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = "test-key"
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is True


def test_embedding_config_disabled_without_key() -> None:
    """Embedding is disabled when config has enabled flag but no API key."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = None
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is False


def test_embedding_config_disabled_explicitly() -> None:
    """Embedding is disabled when config has key but enabled flag is False."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = False
        mock_cfg.return_value.voyage_api_key = "test-key"
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is False
