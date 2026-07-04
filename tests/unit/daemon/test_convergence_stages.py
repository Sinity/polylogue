from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import polylogue.daemon.convergence_stages as stages
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.convergence import StageExecutionResult
from polylogue.daemon.convergence_stages import (
    make_default_convergence_stages,
    make_embed_stage,
    make_fts_stage,
    make_insights_stage,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.insights.session import storage as session_storage
from polylogue.storage.insights.session.runtime import SessionInsightCounts
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.frozen_clock import FrozenClock


class _SessionIdOnly:
    def __init__(self, session_id: str, marker: str) -> None:
        self.session_id = session_id
        self.marker = marker


def test_session_storage_dedupes_records_by_session_id() -> None:
    records = [
        _SessionIdOnly("codex-session:one", "first"),
        _SessionIdOnly("codex-session:one", "second"),
        _SessionIdOnly("codex-session:two", "third"),
    ]

    deduped = session_storage._dedupe_records_by_session(records)

    assert [(record.session_id, record.marker) for record in deduped] == [
        ("codex-session:one", "second"),
        ("codex-session:two", "third"),
    ]


def _main_db_path(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(str(row[2]))


def _seed_raw_source_session(conn: sqlite3.Connection, *, session_id: str, source_path: Path) -> str:
    index_path = _main_db_path(conn)
    if not source_path.exists():
        source_path.write_bytes(b"{}\n")
    raw_id = f"raw-{session_id}"
    source_db = index_path.with_name("source.db")
    with sqlite3.connect(source_db) as source_conn:
        initialize_archive_tier(source_conn, ArchiveTier.SOURCE)
        source_conn.execute(
            """
            INSERT OR REPLACE INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "codex-session",
                session_id,
                str(source_path),
                hashlib.sha256(f"raw:{session_id}".encode()).digest(),
                source_path.stat().st_size,
                1_769_000_000_000,
            ),
        )
        source_conn.commit()
    stored_session_id = write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=session_id,
            title=session_id,
            created_at="2026-05-24T01:00:00+00:00",
            updated_at="2026-05-24T01:00:00+00:00",
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.normalize("user"),
                    text=f"Message for {session_id}",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"Message for {session_id}")],
                )
            ],
        ),
        raw_id=raw_id,
        content_hash=hashlib.sha256(f"session:{session_id}".encode()).hexdigest(),
    )
    return stored_session_id


def _seed_index_session(conn: sqlite3.Connection, *, session_id: str, text: str) -> str:
    return write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=session_id,
            title=session_id,
            created_at="2026-05-24T01:00:00+00:00",
            updated_at="2026-05-24T01:00:00+00:00",
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.normalize("user"),
                    text=text,
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                )
            ],
        ),
        content_hash=hashlib.sha256(f"session:{session_id}".encode()).hexdigest(),
    )


def _seed_empty_text_index_session(conn: sqlite3.Connection, *, session_id: str) -> str:
    return write_parsed_session_to_archive(
        conn,
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=session_id,
            title=session_id,
            messages=[
                ParsedMessage(
                    provider_message_id="msg-1",
                    role=Role.normalize("user"),
                    text="",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="")],
                )
            ],
        ),
        content_hash=hashlib.sha256(f"session:{session_id}".encode()).hexdigest(),
    )


def _truncate(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def _seed_minimal_archive(db_path: Path, source_path: Path, *, session_id: str = "codex-session:s1") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}\n", encoding="utf-8")
    with sqlite3.connect(db_path.with_name("source.db")) as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-s1",
                "codex-session",
                session_id,
                str(source_path),
                hashlib.sha256(b"raw-s1").digest(),
                source_path.stat().st_size,
                1_770_000_000_000,
            ),
        )
        conn.commit()
    with sqlite3.connect(db_path) as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        native_id = session_id.split(":", 1)[-1]
        conn.execute(
            """
            INSERT INTO sessions(
                native_id, origin, raw_id, title, message_count, user_message_count,
                assistant_message_count, tool_use_count, paste_count, content_hash, updated_at_ms
            ) VALUES (?, 'codex-session', 'raw-s1', 'Native session', 1, 1, 0, 0, 0, ?, 1770000000000)
            """,
            (native_id, hashlib.sha256(f"session:{session_id}".encode()).digest()),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, native_id, position, role, message_type, content_hash)
            VALUES (?, 'm1', 0, 'user', 'message', ?)
            """,
            (session_id, hashlib.sha256(f"message:{session_id}".encode()).digest()),
        )
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text)
            VALUES (?, ?, 0, 'text', 'archive searchable block')
            """,
            (f"{session_id}:m1", session_id),
        )
        conn.execute("DELETE FROM messages_fts")
        conn.commit()


def test_fts_stage_skips_archive_source_path_backlog_repair(tmp_path: Path) -> None:
    archive_db = tmp_path / "index.db"
    (tmp_path / "index.db").touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    stage = make_fts_stage(tmp_path / "index.db")

    assert stage.check(source_path) is False
    assert stage.execute(source_path) is True
    with sqlite3.connect(archive_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0


def test_fts_session_debt_uses_targeted_repair_not_full_rebuild(tmp_path: Path) -> None:
    """Session-scoped FTS debt repairs only named sessions.

    Foreground source-path convergence deliberately does not repair historical
    FTS backlog: archive writes already repair newly changed session rows, and
    old debt is handled by bounded session/debt convergence. When a concrete
    session-id debt item is retried, it still takes the targeted delete+insert
    path rather than rebuilding the whole FTS surface.
    """
    from unittest import mock

    import polylogue.storage.fts.fts_lifecycle as fts_lc

    archive_db = tmp_path / "index.db"
    archive_db.touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    stage = make_fts_stage(tmp_path / "index.db")

    with (
        mock.patch.object(
            fts_lc, "repair_message_fts_index_sync", wraps=fts_lc.repair_message_fts_index_sync
        ) as targeted,
        mock.patch.object(fts_lc, "rebuild_fts_index_sync", wraps=fts_lc.rebuild_fts_index_sync) as full_rebuild,
    ):
        assert stage.execute_sessions is not None
        assert stage.execute_sessions(["codex-session:s1"]) is True

    targeted.assert_called_once()
    full_rebuild.assert_not_called()
    with sqlite3.connect(archive_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1


def test_archive_fts_session_repair_defers_sqlite_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_db = tmp_path / "index.db"
    archive_db.touch()

    def locked(_db_path: Path) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(stages, "_open_archive_insight_write_connection", locked)

    assert stages._archive_fts_execute_sessions(archive_db, ["codex-session:s1"]) is False


def test_archive_fts_global_repair_defers_sqlite_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_db = tmp_path / "index.db"
    archive_db.touch()

    def locked(_db_path: Path) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(stages, "_open_archive_insight_write_connection", locked)

    assert stages.repair_messages_fts_surface(archive_db) is False


def test_archive_fts_global_repair_inserts_missing_rows_without_reset(tmp_path: Path) -> None:
    """Global surface debt should converge with bounded missing-row repair."""
    from unittest import mock

    import polylogue.storage.fts.fts_lifecycle as fts_lc

    archive_db = tmp_path / "index.db"
    archive_db.touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    with mock.patch.object(
        fts_lc,
        "reset_message_fts_index_sync",
        wraps=fts_lc.reset_message_fts_index_sync,
    ) as reset_surface:
        assert stages.repair_messages_fts_surface(archive_db) is True

    reset_surface.assert_not_called()
    with sqlite3.connect(archive_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1


def test_archive_fts_global_repair_deletes_excess_rows_without_reset(tmp_path: Path) -> None:
    """Global surface debt should remove excess rows without a full rebuild."""
    from unittest import mock

    import polylogue.storage.fts.fts_lifecycle as fts_lc

    archive_db = tmp_path / "index.db"
    archive_db.touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    with sqlite3.connect(archive_db) as conn:
        fts_lc.rebuild_fts_index_sync(conn)
        row = conn.execute("SELECT rowid FROM blocks LIMIT 1").fetchone()
        assert row is not None
        conn.execute("DROP TRIGGER messages_fts_ad")
        conn.execute("DELETE FROM blocks WHERE rowid = ?", (row[0],))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 1

    with mock.patch.object(
        fts_lc,
        "reset_message_fts_index_sync",
        wraps=fts_lc.reset_message_fts_index_sync,
    ) as reset_surface:
        assert stages.repair_messages_fts_surface(archive_db) is True

    reset_surface.assert_not_called()
    with sqlite3.connect(archive_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 0


def test_archive_repair_sessions_fts_skips_unknown_scope(tmp_path: Path) -> None:
    """Path-scoped convergence must not become a whole-archive FTS rebuild."""
    from unittest import mock

    import polylogue.storage.fts.fts_lifecycle as fts_lc

    archive_db = tmp_path / "index.db"
    archive_db.touch()
    source_path = tmp_path / "codex.jsonl"
    _seed_minimal_archive(archive_db, source_path)

    with (
        sqlite3.connect(archive_db) as conn,
        mock.patch.object(
            fts_lc, "reset_message_fts_index_sync", wraps=fts_lc.reset_message_fts_index_sync
        ) as reset_surface,
    ):
        stages._archive_repair_sessions_fts(conn, [])

    reset_surface.assert_not_called()


def test_archive_insights_path_batch_does_not_fallback_to_global_missing_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_db = tmp_path / "index.db"
    archive_db.touch()
    source_path = tmp_path / "codex.jsonl"

    monkeypatch.setattr(stages, "_schema_archive_session_ids_for_source_paths", lambda _conn, _paths: {source_path: []})

    def fail_global_missing_profiles(_conn: sqlite3.Connection) -> list[str]:
        raise AssertionError("path-scoped insight convergence must not scan global missing profiles")

    monkeypatch.setattr(stages, "_schema_archive_session_ids_missing_profiles", fail_global_missing_profiles)

    assert stages._archive_insights_check_many(archive_db, [source_path]) == set()
    result = stages._archive_insights_execute_many(archive_db, [source_path])
    assert result is True


def test_insights_stage_materializes_archive_profiles_from_archive_tiers(tmp_path: Path) -> None:
    archive_db = tmp_path / "index.db"
    source_path = tmp_path / "codex.jsonl"
    session_id = "codex-session:s1"
    _seed_minimal_archive(archive_db, source_path, session_id=session_id)

    stage = make_insights_stage(tmp_path / "index.db")

    assert stage.check_sessions is not None
    assert stage.execute_sessions is not None
    assert stage.check_sessions([session_id]) == {session_id}
    result = stage.execute_sessions([session_id])
    assert result
    assert isinstance(result, StageExecutionResult)
    assert "insights.analysis.facts" in result.stage_timings_s
    assert "insights.facts.message_flags" in result.stage_timings_s
    assert "insights.facts.message_facts" in result.stage_timings_s
    assert "insights.facts.aggregate_messages" in result.stage_timings_s
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
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "insights",
    ) -> SessionInsightCounts:
        nonlocal rebuilt
        del conn, stage_timing_prefix
        rebuilt = True
        assert session_ids == ["conv-1"]
        assert page_size == 10
        if stage_timings_s is not None:
            stage_timings_s["insights.fake"] = 0.125
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

    def fake_rebuild(conn: FakeConnection) -> None:
        nonlocal rebuilt
        rebuilt = True

    needs_calls: list[list[str]] = []
    marked_ready: list[FakeConnection] = []

    def fake_repair_needs(_conn: FakeConnection, session_ids: list[str]) -> stages._FtsRepairNeeds:
        needs_calls.append(session_ids)
        return stages._FtsRepairNeeds(messages=len(needs_calls) == 1)

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync", fake_repair_messages)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", fake_rebuild)
    monkeypatch.setattr(
        stages,
        "_session_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )
    monkeypatch.setattr(stages, "_fts_repair_needs_for_sessions", fake_repair_needs)
    monkeypatch.setattr(stages, "_mark_message_fts_ready_after_targeted_repair", lambda conn: marked_ready.append(conn))

    stage = make_fts_stage(db_path)
    assert stage.execute_many is not None
    assert stage.execute_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is True
    assert repaired_messages == [["conv-a", "conv-b"]]
    assert needs_calls == [["conv-a", "conv-b"], ["conv-a", "conv-b"]]
    assert len(marked_ready) == 1
    assert committed is True
    assert rebuilt is False


def test_fts_repair_needs_probe_uses_docsize_shadow_tables() -> None:
    queries: list[tuple[str, tuple[str, ...]]] = []
    existing_tables = {"messages_fts_docsize"}

    class FakeConnection:
        def execute(self, sql: str, params: tuple[str, ...] = ()) -> object:
            queries.append((sql, params))
            if "sqlite_master" in sql:
                return _FakeCursor([(1,)] if params and params[0] in existing_tables else [])
            if "blocks AS b" in sql:
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
    assert "LEFT JOIN messages_fts AS" not in probe_sql


def test_fts_repair_needs_ignores_empty_text_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        session_id = _seed_empty_text_index_session(conn, session_id="conv-empty-text")
        conn.commit()

        assert stages._fts_repair_needs_for_sessions(conn, [session_id]) == stages._FtsRepairNeeds()


def test_targeted_fts_ready_marker_preserves_ledger_counts(tmp_path: Path) -> None:
    from polylogue.storage.fts.freshness import record_fts_surface_state_sync

    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        _seed_index_session(conn, session_id="conv-ledger", text="indexed text")
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
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        _seed_index_session(conn, session_id="conv-legacy-ledger", text="indexed text")
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
    db_path = tmp_path / "index.db"
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
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "insights",
    ) -> SessionInsightCounts:
        del conn, stage_timings_s, stage_timing_prefix
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
        session_id = _seed_raw_source_session(conn, session_id="conv-hot", source_path=source_path)
        conn.commit()

    def fail_rebuild(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("hot active sources should wait for convergence debt retry")

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fail_rebuild)

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions([session_id]) is False


def test_session_ids_missing_profiles_includes_stale(tmp_path: Path) -> None:
    """#1620: the path-fallback debt loop must surface stale profiles, not just missing ones.

    Before the fix, sessions whose JSONL had gone quiet but whose
    ``sessions.sort_key_ms`` drifted from the materialized
    ``source_sort_key`` were never picked up by the daemon's debt loop —
    ``remaining=0`` was reported indefinitely.
    """
    db_path = tmp_path / "missing_profiles.sqlite"
    cutoff_safe_sort_key = 1.0  # well below now - HOT_SOURCE_GRACE_SECONDS
    cutoff_safe_sort_key_ms = int(cutoff_safe_sort_key * 1000)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key_ms INTEGER,
                updated_at_ms INTEGER
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
            "INSERT INTO sessions (session_id, sort_key_ms) VALUES ('conv-missing', ?)",
            (cutoff_safe_sort_key_ms,),
        )
        conn.execute(
            "INSERT INTO sessions (session_id, sort_key_ms) VALUES ('conv-stale', ?)",
            (cutoff_safe_sort_key_ms,),
        )
        conn.execute(
            "INSERT INTO sessions (session_id, sort_key_ms) VALUES ('conv-fresh', ?)",
            (cutoff_safe_sort_key_ms,),
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
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER,
                source_sort_key REAL,
                source_updated_at TEXT
            );
            INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms)
            VALUES ('conv-current', 1779606000000, 1779606000000);
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
        session_id = _seed_raw_source_session(conn, session_id="conv-stale", source_path=source_path)
        conn.commit()

    def no_op_rebuild(
        conn: sqlite3.Connection,
        *,
        session_ids: list[str],
        page_size: int,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "insights",
    ) -> SessionInsightCounts:
        del conn, session_ids, page_size, stage_timings_s, stage_timing_prefix
        return SessionInsightCounts(profiles=0, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", no_op_rebuild)

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions([session_id]) is False


def test_insights_stage_rebuilds_large_session_after_quiet_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    db_path = tmp_path / "index.db"
    source_path = tmp_path / "quiet-codex.jsonl"
    _truncate(source_path, stages._HOT_INSIGHT_SOURCE_BYTES + 1)
    quiet_mtime = frozen_clock.now().timestamp() - stages._HOT_INSIGHT_QUIET_SECONDS - 5
    os.utime(source_path, (quiet_mtime, quiet_mtime))
    with open_connection(db_path) as conn:
        session_id = _seed_raw_source_session(conn, session_id="conv-quiet", source_path=source_path)
        conn.commit()

    monkeypatch.setattr(stages, "_stale_session_profile_ids", lambda _conn, _ids: [])

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    result = stage.execute_sessions([session_id])
    assert result
    assert isinstance(result, StageExecutionResult)
    assert "insights.load_batch" in result.stage_timings_s
    assert "insights.build_records.analysis" in result.stage_timings_s
    assert "insights.build_records.profile" in result.stage_timings_s
    assert "insights.analysis.facts" in result.stage_timings_s
    assert "insights.facts.message_flags" in result.stage_timings_s
    assert "insights.facts.message_facts" in result.stage_timings_s
    assert "insights.facts.aggregate_messages" in result.stage_timings_s
    assert "insights.profile.cost_summary" in result.stage_timings_s
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[0] == 1
        )


def test_insights_stage_rebuilds_small_active_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "index.db"
    source_path = tmp_path / "small-active-codex.jsonl"
    _truncate(source_path, 1024)
    with open_connection(db_path) as conn:
        session_id = _seed_raw_source_session(conn, session_id="conv-small", source_path=source_path)
        conn.commit()

    monkeypatch.setattr(stages, "_stale_session_profile_ids", lambda _conn, _ids: [])

    stage = make_insights_stage(db_path)
    assert stage.execute_sessions is not None
    result = stage.execute_sessions([session_id])
    assert result
    assert isinstance(result, StageExecutionResult)
    assert "insights.load_batch" in result.stage_timings_s
    assert "insights.build_records.analysis" in result.stage_timings_s
    assert "insights.build_records.profile" in result.stage_timings_s
    assert "insights.analysis.facts" in result.stage_timings_s
    assert "insights.facts.message_flags" in result.stage_timings_s
    assert "insights.facts.message_facts" in result.stage_timings_s
    assert "insights.facts.aggregate_messages" in result.stage_timings_s
    assert "insights.profile.cost_summary" in result.stage_timings_s
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[0] == 1
        )


def test_insights_stage_scopes_session_debt_to_stale_profiles(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    sessions = {
        "conv-fresh": "2026-05-24T01:00:00+00:00",
        "conv-missing-profile": "2026-05-24T01:01:00+00:00",
        "conv-stale-source": "2026-05-24T01:02:00+00:00",
        "conv-stale-version": "2026-05-24T01:03:00+00:00",
    }
    with open_connection(db_path) as conn:
        for session_id, updated_at in sessions.items():
            del updated_at
            _seed_index_session(conn, session_id=session_id, text=f"Message for {session_id}")
        assert stages._archive_insights_execute_ids(conn, [f"codex-session:{session_id}" for session_id in sessions])
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", ("codex-session:conv-missing-profile",))
        conn.execute(
            "UPDATE sessions SET updated_at_ms = ? WHERE session_id = ?",
            (1_769_000_000_000, "codex-session:conv-stale-source"),
        )
        conn.execute(
            "UPDATE insight_materialization SET materializer_version = ? WHERE session_id = ?",
            (SESSION_INSIGHT_MATERIALIZER_VERSION - 1, "codex-session:conv-stale-version"),
        )
        conn.commit()

    stage = make_insights_stage(db_path)
    assert stage.check_sessions is not None
    assert stage.check_sessions(
        [
            "codex-session:conv-fresh",
            "codex-session:conv-missing-profile",
            "codex-session:conv-stale-source",
            "codex-session:conv-stale-version",
            "codex-session:conv-unknown",
        ]
    ) == {
        "codex-session:conv-missing-profile",
        "codex-session:conv-stale-source",
        "codex-session:conv-stale-version",
    }


def test_archive_insights_execute_ids_preserves_millisecond_sort_key(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    session_id = "codex-session:conv-ms"
    source_sort_key_ms = 1_779_606_000_953
    with open_connection(db_path) as conn:
        _seed_index_session(conn, session_id="conv-ms", text="Message with millisecond sort key")
        conn.execute(
            """
            UPDATE sessions
            SET updated_at_ms = ?
            WHERE session_id = ?
            """,
            (source_sort_key_ms, session_id),
        )
        conn.commit()

        assert stages._archive_insights_execute_ids(conn, [session_id])

        profile = conn.execute(
            "SELECT source_sort_key FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert profile is not None
        assert profile["source_sort_key"] == pytest.approx(source_sort_key_ms / 1000.0)
        assert stages._archive_stale_session_profile_ids(conn, [session_id]) == []


def test_archive_insights_execute_ids_deduplicates_session_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        _seed_index_session(conn, session_id="conv-dupe", text="Message for duplicated session")
        conn.commit()

    seen_session_ids: list[list[str]] = []

    def fake_rebuild(
        conn: sqlite3.Connection,
        *,
        session_ids: list[str],
        page_size: int,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "insights",
    ) -> SimpleNamespace:
        del conn, page_size, stage_timing_prefix
        seen_session_ids.append(session_ids)
        if stage_timings_s is not None:
            stage_timings_s["insights.fake"] = 0.25
        return SimpleNamespace(profiles=1, work_events=0, phases=0, threads=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(stages, "_archive_hot_insight_session_ids", lambda _conn, _ids: set())
    monkeypatch.setattr(stages, "_archive_stale_session_profile_ids", lambda _conn, _ids: [])

    with sqlite3.connect(db_path) as conn:
        result = stages._archive_insights_execute_ids(
            conn,
            [
                "codex-session:conv-dupe",
                "codex-session:conv-dupe",
                "codex-session:conv-dupe",
            ],
        )
        assert result
        assert isinstance(result, StageExecutionResult)
        assert result.stage_timings_s == {"insights.fake": 0.25}

    assert seen_session_ids == [["codex-session:conv-dupe"]]


def test_archive_insights_execute_ids_rebuilds_quiet_subset_when_some_sessions_are_hot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        _seed_index_session(conn, session_id="conv-hot", text="Message for hot session")
        _seed_index_session(conn, session_id="conv-cold", text="Message for cold session")
        conn.commit()

    seen_session_ids: list[list[str]] = []

    def fake_rebuild(
        conn: sqlite3.Connection,
        *,
        session_ids: list[str],
        page_size: int,
        stage_timings_s: dict[str, float] | None = None,
        stage_timing_prefix: str = "insights",
    ) -> SimpleNamespace:
        del conn, page_size, stage_timing_prefix
        seen_session_ids.append(session_ids)
        if stage_timings_s is not None:
            stage_timings_s["insights.fake"] = 0.25
        return SimpleNamespace(profiles=1, work_events=0, phases=0, threads=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(stages, "_archive_hot_insight_session_ids", lambda _conn, _ids: {"codex-session:conv-hot"})
    monkeypatch.setattr(stages, "_archive_stale_session_profile_ids", lambda _conn, _ids: [])

    with sqlite3.connect(db_path) as conn:
        result = stages._archive_insights_execute_ids(
            conn,
            [
                "codex-session:conv-hot",
                "codex-session:conv-cold",
            ],
        )

    assert isinstance(result, StageExecutionResult)
    assert result.success is False
    assert result.stage_timings_s == {"insights.fake": 0.25}
    assert seen_session_ids == [["codex-session:conv-cold"]]


def test_archive_insights_execute_sessions_uses_write_connection_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        session_id = _seed_index_session(conn, session_id="conv-profile", text="Message for connection profile")
        conn.commit()

    seen_busy_timeout: list[int] = []

    def fake_execute_ids(conn: sqlite3.Connection, session_ids: list[str]) -> bool:
        assert session_ids == [session_id]
        seen_busy_timeout.append(int(conn.execute("PRAGMA busy_timeout").fetchone()[0]))
        return True

    monkeypatch.setattr(stages, "_archive_insights_execute_ids", fake_execute_ids)

    assert stages._archive_insights_execute_sessions(db_path, [session_id]) is True
    assert seen_busy_timeout == [stages._ARCHIVE_INSIGHT_WRITE_BUSY_TIMEOUT_MS]


def test_archive_insights_execute_sessions_defers_transient_sqlite_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        session_id = _seed_index_session(conn, session_id="conv-locked", text="Message for locked insight rebuild")
        conn.commit()

    def fail_locked(_conn: sqlite3.Connection, _session_ids: list[str]) -> bool:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(stages, "_archive_insights_execute_ids", fail_locked)

    assert stages._archive_insights_execute_sessions(db_path, [session_id]) is False


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
