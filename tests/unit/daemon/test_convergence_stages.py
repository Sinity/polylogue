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
from tests.infra.storage_records import make_conversation, make_message, store_records


def _seed_raw_source_conversation(conn: sqlite3.Connection, *, conversation_id: str, source_path: Path) -> None:
    conn.execute(
        """
        INSERT INTO raw_conversations (
            raw_id,
            provider_name,
            source_path,
            blob_size,
            acquired_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            f"raw-{conversation_id}",
            "codex",
            str(source_path),
            source_path.stat().st_size,
            "2026-05-24T01:00:00+00:00",
        ),
    )
    store_records(
        conversation=make_conversation(
            conversation_id,
            provider_name="codex",
            title=conversation_id,
            created_at="2026-05-24T01:00:00+00:00",
            updated_at="2026-05-24T01:00:00+00:00",
            raw_id=f"raw-{conversation_id}",
        ),
        messages=[
            make_message(
                f"{conversation_id}:msg-1",
                conversation_id,
                text=f"Message for {conversation_id}",
            )
        ],
        attachments=[],
        conn=conn,
    )


def _truncate(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


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
            if "raw_conversations" in sql:
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
        conversation_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        nonlocal rebuilt
        del conn
        rebuilt = True
        assert conversation_ids == ["conv-1"]
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

    def fake_repair_messages(conn: FakeConnection, conversation_ids: list[str]) -> None:
        repaired_messages.append(conversation_ids)

    def fake_insert_missing_actions(conn: FakeConnection, conversation_ids: list[str]) -> None:
        inserted_actions.append(conversation_ids)

    def fake_rebuild(conn: FakeConnection) -> None:
        nonlocal rebuilt
        rebuilt = True

    needs_calls: list[list[str]] = []
    marked_ready: list[FakeConnection] = []

    def fake_repair_needs(_conn: FakeConnection, conversation_ids: list[str]) -> stages._FtsRepairNeeds:
        needs_calls.append(conversation_ids)
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
        "_conversation_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )
    monkeypatch.setattr(stages, "_fts_repair_needs_for_conversations", fake_repair_needs)
    monkeypatch.setattr(stages, "_action_events_exist_for_conversations", lambda _conn, _ids: False)
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
    assert stages._action_events_exist_for_conversations(conn, ["conv-a", "conv-b"]) is True
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

    assert stages._fts_repair_needs_for_conversations(conn, ["conv-a"]) == stages._FtsRepairNeeds()

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
            conversation=make_conversation("conv-empty-text", provider_name="codex"),
            messages=[make_message("msg-empty-text", "conv-empty-text", text="")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

        assert stages._fts_repair_needs_for_conversations(conn, ["conv-empty-text"]) == stages._FtsRepairNeeds()


def test_targeted_fts_ready_marker_preserves_ledger_counts(tmp_path: Path) -> None:
    from polylogue.storage.fts.freshness import record_fts_surface_state_sync

    db_path = tmp_path / "archive.sqlite"
    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations(conversation_id, provider_name, provider_conversation_id, version) VALUES(?,?,?,1)",
            ("conv-ledger", "codex", "provider-conv"),
        )
        conn.execute(
            "INSERT INTO messages(message_id, conversation_id, role, text, provider_name, version) VALUES(?,?,?,?,?,1)",
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
        "targeted changed-conversation repair complete",
    )


def test_targeted_fts_ready_marker_handles_legacy_freshness_table(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations(conversation_id, provider_name, provider_conversation_id, version) VALUES(?,?,?,1)",
            ("conv-legacy-ledger", "codex", "provider-conv"),
        )
        conn.execute(
            "INSERT INTO messages(message_id, conversation_id, role, text, provider_name, version) VALUES(?,?,?,?,?,1)",
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
        "targeted changed-conversation repair complete",
    )


def test_embed_stage_scopes_changed_conversations_without_asyncio_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    db_path.touch()
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    embedded_calls: list[list[str]] = []
    pending_calls = 0

    class FakeConnection:
        def close(self) -> None:
            pass

        def commit(self) -> None:
            pass

    def fake_open_connection(path: Path, *, timeout: float) -> FakeConnection:
        assert path == db_path
        assert timeout == 5.0
        return FakeConnection()

    def fail_if_used(coro: object) -> object:
        raise AssertionError("embed stage should not open an asyncio runner")

    def fake_pending(_conn: FakeConnection, conversation_ids: list[str]) -> list[str]:
        nonlocal pending_calls
        pending_calls += 1
        if pending_calls > 2:
            return []
        return [conversation_id for conversation_id in conversation_ids if conversation_id in {"conv-a", "conv-c"}]

    def fake_embed(
        _db_path: Path,
        conversation_ids: list[str],
        *,
        max_errors: int | None = None,
        stop_after_seconds: int | None = None,
    ) -> bool:
        assert max_errors is not None
        assert stop_after_seconds is not None
        embedded_calls.append(list(conversation_ids))
        return True

    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.setenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", "1")
    monkeypatch.setattr(asyncio, "run", fail_if_used)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)
    monkeypatch.setattr(
        stages,
        "_conversation_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a", "conv-b"], Path(paths[1]): ["conv-c"]},
    )
    monkeypatch.setattr(stages, "_pending_embedding_conversation_ids", fake_pending)
    monkeypatch.setattr(stages, "_embed_conversations_sync", fake_embed)
    monkeypatch.setattr(stages, "_reconcile_embedding_config_change", lambda _conn: None)

    stage = make_embed_stage(db_path)
    assert stage.check_many is not None
    assert stage.execute_many is not None
    assert stage.check_many([path_a, path_b]) == {path_a, path_b}
    assert stage.execute_many([path_a, path_b]) is True
    assert embedded_calls == [["conv-a", "conv-c"]]


def test_embed_stage_processes_bounded_window_and_leaves_debt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    db_path.touch()
    path_a = tmp_path / "a.jsonl"
    embedded_calls: list[list[str]] = []
    remaining_checks = 0

    class FakeConnection:
        def close(self) -> None:
            pass

        def commit(self) -> None:
            pass

    def fake_open_connection(path: Path, *, timeout: float) -> FakeConnection:
        assert path == db_path
        assert timeout == 5.0
        return FakeConnection()

    def fake_pending(_conn: FakeConnection, conversation_ids: list[str] | tuple[str, ...]) -> list[str]:
        nonlocal remaining_checks
        ids = list(conversation_ids)
        if ids == ["conv-a", "conv-b", "conv-c"]:
            remaining_checks += 1
            return ["conv-c"] if remaining_checks > 1 else ["conv-a", "conv-b"]
        return []

    def fake_embed(
        _db_path: Path,
        conversation_ids: list[str],
        *,
        max_errors: int | None = None,
        stop_after_seconds: int | None = None,
    ) -> bool:
        embedded_calls.append(list(conversation_ids))
        return True

    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.setenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", "1")
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)
    monkeypatch.setattr(
        stages, "_conversation_ids_for_source_path", lambda _conn, _path: ["conv-a", "conv-b", "conv-c"]
    )
    monkeypatch.setattr(stages, "_pending_embedding_conversation_ids", fake_pending)
    monkeypatch.setattr(stages, "_embed_conversations_sync", fake_embed)
    monkeypatch.setattr(stages, "_reconcile_embedding_config_change", lambda _conn: None)

    stage = make_embed_stage(db_path)

    assert stage.execute(path_a) is False
    assert embedded_calls == [["conv-a", "conv-b"]]


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
        conversation_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn
        rebuild_calls.append((conversation_ids, page_size))
        return SessionInsightCounts(profiles=2, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.sqlite.connection.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(
        stages,
        "_conversation_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )
    monkeypatch.setattr(stages, "_hot_insight_conversation_ids", lambda _conn, _ids: set())

    stage = make_insights_stage(db_path)
    assert stage.execute_many is not None
    assert stage.execute_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is True
    assert rebuild_calls == [(["conv-a", "conv-b"], 10)]


def test_insights_stage_defers_hot_large_conversation_debt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    source_path = tmp_path / "active-codex.jsonl"
    _truncate(source_path, stages._HOT_INSIGHT_SOURCE_BYTES + 1)
    with open_connection(db_path) as conn:
        _seed_raw_source_conversation(conn, conversation_id="conv-hot", source_path=source_path)
        conn.commit()

    def fail_rebuild(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("hot active sources should wait for convergence debt retry")

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fail_rebuild)

    stage = make_insights_stage(db_path)
    assert stage.execute_conversations is not None
    assert stage.execute_conversations(["conv-hot"]) is False


def test_insights_staleness_uses_sort_key_not_timestamp_text(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                sort_key REAL,
                updated_at TEXT
            );
            CREATE TABLE session_profiles (
                conversation_id TEXT PRIMARY KEY,
                materializer_version INTEGER,
                source_sort_key REAL,
                source_updated_at TEXT
            );
            INSERT INTO conversations (conversation_id, sort_key, updated_at)
            VALUES ('conv-current', 1779606000.0, '2026-05-24T07:00:00+00:00');
            """
        )
        conn.execute(
            """
            INSERT INTO session_profiles (
                conversation_id,
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


def test_insights_conversation_rebuild_returns_false_when_still_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    source_path = tmp_path / "quiet-codex.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    with open_connection(db_path) as conn:
        _seed_raw_source_conversation(conn, conversation_id="conv-stale", source_path=source_path)
        conn.commit()

    def no_op_rebuild(
        conn: sqlite3.Connection,
        *,
        conversation_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn, conversation_ids, page_size
        return SessionInsightCounts(profiles=0, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", no_op_rebuild)

    stage = make_insights_stage(db_path)
    assert stage.execute_conversations is not None
    assert stage.execute_conversations(["conv-stale"]) is False


def test_insights_stage_rebuilds_large_conversation_after_quiet_window(
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
        _seed_raw_source_conversation(conn, conversation_id="conv-quiet", source_path=source_path)
        conn.commit()

    def fake_rebuild(
        conn: sqlite3.Connection,
        *,
        conversation_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn
        rebuild_calls.append((conversation_ids, page_size))
        return SessionInsightCounts(profiles=1, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(stages, "_stale_session_profile_ids", lambda _conn, _ids: [])

    stage = make_insights_stage(db_path)
    assert stage.execute_conversations is not None
    assert stage.execute_conversations(["conv-quiet"]) is True
    assert rebuild_calls == [(["conv-quiet"], 10)]


def test_insights_stage_rebuilds_small_active_conversation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    source_path = tmp_path / "small-active-codex.jsonl"
    _truncate(source_path, 1024)
    rebuild_calls: list[tuple[list[str], int]] = []
    with open_connection(db_path) as conn:
        _seed_raw_source_conversation(conn, conversation_id="conv-small", source_path=source_path)
        conn.commit()

    def fake_rebuild(
        conn: sqlite3.Connection,
        *,
        conversation_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn
        rebuild_calls.append((conversation_ids, page_size))
        return SessionInsightCounts(profiles=1, work_events=0, phases=0, threads=0, tag_rollups=0)

    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(stages, "_stale_session_profile_ids", lambda _conn, _ids: [])

    stage = make_insights_stage(db_path)
    assert stage.execute_conversations is not None
    assert stage.execute_conversations(["conv-small"]) is True
    assert rebuild_calls == [(["conv-small"], 10)]


def test_insights_stage_scopes_conversation_debt_to_stale_profiles(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    conversations = {
        "conv-fresh": "2026-05-24T01:00:00+00:00",
        "conv-missing-profile": "2026-05-24T01:01:00+00:00",
        "conv-stale-source": "2026-05-24T01:02:00+00:00",
        "conv-stale-version": "2026-05-24T01:03:00+00:00",
    }
    with open_connection(db_path) as conn:
        for conversation_id, updated_at in conversations.items():
            store_records(
                conversation=make_conversation(
                    conversation_id,
                    provider_name="codex",
                    title=conversation_id,
                    created_at=updated_at,
                    updated_at=updated_at,
                ),
                messages=[
                    make_message(
                        f"{conversation_id}:msg-1",
                        conversation_id,
                        text=f"Message for {conversation_id}",
                    )
                ],
                attachments=[],
                conn=conn,
            )
        rebuild_session_insights_sync(
            conn,
            conversation_ids=list(conversations),
            page_size=10,
        )
        conn.execute("DELETE FROM session_profiles WHERE conversation_id = ?", ("conv-missing-profile",))
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            ("2026-05-24T02:02:00+00:00", "conv-stale-source"),
        )
        conn.execute(
            "UPDATE session_profiles SET materializer_version = ? WHERE conversation_id = ?",
            (SESSION_INSIGHT_MATERIALIZER_VERSION - 1, "conv-stale-version"),
        )
        conn.commit()

    stage = make_insights_stage(db_path)
    assert stage.check_conversations is not None
    assert stage.check_conversations(
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
