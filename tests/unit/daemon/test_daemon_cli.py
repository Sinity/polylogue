from __future__ import annotations

import asyncio
import inspect
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.core.json import JSONDocument, loads
from polylogue.daemon.cli import main
from polylogue.daemon.convergence import ConvergenceStage
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.frozen_clock import FrozenClock


def _record_successful_repair(fake_conn: object, repairs: list[Any]) -> SimpleNamespace:
    repairs.append(fake_conn)
    return SimpleNamespace(success=True, repaired_count=0, detail="FTS index in sync")


def test_polylogued_help_lists_watch_command() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "browser-capture" in result.output
    assert "run" in result.output
    assert "status" in result.output
    assert "watch" in result.output
    assert "long-lived Polylogue local services" in result.output


@pytest.mark.contract
def test_polylogued_status_json_reports_daemon_components(
    tmp_path: Path,
) -> None:
    sources = (
        WatchSource(name="exists", root=tmp_path),
        WatchSource(name="missing", root=tmp_path / "missing"),
    )

    with patch("polylogue.daemon.status.default_sources", return_value=sources):
        result = CliRunner().invoke(
            main,
            [
                "status",
                "--spool",
                str(tmp_path / "captures"),
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0
    payload = loads(result.output)
    assert isinstance(payload, dict)
    live = cast(JSONDocument, payload["live"])
    browser_capture = cast(JSONDocument, payload["browser_capture"])
    assert payload["daemon"] == "polylogued"
    assert live["source_count"] == 2
    assert live["existing_source_count"] == 1
    assert browser_capture["spool_path"] == str(tmp_path / "captures")


def test_polylogued_status_plain_reports_daemon_components(tmp_path: Path) -> None:
    sources = (WatchSource(name="exists", root=tmp_path),)

    with patch("polylogue.daemon.status.default_sources", return_value=sources):
        result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0
    assert "Polylogue daemon" in result.output
    assert "Live sources: 1/1 available" in result.output
    assert f"exists: {tmp_path} (available)" in result.output
    assert "Browser capture spool:" in result.output


def test_polylogued_status_json_reports_archive_storage(tmp_path: Path) -> None:
    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("user.db", ArchiveTier.USER),
        ("ops.db", ArchiveTier.OPS),
    ):
        initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "embeddings.db") as conn:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status.db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.index_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = CliRunner().invoke(main, ["status", "--format", "json"])

    assert result.exit_code == 0
    payload = loads(result.output)
    assert isinstance(payload, dict)
    storage = cast(dict[str, object], payload["archive_storage"])
    assert storage["active_store"] == "archive_file_set"
    assert storage["archive_root"] == str(tmp_path)
    assert storage["configured_archive_root"] == str(tmp_path)
    assert storage["archive_root_matches_configured"] is True
    assert storage["archive_ready"] is True
    assert storage["final_shape_ready"] is True
    assert storage["present_tiers"] == ["source", "index", "embeddings", "user", "ops"]
    tiers = cast(list[dict[str, object]], storage["tiers"])
    assert {tier["name"]: tier["user_version"] for tier in tiers} == {
        "source": 1,
        "index": 2,
        "embeddings": 1,
        "user": 1,
        "ops": 1,
    }


def test_polylogued_status_plain_reports_archive_storage(tmp_path: Path) -> None:
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status.db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.index_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0
    assert "Storage: archive_file_set (source, index); missing embeddings, user, ops" in result.output


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_retries_due_items_without_source_failure(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    source = tmp_path / "session.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    cursor = CursorStore(db)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="source_path",
        subject_id=str(source),
        error="initial failure",
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'",
        )
        conn.commit()
    stage = ConvergenceStage(
        name="insights",
        description="retry test",
        check=lambda candidate: candidate == source,
        execute=lambda candidate: candidate == source,
    )
    with patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=(stage,)):
        retried = daemon_cli._drain_convergence_debt_once(db)
        debt_after = cursor.list_convergence_debt()

    assert retried == 1
    assert debt_after == []
    assert cursor.get_record(source) is None


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_retries_session_subjects_without_source_lookup(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    cursor = CursorStore(db)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="conv-1",
        error="initial failure",
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'",
        )
        conn.commit()
    stage = ConvergenceStage(
        name="insights",
        description="retry test",
        check=lambda _candidate: False,
        execute=lambda _candidate: False,
        check_sessions=lambda session_ids: {"conv-1"} if tuple(session_ids) == ("conv-1",) else set(),
        execute_sessions=lambda session_ids: tuple(session_ids) == ("conv-1",),
    )
    with patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=(stage,)):
        retried = daemon_cli._drain_convergence_debt_once(db)
        debt_after = cursor.list_convergence_debt()

    assert retried == 1
    assert debt_after == []


def test_periodic_convergence_check_treats_sqlite_lock_as_archive_busy(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()
    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("database is locked")

    with (
        patch("polylogue.daemon.cli._active_index_db_path", return_value=db),
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        patch.object(daemon_cli.logger, "info") as info,
        patch.object(daemon_cli.logger, "warning") as warning,
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_convergence_check(()))

    info.assert_called_once()
    assert info.call_args.args[0] == "convergence: archive busy; retrying derived debt on next tick: %s"
    warning.assert_not_called()


def test_periodic_convergence_check_warns_on_non_lock_failures(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()
    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected convergence retry failure")

    with (
        patch("polylogue.daemon.cli._active_index_db_path", return_value=db),
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        patch.object(daemon_cli.logger, "info") as info,
        patch.object(daemon_cli.logger, "warning") as warning,
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_convergence_check(()))

    info.assert_not_called()
    warning.assert_called_once()
    assert warning.call_args.args[0] == "convergence: check failed"
    assert warning.call_args.kwargs == {"exc_info": True}


def test_polylogued_browser_capture_help_lists_service_commands() -> None:
    result = CliRunner().invoke(main, ["browser-capture", "--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "status" in result.output


def test_polylogued_run_uses_default_sources() -> None:
    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch("polylogue.daemon.cli.default_sources", return_value=sources) as default_sources,
        patch("polylogue.daemon.cli.asyncio.run") as run,
    ):
        result = CliRunner().invoke(main, ["run", "--no-browser-capture", "--no-api", "--debounce-s", "0.25"])

    assert result.exit_code == 0
    default_sources.assert_called_once_with()
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Starting polylogued (watch=1 source(s)). Ctrl-C to stop." in result.stderr


def test_polylogued_run_rejects_empty_component_set() -> None:
    # All three components default to ON; only when every one is explicitly
    # disabled should `run` refuse to start.
    result = CliRunner().invoke(main, ["run", "--no-watch", "--no-browser-capture", "--no-api"])

    assert result.exit_code != 0
    assert "at least one daemon component must be enabled" in result.output


def test_polylogued_watch_uses_default_sources() -> None:
    runner = CliRunner()
    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch("polylogue.daemon.cli.default_sources", return_value=sources) as default_sources,
        patch("polylogue.daemon.cli.asyncio.run") as run,
    ):
        result = runner.invoke(main, ["watch", "--debounce-s", "0.25"])

    assert result.exit_code == 0
    default_sources.assert_called_once_with()
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Watching 1 source(s); debounce=0.25s" in result.stderr


def test_polylogued_watch_builds_sources_from_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "claude-code"
    root_b = tmp_path / "codex"

    with patch("polylogue.daemon.cli.asyncio.run") as run:
        result = CliRunner().invoke(
            main,
            [
                "watch",
                "--root",
                str(root_a),
                "--root",
                str(root_b),
            ],
        )

    assert result.exit_code == 0
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Watching 2 source(s); debounce=2.0s" in result.stderr


def test_run_live_watcher_stops_on_keyboard_interrupt() -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    stopped: list[bool] = []

    class FakeWatcher:
        stopped = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            raise KeyboardInterrupt

        def stop(self) -> None:
            self.stopped = True
            stopped.append(self.stopped)

    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch.object(daemon_cli, "Polylogue", FakePolylogue),
        patch.object(daemon_cli, "LiveWatcher", FakeWatcher),
    ):
        asyncio.run(daemon_cli.run_live_watcher(sources=sources, debounce_s=1.0))

    assert stopped == [True]


def test_ensure_fts_startup_readiness_skips_old_non_blocks_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.write_bytes(b"sqlite placeholder")

    class FakeCursor:
        def __init__(self, row: tuple[object, ...] | None, rows: list[tuple[object, ...]] | None = None) -> None:
            self._row = row
            self._rows = rows if rows is not None else ([] if row is None else [row])

        def fetchone(self) -> tuple[object, ...] | None:
            return self._row

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class FakeConnection:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.committed = False
            self.closed = False

        def execute(self, sql: str, _params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'":
                return FakeCursor(("messages_fts",))
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'":
                return FakeCursor(("messages",))
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(_params[0]) if isinstance(_params, tuple) and _params else ""
                return FakeCursor((1,)) if name in {"messages", "messages_fts"} else FakeCursor(None)
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                # All six FTS triggers present — no SIGKILL-drift recovery.
                triggers: list[tuple[object, ...]] = [
                    ("messages_fts_ai",),
                    ("messages_fts_ad",),
                    ("messages_fts_au",),
                ]
                return FakeCursor(triggers[0], rows=triggers)
            raise AssertionError(f"unexpected query: {query}")

        def commit(self) -> None:
            self.committed = True

        def close(self) -> None:
            self.closed = True

    conn = FakeConnection()
    rebuilds: list[FakeConnection] = []
    ensured: list[FakeConnection] = []

    def rebuild(fake_conn: FakeConnection) -> None:
        rebuilds.append(fake_conn)

    def ensure(fake_conn: FakeConnection) -> None:
        ensured.append(fake_conn)

    repairs: list[FakeConnection] = []

    monkeypatch.setattr("polylogue.paths.active_index_db_path", lambda: db)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", lambda _db, timeout: conn)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_index_sync", ensure)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", rebuild)
    monkeypatch.setattr("polylogue.storage.fts.freshness.ensure_fts_freshness_table_sync", lambda fake_conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.configure_bounded_repair_connection",
        lambda fake_conn: None,
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.repair_stale_fts_rows",
        lambda fake_conn: _record_successful_repair(fake_conn, repairs),
    )
    freshness_calls: list[FakeConnection] = []
    monkeypatch.setattr(
        "polylogue.daemon.fts_startup.record_fts_freshness_snapshot_sync",
        lambda fake_conn: freshness_calls.append(fake_conn),
    )

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    assert ensured == []
    assert rebuilds == []
    assert repairs == []
    assert conn.committed is False
    assert conn.closed is True
    assert freshness_calls == []


def test_ensure_fts_startup_readiness_does_not_rebuild_old_non_blocks_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.write_bytes(b"sqlite placeholder")

    class FakeCursor:
        def __init__(self, row: tuple[object, ...] | None, rows: list[tuple[object, ...]] | None = None) -> None:
            self._row = row
            self._rows = rows if rows is not None else ([] if row is None else [row])

        def fetchone(self) -> tuple[object, ...] | None:
            return self._row

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class FakeConnection:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.committed = False
            self.closed = False

        def execute(self, sql: str, _params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'":
                return FakeCursor(("messages_fts",))
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'":
                return FakeCursor(("messages",))
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(_params[0]) if isinstance(_params, tuple) and _params else ""
                return FakeCursor((1,)) if name in {"messages", "messages_fts"} else FakeCursor(None)
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                triggers: list[tuple[object, ...]] = [
                    ("messages_fts_ai",),
                    ("messages_fts_ad",),
                    ("messages_fts_au",),
                ]
                return FakeCursor(triggers[0], rows=triggers)
            raise AssertionError(f"unexpected query: {query}")

        def commit(self) -> None:
            self.committed = True

        def close(self) -> None:
            self.closed = True

    conn = FakeConnection()
    rebuilds: list[FakeConnection] = []
    restored: list[FakeConnection] = []

    def rebuild(fake_conn: FakeConnection) -> None:
        rebuilds.append(fake_conn)

    monkeypatch.setattr("polylogue.paths.active_index_db_path", lambda: db)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", lambda _db, timeout: conn)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_index_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.restore_fts_triggers_sync",
        lambda fake_conn: restored.append(fake_conn),
    )
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", rebuild)
    monkeypatch.setattr("polylogue.storage.fts.freshness.ensure_fts_freshness_table_sync", lambda fake_conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.configure_bounded_repair_connection",
        lambda fake_conn: None,
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.repair_stale_fts_rows",
        lambda fake_conn: SimpleNamespace(success=False, repaired_count=1, detail="excess rows"),
    )

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    assert restored == []
    assert rebuilds == []
    assert conn.committed is False
    assert conn.closed is True


def test_ensure_fts_startup_readiness_skips_when_blocks_table_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh-init/current-shape guard: if the canonical ``blocks`` table is not
    visible, startup skips FTS repair instead of probing an old monolithic
    shape.
    """
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.write_bytes(b"sqlite placeholder")

    class FakeCursor:
        def __init__(self, row: tuple[object, ...] | None) -> None:
            self._row = row

        def fetchone(self) -> tuple[object, ...] | None:
            return self._row

    class FakeConnection:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.committed = False
            self.closed = False

        def execute(self, sql: str, _params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                return FakeCursor(None)
            raise AssertionError(f"unexpected query: {query}")

        def commit(self) -> None:
            self.committed = True

        def close(self) -> None:
            self.closed = True

    conn = FakeConnection()
    monkeypatch.setattr("polylogue.paths.active_index_db_path", lambda: db)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", lambda _db, timeout: conn)
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.repair_stale_fts_rows",
        lambda _conn: pytest.fail("repair_stale_fts_rows must not run when blocks table is absent"),
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync",
        lambda _conn: pytest.fail("rebuild_fts_index_sync must not run when blocks table is absent"),
    )

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    assert "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1" in conn.queries
    assert conn.committed is False
    assert conn.closed is True


def test_ensure_fts_startup_readiness_skips_non_current_archive_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.write_bytes(b"sqlite placeholder")

    class FakeCursor:
        def __init__(self, row: tuple[object, ...] | None, rows: list[tuple[object, ...]] | None = None) -> None:
            self._row = row
            self._rows = rows if rows is not None else ([] if row is None else [row])

        def fetchone(self) -> tuple[object, ...] | None:
            return self._row

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class FakeConnection:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.committed = False
            self.closed = False

        def execute(self, sql: str, _params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'":
                return FakeCursor(("messages_fts",))
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'":
                return FakeCursor(("messages",))
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(_params[0]) if isinstance(_params, tuple) and _params else ""
                return FakeCursor((1,)) if name in {"messages", "messages_fts"} else FakeCursor(None)
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                # One missing trigger must send startup through trigger restore
                # before bounded repair can mark the FTS surfaces fresh.
                triggers: list[tuple[object, ...]] = [
                    ("messages_fts_ai",),
                    ("messages_fts_ad",),
                    ("messages_fts_au",),
                ]
                return FakeCursor(triggers[0], rows=triggers)
            if query == "SELECT 1 FROM messages WHERE text IS NOT NULL LIMIT 1":
                return FakeCursor((1,))
            if query == "SELECT 1 FROM messages_fts_docsize LIMIT 1":
                return FakeCursor((1,))
            raise AssertionError(f"unexpected query: {query}")

        def commit(self) -> None:
            self.committed = True

        def close(self) -> None:
            self.closed = True

    conn = FakeConnection()
    restored: list[FakeConnection] = []
    rebuilds: list[FakeConnection] = []

    monkeypatch.setattr("polylogue.paths.active_index_db_path", lambda: db)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", lambda _db, timeout: conn)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_index_sync", lambda fake_conn: None)
    freshness_calls: list[FakeConnection] = []
    monkeypatch.setattr(
        "polylogue.daemon.fts_startup.record_fts_freshness_snapshot_sync",
        lambda fake_conn: freshness_calls.append(fake_conn),
    )

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    assert restored == []
    assert rebuilds == []
    assert conn.committed is False
    assert conn.closed is True
    assert freshness_calls == []


def test_periodic_db_optimize_does_not_run_on_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    class SleepBeforeOptimizeError(Exception):
        pass

    opened: list[Path] = []

    async def fake_sleep(_seconds: float) -> None:
        raise SleepBeforeOptimizeError

    def fake_open_connection(path: Path, *, timeout: float) -> object:
        del timeout
        opened.append(path)
        raise AssertionError("PRAGMA optimize must not run at daemon startup")

    monkeypatch.setattr("polylogue.paths.db_path", lambda: tmp_path / "index.db")
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)

    with pytest.raises(SleepBeforeOptimizeError):
        asyncio.run(daemon_cli._periodic_db_optimize())

    assert opened == []


def test_daemon_cli_active_archive_uses_archive_file_set_from_archive_tiers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))

    assert daemon_cli._active_index_db_path() == index_db


def test_daemon_cli_active_archive_uses_index_when_db_anchor_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    db_anchor = tmp_path / "index.db"
    db_anchor.touch()
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))

    assert daemon_cli._active_index_db_path() == index_db


def test_daemon_cli_heartbeat_counts_archive(tmp_path: Path) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.types import BlockType, Provider

    archive_root = tmp_path
    with ArchiveStore(archive_root) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="daemon-heartbeat-v1",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="heartbeat v1",
                        content_blocks=[ParsedContentBlock(type=BlockType.TEXT, text="heartbeat v1")],
                    )
                ],
            )
        )

    assert daemon_cli._heartbeat_counts(archive_root / "index.db") == (1, 1, "sessions")


def test_ensure_fts_startup_readiness_handles_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.types import BlockType, Provider

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    with ArchiveStore(tmp_path) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="daemon-startup-v1",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="startup v1",
                        content_blocks=[ParsedContentBlock(type=BlockType.TEXT, text="startup v1")],
                    )
                ],
            )
        )

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='messages_fts'").fetchone() is not None
        row = conn.execute(
            """
            SELECT state, source_rows, indexed_rows
            FROM fts_freshness_state
            WHERE surface = 'messages_fts'
            """
        ).fetchone()
    assert row == ("ready", 1, 1)


def test_run_daemon_services_stops_live_watcher_on_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    stopped: list[bool] = []

    class FakeWatcher:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            stopped.append(True)

    with (
        patch.object(daemon_cli, "Polylogue", FakePolylogue),
        patch.object(daemon_cli, "LiveWatcher", FakeWatcher),
        pytest.raises(RuntimeError, match="watch stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                debounce_s=1.0,
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert stopped == [True]


def test_run_daemon_services_closes_browser_capture_server_on_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakeServer:
        shutdown_called = False
        close_called = False

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            assert poll_interval == 0.5
            raise RuntimeError("server stopped")

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.close_called = True

    server = FakeServer()
    with (
        patch.object(daemon_cli, "make_server", return_value=server),
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert server.shutdown_called is False
    assert server.close_called is True


def test_run_daemon_services_shutdowns_running_server_on_watcher_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class FakeWatcher:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            return None

    class BlockingServer:
        shutdown_called = False
        close_called = False

        def __init__(self) -> None:
            self._stopped = threading.Event()

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            assert poll_interval == 0.5
            self._stopped.wait(timeout=5)

        def shutdown(self) -> None:
            self.shutdown_called = True
            self._stopped.set()

        def server_close(self) -> None:
            self.close_called = True

    server = BlockingServer()
    with (
        patch.object(daemon_cli, "Polylogue", FakePolylogue),
        patch.object(daemon_cli, "LiveWatcher", FakeWatcher),
        patch.object(daemon_cli, "make_server", return_value=server),
        pytest.raises(RuntimeError, match="watch stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                debounce_s=1.0,
                enable_watch=True,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert server.shutdown_called is True
    assert server.close_called is True


def test_run_daemon_services_schema_block_skips_db_background_work() -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    class FakeServer:
        shutdown_called = False
        close_called = False

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            assert poll_interval == 0.5
            raise RuntimeError("server stopped")

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.close_called = True

    def fail_background_work(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema-blocked daemon must not start DB background work")

    server = FakeServer()
    critical = HealthAlert(
        check_name="schema_version",
        tier=HealthTier.FAST,
        severity=HealthSeverity.CRITICAL,
        message="archive2 is not runtime v8",
        checked_at="2026-05-24T00:00:00+00:00",
    )
    with (
        patch.object(daemon_cli, "_check_schema_version_fast", return_value=critical),
        patch.object(daemon_cli, "_periodic_wal_checkpoint", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_heartbeat", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_convergence_check", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_health_check", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_db_optimize", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_status_snapshot_refresh", side_effect=fail_background_work),
        patch("polylogue.daemon.convergence.DaemonConverger", side_effect=fail_background_work),
        patch.object(daemon_cli, "make_server", return_value=server),
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                debounce_s=1.0,
                enable_watch=True,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert server.shutdown_called is False
    assert server.close_called is True
