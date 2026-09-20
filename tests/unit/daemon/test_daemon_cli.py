from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from polylogue.config import Config
from polylogue.core.json import JSONDocument, loads
from polylogue.daemon.cli import main
from polylogue.daemon.convergence import ConvergenceStage
from polylogue.daemon.derivation import DerivationReport
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.health import DaemonHealth, HealthSeverity, HealthTier
from polylogue.daemon.session_profile_composition import ComposedSessionProfiles
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.logging import capture
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.derived.raw import RawObservationScope
from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from tests.infra.frozen_clock import FrozenClock
from tests.infra.live_ingest import write_index_session


async def _unused_session_profile_callback(_session_ids: Sequence[str] | None) -> DerivationReport:
    raise AssertionError("session-profile callback should not run")


class _NoIntakeHints:
    def intake_revision(self, source: WatchSource) -> int:
        return 0


def _resolved_config(**overrides: object) -> Any:
    """Return a ``load_polylogue_config`` stand-in answering every config key.

    Daemon modules bind ``load_polylogue_config`` at import time, so a module
    first imported while this seam is patched keeps the stand-in for the rest
    of the process. Only a fully resolved config can answer the keys those
    later readers ask for.
    """
    from polylogue.config import load_polylogue_config

    resolved = load_polylogue_config(cli_overrides=dict(overrides))
    return lambda **_kwargs: resolved


def test_polylogued_help_lists_watch_command() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "browser-capture" in result.output
    assert "run" in result.output
    assert "status" in result.output
    assert "watch" in result.output
    assert "long-lived Polylogue local services" in result.output


def test_polylogued_version_option_reports_version() -> None:
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("polylogued, version ")


def test_polylogued_health_json_runs_against_isolated_workspace(workspace_env: dict[str, Path]) -> None:
    """Health CLI returns structured output without requiring a live daemon."""
    result = CliRunner().invoke(main, ["health", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = loads(result.output)
    assert isinstance(payload, dict)
    assert "overall_status" in payload
    assert isinstance(payload["alerts"], list)


def test_polylogued_health_error_json_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI health propagates an unhealthy aggregate through its process status."""
    monkeypatch.setattr(
        "polylogue.daemon.cli.check_health",
        lambda *, tiers: DaemonHealth(overall_status=HealthSeverity.ERROR, checked_at="2026-07-13T00:00:00+00:00"),
    )

    result = CliRunner().invoke(main, ["health", "--format", "json"])

    assert result.exit_code == 1
    payload = loads(result.output)
    assert isinstance(payload, dict)
    assert payload["overall_status"] == "error"


def test_polylogued_health_expensive_flag_selects_all_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The expensive convenience flag must request all health tiers."""
    observed: list[set[HealthTier]] = []

    def _check_health(*, tiers: set[HealthTier]) -> DaemonHealth:
        observed.append(tiers)
        return DaemonHealth(overall_status=HealthSeverity.OK, checked_at="2026-07-13T00:00:00+00:00")

    monkeypatch.setattr("polylogue.daemon.cli.check_health", _check_health)

    result = CliRunner().invoke(main, ["health", "--expensive", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert observed == [{HealthTier.FAST, HealthTier.MEDIUM, HealthTier.EXPENSIVE}]


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

    assert result.exit_code == 1
    payload = loads(result.output)
    assert isinstance(payload, dict)
    live = cast(JSONDocument, payload["live"])
    browser_capture = cast(JSONDocument, payload["browser_capture"])
    assert payload["daemon"] == "polylogued"
    assert live["source_count"] == 2
    assert live["existing_source_count"] == 1
    assert browser_capture["spool_ready"] is True
    assert browser_capture["spool_path"] == str(tmp_path / "captures")


def test_polylogued_status_plain_reports_daemon_components(tmp_path: Path) -> None:
    sources = (WatchSource(name="exists", root=tmp_path),)

    with patch("polylogue.daemon.status.default_sources", return_value=sources):
        result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 1
    assert "Polylogue daemon" in result.output
    assert "Live sources: 1/1 available" in result.output
    assert f"exists: {tmp_path} (available)" in result.output
    assert "Browser capture spool: ready" in result.output


def test_polylogued_status_json_reports_archive_storage(tmp_path: Path) -> None:
    from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier

    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("user.db", ArchiveTier.USER),
        ("audit.db", ArchiveTier.AUDIT),
        ("ops.db", ArchiveTier.OPS),
    ):
        initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "embeddings.db") as conn:
        conn.execute(f"PRAGMA user_version = {EMBEDDINGS_SCHEMA_VERSION}")
        conn.commit()
    inspect_raw_authority_frontier(
        Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[], db_path=tmp_path / "index.db")
    )

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
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
    assert storage["final_shape_ready"] is True
    assert storage["schema_mismatches"] == []
    assert storage["archive_schema_ready"] is True
    assert storage["archive_ready"] is True
    assert storage["present_tiers"] == ["source", "index", "embeddings", "user", "audit", "ops"]
    tiers = cast(list[dict[str, object]], storage["tiers"])
    assert {tier["name"]: tier["user_version"] for tier in tiers} == {
        "source": SOURCE_SCHEMA_VERSION,
        "index": INDEX_SCHEMA_VERSION,
        "embeddings": EMBEDDINGS_SCHEMA_VERSION,
        "user": USER_SCHEMA_VERSION,
        "audit": AUDIT_SCHEMA_VERSION,
        "ops": 1,
    }
    assert {tier["name"]: tier["version_status"] for tier in tiers} == {
        "source": "ok",
        "index": "ok",
        "embeddings": "ok",
        "user": "ok",
        "audit": "ok",
        "ops": "ok",
    }


def test_polylogued_status_json_reports_schema_mismatch_not_ready(tmp_path: Path) -> None:
    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("embeddings.db", ArchiveTier.EMBEDDINGS),
        ("user.db", ArchiveTier.USER),
        ("audit.db", ArchiveTier.AUDIT),
        ("ops.db", ArchiveTier.OPS),
    ):
        initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = CliRunner().invoke(main, ["status", "--format", "json"])

    assert result.exit_code == 1
    payload = loads(result.output)
    assert isinstance(payload, dict)
    # The refused index tier leaves raw frontier integrity uncollectable, so
    # the daemon is not ready and the command exits non-zero.
    integrity = cast(dict[str, object], payload["raw_frontier_integrity"])
    assert integrity["available"] is False
    assert integrity["overall_status"] == "unknown"
    storage_raw = payload["archive_storage"]
    assert isinstance(storage_raw, dict)
    storage = cast(dict[str, object], storage_raw)
    assert storage["active_store"] == "archive_file_set"
    assert storage["archive_ready"] is False
    assert storage["final_shape_ready"] is True
    assert storage["archive_schema_ready"] is False
    assert storage["schema_mismatches"] == ["index"]
    tiers = cast(list[dict[str, object]], storage["tiers"])
    index_tier = next(tier for tier in tiers if tier["name"] == "index")
    assert index_tier["user_version"] == 1
    assert index_tier["expected_user_version"] == INDEX_SCHEMA_VERSION
    assert index_tier["version_status"] == "mismatch"
    components_raw = payload["component_readiness"]
    assert isinstance(components_raw, dict)
    components = cast(dict[str, dict[str, object]], components_raw)
    archive_component = components["archive_storage"]
    assert archive_component["state"] == "blocked"
    assert archive_component["repair_hint"] == "polylogued run"


def test_polylogued_status_plain_reports_archive_storage(tmp_path: Path) -> None:
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 1
    assert "Storage: archive_file_set (source, index); missing embeddings, user, audit, ops" in result.output


def test_polylogued_status_plain_reports_schema_mismatch(tmp_path: Path) -> None:
    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("embeddings.db", ArchiveTier.EMBEDDINGS),
        ("user.db", ArchiveTier.USER),
        ("audit.db", ArchiveTier.AUDIT),
        ("ops.db", ArchiveTier.OPS),
    ):
        initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 1
    assert (
        "Storage: archive_file_set (source, index, embeddings, user, audit, ops); final split complete; schema mismatch index"
        in result.output
    )


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_migrates_retired_insights_stage(
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
        name="derived",
        description="retry test",
        check=lambda candidate: candidate == source,
        execute=lambda candidate: candidate == source,
    )
    with patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=(stage,)):
        retried = daemon_cli._drain_convergence_debt_once(db)
        debt_after = cursor.list_convergence_debt()

    assert retried == 0
    assert len(debt_after) == 1
    assert debt_after[0].stage == "derived"
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
        stage="convergence",
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
        name="derived",
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


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_preserves_error_for_unimplemented_stage(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    """A stage with no registered implementation leaves its debt row untouched.

    ``lineage_prefix_recompose`` debt names the identity contradiction that
    truncated a child's lineage. No convergence stage implements it, so the
    drain measures nothing about the row; re-recording it would overwrite that
    diagnostic with a note about the missing stage (polylogue-ia88n).

    Anti-vacuity: restoring the ``convergence retry stage unavailable: ...``
    re-record replaces ``last_error`` on the first drain pass and this
    assertion fails.
    """
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    cursor = CursorStore(db)
    cursor.record_convergence_debt(
        stage="lineage_prefix_recompose",
        subject_type="session_id",
        subject_id="claude-code-session:child-1",
        error="alias collision truncated child prefix at message 42",
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'")
        conn.commit()
    stage = ConvergenceStage(
        name="unrelated_stage",
        description="retry test",
        check=lambda _candidate: False,
        execute=lambda _candidate: False,
    )
    with patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=(stage,)):
        retried = daemon_cli._drain_convergence_debt_once(db)
        debt_after = cursor.list_convergence_debt()

    assert retried == 0
    assert len(debt_after) == 1
    row = debt_after[0]
    assert row.stage == "lineage_prefix_recompose"
    assert row.last_error == "alias collision truncated child prefix at message 42"


def test_no_default_sources_makes_root_the_complete_watch_set(tmp_path: Path) -> None:
    """``--no-default-sources`` drops the typed defaults; the default stays additive.

    Anti-vacuity: keeping ``default_sources()`` in the list when
    ``include_defaults=False`` leaves the hermes/inbox/browser-capture roots in
    the watch set, so the isolated-root assertion fails; dropping them
    unconditionally makes the additive assertion fail.
    """
    from polylogue.daemon import cli as daemon_cli

    isolated = tmp_path / "isolated"
    isolated.mkdir()

    with patch("polylogue.paths.archive_root", return_value=tmp_path / "archive"):
        additive = daemon_cli._watch_sources_from_roots((isolated,))
        exclusive = daemon_cli._watch_sources_from_roots((isolated,), include_defaults=False)

    assert len(additive) > 1
    assert isolated in {source.root for source in additive}
    assert [source.root for source in exclusive] == [isolated]

    # The flag is wired onto the daemon entry points, not just the helper.
    for command in (daemon_cli.run_command, daemon_cli.watch_command):
        assert "no_default_sources" in {param.name for param in command.params}


def test_periodic_convergence_check_treats_sqlite_lock_as_archive_busy(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()

    def fake_drain(_db: Path) -> int:
        raise sqlite3.OperationalError("database is locked")

    with (
        patch.object(daemon_cli, "_drain_convergence_debt_once", fake_drain),
        patch.object(
            daemon_cli,
            "daemon_write_coordinator",
            return_value=SimpleNamespace(run_sync=None),
        ),
        capture() as records,
    ):
        asyncio.run(daemon_cli._retry_convergence_debt_once(db))

    # ``.start`` is DEBUG and sits below the default threshold; the terminal
    # event is the one an operator reads.
    terminals = [r for r in records if str(r["event"]).startswith("daemon.convergence_debt.pass.")]
    assert [r["event"] for r in terminals] == ["daemon.convergence_debt.pass.degraded"]
    assert terminals[-1]["outcome"] == "degraded"
    assert terminals[-1]["reason"] == "archive_busy"
    assert terminals[-1]["error_type"] == "OperationalError"


def test_periodic_drive_source_catchup_waits_for_watcher_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote Drive work cannot monopolize startup ahead of local sessions."""
    from polylogue.daemon import cli as daemon_cli

    calls: list[str] = []

    async def fake_run(_callback: object) -> int:
        from polylogue.storage.sqlite.write_lease import current_write_lease

        assert current_write_lease() is None
        calls.append("drive")
        raise asyncio.CancelledError

    async def exercise() -> None:
        watcher_registered = asyncio.Event()
        monkeypatch.setattr(
            daemon_cli,
            "_run_drive_source_catchup_safely",
            fake_run,
        )
        task = asyncio.create_task(
            daemon_cli._periodic_drive_source_catchup(
                session_profile_callback=_unused_session_profile_callback,
                watcher_registered=watcher_registered,
            )
        )
        await asyncio.sleep(0)
        assert calls == []
        watcher_registered.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert calls == ["drive"]


def test_spool_pending_check_ignores_terminal_cursor_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Excluded or repeatedly-failed spool files are owned by exclusion and
    retry policy; they must not park raw materialization indefinitely. Only
    a file with no cursor yet (genuinely awaiting its live route) yields."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.live.watcher import _PARSER_FINGERPRINT

    spool = tmp_path / "browser-capture"
    spool.mkdir()
    capture = spool / "chatgpt-capture.json"
    capture.write_bytes(b"{}\n")

    records: dict[Path, object] = {}
    monkeypatch.setattr("polylogue.paths.browser_capture_spool_root", lambda: spool)
    monkeypatch.setattr(daemon_cli, "_active_index_db_path", lambda: tmp_path / "index.db")
    monkeypatch.setattr(
        "polylogue.sources.live.cursor.CursorStore",
        lambda _db, **_kwargs: SimpleNamespace(get_record=lambda path: records.get(path)),
    )

    # A capture that JUST arrived is in the live route's debounce flow —
    # it must not park the conveyor even without a cursor.
    records.clear()
    assert daemon_cli._browser_capture_spool_has_pending_files() is False

    # Age the file past the grace window: now it is a stalled backlog.
    stale = capture.stat().st_mtime - daemon_cli._SPOOL_PENDING_GRACE_SECONDS - 60
    os.utime(capture, (stale, stale))
    stat = capture.stat()

    def cursor_record(*, excluded: bool = False, failure_count: int = 0) -> SimpleNamespace:
        return SimpleNamespace(
            excluded=excluded,
            failure_count=failure_count,
            parser_fingerprint=_PARSER_FINGERPRINT,
            byte_size=stat.st_size,
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            content_fingerprint="fp",
        )

    records.clear()
    assert daemon_cli._browser_capture_spool_has_pending_files() is True

    records[capture] = cursor_record(excluded=True)
    assert daemon_cli._browser_capture_spool_has_pending_files() is False

    records[capture] = cursor_record(failure_count=3)
    assert daemon_cli._browser_capture_spool_has_pending_files() is False


def test_periodic_wal_checkpoint_targets_archive_root_tiers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.wal_checkpoint import checkpoint_archive_wals

    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_run_sync(actor: str, func: object, *args: object, **kwargs: object) -> object:
        assert actor == daemon_cli.WAL_CHECKPOINT_ACTOR
        calls.append((func, args, kwargs))
        raise asyncio.CancelledError

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch.object(
            daemon_cli,
            "daemon_write_coordinator",
            return_value=SimpleNamespace(run_sync=fake_run_sync),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_wal_checkpoint())

    # Recurring means PASSIVE only, under its own actor, with blocker evidence
    # collected because this is a background route that can afford the scan.
    assert calls == [
        (
            checkpoint_archive_wals,
            (tmp_path,),
            {"reason": "periodic", "escalation": "recurring", "collect_blockers": True},
        )
    ]


def test_periodic_convergence_check_waits_for_watcher_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()
    drains: list[Path] = []
    fts_scopes: list[object] = []
    profile_scopes: list[tuple[str, ...] | None] = []
    drained = asyncio.Event()

    def fake_drain(drain_db: Path) -> int:
        drains.append(drain_db)
        return 0

    async def fake_session_profiles(scope: tuple[str, ...] | None) -> object:
        profile_scopes.append(scope)
        return SimpleNamespace()

    async def fake_fts_converge() -> object:
        fts_scopes.append(None)
        drained.set()
        return SimpleNamespace()

    async def exercise() -> None:
        watcher_registered = asyncio.Event()
        monkeypatch.setattr(daemon_cli, "_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS", 60)
        monkeypatch.setattr(
            daemon_cli,
            "daemon_write_coordinator",
            lambda: SimpleNamespace(run_sync=None),
        )
        monkeypatch.setattr(daemon_cli, "_drain_convergence_debt_once", fake_drain)
        monkeypatch.setattr(daemon_cli, "_active_index_db_path", lambda: db)
        task = asyncio.create_task(
            daemon_cli._periodic_convergence_check(
                (),
                fts_owner=cast(Any, SimpleNamespace(converge=fake_fts_converge)),
                watcher_registered=watcher_registered,
                session_profile_callback=fake_session_profiles,
            )
        )
        await asyncio.sleep(0)
        assert drains == []
        assert profile_scopes == []
        watcher_registered.set()
        await asyncio.wait_for(drained.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert drains == [db]
    assert fts_scopes == [None]
    assert profile_scopes == [None]


def test_periodic_convergence_check_warns_on_non_lock_failures(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()

    def fake_drain(_db: Path) -> int:
        raise RuntimeError("unexpected convergence retry failure")

    # The drain itself runs off the writer lease (polylogue-ssplv); the
    # coordinator is reached only by the admission each stage's write uses.
    with (
        patch.object(daemon_cli, "_drain_convergence_debt_once", fake_drain),
        patch.object(
            daemon_cli,
            "daemon_write_coordinator",
            return_value=SimpleNamespace(run_sync=None),
        ),
        capture() as records,
    ):
        asyncio.run(daemon_cli._retry_convergence_debt_once(db))

    # The span's terminal event is emitted from ``__exit__``, so the swallowed
    # failure is still on the record at ERROR rather than silently dropped.
    errors = [r for r in records if r["event"] == "daemon.convergence_debt.pass.error"]
    assert len(errors) == 1
    assert errors[0]["level"] == "error"
    assert errors[0]["outcome"] == "error"
    assert errors[0]["error_type"] == "RuntimeError"
    assert "unexpected convergence retry failure" in str(errors[0]["error_detail"])
    assert [r for r in records if r["event"] == "daemon.convergence_debt.pass.ok"] == []


def test_polylogued_browser_capture_help_lists_service_commands() -> None:
    result = CliRunner().invoke(main, ["browser-capture", "--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "status" in result.output
    assert "token" in result.output


def test_polylogued_run_help_lists_allow_no_auth_flag() -> None:
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0
    assert "--browser-capture-allow-no-auth" in result.output


class TestBrowserCaptureReceiverTokenAutoMint:
    """The daemon's browser-capture receiver requires a bearer token by
    default (polylogue-gnie): unless an explicit token or the loud
    ``--browser-capture-allow-no-auth`` opt-out is given, one is
    auto-minted so unauthenticated capture requests are refused."""

    @staticmethod
    def _run_with_captured_make_server_kwargs(**run_kwargs: Any) -> dict[str, object]:
        from polylogue.daemon import cli as daemon_cli

        class FakeServer:
            def serve_forever(self, poll_interval: float = 0.5) -> None:
                raise RuntimeError("server stopped")

            def shutdown(self) -> None:
                pass

            def server_close(self) -> None:
                pass

        captured: dict[str, object] = {}

        def _fake_make_server(*_args: object, **kwargs: object) -> FakeServer:
            captured.update(kwargs)
            return FakeServer()

        with (
            patch.object(daemon_cli, "make_server", side_effect=_fake_make_server),
            pytest.raises(RuntimeError, match="server stopped"),
        ):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=True,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                    **run_kwargs,
                )
            )
        return captured

    def test_default_run_mints_a_receiver_token(self) -> None:
        captured = self._run_with_captured_make_server_kwargs()

        token = captured.get("auth_token")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_allow_no_auth_serves_with_no_token(self) -> None:
        captured = self._run_with_captured_make_server_kwargs(browser_capture_allow_no_auth=True)

        assert captured.get("auth_token") is None

    def test_explicit_token_wins_over_auto_mint(self) -> None:
        captured = self._run_with_captured_make_server_kwargs(browser_capture_auth_token="operator-set-token")

        assert captured.get("auth_token") == "operator-set-token"


def test_polylogued_run_uses_default_sources() -> None:
    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch("polylogue.daemon.cli.default_sources", return_value=sources) as default_sources,
        patch("polylogue.daemon.cli.asyncio.run") as run,
    ):
        result = CliRunner().invoke(main, ["run", "--no-browser-capture", "--no-api"])

    assert result.exit_code == 0
    assert default_sources.call_count == 1
    assert default_sources.call_args.kwargs["hermes_root"] == Path.home() / ".hermes"
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Starting polylogued (watch=1 source(s)). Ctrl-C to stop." in result.stderr


def test_polylogued_run_rejects_retired_debounce_option() -> None:
    """The dispatcher owns scheduling, so its retired watcher option is absent.

    Anti-vacuity: restoring the Click option makes this parse successfully
    instead of reporting the unknown option before daemon startup.
    """
    result = CliRunner().invoke(main, ["run", "--debounce-s", "0.25", "--help"])

    assert result.exit_code != 0
    assert "No such option '--debounce-s'" in result.output


def test_spool_override_replaces_default_browser_capture_source() -> None:
    from polylogue.daemon import cli as daemon_cli

    default_spool = Path("/tmp/default-browser-capture")
    override_spool = Path("/tmp/override-browser-capture")
    sources = (
        WatchSource(name="codex", root=Path("/tmp/codex")),
        WatchSource(name="browser-capture", root=default_spool, suffixes=(".json",)),
    )

    with patch("polylogue.daemon.cli.default_sources", return_value=sources):
        resolved = daemon_cli._watch_sources_from_roots((), browser_capture_spool_path=override_spool)

    assert resolved == (
        WatchSource(name="codex", root=Path("/tmp/codex")),
        WatchSource(name="browser-capture", root=override_spool, suffixes=(".json",)),
    )


def test_polylogued_run_can_skip_configured_source_catchup() -> None:
    recorded: dict[str, object] = {}

    async def fake_run_daemon_services(**kwargs: object) -> None:
        recorded.update(kwargs)

    with patch("polylogue.daemon.cli.run_daemon_services", side_effect=fake_run_daemon_services):
        result = CliRunner().invoke(
            main,
            [
                "run",
                "--root",
                "/tmp/codex",
                "--no-source-catchup",
                "--no-browser-capture",
                "--no-api",
            ],
        )

    assert result.exit_code == 0, (result.output, result.exception)
    assert recorded["enable_watch"] is True
    assert recorded["enable_source_catchup"] is False
    recorded_sources = recorded["sources"]
    assert isinstance(recorded_sources, tuple)
    roots = {source.root for source in recorded_sources}
    assert Path("/tmp/codex") in roots
    assert {source.name for source in recorded_sources} >= {
        "claude-code",
        "claude-code-todos",
        "codex",
        "codex-state",
        "gemini-cli",
        "hermes",
        "antigravity",
        "browser-capture",
        "inbox",
        "claude-code-hooks",
        "codex-hooks",
        "hermes-hooks",
    }


def test_polylogued_run_rejects_empty_component_set() -> None:
    # All three components default to ON; only when every one is explicitly
    # disabled should `run` refuse to start.
    result = CliRunner().invoke(main, ["run", "--no-watch", "--no-browser-capture", "--no-api"])

    assert result.exit_code != 0
    assert "at least one daemon component must be enabled" in result.output


def test_polylogued_watch_uses_default_sources(workspace_env: dict[str, Path]) -> None:
    runner = CliRunner()
    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    async def fake_run_daemon_services(**kwargs: object) -> None:
        assert kwargs["enable_watch"] is True

    with (
        patch("polylogue.daemon.cli.default_sources", return_value=sources) as default_sources,
        patch("polylogue.daemon.cli.run_daemon_services", side_effect=fake_run_daemon_services) as run_services,
    ):
        result = runner.invoke(main, ["watch"])

    assert result.exit_code == 0
    assert default_sources.call_count == 1
    assert default_sources.call_args.kwargs["hermes_root"] == Path.home() / ".hermes"
    run_services.assert_called_once()
    assert run_services.call_args.kwargs["startup_message"] == "Watching 1 source(s). Ctrl-C to stop."


def test_polylogued_watch_uses_supervised_fair_intake_composition(
    workspace_env: dict[str, Path],
) -> None:
    """The standalone watch command must not bypass the fair-intake service."""
    recorded: dict[str, object] = {}

    async def fake_run_daemon_services(**kwargs: object) -> None:
        recorded.update(kwargs)

    with patch(
        "polylogue.daemon.cli.run_daemon_services",
        side_effect=fake_run_daemon_services,
    ) as run_services:
        result = CliRunner().invoke(main, ["watch"])

    assert result.exit_code == 0
    run_services.assert_called_once()
    assert recorded["enable_watch"] is True
    assert recorded["enable_source_catchup"] is True
    assert recorded["enable_browser_capture"] is False
    assert recorded["enable_api"] is False


def test_polylogued_watch_reports_archive_ownership_conflict_as_click_error(
    workspace_env: dict[str, Path],
) -> None:
    """A competing daemon or maintenance owner must not escape the Click boundary."""
    root = workspace_env["archive_root"]
    with OwnedArchiveLocation.acquire(ArchiveLocation.resolve(root), owner_id="competing-writer"):
        result = CliRunner().invoke(main, ["watch"])

    assert result.exit_code == 1
    assert "Error: watch could not acquire exclusive archive ownership:" in result.output
    assert "archive location already owned" in result.output
    assert "Watching" not in result.output
    assert "Traceback" not in result.output


def test_polylogued_watch_builds_sources_from_roots(workspace_env: dict[str, Path], tmp_path: Path) -> None:
    root_a = tmp_path / "claude-code"
    root_b = tmp_path / "codex"

    typed_default = WatchSource(name="typed-default", root=tmp_path / "typed-default")
    with (
        patch("polylogue.daemon.cli.default_sources", return_value=(typed_default,)),
        patch("polylogue.daemon.cli.asyncio.run") as run,
    ):
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
    assert "Watching" not in result.stderr


def test_drive_source_catchup_skips_when_no_drive_sources(tmp_path: Path) -> None:
    from polylogue.config import Config
    from polylogue.daemon import cli as daemon_cli

    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        db_path=tmp_path / "index.db",
    )

    with (
        patch("polylogue.config.get_config", return_value=config),
        patch("polylogue.services.build_runtime_services") as build_services,
    ):
        changed = asyncio.run(daemon_cli._run_drive_source_catchup_once(_unused_session_profile_callback))

    assert changed == 0
    build_services.assert_not_called()


def test_drive_source_catchup_ingests_configured_drive_source(tmp_path: Path) -> None:
    """Drive hands every parsed id to the daemon's canonical derivation owner.

    Anti-vacuity: restoring the legacy bulk-refresh caller reaches the patched
    function below and fails instead of silently bypassing the composed owner.
    """
    from polylogue.config import Config, Source
    from polylogue.daemon import cli as daemon_cli

    drive_source = Source(name="aistudio", folder="Google AI Studio", path=tmp_path / "drive-cache" / "gemini")
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[drive_source],
        db_path=tmp_path / "index.db",
    )
    events: list[object] = []

    class FakeServices:
        def get_repository(self) -> object:
            events.append("repository")
            return object()

        async def close(self) -> None:
            events.append("close")

    class FakeParser:
        def __init__(self, *, repository: object, archive_root: Path, config: Config, execution: object) -> None:
            from polylogue.daemon.drive_catchup import DriveCatchupExecution

            assert isinstance(execution, DriveCatchupExecution)
            events.append(("parser", repository, archive_root, config))

        async def ingest_sources(
            self,
            *,
            sources: list[Source],
            stage: str,
            parse_records: bool,
            max_pass_seconds: float | None = None,
        ) -> SimpleNamespace:
            events.append(("ingest", sources, stage, parse_records, max_pass_seconds))
            return SimpleNamespace(
                acquire_result=SimpleNamespace(raw_ids=["raw-1"], errors=0),
                parse_result=SimpleNamespace(
                    processed_ids={"session-b", "session-a"},
                    counts={"sessions": 0},
                    time_budget_exceeded=False,
                ),
            )

    async def canonical_callback(session_ids: Sequence[str] | None) -> DerivationReport:
        events.append(("canonical", session_ids))
        return cast(DerivationReport, object())

    with (
        patch("polylogue.config.get_config", return_value=config),
        patch("polylogue.services.build_runtime_services", return_value=FakeServices()) as build_services,
        patch("polylogue.pipeline.services.parsing.ParsingService", FakeParser),
        patch(
            "polylogue.pipeline.services.ingest_batch.refresh_session_insights_bulk",
            side_effect=AssertionError("Drive catch-up bypassed the composed derivation owner"),
        ),
    ):
        changed = asyncio.run(daemon_cli._run_drive_source_catchup_once(canonical_callback))

    assert changed == 2
    build_services.assert_called_once_with(config=config, db_path=config.db_path)
    assert ("ingest", [drive_source], "all", True, daemon_cli._DRIVE_CATCHUP_MAX_PASS_SECONDS) in events
    assert ("canonical", ("session-a", "session-b")) in events
    assert events[-1] == "close"


def test_drive_source_catchup_keeps_session_derivation_failures_nonfatal(tmp_path: Path) -> None:
    """A derived-output failure does not discard completed Drive source work."""
    from polylogue.config import Config, Source
    from polylogue.daemon import cli as daemon_cli

    drive_source = Source(name="aistudio", folder="Google AI Studio", path=tmp_path / "drive-cache" / "gemini")
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[drive_source],
        db_path=tmp_path / "index.db",
    )

    class FakeServices:
        def get_repository(self) -> object:
            return object()

        async def close(self) -> None:
            return None

    class FakeParser:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def ingest_sources(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                acquire_result=SimpleNamespace(raw_ids=["raw-1"], errors=0),
                parse_result=SimpleNamespace(
                    processed_ids={"raw-link-only-session"},
                    counts={"sessions": 0},
                    time_budget_exceeded=False,
                ),
            )

    async def failing_callback(session_ids: Sequence[str] | None) -> DerivationReport:
        assert session_ids == ("raw-link-only-session",)
        raise RuntimeError("synthetic aggregate failure")

    with (
        patch("polylogue.config.get_config", return_value=config),
        patch("polylogue.services.build_runtime_services", return_value=FakeServices()),
        patch("polylogue.pipeline.services.parsing.ParsingService", FakeParser),
        capture() as records,
    ):
        changed = asyncio.run(daemon_cli._run_drive_source_catchup_once(failing_callback))

    assert changed == 1
    failures = [r for r in records if r["event"] == "daemon.drive_catchup.session_profile_failed"]
    assert len(failures) == 1
    assert failures[0]["outcome"] == "degraded"
    assert failures[0]["error_type"] == "RuntimeError"
    # The pass itself still completed, and says so separately.
    assert [r["event"] for r in records if str(r["event"]).startswith("daemon.drive_catchup.pass.")][-1] == (
        "daemon.drive_catchup.pass.ok"
    )


def test_drive_source_catchup_safe_wrapper_logs_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    async def fail_catchup(_callback: object) -> int:
        raise RuntimeError("drive unavailable")

    with (
        patch.object(daemon_cli, "_run_drive_source_catchup_once", fail_catchup),
        capture() as records,
    ):
        changed = asyncio.run(daemon_cli._run_drive_source_catchup_safely(_unused_session_profile_callback))

    assert changed == 0
    failures = [r for r in records if r["event"] == "daemon.drive_catchup.failed"]
    assert len(failures) == 1
    assert failures[0]["outcome"] == "error"
    assert failures[0]["error_type"] == "RuntimeError"
    assert "drive unavailable" in str(failures[0]["error_detail"])


def test_explicit_archive_inbox_root_keeps_import_suffixes(workspace_env: dict[str, Path]) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.live.watcher import INBOX_SOURCE_SUFFIXES

    inbox = workspace_env["archive_root"] / "inbox"
    ordinary = workspace_env["archive_root"] / "ordinary-jsonl-root"

    sources = daemon_cli._watch_sources_from_roots((inbox, ordinary))

    assert next(source for source in sources if source.root == inbox) == WatchSource(
        name="inbox", root=inbox, suffixes=INBOX_SOURCE_SUFFIXES
    )
    assert next(source for source in sources if source.root == ordinary) == WatchSource(
        name="ordinary-jsonl-root",
        root=ordinary,
        suffixes=(".json", ".jsonl", ".ndjson", ".zip"),
    )
    assert {source.name for source in sources} >= {
        "claude-code",
        "claude-code-todos",
        "codex",
        "gemini-cli",
        "hermes",
        "antigravity",
        "browser-capture",
    }


def test_configured_root_does_not_duplicate_typed_default(workspace_env: dict[str, Path]) -> None:
    from polylogue.daemon import cli as daemon_cli

    default_root = next(source.root for source in daemon_cli.default_sources() if source.name == "codex")
    sources = daemon_cli._watch_sources_from_roots((default_root,))

    assert sum(source.root == default_root for source in sources) == 1
    assert next(source for source in sources if source.root == default_root).name == "codex"


def test_default_sources_watch_the_legacy_data_home_inbox(workspace_env: dict[str, Path]) -> None:
    """An archive root moved off the XDG data home leaves an inbox behind it,
    and exports staged there before the move are under no other watch root.

    Anti-vacuity: drop ``_legacy_data_home_inbox_sources`` and the only inbox
    root is the archive one, so a wipe-and-reconverge never reads the older
    inbox at all.
    """
    from polylogue.daemon import cli as daemon_cli

    inbox_roots = {source.root for source in daemon_cli.default_sources() if source.name in {"inbox", "inbox-legacy"}}

    assert inbox_roots == {
        workspace_env["archive_root"] / "inbox",
        workspace_env["data_root"] / "polylogue" / "inbox",
    }


def test_default_sources_name_one_inbox_when_the_archive_lives_in_the_data_home(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default layout has one inbox, and the legacy root must not double it.

    Anti-vacuity: return the legacy source unconditionally and this root is
    watched, scanned, and cursor-tracked twice under one name.
    """
    from polylogue.daemon import cli as daemon_cli

    data_home = workspace_env["data_root"] / "polylogue"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(data_home))

    inbox_roots = [source.root for source in daemon_cli.default_sources() if source.name == "inbox"]

    assert inbox_roots == [data_home / "inbox"]


def test_workspace_env_isolates_typed_default_source_roots(workspace_env: dict[str, Path]) -> None:
    from polylogue.daemon import cli as daemon_cli

    roots_by_name = {source.name: source.root for source in daemon_cli.default_sources()}
    home_dir = workspace_env["home_dir"]

    assert roots_by_name["claude-code"] == home_dir / ".claude" / "projects"
    assert roots_by_name["codex"] == home_dir / ".codex" / "sessions"
    assert roots_by_name["codex-state"] == home_dir / ".codex"
    assert roots_by_name["hermes"] == home_dir / ".hermes"


def test_hook_carrier_sources_are_named_apart_but_owned_by_their_provider(workspace_env: dict[str, Path]) -> None:
    """A carrier source never shadows its harness's session source by name.

    Anti-vacuity: name the carrier source after the bare harness again and the
    two ``claude-code`` entries collapse to one in the by-name mapping, so the
    first assertion sees the carriers root; drop the alias and the second one
    resolves the carrier source to no provider.
    """
    from polylogue.core.provider_identity import canonical_runtime_provider
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.live.watcher import HOOK_CARRIER_PROVIDERS

    names = [source.name for source in daemon_cli.default_sources()]
    assert len(names) == len(set(names)), names
    for provider in HOOK_CARRIER_PROVIDERS:
        assert f"{provider}-hooks" in names
        assert canonical_runtime_provider(f"{provider}-hooks") == provider


def test_additional_root_excludes_provider_state_suffixes(workspace_env: dict[str, Path]) -> None:
    from polylogue.daemon import cli as daemon_cli

    root = workspace_env["archive_root"] / "export-root"
    source = next(source for source in daemon_cli._watch_sources_from_roots((root,)) if source.root == root)

    assert source.suffixes == (".json", ".jsonl", ".ndjson", ".zip")
    assert ".db" not in source.suffixes
    assert ".sqlite" not in source.suffixes


def test_explicit_browser_capture_root_keeps_capture_suffixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polylogue.paths as polylogue_paths
    from polylogue.daemon import cli as daemon_cli

    spool = tmp_path / "polylogue" / "browser-capture"
    ordinary = tmp_path / "ordinary-jsonl-root"
    monkeypatch.setattr(polylogue_paths, "browser_capture_spool_root", lambda: spool)

    sources = daemon_cli._watch_sources_from_roots((spool, ordinary))

    assert next(source for source in sources if source.root == spool) == WatchSource(
        name="browser-capture", root=spool, suffixes=(".json",)
    )
    assert next(source for source in sources if source.root == ordinary).suffixes == (
        ".json",
        ".jsonl",
        ".ndjson",
        ".zip",
    )
    assert {source.name for source in sources} >= {
        "claude-code",
        "claude-code-todos",
        "codex",
        "gemini-cli",
        "hermes",
        "antigravity",
        "browser-capture",
    }


def test_explicit_browser_capture_root_uses_spool_override_classifier(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    override_spool = tmp_path / "override-browser-capture"
    ordinary = tmp_path / "ordinary-jsonl-root"

    sources = daemon_cli._watch_sources_from_roots(
        (override_spool, ordinary),
        browser_capture_spool_path=override_spool,
    )

    assert next(source for source in sources if source.root == override_spool) == WatchSource(
        name="browser-capture", root=override_spool, suffixes=(".json",)
    )
    assert next(source for source in sources if source.root == ordinary).suffixes == (
        ".json",
        ".jsonl",
        ".ndjson",
        ".zip",
    )


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


def test_periodic_db_optimize_targets_archive_root_tiers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.maintenance import maybe_optimize_archive_tiers

    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_run_sync(actor: str, func: object, *args: object, **kwargs: object) -> object:
        assert actor == "maintenance.db_optimize"
        calls.append((func, args, kwargs))
        raise asyncio.CancelledError

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch.object(
            daemon_cli,
            "daemon_write_coordinator",
            return_value=SimpleNamespace(run_sync=fake_run_sync),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_db_optimize())

    assert calls == [(maybe_optimize_archive_tiers, (tmp_path,), {"reason": "periodic"})]


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
    from polylogue.core.enums import BlockType, Provider
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path
    with ArchiveStore(archive_root) as archive:
        write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="daemon-heartbeat-v1",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="heartbeat v1",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="heartbeat v1")],
                    )
                ],
            ),
        )

    assert daemon_cli._heartbeat_counts(archive_root / "index.db") == (1, 1, "sessions")


def test_daemon_cli_heartbeat_counts_uses_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakeCursor:
        def __init__(self, row: tuple[object, ...] | None = None, rows: list[tuple[object, ...]] | None = None) -> None:
            self._row = row
            self._rows = rows if rows is not None else ([] if row is None else [row])

        def fetchone(self) -> tuple[object, ...] | None:
            return self._row

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class ReadOnlyConnection:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, sql: str) -> FakeCursor:
            normalized = " ".join(sql.split())
            assert not normalized.upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "PRAGMA JOURNAL_MODE"))
            if "FROM sqlite_master" in normalized:
                return FakeCursor(rows=[("sessions",), ("messages",)])
            if normalized == "SELECT COUNT(*) FROM sessions":
                return FakeCursor((7,))
            if normalized == "SELECT COUNT(*) FROM messages":
                return FakeCursor((42,))
            raise AssertionError(f"unexpected heartbeat query: {normalized}")

        def close(self) -> None:
            self.closed = True

    conn = ReadOnlyConnection()

    def open_readonly(path: Path, *, timeout: float) -> ReadOnlyConnection:
        assert path == tmp_path / "index.db"
        assert timeout == 5.0
        return conn

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_readonly_connection", open_readonly)

    assert daemon_cli._heartbeat_counts(tmp_path / "index.db") == (7, 42, "sessions")
    assert conn.closed is True


def test_reconcile_blob_publications_clears_terminal_receipts_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Startup reconciliation must run under real writer exclusion (polylogue-qs0a).

    Without a live ``ArchiveWriterExclusion``, ``reconcile_blob_publication_reservations``'s
    ``may_clear`` gate is always false and every classified row is only ever
    retained -- a durable reservation leak. This proves the daemon startup
    call actually acquires the exclusion and clears the missing-bytes terminal
    bucket while retaining the referenced receipt for explicit abandonment.
    """
    from polylogue.core.enums import Origin
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref

    archive_root_path = tmp_path / "archive"
    initialize_active_archive_root(archive_root_path)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root_path))

    source_db = archive_root_path / "source.db"
    store = BlobStore(archive_root_path / "blob")
    publisher = ArchiveBlobPublisher(source_db, store.root)
    missing_hash, _ = publisher.write_from_bytes(b"startup-missing-terminal")
    referenced_hash, referenced_size = publisher.write_from_bytes(b"startup-referenced-terminal")
    publisher.flush()
    store.blob_path(missing_hash).unlink()
    with sqlite3.connect(source_db) as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin=Origin.CHATGPT_EXPORT,
            source_path="startup-referenced.json",
            source_index=0,
            blob_hash=bytes.fromhex(referenced_hash),
            blob_size=referenced_size,
            acquired_at_ms=1,
            raw_id="startup-referenced-raw",
        )
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 2

    asyncio.run(daemon_cli._reconcile_blob_publications())

    with sqlite3.connect(source_db) as conn:
        # A live referenced blob is retained for explicit abandonment; only
        # the missing-bytes terminal receipt is safe to clear automatically.
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1


def test_daemon_rebuild_lease_refusal_precedes_startup_blob_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An offline rebuild must refuse the daemon before durable startup writes."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root_path = tmp_path / "archive"
    initialize_active_archive_root(archive_root_path)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root_path))

    source_db = archive_root_path / "source.db"
    publisher = ArchiveBlobPublisher(source_db, BlobStore(archive_root_path / "blob").root)
    publisher.write_from_bytes(b"startup-rebuild-refusal")
    publisher.flush()
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1

    def archive_digest() -> str:
        digest = hashlib.sha256()
        for path in sorted(archive_root_path.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(archive_root_path).as_posix().encode()
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    with RebuildLease(archive_root_path):
        before = archive_digest()
        with pytest.raises(
            RebuildLeaseUnavailableError,
            match="offline index rebuild owns archive",
        ):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                )
            )
        assert archive_digest() == before

    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1
    assert not (archive_root_path / "daemon.pid").exists()


def test_run_daemon_services_stops_live_watcher_on_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    async def noop() -> None:
        return None

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    stopped: list[bool] = []

    class FakeWatcher(_NoIntakeHints):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            stopped.append(True)

    with (
        patch.object(daemon_cli, "Polylogue", FakePolylogue),
        patch.object(daemon_cli, "LiveWatcher", FakeWatcher),
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
        pytest.raises(RuntimeError, match="watch stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert stopped == [True]


def test_run_daemon_services_parks_operation_recovery_on_audit_schema_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-39pdi: a version-mismatched audit.db must not crash startup.

    Before the fix, startup called ``recover_interrupted_operations``
    unconditionally, even when audit.db (an ingestion-blocking durable tier)
    is missing/version-mismatched -- letting a ``sqlite3.Error`` escape into
    the ``BaseException`` shutdown path and kill the daemon instead of
    leaving it degraded.

    Anti-vacuity: removing the ``durable_schema_mismatch`` gate around the
    recovery call (reverting to the unconditional call) makes this test fail
    -- the patched ``recover_interrupted_operations`` gets called at least
    once instead of staying at zero calls.

    A durable-tier mismatch also blocks the live watcher itself
    (``watcher_creation_blocked``), so startup idles waiting on signals
    rather than reaching any watcher/browser-capture/API work; this is
    bounded with ``asyncio.wait_for`` and cancelled rather than run to
    completion.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root_path = tmp_path / "archive"
    initialize_active_archive_root(archive_root_path)
    with sqlite3.connect(archive_root_path / "audit.db") as conn:
        conn.execute("PRAGMA user_version = 1")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root_path))

    recover_mock = Mock()

    with (
        patch(
            "polylogue.operations.mutation_transaction.recover_interrupted_operations",
            recover_mock,
        ),
        pytest.raises(TimeoutError),
    ):
        asyncio.run(
            asyncio.wait_for(
                daemon_cli.run_daemon_services(
                    sources=(WatchSource(name="codex", root=archive_root_path),),
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                ),
                timeout=5.0,
            )
        )

    recover_mock.assert_not_called()


def test_daemon_cleanup_failure_retains_rebuild_exclusion_until_process_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup errors before coordinator shutdown must never reopen rebuilds."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.maintenance import raw_authority
    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError

    async def noop() -> None:
        return None

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class FakeWatcher(_NoIntakeHints):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            return None

    captured: list[raw_authority.ArchiveWriterRebuildExclusion] = []
    archive_roots: list[Path] = []
    real_exclusion = raw_authority.archive_writer_rebuild_exclusion

    @contextlib.contextmanager
    def capture_exclusion(archive_root: Path) -> Iterator[raw_authority.ArchiveWriterRebuildExclusion]:
        archive_roots.append(archive_root)
        with real_exclusion(archive_root) as exclusion:
            captured.append(exclusion)
            yield exclusion

    def fail_shutdown_marker() -> None:
        raise RuntimeError("shutdown marker failed")

    monkeypatch.setattr(raw_authority, "archive_writer_rebuild_exclusion", capture_exclusion)
    with (
        patch.object(daemon_cli, "Polylogue", FakePolylogue),
        patch.object(daemon_cli, "LiveWatcher", FakeWatcher),
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
        patch.object(
            daemon_cli,
            "_mark_interrupted_live_ingest_attempts_on_shutdown",
            fail_shutdown_marker,
        ),
        pytest.raises(RuntimeError, match="shutdown marker failed"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert len(captured) == 1
    assert len(archive_roots) == 1
    with pytest.raises(RebuildLeaseUnavailableError, match="index rebuild lease is already held"):
        with RebuildLease(archive_roots[0]):
            pass

    captured[0].release()
    with RebuildLease(archive_roots[0]):
        pass


def test_lifecycle_heartbeat_runs_without_index_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """The degraded daemon heartbeat must not depend on index.db existing."""
    from polylogue.daemon import cli as daemon_cli

    calls: list[str] = []
    actors: list[str] = []

    class Lifecycle:
        def heartbeat(self) -> None:
            calls.append("heartbeat")

    class Coordinator:
        async def run_sync(self, actor: str, function: object, /, *args: object, **kwargs: object) -> object:
            actors.append(actor)
            assert callable(function)
            return function(*args, **kwargs)

    async def exercise() -> None:
        monkeypatch.setattr(daemon_cli, "_daemon_lifecycle", Lifecycle())
        monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: Coordinator())
        task = asyncio.create_task(daemon_cli._periodic_lifecycle_heartbeat(interval_s=0))
        try:
            for _ in range(5):
                await asyncio.sleep(0)
                if calls:
                    break
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(exercise())

    assert calls
    assert actors == ["daemon.lifecycle.heartbeat"]


def test_lifecycle_start_failure_releases_pidfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed forensic start must not strand the daemon's mutual-exclusion lock."""
    from polylogue.daemon import cli as daemon_cli

    class Coordinator:
        async def run_sync(self, _actor: str, _function: object, /, *args: object, **kwargs: object) -> object:
            raise RuntimeError("ops unavailable")

        async def shutdown(self, *, timeout: float) -> bool:
            assert timeout == 5.0
            return True

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.storage.archive_identity.resolve_active_index_path", lambda *_a, **_k: tmp_path / "index.db"
    )
    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: Coordinator())

    with pytest.raises(RuntimeError, match="ops unavailable"):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert not (tmp_path / "daemon.pid").exists()


def test_daemon_startup_reconciles_trains_before_schema_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    events: list[str] = []

    class Coordinator:
        async def run_sync(self, actor: str, _function: object, /, *args: object, **kwargs: object) -> object:
            del args, kwargs
            if actor == "daemon.lifecycle.start":
                raise RuntimeError("startup stopped")
            return None

        async def shutdown(self, *, timeout: float) -> bool:
            assert timeout == 5.0
            return True

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.storage.archive_identity.resolve_active_index_path", lambda *_a, **_k: tmp_path / "index.db"
    )
    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: Coordinator())

    def reconcile(root: Path) -> tuple[Path, ...]:
        events.append(f"reconcile:{root}")
        return (tmp_path / "recovered.json",)

    def schema_ok() -> HealthAlert:
        events.append("schema")
        return HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=HealthSeverity.OK,
            message="ok",
            checked_at="now",
        )

    monkeypatch.setattr(
        "polylogue.operations.durable_change_train.reconcile_durable_change_trains_on_startup", reconcile
    )
    monkeypatch.setattr(
        daemon_cli,
        "_check_schema_version_fast",
        schema_ok,
    )

    with pytest.raises(RuntimeError, match="startup stopped"):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert events == [f"reconcile:{tmp_path}", "schema"]
    assert (tmp_path / ".archive-ownership.lock").exists()
    assert not (tmp_path / "daemon.pid").exists()


def test_daemon_startup_creates_missing_archive_root_before_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live daemon route must own a first-run root, not require a prior bootstrap."""
    from polylogue.daemon import cli as daemon_cli

    archive = tmp_path / "first-run-archive"
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive)

    def stop_after_ownership(root: Path) -> tuple[Path, ...]:
        assert root == archive
        assert archive.is_dir()
        raise RuntimeError("owned first-run archive")

    monkeypatch.setattr(
        "polylogue.operations.durable_change_train.reconcile_durable_change_trains_on_startup",
        stop_after_ownership,
    )

    previous_umask = os.umask(0)
    try:
        with pytest.raises(RuntimeError, match="owned first-run archive"):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                )
            )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(archive.stat().st_mode) == 0o700
    assert (archive / ".archive-ownership.lock").exists()


def test_run_daemon_services_checks_archive_identity_before_component_startup(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.archive_identity import ArchiveIdentityConflictError

    configure = Mock()
    with (
        patch("polylogue.paths.archive_root", return_value=tmp_path / "configured"),
        patch(
            "polylogue.storage.archive_identity.assert_writable_archive_identity",
            side_effect=ArchiveIdentityConflictError("split root"),
        ),
        patch("polylogue.daemon.status_snapshot.configure_runtime_components", configure),
        pytest.raises(ArchiveIdentityConflictError, match="split root"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    configure.assert_not_called()


def test_emit_daemon_lifecycle_event_has_no_dev_loop_launcher_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    calls: list[tuple[str, dict[str, object]]] = []
    actors: list[str] = []

    def fake_emit(kind: str, **kwargs: object) -> None:
        calls.append((kind, kwargs))

    class FakeCoordinator:
        async def run_sync(self, actor: str, function: Any, /, *args: object, **kwargs: object) -> None:
            actors.append(actor)
            function(*args, **kwargs)

    with (
        patch("polylogue.daemon.events.emit_daemon_event", side_effect=fake_emit),
        patch.object(daemon_cli, "daemon_write_coordinator", return_value=FakeCoordinator()),
    ):
        asyncio.run(
            daemon_cli._emit_daemon_lifecycle_event(
                "component_started",
                archive_root_path=tmp_path / "archive",
                component="api",
                payload={"port": 8766},
            )
        )

    assert len(calls) == 1
    assert actors == ["daemon.lifecycle.component_started"]
    kind, kwargs = calls[0]
    assert kind == "daemon.lifecycle"
    assert kwargs["operation_id"] is None
    payload = cast(dict[str, object], kwargs["payload"])
    assert payload["phase"] == "component_started"
    assert payload["component"] == "api"
    assert payload["port"] == 8766
    assert payload["archive_root"] == str(tmp_path / "archive")


def test_pidfile_remains_locked_until_admitted_writers_are_drained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    pidfile = tmp_path / "daemon.pid"
    owner_fd = daemon_cli._acquire_pidfile(pidfile)
    monkeypatch.setattr(daemon_cli, "_pidfile_path", pidfile)

    retained_fd = daemon_cli._release_pidfile_after_writer_drain(owner_fd, writer_drained=False)

    assert retained_fd == owner_fd
    with pytest.raises(RuntimeError, match="another daemon may be running"):
        daemon_cli._acquire_pidfile(pidfile)

    assert daemon_cli._release_pidfile_after_writer_drain(retained_fd, writer_drained=True) is None
    successor_fd = daemon_cli._acquire_pidfile(pidfile)
    os.close(successor_fd)


def test_rebuild_exclusion_survives_an_undrained_writer_timeout(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.maintenance.raw_authority import archive_writer_rebuild_exclusion
    from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError

    archive_root_path = tmp_path / "archive"
    archive_root_path.mkdir()
    with archive_writer_rebuild_exclusion(archive_root_path) as exclusion:
        daemon_cli._retain_rebuild_exclusion_for_undrained_writer(
            exclusion,
            writer_drained=False,
        )

    with pytest.raises(RebuildLeaseUnavailableError, match="index rebuild lease is already held"):
        with RebuildLease(archive_root_path):
            pass

    exclusion.release()
    with RebuildLease(archive_root_path):
        pass


def test_shutdown_lifecycle_event_is_bounded_when_writer_gate_is_stuck(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    class StuckCoordinator:
        async def run_sync(self, *_args: object, **_kwargs: object) -> None:
            await asyncio.Event().wait()

    async def exercise() -> None:
        with patch.object(daemon_cli, "daemon_write_coordinator", return_value=StuckCoordinator()):
            await asyncio.wait_for(
                daemon_cli._emit_daemon_lifecycle_event(
                    "shutdown_started",
                    archive_root_path=tmp_path,
                    status="stopping",
                ),
                timeout=0.75,
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("configured_drive", [False, True])
def test_run_daemon_services_waits_for_fts_startup_before_watcher(tmp_path: Path, configured_drive: bool) -> None:
    from polylogue.config import Source
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.execution import BoundedComputeAdapter, reset_daemon_compute_adapter
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    events: list[str] = []
    lifecycle_payloads: list[dict[str, object]] = []
    watcher_coordinators: list[object] = []
    watcher_profile_callbacks: list[object] = []
    periodic_profile_callbacks: list[object] = []
    drive_called = asyncio.Event()
    ok_schema = HealthAlert(
        check_name="schema_version",
        tier=HealthTier.FAST,
        severity=HealthSeverity.OK,
        message="ok",
        checked_at="now",
    )

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class FakeWatcher(_NoIntakeHints):
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.watcher_ready = asyncio.Event()
            watcher_coordinators.append(kwargs["write_coordinator"])
            watcher_profile_callbacks.append(kwargs["session_profile_callback"])

        async def run(self) -> None:
            events.append("watcher")
            self.watcher_ready.set()
            if configured_drive:
                await asyncio.wait_for(drive_called.wait(), 10)
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            events.append("stop")

    async def fake_fts_owner_run(self: object, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        events.append("fts")
        return SimpleNamespace(failed=0)

    def fake_embedding_lifecycle_startup(_archive_root_path: Path) -> Path:
        events.append("embedding-lifecycle")
        return tmp_path / "embeddings.db"

    def fake_lineage_startup() -> int:
        events.append("lineage")
        return 0

    async def fake_reconcile_blob_publications() -> None:
        events.append("blob-publications")

    async def fake_drive_catchup(callback: object) -> int:
        from polylogue.storage.sqlite.write_lease import current_write_lease

        assert current_write_lease() is None
        assert callback is api_server.session_profile_callback
        events.append("drive-once")
        drive_called.set()
        return 0

    async def fake_configure_fts_automerge() -> None:
        events.append("automerge")

    def fake_operation_recovery(_archive_root_path: Path) -> None:
        events.append("operation-recovery")

    def recording_converger(
        stages: Iterable[ConvergenceStage], *, derivations: Iterable[object] = ()
    ) -> DaemonConverger:
        events.append("converger")
        return DaemonConverger(stages, derivations=derivations)

    async def fake_loop(name: str) -> None:
        events.append(name)
        await asyncio.Event().wait()

    class FakeAPIServer:
        def __init__(self) -> None:
            self.stopped = threading.Event()
            self.session_profile_callback = object()
            self.execution_kernel = BoundedComputeAdapter(
                max_workers=1,
                queue_units=0,
                thread_name_prefix="test-daemon-api",
            )
            self.operation_runtime = SimpleNamespace(shutdown=self._shutdown_operation_runtime)

        async def _shutdown_operation_runtime(self) -> None:
            events.append("operation-runtime-shutdown")

        def serve_forever(self, _poll_interval: float) -> None:
            self.stopped.wait(timeout=2.0)

        def shutdown(self) -> None:
            self.stopped.set()

        def server_close(self) -> None:
            self.execution_kernel.shutdown(wait=False, cancel_futures=True)
            return None

    reset_daemon_compute_adapter()
    api_server = FakeAPIServer()

    def make_api_server(*_args: object, **_kwargs: object) -> FakeAPIServer:
        events.append("api-bind")
        return api_server

    api_server_factory = Mock(side_effect=make_api_server)

    def fake_emit_daemon_event(kind: str, **kwargs: object) -> None:
        assert kind == "daemon.lifecycle"
        lifecycle_payloads.append(cast(dict[str, object], kwargs["payload"]))

    with contextlib.ExitStack() as stack:
        stack.callback(reset_daemon_compute_adapter)
        stack.enter_context(
            patch(
                "polylogue.config.get_config",
                return_value=Config(
                    archive_root=tmp_path,
                    render_root=tmp_path / "render",
                    db_path=tmp_path / "index.db",
                    sources=[Source(name="gemini", folder="fixture")] if configured_drive else [],
                ),
            )
        )
        stack.enter_context(patch.object(daemon_cli, "Polylogue", FakePolylogue))
        stack.enter_context(patch.object(daemon_cli, "LiveWatcher", FakeWatcher))
        stack.enter_context(
            patch.object(daemon_cli, "_ensure_embedding_lifecycle_startup_sync", fake_embedding_lifecycle_startup)
        )
        stack.enter_context(patch("polylogue.daemon.fts_convergence.FtsConvergenceOwner.converge", fake_fts_owner_run))
        stack.enter_context(patch.object(daemon_cli, "_ensure_lineage_startup_readiness_sync", fake_lineage_startup))
        stack.enter_context(patch.object(daemon_cli, "_reconcile_blob_publications", fake_reconcile_blob_publications))
        stack.enter_context(patch.object(daemon_cli, "_check_schema_version_fast", return_value=ok_schema))
        stack.enter_context(patch("polylogue.paths.archive_root", return_value=tmp_path))
        stack.enter_context(patch.object(daemon_cli, "_run_drive_source_catchup_safely", fake_drive_catchup))
        stack.enter_context(patch.object(daemon_cli, "_configure_fts_automerge", fake_configure_fts_automerge))
        stack.enter_context(
            patch("polylogue.operations.mutation_transaction.recover_interrupted_operations", fake_operation_recovery)
        )
        stack.enter_context(patch.object(daemon_cli, "_periodic_wal_checkpoint", lambda: fake_loop("wal")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_fts_merge", lambda: fake_loop("fts-merge")))
        stack.enter_context(
            patch(
                "polylogue.daemon.blob_gc_periodic.periodic_blob_gc_check",
                lambda **_kwargs: fake_loop("blob-gc"),
            )
        )
        stack.enter_context(
            patch(
                "polylogue.daemon.blob_gc_periodic.periodic_blob_publication_reconciliation_check",
                lambda **_kwargs: fake_loop("blob-publication-reconciliation"),
            )
        )
        stack.enter_context(
            patch.object(
                daemon_cli,
                "_periodic_raw_materialization_convergence",
                lambda **_kwargs: fake_loop("raw-observation"),
            )
        )
        stack.enter_context(patch.object(daemon_cli, "_periodic_heartbeat", lambda: fake_loop("heartbeat")))

        def fake_periodic_convergence(_sources: tuple[WatchSource, ...], **kwargs: object) -> object:
            periodic_profile_callbacks.append(kwargs["session_profile_callback"])
            return fake_loop("convergence")

        stack.enter_context(patch.object(daemon_cli, "_periodic_convergence_check", fake_periodic_convergence))
        stack.enter_context(patch.object(daemon_cli, "_periodic_health_check", lambda: fake_loop("health")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_db_optimize", lambda: fake_loop("optimize")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_status_snapshot_refresh", lambda: fake_loop("status")))
        stack.enter_context(
            patch.object(daemon_cli, "_periodic_drive_source_catchup", lambda **_kwargs: fake_loop("drive"))
        )
        stack.enter_context(
            patch(
                "polylogue.daemon.embedding_backlog.periodic_embedding_backlog_check",
                lambda **_kwargs: fake_loop("embedding"),
            )
        )
        stack.enter_context(
            patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=())
        )
        stack.enter_context(patch("polylogue.daemon.convergence.DaemonConverger", recording_converger))
        stack.enter_context(patch("polylogue.daemon.http.DaemonAPIHTTPServer", api_server_factory))
        stack.enter_context(patch("polylogue.daemon.events.emit_daemon_event", side_effect=fake_emit_daemon_event))
        stack.enter_context(patch.object(daemon_cli, "_mark_interrupted_live_ingest_attempts_on_shutdown"))
        stack.enter_context(pytest.raises(RuntimeError, match="watch stopped"))
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                enable_api=True,
            )
        )

    assert "watcher" in events
    assert events.index("embedding-lifecycle") < events.index("api-bind")
    assert events.index("embedding-lifecycle") < events.index("fts") < events.index("watcher")
    assert events.index("operation-recovery") < events.index("watcher")
    assert events.index("fts") < events.index("watcher")
    assert events.index("fts") < events.index("lineage") < events.index("watcher")
    assert events.index("lineage") < events.index("blob-publications") < events.index("watcher")
    assert events.index("fts") < events.index("raw-observation") < events.index("watcher")
    assert "blob-gc" in events
    assert "blob-publication-reconciliation" in events
    # Source acquisition belongs to fair intake after startup readiness.
    if configured_drive:
        assert events.index("lineage") < events.index("drive-once")
    else:
        assert "drive-once" not in events
    assert events.index("lineage") < events.index("convergence")
    assert "raw-observation" in events
    assert "drive" not in events
    assert events.index("converger") < events.index("watcher")
    assert events.count("convergence") == 1
    lifecycle_phases = [str(payload["phase"]) for payload in lifecycle_payloads]
    assert lifecycle_phases[0] == "startup"
    assert "component_ready" in lifecycle_phases
    assert lifecycle_phases[-1] == "shutdown_started"
    assert "shutdown_complete" not in lifecycle_phases
    lifecycle_components = {payload.get("component") for payload in lifecycle_payloads}
    assert {"embedding_lifecycle_startup", "fts", "lineage_startup", "converger", "intake"}.issubset(
        lifecycle_components
    )
    assert len(watcher_coordinators) == 1
    bridge = api_server_factory.call_args.kwargs["write_bridge"]
    assert bridge._coordinator is watcher_coordinators[0]
    assert watcher_profile_callbacks == [api_server.session_profile_callback]
    assert periodic_profile_callbacks == [api_server.session_profile_callback]


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("drives the real filesystem watcher through a controlled daemon lifecycle")
async def test_daemon_startup_catch_up_and_restart_repair_session_profiles(tmp_path: Path) -> None:
    """A real daemon lifecycle restores profiles from configured-source evidence.

    Each pass enters ``run_daemon_services`` with the production watcher and
    composition callback. The first pass catches up a physical JSONL source;
    the second starts after a synthetic, output-only profile removal. Both
    passes terminate only after the real periodic convergence loop invokes
    its post-catch-up no-hint callback. No manual operation invokes the owner.

    Anti-vacuity: omit the watcher callback, run the sweep before catch-up,
    replace it with a scoped live-source call, or retain output rows across
    restart, and the recorded ``None`` scope or repaired durable profile fails.
    """
    from polylogue import Polylogue as RealPolylogue
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import session_profile_composition
    from polylogue.daemon.execution import reset_daemon_compute_adapter
    from polylogue.daemon.services import ServiceProfile
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    archive_root = tmp_path / "archive"
    source_root = tmp_path / "configured-source"
    source_root.mkdir()
    native_session_id = "aaaa0000-0000-0000-0000-000000000001"
    session_id = f"claude-code-session:{native_session_id}"
    source = source_root / "fresh-session.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "parentUuid": None,
                    "sessionId": native_session_id,
                    "type": "user",
                    "message": {"role": "user", "content": "synthetic startup prompt"},
                    "uuid": "message-1",
                    "timestamp": "2026-05-16T00:00:00.000Z",
                    "cwd": "/workspace",
                    "version": "1.0.6",
                    "isSidechain": False,
                    "userType": "external",
                },
                {
                    "parentUuid": "message-1",
                    "sessionId": native_session_id,
                    "type": "assistant",
                    "message": {"role": "assistant", "content": "synthetic startup reply"},
                    "uuid": "message-2",
                    "timestamp": "2026-05-16T00:00:01.000Z",
                    "cwd": "/workspace",
                    "version": "1.0.6",
                    "isSidechain": False,
                    "userType": "external",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(source, (1.0, 1.0))

    observed_scopes: list[tuple[str, ...] | None] = []
    sweep_complete = asyncio.Event()
    current_coordinator: DaemonWriteCoordinator | None = None
    real_compose = session_profile_composition.compose_session_profile_callback

    async def idle_loop(**_kwargs: object) -> None:
        await asyncio.Event().wait()

    async def noop_periodic_work(*_args: object, **_kwargs: object) -> None:
        return None

    def compose_with_oracle(
        root: Path,
        *,
        compute_adapter: BoundedComputeAdapter,
        write_bridge: DaemonWriteThreadBridge,
        now: Callable[[], float],
    ) -> ComposedSessionProfiles:
        composed = real_compose(
            root,
            compute_adapter=compute_adapter,
            write_bridge=write_bridge,
            now=now,
        )

        async def observe(scope: Sequence[str] | None) -> DerivationReport:
            report = await composed(scope)
            if scope is None and profile_exists():
                observed_scopes.append(scope)
                sweep_complete.set()
            return report

        return session_profile_composition.ComposedSessionProfiles(observe, composed.maintenance)

    def daemon_coordinator() -> DaemonWriteCoordinator:
        assert current_coordinator is not None
        return current_coordinator

    def profile_exists() -> bool:
        # A read, so it opens read-only: the daemon under test holds the
        # process-wide writer boundary and a writable open here would be a
        # second in-process writer (polylogue-8qm4k).
        with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'session_profiles'").fetchone():
                return False
            return (
                conn.execute("SELECT 1 FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()
                is not None
            )

    async def run_until_observed_sweep() -> None:
        nonlocal current_coordinator
        current_coordinator = DaemonWriteCoordinator(archive_root=archive_root)
        sweep_complete.clear()
        task = asyncio.create_task(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="configured", root=source_root),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                enable_api=False,
                service_profile=ServiceProfile.PRODUCTION,
            )
        )
        try:
            await asyncio.wait_for(sweep_complete.wait(), timeout=20.0)
            assert profile_exists()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10.0)

    try:
        with contextlib.ExitStack() as stack:
            _daemon_startup_stubs(stack, daemon_cli, archive_root)
            stack.enter_context(patch.object(daemon_cli, "Polylogue", lambda: RealPolylogue(archive_root=archive_root)))
            stack.enter_context(patch.object(daemon_cli, "daemon_write_coordinator", daemon_coordinator))
            stack.enter_context(
                patch.object(session_profile_composition, "compose_session_profile_callback", compose_with_oracle)
            )
            stack.enter_context(patch.object(daemon_cli, "_retry_convergence_debt_once", noop_periodic_work))
            stack.enter_context(patch.object(daemon_cli, "_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS", 0.05))
            for attribute in (
                "_periodic_lifecycle_heartbeat",
                "_periodic_health_check",
                "_periodic_wal_checkpoint",
                "_periodic_fts_merge",
                "_periodic_heartbeat",
                "_periodic_db_optimize",
                "_periodic_status_snapshot_refresh",
                "_periodic_raw_materialization_convergence",
                "_periodic_drive_source_catchup",
            ):
                stack.enter_context(patch.object(daemon_cli, attribute, idle_loop))
            for target in (
                "polylogue.daemon.embedding_backlog.periodic_embedding_backlog_check",
                "polylogue.daemon.embedding_backlog.periodic_embedding_orphan_reconcile_check",
                "polylogue.daemon.judgment_automation.periodic_judgment_automation_sweep",
                "polylogue.daemon.blob_gc_periodic.periodic_blob_gc_check",
                "polylogue.daemon.blob_gc_periodic.periodic_blob_publication_reconciliation_check",
                "polylogue.daemon.secret_scan_sweep.periodic_secret_scan_sweep",
            ):
                stack.enter_context(patch(target, idle_loop))

            await run_until_observed_sweep()
            # This fixture really does write the archive out from under the
            # running daemon, to force reconvergence. It is a declared test
            # authority, not a production route, so it says so.
            from polylogue.storage.sqlite.write_lease import declared_unguarded_write

            with (
                declared_unguarded_write("test fixture clears derived rows to force reconvergence"),
                sqlite3.connect(archive_root / "index.db") as conn,
            ):
                assert conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                for table in ("session_latency_profiles", "session_profiles"):
                    conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                conn.commit()
            assert not profile_exists()

            await run_until_observed_sweep()
    finally:
        reset_daemon_compute_adapter()

    assert observed_scopes == [None, None]
    assert profile_exists()


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("runs supervised intake and threaded derivation with controlled filesystem events")
@pytest.mark.parametrize("source_name", ["configured", "browser-capture"])
@pytest.mark.parametrize("directory_event", [False, True])
async def test_daemon_watcher_hints_wake_fair_intake_and_canonical_derivation(
    tmp_path: Path, source_name: str, directory_event: bool
) -> None:
    """Only fair intake admits created/updated sources and derives their profiles.

    Anti-vacuity: restore watcher debounce admission, omit the wake connection,
    or bypass the canonical profile kernel, and task ownership, bounded wake,
    or the persisted profile fails. JSONL prefix growth derives a second profile;
    a competing browser snapshot retains the canonical raw frontier deferral.
    Other maintenance cannot repair the output.
    """
    from watchfiles import Change

    from polylogue import Polylogue as RealPolylogue
    from polylogue.archive.revision_authority import RawRevisionAuthority
    from polylogue.archive.session_revision_membership import MembershipDecision
    from polylogue.browser_capture.models import BrowserCaptureEnvelope
    from polylogue.browser_capture.receiver import write_capture_envelope
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.execution import reset_daemon_compute_adapter
    from polylogue.daemon.intake import FairIntakeDispatcher
    from polylogue.daemon.intake_adapters import DaemonIntakeService
    from polylogue.daemon.supervisor import DaemonSupervisor
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.sources.live.watcher import LiveWatcher

    archive_root = tmp_path / "archive"
    source_root = tmp_path / "source"
    source_root.mkdir()
    first_pass = asyncio.Event()
    first_admission = asyncio.Event()
    completed = asyncio.Event()
    intake_tasks: list[object] = []
    admission_tasks: list[object] = []
    admission_metrics: list[object] = []
    kernel_reports: list[DerivationReport] = []
    passes: list[object] = []
    carriers: list[Path] = []
    browser = source_name == "browser-capture"
    native_id = "aaaa0000-0000-0000-0000-000000000001"
    session_id = "chatgpt-export:intake-law" if browser else f"claude-code-session:{native_id}"
    real_start = DaemonSupervisor.start
    real_pass = FairIntakeDispatcher.run_once
    real_ingest = LiveWatcher._ingest_files
    real_kernel = DaemonConverger.converge_derivations

    async def idle() -> None:
        await asyncio.Event().wait()

    def start(self: DaemonSupervisor, name: str, factory: Any, **kwargs: Any) -> Any:
        return real_start(self, name, factory if name in {"watcher", "fair_intake"} else idle, **kwargs)

    async def dispatch(self: FairIntakeDispatcher, **kwargs: Any) -> Any:
        intake_tasks.append(asyncio.current_task())
        result = await real_pass(self, **kwargs)
        passes.append(result)
        first_pass.set()
        if result.progressed:
            if first_admission.is_set():
                completed.set()
            else:
                first_admission.set()
        return result

    async def ingest(self: LiveWatcher, *args: Any, **kwargs: Any) -> Any:
        admission_tasks.append(asyncio.current_task())
        metrics = await real_ingest(self, *args, **kwargs)
        admission_metrics.append(metrics)
        return metrics

    def kernel(self: DaemonConverger, *args: Any, **kwargs: Any) -> DerivationReport:
        report = real_kernel(self, *args, **kwargs)
        kernel_reports.append(report)
        return report

    def claude_record(role: str, uuid: str, parent: str | None, text: str) -> str:
        return (
            json.dumps(
                {
                    "parentUuid": parent,
                    "sessionId": native_id,
                    "type": role,
                    "message": {"role": role, "content": text},
                    "uuid": uuid,
                    "timestamp": "2026-04-24T00:00:00.000Z",
                    "cwd": "/workspace",
                    "version": "1.0.6",
                    "isSidechain": False,
                    "userType": "external",
                }
            )
            + "\n"
        )

    async def watch_events(*_args: Any, **_kwargs: Any) -> Any:
        await first_pass.wait()
        envelope = BrowserCaptureEnvelope.model_validate(
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "capture_id": "chatgpt:intake-law",
                "provenance": {
                    "source_url": "https://chatgpt.com/c/intake-law",
                    "page_title": "Intake law",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                    "capture_mode": "snapshot",
                },
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "intake-law",
                    "title": "Intake law",
                    "turns": [{"provider_turn_id": "u1", "role": "user", "text": "Synthetic", "ordinal": 0}],
                },
            }
        )
        spool = source_root / "new-directory" if directory_event else source_root
        if browser:
            artifact = write_capture_envelope(envelope, spool_path=spool).path
        else:
            spool.mkdir(exist_ok=True)
            artifact = spool / "session.jsonl"
            artifact.write_text(claude_record("user", "u1", None, "Synthetic"), encoding="utf-8")
        os.utime(artifact, (1.0, 1.0))
        carriers.append(artifact)
        yield {(Change.added, str(spool if directory_event else artifact))}
        await first_admission.wait()
        root_mtime = source_root.stat().st_mtime_ns
        if browser:
            payload = envelope.model_dump(mode="json")
            payload["session"]["turns"][0]["text"] = "Competing synthetic revision"
            payload["session"]["turns"].append(
                {"provider_turn_id": "a1", "role": "assistant", "text": "Synthetic reply", "ordinal": 1}
            )
            assert (
                write_capture_envelope(BrowserCaptureEnvelope.model_validate(payload), spool_path=spool).path
                == artifact
            )
        else:
            with artifact.open("a", encoding="utf-8") as stream:
                stream.write(claude_record("assistant", "a1", "u1", "Synthetic reply"))
        os.utime(artifact, (2.0, 2.0))
        assert source_root.stat().st_mtime_ns == root_mtime
        yield {(Change.modified, str(artifact))}
        await idle()

    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    reset_daemon_compute_adapter()
    try:
        with contextlib.ExitStack() as stack:
            _daemon_startup_stubs(stack, daemon_cli, archive_root)
            stack.enter_context(patch.object(daemon_cli, "Polylogue", lambda: RealPolylogue(archive_root=archive_root)))
            stack.enter_context(patch.object(daemon_cli, "daemon_write_coordinator", return_value=coordinator))
            stack.enter_context(patch.object(DaemonSupervisor, "start", start))
            stack.enter_context(patch.object(FairIntakeDispatcher, "run_once", dispatch))
            stack.enter_context(patch.object(LiveWatcher, "_ingest_files", ingest))
            stack.enter_context(patch.object(DaemonConverger, "converge_derivations", kernel))
            stack.enter_context(patch("watchfiles.awatch", watch_events))
            stack.enter_context(
                patch(
                    "polylogue.daemon.intake_adapters.DaemonIntakeService",
                    lambda dispatcher, **kwargs: DaemonIntakeService(
                        dispatcher, budget=4096, idle_delay_s=0.05, **kwargs
                    ),
                )
            )
            task = asyncio.create_task(
                daemon_cli.run_daemon_services(
                    sources=(
                        WatchSource(name=source_name, root=source_root, suffixes=(".json" if browser else ".jsonl",)),
                    ),
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                    enable_api=False,
                    enable_source_catchup=False,
                )
            )
            try:
                try:
                    await asyncio.wait_for(completed.wait(), timeout=20)
                except TimeoutError:
                    if task.done():
                        await task
                    pytest.fail(
                        f"intake did not complete: passes={passes}, carriers={carriers}, admissions={admission_tasks}"
                    )
                assert admission_tasks and all(item is intake_tasks[0] for item in admission_tasks)
                expected_versions = 1 if browser else 2
                with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
                    evidence = conn.execute(
                        "SELECT decision, revision_authority FROM raw_session_memberships WHERE logical_source_key = ?",
                        (session_id,),
                    ).fetchall()
                # polylogue-f7pdm: on a fresh root the raw-materialization
                # intake class used to be latched off for the daemon's whole
                # lifetime, so every kernel report here was a session-profile
                # one. The class now registers and converges the raw
                # observations its own admissions produce, so the session
                # count is taken from the session-scoped reports and the raw
                # ones are asserted separately rather than folded in.
                session_reports = [
                    report for report in kernel_reports if not isinstance(report.frame.scope, RawObservationScope)
                ]
                assert sum(report.done == 1 for report in session_reports) == expected_versions, str(
                    ([(report.frame.scope, report.done) for report in kernel_reports], admission_metrics, evidence)
                )
                with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
                    assert conn.execute("SELECT session_id FROM session_profiles").fetchall() == [(session_id,)]
                    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (expected_versions,)
                if browser:
                    assert (
                        evidence == [(MembershipDecision.AMBIGUOUS.value, RawRevisionAuthority.QUARANTINED.value)] * 2
                    )
                assert all(path.is_file() for path in carriers)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=10)
    finally:
        reset_daemon_compute_adapter()


def test_run_daemon_services_closes_browser_capture_server_on_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    async def noop() -> None:
        return None

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
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
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

    async def noop() -> None:
        return None

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class FakeWatcher(_NoIntakeHints):
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
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
        pytest.raises(RuntimeError, match="watch stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert server.shutdown_called is True
    assert server.close_called is True


def test_shutdown_server_runs_even_when_to_thread_task_is_cancelled() -> None:
    from polylogue.daemon import cli as daemon_cli

    class BlockingServer:
        shutdown_called = False

        def __init__(self) -> None:
            self._stopped = threading.Event()

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            assert poll_interval == 0.5
            self._stopped.wait(timeout=5)

        def shutdown(self) -> None:
            self.shutdown_called = True
            self._stopped.set()

    async def exercise() -> BlockingServer:
        server = BlockingServer()
        task = asyncio.create_task(asyncio.to_thread(server.serve_forever, 0.5))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await daemon_cli._shutdown_server_if_serving(cast(Any, server), task, label="browser-capture")
        return server

    server = asyncio.run(exercise())

    assert server.shutdown_called is True


async def _await_server_readiness_or_daemon_exit(
    task: asyncio.Task[None], servers: tuple[threading.Event, ...]
) -> None:
    """Bound test startup readiness and preserve the daemon's original failure."""
    try:
        async with asyncio.timeout(0.75):
            while not all(server.is_set() for server in servers):
                # A service that dies during startup must surface its failure
                # now, rather than leaving this probe spinning until an
                # external test timeout.
                if task.done():
                    await task
                    raise AssertionError("daemon exited before server readiness")
                await asyncio.sleep(0.01)
    except BaseException:
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise


@pytest.mark.parametrize(
    ("received_signal_name", "expected_interrupted_cleanup_calls"),
    [(None, 1), ("SIGTERM", 0)],
)
def test_daemon_shutdown_marks_interrupted_attempts_only_without_signal(
    received_signal_name: str | None,
    expected_interrupted_cleanup_calls: int,
) -> None:
    """Signal shutdown defers OPS recovery, while ordinary shutdown performs it."""
    from polylogue.daemon import cli as daemon_cli

    class BlockingServer:
        shutdown_called = False
        close_called = False

        def __init__(self) -> None:
            self.ready = threading.Event()
            self._stopped = threading.Event()

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            assert poll_interval == 0.5
            self.ready.set()
            self._stopped.wait(timeout=5)

        def shutdown(self) -> None:
            self.shutdown_called = True
            self._stopped.set()

        def server_close(self) -> None:
            self.close_called = True

    class APIBlockingServer(BlockingServer):
        execution_kernel: BoundedComputeAdapter
        session_profile_callback: None
        operation_runtime: SimpleNamespace

        def server_close(self) -> None:
            super().server_close()
            self.execution_kernel.shutdown(wait=False, cancel_futures=True)

    class FakeConverger:
        pass

    async def noop() -> None:
        return None

    async def ready_fts(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(failed=0)

    def noop_sync(*_args: object) -> None:
        return None

    async def no_drive_changes() -> int:
        return 0

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    browser_server = BlockingServer()
    api_server = APIBlockingServer()
    from polylogue.daemon.execution import reset_daemon_compute_adapter

    api_server.execution_kernel = BoundedComputeAdapter(
        max_workers=1,
        queue_units=0,
        thread_name_prefix="test-daemon-api",
    )
    api_server.session_profile_callback = None

    async def shutdown_operation_runtime() -> None:
        return None

    api_server.operation_runtime = SimpleNamespace(shutdown=shutdown_operation_runtime)
    interrupted_cleanup_calls = 0

    def mark_interrupted_cleanup() -> None:
        nonlocal interrupted_cleanup_calls
        interrupted_cleanup_calls += 1

    async def exercise() -> None:
        task = asyncio.create_task(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                enable_api=True,
                api_host="127.0.0.1",
                api_port=8766,
            )
        )
        await _await_server_readiness_or_daemon_exit(task, (browser_server.ready, api_server.ready))
        if received_signal_name is not None:
            lifecycle = daemon_cli._daemon_lifecycle
            assert lifecycle is not None
            lifecycle.received_signal_name = received_signal_name
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

    patches = (
        patch.object(daemon_cli, "make_server", return_value=browser_server),
        patch.object(daemon_cli, "_ensure_embedding_lifecycle_startup_sync", noop_sync),
        patch("polylogue.daemon.fts_convergence.FtsConvergenceOwner.converge", ready_fts),
        patch.object(daemon_cli, "_ensure_lineage_startup_readiness_sync", noop_sync),
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
        patch.object(daemon_cli, "_configure_fts_automerge", noop),
        patch.object(daemon_cli, "_run_drive_source_catchup_safely", no_drive_changes),
        patch.object(daemon_cli, "_periodic_wal_checkpoint", wait_forever),
        patch.object(daemon_cli, "_periodic_fts_merge", wait_forever),
        patch(
            "polylogue.daemon.blob_gc_periodic.periodic_blob_publication_reconciliation_check",
            lambda **_kwargs: wait_forever(),
        ),
        patch.object(daemon_cli, "_periodic_heartbeat", wait_forever),
        patch.object(daemon_cli, "_periodic_drive_source_catchup", wait_forever),
        patch.object(daemon_cli, "_periodic_health_check", wait_forever),
        patch.object(daemon_cli, "_periodic_db_optimize", wait_forever),
        patch.object(daemon_cli, "_periodic_status_snapshot_refresh", wait_forever),
        patch.object(daemon_cli, "_periodic_convergence_check", lambda _sources, **_kwargs: wait_forever()),
        patch.object(daemon_cli, "_mark_interrupted_live_ingest_attempts_on_shutdown", mark_interrupted_cleanup),
        patch("polylogue.daemon.embedding_backlog.periodic_embedding_backlog_check", lambda **_kwargs: wait_forever()),
        patch("polylogue.daemon.convergence.DaemonConverger", return_value=FakeConverger()),
        patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=()),
        patch("polylogue.daemon.http.DaemonAPIHTTPServer", return_value=api_server),
    )
    with contextlib.ExitStack() as stack:
        for scoped_patch in patches:
            stack.enter_context(scoped_patch)
        reset_daemon_compute_adapter()
        try:
            asyncio.run(exercise())
        finally:
            reset_daemon_compute_adapter()

    assert browser_server.shutdown_called is True
    assert browser_server.close_called is True
    assert api_server.shutdown_called is True
    assert api_server.close_called is True
    assert interrupted_cleanup_calls == expected_interrupted_cleanup_calls


def test_daemon_shutdown_readiness_surfaces_startup_failure_without_spinning() -> None:
    """A startup exception reaches the shutdown test before its short bound expires."""

    async def failed_daemon() -> None:
        raise RuntimeError("api startup failed")

    async def exercise() -> None:
        task = asyncio.create_task(failed_daemon())
        never_ready = threading.Event()
        with pytest.raises(RuntimeError, match="api startup failed"):
            await _await_server_readiness_or_daemon_exit(task, (never_ready,))
        assert task.done()

    asyncio.run(exercise())


def test_run_daemon_services_schema_block_skips_write_but_starts_health_check() -> None:
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

    lifecycle_tick_started = False
    health_check_started = False

    async def lifecycle_heartbeat() -> None:
        nonlocal lifecycle_tick_started
        lifecycle_tick_started = True
        await asyncio.Event().wait()

    async def fake_health_check() -> None:
        # polylogue-7eo7 #4: FAST-tier health checks are read-only and stay
        # meaningful precisely when the watcher is schema-blocked -- unlike
        # the other periodic loops here, this one MUST start even while
        # blocked, or the daemon goes completely blind for the whole
        # blocked duration.
        nonlocal health_check_started
        health_check_started = True
        await asyncio.Event().wait()

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
        patch.object(daemon_cli, "_periodic_lifecycle_heartbeat", lifecycle_heartbeat),
        patch.object(daemon_cli, "_periodic_convergence_check", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_health_check", fake_health_check),
        patch.object(daemon_cli, "_periodic_db_optimize", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_status_snapshot_refresh", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_drive_source_catchup", side_effect=fail_background_work),
        patch("polylogue.daemon.convergence.DaemonConverger", side_effect=fail_background_work),
        patch.object(daemon_cli, "make_server", return_value=server),
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert server.shutdown_called is False
    assert server.close_called is True
    assert lifecycle_tick_started is True
    assert health_check_started is True


def test_periodic_schema_preflight_recheck_exits_on_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-blocked daemon must not stay watcher-less after tiers heal:
    when the preflight turns non-critical the recheck loop raises so the
    supervisor restarts the daemon into a healthy boot."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    def alert(severity: HealthSeverity) -> HealthAlert:
        return HealthAlert(
            check_name="schema_version",
            tier=HealthTier.FAST,
            severity=severity,
            message="probe",
            checked_at="now",
        )

    schedule = [alert(HealthSeverity.CRITICAL), alert(HealthSeverity.OK)]
    sleeps = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        # The runner jitters each tick, so the declared cadence is the floor
        # of the wait, not its exact value (polylogue-74wvj).
        assert (
            daemon_cli._SCHEMA_PREFLIGHT_RECHECK_INTERVAL_SECONDS
            <= seconds
            <= (daemon_cli._SCHEMA_PREFLIGHT_RECHECK_INTERVAL_SECONDS * 1.1)
        )

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", lambda: schedule.pop(0))
    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        pytest.raises(RuntimeError, match="schema preflight recovered"),
    ):
        asyncio.run(daemon_cli._periodic_schema_preflight_recheck())

    assert schedule == []
    assert sleeps == 2


# polylogue-t93b: the daemon's whale-pass escalation tier. A component
# permanently resource-blocked at the ordinary fast-path envelope must not
# stay blocked forever -- the periodic conveyor schedules a dedicated,
# bounded pass for it once (and only once) the ordinary trickle backlog is
# genuinely quiescent for the tick.


def test_periodic_raw_materialization_wakes_fair_intake_without_legacy_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed periodic route only wakes the fair owner.

    Anti-vacuity: restoring the legacy drain or its all-valid scanner makes
    this fail before the bounded canonical whale probe can run.
    """
    from polylogue.daemon import cli as daemon_cli

    wakeup = asyncio.Event()
    whale_calls: list[tuple[object, object]] = []

    async def fake_whale(**kwargs: object) -> bool:
        whale_calls.append((kwargs["raw_observation_owner"], kwargs["raw_intake_discovery"]))
        return False

    async def stop_after_one_tick(seconds: float) -> None:
        assert (
            daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS
            <= seconds
            <= (daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS * 1.1)
        )
        assert wakeup.is_set()
        raise asyncio.CancelledError

    owner = object()
    discovery = object()
    monkeypatch.setattr(daemon_cli, "_maybe_run_raw_materialization_whale_pass", fake_whale)
    with patch("asyncio.sleep", side_effect=stop_after_one_tick), pytest.raises(asyncio.CancelledError):
        asyncio.run(
            daemon_cli._periodic_raw_materialization_convergence(
                raw_observation_owner=owner,
                raw_intake_wakeup=wakeup,
                raw_intake_discovery=discovery,
            )
        )

    assert whale_calls == [(owner, discovery)]


@pytest.mark.parametrize("watcher_initially_registered", [False, True])
def test_periodic_raw_materialization_respects_watcher_registration_gate(
    monkeypatch: pytest.MonkeyPatch,
    watcher_initially_registered: bool,
) -> None:
    """Canonical raw maintenance wakes only after watcher registration."""
    from polylogue.daemon import cli as daemon_cli

    async def exercise() -> bool:
        watcher_registered = asyncio.Event()
        if watcher_initially_registered:
            watcher_registered.set()
        raw_intake_wakeup = asyncio.Event()
        whale_calls: list[tuple[object, object]] = []
        owner = object()
        discovery = object()

        async def fake_whale(**kwargs: object) -> bool:
            whale_calls.append((kwargs["raw_observation_owner"], kwargs["raw_intake_discovery"]))
            return False

        monkeypatch.setattr(daemon_cli, "_maybe_run_raw_materialization_whale_pass", fake_whale)
        task = asyncio.create_task(
            daemon_cli._periodic_raw_materialization_convergence(
                watcher_registered=watcher_registered,
                raw_observation_owner=owner,
                raw_intake_wakeup=raw_intake_wakeup,
                raw_intake_discovery=discovery,
            )
        )
        await asyncio.sleep(0)
        if not watcher_initially_registered:
            assert whale_calls == []
            assert not raw_intake_wakeup.is_set()
            watcher_registered.set()

            async def stop_after_one_tick(seconds: float) -> None:
                assert (
                    daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS
                    <= seconds
                    <= (daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS * 1.1)
                )
                raise asyncio.CancelledError

            with patch(
                "asyncio.sleep",
                side_effect=stop_after_one_tick,
            ):
                with pytest.raises(asyncio.CancelledError):
                    await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert whale_calls == [(owner, discovery)]
        return raw_intake_wakeup.is_set()

    assert asyncio.run(exercise()) is True


def test_canonical_whale_pass_uses_bounded_derivation_discovery_not_legacy_scanner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One oversized pending raw reaches the canonical owner at whale capacity.

    Anti-vacuity: the bounded discovery limit and the owner call each assert
    their canonical arguments, so a legacy scanner cannot satisfy this proof.
    """
    from polylogue.daemon import cli as daemon_cli

    raw_id = "oversized-raw"
    calls: list[tuple[str, int]] = []
    receipts: list[dict[str, object]] = []

    class Discovery:
        def discover_pending_raw_ids(self, limit: int) -> tuple[tuple[str, int], ...]:
            assert limit == 1
            return ((raw_id, daemon_cli._RAW_MATERIALIZATION_DAEMON_BLOB_LIMIT_BYTES + 1),)

    class Owner:
        async def converge_raw_id(self, candidate: str, *, max_payload_bytes: int) -> object:
            calls.append((candidate, max_payload_bytes))
            return SimpleNamespace(done=1, outcomes=())

    async def capture_receipt(**kwargs: object) -> None:
        receipts.append(cast(dict[str, object], kwargs["payload"]))

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(daemon_cli, "_resolve_raw_materialization_whale_blob_limit_bytes", lambda: 4096)
    monkeypatch.setattr(daemon_cli, "_drain_whale_receipt_outbox", lambda: asyncio.sleep(0))
    monkeypatch.setattr(daemon_cli, "_publish_whale_receipt", capture_receipt)
    assert asyncio.run(
        daemon_cli._maybe_run_raw_materialization_whale_pass(
            raw_observation_owner=Owner(),
            raw_intake_discovery=Discovery(),
        )
    )
    assert calls == [(raw_id, 4096)]
    assert receipts[-1]["status"] == "success"
    assert receipts[-1]["repaired_count"] == 1


def test_canonical_whale_pass_without_candidate_does_not_call_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bounded empty discovery page must not acquire publication machinery."""
    from polylogue.daemon import cli as daemon_cli

    class Discovery:
        def discover_pending_raw_ids(self, limit: int) -> tuple[tuple[str, int], ...]:
            assert limit == 1
            return ()

    class Owner:
        async def converge_raw_id(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("empty whale discovery must not call the owner")

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(daemon_cli, "_drain_whale_receipt_outbox", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        daemon_cli,
        "_resolve_raw_materialization_whale_blob_limit_bytes",
        lambda: 4096,
    )
    monkeypatch.setattr(
        daemon_cli,
        "_publish_whale_receipt",
        lambda **_kwargs: pytest.fail("empty whale discovery must not publish a receipt"),
    )

    assert (
        asyncio.run(
            daemon_cli._maybe_run_raw_materialization_whale_pass(
                raw_observation_owner=Owner(),
                raw_intake_discovery=Discovery(),
            )
        )
        is False
    )


@pytest.mark.parametrize("outcome_kind", ["pending", "failed"])
def test_canonical_whale_pass_maps_pending_and_failed_reports_to_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome_kind: str,
) -> None:
    """Canonical derivation outcomes remain truthful in the whale receipt."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.derivation import Outcome, PendingReason

    events: list[tuple[str, dict[str, object]]] = []
    receipts: list[dict[str, object]] = []
    raw_id = f"{outcome_kind}-raw"

    class Discovery:
        def discover_pending_raw_ids(self, limit: int) -> tuple[tuple[str, int], ...]:
            assert limit == 1
            return ((raw_id, daemon_cli._RAW_MATERIALIZATION_DAEMON_BLOB_LIMIT_BYTES + 1),)

    class Owner:
        async def converge_raw_id(self, candidate: str, *, max_payload_bytes: int) -> object:
            assert candidate == raw_id
            assert max_payload_bytes == 4096
            if outcome_kind == "pending":
                outcome = SimpleNamespace(
                    key=SimpleNamespace(key=raw_id),
                    outcome=Outcome.PENDING,
                    reason=PendingReason.BLOCKED,
                )
            else:
                outcome = SimpleNamespace(
                    key=SimpleNamespace(key=raw_id),
                    outcome=Outcome.FAILED,
                    error="source frontier changed",
                )
            return SimpleNamespace(done=0, outcomes=(outcome,))

    async def capture_receipt(**kwargs: object) -> None:
        receipts.append(cast(dict[str, object], kwargs["payload"]))

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(daemon_cli, "_resolve_raw_materialization_whale_blob_limit_bytes", lambda: 4096)
    monkeypatch.setattr(daemon_cli, "_drain_whale_receipt_outbox", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "polylogue.daemon.events.emit_daemon_event",
        lambda kind, *, payload: events.append((str(kind), cast(dict[str, object], payload))),
    )
    monkeypatch.setattr(daemon_cli, "_publish_whale_receipt", capture_receipt)

    assert (
        asyncio.run(
            daemon_cli._maybe_run_raw_materialization_whale_pass(
                raw_observation_owner=Owner(),
                raw_intake_discovery=Discovery(),
            )
        )
        is True
    )
    assert [kind for kind, _payload in events] == ["raw_materialization_whale_pass_started"]
    assert len(receipts) == 1
    assert receipts[0]["status"] == "error"
    assert receipts[0]["success"] is False
    assert receipts[0]["detail"] == ("blocked" if outcome_kind == "pending" else "source frontier changed")


@pytest.mark.parametrize(
    ("refusal", "pattern"),
    [
        (
            SimpleNamespace(source_paths=frozenset(), unattributed_reason="source tier is unreadable"),
            "source-selection gate blocked",
        ),
        (
            SimpleNamespace(source_paths=frozenset({"/archive/refused.jsonl"}), unattributed_reason=None),
            "refused source path",
        ),
    ],
)
def test_raw_observation_owner_preserves_source_frontier_refusal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    refusal: object,
    pattern: str,
) -> None:
    """Canonical owner admission refuses only what the frontier proves unsafe."""
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner

    raw_id = "refused-raw"
    observed_raw_ids: list[tuple[str, ...]] = []

    def refusal_gate(_root: Path, *, raw_ids: Sequence[str]) -> object:
        observed_raw_ids.append(tuple(raw_ids))
        return refusal

    monkeypatch.setattr(
        "polylogue.readiness.capability.raw_frontier_source_selection_refusal",
        refusal_gate,
    )

    class Derivation:
        def source_paths(self, raw_ids: Sequence[str]) -> dict[str, str]:
            assert tuple(raw_ids) == (raw_id,)
            return {raw_id: "/archive/refused.jsonl"}

    monkeypatch.setattr(
        "polylogue.operations.raw_observation_derivation.make_raw_observation_derivation",
        lambda *_args, **_kwargs: Derivation(),
    )
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=cast(BoundedComputeAdapter, object()),
        write_bridge=cast(DaemonWriteThreadBridge, object()),
        max_payload_bytes=4096,
    )

    with pytest.raises(RuntimeError, match=pattern):
        owner._require_source_frontier_authority(raw_id)
    assert observed_raw_ids == [(raw_id,)]


def test_raw_observation_publication_holds_writer_lease_through_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canonical replay keeps FTS/index publication under one writer lease."""
    from contextlib import contextmanager

    from polylogue.sources.revision_backfill import RawParsePrefetchCache
    from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement

    held = 0
    replay_held: list[int] = []
    blocked_checks: list[tuple[str, ...]] = []

    class Lease:
        def __init__(self, _root: Path) -> None:
            pass

        def acquire(self) -> None:
            nonlocal held
            held += 1

        def close(self) -> None:
            nonlocal held
            held -= 1

    class FakeArchive:
        def expand_raw_membership_selection(self, _raw_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
            return ("raw-1",), ()

        def raw_revision_descriptor(self, _raw_id: str) -> tuple[str, str, str, str, int]:
            return ("codex-session", "blob-hash", "/archive/source.jsonl", "full", 1)

    @contextmanager
    def open_archive() -> Any:
        yield FakeArchive()

    def fake_replay(*_args: object, **_kwargs: object) -> None:
        replay_held.append(held)

    monkeypatch.setattr("polylogue.storage.index_generation.ActiveWriterLease", Lease)

    def record_blocked_check(_root: Path, raw_ids: Sequence[str]) -> object:
        blocked_checks.append(tuple(raw_ids))
        return SimpleNamespace(source_paths=frozenset(), unattributed_reason=None)

    monkeypatch.setattr(
        "polylogue.storage.raw_retention.raw_frontier_blocked_raw_ids",
        record_blocked_check,
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        lambda *_args, **_kwargs: open_archive(),
    )
    monkeypatch.setattr("polylogue.storage.blob_store.BlobStore.verify", lambda _self, _blob_hash: True)
    monkeypatch.setattr("polylogue.sources.revision_backfill.backfill_historical_revision_evidence", fake_replay)

    adapter = RawObservationDerivation(tmp_path, max_payload_bytes=4096)
    monkeypatch.setattr(adapter, "_current", lambda _frame: True)
    monkeypatch.setattr(adapter, "_binding", lambda _raw_ids: "binding")
    monkeypatch.setattr(adapter, "source_paths", lambda _raw_ids: {"raw-1": "/archive/source.jsonl"})
    frame = SimpleNamespace(
        archive_root=str(tmp_path),
        source_revision=str(tmp_path / "index.db"),
        recipe_version=lambda _domain: adapter.recipe_version,
    )
    replacement = RawObservationReplacement(
        key="raw-1",
        input_binding="binding",
        payload=RawParsePrefetchCache(max_inflight_bytes=4096),
        raw_ids=("raw-1",),
    )

    assert adapter.publish(frame, replacement) is True
    assert blocked_checks == [("raw-1",)]
    assert replay_held == [1]
    assert held == 0


def test_raw_owner_cancellation_settles_publication_and_fts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancelling preparation still settles the shielded canonical publication."""
    from polylogue.core.enums import Provider
    from polylogue.daemon.derivation import DerivationFrame, ReplacementLike
    from polylogue.daemon.execution import BoundedComputeAdapter
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import (
        DaemonWriteCoordinator,
        DaemonWriteThreadBridge,
        daemon_write_lease_active,
    )
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    payload = [
        {
            "id": "cancelled-owner",
            "title": "cancelled-owner",
            "create_time": 1,
            "current_node": "m",
            "mapping": {
                "m": {
                    "id": "m",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": ["cancelled-owner"]},
                    },
                }
            },
        }
    ]
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path="cancelled-owner.json",
            acquired_at_ms=1,
        )

    async def scenario() -> None:
        compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
        coordinator = DaemonWriteCoordinator()
        owner = RawObservationConvergenceOwner(
            tmp_path,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            max_payload_bytes=1_000_000,
        )
        adapter = owner._converger._derivation_adapter("raw_observation")
        original_compute = adapter.compute
        started = threading.Event()
        release = threading.Event()

        def paused_compute(frame: DerivationFrame, key: str) -> ReplacementLike:
            assert not daemon_write_lease_active()
            started.set()
            assert release.wait(timeout=2.0)
            return original_compute(frame, key)

        monkeypatch.setattr(adapter, "compute", paused_compute)
        task = asyncio.create_task(owner.converge_raw_id(raw_id))
        try:
            assert await asyncio.to_thread(started.wait, 2.0)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            with sqlite3.connect(tmp_path / "index.db") as conn:
                assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
                assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] > 0
        finally:
            release.set()
            if not task.done():
                await task
            compute.shutdown(wait=True)
            assert await coordinator.shutdown(timeout=2.0) is True

    asyncio.run(scenario())


def test_whale_cancellation_publishes_canonical_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation of canonical whale admission leaves a terminal receipt."""
    from polylogue.daemon import cli as daemon_cli

    started = asyncio.Event()
    receipts: list[dict[str, object]] = []

    class Discovery:
        def discover_pending_raw_ids(self, limit: int) -> tuple[tuple[str, int], ...]:
            assert limit == 1
            return (("cancelled-raw", daemon_cli._RAW_MATERIALIZATION_DAEMON_BLOB_LIMIT_BYTES + 1),)

    class Owner:
        async def converge_raw_id(self, _raw_id: str, *, max_payload_bytes: int) -> object:
            assert max_payload_bytes == 4096
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def capture_receipt(**kwargs: object) -> None:
        receipts.append(cast(dict[str, object], kwargs["payload"]))

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(daemon_cli, "_resolve_raw_materialization_whale_blob_limit_bytes", lambda: 4096)
    monkeypatch.setattr(daemon_cli, "_drain_whale_receipt_outbox", lambda: asyncio.sleep(0))
    monkeypatch.setattr(daemon_cli, "_publish_whale_receipt", capture_receipt)

    async def scenario() -> None:
        task = asyncio.create_task(
            daemon_cli._maybe_run_raw_materialization_whale_pass(
                raw_observation_owner=Owner(),
                raw_intake_discovery=Discovery(),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(receipts) == 1
    assert receipts[0]["seed_raw_id"] == "cancelled-raw"
    assert receipts[0]["status"] == "cancelled"
    assert receipts[0]["cancelled"] is True
    assert receipts[0]["success"] is False


def test_raw_source_refusal_is_retryable_and_does_not_starve_sibling(
    tmp_path: Path,
) -> None:
    """The raw adapter isolates a refused ID while fair intake admits its sibling."""
    from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
    from polylogue.operations.intake_adapters import RawMaterializationIntakeAdapter

    pending = [("refused-raw", 1), ("healthy-raw", 1)]
    admitted: list[str] = []

    def discover(limit: int) -> tuple[tuple[str, int], ...]:
        return tuple(pending[:limit])

    def admit(raw_id: str) -> int:
        if raw_id == "refused-raw":
            raise RuntimeError("source-selection gate blocked: refused source path")
        admitted.append(raw_id)
        return 1

    adapter = RawMaterializationIntakeAdapter(discover, admit)
    dispatcher = FairIntakeDispatcher(
        (IntakeClassSpec(name="raw_materialization", adapter=adapter, page_size=8, max_attempts=3),),
        frame=f"daemon:{tmp_path}",
    )

    result = asyncio.run(dispatcher.run_once(budget=8))
    report = result.require_report("raw_materialization")
    assert report.admitted == 1
    assert report.retried == 1
    assert report.isolated == 0
    assert admitted == ["healthy-raw"]


def test_startup_drain_recovers_valid_recovery_receipt_and_acknowledges_after_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Startup recovery delivers a readable quarantine receipt before removing it."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import whale_outbox

    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(
        kind="whale.recovery",
        idempotency_key="recovery-receipt",
        operation_id="operation-recovery",
        payload={"status": "completed"},
        root=tmp_path,
    )
    recovery = target.with_name(f"{target.name}.recovery.0123456789abcdef0123456789abcdef.json")
    target.rename(recovery)
    assert whale_outbox.list_pending(root=tmp_path)[0]["_name"] == recovery.name

    published: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "polylogue.daemon.events.emit_daemon_event",
        lambda kind, *, payload: published.append((str(kind), cast(dict[str, object], payload))),
    )

    async def run_sync(_label: str, callback: object, *args: object, **kwargs: object) -> None:
        callback_result = callback(*args, **kwargs)  # type: ignore[operator]
        assert callback_result is None

    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: SimpleNamespace(run_sync=run_sync))
    delivered = asyncio.run(daemon_cli._drain_whale_receipt_outbox(root=tmp_path))

    assert delivered == 1
    assert published == [("whale.recovery", {"status": "completed"})]
    assert not recovery.exists()
    assert whale_outbox.list_pending(root=tmp_path) == []


def test_second_order_recovery_race_remains_startup_drainable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A recovery receipt acknowledged during a second race remains recoverable."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import whale_outbox

    tmp_path.chmod(0o700)
    canonical = whale_outbox.enqueue(
        kind="whale.recovery",
        idempotency_key="second-order",
        operation_id="operation-second-order",
        payload={"status": "completed"},
        root=tmp_path,
    )
    target = canonical.with_name(f"{canonical.name}.recovery.0123456789abcdef0123456789abcdef.json")
    canonical.rename(target)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    real_move = whale_outbox._rename_noreplace
    real_rename = os.rename
    replaced = False
    inserted = False

    def replace_before_move(source: str, destination: str, *, src_dir_fd: int = -1, dst_dir_fd: int = -1) -> None:
        nonlocal replaced
        if not replaced and source == target.name:
            target.unlink()
            target.write_bytes(
                b'{"kind":"whale.recovery","idempotency_key":"second-order",'
                b'"operation_id":"operation-second-order","payload":{"status":"replacement"}}'
            )
            target.chmod(0o600)
            replaced = True
        real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def insert_before_restore(source: str, destination: str, *, directory_fd: int) -> None:
        nonlocal inserted
        if not inserted and source.endswith(".ack"):
            target.write_bytes(b"replacement-at-restore")
            target.chmod(0o600)
            inserted = True
        real_move(source, destination, directory_fd=directory_fd)

    monkeypatch.setattr(os, "rename", replace_before_move)
    monkeypatch.setattr(whale_outbox, "_rename_noreplace", insert_before_restore)
    whale_outbox.acknowledge(record)
    pending = whale_outbox.list_pending(root=tmp_path)
    assert len(pending) == 1
    assert pending[0]["_name"].startswith("second-order.json.recovery.")

    published: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "polylogue.daemon.events.emit_daemon_event",
        lambda kind, *, payload: published.append((str(kind), cast(dict[str, object], payload))),
    )

    async def run_sync(_label: str, callback: object, *args: object, **kwargs: object) -> None:
        callback_result = callback(*args, **kwargs)  # type: ignore[operator]
        assert callback_result is None

    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: SimpleNamespace(run_sync=run_sync))
    assert asyncio.run(daemon_cli._drain_whale_receipt_outbox(root=tmp_path)) == 1
    assert published == [("whale.recovery", {"status": "replacement"})]
    assert whale_outbox.list_pending(root=tmp_path) == []


def test_recovery_name_exhaustion_leaves_fallback_receipt_startup_drainable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forced recovery-name exhaustion keeps a valid fallback drainable."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import whale_outbox

    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(
        kind="whale.recovery",
        idempotency_key="exhaustion",
        operation_id="operation-exhaustion",
        payload={"status": "completed"},
        root=tmp_path,
    )
    replacement = (
        b'{"kind":"whale.recovery","idempotency_key":"exhaustion",'
        b'"operation_id":"operation-exhaustion","payload":{"status":"fallback"}}'
    )
    real_rename = os.rename
    replaced = False

    def replace_before_move(source: str, destination: str, *, src_dir_fd: int = -1, dst_dir_fd: int = -1) -> None:
        nonlocal replaced
        if not replaced and source == target.name:
            target.unlink()
            target.write_bytes(replacement)
            target.chmod(0o600)
            replaced = True
        real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def exhausted(_source: str, _destination: str, *, directory_fd: int) -> None:
        raise FileExistsError("forced recovery allocation exhaustion")

    monkeypatch.setattr(os, "rename", replace_before_move)
    monkeypatch.setattr(whale_outbox, "_rename_noreplace", exhausted)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    whale_outbox.acknowledge(record)
    pending = whale_outbox.list_pending(root=tmp_path)
    assert len(pending) == 1
    assert pending[0]["_name"].endswith(".ack")

    published: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "polylogue.daemon.events.emit_daemon_event",
        lambda kind, *, payload: published.append((str(kind), cast(dict[str, object], payload))),
    )

    async def run_sync(_label: str, callback: object, *args: object, **kwargs: object) -> None:
        callback_result = callback(*args, **kwargs)  # type: ignore[operator]
        assert callback_result is None

    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: SimpleNamespace(run_sync=run_sync))
    assert asyncio.run(daemon_cli._drain_whale_receipt_outbox(root=tmp_path)) == 1
    assert published == [("whale.recovery", {"status": "fallback"})]
    assert whale_outbox.list_pending(root=tmp_path) == []


def _daemon_startup_stubs(
    stack: contextlib.ExitStack,
    daemon_cli: Any,
    tmp_path: Path,
    *,
    loops: dict[str, str] | None = None,
) -> None:
    """Stub the startup work the composition route performs before scheduling.

    Everything stubbed here is I/O the service inventory does not depend on;
    the composition route, the supervisor, and the registry stay real.
    """
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    ok_schema = HealthAlert(
        check_name="schema_version",
        tier=HealthTier.FAST,
        severity=HealthSeverity.OK,
        message="ok",
        checked_at="now",
    )

    async def _noop() -> None:
        return None

    async def _noop_fts(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(failed=0)

    stack.enter_context(patch("polylogue.paths.archive_root", return_value=tmp_path))
    stack.enter_context(patch.object(daemon_cli, "_check_schema_version_fast", return_value=ok_schema))
    stack.enter_context(
        patch.object(daemon_cli, "_ensure_embedding_lifecycle_startup_sync", lambda _root: tmp_path / "embeddings.db")
    )
    stack.enter_context(
        patch(
            "polylogue.daemon.fts_convergence.FtsConvergenceOwner.converge",
            _noop_fts,
        )
    )
    stack.enter_context(patch.object(daemon_cli, "_ensure_lineage_startup_readiness_sync", lambda: 0))
    stack.enter_context(patch.object(daemon_cli, "_reconcile_blob_publications", _noop))
    stack.enter_context(patch.object(daemon_cli, "_configure_fts_automerge", _noop))
    stack.enter_context(
        patch("polylogue.operations.mutation_transaction.recover_interrupted_operations", lambda _root: None)
    )
    stack.enter_context(patch.object(daemon_cli, "_mark_interrupted_live_ingest_attempts_on_shutdown"))
    stack.enter_context(patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=()))


def _record_task_creation(stack: contextlib.ExitStack) -> list[tuple[str, str | None]]:
    """Record every ``asyncio.create_task`` call with its immediate caller."""
    import sys as _sys

    real_create_task = asyncio.create_task
    created: list[tuple[str, str | None]] = []

    def recording(coro: Any, **kwargs: Any) -> Any:
        caller = _sys._getframe(1)
        created.append((caller.f_code.co_filename, kwargs.get("name")))
        return real_create_task(coro, **kwargs)

    stack.enter_context(patch.object(asyncio, "create_task", recording))
    return created


def _capture_supervisor(stack: contextlib.ExitStack, daemon_cli: Any) -> list[Any]:
    captured: list[Any] = []
    real_setter = daemon_cli._set_active_supervisor

    def capture(supervisor: Any) -> None:
        if supervisor is not None:
            captured.append(supervisor)
        real_setter(supervisor)

    stack.enter_context(patch.object(daemon_cli, "_set_active_supervisor", capture))
    return captured


def test_the_composition_route_spawns_only_declared_supervised_services(tmp_path: Path) -> None:
    """Every task the daemon spawns is one the registry declares.

    Anti-vacuity: add ``asyncio.create_task(...)`` anywhere in
    ``polylogue/daemon/cli.py`` and the first assertion names that file;
    start a name the registry does not carry and ``service_spec`` raises
    with the missing identity before a task exists.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.services import ServiceState, service_spec
    from polylogue.daemon.supervisor import TASK_NAME_PREFIX

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class FakeWatcher(_NoIntakeHints):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.watcher_ready = asyncio.Event()

        async def run(self) -> None:
            self.watcher_ready.set()
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            return None

    async def idle_loop(**_kwargs: object) -> None:
        await asyncio.Event().wait()

    with contextlib.ExitStack() as stack:
        _daemon_startup_stubs(stack, daemon_cli, tmp_path)
        created = _record_task_creation(stack)
        supervisors = _capture_supervisor(stack, daemon_cli)
        stack.enter_context(patch.object(daemon_cli, "Polylogue", FakePolylogue))
        stack.enter_context(patch.object(daemon_cli, "LiveWatcher", FakeWatcher))
        for attribute in (
            "_periodic_lifecycle_heartbeat",
            "_periodic_health_check",
            "_periodic_wal_checkpoint",
            "_periodic_fts_merge",
            "_periodic_heartbeat",
            "_periodic_db_optimize",
            "_periodic_status_snapshot_refresh",
            "_periodic_raw_materialization_convergence",
            "_periodic_drive_source_catchup",
        ):
            stack.enter_context(patch.object(daemon_cli, attribute, idle_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_convergence_check", lambda *_a, **_k: idle_loop()))
        for target in (
            "polylogue.daemon.embedding_backlog.periodic_embedding_backlog_check",
            "polylogue.daemon.embedding_backlog.periodic_embedding_orphan_reconcile_check",
            "polylogue.daemon.judgment_automation.periodic_judgment_automation_sweep",
            "polylogue.daemon.blob_gc_periodic.periodic_blob_gc_check",
            "polylogue.daemon.blob_gc_periodic.periodic_blob_publication_reconciliation_check",
            "polylogue.daemon.secret_scan_sweep.periodic_secret_scan_sweep",
        ):
            stack.enter_context(patch(target, idle_loop))
        stack.enter_context(pytest.raises(RuntimeError, match="watch stopped"))
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    direct_from_composition_root = [entry for entry in created if entry[0].endswith("polylogue/daemon/cli.py")]
    assert direct_from_composition_root == [], (
        f"the composition root created an unsupervised task: {direct_from_composition_root}"
    )

    supervised_names = sorted(
        name[len(TASK_NAME_PREFIX) :]
        for _filename, name in created
        if name is not None and name.startswith(TASK_NAME_PREFIX)
    )
    assert supervised_names, "no supervised service was started on the production route"
    for name in supervised_names:
        service_spec(name)

    assert len(supervisors) == 1
    supervisor = supervisors[0]
    assert set(supervised_names) <= set(supervisor.states())
    unresolved = [spec.name for spec in supervisor.selected if supervisor.state(spec.name) is ServiceState.PENDING]
    assert unresolved == [], f"declared services the composition route never resolved: {unresolved}"


def test_daemon_composition_gives_raw_whale_its_own_discovery_cursor(tmp_path: Path) -> None:
    """Whale selection cannot consume fair intake's raw continuation.

    ``RawMaterializationDiscovery`` owns a process-local traversal cursor.  If
    the periodic whale route borrows fair intake's instance, a whale probe
    advances that cursor and changes which raw the fair adapter sees next.

    Anti-vacuity: passing ``raw_intake_discovery`` to periodic raw convergence
    instead of a separately constructed whale discovery makes the identities
    below equal.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import intake_adapters as daemon_intake_adapters
    from polylogue.operations.intake_adapters import build_intake_adapters as build_real_intake_adapters

    created: list[object] = []
    periodic_discoveries: list[object] = []
    fair_discover: list[object] = []

    class Discovery:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            created.append(self)

        def discover_pending_raw_ids(self, _limit: int) -> tuple[tuple[str, int], ...]:
            return ()

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class FakeWatcher(_NoIntakeHints):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.watcher_ready = asyncio.Event()

        async def run(self) -> None:
            await asyncio.sleep(0)
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            return None

    async def capture_periodic(**kwargs: object) -> None:
        periodic_discoveries.append(kwargs["raw_intake_discovery"])
        await asyncio.Event().wait()

    def capture_build(*args: Any, **kwargs: Any) -> tuple[tuple[str, object], ...]:
        fair_discover.append(kwargs["raw_discover"])
        return cast(tuple[tuple[str, object], ...], build_real_intake_adapters(*args, **kwargs))

    with contextlib.ExitStack() as stack:
        _daemon_startup_stubs(stack, daemon_cli, tmp_path)
        stack.enter_context(patch.object(daemon_cli, "Polylogue", FakePolylogue))
        stack.enter_context(patch.object(daemon_cli, "LiveWatcher", FakeWatcher))
        stack.enter_context(patch.object(daemon_intake_adapters, "RawMaterializationDiscovery", Discovery))
        stack.enter_context(patch.object(daemon_intake_adapters, "build_intake_adapters", capture_build))
        stack.enter_context(patch.object(daemon_cli, "_periodic_raw_materialization_convergence", capture_periodic))
        stack.enter_context(pytest.raises(RuntimeError, match="watch stopped"))
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert len(created) == 2
    assert len(fair_discover) == 1
    fair_cursor = inspect.getclosurevars(cast(Callable[..., object], fair_discover[0])).nonlocals[
        "raw_intake_discovery"
    ]
    assert periodic_discoveries == [created[1]]
    assert periodic_discoveries[0] is not fair_cursor


@pytest.mark.uses_real_clock("bounds the focused-profile fixture's own wall-clock cost")
def test_a_focused_profile_starts_no_materialization_and_finishes_promptly(tmp_path: Path) -> None:
    """The API-disabled fixture profile.

    Anti-vacuity: pass ``ServiceProfile.PRODUCTION`` instead and the
    materialization assertion fails, because the production profile starts
    the raw-materialization loop this profile exists to exclude.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.services import ServiceProfile, ServiceState

    started: list[str] = []

    async def resident_loop(**_kwargs: object) -> None:
        started.append("resident")

    async def materialization(**_kwargs: object) -> None:
        started.append("raw_materialization")
        await asyncio.Event().wait()

    with contextlib.ExitStack() as stack:
        _daemon_startup_stubs(stack, daemon_cli, tmp_path)
        supervisors = _capture_supervisor(stack, daemon_cli)
        stack.enter_context(patch.object(daemon_cli, "_periodic_lifecycle_heartbeat", resident_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_health_check", resident_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_raw_materialization_convergence", materialization))
        started_at = time.monotonic()
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                enable_api=False,
                service_profile=ServiceProfile.RESIDENT_CORE,
            )
        )
        elapsed = time.monotonic() - started_at

    assert "raw_materialization" not in started
    assert started == ["resident", "resident"]
    assert elapsed < 10.0

    supervisor = supervisors[0]
    assert supervisor.state("raw_observation_convergence") is ServiceState.SKIPPED
    assert supervisor.state("lifecycle_heartbeat") is ServiceState.STOPPED


def test_whale_pass_on_an_empty_root_is_quiet_not_a_repeating_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh archive root has no raw tier yet, and that is not an error.

    polylogue-f7pdm: the periodic whale pass ran discovery against a
    non-existent ``source.db`` and logged
    ``daemon.raw_materialization.whale_schedule_failed`` at WARNING every
    thirty seconds for the daemon's whole lifetime on the declared build
    route.

    Anti-vacuity: removing the missing-tier guard in
    ``RawMaterializationDiscovery.discover_pending_raw_ids`` makes this raise
    ``sqlite3.OperationalError`` instead of returning ``False``.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.operations.intake_adapters import RawMaterializationDiscovery

    monkeypatch.setattr(daemon_cli, "archive_root", lambda: tmp_path, raising=False)
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)

    async def no_outbox() -> None:
        return None

    monkeypatch.setattr(daemon_cli, "_drain_whale_receipt_outbox", no_outbox)

    assert not (tmp_path / "source.db").exists()
    attempted = asyncio.run(
        daemon_cli._maybe_run_raw_materialization_whale_pass(
            raw_observation_owner=object(),
            raw_intake_discovery=RawMaterializationDiscovery(tmp_path, max_payload_bytes=1024),
        )
    )

    assert attempted is False


@pytest.mark.asyncio
async def test_an_undrained_writer_retains_rebuild_exclusion_for_the_process(tmp_path: Path) -> None:
    """Drain authority decides whether an offline rebuild may ever start.

    This is the invariant the deleted standalone ``run_live_watcher`` entry
    point used to carry: the daemon's own shutdown path
    (``_shutdown_writer_coordinator_with_rebuild_exclusion``) is the one route
    that holds it now.

    Anti-vacuity: drop the ``retain_until_process_exit`` call for an undrained
    writer, or swallow the shutdown exception, and the retained flags below go
    False.
    """
    from polylogue.daemon import cli as daemon_cli

    class _Exclusion:
        def __init__(self) -> None:
            self.retained = False

        def retain_until_process_exit(self) -> None:
            self.retained = True

    class _Coordinator:
        def __init__(self, drained: bool | BaseException) -> None:
            self._drained = drained

        async def shutdown(self, *, timeout: float) -> bool:
            assert timeout == 5.0
            if isinstance(self._drained, BaseException):
                raise self._drained
            return self._drained

    drained_exclusion = _Exclusion()
    assert (
        await daemon_cli._shutdown_writer_coordinator_with_rebuild_exclusion(
            cast(Any, _Coordinator(True)), cast(Any, drained_exclusion), timeout=5.0
        )
        is True
    )
    assert drained_exclusion.retained is False

    undrained_exclusion = _Exclusion()
    assert (
        await daemon_cli._shutdown_writer_coordinator_with_rebuild_exclusion(
            cast(Any, _Coordinator(False)), cast(Any, undrained_exclusion), timeout=5.0
        )
        is False
    )
    assert undrained_exclusion.retained is True

    raising_exclusion = _Exclusion()
    with pytest.raises(RuntimeError, match="drain failed"):
        await daemon_cli._shutdown_writer_coordinator_with_rebuild_exclusion(
            cast(Any, _Coordinator(RuntimeError("drain failed"))), cast(Any, raising_exclusion), timeout=5.0
        )
    assert raising_exclusion.retained is True
