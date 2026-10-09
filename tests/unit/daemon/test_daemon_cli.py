from __future__ import annotations

import asyncio
import builtins
import contextlib
import dataclasses
import errno
import hashlib
import inspect
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any, ParamSpec, TypeVar, cast
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig
from polylogue.config import Config
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.json import JSONDocument, loads
from polylogue.daemon.commands import main
from polylogue.daemon.convergence import ConvergenceStage
from polylogue.daemon.derivation import DerivationReport, Outcome
from polylogue.daemon.health import DaemonHealth, HealthSeverity, HealthTier
from polylogue.daemon.lineage_startup import LineageStartupCensus
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.session_profile_composition import ComposedSessionProfiles
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.logging import capture
from polylogue.operations.drive_readiness import DriveCatchupReport, DriveCatchupState
from polylogue.paths import browser_capture_spool_root
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import RetainedReplayOutcome
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.derived.raw import RawObservationScope
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture
from tests.infra.frozen_clock import FrozenClock
from tests.infra.live_ingest import write_index_session

P = ParamSpec("P")
T = TypeVar("T")


def _converged_lineage_census() -> LineageStartupCensus:
    """A startup census that saw a converged archive: nothing to report."""
    return LineageStartupCensus(dangling_edges=0, dangling_sessions=0)


async def _unused_session_profile_callback(_session_ids: Sequence[str] | None) -> DerivationReport:
    raise AssertionError("session-profile callback should not run")


class _NoIntakeHints:
    def intake_revision(self, source: WatchSource) -> int:
        return 0

    def prepare_watch_roots(self) -> list[Path]:
        return [Path("/synthetic-watch-root")]


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


def _render_observed_status(args: list[str]) -> Any:
    """Exercise command rendering of the actual producer's unit-test view.

    The resident transport has separate real-socket controls. These existing
    component assertions keep exercising the canonical status producer.
    """
    from polylogue.daemon.status import daemon_status_payload

    payload = daemon_status_payload()
    with patch("polylogue.daemon.commands._live_daemon_status_payload", return_value=payload):
        return CliRunner().invoke(main, ["status", *args])


@pytest.mark.contract
def test_polylogued_status_json_reports_daemon_components(
    tmp_path: Path,
) -> None:
    sources = (
        WatchSource(name="exists", root=tmp_path),
        WatchSource(name="missing", root=tmp_path / "missing"),
    )

    with patch("polylogue.daemon.status.default_sources", return_value=sources):
        result = _render_observed_status(
            ["--format", "json"],
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
    assert "spool_path" not in browser_capture


def test_polylogued_status_plain_reports_daemon_components(tmp_path: Path) -> None:
    sources = (WatchSource(name="exists", root=tmp_path),)

    with patch("polylogue.daemon.status.default_sources", return_value=sources):
        result = _render_observed_status([])

    assert result.exit_code == 1
    assert "Polylogue daemon" in result.output
    assert "Live sources: 1/1 available" in result.output
    assert f"exists: {tmp_path} (available)" in result.output
    assert "Browser capture spool: ready" in result.output


def test_polylogued_status_json_reports_archive_storage(tmp_path: Path) -> None:
    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.live_ingest import prepared_live_convergence_owner

    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("embeddings.db", ArchiveTier.EMBEDDINGS),
        ("user.db", ArchiveTier.USER),
        ("audit.db", ArchiveTier.AUDIT),
        ("ops.db", ArchiveTier.OPS),
    ):
        if tier is ArchiveTier.SOURCE:
            initialize_runtime_source_fixture(tmp_path / filename)
        else:
            initialize_archive_database(tmp_path / filename, tier)

    async def inspect_fixture() -> None:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            inspection = await owner.run_convergence_sync(
                "fixture.frontier.inspect",
                inspect_prepared_raw_authority_frontier,
                tmp_path,
                input_demand=owner._compute_adapter.amend_current_input_demand,
            )
            assert inspection.mode == "full" and inspection.healthy
            assert inspection.accepted_head_checks == inspection.cursor_checks == 0

    asyncio.run(inspect_fixture())

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = _render_observed_status(["--format", "json"])

    # A schema-complete but empty archive has no raw revisions from which to
    # prove materialization readiness, so status must report it as unmeasured.
    assert result.exit_code == 1
    payload = loads(result.output)
    assert isinstance(payload, dict)
    frontier = cast(dict[str, object], payload["raw_frontier_integrity"])
    assert frontier["available"] is True
    assert frontier["broken_head_status"] == "healthy"
    assert frontier["cursor_ahead_status"] == "healthy"
    storage = cast(dict[str, object], payload["archive_storage"])
    assert storage["active_store"] == "archive_file_set"
    assert storage["archive_root"] == str(tmp_path)
    assert storage["configured_archive_root"] == str(tmp_path)
    assert storage["archive_root_matches_configured"] is True
    assert storage["final_shape_ready"] is True
    assert storage["schema_mismatches"] == []
    assert storage["archive_schema_ready"] is True
    assert storage["archive_materialization_ready"] is False
    assert cast(dict[str, object], storage["archive_materialization_assessment"])["reason"] == "zero_denominator"
    assert storage["archive_ready"] is False
    assert storage["present_tiers"] == ["source", "index", "embeddings", "user", "audit", "ops"]
    tiers = cast(list[dict[str, object]], storage["tiers"])
    # The daemon reports what each tier stamps, so the expectation is the one
    # runtime declaration of that -- not a second per-tier constant that can
    # silently disagree with it.
    assert {tier["name"]: tier["user_version"] for tier in tiers} == {
        tier.value: ARCHIVE_VERSION_BY_TIER[tier] for tier in ArchiveTier
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
        if tier is ArchiveTier.SOURCE:
            initialize_runtime_source_fixture(tmp_path / filename)
        else:
            initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1}")

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = _render_observed_status(["--format", "json"])

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
    assert index_tier["user_version"] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1
    assert index_tier["expected_user_version"] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
    assert index_tier["version_status"] == "mismatch"
    components_raw = payload["component_readiness"]
    assert isinstance(components_raw, dict)
    components = cast(dict[str, dict[str, object]], components_raw)
    archive_component = components["archive_storage"]
    assert archive_component["state"] == "blocked"
    assert archive_component["repair_hint"] == "polylogued run"


def test_polylogued_status_plain_reports_archive_storage(tmp_path: Path) -> None:
    initialize_runtime_source_fixture(tmp_path / "source.db")
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = _render_observed_status([])

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
        if tier is ArchiveTier.SOURCE:
            initialize_runtime_source_fixture(tmp_path / filename)
        else:
            initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] + 1}")

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = _render_observed_status([])

    assert result.exit_code == 1
    assert (
        "Storage: archive_file_set (source, index, embeddings, user, audit, ops); final split complete; schema mismatch index"
        in result.output
    )


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_retries_session_subjects_without_source_lookup(
    tmp_path: Path,
    frozen_clock: FrozenClock,
    bounded_compute_adapter: BoundedComputeAdapter,
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
    frozen_clock.advance(1.0)
    stage = ConvergenceStage(
        name="derived",
        description="retry test",
        check=lambda _candidate: False,
        execute=lambda _candidate: False,
        check_sessions=lambda session_ids: {"conv-1"} if tuple(session_ids) == ("conv-1",) else set(),
        execute_sessions=lambda session_ids: tuple(session_ids) == ("conv-1",),
    )
    with patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=(stage,)):
        retried = daemon_cli._drain_convergence_debt_once(db, compute_adapter=bounded_compute_adapter)
        debt_after = cursor.list_convergence_debt()

    assert retried == 1
    assert debt_after == []


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_preserves_error_for_unimplemented_stage(
    tmp_path: Path,
    frozen_clock: FrozenClock,
    bounded_compute_adapter: BoundedComputeAdapter,
) -> None:
    """A stage with no registered implementation leaves its debt row untouched.

    The drain measures nothing about a row whose stage nothing can run, so
    re-recording it would overwrite the producer's diagnostic with a note about
    the missing stage (polylogue-ia88n). The stage name below is now registered
    in the real default set -- this test patches the set to an unrelated stage,
    because the invariant is about ANY unimplemented stage, not about that one.

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
        retried = daemon_cli._drain_convergence_debt_once(db, compute_adapter=bounded_compute_adapter)
        debt_after = cursor.list_convergence_debt()

    assert retried == 0
    assert len(debt_after) == 1
    row = debt_after[0]
    assert row.stage == "lineage_prefix_recompose"
    assert row.last_error == "alias collision truncated child prefix at message 42"


def test_periodic_convergence_check_treats_sqlite_lock_as_archive_busy(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()

    def fake_drain(_db: Path, **_kwargs: object) -> int:
        raise sqlite3.OperationalError("database is locked")

    with (
        patch.object(daemon_cli, "_drain_convergence_debt_and_frontier", fake_drain),
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


@pytest.mark.parametrize("raw_failure", (False, True))
def test_periodic_convergence_check_waits_for_watcher_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_failure: bool,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()
    drains: list[Path] = []
    fts_scopes: list[object] = []
    raw_retention_calls: list[None] = []
    drained = asyncio.Event()

    def fake_drain(drain_db: Path, **_kwargs: object) -> int:
        drains.append(drain_db)
        return 0

    async def fake_fts_converge() -> object:
        fts_scopes.append(None)
        drained.set()
        return SimpleNamespace()

    async def fake_raw_retention() -> None:
        raw_retention_calls.append(None)
        if raw_failure:
            raise sqlite3.OperationalError("retention unavailable")

    async def exercise() -> None:
        watcher_registered = asyncio.Event()
        monkeypatch.setattr(daemon_cli, "_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS", 60)
        monkeypatch.setattr(
            daemon_cli,
            "daemon_write_coordinator",
            lambda: SimpleNamespace(run_sync=None),
        )
        monkeypatch.setattr(daemon_cli, "_drain_convergence_debt_and_frontier", fake_drain)
        monkeypatch.setattr(daemon_cli, "_active_index_db_path", lambda: db)
        task = asyncio.create_task(
            daemon_cli._periodic_convergence_check(
                (),
                fts_owner=cast(Any, SimpleNamespace(converge=fake_fts_converge)),
                watcher_registered=watcher_registered,
                raw_retention_callback=fake_raw_retention,
            )
        )
        await asyncio.sleep(0)
        assert drains == []
        watcher_registered.set()
        await asyncio.wait_for(drained.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with capture() as records:
        asyncio.run(exercise())

    assert drains == [db]
    assert fts_scopes == [None]
    assert raw_retention_calls == [None]
    failures = [record for record in records if record["event"] == "daemon.raw_retention.retry_failed"]
    assert len(failures) == int(raw_failure)


def test_periodic_convergence_check_warns_on_non_lock_failures(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()

    def fake_drain(_db: Path, **_kwargs: object) -> int:
        raise RuntimeError("unexpected convergence retry failure")

    # The drain itself runs off the writer lease (polylogue-ssplv); the
    # coordinator is reached only by the admission each stage's write uses.
    with (
        patch.object(daemon_cli, "_drain_convergence_debt_and_frontier", fake_drain),
        patch.object(
            daemon_cli,
            "daemon_write_coordinator",
            return_value=SimpleNamespace(run_sync=None),
        ),
        capture() as records,
    ):
        failure = asyncio.run(daemon_cli._retry_convergence_debt_once(db))

    # The failure is handed back so the periodic loop re-raises it after its
    # other stages and records it as ``last_error`` (polylogue-hu24g); before,
    # it was suppressed and only the span's event remained.
    assert isinstance(failure, RuntimeError)
    # The span's terminal event is emitted from ``__exit__``, so the
    # failure is also on the record at ERROR.
    errors = [r for r in records if r["event"] == "daemon.convergence_debt.pass.error"]
    assert len(errors) == 1
    assert errors[0]["level"] == "error"
    assert errors[0]["outcome"] == "error"
    assert errors[0]["error_type"] == "RuntimeError"
    assert "unexpected convergence retry failure" in str(errors[0]["error_detail"])
    assert [r for r in records if r["event"] == "daemon.convergence_debt.pass.ok"] == []


def test_an_unreadable_config_is_reported_not_read_as_no_drive_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity (polylogue-hu24g): ``suppress(Exception)`` left the Drive
    intake class unregistered for the daemon's life with no event at all."""
    import polylogue.config as config_module
    from polylogue.daemon import cli as daemon_cli

    def unreadable() -> Config:
        raise OSError("config unreadable")

    monkeypatch.setattr(config_module, "get_config", unreadable)
    with capture() as records:
        assert daemon_cli._drive_sources_configured() is False
    events = [record for record in records if record["event"] == "daemon.intake.drive_config_unreadable"]
    assert len(events) == 1
    assert events[0]["error_type"] == "OSError"


def test_an_alert_probe_that_raises_is_itself_an_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (polylogue-hu24g): both heartbeat probes ran under
    ``suppress(Exception)``, so a probe that raised could never alert."""
    import polylogue.hooks as hooks
    from polylogue.daemon import cli as daemon_cli

    def broken_drift(_harness: str) -> list[Path]:
        raise PermissionError("hook directory unreadable")

    def broken_depth(*, cap: int) -> int:
        raise OSError("spool unreadable")

    monkeypatch.setattr(hooks, "hook_install_sidecar_drift", broken_drift)
    monkeypatch.setattr(daemon_cli, "_browser_capture_spool_pending_file_count", broken_depth)
    with capture() as records:
        daemon_cli._log_spool_depth_if_notable()

    failed = [record for record in records if record["event"] == "daemon.alert_probe.failed"]
    assert sorted((record["operation"], record["component"]) for record in failed) == [
        ("browser_capture_spool_depth", "browser-capture"),
        ("hook_install_sidecar_drift", "claude-code"),
        ("hook_install_sidecar_drift", "codex"),
    ]


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
        from polylogue.daemon.services import ServiceProfile

        class FakeServer:
            def __init__(self, auth_token: str | None) -> None:
                self.config = BrowserCaptureReceiverConfig(
                    spool_path=browser_capture_spool_root(), auth_token=auth_token
                )

            def serve_forever(self, poll_interval: float = 0.5) -> None:
                raise RuntimeError("server stopped")

            def shutdown(self) -> None:
                pass

            def server_close(self) -> None:
                pass

        captured: dict[str, object] = {}

        def _fake_make_server(*_args: object, **kwargs: object) -> FakeServer:
            captured.update(kwargs)
            return FakeServer(cast(str | None, kwargs.get("auth_token")))

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
                    service_profile=ServiceProfile.SURFACES,
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

    def test_receiver_preserves_explicit_machine_policy_separately(self) -> None:
        captured = self._run_with_captured_make_server_kwargs(
            browser_capture_auth_token="neutral-pairing-token",
            api_auth_token="neutral-machine-token",
            api_allow_no_auth=True,
        )
        assert captured["auth_token"] == "neutral-pairing-token"
        assert captured["api_auth_token"] == "neutral-machine-token"
        assert captured["api_allow_no_auth"] is True

    def test_explicit_token_wins_over_auto_mint(self) -> None:
        captured = self._run_with_captured_make_server_kwargs(browser_capture_auth_token="operator-set-token")

        assert captured.get("auth_token") == "operator-set-token"


def test_polylogued_run_uses_default_sources() -> None:
    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch("polylogue.sources.live.watcher.default_sources", return_value=sources) as default_sources,
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


def test_polylogued_run_can_skip_configured_source_catchup() -> None:
    recorded: dict[str, object] = {}

    async def fake_run_daemon_services(**kwargs: object) -> None:
        recorded.update(kwargs)

    with patch("polylogue.daemon.cli.run_daemon_services", side_effect=fake_run_daemon_services):
        result = CliRunner().invoke(
            main,
            [
                "run",
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
        patch("polylogue.sources.live.watcher.default_sources", return_value=sources) as default_sources,
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


def test_drive_source_catchup_skips_when_no_drive_sources(
    tmp_path: Path, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
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
        changed = asyncio.run(
            daemon_cli._run_drive_source_catchup_once(
                _unused_session_profile_callback, raw_owner=None, compute_owner=bounded_compute_adapter
            )
        )

    assert changed.state.value == "complete"
    build_services.assert_not_called()


def test_drive_source_catchup_ingests_configured_drive_source(
    tmp_path: Path, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
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
        def __init__(
            self,
            *,
            repository: object,
            archive_root: Path,
            config: Config,
            execution: object,
            retained_runner: object,
        ) -> None:
            from polylogue.daemon.drive_catchup import DriveCatchupExecution

            assert isinstance(execution, DriveCatchupExecution)
            assert retained_runner == raw_owner.ingest_retained_raw_ids
            events.append(("parser", repository, archive_root, config))

        async def ingest_sources(
            self,
            *,
            sources: list[Source],
            stage: str,
            parse_records: bool,
            max_pass_seconds: float | None = None,
            skip_acquire: bool = False,
        ) -> SimpleNamespace:
            events.append(("ingest", sources, stage, parse_records, max_pass_seconds))
            return SimpleNamespace(
                acquire_result=SimpleNamespace(raw_ids=["raw-1"], errors=0, drive_witnesses={}),
                parse_result=SimpleNamespace(
                    processed_ids={"session-b", "session-a"},
                    counts={"sessions": 0},
                    time_budget_exceeded=False,
                ),
            )

    class FakeRawOwner:
        async def ingest_retained_raw_ids(self, raw_ids: Sequence[str]) -> RetainedReplayOutcome:
            del raw_ids
            return RetainedReplayOutcome()

    raw_owner = FakeRawOwner()

    async def canonical_callback(session_ids: Sequence[str] | None) -> DerivationReport:
        events.append(("canonical", session_ids))
        return cast(DerivationReport, object())

    with (
        patch("polylogue.config.get_config", return_value=config),
        patch("polylogue.services.build_runtime_services", return_value=FakeServices()) as build_services,
        patch("polylogue.pipeline.services.parsing.ParsingService", FakeParser),
    ):
        changed = asyncio.run(
            daemon_cli._run_drive_source_catchup_once(
                canonical_callback,
                raw_owner=cast(RawObservationConvergenceOwner, raw_owner),
                compute_owner=bounded_compute_adapter,
            )
        )

    assert changed.changed_count == 2
    assert changed.state.value == "unknown"
    build_services.assert_called_once_with(config=config, db_path=config.db_path)
    assert ("ingest", [drive_source], "all", True, daemon_cli._DRIVE_CATCHUP_MAX_PASS_SECONDS) in events
    assert ("canonical", ("session-a", "session-b")) in events
    assert events[-1] == "close"


def test_drive_source_catchup_keeps_session_derivation_failures_nonfatal(
    tmp_path: Path, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
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
                acquire_result=SimpleNamespace(raw_ids=["raw-1"], errors=0, drive_witnesses={}),
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
        changed = asyncio.run(
            daemon_cli._run_drive_source_catchup_once(
                failing_callback, raw_owner=None, compute_owner=bounded_compute_adapter
            )
        )

    assert changed.changed_count == 1
    assert changed.state.value == "unknown"
    failures = [r for r in records if r["event"] == "daemon.drive_catchup.session_profile_failed"]
    assert len(failures) == 1
    assert failures[0]["outcome"] == "degraded"
    assert failures[0]["error_type"] == "RuntimeError"
    # The pass itself still completed, and says so separately.
    assert [r["event"] for r in records if str(r["event"]).startswith("daemon.drive_catchup.pass.")][-1] == (
        "daemon.drive_catchup.pass.degraded"
    )


def test_drive_source_catchup_safe_wrapper_logs_failure(bounded_compute_adapter: BoundedComputeAdapter) -> None:
    from polylogue.daemon import cli as daemon_cli

    async def fail_catchup(_callback: object, **_owners: object) -> int:
        raise RuntimeError("drive unavailable")

    with (
        patch.object(daemon_cli, "_run_drive_source_catchup_once", fail_catchup),
        capture() as records,
    ):
        changed = asyncio.run(
            daemon_cli._run_drive_source_catchup_safely(
                _unused_session_profile_callback, raw_owner=None, compute_owner=bounded_compute_adapter
            )
        )

    assert changed.state.value == "retryable"
    failures = [r for r in records if r["event"] == "daemon.drive_catchup.failed"]
    assert len(failures) == 1
    assert failures[0]["outcome"] == "error"
    assert failures[0]["error_type"] == "RuntimeError"
    assert "drive unavailable" in str(failures[0]["error_detail"])


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
    from polylogue.storage.sqlite.write_lease import write_lease

    with write_lease("test.startup-blob-reservations", archive_root=archive_root_path):
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
            canonical_source_path="startup-referenced.json",
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
    from polylogue.storage.sqlite.write_lease import write_lease

    with write_lease("test.startup-blob-reservations", archive_root=archive_root_path):
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
    # The audit tier's declared version is the archive format floor, so the
    # hardcoded ``user_version = 1`` this test used to write stopped being a
    # mismatch once the schema floor was reset (b56699c80, #5275): the test
    # kept running but no longer produced the condition it names. Derive the
    # skew from the declaration instead.
    #
    # The skew is deliberately backward. A FORWARD audit version is a typed
    # startup refusal (see
    # test_forward_versioned_durable_tier_is_a_typed_startup_refusal), a
    # different contract from the one under test here; a backward version is
    # the plain version-mismatch this startup gate exists to park.
    mismatched_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT] - 1
    with sqlite3.connect(archive_root_path / "audit.db") as conn:
        conn.execute(f"PRAGMA user_version = {mismatched_version}")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root_path))

    recover_mock = Mock()

    with (
        patch(
            "polylogue.operations.mutation_replay.recover_interrupted_operations",
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
                ),
                timeout=5.0,
            )
        )

    recover_mock.assert_not_called()


def test_run_daemon_services_applies_staged_resets_before_any_tier_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-9kemf 07.F012: a staged tier reset lands before startup opens a tier.

    The seam must run under the daemon's archive ownership and before the
    schema preflight, lifecycle start and operation recovery, which all open
    tier connections. At the seam this process holds no descriptor on any tier
    file.

    Anti-vacuity: move ``apply_staged_archive_resets`` after the schema
    preflight (or drop it) and the recorded order no longer starts with the
    seam.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root_path = tmp_path / "archive"
    initialize_active_archive_root(archive_root_path)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root_path))
    tier_names = {f"{tier.value}.db{suffix}" for tier in ArchiveTier for suffix in ("", "-wal", "-shm")}
    order: list[str] = []
    open_tier_files: list[str] = []

    class _StopStartupError(Exception):
        pass

    def seam(root: Path) -> tuple[str, ...]:
        order.append("seam")
        assert root == archive_root_path
        for fd in Path("/proc/self/fd").iterdir():
            with contextlib.suppress(OSError):
                target = Path(os.readlink(fd))
                if target.parent == archive_root_path and target.name in tier_names:
                    open_tier_files.append(target.name)
        return ()

    def preflight() -> object:
        order.append("schema_preflight")
        raise _StopStartupError

    with (
        patch("polylogue.operations.mutation_replay.apply_staged_archive_resets", seam),
        patch.object(daemon_cli, "_check_schema_version_fast", preflight),
        pytest.raises(_StopStartupError),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=archive_root_path),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )

    assert order == ["seam", "schema_preflight"]
    assert open_tier_files == []


def test_forward_versioned_durable_tier_is_a_typed_startup_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-w6nrl: a durable tier newer than the runtime refuses startup cleanly.

    Forward skew means a newer release wrote the tier, so the daemon must not
    start at all -- but as a typed refusal the ``run`` command reports with an
    actionable message and exit 1, not as an untyped train-evidence error
    escaping through the shutdown path.

    Anti-vacuity: deleting the newer-than-runtime check in startup
    reconciliation makes the service raise the generic "lacks released train evidence" error (not
    the typed one), and deleting the ``DurableChangeTrainError`` mapping in
    ``run_command`` makes the CLI exit with an unhandled exception.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.migration_runner import DurableTierNewerThanRuntimeError

    archive_root_path = tmp_path / "archive"
    initialize_active_archive_root(archive_root_path)
    forward_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT] + 1
    with sqlite3.connect(archive_root_path / "audit.db") as conn:
        conn.execute(f"PRAGMA user_version = {forward_version}")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root_path))

    with pytest.raises(DurableTierNewerThanRuntimeError) as refused:
        asyncio.run(
            asyncio.wait_for(
                daemon_cli.run_daemon_services(
                    sources=(WatchSource(name="codex", root=archive_root_path),),
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                ),
                timeout=5.0,
            )
        )
    assert refused.value.tier is ArchiveTier.AUDIT
    assert refused.value.live_version == forward_version

    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "absent.toml"))
    result = CliRunner().invoke(main, ["run", "--no-watch", "--no-browser-capture"])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "newer than this runtime supports" in result.output


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

    class Coordinator(DaemonWriteCoordinator):
        async def run_sync(self, _actor: str, _function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
            raise RuntimeError("ops unavailable")

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.storage.archive_identity.resolve_active_index_path", lambda *_a, **_k: tmp_path / "index.db"
    )
    coordinator = Coordinator(archive_root=tmp_path)
    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: coordinator)

    with pytest.raises(RuntimeError, match="ops unavailable"):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )

    assert not (tmp_path / "daemon.pid").exists()


def test_daemon_startup_reconciles_trains_before_schema_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    events: list[str] = []

    class Coordinator(DaemonWriteCoordinator):
        async def run_sync(self, actor: str, _function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
            del args, kwargs
            if actor == "daemon.lifecycle.start":
                raise RuntimeError("startup stopped")
            return cast(T, None)

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.storage.archive_identity.resolve_active_index_path", lambda *_a, **_k: tmp_path / "index.db"
    )
    coordinator = Coordinator(archive_root=tmp_path)
    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: coordinator)

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
    # The refusal is the contract, not its wording.
    with pytest.raises(RuntimeError, match=re.escape(str(pidfile))):
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
    from polylogue.core.compute import BoundedComputeAdapter, reset_compute_adapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.convergence import DaemonConverger
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

    def fake_lineage_startup() -> LineageStartupCensus:
        events.append("lineage")
        return LineageStartupCensus(dangling_edges=0, dangling_sessions=0)

    async def fake_reconcile_blob_publications() -> None:
        events.append("blob-publications")

    async def fake_drive_catchup(callback: object, *, raw_owner: object, compute_owner: object) -> DriveCatchupReport:
        from polylogue.storage.sqlite.write_lease import current_write_lease

        assert current_write_lease() is None
        assert callback is api_server.session_profile_callback
        assert raw_owner is api_server.operation_runtime.raw_observation_owner
        assert compute_owner is api_server.execution_kernel
        events.append("drive-once")
        drive_called.set()
        return DriveCatchupReport(DriveCatchupState.COMPLETE)

    async def fake_configure_fts_automerge() -> None:
        events.append("automerge")

    def fake_operation_recovery(
        _archive_root_path: Path, *, resolver_actor_ref: str, input_demand: object, startup: bool
    ) -> None:
        assert startup
        events.append("operation-recovery")

    def recording_converger(
        stages: Iterable[ConvergenceStage], *, derivations: Iterable[object] = (), **kwargs: Any
    ) -> DaemonConverger:
        events.append("converger")
        return DaemonConverger(stages, derivations=derivations, **kwargs)

    async def fake_loop(name: str) -> None:
        events.append(name)
        await asyncio.Event().wait()

    class FakeAPIServer:
        execution_kernel: BoundedComputeAdapter

        def __init__(self) -> None:
            self.stopped = threading.Event()
            self.session_profile_callback = object()
            self.operation_runtime = SimpleNamespace(
                shutdown=self._shutdown_operation_runtime,
                embedding_convergence=None,
                accepted_ingest_redrive_claimed=_no_redrive_claims,
            )

        async def _shutdown_operation_runtime(self) -> None:
            events.append("operation-runtime-shutdown")

        def serve_forever(self, _poll_interval: float) -> None:
            self.stopped.wait(timeout=2.0)

        def shutdown(self) -> None:
            self.stopped.set()

        def server_close(self) -> None:
            self.execution_kernel.shutdown(wait=False, cancel_futures=True)
            return None

    assert reset_compute_adapter(join_timeout_s=5) == ()
    api_server = FakeAPIServer()

    def make_api_server(*_args: object, **_kwargs: object) -> FakeAPIServer:
        events.append("api-bind")
        kernel = _kwargs["execution_kernel"]
        assert isinstance(kernel, BoundedComputeAdapter)
        api_server.execution_kernel = kernel
        bridge = _kwargs["write_bridge"]
        root = _kwargs["archive_root"]
        assert isinstance(bridge, DaemonWriteThreadBridge)
        assert isinstance(root, Path)
        api_server.operation_runtime.raw_observation_owner = RawObservationConvergenceOwner(
            root,
            compute_adapter=kernel,
            write_bridge=bridge,
            write_coordinator=bridge.coordinator,
        )
        return api_server

    api_server_factory = Mock(side_effect=make_api_server)

    def fake_emit_daemon_event(kind: str, **kwargs: object) -> None:
        assert kind == "daemon.lifecycle"
        lifecycle_payloads.append(cast(dict[str, object], kwargs["payload"]))

    with contextlib.ExitStack() as stack:
        stack.callback(reset_compute_adapter)
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
        stack.enter_context(patch.object(daemon_cli, "_census_lineage_startup_sync", fake_lineage_startup))
        stack.enter_context(patch.object(daemon_cli, "_reconcile_blob_publications", fake_reconcile_blob_publications))
        stack.enter_context(patch.object(daemon_cli, "_check_schema_version_fast", return_value=ok_schema))
        stack.enter_context(patch("polylogue.paths.archive_root", return_value=tmp_path))
        stack.enter_context(patch.object(daemon_cli, "_run_drive_source_catchup_safely", fake_drive_catchup))
        stack.enter_context(patch.object(daemon_cli, "_configure_fts_automerge", fake_configure_fts_automerge))
        stack.enter_context(
            patch("polylogue.operations.mutation_replay.recover_interrupted_operations", fake_operation_recovery)
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
        stack.enter_context(patch.object(daemon_cli, "_periodic_heartbeat", lambda **_kwargs: fake_loop("heartbeat")))

        def fake_session_profile_audit(callback: object, **_kwargs: object) -> object:
            periodic_profile_callbacks.append(callback)
            return fake_loop("session-profile-audit")

        stack.enter_context(
            patch.object(
                daemon_cli, "_periodic_convergence_check", lambda _sources, **_kwargs: fake_loop("convergence")
            )
        )
        stack.enter_context(patch.object(daemon_cli, "_periodic_session_profile_audit", fake_session_profile_audit))
        stack.enter_context(patch.object(daemon_cli, "_periodic_health_check", lambda **_kwargs: fake_loop("health")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_db_optimize", lambda: fake_loop("optimize")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_status_snapshot_refresh", lambda: fake_loop("status")))
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
    passes terminate only after the real periodic session-profile audit loop
    invokes its post-catch-up no-hint callback. No manual operation invokes the owner.

    Anti-vacuity: omit the watcher callback, run the sweep before catch-up,
    replace it with a scoped live-source call, or retain output rows across
    restart, and the recorded ``None`` scope or repaired durable profile fails.
    """
    from polylogue import Polylogue as RealPolylogue
    from polylogue.core.compute import reset_compute_adapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import session_profile_composition
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

        return session_profile_composition.ComposedSessionProfiles(
            observe, composed.promoted_callback, composed.maintenance
        )

    def daemon_coordinator() -> DaemonWriteCoordinator:
        assert current_coordinator is not None
        return current_coordinator

    def profile_exists() -> bool:
        # A read, so it opens read-only: the daemon under test holds the
        # process-wide writer boundary and a writable open here would be a
        # second in-process writer (polylogue-8qm4k).
        with contextlib.closing(sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)) as conn:
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
                enable_api=False,
                service_profile=ServiceProfile.REPLAY,
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
            stack.enter_context(patch.object(daemon_cli, "_SESSION_PROFILE_AUDIT_INTERVAL_SECONDS", 0.05))
            await run_until_observed_sweep()
            # This fixture writes derived output out from under the daemon and
            # retains the matching transaction-owned demand obligation. The
            # periodic owner must discover that obligation without a file hint.
            from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
            from polylogue.storage.sqlite.write_lease import write_lease

            def clear_projection() -> None:
                with (
                    write_lease("test.fixture.profile-demand", archive_root=archive_root),
                    contextlib.closing(
                        open_isolated_write_connection(
                            archive_root / "index.db", purpose="test.fixture.profile-demand", archive_root=archive_root
                        )
                    ) as conn,
                ):
                    assert conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                    for table in ("session_latency_profiles", "session_profiles"):
                        conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                    conn.execute(
                        "INSERT INTO session_profile_demand(session_id, revision) VALUES (?, 1) "
                        "ON CONFLICT(session_id) DO UPDATE SET revision = revision + 1",
                        (session_id,),
                    )
                    conn.commit()

            await asyncio.to_thread(clear_projection)
            assert not profile_exists()

            await run_until_observed_sweep()
    finally:
        assert reset_compute_adapter(join_timeout_s=5) == ()

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
    from polylogue.core.compute import reset_compute_adapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.intake import FairIntakeDispatcher
    from polylogue.daemon.intake_adapters import DaemonIntakeService
    from polylogue.daemon.services import ServiceProfile
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.sources.live.watcher import LiveWatcher

    archive_root = tmp_path / "archive"
    from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop

    # The law is intake, not first bootstrap: a fresh archive's startup
    # applies every declared durable migration before intake begins.
    archive_root.mkdir()
    run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
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
    intake_outcomes: list[tuple[str, str, str, str | None]] = []
    carriers: list[Path] = []
    browser = source_name == "browser-capture"
    native_id = "aaaa0000-0000-0000-0000-000000000001"
    session_id = "chatgpt-export:intake-law" if browser else f"claude-code-session:{native_id}"
    real_pass = FairIntakeDispatcher.run_once
    real_admit_page = FairIntakeDispatcher._admit_page
    real_ingest = LiveWatcher._ingest_files
    real_kernel = DaemonConverger.converge_derivations
    real_promoted = ComposedSessionProfiles.converge_promoted

    async def delayed_promoted(self: ComposedSessionProfiles) -> DerivationReport:
        # The active pointer becomes readable before its profile pass finishes.
        # Hold that real pass past the old one-second observation window.
        await asyncio.sleep(1.25)
        return await real_promoted(self)

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

    async def record_admit_page(self: FairIntakeDispatcher, spec: Any, items: Any) -> Any:
        results = await real_admit_page(self, spec, items)
        intake_outcomes.extend(
            (spec.name, item.item_id, result.outcome.value, result.reason)
            for item, result in zip(items, results, strict=True)
        )
        return results

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
        await asyncio.Event().wait()

    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    assert reset_compute_adapter(join_timeout_s=5) == ()
    try:
        with contextlib.ExitStack() as stack:
            _daemon_startup_stubs(stack, daemon_cli, archive_root)
            stack.enter_context(patch.object(daemon_cli, "Polylogue", lambda: RealPolylogue(archive_root=archive_root)))
            stack.enter_context(patch.object(daemon_cli, "daemon_write_coordinator", return_value=coordinator))
            stack.enter_context(patch.object(FairIntakeDispatcher, "run_once", dispatch))
            stack.enter_context(patch.object(FairIntakeDispatcher, "_admit_page", record_admit_page))
            stack.enter_context(patch.object(LiveWatcher, "_ingest_files", ingest))
            stack.enter_context(patch.object(DaemonConverger, "converge_derivations", kernel))
            if browser:
                stack.enter_context(patch.object(ComposedSessionProfiles, "converge_promoted", delayed_promoted))
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
                        WatchSource(
                            name=source_name,
                            root=source_root,
                            layout=export_drop_layout((".json" if browser else ".jsonl",)),
                        ),
                    ),
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    enable_api=False,
                    enable_source_catchup=False,
                    service_profile=ServiceProfile.INTAKE,
                )
            )
            try:
                try:
                    # Startup, the first admission, and the changed revision
                    # each have their own bounded work. A single wall-clock
                    # window can expire after a successful first admission
                    # before the second is even scheduled.
                    await asyncio.wait_for(first_pass.wait(), timeout=20)
                    await asyncio.wait_for(first_admission.wait(), timeout=20)
                    await asyncio.wait_for(completed.wait(), timeout=20)
                except TimeoutError:
                    if task.done():
                        await task
                    pytest.fail(
                        f"intake did not complete: outcomes={intake_outcomes[-12:]}, "
                        f"passes={passes}, carriers={carriers}, admissions={admission_tasks}"
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
                # observations its own admissions produce, so the raw reports
                # are separated from the session-scoped ones here.
                #
                # Session scope carries three ordered domains
                # (session_summary -> session_usage_rollup -> session_profile)
                # and one bounded pass converges as many of them as it can, so
                # counting passes whose ``done`` is exactly 1 names nothing
                # about the archive. Inspect this session's persisted profile
                # below. ``outcomes`` is a retained sample and may be empty.
                session_reports = [
                    report for report in kernel_reports if not isinstance(report.frame.scope, RawObservationScope)
                ]
                assert session_reports
                failed = sum(report.count(Outcome.FAILED) for report in session_reports)
                assert failed == 0, str(
                    [
                        (item.key, item.outcome, item.reason, item.error)
                        for report in session_reports
                        for item in report.outcomes
                    ]
                )

                # A returning intake pass is not the point at which its index
                # publication becomes visible: the replace commits shortly
                # after, so reading the index the instant ``completed`` fires
                # observes the previous revision (measured on this test: the
                # claude case reads one message for roughly 200ms before the
                # second admission's replace lands). Wait for the converged
                # count and then require it to STAY there. The settle window is
                # what keeps the browser expectation honest: a competing
                # snapshot that was wrongly admitted would move the count off
                # one during the window instead of passing a lucky first read.
                # ``sqlite3.connect`` as a context manager ends the
                # transaction but does NOT close the connection, so each probe
                # closes its own read explicitly -- sampling in a loop with the
                # bare ``with`` form leaks one descriptor per tick and trips the
                # suite's descriptor-balance teardown check.
                def index_probe() -> tuple[int, list[tuple[str, ...]]]:
                    with contextlib.closing(
                        sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
                    ) as conn:
                        return (
                            int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]),
                            conn.execute("SELECT session_id FROM session_profiles").fetchall(),
                        )

                def message_count() -> int:
                    return index_probe()[0]

                for _ in range(200):
                    if message_count() == expected_versions:
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail(
                        f"index never converged to {expected_versions} message(s): {message_count()}; "
                        f"membership evidence={evidence}"
                    )
                # The active pointer exposes messages before the promotion
                # callback has finished deriving profiles. Wait for that
                # observable publication with its own deadline, then hold the
                # message-count stability window. The browser profile may be
                # withdrawn by the competing quarantined snapshot's replace,
                # so retain every observed row rather than requiring it at
                # the end. Bypassing the canonical kernel never satisfies the
                # publication wait.
                observed_profiles: list[list[tuple[str, ...]]] = [index_probe()[1]]
                for _ in range(200):
                    if [(session_id,)] in observed_profiles:
                        break
                    await asyncio.sleep(0.05)
                    settled_count, profile_rows = index_probe()
                    assert settled_count == expected_versions
                    observed_profiles.append(profile_rows)
                for _ in range(20):
                    await asyncio.sleep(0.05)
                    settled_count, profile_rows = index_probe()
                    assert settled_count == expected_versions
                    observed_profiles.append(profile_rows)
                assert [(session_id,)] in observed_profiles, str(
                    {
                        "profiles": observed_profiles,
                        "reports": [
                            [(item.key, item.outcome, item.reason, item.error) for item in report.outcomes]
                            for report in session_reports
                        ],
                        "scopes": [str(report.frame.scope) for report in session_reports],
                        "changed_sessions": [getattr(item, "changed_session_ids", ()) for item in admission_metrics],
                    }
                )
                assert all(rows in ([], [(session_id,)]) for rows in observed_profiles), str(observed_profiles)
                if browser:
                    # Intake admits the competing snapshot; its retained
                    # publication records the decision afterwards. Read the
                    # settled evidence, not the admission-time sample.
                    decided = [(MembershipDecision.AMBIGUOUS.value, RawRevisionAuthority.QUARANTINED.value)] * 2
                    for _ in range(200):
                        with contextlib.closing(
                            sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
                        ) as conn:
                            evidence = conn.execute(
                                "SELECT decision, revision_authority FROM raw_session_memberships "
                                "WHERE logical_source_key = ?",
                                (session_id,),
                            ).fetchall()
                        if evidence == decided:
                            break
                        await asyncio.sleep(0.05)
                    assert evidence == decided
                    assert message_count() == expected_versions
                assert all(path.is_file() for path in carriers)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=10)
    finally:
        assert reset_compute_adapter(join_timeout_s=5) == ()


def test_run_daemon_services_closes_browser_capture_server_on_failure() -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.services import ServiceProfile

    async def noop() -> None:
        return None

    class FakeServer:
        @property
        def config(self) -> BrowserCaptureReceiverConfig:
            return BrowserCaptureReceiverConfig(spool_path=browser_capture_spool_root())

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
                service_profile=ServiceProfile.SURFACES,
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

    class EmptyCursor:
        def has_pending_retries(self, roots: Iterable[Path] | None = None) -> bool:
            return False

        def release_deferred_convergence_debt(self) -> int:
            return 0

    class FakeWatcher(_NoIntakeHints):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._cursor = EmptyCursor()

        async def run(self) -> None:
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            return None

    class BlockingServer:
        @property
        def config(self) -> BrowserCaptureReceiverConfig:
            return BrowserCaptureReceiverConfig(spool_path=browser_capture_spool_root())

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
        async with asyncio.timeout(5.0):
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signal shutdown defers OPS recovery, while ordinary shutdown performs it."""
    from polylogue.daemon import cli as daemon_cli
    from tests.infra.archive_templates import bootstrap_archive_root

    # The law is shutdown, not first bootstrap: a fresh archive's startup
    # applies every declared durable migration before the servers start.
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    bootstrap_archive_root(archive_root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

    class BlockingServer:
        @property
        def config(self) -> BrowserCaptureReceiverConfig:
            return BrowserCaptureReceiverConfig(spool_path=browser_capture_spool_root())

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

    async def no_drive_changes(_callback: object, *, raw_owner: object, compute_owner: object) -> DriveCatchupReport:
        assert raw_owner is api_server.operation_runtime.raw_observation_owner
        assert compute_owner is api_server.execution_kernel
        return DriveCatchupReport(DriveCatchupState.COMPLETE)

    async def wait_forever(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    browser_server = BlockingServer()
    api_server = APIBlockingServer()
    from polylogue.core.compute import reset_compute_adapter

    def make_api_server(*_args: object, **kwargs: object) -> APIBlockingServer:
        kernel = kwargs["execution_kernel"]
        assert isinstance(kernel, BoundedComputeAdapter)
        api_server.execution_kernel = kernel
        bridge = kwargs["write_bridge"]
        root = kwargs["archive_root"]
        assert isinstance(bridge, DaemonWriteThreadBridge)
        assert isinstance(root, Path)
        api_server.operation_runtime.raw_observation_owner = RawObservationConvergenceOwner(
            root,
            compute_adapter=kernel,
            write_bridge=bridge,
            write_coordinator=bridge.coordinator,
        )
        return api_server

    api_server.session_profile_callback = None

    async def shutdown_operation_runtime() -> None:
        return None

    api_server.operation_runtime = SimpleNamespace(
        shutdown=shutdown_operation_runtime,
        accepted_ingest_redrive_claimed=_no_redrive_claims,
        embedding_convergence=None,
    )
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
        patch.object(daemon_cli, "_census_lineage_startup_sync", _converged_lineage_census),
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
        patch.object(daemon_cli, "_configure_fts_automerge", noop),
        patch.object(daemon_cli, "_run_drive_source_catchup_safely", no_drive_changes),
        patch.object(daemon_cli, "_periodic_raw_materialization_convergence", lambda **_kwargs: wait_forever()),
        patch("polylogue.daemon.blob_gc_periodic.periodic_blob_gc_check", lambda **_kwargs: wait_forever()),
        patch.object(daemon_cli, "_periodic_wal_checkpoint", wait_forever),
        patch.object(daemon_cli, "_periodic_fts_merge", wait_forever),
        patch(
            "polylogue.daemon.blob_gc_periodic.periodic_blob_publication_reconciliation_check",
            lambda **_kwargs: wait_forever(),
        ),
        patch.object(daemon_cli, "_periodic_heartbeat", wait_forever),
        patch.object(daemon_cli, "_periodic_health_check", wait_forever),
        patch.object(daemon_cli, "_periodic_db_optimize", wait_forever),
        patch.object(daemon_cli, "_periodic_status_snapshot_refresh", wait_forever),
        patch.object(daemon_cli, "_periodic_convergence_check", lambda _sources, **_kwargs: wait_forever()),
        patch.object(daemon_cli, "_periodic_session_profile_audit", lambda _callback, **_kwargs: wait_forever()),
        patch.object(daemon_cli, "_mark_interrupted_live_ingest_attempts_on_shutdown", mark_interrupted_cleanup),
        patch("polylogue.daemon.embedding_backlog.periodic_embedding_backlog_check", lambda **_kwargs: wait_forever()),
        patch("polylogue.daemon.convergence.DaemonConverger", return_value=FakeConverger()),
        patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=()),
        patch("polylogue.daemon.http.DaemonAPIHTTPServer", side_effect=make_api_server),
    )
    with contextlib.ExitStack() as stack:
        for scoped_patch in patches:
            stack.enter_context(scoped_patch)
        assert reset_compute_adapter(join_timeout_s=5) == ()
        try:
            asyncio.run(exercise())
        finally:
            assert reset_compute_adapter(join_timeout_s=5) == ()

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
        @property
        def config(self) -> BrowserCaptureReceiverConfig:
            return BrowserCaptureReceiverConfig(spool_path=browser_capture_spool_root())

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

    async def fake_health_check(**_kwargs: object) -> None:
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
        patch.object(daemon_cli, "_periodic_session_profile_audit", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_health_check", fake_health_check),
        patch.object(daemon_cli, "_periodic_db_optimize", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_status_snapshot_refresh", side_effect=fail_background_work),
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


def test_periodic_raw_materialization_wakes_fair_intake_without_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The periodic route wakes fair intake without selecting a raw itself."""
    from polylogue.daemon import cli as daemon_cli

    wakeup = asyncio.Event()

    async def stop_after_one_tick(seconds: float) -> None:
        assert (
            daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS
            <= seconds
            <= daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS * 1.1
        )
        assert wakeup.is_set()
        raise asyncio.CancelledError

    with patch("asyncio.sleep", side_effect=stop_after_one_tick), pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon_cli._periodic_raw_materialization_convergence(raw_intake_wakeup=wakeup))


@pytest.mark.parametrize("watcher_initially_registered", [False, True])
def test_periodic_raw_materialization_respects_watcher_registration_gate(
    monkeypatch: pytest.MonkeyPatch,
    watcher_initially_registered: bool,
) -> None:
    """Periodic raw maintenance starts only after watcher registration."""
    from polylogue.daemon import cli as daemon_cli

    async def exercise() -> bool:
        watcher_registered = asyncio.Event()
        if watcher_initially_registered:
            watcher_registered.set()
        raw_intake_wakeup = asyncio.Event()
        task = asyncio.create_task(
            daemon_cli._periodic_raw_materialization_convergence(
                watcher_registered=watcher_registered,
                raw_intake_wakeup=raw_intake_wakeup,
            )
        )
        await asyncio.sleep(0)
        if not watcher_initially_registered:
            assert not raw_intake_wakeup.is_set()
            watcher_registered.set()

            async def stop_after_one_tick(seconds: float) -> None:
                assert (
                    daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS
                    <= seconds
                    <= daemon_cli._RAW_MATERIALIZATION_CONVERGENCE_INTERVAL_SECONDS * 1.1
                )
                raise asyncio.CancelledError

            with patch("asyncio.sleep", side_effect=stop_after_one_tick), pytest.raises(asyncio.CancelledError):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        return raw_intake_wakeup.is_set()

    assert asyncio.run(exercise()) is True


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
        "polylogue.operations.raw_observation_owner.make_raw_observation_derivation",
        lambda *_args, **_kwargs: Derivation(),
    )
    owner = RawObservationConvergenceOwner(
        tmp_path,
        compute_adapter=cast(BoundedComputeAdapter, object()),
        write_bridge=cast(DaemonWriteThreadBridge, object()),
        write_coordinator=cast(DaemonWriteCoordinator, object()),
    )

    with pytest.raises(RuntimeError, match=pattern):
        owner._require_source_frontier_authority(raw_id)
    assert observed_raw_ids == [(raw_id,)]


def test_raw_observation_publication_holds_writer_lease_through_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canonical replay applies its prepared publication under the writer lease.

    Real retained bytes are acquired and replayed through the canonical owner;
    every prepared replay apply must observe a held ``ActiveWriterLease``, and
    the raw frontier gate is consulted for the published raw first.
    Anti-vacuity: acquiring the lease after, or releasing it before, the
    prepared apply records an unheld apply; skipping the frontier gate leaves
    ``blocked_checks`` without the raw.
    """
    import asyncio

    from polylogue.core.enums import Provider
    from polylogue.sources import revision_backfill
    from polylogue.storage import index_generation, raw_retention
    from tests.infra.retained_replay import publish_retained_payload

    leases: list[index_generation.ActiveWriterLease] = []
    replay_held: list[bool] = []
    blocked_checks: list[tuple[str, ...]] = []

    class RecordingLease(index_generation.ActiveWriterLease):
        def __init__(self, archive_root: Path) -> None:
            super().__init__(archive_root)
            leases.append(self)

    real_apply = revision_backfill.apply_prepared_revision_replay
    real_blocked = raw_retention.raw_frontier_blocked_raw_ids

    def recording_apply(*args: Any, **kwargs: Any) -> Any:
        replay_held.append(any(lease.held for lease in leases))
        return real_apply(*args, **kwargs)

    def recording_blocked(root: Path, raw_ids: Sequence[str]) -> Any:
        blocked_checks.append(tuple(raw_ids))
        return real_blocked(root, raw_ids)

    monkeypatch.setattr(index_generation, "ActiveWriterLease", RecordingLease)
    monkeypatch.setattr(revision_backfill, "apply_prepared_revision_replay", recording_apply)
    monkeypatch.setattr(raw_retention, "raw_frontier_blocked_raw_ids", recording_blocked)

    payload = (
        b'{"type":"session_meta","payload":{"id":"lease-replay","timestamp":"2025-01-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user","content":'
        b'[{"type":"input_text","text":"Hello"}]}}\n'
    )
    raw_id, written = asyncio.run(
        publish_retained_payload(
            tmp_path / "archive",
            provider=Provider.CODEX,
            payload=payload,
            source_path=str(tmp_path / "sessions" / "lease-replay.jsonl"),
            acquired_at_ms=1,
        )
    )

    assert written
    assert replay_held and all(replay_held)
    assert (raw_id,) in blocked_checks
    assert not any(lease.held for lease in leases)


def test_raw_owner_cancellation_stops_preparation_and_the_next_pass_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cancelled owner stops its preparation; the raw stays pending, not lost.

    Since #5691 the owner publishes its cancellation to the compute pass
    (``compute_cancel``), and retained preparation stops at its next check
    instead of publishing after the owner is gone. Anti-vacuity: a cancelled
    pass that publishes anyway fails the zero-session check, and one that
    leaves the raw terminal fails the next pass's session and FTS checks.
    """
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.enums import Provider
    from polylogue.core.write_lease import coordinator_write_lease_active
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import (
        DaemonWriteCoordinator,
        DaemonWriteThreadBridge,
    )
    from polylogue.sources import revision_backfill
    from polylogue.storage.derived.raw import RawFrame, RawObservationDerivation, RawObservationReplacement
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
            canonical_source_path="cancelled-owner.json",
            acquired_at_ms=1,
        )

    async def scenario() -> None:
        compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
        coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
        owner = RawObservationConvergenceOwner(
            tmp_path,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            write_coordinator=coordinator,
        )
        adapter, index_path, index_destination = owner._archive.destination_adapter()
        monkeypatch.setattr(owner._archive, "destination_adapter", lambda: (adapter, index_path, index_destination))
        original_compute = RawObservationDerivation.compute
        original_publish = RawObservationDerivation.publish
        publication_flags: list[tuple[bool, bool, bool, bool]] = []
        publication_line_sites: list[tuple[int, ...]] = []
        replay_failures: list[tuple[tuple[str, str], ...]] = []
        membership_gate: list[tuple[bool, int, int, bool, bool]] = []
        original_replay = revision_backfill.apply_prepared_revision_replay

        def observed_replay(*args: Any, **kwargs: Any) -> Any:
            try:
                return original_replay(*args, **kwargs)
            except BaseException as failure:
                causes: list[tuple[str, str]] = []
                current: BaseException | None = failure
                seen: set[int] = set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    causes.append((type(current).__name__, str(current)))
                    current = current.__cause__
                replay_failures.append(tuple(causes))
                raise

        monkeypatch.setattr(revision_backfill, "apply_prepared_revision_replay", observed_replay)
        started = threading.Event()
        loop = asyncio.get_running_loop()
        compute_started = loop.create_future()
        release = threading.Event()

        def paused_compute(
            instance: RawObservationDerivation, frame: RawFrame, key: str, **kwargs: Any
        ) -> RawObservationReplacement:
            assert not coordinator_write_lease_active()
            started.set()
            loop.call_soon_threadsafe(lambda: None if compute_started.done() else compute_started.set_result(None))
            release.wait()
            return original_compute(instance, frame, key, **kwargs)

        monkeypatch.setattr(RawObservationDerivation, "compute", paused_compute)

        def observed_publish(
            instance: RawObservationDerivation, frame: RawFrame, replacement: RawObservationReplacement, **kwargs: Any
        ) -> bool:
            assert isinstance(replacement, RawObservationReplacement)
            flags = (
                replacement.already_valid,
                replacement.needs_source_census,
                replacement.needs_source_classification,
            )
            previous_trace = sys.gettrace()
            observed_lines: list[int] = []

            def observed_line(current: Any, event: str, argument: object) -> Any:
                if current.f_code is original_replay.__code__:
                    if event == "exception" and isinstance(argument, tuple):
                        failure = argument[1]
                        if isinstance(failure, revision_backfill.RetainedPreparationRetryableError) and str(
                            failure
                        ).startswith("prepared membership authority changed for "):
                            fields = current.f_locals
                            plan = fields.get("membership_plan")
                            candidates = fields.get("candidate_raw_ids", ())
                            membership_gate.append(
                                (
                                    plan is not None,
                                    len(plan.candidate_raw_ids) if plan is not None else 0,
                                    len(candidates),
                                    plan.head_raw_id == fields.get("head_raw_id") if plan is not None else False,
                                    plan.candidate_raw_ids == tuple(sorted(candidates)) if plan is not None else False,
                                )
                            )
                    return observed_line
                if current.f_code is not original_publish.__code__:
                    return None
                if event == "line":
                    observed_lines.append(current.f_lineno)
                return observed_line

            # Do not replace an existing diagnostic/coverage trace. A missing
            # sequence then explicitly means this observation was unavailable.
            if previous_trace is None:
                sys.settrace(observed_line)
            try:
                result = original_publish(instance, frame, replacement, **kwargs)
            finally:
                sys.settrace(previous_trace)
                publication_line_sites.append(tuple(observed_lines))
            publication_flags.append((*flags, result))
            return result

        monkeypatch.setattr(RawObservationDerivation, "publish", observed_publish)
        task = asyncio.create_task(owner.replay_retained_raw_ids((raw_id,)))
        try:
            done, _ = await asyncio.wait((task, compute_started), return_when=asyncio.FIRST_COMPLETED)
            assert compute_started in done and started.is_set(), task.result() if task in done else None
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            with sqlite3.connect(tmp_path / "index.db") as conn:
                assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
            next_report = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
            with sqlite3.connect(tmp_path / "index.db") as conn:
                assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,), (
                    next_report,
                    publication_flags,
                    publication_line_sites,
                    replay_failures,
                    membership_gate,
                )
                assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] > 0
        finally:
            release.set()
            if not task.done():
                await task
            assert await coordinator.shutdown(timeout=2.0) is True
            compute.shutdown(wait=True)

    asyncio.run(scenario())


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

    # Modules that bind ``archive_root`` at import are imported before the
    # patch, or the first test to import them lazily would pin its own root
    # for every later test. The environment names the same root, so the
    # real resolver and the patched one agree.
    import polylogue.daemon.events
    import polylogue.daemon.lifecycle  # noqa: F401

    stack.enter_context(patch.dict(os.environ, {"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path)}))
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
    stack.enter_context(patch.object(daemon_cli, "_census_lineage_startup_sync", _converged_lineage_census))
    stack.enter_context(patch.object(daemon_cli, "_reconcile_blob_publications", _noop))
    stack.enter_context(patch.object(daemon_cli, "_configure_fts_automerge", _noop))
    stack.enter_context(
        patch(
            "polylogue.operations.mutation_replay.recover_interrupted_operations",
            lambda _root, *, resolver_actor_ref, input_demand, startup: None,
        )
    )
    stack.enter_context(patch.object(daemon_cli, "_mark_interrupted_live_ingest_attempts_on_shutdown"))
    stack.enter_context(patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=()))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    (
        ("busy", "sqlite_busy"),
        ("cantopen", "sqlite_open_unavailable"),
        ("ioerr", "candidate_storage_unavailable"),
        ("ioerr_read", "candidate_storage_unavailable"),
    ),
)
async def test_cold_build_transient_sqlite_settlement_retries_in_running_daemon(
    tmp_path: Path, fault: str, expected_reason: str
) -> None:
    from polylogue import Polylogue as RealPolylogue
    from polylogue.core.compute import reset_compute_adapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.catchup_status import _cold_build_settlement
    from polylogue.daemon.intake_adapters import DaemonIntakeService
    from polylogue.daemon.services import ServiceProfile
    from polylogue.sources.live.cold_build import ColdBuildGeneration, active_cold_build_generation
    from polylogue.sources.live.production_baseline import ProductionSourceBaseline
    from polylogue.storage.derived import raw as raw_derivation
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

    archive_root = tmp_path / "archive"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "session.jsonl"
    source_file.write_text(
        '{"type":"session_meta","payload":{"id":"cold-retry","timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        '"role":"user","content":[{"type":"input_text","text":"Synthetic"}]}}\n',
        encoding="utf-8",
    )
    os.utime(source_file, (1.0, 1.0))
    lock_path = tmp_path / "lock.db"
    holder = sqlite3.connect(lock_path)
    contender = sqlite3.connect(lock_path, timeout=0)
    try:
        holder.execute("CREATE TABLE lock_probe (id INTEGER)")
        holder.commit()
        holder.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError) as busy:
            contender.execute("INSERT INTO lock_probe VALUES (1)")
        assert busy.value.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY
    finally:
        contender.close()
        holder.close()

    missing_database = tmp_path / "temporarily-unavailable.db"
    with pytest.raises(sqlite3.OperationalError) as cantopen:
        sqlite3.connect(f"file:{missing_database}?mode=ro", uri=True)
    assert cantopen.value.sqlite_errorcode & 0xFF == sqlite3.SQLITE_CANTOPEN

    ioerr = sqlite3.OperationalError("synthetic transient candidate read failure")
    ioerr.sqlite_errorcode = sqlite3.SQLITE_IOERR_READ if fault == "ioerr_read" else sqlite3.SQLITE_IOERR
    real_readiness = ArchiveStore.run_generation_readiness_pass
    real_verify = ProductionSourceBaseline.verify
    calls = 0
    cleanup_owners: dict[int, tuple[object, ...]] = {}
    real_cleanup = raw_derivation._cleanup_scratch

    def observed_cleanup(scratch: tempfile.TemporaryDirectory[str]) -> None:
        # Observe on the actual creator immediately before delegating unchanged.
        for owner in retained_native_sql_owners_for_lifetime(scratch):
            cleanup_owners.setdefault(
                id(owner),
                (
                    id(owner),
                    owner._connection_identity,
                    owner.connection is None,
                    owner._settled,
                    owner.close_required,
                    id(owner._terminal_parent),
                    type(owner._terminal_parent).__name__,
                    owner.thread is threading.current_thread(),
                    any(item is scratch for item in owner._lifetime_dependencies),
                ),
            )
        real_cleanup(scratch)

    def busy_once(self: ArchiveStore) -> None:
        nonlocal calls
        calls += 1
        if fault != "cantopen" and calls == 1:
            raise busy.value if fault == "busy" else ioerr
        real_readiness(self)

    verify_calls = 0

    def cantopen_once(self: ProductionSourceBaseline, source_db: Path) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if fault == "cantopen" and verify_calls == 1:
            raise cantopen.value
        real_verify(self, source_db)

    assert reset_compute_adapter(join_timeout_s=5) == ()
    try:
        with contextlib.ExitStack() as stack:
            _daemon_startup_stubs(stack, daemon_cli, archive_root)
            stack.enter_context(patch.object(daemon_cli, "Polylogue", lambda: RealPolylogue(archive_root=archive_root)))
            stack.enter_context(patch.object(ArchiveStore, "run_generation_readiness_pass", busy_once))
            stack.enter_context(patch.object(ProductionSourceBaseline, "verify", cantopen_once))
            stack.enter_context(patch.object(raw_derivation, "_cleanup_scratch", observed_cleanup))
            stack.enter_context(
                patch(
                    "polylogue.daemon.intake_adapters.DaemonIntakeService",
                    lambda dispatcher, **kwargs: DaemonIntakeService(dispatcher, idle_delay_s=0.05, **kwargs),
                )
            )
            task = asyncio.create_task(
                daemon_cli.run_daemon_services(
                    sources=(WatchSource("codex", source_root, layout=export_drop_layout((".jsonl",))),),
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    enable_api=False,
                    enable_source_catchup=False,
                    service_profile=ServiceProfile.INTAKE,
                )
            )
            primary_failure: BaseException | None = None
            try:
                async with asyncio.timeout(20):
                    while _cold_build_settlement().get("cold_build_settlement_state") != "retryable":
                        if task.done():
                            await task
                        await asyncio.sleep(0.05)
                candidate = active_cold_build_generation(archive_root)
                assert isinstance(candidate, ColdBuildGeneration)
                candidate_id = candidate.generation_id
                with contextlib.closing(
                    sqlite3.connect(f"file:{candidate.generation.index_path}?mode=ro", uri=True)
                ) as candidate_reader:
                    assert candidate_reader.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
                assert _cold_build_settlement()["cold_build_candidate_id"] == candidate_id
                assert _cold_build_settlement()["cold_build_settlement_reason"] == expected_reason
                with contextlib.closing(
                    sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
                ) as active:
                    assert active.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
                assert not task.done()
                async with asyncio.timeout(15):
                    while _cold_build_settlement().get("cold_build_settlement_state") != "complete":
                        if task.done():
                            await task
                        await asyncio.sleep(0.05)
                assert (verify_calls if fault == "cantopen" else calls) == 2
                assert candidate.publication_complete
                assert _cold_build_settlement()["cold_build_candidate_id"] == candidate_id
                with contextlib.closing(
                    sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
                ) as active:
                    assert active.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            except BaseException as failure:
                primary_failure = failure
                failure.add_note(
                    f"settlement fault={fault}, readiness_calls={calls}, verification_calls={verify_calls}, "
                    f"state={_cold_build_settlement().get('cold_build_settlement_state')}, "
                    f"reason={_cold_build_settlement().get('cold_build_settlement_reason')}, "
                    f"error={_cold_build_settlement().get('cold_build_settlement_last_error')}"
                )
                failure.add_note(f"original scratch cleanup owners={tuple(cleanup_owners.values())!r}")
                thread_sites: list[tuple[str, tuple[tuple[str, str, int], ...]]] = []
                thread_names = {thread.ident: thread.name for thread in threading.enumerate()}
                for thread_id, thread_frame in sys._current_frames().items():
                    if thread_id == threading.get_ident():
                        continue
                    sites: list[tuple[str, str, int]] = []
                    frame: FrameType | None = thread_frame
                    while frame is not None:
                        filename = frame.f_code.co_filename
                        if "/polylogue/" in filename:
                            sites.append((filename.split("/polylogue/", 1)[1], frame.f_code.co_name, frame.f_lineno))
                        frame = frame.f_back
                    if sites:
                        thread_sites.append((thread_names.get(thread_id, "unknown"), tuple(sites)))
                failure.add_note(f"observed publication worker sites={thread_sites!r}")
                raise
            finally:
                task.cancel()
                try:
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=10)
                except BaseException as cleanup:
                    if primary_failure is not None:
                        raise builtins.BaseExceptionGroup(
                            "cold settlement and daemon cleanup failed", [primary_failure, cleanup]
                        ) from None
                    raise
    finally:
        assert reset_compute_adapter(join_timeout_s=5) == ()


def test_cold_build_settlement_classifies_typed_faults(tmp_path: Path) -> None:
    from polylogue.core.durable_fs import DurableFilesystemError
    from polylogue.daemon.intake_adapters import classify_cold_build_settlement_failure
    from polylogue.maintenance.candidate_capacity import ArchiveCapacityError
    from polylogue.operations.cold_build_coverage import ColdBuildCoverageError
    from polylogue.sources.live.production_baseline import (
        ProductionBaselineError,
        ProductionBaselineReadUnavailableError,
    )
    from polylogue.storage.archive_identity import ArchiveLocationError
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError, ReferenceSealStaleError

    assert classify_cold_build_settlement_failure(ProductionBaselineError("missing source revision")) == (
        "source_integrity",
        False,
    )
    assert classify_cold_build_settlement_failure(ProductionBaselineReadUnavailableError("read failed")) == (
        "source_integrity",
        True,
    )
    assert classify_cold_build_settlement_failure(
        ColdBuildCoverageError(missing_count=1, first_missing_session_id="codex:synthetic")
    ) == ("active_coverage_incomplete", False)
    assert classify_cold_build_settlement_failure(ReferenceSealStaleError("snapshot changed")) == (
        "promotion_evidence_changed",
        True,
    )
    assert classify_cold_build_settlement_failure(ReferenceSealError("durable ref cannot be preserved")) == (
        "durable_reference_preservation",
        False,
    )
    assert classify_cold_build_settlement_failure(RuntimeError("database is locked")) is None
    assert classify_cold_build_settlement_failure(OSError(errno.EIO, "transient pointer I/O")) == (
        "storage_io_unavailable",
        True,
    )
    not_database = tmp_path / "not-a-database.db"
    not_database.write_bytes(b"not a SQLite database")
    with contextlib.closing(sqlite3.connect(not_database)) as conn:
        with pytest.raises(sqlite3.DatabaseError) as corrupt:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    assert corrupt.value.sqlite_errorcode & 0xFF == sqlite3.SQLITE_NOTADB
    assert classify_cold_build_settlement_failure(corrupt.value) is None
    try:
        raise DurableFilesystemError("receipt publication failed") from OSError(errno.ENOSPC, "full")
    except DurableFilesystemError as wrapped:
        assert classify_cold_build_settlement_failure(wrapped) == ("capacity_unavailable", False)
    try:
        raise ArchiveCapacityError("capacity inventory unavailable") from OSError(errno.EIO, "temporary scan failure")
    except ArchiveCapacityError as wrapped:
        assert classify_cold_build_settlement_failure(wrapped) == ("capacity_inventory_unavailable", True)
    try:
        try:
            raise ArchiveLocationError("cannot inspect active index pointer") from PermissionError(
                errno.EACCES, "pointer unavailable"
            )
        except ArchiveLocationError as location_error:
            raise ArchiveCapacityError("cannot resolve archive identity") from location_error
    except ArchiveCapacityError as wrapped:
        assert classify_cold_build_settlement_failure(wrapped) == ("capacity_inventory_unavailable", True)
    try:
        raise ArchiveCapacityError("cannot resolve archive identity") from ArchiveLocationError("invalid pointer")
    except ArchiveCapacityError as malformed:
        assert classify_cold_build_settlement_failure(malformed) is None
    assert classify_cold_build_settlement_failure(ArchiveCapacityError("invalid capacity topology")) is None


@pytest.mark.asyncio
async def test_cold_build_integrity_fault_stays_blocked_in_running_daemon(tmp_path: Path) -> None:
    from polylogue import Polylogue as RealPolylogue
    from polylogue.core.compute import reset_compute_adapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.catchup_status import _cold_build_settlement
    from polylogue.daemon.intake_adapters import DaemonIntakeService
    from polylogue.daemon.services import ServiceProfile
    from polylogue.sources.live.cold_build import active_cold_build_generation
    from polylogue.sources.live.production_baseline import ProductionBaselineError, ProductionSourceBaseline

    archive_root = tmp_path / "archive"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "session.jsonl"
    source_file.write_text(
        '{"type":"session_meta","payload":{"id":"cold-blocked","timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        '"role":"user","content":[{"type":"input_text","text":"Synthetic"}]}}\n',
        encoding="utf-8",
    )
    os.utime(source_file, (1.0, 1.0))
    verifications = 0

    def refuse_integrity(self: ProductionSourceBaseline, _source_db: Path) -> None:
        nonlocal verifications
        verifications += 1
        raise ProductionBaselineError("missing retained revision")

    assert reset_compute_adapter(join_timeout_s=5) == ()
    try:
        with contextlib.ExitStack() as stack:
            _daemon_startup_stubs(stack, daemon_cli, archive_root)
            stack.enter_context(patch.object(daemon_cli, "Polylogue", lambda: RealPolylogue(archive_root=archive_root)))
            stack.enter_context(patch.object(ProductionSourceBaseline, "verify", refuse_integrity))
            stack.enter_context(
                patch(
                    "polylogue.daemon.intake_adapters.DaemonIntakeService",
                    lambda dispatcher, **kwargs: DaemonIntakeService(dispatcher, idle_delay_s=0.05, **kwargs),
                )
            )
            task = asyncio.create_task(
                daemon_cli.run_daemon_services(
                    sources=(WatchSource("codex", source_root, layout=export_drop_layout((".jsonl",))),),
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    enable_api=False,
                    enable_source_catchup=False,
                    service_profile=ServiceProfile.INTAKE,
                )
            )
            try:
                async with asyncio.timeout(20):
                    while _cold_build_settlement().get("cold_build_settlement_state") != "blocked":
                        if task.done():
                            await task
                        await asyncio.sleep(0.05)
                status = _cold_build_settlement()
                candidate = active_cold_build_generation(archive_root)
                assert candidate is not None
                assert status["cold_build_candidate_id"] == candidate.generation_id
                assert status["cold_build_settlement_reason"] == "source_integrity"
                assert status["cold_build_settlement_attempts"] == 1
                assert status["cold_build_settlement_retry_due_in_s"] is None
                assert verifications == 1
                await asyncio.sleep(0.3)
                assert _cold_build_settlement()["cold_build_settlement_attempts"] == 1
                assert verifications == 1
                assert not task.done()
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=10)
            assert candidate.discarded
            assert not candidate.generation_root.exists()
    finally:
        assert reset_compute_adapter(join_timeout_s=5) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("imported_source_kept", [False, True])
async def test_explicit_cold_build_keeps_sessions_the_active_index_serves(
    tmp_path: Path, imported_source_kept: bool
) -> None:
    """``--cold-build-index`` never promotes a candidate that drops a served session.

    The active index serves two sessions ingested from two roots. When the
    imported file is deleted and the daemon runs ``--cold-build-index`` over
    only the first root, the imported session's raw is retained in
    ``source.db`` but outside the build's source baseline, so the candidate
    never sees it: promotion refuses as ``active_coverage_incomplete`` and the
    active index keeps serving both sessions. When the daemon watches both
    roots the candidate covers every served session and promotes.

    Anti-vacuity: dropping the active-coverage census from the retained
    ``PreparedIndexPromotion`` lets the one-session candidate publish, and the
    active index loses ``codex-session:cold-imported``.
    """
    from polylogue import Polylogue as RealPolylogue
    from polylogue.core.compute import reset_compute_adapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.catchup_status import _cold_build_settlement
    from polylogue.daemon.intake_adapters import DaemonIntakeService
    from polylogue.daemon.services import ServiceProfile
    from polylogue.sources.live.cold_build import active_cold_build_generation, active_index_generation_is_empty
    from polylogue.sources.live.watcher import _PARSER_FINGERPRINT
    from polylogue.storage.archive_identity import resolve_active_index_path

    def codex_session(native_id: str) -> str:
        return (
            f'{{"type":"session_meta","payload":{{"id":"{native_id}","timestamp":"2026-06-02T00:00:00Z"}}}}\n'
            '{"type":"response_item","payload":{"type":"message","id":"message-0",'
            '"role":"user","content":[{"type":"input_text","text":"Synthetic"}]}}\n'
        )

    def session_ids(index_path: Path) -> set[str]:
        with contextlib.closing(sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)) as db:
            return {str(row[0]) for row in db.execute("SELECT session_id FROM sessions")}

    archive_root = tmp_path / "archive"
    from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop

    # Source-only acquisition refuses an archive without its durable Source tier.
    run_off_event_loop(lambda: bootstrap_archive_root(archive_root))
    source_root = tmp_path / "source"
    source_root.mkdir()
    imports_root = tmp_path / "imports"
    imports_root.mkdir()
    watched_file = source_root / "watched.jsonl"
    watched_file.write_text(codex_session("cold-watched"), encoding="utf-8")
    os.utime(watched_file, (1.0, 1.0))
    imported_file = imports_root / "imported.jsonl"
    imported_file.write_text(codex_session("cold-imported"), encoding="utf-8")
    os.utime(imported_file, (1.0, 1.0))
    served = {"codex-session:cold-watched", "codex-session:cold-imported"}

    from tests.infra.archive_templates import run_archive_fixture_write

    assert await run_archive_fixture_write(archive_root, lambda: active_index_generation_is_empty(archive_root))
    from tests.infra.live_batch import prepared_live_batch_processor

    for root, path in ((source_root, watched_file), (imports_root, imported_file)):
        # The supplied live owners: writer, retained publication and convergence.
        async with prepared_live_batch_processor(
            archive_root,
            (WatchSource("codex", root, layout=export_drop_layout((".jsonl",))),),
            parser_fingerprint=_PARSER_FINGERPRINT,
        ) as processor:
            metrics = await processor.ingest_files([path], emit_event=False)
        assert metrics.succeeded_file_count == 1, metrics
    assert session_ids(resolve_active_index_path(archive_root)) == served

    daemon_sources: tuple[WatchSource, ...] = (
        WatchSource("codex", source_root, layout=export_drop_layout((".jsonl",))),
    )
    if imported_source_kept:
        daemon_sources += (WatchSource("imports", imports_root, layout=export_drop_layout((".jsonl",))),)
    else:
        imported_file.unlink()

    assert reset_compute_adapter(join_timeout_s=5) == ()
    try:
        with contextlib.ExitStack() as stack:
            _daemon_startup_stubs(stack, daemon_cli, archive_root)
            stack.enter_context(patch.object(daemon_cli, "Polylogue", lambda: RealPolylogue(archive_root=archive_root)))
            stack.enter_context(
                patch(
                    "polylogue.daemon.intake_adapters.DaemonIntakeService",
                    lambda dispatcher, **kwargs: DaemonIntakeService(dispatcher, idle_delay_s=0.05, **kwargs),
                )
            )
            task = asyncio.create_task(
                daemon_cli.run_daemon_services(
                    sources=daemon_sources,
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    enable_api=False,
                    enable_source_catchup=False,
                    service_profile=ServiceProfile.INTAKE,
                    cold_build_index=True,
                )
            )
            try:
                try:
                    async with asyncio.timeout(45):
                        while _cold_build_settlement().get("cold_build_settlement_state") not in (
                            "complete",
                            "blocked",
                        ):
                            if task.done():
                                await task
                            await asyncio.sleep(0.05)
                except TimeoutError as exc:
                    raise AssertionError(f"settlement={_cold_build_settlement()}") from exc
                status = _cold_build_settlement()
                candidate = active_cold_build_generation(archive_root)
                assert session_ids(resolve_active_index_path(archive_root)) == served
                if imported_source_kept:
                    assert status["cold_build_settlement_state"] == "complete"
                    assert status["cold_build_settlement_reason"] is None
                    assert candidate is None
                    assert (
                        resolve_active_index_path(archive_root).resolve().parent.name
                        == status["cold_build_candidate_id"]
                    )
                else:
                    assert status["cold_build_settlement_state"] == "blocked"
                    assert status["cold_build_settlement_reason"] == "active_coverage_incomplete"
                    assert status["cold_build_settlement_retry_due_in_s"] is None
                    assert candidate is not None
                    assert candidate.generation_id == status["cold_build_candidate_id"]
                    assert not candidate.promoted
                    assert session_ids(Path(candidate.generation.index_path)) == {"codex-session:cold-watched"}
                    assert resolve_active_index_path(archive_root).resolve().parent.name != candidate.generation_id
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=10)
    finally:
        assert reset_compute_adapter(join_timeout_s=5) == ()


@pytest.mark.asyncio
async def test_cold_baseline_observation_cancel_stops_owned_worker() -> None:
    """Cancelling fair intake must not join an archive-sized source scan at exit."""
    from polylogue.daemon.cli import _observe_faulted_baseline_cancellable

    started = threading.Event()
    stopped = threading.Event()

    def observe(_sources: tuple[WatchSource, ...], *, cancel: threading.Event) -> None:
        started.set()
        cancel.wait(timeout=2.0)
        stopped.set()

    generation = cast(Any, SimpleNamespace(observe_faulted_baseline=observe))
    task = asyncio.create_task(_observe_faulted_baseline_cancellable(generation, ()))
    async with asyncio.timeout(1):
        while not started.is_set():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with asyncio.timeout(1):
        while not stopped.is_set():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_cold_build_repairs_faulted_baseline_in_running_daemon(tmp_path: Path) -> None:
    from polylogue import Polylogue as RealPolylogue
    from polylogue.core.compute import reset_compute_adapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.catchup_status import _cold_build_settlement
    from polylogue.daemon.intake_adapters import DaemonIntakeService
    from polylogue.daemon.services import ServiceProfile
    from polylogue.sources.live import production_baseline
    from polylogue.sources.live.cold_build import ColdBuildGeneration, active_cold_build_generation

    def candidate_rows(candidate: ColdBuildGeneration) -> int:
        with contextlib.closing(sqlite3.connect(f"file:{candidate.generation.index_path}?mode=ro", uri=True)) as db:
            return int(db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])

    archive_root = tmp_path / "archive"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "session.jsonl"
    source_file.write_text(
        '{"type":"session_meta","payload":{"id":"cold-repair","timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        '"role":"user","content":[{"type":"input_text","text":"Synthetic"}]}}\n',
        encoding="utf-8",
    )
    os.utime(source_file, (1.0, 1.0))
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    repaired = False
    real_capture = production_baseline.capture_production_source_baseline
    real_observe = ColdBuildGeneration.observe_faulted_baseline
    real_refresh = ColdBuildGeneration.refresh_faulted_baseline
    observation_threads: list[str] = []
    binding_threads: list[str] = []

    def observe_off_writer(
        self: ColdBuildGeneration, sources: tuple[WatchSource, ...], *, cancel: threading.Event | None = None
    ) -> Any:
        observation_threads.append(threading.current_thread().name)
        return real_observe(self, sources, cancel=cancel)

    def bind_on_writer(self: ColdBuildGeneration, observed: Any) -> bool:
        binding_threads.append(threading.current_thread().name)
        return real_refresh(self, observed)

    def capture_with_transient_fault(
        sources: tuple[WatchSource, ...],
        *,
        operation_id: str,
        cancelled: Callable[[], bool] | None = None,
        progress: production_baseline.BaselineProgress | None = None,
    ) -> production_baseline.ProductionSourceBaseline:
        baseline = real_capture(sources, operation_id=operation_id, cancelled=cancelled, progress=progress)
        if repaired:
            return baseline
        rows = tuple(
            dataclasses.replace(row, disposition="fault", reason="absent_root")
            if row.path == str(missing_root)
            else row
            for row in baseline.decisions
        )
        return production_baseline._seal(baseline.operation_id, baseline.source_signature, rows)

    assert reset_compute_adapter(join_timeout_s=5) == ()
    try:
        with contextlib.ExitStack() as stack:
            _daemon_startup_stubs(stack, daemon_cli, archive_root)
            stack.enter_context(patch.object(daemon_cli, "Polylogue", lambda: RealPolylogue(archive_root=archive_root)))
            stack.enter_context(
                patch.object(production_baseline, "capture_production_source_baseline", capture_with_transient_fault)
            )
            stack.enter_context(patch.object(ColdBuildGeneration, "observe_faulted_baseline", observe_off_writer))
            stack.enter_context(patch.object(ColdBuildGeneration, "refresh_faulted_baseline", bind_on_writer))
            stack.enter_context(
                patch(
                    "polylogue.daemon.intake_adapters.DaemonIntakeService",
                    lambda dispatcher, **kwargs: DaemonIntakeService(dispatcher, idle_delay_s=0.05, **kwargs),
                )
            )
            task = asyncio.create_task(
                daemon_cli.run_daemon_services(
                    sources=(
                        WatchSource("codex", source_root, layout=export_drop_layout((".jsonl",)), required=True),
                        WatchSource("missing", missing_root, layout=export_drop_layout((".jsonl",)), required=True),
                    ),
                    enable_watch=True,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    enable_api=False,
                    enable_source_catchup=False,
                    service_profile=ServiceProfile.INTAKE,
                )
            )
            try:
                try:
                    async with asyncio.timeout(45):
                        while _cold_build_settlement().get("cold_build_settlement_state") != "blocked":
                            if task.done():
                                await task
                            await asyncio.sleep(0.05)
                except TimeoutError as exc:
                    candidate = active_cold_build_generation(archive_root)
                    raise AssertionError(
                        f"settlement={_cold_build_settlement()}, candidate_sessions="
                        f"{candidate_rows(candidate) if candidate is not None else None}"
                    ) from exc
                candidate = active_cold_build_generation(archive_root)
                assert candidate is not None
                candidate_id = candidate.generation_id
                assert candidate_rows(candidate) == 1
                assert _cold_build_settlement()["cold_build_settlement_reason"] == "source_integrity"
                repaired = True
                os.utime(missing_root, (2.0, 2.0))
                async with asyncio.timeout(25):
                    while not candidate.publication_complete:
                        if task.done():
                            await task
                        await asyncio.sleep(0.05)
                assert candidate.generation_id == candidate_id
                assert observation_threads and binding_threads
                assert set(observation_threads) == {"cold-source-observation"}
                assert set(observation_threads).isdisjoint(binding_threads)
                assert _cold_build_settlement()["cold_build_settlement_state"] == "complete"
                assert not task.done()
                with contextlib.closing(
                    sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
                ) as active:
                    assert active.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=10)
    finally:
        assert reset_compute_adapter(join_timeout_s=5) == ()


#: Task-name prefixes the daemon may create outside the supervisor, with the
#: component whose call owns each one's whole lifetime.
#:
#: Every entry is a *bounded child of one admitted operation*: it has no
#: cadence, no retry loop and no scheduler, and it settles inside the call
#: that created it. That is the line the epic draws -- "a registry entry does
#: not create a thread pool, retry loop or independent scheduler", and
#: symmetrically, anything that acquires one stops being a bounded child and
#: has to be declared in :mod:`polylogue.daemon.services` instead of being
#: added here. An anonymous ``Task-N`` matches nothing and fails the
#: inventory, which is why every one of these carries a name.
async def _no_redrive_claims() -> None:
    """A fake ingest owner with no interrupted accepted ingest to claim."""


_DECLARED_UNSUPERVISED_TASK_PREFIXES: dict[str, str] = {
    "polylogue-writer:": "DaemonWriteCoordinator: one admitted mutation",
    "polylogue-writer-staged:": "DaemonWriteThreadBridge: one staged publication",
    "polylogue-managed:": "DaemonWriteCoordinator: one tracked post-write effect",
    "polylogue-drive-catchup:": "DriveCatchupExecution: one settled catch-up step",
    "polylogue-ingest-redrive:": "DaemonOperationRuntime: the ingest owner's accepted-ingest re-drive",
    "polylogue-prepared-writer:": "DaemonWriteCoordinator: one prepared writer-worker body",
    "polylogue-writer-custody:": "async_write_lease: archive custody acquisition and settlement wait",
    "polylogue-writer-hold:": "DaemonWriteThreadBridge.hold: the write gate held for one delegated thread",
}


@dataclasses.dataclass(frozen=True, slots=True)
class _SpawnedTask:
    """One task the event loop actually created, and who asked for it."""

    name: str
    frame: str
    """``file:line`` of the innermost non-asyncio frame that requested it."""

    def __str__(self) -> str:  # pragma: no cover - only read from a failure
        return f"{self.name} ({self.frame})"


def _run_with_task_inventory(
    coro: Any,
    *,
    into: list[_SpawnedTask],
    orphans: list[str] | None = None,
    thread_orphans: list[str] | None = None,
) -> None:
    """Run *coro* under ``asyncio.run``, recording every task the loop creates.

    The denominator is taken from the event loop, not from the registry.
    ``loop.set_task_factory`` is the single construction point that
    ``asyncio.create_task``, ``asyncio.ensure_future`` and ``loop.create_task``
    all funnel through, so a child spawned by *any* module through *any* of
    those spellings lands in ``into`` whether or not something declared it.
    Nothing here consults :mod:`polylogue.daemon.services`, which is what
    stops the inventory from being a mirror of the thing it audits.

    Not covered, and deliberately named rather than implied: a task created
    on a *different* loop inside a worker thread. The daemon's threaded work
    goes through the compute adapter and ``asyncio.to_thread``, neither of
    which creates a task. Those threads are counted separately:
    ``thread_orphans`` receives every thread started during the run that is
    still alive once ``asyncio.run`` has joined its default executor, because
    cancelling the task that awaited a thread does not stop the thread.
    """
    import threading as _threading
    import traceback as _tb

    threads_before = set(_threading.enumerate())

    tasks: list[asyncio.Task[Any]] = []

    def factory(loop: Any, task_coro: Any, **kwargs: Any) -> Any:
        task = asyncio.Task(task_coro, loop=loop, **kwargs)
        tasks.append(task)
        frame = "<unknown>"
        for entry in reversed(_tb.extract_stack()[:-1]):
            if "/asyncio/" in entry.filename or entry.filename == __file__:
                continue
            frame = f"{entry.filename}:{entry.lineno}"
            break
        into.append(_SpawnedTask(name=task.get_name(), frame=frame))
        return task

    async def _main() -> None:
        loop = asyncio.get_running_loop()
        loop.set_task_factory(factory)
        try:
            await coro
        finally:
            # ``asyncio.run`` creates its own shutdown tasks after the main
            # coroutine settles. Those belong to the runner, not to the
            # daemon, so the inventory closes with the route it audits.
            loop.set_task_factory(None)
            # ``asyncio.run`` would cancel any survivor after this point, which
            # would hide an orphan; read what the route itself left running.
            if orphans is not None:
                orphans.extend(task.get_name() for task in tasks if not task.done())

    try:
        asyncio.run(_main())
    finally:
        if thread_orphans is not None:
            thread_orphans.extend(
                thread.name for thread in _threading.enumerate() if thread not in threads_before and thread.is_alive()
            )


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
    """Every task the daemon spawns is one the registry or a named owner declares.

    The denominator is the event loop's task factory
    (:func:`_run_with_task_inventory`), which sees every
    ``create_task``/``ensure_future``/``loop.create_task`` regardless of which
    module spelled it. It never reads
    :mod:`polylogue.daemon.services`, so this inventory cannot be satisfied by
    a registry that agrees with itself.

    Anti-vacuity, all three executed:

    * spawn one unregistered child anywhere on the route
      (``asyncio.ensure_future(asyncio.sleep(0))`` in ``run_daemon_services``)
      and ``unowned`` names it with its ``file:line``;
    * delete one registration from the registry and ``supervisor.start``
      raises ``UnknownServiceError`` naming that exact identity before a task
      exists;
    * drop one ``supervisor.start`` call and ``unresolved`` names the declared
      service the route never resolved.

    The route's own exit is also the orphan count: every task it spawned must
    be finished when ``run_daemon_services`` returns, before ``asyncio.run``
    would cancel survivors on its behalf. Skipping the supervisor's
    cancel-and-await on shutdown leaves the idle services pending here.
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
        created: list[_SpawnedTask] = []
        orphans: list[str] = []
        thread_orphans: list[str] = []
        supervisors = _capture_supervisor(stack, daemon_cli)
        stack.enter_context(patch.object(daemon_cli, "Polylogue", FakePolylogue))
        stack.enter_context(patch.object(daemon_cli, "LiveWatcher", FakeWatcher))
        from polylogue.daemon import embedding_owner

        ingest_owners: list[Any] = []
        embedding_owners: list[object] = []
        compose_owner = daemon_cli.compose_ingest_owner
        compose_embedding = embedding_owner.compose_embedding_convergence

        def capture_ingest_owner(*args: Any, **kwargs: Any) -> Any:
            runtime, profiles = compose_owner(*args, **kwargs)
            ingest_owners.append(runtime)
            return runtime, profiles

        def capture_embedding_owner(*args: Any, **kwargs: Any) -> Any:
            owner = compose_embedding(*args, **kwargs)
            embedding_owners.append(owner)
            return owner

        stack.enter_context(patch.object(daemon_cli, "compose_ingest_owner", capture_ingest_owner))
        stack.enter_context(patch.object(embedding_owner, "compose_embedding_convergence", capture_embedding_owner))
        for attribute in (
            "_periodic_lifecycle_heartbeat",
            "_periodic_health_check",
            "_periodic_wal_checkpoint",
            "_periodic_fts_merge",
            "_periodic_heartbeat",
            "_periodic_db_optimize",
            "_periodic_status_snapshot_refresh",
            "_periodic_raw_materialization_convergence",
        ):
            stack.enter_context(patch.object(daemon_cli, attribute, idle_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_convergence_check", lambda *_a, **_k: idle_loop()))
        stack.enter_context(patch.object(daemon_cli, "_periodic_session_profile_audit", lambda *_a, **_k: idle_loop()))
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
        _run_with_task_inventory(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            ),
            into=created,
            orphans=orphans,
            thread_orphans=thread_orphans,
        )
    assert created, "the task factory recorded nothing; the inventory never observed the route"
    # A daemon serving no API still composes its ingest owner (Codex P1, #5717).
    assert any(entry.name.startswith("polylogue-ingest-redrive:") for entry in created)
    # ...and that owner shares the daemon's one embedding owner rather than
    # composing a second one lazily on its first embedding operation.
    assert len(ingest_owners) == 1 and len(embedding_owners) == 1
    assert ingest_owners[0].embedding_convergence is embedding_owners[0]
    assert orphans == [], f"the composition route returned with live children: {orphans}"
    assert thread_orphans == [], f"the composition route returned with live threads: {thread_orphans}"

    unowned = [
        entry
        for entry in created
        if not entry.name.startswith(TASK_NAME_PREFIX)
        and not any(entry.name.startswith(prefix) for prefix in _DECLARED_UNSUPERVISED_TASK_PREFIXES)
    ]
    assert unowned == [], (
        "the composition route spawned a task no owner declares: "
        + ", ".join(str(entry) for entry in unowned)
        + " -- register it in polylogue.daemon.services or give its owner a declared prefix"
    )

    supervised_names = sorted(
        entry.name[len(TASK_NAME_PREFIX) :] for entry in created if entry.name.startswith(TASK_NAME_PREFIX)
    )
    assert supervised_names, "no supervised service was started on the production route"
    for name in supervised_names:
        service_spec(name)

    assert len(supervisors) == 1
    supervisor = supervisors[0]
    assert set(supervised_names) <= set(supervisor.states())
    unresolved = [spec.name for spec in supervisor.selected if supervisor.state(spec.name) is ServiceState.PENDING]
    assert unresolved == [], f"declared services the composition route never resolved: {unresolved}"

    # The two halves are read from different places on purpose. ``supervised``
    # is what the loop built; ``resolved`` is what the registry-driven
    # supervisor decided. A service that lost its ``supervisor.start`` call
    # shows up as a name the supervisor left PENDING; a task started under a
    # name the registry does not carry never reaches the loop at all, because
    # ``service_spec`` raises with that exact name first.
    resolved_with_a_task = {
        spec.name
        for spec in supervisor.selected
        if supervisor.state(spec.name) not in {ServiceState.SKIPPED, ServiceState.HALTED, ServiceState.PENDING}
    }
    assert set(supervised_names) == resolved_with_a_task, (
        f"supervised tasks {sorted(set(supervised_names) ^ resolved_with_a_task)} do not match resolved services"
    )


def test_a_watcher_with_no_roots_is_unavailable_on_the_production_route(tmp_path: Path) -> None:
    """A watcher with nothing to watch settles ``unavailable``, never ``completed``.

    ``watcher`` is declared ``FAIL_DAEMON``. A watcher that returned when no
    root existed settled ``stopped: completed``: the policy never applied and
    status read a watch that did its work. The composition route now resolves
    it before a task exists, names the reason, releases the maintenance gate
    that waits on watch registration, and keeps fair intake running.

    Anti-vacuity: drop the ``prepare_watch_roots`` check in
    ``run_daemon_services`` and the watcher is started, which this fake refuses.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.services import ServiceState
    from polylogue.daemon.supervisor import TASK_NAME_PREFIX

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class RootlessWatcher(_NoIntakeHints):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.watcher_ready = asyncio.Event()

        def prepare_watch_roots(self) -> list[Path]:
            return []

        async def run(self) -> None:
            raise AssertionError("a watcher with no roots was started")

        def stop(self) -> None:
            return None

    async def idle_loop(**_kwargs: object) -> None:
        await asyncio.Event().wait()

    async def intake_stops(_self: object) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("intake stopped")

    with contextlib.ExitStack() as stack:
        _daemon_startup_stubs(stack, daemon_cli, tmp_path)
        created: list[_SpawnedTask] = []
        supervisors = _capture_supervisor(stack, daemon_cli)
        stack.enter_context(patch.object(daemon_cli, "Polylogue", FakePolylogue))
        stack.enter_context(patch.object(daemon_cli, "LiveWatcher", RootlessWatcher))
        stack.enter_context(patch("polylogue.operations.intake_adapters.DaemonIntakeService.run", intake_stops))
        for attribute in (
            "_periodic_lifecycle_heartbeat",
            "_periodic_health_check",
            "_periodic_wal_checkpoint",
            "_periodic_fts_merge",
            "_periodic_heartbeat",
            "_periodic_db_optimize",
            "_periodic_status_snapshot_refresh",
            "_periodic_raw_materialization_convergence",
        ):
            stack.enter_context(patch.object(daemon_cli, attribute, idle_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_convergence_check", lambda *_a, **_k: idle_loop()))
        stack.enter_context(patch.object(daemon_cli, "_periodic_session_profile_audit", lambda *_a, **_k: idle_loop()))
        for target in (
            "polylogue.daemon.embedding_backlog.periodic_embedding_backlog_check",
            "polylogue.daemon.embedding_backlog.periodic_embedding_orphan_reconcile_check",
            "polylogue.daemon.judgment_automation.periodic_judgment_automation_sweep",
            "polylogue.daemon.blob_gc_periodic.periodic_blob_gc_check",
            "polylogue.daemon.blob_gc_periodic.periodic_blob_publication_reconciliation_check",
            "polylogue.daemon.secret_scan_sweep.periodic_secret_scan_sweep",
        ):
            stack.enter_context(patch(target, idle_loop))
        stack.enter_context(pytest.raises(RuntimeError, match="intake stopped"))
        _run_with_task_inventory(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=tmp_path / "absent-codex"),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            ),
            into=created,
        )

    supervisor = supervisors[0]
    assert supervisor.state("watcher") is ServiceState.UNAVAILABLE
    assert supervisor.state("watcher_registered_bridge") is ServiceState.UNAVAILABLE
    reasons = {transition.service: transition.reason for transition in supervisor.transitions()}
    assert reasons["watcher"] == "no configured source root exists"
    assert f"{TASK_NAME_PREFIX}watcher" not in {entry.name for entry in created}
    assert supervisor.state("fair_intake") is ServiceState.FAILED


@pytest.mark.parametrize("embeddings_configured", [False, True])
def test_unconfigured_embeddings_skip_the_backlog_service_on_the_production_route(
    tmp_path: Path, embeddings_configured: bool
) -> None:
    """An embedding backlog that can only refuse is never given a task.

    With embeddings unconfigured -- the default -- ``compose_embedding_convergence``
    returns a constant policy deferral for the life of the process, while this
    loop is woken by every ``IngestCommitted``. Selected-and-refusing therefore
    costs one identical refusal per commit and reports nothing: measured at
    head, 25 ingest wakes produced 25 refusals. The capability moves that to
    selection, where the supervisor resolves one ``skipped`` state that
    ``supervised_service_snapshot`` publishes.

    ``embedding_orphan_reconcile`` is selected in both rows: stale embedding
    rows are debt to drain regardless, so this is a gate on one loop rather
    than on the embeddings owner.

    Anti-vacuity: drop ``ServiceCapability.EMBEDDINGS`` from the
    ``embedding_backlog`` spec, or stop resolving it in the composition root,
    and the unconfigured row gets a task and settles ``stopped`` instead of
    ``skipped``. Both mutations were executed and both fail here.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.services import ServiceState
    from polylogue.daemon.status import supervised_service_snapshot
    from polylogue.daemon.supervisor import TASK_NAME_PREFIX
    from tests.infra.embedding_config import embedding_config

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

    config = embedding_config(
        embedding_enabled=embeddings_configured,
        voyage_api_key="vk-synthetic" if embeddings_configured else None,
    )
    supervisors: list[Any] = []
    projections: list[dict[str, str] | None] = []
    real_setter = daemon_cli._set_active_supervisor

    def capture(supervisor: Any) -> None:
        if supervisor is not None:
            supervisors.append(supervisor)
        else:
            # The last moment the process still has a composed supervisor, so
            # the status projection is read the way a live daemon reads it.
            snapshot = supervised_service_snapshot()
            projections.append(None if snapshot is None else snapshot[0])
        real_setter(supervisor)

    with contextlib.ExitStack() as stack:
        _daemon_startup_stubs(stack, daemon_cli, tmp_path)
        created: list[_SpawnedTask] = []
        stack.enter_context(patch.object(daemon_cli, "_set_active_supervisor", capture))
        stack.enter_context(patch("polylogue.config.load_polylogue_config", return_value=config))
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
        ):
            stack.enter_context(patch.object(daemon_cli, attribute, idle_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_convergence_check", lambda *_a, **_k: idle_loop()))
        stack.enter_context(patch.object(daemon_cli, "_periodic_session_profile_audit", lambda *_a, **_k: idle_loop()))
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
        _run_with_task_inventory(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            ),
            into=created,
        )

    supervisor = supervisors[0]
    # ``skipped`` is resolved at selection and survives shutdown; anything the
    # supervisor did give a task to settles ``stopped`` when this run unwinds.
    assert supervisor.state("embedding_backlog") is (
        ServiceState.SKIPPED if not embeddings_configured else ServiceState.STOPPED
    )
    assert supervisor.state("embedding_orphan_reconcile") is ServiceState.STOPPED

    backlog_tasks = [entry.name for entry in created if entry.name == f"{TASK_NAME_PREFIX}embedding_backlog"]
    assert backlog_tasks == ([] if not embeddings_configured else [f"{TASK_NAME_PREFIX}embedding_backlog"])

    # The unschedulable half is only half the property: status must name it.
    assert projections and projections[-1] is not None
    projected = projections[-1]
    assert projected["embedding_backlog"] == supervisor.state("embedding_backlog").value
    assert (projected["embedding_backlog"] == "skipped") is not embeddings_configured

    if not embeddings_configured:
        skip = next(transition for transition in supervisor.transitions() if transition.service == "embedding_backlog")
        assert skip.reason is not None and "embeddings" in skip.reason


def _run_focused_profile_iteration(tmp_path: Path) -> dict[str, float | int]:
    """The API-disabled fixture profile.

    Two separate things are asserted, because the profile is only half the
    property. It must not *start* raw materialization, and it must not
    *construct* the intake stack on behalf of services it will never
    schedule: registering adapters and opening a cold-build generation for a
    discarded `fair_intake` is real archive work that a focused test pays
    for. Each iteration measures the current focused composition and
    supervisor shutdown; no historical timing baseline is available.

    Anti-vacuity, both executed: pass ``ServiceProfile.PRODUCTION`` instead
    and the materialization assertion fails, because the production profile
    starts the loop this profile excludes; drop the ``intake_scheduled``
    guard in ``run_daemon_services`` and ``intake_builds`` is non-empty.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon import intake_adapters as daemon_intake_adapters
    from polylogue.daemon.services import ServiceProfile, ServiceState
    from polylogue.daemon.supervisor import DaemonSupervisor, ShutdownReport

    started: list[str] = []
    intake_builds: list[object] = []
    startup_archive_work: list[str] = []
    shutdown_reports: list[ShutdownReport] = []
    actual_shutdown = DaemonSupervisor.shutdown

    async def measured_shutdown(supervisor: DaemonSupervisor) -> ShutdownReport:
        report = await actual_shutdown(supervisor)
        shutdown_reports.append(report)
        return report

    async def resident_loop(**_kwargs: object) -> None:
        started.append("resident")

    async def materialization(**_kwargs: object) -> None:
        started.append("raw_materialization")
        await asyncio.Event().wait()

    with contextlib.ExitStack() as stack:
        _daemon_startup_stubs(stack, daemon_cli, tmp_path)
        stack.enter_context(patch.object(DaemonSupervisor, "shutdown", measured_shutdown))
        supervisors = _capture_supervisor(stack, daemon_cli)
        stack.enter_context(patch.object(daemon_cli, "_periodic_lifecycle_heartbeat", resident_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_health_check", resident_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_raw_materialization_convergence", materialization))
        stack.enter_context(
            patch.object(
                daemon_cli,
                "_ensure_embedding_lifecycle_startup_sync",
                lambda _root: startup_archive_work.append("embedding_lifecycle"),
            )
        )
        stack.enter_context(
            patch(
                "polylogue.daemon.fts_convergence.FtsConvergenceOwner.converge",
                lambda *_args, **_kwargs: startup_archive_work.append("fts"),
            )
        )

        def _record_intake_build(*args: object, **_kwargs: object) -> tuple[object, ...]:
            intake_builds.append(args)
            return ()

        stack.enter_context(patch.object(daemon_intake_adapters, "build_intake_adapters", _record_intake_build))
        started_at = time.monotonic()
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                enable_api=False,
                service_profile=ServiceProfile.RESIDENT_CORE,
            )
        )
        elapsed = time.monotonic() - started_at

    assert "raw_materialization" not in started
    assert startup_archive_work == [], "the focused profile entered archive startup work"
    assert started == ["resident", "resident"]
    assert intake_builds == [], "the intake stack was built for services this profile never schedules"
    supervisor = supervisors[0]
    assert "raw_observation_convergence" not in {spec.name for spec in supervisor.selected}
    assert supervisor.state("lifecycle_heartbeat") is ServiceState.STOPPED
    assert len(shutdown_reports) == 1
    shutdown_report = shutdown_reports[0]
    assert shutdown_report.clean
    assert shutdown_report.orphaned == ()
    return {
        "elapsed_s": elapsed,
        "shutdown_s": shutdown_report.duration_s,
        "orphan_count": len(shutdown_report.orphaned),
        "unexpected_resident_runs": max(0, len(started) - 2),
    }


@pytest.mark.uses_real_clock("bounds ten consecutive focused-profile fixture runs")
def test_a_focused_profile_starts_no_materialization_and_finishes_promptly(tmp_path: Path) -> None:
    """Ten consecutive focused selections of the production registry stay below 10 seconds."""
    from tests.infra.daemon_service_harness import record_private_lifecycle_probe

    measures: list[dict[str, float | int]] = []
    for index in range(10):
        root = tmp_path / f"run-{index}"
        root.mkdir()
        measure = _run_focused_profile_iteration(root)
        measures.append(measure)
        assert measure["elapsed_s"] < 10.0, measures
        assert measure["orphan_count"] == 0, measures
        assert measure["unexpected_resident_runs"] == 0, measures
    record_private_lifecycle_probe(
        "focused-profile",
        {"profile": "resident_core", "iterations": measures, "historical_before": "unmeasured"},
    )


@pytest.mark.asyncio
async def test_browser_host_child_is_terminated_when_its_service_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supervised process cannot outlive cancellation of its service."""
    from polylogue.daemon import cli as daemon_cli

    started = asyncio.Event()
    exited = asyncio.Event()
    arguments: tuple[str, ...] = ()

    class Process:
        returncode: int | None = None
        terminate_calls = 0

        async def wait(self) -> int:
            await exited.wait()
            assert self.returncode is not None
            return self.returncode

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.returncode = 0
            exited.set()

        def kill(self) -> None:
            raise AssertionError("cooperative child was killed")

    process = Process()

    async def spawn(*args: str) -> Process:
        nonlocal arguments
        arguments = args
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(
        daemon_cli._run_browser_host(host="127.0.0.1", port=8767, daemon_origin="http://127.0.0.1:8766")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminate_calls == 1
    assert arguments[-6:] == (
        "--host",
        "127.0.0.1",
        "--port",
        "8767",
        "--daemon-origin",
        "http://127.0.0.1:8766",
    )


@pytest.mark.asyncio
async def test_established_missing_source_refuses_before_starting_services(tmp_path: Path) -> None:
    """Lost Source custody never becomes a newly created empty acquisition tier."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.health import _check_schema_version_fast
    from polylogue.daemon.services import ServiceProfile
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.migration_runner import DurableChangeTrainError

    archive_root = tmp_path / "archive"
    from tests.infra.archive_templates import run_archive_fixture_write

    await run_archive_fixture_write(archive_root, lambda: initialize_active_archive_root(archive_root))
    (archive_root / "source.db").unlink()
    retained = {name: (archive_root / name).read_bytes() for name in ("user.db", "audit.db")}
    with contextlib.ExitStack() as stack:
        _daemon_startup_stubs(stack, daemon_cli, archive_root)
        stack.enter_context(patch.object(daemon_cli, "_check_schema_version_fast", _check_schema_version_fast))
        supervisors = _capture_supervisor(stack, daemon_cli)
        with pytest.raises(DurableChangeTrainError):
            await daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                enable_api=False,
                service_profile=ServiceProfile.PRODUCTION,
            )
    assert not supervisors
    assert not (archive_root / "source.db").exists()
    assert {name: (archive_root / name).read_bytes() for name in retained} == retained


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


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("a shutdown deadline is wall-clock; the barrier bounds this one to 50ms")
async def test_an_orphaned_service_retains_archive_ownership_on_the_production_route(tmp_path: Path) -> None:
    """Incomplete shutdown keeps ownership; a successor writer cannot start beside it.

    The supervisor cancels ``health_check`` and stops waiting after its
    declared deadline. That child is still running. Every authority that
    would let some *other* writer in -- the durable archive lease, the
    pidfile, and rebuild exclusion -- must therefore stay held, and the
    shutdown must be reported as incomplete rather than as a clean stop.

    The deadline is made controllable rather than waited out: one registry
    field is narrowed to 50 ms and the child is released explicitly at the
    end, so nothing here depends on a wall-clock timeout elapsing.

    Anti-vacuity: drop ``orphaned_services=shutdown_report.orphaned`` from
    the ``_ownership_retention_reason`` call in ``run_daemon_services`` and
    all three assertions invert -- the lease is released, the pidfile is
    cleaned, and rebuild exclusion is dropped while the child still runs.
    Executed.
    """
    import dataclasses as _dataclasses

    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.services import ServiceProfile, ServiceState, service_spec
    from polylogue.maintenance.raw_authority import ArchiveWriterRebuildExclusion
    from polylogue.operations import durable_change_train

    real_spec = service_spec
    ignoring = True
    running = asyncio.Event()
    retained: list[str] = []
    released: list[str] = []
    pidfile_cleanups: list[str] = []

    def narrowed_spec(name: str) -> Any:
        spec = real_spec(name)
        if name != "health_check":
            return spec
        return _dataclasses.replace(spec, shutdown_deadline_s=0.05)

    async def resident_loop(**_kwargs: object) -> None:
        return None

    async def uncancellable_health(**_kwargs: object) -> None:
        running.set()
        while True:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                # Releasing at the end is teardown, not the behaviour under
                # test: a child that ignores cancellation forever would hang
                # the event loop's own shutdown instead of this one.
                if not ignoring:
                    raise

    # Startup train reconciliation acquires and releases an archive ownership
    # token of its own, so the daemon's token is identified by object, not by
    # class: only *its* release is the ownership handover under test.
    daemon_owners: list[OwnedArchiveLocation] = []
    real_acquire = durable_change_train.acquire_durable_archive_ownership
    real_release = OwnedArchiveLocation.release

    def recording_acquire(root: Path, *, owner_id: str) -> OwnedArchiveLocation:
        owner = real_acquire(root, owner_id=owner_id)
        daemon_owners.append(owner)
        return owner

    def recording_release(self: OwnedArchiveLocation) -> None:
        if daemon_owners and self is daemon_owners[0]:
            released.append("archive_owner")
        real_release(self)

    def recording_cleanup() -> None:
        pidfile_cleanups.append("pidfile")

    with contextlib.ExitStack() as stack, capture() as events:
        _daemon_startup_stubs(stack, daemon_cli, tmp_path)
        supervisors = _capture_supervisor(stack, daemon_cli)
        stack.enter_context(patch("polylogue.daemon.supervisor.service_spec", narrowed_spec))
        stack.enter_context(patch.object(daemon_cli, "_periodic_lifecycle_heartbeat", resident_loop))
        stack.enter_context(patch.object(daemon_cli, "_periodic_health_check", uncancellable_health))
        stack.enter_context(
            patch.object(ArchiveWriterRebuildExclusion, "retain_until_process_exit", lambda _self: retained.append("x"))
        )
        stack.enter_context(patch.object(durable_change_train, "acquire_durable_archive_ownership", recording_acquire))
        stack.enter_context(patch.object(OwnedArchiveLocation, "release", recording_release))
        stack.enter_context(patch.object(daemon_cli, "_cleanup_pidfile", recording_cleanup))

        task = asyncio.create_task(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                enable_api=False,
                service_profile=ServiceProfile.RESIDENT_CORE,
            )
        )
        try:
            await asyncio.wait_for(running.wait(), timeout=10.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10.0)
        finally:
            ignoring = False

    supervisor = supervisors[0]
    assert supervisor.state("health_check") is ServiceState.ORPHANED

    assert released == [], "the durable archive lease was released beside a still-running child"
    assert pidfile_cleanups == [], "the pidfile was released beside a still-running child"
    assert retained == ["x"], "rebuild exclusion was not retained for the process"

    records = [record for record in events if record.get("event") == "daemon.pidfile.retained"]
    assert records, "the retained pidfile was not reported"
    assert records[-1].get("reason") == "services_outlived_shutdown_deadline"
    assert "health_check" in str(records[-1].get("error_detail"))

    orphan_reports = [record for record in events if record.get("event") == "daemon.shutdown.services_orphaned"]
    assert orphan_reports, "incomplete shutdown was not reported"
    assert "health_check" in str(orphan_reports[-1].get("error_detail"))

    # ... and it is never reported as a stop. On this route the daemon was
    # cancelled, so ``daemon.stopped`` is not reached at all; the assertion
    # exists because the one thing shutdown must not do with a live child is
    # claim the process stopped cleanly.
    assert [record for record in events if record.get("event") == "daemon.stopped"] == []


@pytest.mark.parametrize("command", ("run", "watch"))
@pytest.mark.parametrize("flag", ("--root", "--default-source", "--no-default-sources"))
def test_daemon_has_no_custom_source_roots_or_source_narrowing(command: str, flag: str) -> None:
    """Every origin is acquired only from its canonical location.

    Anti-vacuity: restoring any of the three options lets it parse instead of
    failing as an unknown option.
    """
    result = CliRunner().invoke(main, [command, flag, "/tmp/elsewhere", "--help"])

    assert result.exit_code != 0
    assert f"No such option '{flag}'" in result.output


def test_owned_source_roots_are_decided_by_role_not_resolved_location(tmp_path: Path) -> None:
    """A provider root relocated into the archive tree is still the provider's.

    Anti-vacuity: classify by ``resolve(strict=False).is_relative_to(archive)``
    again and the dangling ``claude-code`` symlink below is treated as owned,
    so startup's ``mkdir(exist_ok=True)`` raises ``FileExistsError`` on it.
    """
    from polylogue.daemon.cli import _is_polylogue_owned_source, _watch_sources
    from polylogue.sources.hooks import HookSpoolSourceSpec
    from polylogue.sources.live.watcher import WatchSource, hook_carrier_watch_sources

    archive = tmp_path / "archive"
    archive.mkdir()
    relocated = tmp_path / "home" / ".claude" / "projects"
    relocated.parent.mkdir(parents=True)
    relocated.symlink_to(archive / "provider-data" / "claude")  # target absent: dangling
    provider = WatchSource(name="claude-code", root=relocated)
    assert not _is_polylogue_owned_source(provider)

    owned = [source for source in _watch_sources() if _is_polylogue_owned_source(source)]
    assert {"browser-capture", "inbox"} <= {source.name for source in owned}
    assert all(source.name in {"browser-capture", "inbox"} or source.role == "primary-writable" for source in owned)
    assert not any(source.name in {"claude-code", "codex", "gemini-cli", "hermes"} for source in owned)
    primary = hook_carrier_watch_sources(
        (HookSpoolSourceSpec(source_id="hooks", role="primary-writable", root=tmp_path / "hooks"),)
    )
    legacy = hook_carrier_watch_sources(
        (HookSpoolSourceSpec(source_id="old-hooks", role="legacy-read-only", root=tmp_path / "old"),)
    )
    assert primary and all(_is_polylogue_owned_source(source) for source in primary)
    assert legacy and not any(_is_polylogue_owned_source(source) for source in legacy)


def test_session_profile_audit_resumes_the_promoted_audit_each_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """The periodic audit service drives demand plus bounded audit passes.

    Anti-vacuity: without this service a profile that runs intake but not
    ``convergence_check`` (INTAKE) never resumes a promoted audit.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.session_profile_composition import ComposedSessionProfiles

    budgets: list[float] = []
    ran = asyncio.Event()

    class _Profiles(ComposedSessionProfiles):
        async def converge_backlog(self, budget_s: float) -> Any:
            budgets.append(budget_s)
            ran.set()
            return SimpleNamespace()

    async def unused(_scope: object) -> Any:
        raise AssertionError("the audit service drives converge_backlog, not a bare demand call")

    async def unused_promoted() -> Any:
        raise AssertionError("the audit service never promotes")

    profiles = _Profiles(unused, unused_promoted, cast(Any, None))

    async def exercise() -> None:
        task = asyncio.create_task(daemon_cli._periodic_session_profile_audit(profiles))
        await asyncio.wait_for(ran.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert budgets == [daemon_cli._SESSION_PROFILE_BACKLOG_SECONDS]


@pytest.mark.asyncio
async def test_no_watch_fresh_audit_fault_returns_to_its_periodic_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed no-watch startup service must yield after one unavailable audit pass."""
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.periodic import PeriodicRunner
    from polylogue.daemon.session_profile_composition import compose_session_profile_callback

    archive_root = tmp_path / "fresh"
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    tick_finished = asyncio.Event()
    delays: list[float] = []
    calls = 0

    async def scheduled_wait(delay: float) -> None:
        delays.append(delay)
        tick_finished.set()
        await asyncio.Event().wait()

    runner = PeriodicRunner(jitter_ratio=0, sleep=scheduled_wait)
    monkeypatch.setattr(daemon_cli, "daemon_periodic_runner", lambda: runner)
    profiles = compose_session_profile_callback(
        archive_root,
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        now=lambda: 0.0,
    )
    real_pass = profiles.audit_pass
    assert real_pass is not None
    failures: list[DerivationReport] = []

    async def audit(deadline: float) -> DerivationReport | None:
        nonlocal calls
        calls += 1
        assert calls == 1, "bootstrap fault repeated before the next periodic tick"
        report = await real_pass(deadline)
        assert report is not None
        failures.append(report)
        return report

    profiles = dataclasses.replace(profiles, audit_pass=audit)
    task = asyncio.create_task(daemon_cli._periodic_session_profile_audit(profiles, watcher_registered=None))
    try:
        await tick_finished.wait()
        state = runner.state("session_profile_audit")
        assert state is not None and state.runs == 1 and state.failures == 0
        assert calls == 1 and failures[0].failed == 1
        assert profiles.audit_pending()
        assert delays == [daemon_cli._SESSION_PROFILE_AUDIT_INTERVAL_SECONDS]
        assert not archive_root.exists()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


def test_startup_archive_admission_uses_eventual_compute_creator_and_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from math import inf

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.operations import durable_change_train
    from polylogue.storage.sqlite.connection_profile import retained_native_settlement_owners_on_current_thread
    from polylogue.storage.sqlite.write_lease import require_write_lease
    from tests.infra.excision_embeddings import seed_excision_session

    seed_excision_session(tmp_path, native_id="startup-original-owner", with_embedding=True)
    events: list[str] = []
    main_thread = threading.get_ident()
    original_reconcile = durable_change_train.reconcile_durable_change_trains_on_startup

    class SchemaProbeReachedError(Exception):
        pass

    stopped = SchemaProbeReachedError("after physically settled bootstrap")

    async def run() -> None:
        coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
        kernel = BoundedComputeAdapter(max_workers=1, queue_units=1, queue_bytes=0)
        monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: coordinator)
        monkeypatch.setattr("polylogue.core.compute.compute_adapter", lambda: kernel)
        monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)

        def reconcile(root: Path) -> tuple[Path, ...]:
            assert threading.get_ident() != main_thread
            kernel.require_current_creator()
            require_write_lease("startup control", archive_root=tmp_path)
            result = original_reconcile(root)
            assert not retained_native_settlement_owners_on_current_thread()
            events.append("original-creator-reconciled")
            return result

        def schema_probe() -> None:
            assert threading.get_ident() == main_thread
            assert events == ["original-creator-reconciled"]
            events.append("schema-after-settlement")
            raise stopped

        monkeypatch.setattr(durable_change_train, "reconcile_durable_change_trains_on_startup", reconcile)
        monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", schema_probe)
        try:
            with pytest.raises(SchemaProbeReachedError) as caught:
                await daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                )
            assert caught.value is stopped
            assert events == ["original-creator-reconciled", "schema-after-settlement"]
            assert not (tmp_path / "daemon.pid").exists()
        finally:
            kernel.shutdown(wait=True)
            assert await coordinator.shutdown(timeout=inf)

    asyncio.run(run())


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
@pytest.mark.parametrize("unavailable_stage", ["live_ingest_source_read", "live_ingest_deferred", "sinex_publication"])
@pytest.mark.parametrize("retry_stage", ["hook_paste_enrichment", "lineage_prefix_recompose"])
def test_debt_retry_selects_executable_stage_before_page_limit(
    tmp_path: Path,
    frozen_clock: FrozenClock,
    bounded_compute_adapter: BoundedComputeAdapter,
    unavailable_stage: str,
    retry_stage: str,
) -> None:
    """Filtering unavailable stages after LIMIT100 starves the older subject."""
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    cursor = CursorStore(db)
    cursor.record_convergence_debt(stage=retry_stage, subject_type="session_id", subject_id="owed", error="retry owed")
    for number in range(100):
        cursor.record_convergence_debt(
            stage=unavailable_stage,
            subject_type="source_path",
            subject_id=f"/synthetic/input-{number}",
            error="original acquisition debt",
            deferred=True,
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = NULL, updated_at_ms = 2")
        conn.execute("UPDATE convergence_debt SET updated_at_ms = 1 WHERE stage = ?", (retry_stage,))
    before = cursor.list_convergence_debt(limit=101)
    calls: list[tuple[str, ...]] = []

    def execute(session_ids: Sequence[str]) -> bool:
        calls.append(tuple(session_ids))
        return True

    stage = ConvergenceStage(
        name=retry_stage,
        description="synthetic executable retry",
        check=lambda path: False,
        execute=lambda path: True,
        check_sessions=lambda ids: set(ids),
        execute_sessions=execute,
    )
    with (
        patch(
            "polylogue.daemon.convergence_stages.make_default_convergence_stages",
            return_value=(stage,) if retry_stage != "hook_paste_enrichment" else (),
        ),
        patch(
            "polylogue.daemon.convergence_stages.make_hook_paste_enrichment_stage",
            return_value=stage
            if retry_stage == "hook_paste_enrichment"
            else ConvergenceStage(
                name="hook_paste_enrichment",
                description="idle hook",
                check=lambda path: False,
                execute=lambda path: True,
            ),
        ),
    ):
        assert daemon_cli._drain_convergence_debt_once(db, compute_adapter=bounded_compute_adapter) == 1
        assert daemon_cli._drain_convergence_debt_once(db, compute_adapter=bounded_compute_adapter) == 0
    assert calls == [("owed",)]
    after = cursor.list_convergence_debt(limit=101)
    assert after == [row for row in before if row.stage == unavailable_stage]


@pytest.mark.contract
def test_convergence_debt_empty_stage_selection_preserves_visible_rows(tmp_path: Path) -> None:
    """An empty executable roster selects no debt, never every debt."""
    cursor = CursorStore(tmp_path / "index.db")
    cursor.record_convergence_debt(stage="owed", subject_type="session_id", subject_id="session", error="original")
    assert cursor.list_convergence_debt(include_stages=()) == []
    assert cursor.list_convergence_debt(subject_types=()) == []
    assert len(cursor.list_convergence_debt()) == 1
    assert len(cursor.list_convergence_debt(include_stages=("owed",), subject_types=("session_id",))) == 1
