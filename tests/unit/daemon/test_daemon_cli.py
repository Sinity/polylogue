from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import sqlite3
import threading
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
from polylogue.daemon.health import DaemonHealth, HealthSeverity, HealthTier
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
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

    assert result.exit_code == 0
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

    assert result.exit_code == 0
    assert "Polylogue daemon" in result.output
    assert "Live sources: 1/1 available" in result.output
    assert f"exists: {tmp_path} (available)" in result.output
    assert "Browser capture spool: ready" in result.output


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
    assert storage["archive_schema_ready"] is True
    assert storage["schema_mismatches"] == []
    assert storage["present_tiers"] == ["source", "index", "embeddings", "user", "ops"]
    tiers = cast(list[dict[str, object]], storage["tiers"])
    assert {tier["name"]: tier["user_version"] for tier in tiers} == {
        "source": SOURCE_SCHEMA_VERSION,
        "index": INDEX_SCHEMA_VERSION,
        "embeddings": 1,
        "user": USER_SCHEMA_VERSION,
        "ops": 1,
    }
    assert {tier["name"]: tier["version_status"] for tier in tiers} == {
        "source": "ok",
        "index": "ok",
        "embeddings": "ok",
        "user": "ok",
        "ops": "ok",
    }


def test_polylogued_status_json_reports_rebuild_index_not_ready(tmp_path: Path) -> None:
    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("embeddings.db", ArchiveTier.EMBEDDINGS),
        ("user.db", ArchiveTier.USER),
        ("ops.db", ArchiveTier.OPS),
    ):
        initialize_archive_database(tmp_path / filename, tier)
    now_ms = 1_700_000_001_000
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        record_ingest_attempt(
            conn,
            attempt_id="rebuild-active",
            source_path=str(tmp_path / "source.db"),
            status="running",
            phase="rebuild-index",
            started_at_ms=now_ms - 1_000,
            heartbeat_at_ms=now_ms,
            storage_route="maintenance",
        )

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status.db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.index_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
        patch("polylogue.storage.archive_readiness.time.time", return_value=now_ms / 1000),
    ):
        result = CliRunner().invoke(main, ["status", "--format", "json"])

    assert result.exit_code == 0
    payload = loads(result.output)
    assert isinstance(payload, dict)
    storage = cast(dict[str, object], payload["archive_storage"])
    assert storage["archive_schema_ready"] is True
    assert storage["archive_ready"] is False
    assert storage["archive_materialization_ready"] is False
    assert storage["active_rebuild_index_attempts"] == [
        {
            "attempt_id": "rebuild-active",
            "phase": "rebuild-index",
            "started_at_ms": now_ms - 1_000,
            "heartbeat_at_ms": now_ms,
            "parsed_raw_count": 0,
            "materialized_count": 0,
        }
    ]


def test_polylogued_status_json_reports_schema_mismatch_not_ready(tmp_path: Path) -> None:
    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("embeddings.db", ArchiveTier.EMBEDDINGS),
        ("user.db", ArchiveTier.USER),
        ("ops.db", ArchiveTier.OPS),
    ):
        initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")

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
    assert archive_component["repair_hint"] == "polylogue ops reset --index && polylogued run"


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


def test_polylogued_status_plain_reports_schema_mismatch(tmp_path: Path) -> None:
    for filename, tier in (
        ("source.db", ArchiveTier.SOURCE),
        ("index.db", ArchiveTier.INDEX),
        ("embeddings.db", ArchiveTier.EMBEDDINGS),
        ("user.db", ArchiveTier.USER),
        ("ops.db", ArchiveTier.OPS),
    ):
        initialize_archive_database(tmp_path / filename, tier)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("PRAGMA user_version = 1")

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status.db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.index_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status.default_sources", return_value=()),
    ):
        result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0
    assert (
        "Storage: archive_file_set (source, index, embeddings, user, ops); final split complete; schema mismatch index"
        in result.output
    )


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


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_retries_global_messages_fts_surface(
    tmp_path: Path,
    frozen_clock: FrozenClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    cursor = CursorStore(db)
    cursor.record_convergence_debt(
        stage="fts",
        subject_type="fts_surface",
        subject_id="messages_fts",
        error="startup found stale messages_fts freshness ledger",
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'",
        )
        conn.commit()
    repairs: list[tuple[Path, str]] = []

    def fake_repair_fts_surface(path: Path, surface: str) -> bool:
        repairs.append((path, surface))
        return True

    monkeypatch.setattr(
        "polylogue.daemon.convergence_stages.repair_fts_surface",
        fake_repair_fts_surface,
    )

    retried = daemon_cli._drain_convergence_debt_once(db)
    debt_after = cursor.list_convergence_debt()

    assert retried == 1
    assert repairs == [(db, "messages_fts")]
    assert debt_after == []


@pytest.mark.contract
@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_drain_convergence_debt_retries_optional_fts_surface(
    tmp_path: Path,
    frozen_clock: FrozenClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    cursor = CursorStore(db)
    cursor.record_convergence_debt(
        stage="fts",
        subject_type="fts_surface",
        subject_id="threads_fts",
        error="optional FTS startup repair failed",
    )
    repairs: list[tuple[Path, str]] = []

    def fake_repair_fts_surface(path: Path, surface: str) -> bool:
        repairs.append((path, surface))
        return True

    monkeypatch.setattr(
        "polylogue.daemon.convergence_stages.repair_fts_surface",
        fake_repair_fts_surface,
    )

    retried = daemon_cli._drain_convergence_debt_once(db)
    debt_after = cursor.list_convergence_debt()

    assert retried == 1
    assert repairs == [(db, "threads_fts")]
    assert debt_after == []


def test_periodic_convergence_check_treats_sqlite_lock_as_archive_busy(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("database is locked")

    with (
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        patch.object(daemon_cli.logger, "info") as info,
        patch.object(daemon_cli.logger, "warning") as warning,
    ):
        asyncio.run(daemon_cli._retry_convergence_debt_once(db))

    info.assert_called_once()
    assert info.call_args.args[0] == "convergence: archive busy; retrying derived debt on next tick: %s"
    warning.assert_not_called()


def test_drain_raw_materialization_once_uses_bounded_daemon_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    calls: dict[str, object] = {}
    order: list[str] = []

    class FakeResult:
        success = True
        repaired_count = 7
        detail = "ok"

    class FakeRestoreResult:
        restored_count = 2

    def fake_restore_direct_blob_reference_debt(
        db_path: Path,
        *,
        dry_run: bool,
        max_count: int,
        sample_size: int,
    ) -> FakeRestoreResult:
        order.append("restore")
        calls["restore_db_path"] = db_path
        calls["restore_dry_run"] = dry_run
        calls["restore_max_count"] = max_count
        calls["restore_sample_size"] = sample_size
        return FakeRestoreResult()

    def fake_repair_raw_materialization(config: Config, *, dry_run: bool, raw_artifact_limit: int) -> FakeResult:
        order.append("materialize")
        calls["archive_root"] = config.archive_root
        calls["render_root"] = config.render_root
        calls["dry_run"] = dry_run
        calls["raw_artifact_limit"] = raw_artifact_limit
        return FakeResult()

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path / "archive")
    monkeypatch.setattr("polylogue.paths.render_root", lambda: tmp_path / "render")
    monkeypatch.setattr(
        "polylogue.storage.blob_integrity.restore_direct_blob_reference_debt",
        fake_restore_direct_blob_reference_debt,
    )
    monkeypatch.setattr("polylogue.storage.repair.repair_raw_materialization", fake_repair_raw_materialization)

    assert daemon_cli._drain_raw_materialization_once(limit=11) == 7
    assert order == ["restore", "materialize"]
    assert calls == {
        "restore_db_path": tmp_path / "archive" / "source.db",
        "restore_dry_run": False,
        "restore_max_count": 25,
        "restore_sample_size": 0,
        "archive_root": tmp_path / "archive",
        "render_root": tmp_path / "render",
        "dry_run": False,
        "raw_artifact_limit": 11,
    }


def test_raw_materialization_pass_emits_conserved_plan_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.daemon import cli as daemon_cli

    events: list[tuple[str, dict[str, object]]] = []
    outcome = SimpleNamespace(
        status=SimpleNamespace(value="executed"),
        plan_id="raw-replay:stable",
        reason="applied",
        to_dict=lambda: {
            "plan_id": "raw-replay:stable",
            "input_raw_ids": ["raw-a"],
            "status": "executed",
            "reason": "applied",
            "next_action": "none",
        },
    )
    monkeypatch.setattr(
        "polylogue.daemon.events.emit_daemon_event",
        lambda kind, *, payload: events.append((kind, payload)),
    )
    monkeypatch.setattr(os, "urandom", lambda _size: b"x" * 16)

    daemon_cli._emit_raw_materialization_pass(
        SimpleNamespace(
            success=True,
            repaired_count=1,
            detail="done",
            metrics={
                "raw_materialization_candidate_count": 1.0,
                "raw_materialization_remaining_candidate_count": 0.0,
            },
            plan_outcomes=(outcome,),
        )
    )

    assert events == [
        (
            "raw_materialization_pass",
            {
                "pass_id": f"raw-materialization:{(b'x' * 16).hex()}",
                "success": True,
                "repaired_count": 1,
                "detail": "done",
                "metrics": {
                    "raw_materialization_candidate_count": 1.0,
                    "raw_materialization_remaining_candidate_count": 0.0,
                },
                "plan_outcomes": [outcome.to_dict()],
            },
        )
    ]


def test_raw_materialization_pass_emits_zero_work_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.daemon import cli as daemon_cli

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "polylogue.daemon.events.emit_daemon_event",
        lambda kind, *, payload: events.append((kind, payload)),
    )
    monkeypatch.setattr(os, "urandom", lambda _size: b"z" * 16)

    daemon_cli._emit_raw_materialization_pass(
        SimpleNamespace(
            success=True,
            repaired_count=0,
            detail="Executable raw replay converged",
            metrics={"raw_materialization_candidate_count": 0.0},
            plan_outcomes=(),
        )
    )

    assert events[0][1]["success"] is True
    assert events[0][1]["plan_outcomes"] == []


def test_raw_materialization_closes_fts_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    archive = tmp_path / "archive"
    closed: list[Path] = []

    class FakeRestoreResult:
        restored_count = 0

    def cancel_repair(*_args: object, **_kwargs: object) -> object:
        raise asyncio.CancelledError

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive)
    monkeypatch.setattr("polylogue.paths.render_root", lambda: tmp_path / "render")
    monkeypatch.setattr(
        "polylogue.storage.blob_integrity.restore_direct_blob_reference_debt",
        lambda *_args, **_kwargs: FakeRestoreResult(),
    )
    monkeypatch.setattr("polylogue.storage.repair.repair_raw_materialization", cancel_repair)
    monkeypatch.setattr(daemon_cli, "_close_raw_materialization_fts", closed.append)

    with pytest.raises(asyncio.CancelledError):
        daemon_cli._drain_raw_materialization_once()

    assert closed == [archive / "index.db"]


def test_raw_materialization_fts_failure_records_durable_debt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    index_db = tmp_path / "index.db"
    index_db.touch()
    calls: list[tuple[str, str, str, str | None]] = []

    class FakeCursor:
        def __init__(self, db: Path) -> None:
            assert db == index_db

        def clear_convergence_debt(self, **_kwargs: object) -> None:
            raise AssertionError("failed FTS repair must not clear debt")

        def record_convergence_debt(
            self,
            *,
            stage: str,
            subject_type: str,
            subject_id: str,
            error: str | None = None,
        ) -> None:
            calls.append((stage, subject_type, subject_id, error))

    monkeypatch.setattr(daemon_cli, "_raw_materialization_fts_needs_repair", lambda _db: True)
    monkeypatch.setattr("polylogue.daemon.convergence_stages.repair_fts_surface", lambda *_args: False)
    monkeypatch.setattr("polylogue.sources.live.cursor.CursorStore", FakeCursor)

    daemon_cli._close_raw_materialization_fts(index_db)

    assert calls == [
        (
            "fts",
            "fts_surface",
            "messages_fts",
            "raw materialization exited without restoring message FTS readiness",
        )
    ]


def test_raw_materialization_fts_success_clears_prior_debt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    index_db = tmp_path / "index.db"
    index_db.touch()
    cleared: list[dict[str, object]] = []

    class FakeCursor:
        def __init__(self, db: Path) -> None:
            assert db == index_db

        def clear_convergence_debt(self, **kwargs: object) -> None:
            cleared.append(kwargs)

        def record_convergence_debt(self, **_kwargs: object) -> None:
            raise AssertionError("successful FTS repair must not record debt")

    monkeypatch.setattr(daemon_cli, "_raw_materialization_fts_needs_repair", lambda _db: True)
    monkeypatch.setattr("polylogue.daemon.convergence_stages.repair_fts_surface", lambda *_args: True)
    monkeypatch.setattr("polylogue.sources.live.cursor.CursorStore", FakeCursor)

    daemon_cli._close_raw_materialization_fts(index_db)

    assert cleared == [{"subject_type": "fts_surface", "subject_id": "messages_fts", "stage": "fts"}]


def test_raw_materialization_fts_exception_becomes_explicit_debt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    index_db = tmp_path / "index.db"
    index_db.touch()
    errors: list[str | None] = []

    class FakeCursor:
        def __init__(self, db: Path) -> None:
            assert db == index_db

        def record_convergence_debt(self, *, error: str | None = None, **_kwargs: object) -> None:
            errors.append(error)

    monkeypatch.setattr(daemon_cli, "_raw_materialization_fts_needs_repair", lambda _db: True)
    monkeypatch.setattr(
        "polylogue.daemon.convergence_stages.repair_fts_surface",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected FTS failure")),
    )
    monkeypatch.setattr("polylogue.sources.live.cursor.CursorStore", FakeCursor)

    daemon_cli._close_raw_materialization_fts(index_db)

    assert errors == ["FTS repair failed after raw materialization: RuntimeError: injected FTS failure"]


def test_periodic_raw_materialization_convergence_treats_sqlite_lock_as_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        raise asyncio.CancelledError

    async def fake_run_sync(_actor: str, _func: object, *_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        daemon_cli,
        "daemon_write_coordinator",
        lambda: SimpleNamespace(run_sync=fake_run_sync),
    )
    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch.object(daemon_cli.logger, "info") as info,
        patch.object(daemon_cli.logger, "warning") as warning,
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_raw_materialization_convergence())

    info.assert_called_once()
    assert info.call_args.args[0] == "raw materialization: archive busy; retrying on next tick: %s"
    warning.assert_not_called()


def test_periodic_raw_materialization_convergence_waits_for_catch_up_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    calls: list[str] = []

    async def fake_run_sync(_actor: str, _func: object, *_args: object, **_kwargs: object) -> object:
        calls.append("drain")
        raise asyncio.CancelledError

    async def exercise() -> None:
        catch_up_complete = asyncio.Event()
        monkeypatch.setattr(
            daemon_cli,
            "daemon_write_coordinator",
            lambda: SimpleNamespace(run_sync=fake_run_sync),
        )
        task = asyncio.create_task(
            daemon_cli._periodic_raw_materialization_convergence_after(catch_up_complete=catch_up_complete)
        )
        await asyncio.sleep(0)
        assert calls == []
        catch_up_complete.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert calls == ["drain"]


def test_periodic_session_insight_convergence_waits_for_catch_up_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    calls: list[str] = []

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        calls.append("drain")
        raise asyncio.CancelledError

    async def exercise() -> None:
        catch_up_complete = asyncio.Event()
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        task = asyncio.create_task(
            daemon_cli._periodic_session_insight_convergence_after(catch_up_complete=catch_up_complete)
        )
        await asyncio.sleep(0)
        assert calls == []
        catch_up_complete.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert calls == ["drain"]


def test_periodic_session_insight_convergence_bursts_successful_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    drains: list[int] = []
    sleeps: list[float] = []

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> int:
        drains.append(len(drains))
        return 100 if len(drains) <= 2 else 0

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if seconds == daemon_cli._SESSION_INSIGHT_CONVERGENCE_INTERVAL_SECONDS:
            raise asyncio.CancelledError

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_session_insight_convergence_after())

    assert drains == [0, 1, 2]
    assert sleeps == [
        daemon_cli._SESSION_INSIGHT_CONVERGENCE_BURST_PAUSE_SECONDS,
        daemon_cli._SESSION_INSIGHT_CONVERGENCE_BURST_PAUSE_SECONDS,
        daemon_cli._SESSION_INSIGHT_CONVERGENCE_INTERVAL_SECONDS,
    ]


def test_periodic_session_insight_convergence_treats_sqlite_lock_as_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    async def fake_sleep(_seconds: float) -> None:
        raise asyncio.CancelledError

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("database is locked")

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        patch.object(daemon_cli.logger, "info") as info,
        patch.object(daemon_cli.logger, "warning") as warning,
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_session_insight_convergence_after())

    info.assert_called_once()
    assert info.call_args.args[0] == "insights: archive busy; retrying profile backlog on next tick: %s"
    warning.assert_not_called()


def test_periodic_wal_checkpoint_targets_archive_root_tiers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.wal_checkpoint import maybe_checkpoint_archive_wals

    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        calls.append((func, args, kwargs))
        raise asyncio.CancelledError

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(daemon_cli._periodic_wal_checkpoint())

    assert calls == [(maybe_checkpoint_archive_wals, (tmp_path,), {"reason": "periodic"})]


def test_periodic_convergence_check_waits_for_catch_up_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()
    calls: list[str] = []
    drained = asyncio.Event()

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        calls.append("drain")
        drained.set()
        return 0

    async def exercise() -> None:
        catch_up_complete = asyncio.Event()
        monkeypatch.setattr(daemon_cli, "_CONVERGENCE_DEBT_RETRY_INTERVAL_SECONDS", 60)
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        monkeypatch.setattr(daemon_cli, "_active_index_db_path", lambda: db)
        task = asyncio.create_task(daemon_cli._periodic_convergence_check((), catch_up_complete=catch_up_complete))
        await asyncio.sleep(0)
        assert calls == []
        catch_up_complete.set()
        await asyncio.wait_for(drained.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert calls == ["drain"]


def test_periodic_convergence_check_warns_on_non_lock_failures(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    db = tmp_path / "index.db"
    db.touch()

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected convergence retry failure")

    with (
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        patch.object(daemon_cli.logger, "info") as info,
        patch.object(daemon_cli.logger, "warning") as warning,
    ):
        asyncio.run(daemon_cli._retry_convergence_debt_once(db))

    info.assert_not_called()
    warning.assert_called_once()
    assert warning.call_args.args[0] == "convergence: check failed"
    assert warning.call_args.kwargs == {"exc_info": True}


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
                    debounce_s=1.0,
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
        result = CliRunner().invoke(main, ["run", "--no-browser-capture", "--no-api", "--debounce-s", "0.25"])

    assert result.exit_code == 0
    default_sources.assert_called_once_with()
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Starting polylogued (watch=1 source(s)). Ctrl-C to stop." in result.stderr


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
    assert tuple(source.root for source in recorded_sources) == (Path("/tmp/codex"),)


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
        changed = asyncio.run(daemon_cli._run_drive_source_catchup_once())

    assert changed == 0
    build_services.assert_not_called()


def test_drive_source_catchup_ingests_configured_drive_source(tmp_path: Path) -> None:
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

        def get_backend(self) -> object:
            events.append("backend")
            return object()

        async def close(self) -> None:
            events.append("close")

    class FakeParser:
        def __init__(self, *, repository: object, archive_root: Path, config: Config) -> None:
            events.append(("parser", repository, archive_root, config))

        async def ingest_sources(self, *, sources: list[Source], stage: str, parse_records: bool) -> SimpleNamespace:
            events.append(("ingest", sources, stage, parse_records))
            return SimpleNamespace(
                acquire_result=SimpleNamespace(raw_ids=["raw-1"], errors=0),
                parse_result=SimpleNamespace(
                    processed_ids={"session-b", "session-a"},
                    counts={"sessions": 2},
                ),
            )

    async def fake_refresh(_backend: object, session_ids: list[str]) -> None:
        events.append(("refresh", session_ids))

    with (
        patch("polylogue.config.get_config", return_value=config),
        patch("polylogue.services.build_runtime_services", return_value=FakeServices()) as build_services,
        patch("polylogue.pipeline.services.parsing.ParsingService", FakeParser),
        patch("polylogue.pipeline.services.ingest_batch.refresh_session_insights_bulk", fake_refresh),
    ):
        changed = asyncio.run(daemon_cli._run_drive_source_catchup_once())

    assert changed == 2
    build_services.assert_called_once_with(config=config, db_path=config.db_path)
    assert ("ingest", [drive_source], "all", True) in events
    assert ("refresh", ["session-a", "session-b"]) in events
    assert events[-1] == "close"


def test_drive_source_catchup_safe_wrapper_logs_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    async def fail_catchup() -> int:
        raise RuntimeError("drive unavailable")

    with (
        patch.object(daemon_cli, "_run_drive_source_catchup_once", fail_catchup),
        patch.object(daemon_cli.logger, "warning") as warning,
    ):
        changed = asyncio.run(daemon_cli._run_drive_source_catchup_safely())

    assert changed == 0
    warning.assert_called_once_with("daemon: Drive source catch-up failed", exc_info=True)


def test_explicit_archive_inbox_root_keeps_import_suffixes(workspace_env: dict[str, Path]) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.live.watcher import INBOX_SOURCE_SUFFIXES

    inbox = workspace_env["archive_root"] / "inbox"
    ordinary = workspace_env["archive_root"] / "ordinary-jsonl-root"

    sources = daemon_cli._watch_sources_from_roots((inbox, ordinary))

    assert sources == (
        WatchSource(name="inbox", root=inbox, suffixes=INBOX_SOURCE_SUFFIXES),
        WatchSource(name="ordinary-jsonl-root", root=ordinary, suffixes=(".jsonl",)),
    )


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

    assert sources == (
        WatchSource(name="browser-capture", root=spool, suffixes=(".json",)),
        WatchSource(name="ordinary-jsonl-root", root=ordinary, suffixes=(".jsonl",)),
    )


def test_explicit_browser_capture_root_uses_spool_override_classifier(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli

    override_spool = tmp_path / "override-browser-capture"
    ordinary = tmp_path / "ordinary-jsonl-root"

    sources = daemon_cli._watch_sources_from_roots(
        (override_spool, ordinary),
        browser_capture_spool_path=override_spool,
    )

    assert sources == (
        WatchSource(name="browser-capture", root=override_spool, suffixes=(".json",)),
        WatchSource(name="ordinary-jsonl-root", root=ordinary, suffixes=(".jsonl",)),
    )


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


def test_archive_message_fts_startup_large_drift_is_deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.daemon import fts_startup

    class FakeCursor:
        def __init__(self, row: tuple[object, ...] | None, rows: list[tuple[object, ...]] | None = None) -> None:
            self._row = row
            self._rows = rows if rows is not None else ([] if row is None else [row])

        def fetchone(self) -> tuple[object, ...] | None:
            return self._row

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class FakeConnection:
        def execute(self, sql: str, params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(params[0]) if isinstance(params, tuple) and params else ""
                return (
                    FakeCursor((1,)) if name in {"blocks", "messages_fts", "messages_fts_docsize"} else FakeCursor(None)
                )
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                triggers: list[tuple[object, ...]] = [("messages_fts_ai",), ("messages_fts_ad",), ("messages_fts_au",)]
                return FakeCursor(triggers[0], rows=triggers)
            if query.startswith("SELECT state, source_rows, indexed_rows"):
                return FakeCursor(None)
            if query == "SELECT COUNT(*) FROM blocks WHERE search_text != ''":
                return FakeCursor((250_000,))
            if query == "SELECT COUNT(*) FROM messages_fts_docsize":
                return FakeCursor((100_000,))
            raise AssertionError(f"unexpected query: {query}")

    rebuilds: list[FakeConnection] = []
    records: list[dict[str, object]] = []
    debts: list[dict[str, object]] = []

    monkeypatch.setattr("polylogue.storage.sqlite.archive_tiers.bootstrap.initialize_archive_tier", lambda *_args: None)
    monkeypatch.setattr("polylogue.storage.fts.freshness.ensure_fts_freshness_table_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", lambda conn: rebuilds.append(conn)
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.freshness.record_fts_surface_state_sync",
        lambda _conn, **kwargs: records.append(kwargs),
    )

    class FakeCursorStore:
        def __init__(self, db_path: Path) -> None:
            assert db_path == Path("/archive/index.db")

        def record_convergence_debt(self, **kwargs: object) -> None:
            debts.append(kwargs)

    monkeypatch.setattr("polylogue.sources.live.cursor.CursorStore", FakeCursorStore)

    assert (
        fts_startup._ensure_archive_messages_fts_startup_readiness_sync(
            cast(sqlite3.Connection, FakeConnection()),
            db_path=Path("/archive/index.db"),
        )
        is True
    )

    assert rebuilds == []
    assert records == [
        {
            "surface": "messages_fts",
            "state": "stale",
            "source_rows": 250_000,
            "indexed_rows": 100_000,
            "missing_rows": 150_000,
            "excess_rows": 0,
            "detail": (
                "archive message FTS drift exceeds bounded startup reconciliation; scheduled global FTS freshness debt"
            ),
        }
    ]
    assert debts == [
        {
            "stage": "fts",
            "subject_type": "fts_surface",
            "subject_id": "messages_fts",
            "error": (
                "archive message FTS drift exceeds bounded startup reconciliation; scheduled global FTS freshness debt"
            ),
        }
    ]


def test_archive_message_fts_startup_records_known_stale_ledger_without_global_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import fts_startup

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

        def execute(self, sql: str, params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(params[0]) if isinstance(params, tuple) and params else ""
                return (
                    FakeCursor((1,)) if name in {"blocks", "messages_fts", "messages_fts_docsize"} else FakeCursor(None)
                )
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                triggers: list[tuple[object, ...]] = [("messages_fts_ai",), ("messages_fts_ad",), ("messages_fts_au",)]
                return FakeCursor(triggers[0], rows=triggers)
            if query.startswith("SELECT state, source_rows, indexed_rows"):
                return FakeCursor(("stale", 250_000, 100_000, 150_000, 0, 0))
            raise AssertionError(f"unexpected query: {query}")

    conn = FakeConnection()
    rebuilds: list[FakeConnection] = []
    records: list[dict[str, object]] = []

    monkeypatch.setattr("polylogue.storage.sqlite.archive_tiers.bootstrap.initialize_archive_tier", lambda *_args: None)
    monkeypatch.setattr("polylogue.storage.fts.freshness.ensure_fts_freshness_table_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", lambda fake_conn: rebuilds.append(fake_conn)
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.freshness.record_fts_surface_state_sync",
        lambda _conn, **kwargs: records.append(kwargs),
    )

    assert fts_startup._ensure_archive_messages_fts_startup_readiness_sync(cast(sqlite3.Connection, conn)) is True

    assert rebuilds == []
    assert records == [
        {
            "surface": "messages_fts",
            "state": "stale",
            "source_rows": 250_000,
            "indexed_rows": 100_000,
            "missing_rows": 150_000,
            "excess_rows": 0,
            "duplicate_rows": 0,
            "detail": (
                "archive message FTS drift exceeds bounded startup reconciliation; scheduled global FTS freshness debt"
            ),
        }
    ]
    assert "SELECT COUNT(*) FROM blocks WHERE search_text != ''" not in conn.queries
    assert "SELECT COUNT(*) FROM messages_fts_docsize" not in conn.queries


def test_archive_message_fts_startup_downgrades_inconsistent_ready_ledger_without_global_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import fts_startup

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

        def execute(self, sql: str, params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(params[0]) if isinstance(params, tuple) and params else ""
                return (
                    FakeCursor((1,)) if name in {"blocks", "messages_fts", "messages_fts_docsize"} else FakeCursor(None)
                )
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                triggers: list[tuple[object, ...]] = [("messages_fts_ai",), ("messages_fts_ad",), ("messages_fts_au",)]
                return FakeCursor(triggers[0], rows=triggers)
            if query.startswith("SELECT state, source_rows, indexed_rows"):
                return FakeCursor(("ready", 250_000, 100_000, 0, 0, 0))
            raise AssertionError(f"unexpected query: {query}")

    conn = FakeConnection()
    records: list[dict[str, object]] = []
    debts: list[dict[str, object]] = []

    monkeypatch.setattr("polylogue.storage.sqlite.archive_tiers.bootstrap.initialize_archive_tier", lambda *_args: None)
    monkeypatch.setattr("polylogue.storage.fts.freshness.ensure_fts_freshness_table_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.freshness.record_fts_surface_state_sync",
        lambda _conn, **kwargs: records.append(kwargs),
    )

    class FakeCursorStore:
        def __init__(self, db_path: Path) -> None:
            assert db_path == Path("/archive/index.db")

        def record_convergence_debt(self, **kwargs: object) -> None:
            debts.append(kwargs)

    monkeypatch.setattr("polylogue.sources.live.cursor.CursorStore", FakeCursorStore)

    assert (
        fts_startup._ensure_archive_messages_fts_startup_readiness_sync(
            cast(sqlite3.Connection, conn),
            db_path=Path("/archive/index.db"),
        )
        is True
    )

    assert records == [
        {
            "surface": "messages_fts",
            "state": "stale",
            "source_rows": 250_000,
            "indexed_rows": 100_000,
            "missing_rows": 150_000,
            "excess_rows": 0,
            "duplicate_rows": 0,
            "detail": (
                "archive message FTS drift exceeds bounded startup reconciliation; scheduled global FTS freshness debt"
            ),
        }
    ]
    assert debts == [
        {
            "stage": "fts",
            "subject_type": "fts_surface",
            "subject_id": "messages_fts",
            "error": "startup found inconsistent messages_fts ready freshness ledger",
        }
    ]
    assert "SELECT COUNT(*) FROM blocks WHERE search_text != ''" not in conn.queries
    assert "SELECT COUNT(*) FROM messages_fts_docsize" not in conn.queries


def test_archive_message_fts_startup_records_poisoned_stale_zero_ledger_without_global_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.daemon import fts_startup

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

        def execute(self, sql: str, params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(params[0]) if isinstance(params, tuple) and params else ""
                return (
                    FakeCursor((1,)) if name in {"blocks", "messages_fts", "messages_fts_docsize"} else FakeCursor(None)
                )
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                triggers: list[tuple[object, ...]] = [("messages_fts_ai",), ("messages_fts_ad",), ("messages_fts_au",)]
                return FakeCursor(triggers[0], rows=triggers)
            if query.startswith("SELECT state, source_rows, indexed_rows"):
                return FakeCursor(("stale", 0, 0, 0, 0, 0))
            if query == "SELECT 1 FROM blocks WHERE search_text != '' LIMIT 1":
                return FakeCursor((1,))
            if query == "SELECT 1 FROM messages_fts_docsize LIMIT 1":
                return FakeCursor((1,))
            raise AssertionError(f"unexpected query: {query}")

    conn = FakeConnection()
    records: list[dict[str, object]] = []
    debts: list[dict[str, object]] = []

    monkeypatch.setattr("polylogue.storage.sqlite.archive_tiers.bootstrap.initialize_archive_tier", lambda *_args: None)
    monkeypatch.setattr("polylogue.storage.fts.freshness.ensure_fts_freshness_table_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.freshness.record_fts_surface_state_sync",
        lambda _conn, **kwargs: records.append(kwargs),
    )

    class FakeCursorStore:
        def __init__(self, db_path: Path) -> None:
            assert db_path == Path("/archive/index.db")

        def record_convergence_debt(self, **kwargs: object) -> None:
            debts.append(kwargs)

    monkeypatch.setattr("polylogue.sources.live.cursor.CursorStore", FakeCursorStore)

    assert (
        fts_startup._ensure_archive_messages_fts_startup_readiness_sync(
            cast(sqlite3.Connection, conn),
            db_path=Path("/archive/index.db"),
        )
        is True
    )

    assert records == [
        {
            "surface": "messages_fts",
            "state": "stale",
            "source_rows": 0,
            "indexed_rows": 0,
            "missing_rows": 0,
            "excess_rows": 0,
            "duplicate_rows": 0,
            "detail": (
                "archive message FTS drift exceeds bounded startup reconciliation; scheduled global FTS freshness debt"
            ),
        }
    ]
    assert debts == [
        {
            "stage": "fts",
            "subject_type": "fts_surface",
            "subject_id": "messages_fts",
            "error": "startup found stale messages_fts freshness ledger",
        }
    ]
    assert "SELECT COUNT(*) FROM blocks WHERE search_text != ''" not in conn.queries
    assert "SELECT COUNT(*) FROM messages_fts_docsize" not in conn.queries


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
            if query == "PRAGMA busy_timeout = 120000":
                return FakeCursor(None)
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


def test_ensure_fts_startup_readiness_trusts_ready_freshness_without_counts(
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

        def execute(self, sql: str, params: object = ()) -> FakeCursor:
            query = " ".join(sql.split())
            self.queries.append(query)
            lowered = query.lower()
            if "count(*)" in lowered:
                raise AssertionError(f"startup readiness must not exact-count ready FTS ledgers: {query}")
            if query == "PRAGMA busy_timeout = 120000":
                return FakeCursor(None)
            if query.startswith("CREATE TABLE IF NOT EXISTS fts_freshness_state"):
                return FakeCursor(None)
            if query == "PRAGMA table_info(fts_freshness_state)":
                return FakeCursor(
                    None,
                    rows=[
                        (0, "surface"),
                        (1, "state"),
                        (2, "checked_at"),
                        (3, "source_rows"),
                        (4, "indexed_rows"),
                        (5, "missing_rows"),
                        (6, "excess_rows"),
                        (7, "duplicate_rows"),
                        (8, "detail"),
                    ],
                )
            if query == "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1":
                name = str(params[0]) if isinstance(params, tuple) and params else ""
                return (
                    FakeCursor((1,)) if name in {"blocks", "messages_fts", "messages_fts_docsize"} else FakeCursor(None)
                )
            if query.startswith("SELECT name FROM sqlite_master WHERE type='trigger'"):
                triggers: list[tuple[object, ...]] = [
                    ("messages_fts_ai",),
                    ("messages_fts_ad",),
                    ("messages_fts_au",),
                ]
                return FakeCursor(None, rows=triggers)
            if query.startswith("SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows"):
                return FakeCursor(("ready", 10, 10, 0, 0, 0))
            raise AssertionError(f"unexpected query: {query}")

        def commit(self) -> None:
            self.committed = True

        def close(self) -> None:
            self.closed = True

    conn = FakeConnection()
    rebuilds: list[FakeConnection] = []
    optional_repairs: list[FakeConnection] = []
    monkeypatch.setattr("polylogue.paths.active_index_db_path", lambda: db)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", lambda _db, timeout: conn)
    monkeypatch.setattr("polylogue.storage.sqlite.archive_tiers.bootstrap.initialize_archive_tier", lambda *_args: None)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", lambda c: rebuilds.append(c))
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.configure_bounded_repair_connection",
        lambda _conn: None,
    )

    def fake_repair_stale_fts_rows(c: FakeConnection) -> SimpleNamespace:
        optional_repairs.append(c)
        return SimpleNamespace(success=True, repaired_count=2, detail="repaired optional surfaces")

    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.repair_stale_fts_rows",
        fake_repair_stale_fts_rows,
    )

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    assert rebuilds == []
    assert optional_repairs == [conn]
    assert conn.committed is True
    assert conn.closed is True


def test_optional_fts_startup_failure_records_convergence_debt(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.daemon import fts_startup

    debts: list[dict[str, object]] = []

    class FakeCursorStore:
        def __init__(self, db_path: Path) -> None:
            assert db_path == Path("/archive/index.db")

        def record_convergence_debt(self, **kwargs: object) -> None:
            debts.append(kwargs)

    monkeypatch.setattr("polylogue.daemon.fts_startup._message_fts_freshness_ready_sync", lambda _conn: True)
    monkeypatch.setattr("polylogue.storage.fts.dangling_repair.configure_bounded_repair_connection", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.dangling_repair.repair_stale_fts_rows",
        lambda _conn: SimpleNamespace(success=False, repaired_count=0, detail="optional surfaces still stale"),
    )
    monkeypatch.setattr("polylogue.sources.live.cursor.CursorStore", FakeCursorStore)

    fts_startup._repair_startup_optional_fts_surfaces_sync(
        cast(sqlite3.Connection, object()),
        db_path=Path("/archive/index.db"),
    )

    assert debts == [
        {
            "stage": "fts",
            "subject_type": "fts_surface",
            "subject_id": "session_work_events_fts",
            "error": "optional surfaces still stale",
        },
        {
            "stage": "fts",
            "subject_type": "fts_surface",
            "subject_id": "threads_fts",
            "error": "optional surfaces still stale",
        },
    ]


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


def test_periodic_db_optimize_targets_archive_root_tiers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.sqlite.maintenance import maybe_optimize_archive_tiers

    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def fake_sleep(_seconds: float) -> None:
        return None

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        calls.append((func, args, kwargs))
        raise asyncio.CancelledError

    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
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
        archive.write_parsed(
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
            )
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


def test_ensure_fts_startup_readiness_handles_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

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
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="startup v1")],
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


def test_ensure_fts_startup_readiness_uses_extended_write_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import polylogue.daemon.fts_startup as fts_startup
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.daemon import cli as daemon_cli
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    with ArchiveStore(tmp_path) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="daemon-startup-timeout-v1",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="startup timeout v1",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="startup timeout v1")],
                    )
                ],
            )
        )

    seen_busy_timeout: list[int] = []
    real_ensure = fts_startup._ensure_archive_messages_fts_startup_readiness_sync

    def wrapped_ensure(conn: sqlite3.Connection, *, db_path: Path | None = None) -> bool:
        seen_busy_timeout.append(int(conn.execute("PRAGMA busy_timeout").fetchone()[0]))
        return real_ensure(conn, db_path=db_path)

    monkeypatch.setattr(fts_startup, "_ensure_archive_messages_fts_startup_readiness_sync", wrapped_ensure)

    asyncio.run(daemon_cli._ensure_fts_startup_readiness())

    assert seen_busy_timeout == [fts_startup._FTS_STARTUP_BUSY_TIMEOUT_MS]


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
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
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
    monkeypatch.setattr("polylogue.paths.active_index_db_path", lambda: tmp_path / "index.db")
    monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: Coordinator())

    with pytest.raises(RuntimeError, match="ops unavailable"):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert not (tmp_path / "daemon.pid").exists()


def test_run_daemon_services_checks_archive_identity_before_component_startup(tmp_path: Path) -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.storage.archive_identity import ArchiveIdentityConflictError

    configure = Mock()
    with (
        patch("polylogue.paths.archive_root", return_value=tmp_path / "configured"),
        patch("polylogue.paths.active_index_db_path", return_value=tmp_path / "active" / "index.db"),
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
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    configure.assert_not_called()


def test_emit_daemon_lifecycle_event_carries_dev_loop_context(
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

    monkeypatch.setenv("POLYLOGUE_DEV_LOOP_RUN_ID", "dev-loop-run")
    monkeypatch.setenv("POLYLOGUE_DEV_LOOP_LOG_DIR", str(tmp_path / "logs"))

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
    assert kwargs["operation_id"] == "dev-loop-run"
    payload = cast(dict[str, object], kwargs["payload"])
    assert payload["phase"] == "component_started"
    assert payload["component"] == "api"
    assert payload["port"] == 8766
    assert payload["archive_root"] == str(tmp_path / "archive")
    assert payload["dev_loop_run_id"] == "dev-loop-run"
    assert payload["dev_loop_log_dir"] == str(tmp_path / "logs")


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


def test_run_daemon_services_waits_for_fts_startup_before_watcher() -> None:
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    events: list[str] = []
    lifecycle_payloads: list[dict[str, object]] = []
    watcher_coordinators: list[object] = []
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

    class FakeWatcher:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.catch_up_complete = asyncio.Event()
            watcher_coordinators.append(kwargs["write_coordinator"])

        async def run(self) -> None:
            events.append("watcher")
            self.catch_up_complete.set()
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            events.append("stop")

    def fake_fts_startup() -> None:
        events.append("fts")

    def fake_lineage_startup() -> int:
        events.append("lineage")
        return 0

    async def fake_reconcile_blob_publications() -> None:
        events.append("blob-publications")

    async def fake_drive_catchup() -> int:
        events.append("drive-once")
        return 0

    async def fake_configure_fts_automerge() -> None:
        events.append("automerge")

    async def fake_loop(name: str) -> None:
        events.append(name)
        await asyncio.Event().wait()

    class FakeConverger:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            events.append("converger")

        async def stop(self) -> None:
            events.append("converger-stop")

    class FakeAPIServer:
        def __init__(self) -> None:
            self.stopped = threading.Event()

        def serve_forever(self, _poll_interval: float) -> None:
            self.stopped.wait(timeout=2.0)

        def shutdown(self) -> None:
            self.stopped.set()

        def server_close(self) -> None:
            return None

    api_server = FakeAPIServer()
    api_server_factory = Mock(return_value=api_server)

    def fake_emit_daemon_event(kind: str, **kwargs: object) -> None:
        assert kind == "daemon.lifecycle"
        lifecycle_payloads.append(cast(dict[str, object], kwargs["payload"]))

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(daemon_cli, "Polylogue", FakePolylogue))
        stack.enter_context(patch.object(daemon_cli, "LiveWatcher", FakeWatcher))
        stack.enter_context(patch.object(daemon_cli, "_ensure_fts_startup_readiness_sync", fake_fts_startup))
        stack.enter_context(patch.object(daemon_cli, "_ensure_lineage_startup_readiness_sync", fake_lineage_startup))
        stack.enter_context(patch.object(daemon_cli, "_reconcile_blob_publications", fake_reconcile_blob_publications))
        stack.enter_context(patch.object(daemon_cli, "_check_schema_version_fast", return_value=ok_schema))
        stack.enter_context(patch("polylogue.paths.archive_root", return_value=Path("/tmp/polylogue-test-archive")))
        stack.enter_context(patch.object(daemon_cli, "_run_drive_source_catchup_safely", fake_drive_catchup))
        stack.enter_context(patch.object(daemon_cli, "_configure_fts_automerge", fake_configure_fts_automerge))
        stack.enter_context(patch.object(daemon_cli, "_periodic_wal_checkpoint", lambda: fake_loop("wal")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_fts_merge", lambda: fake_loop("fts-merge")))
        stack.enter_context(
            patch.object(
                daemon_cli,
                "_periodic_raw_materialization_convergence_after",
                lambda _gate=None: fake_loop("raw-materialization"),
            )
        )
        stack.enter_context(
            patch.object(
                daemon_cli,
                "_periodic_session_insight_convergence_after",
                lambda _gate=None: fake_loop("session-insights"),
            )
        )
        stack.enter_context(patch.object(daemon_cli, "_periodic_heartbeat", lambda: fake_loop("heartbeat")))
        stack.enter_context(
            patch.object(
                daemon_cli, "_periodic_convergence_check", lambda _sources, **_kwargs: fake_loop("convergence")
            )
        )
        stack.enter_context(patch.object(daemon_cli, "_periodic_health_check", lambda: fake_loop("health")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_db_optimize", lambda: fake_loop("optimize")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_status_snapshot_refresh", lambda: fake_loop("status")))
        stack.enter_context(patch.object(daemon_cli, "_periodic_drive_source_catchup", lambda: fake_loop("drive")))
        stack.enter_context(
            patch(
                "polylogue.daemon.embedding_backlog.periodic_embedding_backlog_check",
                lambda **_kwargs: fake_loop("embedding"),
            )
        )
        stack.enter_context(patch("polylogue.daemon.convergence.DaemonConverger", FakeConverger))
        stack.enter_context(
            patch("polylogue.daemon.convergence_stages.make_default_convergence_stages", return_value=())
        )
        stack.enter_context(patch("polylogue.daemon.http.DaemonAPIHTTPServer", api_server_factory))
        stack.enter_context(patch("polylogue.daemon.events.emit_daemon_event", side_effect=fake_emit_daemon_event))
        stack.enter_context(pytest.raises(RuntimeError, match="watch stopped"))
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                debounce_s=1.0,
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                enable_api=True,
            )
        )

    assert "watcher" in events
    assert events.index("fts") < events.index("watcher")
    assert events.index("fts") < events.index("lineage") < events.index("watcher")
    assert events.index("lineage") < events.index("blob-publications") < events.index("watcher")
    assert events.index("drive-once") < events.index("watcher")
    assert events.index("blob-publications") < events.index("drive-once")
    assert events.index("lineage") < events.index("convergence")
    assert events.index("lineage") < events.index("raw-materialization")
    assert events.index("lineage") < events.index("drive")
    assert events.index("lineage") < events.index("converger")
    assert events.count("convergence") == 1
    lifecycle_phases = [str(payload["phase"]) for payload in lifecycle_payloads]
    assert lifecycle_phases[0] == "startup"
    assert "component_ready" in lifecycle_phases
    assert lifecycle_phases[-1] == "shutdown_started"
    assert "shutdown_complete" not in lifecycle_phases
    lifecycle_components = {payload.get("component") for payload in lifecycle_payloads}
    assert {"fts_startup", "lineage_startup", "converger", "watcher"}.issubset(lifecycle_components)
    assert len(watcher_coordinators) == 1
    bridge = api_server_factory.call_args.kwargs["write_bridge"]
    assert bridge._coordinator is watcher_coordinators[0]


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

    async def noop() -> None:
        return None

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
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
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


def test_report_drain_exceptions_ignores_expected_cancellations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from polylogue.daemon import cli as daemon_cli

    daemon_cli._report_drain_exceptions([None, asyncio.CancelledError()])

    assert "task raised during shutdown" not in caplog.text


def test_run_daemon_services_drains_servers_when_main_task_is_cancelled() -> None:
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

    class FakeConverger:
        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    async def noop() -> None:
        return None

    def noop_sync() -> None:
        return None

    async def no_drive_changes() -> int:
        return 0

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    browser_server = BlockingServer()
    api_server = BlockingServer()
    interrupted_cleanup_calls = 0

    def mark_interrupted_cleanup() -> None:
        nonlocal interrupted_cleanup_calls
        interrupted_cleanup_calls += 1

    async def exercise() -> None:
        task = asyncio.create_task(
            daemon_cli.run_daemon_services(
                sources=(),
                debounce_s=1.0,
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
        while not (browser_server.ready.is_set() and api_server.ready.is_set()):
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

    with (
        patch.object(daemon_cli, "make_server", return_value=browser_server),
        patch.object(daemon_cli, "_ensure_fts_startup_readiness_sync", noop_sync),
        patch.object(daemon_cli, "_ensure_lineage_startup_readiness_sync", noop_sync),
        patch.object(daemon_cli, "_reconcile_blob_publications", noop),
        patch.object(daemon_cli, "_configure_fts_automerge", noop),
        patch.object(daemon_cli, "_run_drive_source_catchup_safely", no_drive_changes),
        patch.object(daemon_cli, "_periodic_wal_checkpoint", wait_forever),
        patch.object(daemon_cli, "_periodic_fts_merge", wait_forever),
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
    ):
        asyncio.run(exercise())

    assert browser_server.shutdown_called is True
    assert browser_server.close_called is True
    assert api_server.shutdown_called is True
    assert api_server.close_called is True
    assert interrupted_cleanup_calls == 1


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

    lifecycle_tick_started = False

    async def lifecycle_heartbeat() -> None:
        nonlocal lifecycle_tick_started
        lifecycle_tick_started = True
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
        patch.object(daemon_cli, "_periodic_health_check", side_effect=fail_background_work),
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
    assert lifecycle_tick_started is True
