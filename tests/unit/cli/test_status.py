"""Tests for first-run status UX."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.status import (
    _render_assertion_candidate_queue,
    _render_raw_replay_backlog,
    _show_daemon_status,
    _show_status_json,
    _status_ok,
    status_command,
)
from polylogue.cli.operation_kernel import OperationKernelError, OperationResult
from polylogue.cli.shared.types import AppEnv
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.conftest import _MANAGED_VERIFY_ENV
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.frozen_clock import FrozenClock


class _ArchiveStats(TypedDict):
    total_sessions: int
    total_messages: int


class _TierStatus(TypedDict):
    table_counts: dict[str, int]


class _ArchiveReadinessSurface(TypedDict):
    ready: bool
    blockers: list[str]


class _ArchiveReadiness(TypedDict):
    reason: str
    checked: bool
    surfaces: dict[str, _ArchiveReadinessSurface]


class _RawFrontierIntegrity(TypedDict):
    overall_status: str


class _ConvergedClaim(TypedDict):
    value: bool


class _ClaimGuard(TypedDict):
    converged: _ConvergedClaim


class _DirectStatusPayload(TypedDict):
    ok: bool
    archive_stats: _ArchiveStats
    archive_tiers: dict[str, _TierStatus]
    archive_readiness: _ArchiveReadiness
    raw_frontier_integrity: _RawFrontierIntegrity
    claim_guard: _ClaimGuard


class _CapturingConsole:
    """Console mock that captures print calls in a list."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.calls.append(" ".join(str(a) for a in args))


def _make_app_env() -> AppEnv:
    """Create a minimal AppEnv for testing."""
    ui: Any = MagicMock()
    ui.plain = True
    ui.console = _CapturingConsole()
    return AppEnv(ui=ui)


def _combined_calls(env: AppEnv) -> str:
    """Get combined output from the capturing console."""
    console: Any = env.ui.console
    return " ".join(console.calls)


def _healthy_raw_frontier_integrity() -> dict[str, Any]:
    return {
        "available": True,
        "overall_status": "healthy",
        "broken_head_status": "healthy",
        "broken_head_count": 0,
        "broken_head_checked_count": 1,
        "broken_head_samples": [],
        "broken_head_reason": "",
        "missing_source_raw_status": "healthy",
        "missing_source_raw_count": 0,
        "missing_source_raw_samples": [],
        "missing_source_raw_reason": "",
        "cursor_ahead_status": "healthy",
        "cursor_ahead_count": 0,
        "cursor_ahead_checked_count": 1,
        "cursor_head_comparison_count": 1,
        "cursor_ahead_comparison_count": 0,
        "cursor_ahead_samples": [],
        "cursor_authority_gap_count": 0,
        "cursor_authority_gap_samples": [],
        "cursor_ahead_reason": "",
    }


def _fresh_status_snapshot(frozen_clock: FrozenClock) -> dict[str, object]:
    return {
        "state": "fresh",
        "captured_at": frozen_clock.now().isoformat(),
        "age_s": 0.1,
    }


@pytest.mark.parametrize("state", ["scheduler-stalled", "parked-pending"])
def test_status_renderer_includes_red_queue_state_and_receipt_details(state: str) -> None:
    """Human status keeps receipt timing evidence visible with its verdict."""
    env = _make_app_env()

    _render_assertion_candidate_queue(
        env,
        {
            "state": state,
            "pending_count": 2,
            "judgment_scheduler_receipt_status": "completed",
            "judgment_scheduler_receipt_at_ms": 1_800_000_000_000,
            "judgment_scheduler_receipt_age_ms": 90_000,
            "judgment_scheduler_receipt_reason": "sweep_completed",
        },
    )

    output = _combined_calls(env)
    assert f"[red]{state}, 2 pending[/red]" in output
    assert "judgment scheduler receipt: completed" in output
    assert "at=2027-01-15T08:00:00+00:00" in output
    assert "age=90.0s" in output
    assert "reason=sweep_completed" in output


def test_status_renderer_scales_multi_day_receipt_age_to_days() -> None:
    env = _make_app_env()

    _render_assertion_candidate_queue(
        env,
        {
            "state": "scheduler-stalled",
            "pending_count": 2,
            "judgment_scheduler_receipt_status": "failed",
            "judgment_scheduler_receipt_age_ms": 2 * 24 * 60 * 60 * 1000,
            "judgment_scheduler_receipt_reason": "sweep_failed",
        },
    )

    output = _combined_calls(env)
    assert "age=2.0d" in output
    assert "age=172800.0s" not in output


def test_status_command_renders_canonical_daemon_operation_result() -> None:
    """The CLI consumes the typed canonical operation result for daemon authority."""
    env = _make_app_env()
    config = SimpleNamespace(archive_root=Path("/tmp/status-test"))
    result_value = {
        "ok": True,
        "daemon_liveness": True,
        "status_snapshot": {"state": "fresh", "captured_at": "2026-09-11T00:00:00+00:00", "age_s": 0.1},
    }
    operation_result = OperationResult("status", result_value, {"mode": "daemon"})

    with (
        patch("polylogue.cli.shared.helpers.load_effective_config", return_value=config),
        patch("polylogue.cli.operation_kernel.configured_read_operation", return_value=operation_result) as producer,
    ):
        result = CliRunner().invoke(
            status_command,
            ["--daemon-url", "http://127.0.0.1:8766", "--json"],
            obj=env,
        )

    assert result.exit_code == 1
    producer.assert_called_once_with(
        config,
        "status",
        {"include_archive_readiness": False},
        daemon_disabled=False,
    )
    payload = json.loads(_combined_calls(env))
    assert payload["source"] == "daemon"
    assert payload["daemon_liveness"] is True
    assert payload["status_snapshot"]["state"] == "fresh"


def test_status_command_reports_operation_unavailable_as_live_daemon_snapshot() -> None:
    """A canonical transport failure remains an unavailable live-daemon result."""
    env = _make_app_env()
    config = SimpleNamespace(archive_root=Path("/tmp/status-test"))
    with (
        patch("polylogue.cli.shared.helpers.load_effective_config", return_value=config),
        patch(
            "polylogue.cli.operation_kernel.configured_read_operation",
            side_effect=OperationKernelError("status unavailable"),
        ),
    ):
        result = CliRunner().invoke(
            status_command,
            ["--daemon-url", "http://127.0.0.1:8766", "--json"],
            obj=env,
        )

    assert result.exit_code == 1
    payload = json.loads(_combined_calls(env))
    assert payload["daemon_liveness"] is True
    assert payload["status_snapshot"]["state"] == "unavailable"


def test_status_command_does_not_misclassify_sqlite_failure_with_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing archive turns unrelated SQLite failures into unavailable status."""
    archive_root = bootstrap_archive_root(tmp_path / "archive")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    env = _make_app_env()
    config = SimpleNamespace(archive_root=archive_root)
    with (
        patch("polylogue.cli.shared.helpers.load_effective_config", return_value=config),
        patch(
            "polylogue.cli.operation_kernel.configured_read_operation",
            side_effect=sqlite3.OperationalError("unexpected status query failure"),
        ),
    ):
        result = CliRunner().invoke(
            status_command,
            ["--daemon-url", "http://127.0.0.1:8766", "--json"],
            obj=env,
        )

    assert result.exit_code == 1
    payload = json.loads(_combined_calls(env))
    assert "diagnostic" not in payload
    assert payload["daemon_liveness"] is True
    assert payload["status_snapshot"]["state"] == "unavailable"


def test_raw_replay_backlog_plain_status_explains_containment() -> None:
    env = _make_app_env()
    reason = "raw source-to-index replay is disabled pending per-session revision authority"

    _render_raw_replay_backlog(
        env,
        {
            "available": True,
            "execution_blocked": True,
            "execution_block_reason": reason,
            "candidate_count": 2,
            "missing_blob_count": 0,
            "total_blob_bytes": 12_000_000,
            "max_blob_bytes": 10_000_000,
            "oversized_count": 0,
            "origin_summary": [],
        },
    )

    output = _combined_calls(env)
    assert "Raw replay backlog:" in output
    assert reason in output


def test_daemon_status_uses_reported_fts_coverage_pct() -> None:
    env = _make_app_env()

    _show_daemon_status(
        env,
        {
            "daemon_liveness": True,
            "fts_readiness": {
                "messages_ready": False,
                "coverage_pct": 87.5,
            },
        },
    )

    assert "87.5% indexed" in _combined_calls(env)


def test_daemon_status_renders_queue_health_from_daemon_payload() -> None:
    env = _make_app_env()

    _show_daemon_status(
        env,
        {
            "daemon_liveness": True,
            "assertion_candidate_queue": {
                "state": "scheduler-stalled",
                "pending_count": 2,
                "judgment_scheduler_receipt_status": "failed",
                "judgment_scheduler_receipt_at_ms": 1_800_000_000_000,
                "judgment_scheduler_receipt_age_ms": 90_000,
                "judgment_scheduler_receipt_reason": "sweep_failed:RuntimeError",
            },
        },
    )

    output = _combined_calls(env)
    assert "[red]scheduler-stalled, 2 pending[/red]" in output
    assert "judgment scheduler receipt: failed" in output
    assert "age=90.0s" in output
    assert "reason=sweep_failed:RuntimeError" in output


def test_daemon_status_renders_failed_and_deferred_convergence_debt_separately() -> None:
    env = _make_app_env()

    _show_daemon_status(
        env,
        {
            "daemon_liveness": True,
            "convergence": {
                "failed_count": 1,
                "deferred_count": 2,
                "retry_due_count": 1,
            },
        },
    )

    assert "Convergence debt: 1 failed, 2 deferred, 1 retry due" in _combined_calls(env)


def test_daemon_status_treats_null_fts_coverage_as_unknown_progress() -> None:
    """A null coverage_pct must render as explicitly unknown, never a
    fabricated percentage (polylogue-roax): defaulting an unmeasured
    value to 0.0% (or, before this fix, 100.0% when ``messages_ready``
    was true) is exactly the "measuring the wrong thing" bug that let
    `ops status` and `find` disagree while pointing at the same archive.
    """
    env = _make_app_env()

    _show_daemon_status(
        env,
        {
            "daemon_liveness": True,
            "fts_readiness": {
                "messages_ready": False,
                "coverage_pct": None,
            },
        },
    )

    assert "FTS: [yellow]coverage unknown[/yellow]" in _combined_calls(env)


def test_daemon_status_treats_null_fts_coverage_as_unknown_even_when_ready() -> None:
    """The ready-but-unmeasured shape must not render as 100% either."""
    env = _make_app_env()

    _show_daemon_status(
        env,
        {
            "daemon_liveness": True,
            "fts_readiness": {
                "messages_ready": True,
                "coverage_pct": None,
            },
        },
    )

    assert "FTS: [green]coverage unknown[/green]" in _combined_calls(env)
    assert "100.0%" not in _combined_calls(env)


def test_daemon_status_archive_fts_reports_message_surface() -> None:
    env = _make_app_env()

    _show_daemon_status(
        env,
        {
            "daemon_liveness": True,
            "fts_readiness": {
                "indexed_surface": "messages_fts",
                "messages_ready": True,
                "coverage_pct": 100.0,
            },
        },
    )

    assert "FTS: [green]100.0% indexed[/green]" in _combined_calls(env)


@pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
def test_daemon_status_renders_raw_frontier_integrity(frozen_clock: FrozenClock) -> None:
    env = _make_app_env()
    unknown = _healthy_raw_frontier_integrity()
    unknown.update(
        {
            "available": False,
            "overall_status": "unknown",
            "broken_head_status": "unknown",
            "missing_source_raw_status": "unknown",
            "cursor_ahead_status": "unknown",
            "cursor_ahead_reason": "ops cursor authority unavailable",
        }
    )

    _show_daemon_status(
        env,
        {
            "daemon_liveness": True,
            "status_snapshot": _fresh_status_snapshot(frozen_clock),
            "raw_frontier_integrity": unknown,
        },
    )

    rendered = _combined_calls(env)
    assert "Raw frontier: [yellow]unknown[/yellow]" in rendered
    assert "ops cursor authority unavailable" in rendered


@pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
def test_compact_daemon_status_cannot_preserve_stale_green_frontier(frozen_clock: FrozenClock) -> None:
    env = _make_app_env()
    violated = _healthy_raw_frontier_integrity()
    violated.update(
        {
            "overall_status": "violated",
            "broken_head_status": "violated",
            "broken_head_count": 1,
            "broken_head_samples": [{"accepted_raw_id": "raw-1"}],
            "broken_head_reason": "1 active seed is broken",
        }
    )
    _show_status_json(
        env,
        {
            "ok": True,
            "daemon_liveness": True,
            "status_snapshot": _fresh_status_snapshot(frozen_clock),
            "raw_frontier_integrity": violated,
        },
    )

    payload = json.loads(_combined_calls(env))
    assert payload["ok"] is False
    assert payload["raw_frontier_integrity"]["overall_status"] == "violated"
    assert "broken_head_samples" not in payload["raw_frontier_integrity"]


@pytest.mark.parametrize("full", [False, True])
def test_daemon_status_json_missing_frontier_is_explicit_unknown(full: bool) -> None:
    env = _make_app_env()

    _show_status_json(env, {"ok": True, "daemon_liveness": True}, full=full)

    payload = json.loads(_combined_calls(env))
    assert payload["ok"] is False
    assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
    assert payload["component_readiness"]["raw_frontier_integrity"]["state"] == "unknown"


@pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
def test_status_ok_requires_complete_fresh_frontier_authority(frozen_clock: FrozenClock) -> None:
    assert _status_ok({"ok": True, "daemon_liveness": True}) is False
    healthy = _healthy_raw_frontier_integrity()
    assert _status_ok({"ok": True, "raw_frontier_integrity": healthy}) is False
    assert (
        _status_ok(
            {"ok": True, "raw_frontier_integrity": healthy},
            require_fresh_snapshot=True,
        )
        is False
    )
    assert (
        _status_ok(
            {
                "ok": True,
                "status_snapshot": _fresh_status_snapshot(frozen_clock),
                "raw_frontier_integrity": healthy,
                "raw_failure_lifecycle_available": True,
                "raw_failure_lifecycle_state": "healthy",
                "raw_parse_failures": 0,
                "raw_validation_failures": 0,
                "raw_unexplained_failures": 0,
            },
            require_fresh_snapshot=True,
        )
        is True
    )
    assert (
        _status_ok(
            {
                "ok": True,
                "daemon_liveness": True,
                "status_snapshot": {"state": "stale"},
                "raw_frontier_integrity": healthy,
                "raw_failure_lifecycle_available": True,
                "raw_failure_lifecycle_state": "healthy",
                "raw_parse_failures": 0,
                "raw_validation_failures": 0,
                "raw_unexplained_failures": 0,
            }
        )
        is False
    )


def test_daemon_status_text_marks_missing_frontier_non_green() -> None:
    env = _make_app_env()

    _show_daemon_status(env, {"ok": True, "daemon_liveness": True})

    rendered = _combined_calls(env)
    assert "[bold yellow]Daemon: running; status degraded[/bold yellow]" in rendered
    assert "Raw frontier: [yellow]unknown[/yellow]" in rendered


def test_daemon_status_stale_healthy_frontier_becomes_unknown() -> None:
    env = _make_app_env()
    _show_status_json(
        env,
        {
            "ok": True,
            "daemon_liveness": True,
            "status_snapshot": {"state": "stale"},
            "raw_frontier_integrity": _healthy_raw_frontier_integrity(),
        },
        full=True,
    )

    payload = json.loads(_combined_calls(env))
    assert payload["ok"] is False
    assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
    assert "omitted freshness provenance" in payload["raw_frontier_integrity"]["broken_head_reason"]


def test_daemon_status_json_is_compact_by_default() -> None:
    env = _make_app_env()
    full_payload = {
        "ok": True,
        "daemon_liveness": True,
        "checked_at": "2026-07-02T17:00:00+00:00",
        "component_readiness": {
            "search": {
                "component": "search",
                "state": "ready",
                "counts": {"messages_fts_count": 5},
            }
        },
        "live_ingest_attempts": {
            "running_count": 0,
            "recent": [
                {
                    "attempt_id": "attempt-1",
                    "status": "completed",
                    "stage": "idle",
                    "worker_completed_count": 4,
                    "worker_total_count": 4,
                    "large_debug_payload": "x" * 1000,
                }
            ],
        },
        "archive_debt": {
            "available": True,
            "totals": {"total": 1},
            "rows": [{"debt_ref": "debt-1", "summary": "large raw row"}],
        },
        "raw_materialization_readiness": {
            "total": 1,
            "sampled_rows": [{"raw_id": "raw-1"}],
        },
        "live_cursor": {"failing_files": ["large"]},
        "catchup": {"debug": True},
        "convergence": {"debug": True},
        "failing_files": ["large"],
        "last_ingestion_batch": {"debug": True},
    }

    _show_status_json(env, full_payload)

    payload = json.loads(_combined_calls(env))
    assert payload["source"] == "daemon"
    assert payload["daemon_liveness"] is True
    assert payload["component_readiness"]["search"]["state"] == "ready"
    assert payload["ingest"]["latest"]["attempt_id"] == "attempt-1"
    assert "large_debug_payload" not in payload["ingest"]["latest"]
    assert payload["archive_debt"]["row_count"] == 1
    assert "rows" not in payload["archive_debt"]
    assert "sampled_rows" not in payload["raw_materialization_readiness"]
    for heavy_key in ("live_cursor", "catchup", "convergence", "failing_files", "last_ingestion_batch"):
        assert heavy_key not in payload


def test_daemon_status_json_full_preserves_fields_but_normalizes_missing_authority() -> None:
    env = _make_app_env()
    full_payload = {
        "daemon_liveness": True,
        "live_cursor": {"tracked_file_count": 2},
        "archive_debt": {"rows": [{"debt_ref": "debt-1"}]},
        "assertion_candidate_queue": {"state": "scheduler-stalled", "pending_count": 2},
    }

    _show_status_json(env, full_payload, full=True)

    payload = json.loads(_combined_calls(env))
    assert payload["daemon_liveness"] is True
    assert payload["live_cursor"] == full_payload["live_cursor"]
    assert payload["archive_debt"] == full_payload["archive_debt"]
    assert payload["ok"] is False
    assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
    assert payload["component_readiness"]["raw_frontier_integrity"]["state"] == "unknown"


class TestCanonicalStatusOperation:
    """Status command tests pin the canonical operation-result producer."""

    @staticmethod
    def _direct_status(root: Path, *, include_archive_readiness: bool = False) -> _DirectStatusPayload:
        from polylogue.cli.operation_kernel import configured_read_operation
        from polylogue.config import Config

        config = Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")
        result = configured_read_operation(
            config,
            "status",
            {"include_archive_readiness": include_archive_readiness},
            daemon_disabled=True,
        )
        assert result.operation == "status"
        assert result.authority["mode"] == "direct"
        return cast(_DirectStatusPayload, result.value)

    def test_status_command_renders_canonical_direct_operation_result(self, tmp_path: Path) -> None:
        env = _make_app_env()
        config = SimpleNamespace(archive_root=tmp_path)
        result_value = {
            "ok": False,
            "daemon_liveness": False,
            "total_sessions": 2,
            "total_messages": 5,
            "raw_failure_lifecycle_available": False,
            "raw_failure_lifecycle_state": "unavailable",
            "raw_parse_failures": 0,
            "raw_validation_failures": 0,
            "raw_unexplained_failures": 0,
        }
        operation_result = OperationResult("status", result_value, {"mode": "direct"})

        with (
            patch("polylogue.cli.shared.helpers.load_effective_config", return_value=config),
            patch(
                "polylogue.cli.operation_kernel.configured_read_operation", return_value=operation_result
            ) as producer,
        ):
            result = CliRunner().invoke(
                status_command,
                ["--daemon-url", "http://127.0.0.1:8766", "--json"],
                obj=env,
            )

        assert result.exit_code == 1
        producer.assert_called_once_with(
            config,
            "status",
            {"include_archive_readiness": False},
            daemon_disabled=False,
        )
        payload = json.loads(_combined_calls(env))
        assert payload["source"] == "direct"
        assert payload["daemon_liveness"] is False
        assert payload["sessions"] == 2
        assert payload["messages"] == 5
        assert payload["ok"] is False

    def test_status_command_passes_exact_readiness_to_canonical_operation(self, tmp_path: Path) -> None:
        env = _make_app_env()
        config = SimpleNamespace(archive_root=tmp_path)
        result_value = {
            "ok": False,
            "daemon_liveness": False,
            "raw_failure_lifecycle_available": False,
            "raw_failure_lifecycle_state": "unavailable",
            "raw_parse_failures": 0,
            "raw_validation_failures": 0,
            "raw_unexplained_failures": 0,
        }
        operation_result = OperationResult("status", result_value, {"mode": "direct"})

        with (
            patch("polylogue.cli.shared.helpers.load_effective_config", return_value=config),
            patch(
                "polylogue.cli.operation_kernel.configured_read_operation", return_value=operation_result
            ) as producer,
        ):
            result = CliRunner().invoke(
                status_command,
                [
                    "--daemon-url",
                    "http://127.0.0.1:8766",
                    "--json",
                    "--full",
                    "--exact-archive-readiness",
                ],
                obj=env,
            )

        assert result.exit_code == 1
        producer.assert_called_once_with(
            config,
            "status",
            {"include_archive_readiness": True},
            daemon_disabled=False,
        )
        payload = json.loads(_combined_calls(env))
        assert payload["daemon_liveness"] is False
        assert payload["source"] == "direct"

    def test_status_command_compacts_canonical_daemon_operation_payload(self) -> None:
        """CLI JSON compaction applies after the canonical operation boundary."""
        env = _make_app_env()
        config = SimpleNamespace(archive_root=Path("/tmp/status-test"))
        operation_result = OperationResult(
            "status",
            {
                "ok": True,
                "daemon_liveness": True,
                "component_readiness": {"search": {"state": "ready"}},
                "live_ingest_attempts": {
                    "running_count": 0,
                    "recent": [{"attempt_id": "attempt-1", "large_debug_payload": "x" * 1000}],
                },
                "archive_debt": {"available": True, "totals": {"total": 1}, "rows": [{"debt_ref": "debt-1"}]},
                "raw_materialization_readiness": {"sampled_rows": [{"raw_id": "raw-1"}]},
                "live_cursor": {"failing_files": ["large"]},
            },
            {"mode": "daemon"},
        )

        with (
            patch("polylogue.cli.shared.helpers.load_effective_config", return_value=config),
            patch("polylogue.cli.operation_kernel.configured_read_operation", return_value=operation_result),
        ):
            result = CliRunner().invoke(
                status_command,
                ["--daemon-url", "http://127.0.0.1:8766", "--json"],
                obj=env,
            )

        assert result.exit_code == 1
        payload = json.loads(_combined_calls(env))
        assert payload["source"] == "daemon"
        assert payload["component_readiness"]["search"]["state"] == "ready"
        assert payload["ingest"]["latest"]["attempt_id"] == "attempt-1"
        assert "large_debug_payload" not in payload["ingest"]["latest"]
        assert payload["archive_debt"]["row_count"] == 1
        assert "rows" not in payload["archive_debt"]
        assert "sampled_rows" not in payload["raw_materialization_readiness"]
        assert "live_cursor" not in payload

    def test_direct_operation_preserves_archive_and_audit_workload(self, tmp_path: Path) -> None:
        """The real direct producer retains all pinned tier/workload sections."""
        bootstrap_archive_root(tmp_path)
        payload = self._direct_status(tmp_path)

        assert payload["archive_stats"]["total_sessions"] == 0
        assert payload["archive_stats"]["total_messages"] == 0
        assert payload["archive_tiers"]["audit"]["table_counts"] == {
            "operation_previews": 0,
            "operation_authorizations": 0,
            "operation_attempts": 0,
        }
        assert payload["archive_readiness"]["reason"] == "direct_status_default_skips_exact_archive_readiness"

    def test_direct_operation_fails_closed_for_missing_ops_frontier_authority(self, tmp_path: Path) -> None:
        """A missing ops cursor cannot produce a green direct claim."""
        bootstrap_archive_root(tmp_path)
        with sqlite3.connect(tmp_path / "ops.db") as conn:
            conn.execute("DROP TABLE ingest_cursor")
            conn.commit()

        payload = self._direct_status(tmp_path)
        assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
        assert payload["claim_guard"]["converged"]["value"] is False
        assert payload["ok"] is False

    def test_direct_operation_exact_readiness_blocks_missing_raw_evidence(self, tmp_path: Path) -> None:
        """Exact readiness preserves the raw-session evidence blocker."""
        bootstrap_archive_root(tmp_path)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, content_hash)
                VALUES ('native-1', 'codex-session', 'raw-missing', ?)
                """,
                (b"y" * 32,),
            )
            conn.commit()

        payload = self._direct_status(tmp_path, include_archive_readiness=True)
        readiness = payload["archive_readiness"]
        assert readiness["checked"] is True
        assert readiness["surfaces"]["raw_artifacts"]["ready"] is False
        assert readiness["surfaces"]["raw_artifacts"]["blockers"] == ["missing_source_raw_sessions"]


class TestStatusDiagnosticIntegration:
    """End-to-end coverage that the new diagnostics never leak tracebacks (#1263)."""

    def _xdg_env(self, tmp_path: Path) -> dict[str, str]:
        return {
            **os.environ,
            "POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "polylogue"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "HOME": str(tmp_path),
            "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:8766",
        }

    def _malformed_convergence_debt_archive(self, tmp_path: Path) -> Path:
        archive_root = tmp_path / "polylogue"
        for tier in (
            ArchiveTier.SOURCE,
            ArchiveTier.INDEX,
            ArchiveTier.EMBEDDINGS,
            ArchiveTier.USER,
            ArchiveTier.OPS,
        ):
            initialize_archive_database(archive_root / f"{tier.value}.db", tier)
        with sqlite3.connect(archive_root / "ops.db") as conn:
            conn.execute("DROP TABLE convergence_debt")
            conn.execute("CREATE TABLE convergence_debt (wrong_column TEXT)")
            conn.commit()
        return archive_root

    def _malformed_ingest_attempts_archive(self, tmp_path: Path) -> Path:
        archive_root = tmp_path / "polylogue"
        for tier in (
            ArchiveTier.SOURCE,
            ArchiveTier.INDEX,
            ArchiveTier.EMBEDDINGS,
            ArchiveTier.USER,
            ArchiveTier.AUDIT,
            ArchiveTier.OPS,
        ):
            initialize_archive_database(archive_root / f"{tier.value}.db", tier)
        with sqlite3.connect(archive_root / "ops.db") as conn:
            conn.execute("DROP TABLE ingest_attempts")
            conn.execute("CREATE TABLE ingest_attempts (wrong_column TEXT)")
            conn.commit()
        return archive_root

    def _malformed_archive_env(self, tmp_path: Path, archive_root: Path) -> dict[str, str]:
        env = self._xdg_env(tmp_path)
        env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
        return env

    @pytest.mark.integration
    def test_status_subprocess_malformed_convergence_debt_json_is_explicitly_unavailable(self, tmp_path: Path) -> None:
        """Malformed convergence debt must yield valid JSON and block convergence claims."""
        from tests.infra.cli_subprocess import run_cli

        archive_root = self._malformed_convergence_debt_archive(tmp_path)
        result = run_cli(
            ["--plain", "ops", "status", "--json", "--full"],
            env=self._malformed_archive_env(tmp_path, archive_root),
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["ingest_workload"]["available"] is False
        assert "convergence debt status unavailable" in payload["ingest_workload"]["reason"]
        assert payload["convergence"]["available"] is False
        assert "convergence debt status unavailable" in payload["convergence"]["error"]
        assert payload["claim_guard"]["converged"]["value"] is False

    @pytest.mark.integration
    def test_status_subprocess_malformed_convergence_debt_human_is_explicitly_unavailable(self, tmp_path: Path) -> None:
        """Malformed convergence debt must not collapse human status into generic unavailable output."""
        from tests.infra.cli_subprocess import run_cli

        archive_root = self._malformed_convergence_debt_archive(tmp_path)
        result = run_cli(
            ["--plain", "ops", "status"],
            env=self._malformed_archive_env(tmp_path, archive_root),
        )

        output_lower = result.output.lower()
        assert result.exit_code == 0, result.output
        assert "convergence debt: unavailable" in output_lower
        assert "convergence debt status unavailable" in output_lower
        assert "could not be queried" not in output_lower
        assert "traceback" not in output_lower

    @pytest.mark.integration
    def test_status_subprocess_malformed_ingest_attempts_json_keeps_convergence_healthy(self, tmp_path: Path) -> None:
        """A malformed workload table must not mislabel the independent debt ledger."""
        from tests.infra.cli_subprocess import run_cli

        archive_root = self._malformed_ingest_attempts_archive(tmp_path)
        result = run_cli(
            ["--plain", "ops", "status", "--json", "--full"],
            env=self._malformed_archive_env(tmp_path, archive_root),
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["ingest_workload"] == {
            "available": False,
            "reason": "ops workload status unavailable: no such column: phase",
        }
        assert payload["convergence"]["available"] is True
        assert payload["convergence"]["error"] is None
        assert payload["claim_guard"]["converged"]["reason"] == "ready"
        assert payload["claim_guard"]["perf_measurable"]["value"] is False

    @pytest.mark.integration
    def test_status_subprocess_malformed_ingest_attempts_human_keeps_convergence_healthy(self, tmp_path: Path) -> None:
        """Human status must keep the healthy ledger message for a workload-only failure."""
        from tests.infra.cli_subprocess import run_cli

        archive_root = self._malformed_ingest_attempts_archive(tmp_path)
        result = run_cli(
            ["--plain", "ops", "status"],
            env=self._malformed_archive_env(tmp_path, archive_root),
        )

        output_lower = result.output.lower()
        assert result.exit_code == 0, result.output
        assert "convergence debt:" in output_lower
        assert "convergence debt: unavailable" not in output_lower
        assert "convergence debt status unavailable" not in output_lower
        assert "ops workload status unavailable" in output_lower
        assert "could not be queried" not in output_lower
        assert "traceback" not in output_lower

    @pytest.mark.integration
    def test_status_subprocess_schema_mismatch(self, tmp_path: Path) -> None:
        """A db with the wrong PRAGMA user_version yields actionable text, no traceback."""
        import sqlite3

        from tests.infra.cli_subprocess import run_cli

        data_home = tmp_path / "data" / "polylogue"
        data_home.mkdir(parents=True, exist_ok=True)
        db = data_home / "index.db"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()

        result = run_cli(["--plain", "ops", "status"], env=self._xdg_env(tmp_path))
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower
        assert "daemon not running" in output_lower or "polylogued" in output_lower

    @pytest.mark.integration
    def test_status_subprocess_stale_pidfile(self, tmp_path: Path) -> None:
        """A stale pidfile yields actionable text, no traceback."""
        import sqlite3

        from tests.infra.cli_subprocess import run_cli

        data_home = tmp_path / "data" / "polylogue"
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "index.db").close()
        archive = tmp_path / "polylogue"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "daemon.pid").write_text("99999999\n")

        result = run_cli(["--plain", "ops", "status"], env=self._xdg_env(tmp_path))
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower
        assert "pidfile" in output_lower or "polylogued" in output_lower

    @pytest.mark.integration
    def test_status_subprocess_no_sources(self, tmp_path: Path) -> None:
        """An empty roots config surfaces the no-sources hint."""
        import sqlite3

        from tests.infra.cli_subprocess import run_cli

        data_home = tmp_path / "data" / "polylogue"
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "index.db").close()
        config_home = tmp_path / "config" / "polylogue"
        config_home.mkdir(parents=True, exist_ok=True)
        (config_home / "polylogue.toml").write_text("[sources]\nroots = []\n")

        result = run_cli(["--plain", "ops", "status"], env=self._xdg_env(tmp_path))
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower


class TestEnvIsolation:
    """Regression coverage for #1325: workspace_env strips host POLYLOGUE_* env vars."""

    def test_autouse_clears_host_polylogue_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Setting POLYLOGUE_* before the autouse runs must be wiped.

        ``monkeypatch`` here is a *different* instance than the autouse
        fixture, but pytest's fixture finalisation runs LIFO: the autouse
        runs first (clearing host env) and the per-test monkeypatch runs
        after. We simulate the "operator daemon already running" case by
        re-asserting that no POLYLOGUE_* leaked through.
        """
        # The autouse ``_clear_polylogue_env`` fixture has already run and
        # stripped every POLYLOGUE_* var the host had. Verify it.
        leaked = [
            k
            for k in os.environ
            if k.startswith("POLYLOGUE_")
            and k
            not in {
                "POLYLOGUE_SITE_CONFIG",
                "POLYLOGUE_DAEMON_URL",
            }
            # The managed verification supervisor's own run-identity plumbing is
            # not host configuration, and conftest deliberately exempts it from
            # the scrub: clearing it mid-run detaches this process from the
            # basetemp it owns and from the event ledger. What this case is
            # about is operator configuration leaking in, so it defers to the
            # same canonical set rather than keeping a second list that silently
            # disagrees whenever the supervisor gains a variable.
            and k not in _MANAGED_VERIFY_ENV
        ]
        assert leaked == [], f"host POLYLOGUE_* vars leaked into test env: {leaked}"
        # And the daemon URL must be routed to an unreachable address.
        assert os.environ["POLYLOGUE_DAEMON_URL"] == "http://127.0.0.1:1"
        # And site config lookup must be disabled.
        assert os.environ["POLYLOGUE_SITE_CONFIG"] == ""

    def test_workspace_env_overrides_polluted_host(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``workspace_env`` must win over a contaminated host environment.

        Set the offending vars in this test body to simulate a polluted
        host that escaped the autouse clear (e.g. a vendoring agent), then
        re-invoke the relevant resolver paths.
        """
        # Simulate an operator who has POLYLOGUE_ARCHIVE_ROOT set in their
        # shell pointing at the production archive.
        production_archive = Path("/var/lib/polylogue/archive")
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(production_archive))

        # workspace_env declared its own archive root via fixture wiring;
        # _clear_polylogue_env ran before workspace_env so the production
        # value above was set AFTER workspace_env. The fixture's contract
        # is that *its* value is the one tests should see at fixture-setup
        # time. We assert the documented value is the tmp_path one.
        assert workspace_env["archive_root"] == tmp_path / "archive"
        # The CLI default daemon URL resolution must not point at the host
        # ``polylogued`` listening on 8766.
        from polylogue.cli.commands.status import _default_daemon_url

        assert _default_daemon_url() == "http://127.0.0.1:1"


class TestDaemonStatus:
    """Daemon status rendering tests."""

    def test_empty_archive_with_sources(self) -> None:
        """When daemon runs but archive is empty, source discovery is shown."""
        env = _make_app_env()

        status_payload: dict[str, object] = {
            "daemon_liveness": True,
            "component_state": {
                "watcher": {"state": "running", "description": "watching 2 sources"},
            },
            "insight_freshness": {"total_sessions": 0, "sessions_with_profiles": 0},
            "live": {
                "sources": [
                    {"name": "claude-code", "root": "/tmp/claude", "exists": True},
                    {"name": "codex", "root": "/tmp/codex", "exists": False},
                ]
            },
            "watcher_roots": ["/tmp/claude", "/tmp/codex"],
            "live_ingest_attempts": {},
            "fts_readiness": {},
            "db_size_bytes": 0,
            "checked_at": "",
        }
        _show_daemon_status(env, status_payload)
        combined = _combined_calls(env)
        # Daemon-status output should report watching sources.
        assert "watching" in combined.lower(), f"expected 'watching' in status output, got: {combined[:200]}"

    def test_running_daemon_with_data(self) -> None:
        """When daemon runs with data, normal status is shown without first-run hints."""
        env = _make_app_env()

        status_payload: dict[str, object] = {
            "daemon_liveness": True,
            "component_state": {
                "watcher": {"state": "running", "description": "watching 2 sources"},
            },
            "insight_freshness": {"total_sessions": 42, "sessions_with_profiles": 40},
            "live": {
                "sources": [
                    {"name": "claude-code", "root": "/tmp/claude", "exists": True},
                ]
            },
            "live_ingest_attempts": {
                "completed_count": 10,
                "total_count": 10,
            },
            "fts_readiness": {"coverage_pct": 98.5},
            "db_size_bytes": 1_048_576,
            "disk_free_bytes": 107_374_182_400,
            "checked_at": "2026-05-07T12:00:00",
        }
        _show_daemon_status(env, status_payload)
        combined = _combined_calls(env)
        assert "no sessions" not in combined.lower()
        assert "running" in combined.lower()


@pytest.mark.integration
def test_status_command_accepts_json_alias_flag(tmp_path: Path) -> None:
    """`polylogue ops status --json` is a documented alias for `--format json`.

    Sibling subcommands (`list`, `tags`, `sources`, `stats`) accept `--json`;
    `status` previously rejected it with "No such option '--json'". Closes #1612.
    """
    from tests.infra.cli_subprocess import run_cli

    env = {
        **os.environ,
        "POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "polylogue"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "HOME": str(tmp_path),
        "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:8766",
    }
    result = run_cli(["--plain", "ops", "status", "--json"], env=env)
    assert result.exit_code == 0, result.output
    assert "No such option" not in result.output
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
