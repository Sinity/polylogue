"""A status component that was never measured must not read as a positive claim.

Every test here names the mutation that makes it red: restoring the optimistic
default, the literal, or the hardcoded component state that
``polylogue-bu47u`` catalogued.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.daemon import status as status_module
from polylogue.daemon.health import DaemonHealth, HealthTier
from polylogue.daemon.status import build_daemon_status
from polylogue.operations.status_protocol import StatusComponentRegistry
from polylogue.readiness.capability import CapabilityReadinessState, ComponentReadiness


def _patch_healthy_collectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    raw_materialization: Any = None,
    live_ingest: Any = None,
    daemon_alive: bool = False,
) -> None:
    """Pin every status collector to a measured, healthy value.

    Each test then replaces exactly one of them, so a changed claim can only
    come from the component under test.
    """

    storage = status_module.ArchiveStorageStatus(
        active_store="archive_file_set",
        active_db_path=str(tmp_path / "index.db"),
        archive_root=str(tmp_path),
        configured_archive_root=str(tmp_path),
        archive_ready=True,
        archive_materialization_ready=True,
        final_shape_ready=True,
        archive_schema_ready=True,
        present_tiers=["source", "index", "embeddings", "user", "audit", "ops"],
    )
    healthy_raw = status_module.RawMaterializationReadiness(
        available=True,
        raw_authority_parser_census={"available": True},
    )
    frontier = status_module.RawFrontierIntegrity(available=True, overall_status="healthy")

    monkeypatch.setattr(status_module, "_db_size_info", lambda: {})
    monkeypatch.setattr(status_module, "_blob_size_info", lambda: 0)
    monkeypatch.setattr(status_module, "_archive_storage_info", lambda: storage)
    monkeypatch.setattr(
        status_module,
        "_fts_readiness_info",
        lambda: {"messages_ready": True, "invariant_ready": True, "coverage_exact": True},
    )
    monkeypatch.setattr(
        status_module,
        "_insight_freshness_info",
        lambda: {"sessions_with_profiles": 0, "total_sessions": 0, "profile_ready": True},
    )
    monkeypatch.setattr(
        status_module,
        "_session_summary_readiness_info",
        lambda: ComponentReadiness(
            component="session_summary",
            state=CapabilityReadinessState.READY,
            summary="ready",
        ),
    )
    monkeypatch.setattr(
        status_module,
        "_raw_materialization_readiness_info",
        raw_materialization if raw_materialization is not None else (lambda **_: healthy_raw),
    )
    monkeypatch.setattr(status_module, "_raw_replay_backlog_info", lambda **_: {})
    monkeypatch.setattr(status_module, "_live_cursor_summary_info", lambda: status_module.LiveCursorSummary())
    monkeypatch.setattr(
        status_module,
        "_live_ingest_attempt_summary_info",
        live_ingest if live_ingest is not None else (lambda: status_module.LiveIngestAttemptSummary()),
    )
    monkeypatch.setattr(
        status_module, "convergence_debt_summary_info", lambda *_a, **_k: status_module.ConvergenceDebtSummary()
    )
    monkeypatch.setattr(status_module, "cursor_lag_summary_info", lambda **_: status_module.CursorLagSummary())
    monkeypatch.setattr(status_module, "_raw_failure_info", lambda: {})
    monkeypatch.setattr(
        status_module,
        "_blob_publication_reservation_info",
        lambda: status_module.BlobPublicationReservationStatus(),
    )
    monkeypatch.setattr(status_module, "embedding_readiness_info", lambda *_: {})
    monkeypatch.setattr(status_module, "_active_status_db_path", lambda: tmp_path / "index.db")
    monkeypatch.setattr(status_module, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(status_module, "_raw_frontier_integrity_info", lambda _: frontier)
    monkeypatch.setattr(status_module, "catchup_status_info", lambda *_a, **_k: status_module.CatchupStatus())
    monkeypatch.setattr(status_module, "_check_daemon_liveness", lambda *_: daemon_alive)
    monkeypatch.setattr(status_module, "check_health", lambda **_: DaemonHealth())


def _build(**kwargs: Any) -> Any:
    specs = status_module._daemon_status_component_specs(
        checked_health=lambda _tiers: DaemonHealth(),
        health_tiers=lambda: {HealthTier.FAST},
        include_raw_replay_backlog=False,
        include_exact_raw_materialization_readiness=False,
    )
    return build_daemon_status(
        sources=(),
        browser_capture_enabled=False,
        include_raw_replay_backlog=False,
        include_exact_raw_materialization_readiness=False,
        registry=StatusComponentRegistry(specs),
        **kwargs,
    )


def test_timed_out_raw_materialization_collector_reports_unmeasured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A collector that blew its deadline must not fall back to the all-zero model.

    Anti-vacuity: restore ``_v("raw_materialization", RawMaterializationReadiness())``
    -- whose ``available`` defaults to True with every count zero -- and the
    payload reports an inspected, converged archive again, so this fails.
    """

    def slow_collector(**_kwargs: object) -> status_module.RawMaterializationReadiness:
        time.sleep(5.0)
        raise AssertionError("collector must not finish within the component deadline")

    _patch_healthy_collectors(monkeypatch, tmp_path, raw_materialization=slow_collector)

    status = _build()

    assert status.raw_materialization_readiness.available is False
    claim_guard = cast(dict[str, dict[str, object]], status.claim_guard)
    # Withheld, not refuted: nothing measured the domain (polylogue-kjy0a).
    assert claim_guard["converged"]["value"] is None
    assert "raw_materialization" in str(claim_guard["converged"]["reason"]) or "raw-materialization" in str(
        claim_guard["converged"]["reason"]
    )


def test_unreadable_ingest_ledger_cannot_certify_performance_is_measurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unreadable attempt table must not prove no writer is running.

    Anti-vacuity: restore ``active_writer = bool(summary.running_count)`` and a
    ledger that could not be read yields ``perf_measurable: true`` again.
    """

    unreadable = status_module.LiveIngestAttemptSummary(
        available=False,
        unavailable_reason="live_ingest_attempt table unreadable: disk I/O error",
    )
    _patch_healthy_collectors(monkeypatch, tmp_path, live_ingest=lambda: unreadable)

    status = _build()

    claim_guard = cast(dict[str, dict[str, object]], status.claim_guard)
    # Withheld, not refuted: the ledger was never read (polylogue-g88v4).
    assert claim_guard["perf_measurable"]["value"] is None
    assert claim_guard["perf_measurable"]["determinate"] is False
    readiness = cast(dict[str, dict[str, object]], status.component_readiness)
    assert readiness["daemon_ingest"]["state"] == "unknown"


def test_measured_idle_ingest_ledger_still_certifies_performance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The honest branch must not swallow the measured case as well.

    Anti-vacuity: make ``available`` irrelevant (always treat the ledger as
    unavailable) and this fails, proving the previous test is not passing for a
    blanket reason.
    """

    _patch_healthy_collectors(
        monkeypatch, tmp_path, live_ingest=lambda: status_module.LiveIngestAttemptSummary(available=True)
    )

    status = _build()

    claim_guard = cast(dict[str, dict[str, object]], status.claim_guard)
    assert claim_guard["perf_measurable"]["value"] is True


def test_fts_read_error_is_unknown_not_an_empty_source_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The production status builder must not turn a read error into zero.

    Anti-vacuity: restoring FTSReadiness' numeric zero defaults makes this
    route report an empty source relation and changes the component from
    unknown to missing.
    """

    _patch_healthy_collectors(monkeypatch, tmp_path)
    (tmp_path / "index.db").write_bytes(b"sqlite source placeholder")
    from polylogue.daemon.fts_status import fts_readiness_info

    def unreadable_source(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("source table unreadable")

    # ``_build`` reaches _fts_readiness_info -> fts_readiness_info through the
    # normal component registry; only the source connection is faulted.
    monkeypatch.setattr("polylogue.daemon.fts_status.open_readonly_connection", unreadable_source)
    monkeypatch.setattr(
        status_module,
        "_fts_readiness_info",
        lambda: fts_readiness_info(status_module._active_status_db_path()),
    )

    status = _build()

    assert status.fts_readiness.message_indexed_count is None
    assert status.fts_readiness.message_indexable_count is None
    assert status.fts_readiness.coverage_pct is None
    readiness = cast(dict[str, dict[str, object]], status.component_readiness)
    assert readiness["search"]["state"] == "unknown"


@pytest.mark.parametrize("daemon_alive", [True, False])
def test_api_component_state_tracks_observed_daemon_liveness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, daemon_alive: bool
) -> None:
    """``daemon_liveness: false`` and ``api: running`` can never share a payload.

    Anti-vacuity: restore ``api="running"`` in ``ComponentState(...)`` and the
    ``daemon_alive=False`` case emits both claims at once, so this fails.
    """

    _patch_healthy_collectors(monkeypatch, tmp_path, daemon_alive=daemon_alive)

    status = _build()

    assert status.daemon_liveness is daemon_alive
    assert status.component_state.api == ("running" if daemon_alive else "stopped")
