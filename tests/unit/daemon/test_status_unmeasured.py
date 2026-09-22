"""A status component that was never measured must not read as a positive claim.

Every test here names the mutation that makes it red: restoring the optimistic
default, the literal, or the hardcoded component state that
``polylogue-bu47u`` catalogued.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.json import JSONValue
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


def test_health_fallback_probe_is_not_run_when_the_component_answered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``check_health`` is a status component, not also a per-call fallback.

    ``build_daemon_status`` read health through
    ``_v("health", _checked_health(health_tiers))``. Python evaluates that
    argument before ``_v`` can decide whether it is needed, so the probe the
    registry had just run as a component was re-run on every single call --
    a fresh snapshot bought nothing (polylogue-20d.17 AC10).

    Anti-vacuity: pass the fallback positionally again and a third call
    appears, carrying the whole configured tier set at once instead of the
    per-tier split the two components use.
    """
    configured = {HealthTier.FAST, HealthTier.MEDIUM}
    monkeypatch.setattr(status_module, "_configured_health_tiers", lambda **_: set(configured))

    _patch_healthy_collectors(monkeypatch, tmp_path)

    calls: list[frozenset[HealthTier]] = []

    def counting_check_health(**kwargs: Any) -> DaemonHealth:
        tiers = kwargs.get("tiers") or ()
        calls.append(frozenset(cast(set[HealthTier], tiers)))
        return DaemonHealth()

    monkeypatch.setattr(status_module, "check_health", counting_check_health)

    status = build_daemon_status(
        sources=(),
        browser_capture_enabled=False,
        include_raw_replay_backlog=False,
        include_exact_raw_materialization_readiness=False,
    )

    assert isinstance(status.health, DaemonHealth)
    assert sorted(calls, key=lambda tiers: sorted(tier.value for tier in tiers)) == [
        frozenset({HealthTier.FAST}),
        frozenset({HealthTier.MEDIUM}),
    ], calls


# ---------------------------------------------------------------------------
# The ``unmeasured=`` opt-in gap, closed and proven (polylogue-20d.17.4)
# ---------------------------------------------------------------------------


def _specs_with_per_component_fingerprints(
    fingerprints: dict[str, str],
) -> list[Any]:
    """Declared status specs, each keyed on its own mutable fingerprint.

    The production specs share one fingerprint callable, so bumping it would
    stale every component at once. Rebinding per component lets exactly one
    collection go stale while its siblings stay fresh -- which is the shape
    a real archive produces when one source changes under one collector.
    """
    import dataclasses

    specs = status_module._daemon_status_component_specs(
        checked_health=lambda _tiers: DaemonHealth(),
        health_tiers=lambda: {HealthTier.FAST},
        include_raw_replay_backlog=False,
        include_exact_raw_materialization_readiness=False,
    )

    def _fingerprint_for(name: str) -> Callable[[], str]:
        return lambda: fingerprints[name]

    rebound = []
    for spec in specs:
        fingerprints.setdefault(spec.name, "gen-1")
        rebound.append(dataclasses.replace(spec, fingerprint=_fingerprint_for(spec.name)))
    return rebound


def _stale_last_good_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    component: str,
    good_collector: Any,
    attribute: str,
) -> tuple[Any, Any]:
    """Return (fresh_status, stale_status) for one component.

    The first build collects a real value. The component's fingerprint then
    changes and its collector stops answering, so the registry reports the
    component ``stale`` while still holding that first value as last-good --
    exactly the state in which ``_v`` used to return it as a current reading.
    """
    import time

    fingerprints: dict[str, str] = {}
    answering = {"value": True}

    def collector(*args: object, **kwargs: object) -> object:
        if not answering["value"]:
            time.sleep(5.0)
            raise AssertionError("collector must not finish within the component deadline")
        return good_collector(*args, **kwargs)

    _patch_healthy_collectors(monkeypatch, tmp_path)
    monkeypatch.setattr(status_module, attribute, collector)
    registry = StatusComponentRegistry(_specs_with_per_component_fingerprints(fingerprints))

    fresh = build_daemon_status(
        sources=(),
        browser_capture_enabled=False,
        include_raw_replay_backlog=False,
        include_exact_raw_materialization_readiness=False,
        registry=registry,
    )
    answering["value"] = False
    fingerprints[component] = "gen-2"
    stale = build_daemon_status(
        sources=(),
        browser_capture_enabled=False,
        include_raw_replay_backlog=False,
        include_exact_raw_materialization_readiness=False,
        registry=registry,
    )
    return fresh, stale


def test_stale_db_size_is_unmeasured_not_the_previous_reading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale ``db_size`` collection publishes null sizes, not last week's.

    Anti-vacuity: drop ``unmeasured={}`` from the ``_v("db_size", ...)`` read
    and the stale build republishes 4242/11/99 as a current measurement.
    """
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="db_size",
        good_collector=lambda: {"db_size_bytes": 4242, "wal_size_bytes": 11, "disk_free_bytes": 99},
        attribute="_db_size_info",
    )

    assert (fresh.db_size_bytes, fresh.wal_size_bytes, fresh.disk_free_bytes) == (4242, 11, 99)
    assert (stale.db_size_bytes, stale.wal_size_bytes, stale.disk_free_bytes) == (None, None, None)


def test_a_measured_empty_archive_still_publishes_its_zeros(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A true zero survives. Without this the test above passes vacuously.

    Anti-vacuity: make ``_v`` return the unmeasured substitute unconditionally
    and a genuinely empty archive stops reporting its measured zeros.
    """
    fresh, _stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="db_size",
        good_collector=lambda: {"db_size_bytes": 0, "wal_size_bytes": 0, "disk_free_bytes": 0},
        attribute="_db_size_info",
    )

    assert (fresh.db_size_bytes, fresh.wal_size_bytes, fresh.disk_free_bytes) == (0, 0, 0)


def test_stale_fts_readiness_is_unmeasured_not_the_previous_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale FTS evidence never republishes a previous 100% as current.

    Anti-vacuity: drop ``unmeasured={}`` from ``_v("fts_readiness", ...)`` and
    the stale build reports ``messages_ready`` with a measured 7/7 coverage.
    """
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="fts_readiness",
        good_collector=lambda: {
            "messages_ready": True,
            "invariant_ready": True,
            "message_indexed_count": 7,
            "message_indexable_count": 7,
            "coverage_pct": 100.0,
            "coverage_exact": True,
        },
        attribute="_fts_readiness_info",
    )

    assert fresh.fts_readiness.message_indexed_count == 7
    assert fresh.fts_readiness.coverage_pct == 100.0
    assert stale.fts_readiness.messages_ready is False
    assert stale.fts_readiness.message_indexed_count is None
    assert stale.fts_readiness.message_indexable_count is None
    assert stale.fts_readiness.coverage_pct is None
    assert stale.fts_readiness.coverage_exact is False
    readiness = cast(dict[str, dict[str, object]], stale.component_readiness)
    assert readiness["search"]["state"] == "unknown"


def test_stale_insight_freshness_is_unmeasured_not_the_previous_profile_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale profile evidence withholds readiness instead of certifying it.

    Anti-vacuity: drop ``unmeasured={}`` from ``_v("insight_freshness", ...)``
    and the stale build still reports ``profile_ready: true`` over 5/5.
    """
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="insight_freshness",
        good_collector=lambda: {"sessions_with_profiles": 5, "total_sessions": 5, "profile_ready": True},
        attribute="_insight_freshness_info",
    )

    assert fresh.insight_freshness.profile_ready is True
    assert fresh.insight_freshness.total_sessions == 5
    assert stale.insight_freshness.profile_ready is None
    assert stale.insight_freshness.sessions_with_profiles is None
    assert stale.insight_freshness.total_sessions is None


def test_stale_archive_storage_never_certifies_a_root_match_it_did_not_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale storage probe claims neither readiness nor a matching root.

    ``ArchiveStorageStatus.archive_root_matches_configured`` defaulted to
    ``True``, so an uncollected storage probe asserted the active root matched
    the configured one on behalf of a collection that resolved neither.

    Anti-vacuity: restore that ``True`` default (or drop ``unmeasured=`` from
    the ``_v("archive_storage", ...)`` read) and the stale build republishes
    ``archive_ready`` over the previous tier inventory.
    """
    ready = status_module.ArchiveStorageStatus(
        active_store="archive_file_set",
        archive_root=str(tmp_path),
        configured_archive_root=str(tmp_path),
        archive_root_matches_configured=True,
        archive_ready=True,
        archive_materialization_ready=True,
        final_shape_ready=True,
        archive_schema_ready=True,
        present_tiers=["source", "index"],
    )
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="archive_storage",
        good_collector=lambda: ready,
        attribute="_archive_storage_info",
    )

    assert fresh.archive_storage.archive_ready is True
    assert fresh.archive_storage.archive_root_matches_configured is True
    assert stale.archive_storage.archive_ready is False
    assert stale.archive_storage.archive_root_matches_configured is None
    assert stale.archive_storage.present_tiers == []


def test_stale_blob_reservation_ledger_reports_null_counts_not_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An uninspected receipt ledger has no counts; zero would be a measurement.

    Anti-vacuity: restore the ``int = 0`` model defaults (or drop
    ``unmeasured=``) and an uninspected ledger reports a clean, empty one.
    """
    measured = status_module.BlobPublicationReservationStatus(
        total_reserved_count=3,
        retained_referenced_count=2,
        retained_missing_count=0,
        unresolved_count=1,
    )
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="blob_publication_reservations",
        good_collector=lambda: measured,
        attribute="_blob_publication_reservation_info",
    )

    assert fresh.blob_publication_reservations.unresolved_count == 1
    # A measured zero is preserved on the fresh side ...
    assert fresh.blob_publication_reservations.retained_missing_count == 0
    # ... and an unmeasured one is null, not zero.
    assert stale.blob_publication_reservations.unresolved_count is None
    assert stale.blob_publication_reservations.total_reserved_count is None


def test_stale_raw_failure_counts_are_null_not_a_clean_source_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale raw-failure evidence reports unavailable, not zero failures.

    Anti-vacuity: drop ``unmeasured={}`` from ``_v("raw_failures", ...)`` and
    the stale build republishes the previous failure counts and a ``healthy``
    lifecycle for evidence it did not read.
    """
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="raw_failures",
        good_collector=lambda: {
            "parse_failures": 2,
            "validation_failures": 0,
            "raw_failure_lifecycle_available": True,
            "raw_failure_lifecycle_state": "healthy",
        },
        attribute="_raw_failure_info",
    )

    assert fresh.raw_parse_failures == 2
    assert fresh.raw_validation_failures == 0
    assert fresh.raw_failure_lifecycle_available is True
    assert stale.raw_parse_failures is None
    assert stale.raw_failure_lifecycle_available is False
    assert stale.raw_failure_lifecycle_state == "unavailable"


def test_stale_live_cursor_reports_an_unreadable_ledger_not_a_clean_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale cursor evidence cannot report "0 failed, 0 excluded".

    Anti-vacuity: drop ``unmeasured=`` from ``_v("live_cursor", ...)`` and the
    stale build republishes the previous failing-file counts as current.
    """
    measured = status_module.LiveCursorSummary(available=True, tracked_file_count=9, failed_file_count=3)
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="live_cursor",
        good_collector=lambda: measured,
        attribute="_live_cursor_summary_info",
    )

    assert fresh.live_cursor.failed_file_count == 3
    assert stale.live_cursor.available is False
    assert stale.live_cursor.unavailable_reason is not None


def test_every_status_component_read_declares_its_unmeasured_substitute() -> None:
    """No ``_v`` call site may fall back to a bare default.

    ``unmeasured=`` is opt-in per call site, so the gap AC3 closed can reopen
    silently the moment someone adds a component read without it. This reads
    ``build_daemon_status``' own source and names every exemption
    (polylogue-20d.17.4 T4-2).

    Anti-vacuity: delete ``unmeasured=`` from any guarded ``_v`` call and this
    lists that component; add a new bare one and it lists that instead.
    """
    import ast
    import inspect
    import textwrap

    #: Component -> why it is allowed to read without ``unmeasured=``.
    exemptions = {
        "health": (
            "escalated explicitly against its own snapshot state immediately "
            "after the read, so a stale OK becomes a named alert rather than "
            "silently disappearing"
        ),
    }

    tree = ast.parse(textwrap.dedent(inspect.getsource(build_daemon_status)))
    offenders: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "_v":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            offenders.append("a _v call whose component name is not a literal")
            continue
        name = str(node.args[0].value)
        seen.add(name)
        if any(keyword.arg == "unmeasured" for keyword in node.keywords):
            continue
        if name in exemptions:
            continue
        offenders.append(name)

    assert not offenders, (
        "status component reads without an explicit unmeasured= substitute: "
        f"{sorted(offenders)}. Either pass one, or add the component to this "
        "test's exemption map with the reason its own guard is stronger."
    )
    # The four the bead named plus every other site measured at this head.
    assert {"db_size", "archive_storage", "fts_readiness", "insight_freshness"} <= seen
    assert seen >= set(exemptions)


def test_stale_cursor_lag_reports_an_unreadable_ledger_not_a_quiet_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale cursor-lag evidence cannot report "no lag".

    ``_v(..., unmeasured=...)`` already stopped the previous reading being
    served as current, but its substitute was a bare ``CursorLagSummary()``,
    which renders exactly like a caught-up archive. The substitute now carries
    the availability pair, so the stale build is distinguishable.

    Anti-vacuity: restore ``_UNMEASURED_CURSOR_LAG = CursorLagSummary()`` and
    the stale build's ``available`` is True with the same zeros a measured
    quiet archive publishes.
    """
    from polylogue.daemon.cursor_lag_status import CursorLagFamilySummary

    measured = status_module.CursorLagSummary(
        tracked_file_count=4,
        stuck_file_count=2,
        max_lag_s=900.0,
        family_summaries=[
            CursorLagFamilySummary(
                family="claude-code-session", tracked_file_count=4, stuck_file_count=2, max_lag_s=900.0
            )
        ],
    )
    fresh, stale = _stale_last_good_status(
        monkeypatch,
        tmp_path,
        component="cursor_lag",
        good_collector=lambda *_a, **_k: measured,
        attribute="cursor_lag_summary_info",
    )

    assert fresh.cursor_lag.available is True
    assert fresh.cursor_lag.stuck_file_count == 2
    assert stale.cursor_lag.available is False
    assert stale.cursor_lag.unavailable_reason is not None
    assert stale.cursor_lag.stuck_file_count == 0


def test_plaintext_status_names_an_unmeasured_cursor_ledger() -> None:
    """The ops-status text block must not stay silent for an unreadable ledger.

    It only printed when a count was positive, so an unavailable projection
    rendered identically to a quiet archive: no line at all.

    Anti-vacuity: delete the ``available is False`` branch in
    ``format_daemon_status_lines`` and the unmeasured payload produces no
    cursor-lag line, matching the measured-quiet payload exactly.
    """
    unreadable: dict[str, JSONValue] = {
        "available": False,
        "unavailable_reason": "cursor ledger unreadable: DatabaseError: malformed",
        "stuck_file_count": 0,
        "degraded_file_count": 0,
    }
    quiet: dict[str, JSONValue] = {"available": True, "stuck_file_count": 0, "degraded_file_count": 0}

    unreadable_lines = status_module.format_daemon_status_lines({"cursor_lag": unreadable})
    quiet_lines = status_module.format_daemon_status_lines({"cursor_lag": quiet})

    assert any("Cursor lag: UNMEASURED" in line for line in unreadable_lines)
    assert any("cursor ledger unreadable" in line for line in unreadable_lines)
    assert not any("Cursor lag" in line for line in quiet_lines)
