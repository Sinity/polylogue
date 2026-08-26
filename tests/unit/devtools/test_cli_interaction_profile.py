"""Anti-vacuity tests for the named CLI benchmark profile."""

from __future__ import annotations

from typing import cast

from tests.benchmarks.cli_profile import (
    INTERACTION_WORKLOADS,
    PROFILE_METRICS,
    RESPONSE_SIZES,
    TERMINAL_COLUMNS,
    profile_manifest,
)


def test_cli_profile_declares_all_interactive_and_background_workloads() -> None:
    names = {workload.name for workload in INTERACTION_WORKLOADS}
    assert names == {
        "cold-status",
        "warm-status",
        "find-read",
        "static-completion",
        "live-completion",
        "fuzzy-launch",
        "pagination",
        "cancellation",
        "concurrent-reads",
        "incremental-ingest",
        "derivation-catch-up",
        "inactive-candidate",
    }
    assert any(workload.background for workload in INTERACTION_WORKLOADS)


def test_cli_profile_has_reproducible_denominators_and_required_measurements() -> None:
    manifest = profile_manifest()
    assert manifest["terminal_columns"] == list(TERMINAL_COLUMNS)
    assert manifest["response_rows"] == list(RESPONSE_SIZES)
    metrics = manifest["metrics"]
    assert isinstance(metrics, list)
    assert set(PROFILE_METRICS).issubset({str(metric) for metric in metrics})
    denominators = manifest["denominators"]
    assert isinstance(denominators, dict)
    assert "background_throughput" in denominators


def test_profile_manifest_is_not_a_helper_only_benchmark() -> None:
    workloads = profile_manifest()["workloads"]
    assert isinstance(workloads, list)
    routes = {str(cast(dict[str, object], item)["route"]) for item in workloads}
    assert "installed-cli.status" in routes
    assert "uds.cli.query" in routes
    assert "daemon.ingest" in routes
