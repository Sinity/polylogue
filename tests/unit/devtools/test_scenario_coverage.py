from __future__ import annotations

from devtools.scenario_coverage import build_runtime_scenario_coverage


def test_build_runtime_scenario_coverage_tracks_the_current_authored_map() -> None:
    coverage = build_runtime_scenario_coverage()

    assert "raw_validation_state" in coverage.artifacts
    assert "tool_use_source_blocks" in coverage.artifacts
    assert "session_product_source_conversations" in coverage.artifacts
    assert "plan-validation-backlog" in coverage.operations
    assert "materialize-action-events" in coverage.operations
    assert "materialize-session-products" in coverage.operations
    assert coverage.uncovered_artifacts == ()
    assert coverage.uncovered_operations == ()
