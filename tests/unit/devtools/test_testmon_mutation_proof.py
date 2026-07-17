"""Real-route tests for the bounded testmon anti-vacuity proof."""

from __future__ import annotations

from devtools.testmon_mutation_proof import _TARGET_NODEID, run_proof


def test_real_testmon_mutation_proof() -> None:
    result = run_proof()

    assert result.ok, result.failure
    assert result.target_nodeid == _TARGET_NODEID
    assert _TARGET_NODEID in result.selected_nodeids
    assert result.mutation_exit_code != 0
    assert result.restored_exit_code == 0
    assert result.severed_edge_rejected
    assert result.unrelated_selected_count < result.total_seeded_nodes
    assert result.cleanup_complete
