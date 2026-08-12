"""Derived snapshots and readiness agree across convergence schedules."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    assert_derived_readiness_equivalent,
    build_converged_archive,
    rich_convergence_pathology,
    rotated_session_order,
)


@settings(
    max_examples=8,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=1, max_value=len(rich_convergence_pathology().sessions) - 1))
def test_convergence_property_derived_snapshot_readiness_equality(tmp_path: Path, shift: int) -> None:
    """Bulk and incremental production convergence expose the same derived state."""
    pathology = rich_convergence_pathology()
    order = rotated_session_order(pathology, shift)
    bulk = build_converged_archive(tmp_path / "bulk", pathology, session_order=order)
    incremental = build_converged_archive(tmp_path / "incremental", pathology, session_order=order, incremental=True)

    assert_derived_readiness_equivalent(bulk.root, incremental.root)


def test_production_insights_convergence_publishes_final_fts_freshness(tmp_path: Path) -> None:
    """The production stage, not this harness, owns post-insight FTS readiness."""
    archive = build_converged_archive(tmp_path / "archive", rich_convergence_pathology())

    with sqlite3.connect(archive.root / "index.db") as conn:
        row = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows
            FROM fts_freshness_state
            WHERE surface = 'session_work_events_fts'
            """
        ).fetchone()

    assert row is not None
    state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows = row
    assert state == "ready"
    assert source_rows > 0
    assert indexed_rows == source_rows
    assert (missing_rows, excess_rows, duplicate_rows) == (0, 0, 0)
