"""A real live append delta reaches the same normalized archive as full replay."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from tests.infra.convergence_harness import (
    CorpusMember,
    append_convergence_members,
    assert_append_provenance,
    assert_archives_equivalent,
    build_append_prefix_archive,
    build_converged_archive,
    build_full_live_archive,
    convergence_max_examples,
    drop_one_insight_row,
    rich_convergence_pathology,
)


@settings(
    max_examples=convergence_max_examples(),
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.parametrize(
    "member",
    append_convergence_members(),
    ids=lambda member: f"{member.provider}-{member.spec.package_version}-{member.spec.element_kind}",
)
@given(st.integers(min_value=2, max_value=6))
def test_convergence_property_append_prefix_matches_full(tmp_path: Path, member: CorpusMember, split_line: int) -> None:
    full = build_full_live_archive(tmp_path / "full", member)
    appended = build_append_prefix_archive(tmp_path / "append", member, split_line=split_line)

    assert_append_provenance(appended.root)
    assert_archives_equivalent(full, appended, compare_acquisition_route=False)
    with sqlite3.connect(full.root / "index.db") as left, sqlite3.connect(appended.root / "index.db") as right:
        assert (
            left.execute("SELECT COUNT(*) FROM attachments").fetchone()
            == right.execute("SELECT COUNT(*) FROM attachments").fetchone()
        )
        assert (
            left.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()
            == right.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()
        )


def test_convergence_property_append_prefix_red_twin_detects_dropped_insight(tmp_path: Path) -> None:
    corpus = rich_convergence_pathology()
    baseline = build_converged_archive(tmp_path / "baseline", type(corpus)((corpus.members[0],)))
    mutated = build_converged_archive(tmp_path / "mutated", type(corpus)((corpus.members[0],)))
    drop_one_insight_row(mutated.root)

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(baseline, mutated)
