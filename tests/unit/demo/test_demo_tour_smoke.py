"""Smoke coverage for the public one-command demo tour."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.demo.tour import run_demo_tour
from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_NARRATED_STEP_COUNT = 4


@pytest.mark.slow
def test_public_demo_tour_runs_green_end_to_end(tmp_path: Path) -> None:
    """``polylogue demo tour`` is the first command a cold outsider runs.

    It must seed, verify, and narrate every step against a real archive
    without raising and without a failed step, and the cost it presents must
    come from the canonical computed projection
    (``session_model_usage`` falling back to ``sessions.reported_cost_usd``),
    read through ``ArchiveStore.list_session_cost_insights`` -- session cost
    is not stored per session.

    ANTI-VACUITY: the production entry point is ``run_demo_tour`` on a real
    temporary archive, with nothing stubbed. Any exception from seeding,
    augmentation, or verification fails the test rather than being recorded;
    a tour that produced no work fails the step-count, byte, and
    session/message-count assertions rather than passing vacuously. Pointing
    the seed's cost read-back at a column that does not exist (its
    pre-fix ``SELECT total_cost_usd FROM session_profiles``) raises
    ``sqlite3.OperationalError`` out of ``run_demo_tour`` and turns this red;
    dropping the usage injection instead leaves the tour standing but drives
    the asserted cost to zero.
    """

    result = run_demo_tour(output_dir=tmp_path / "tour", force=True)

    assert result.ok, result.problems
    assert result.problems == ()
    assert result.verify.ok, result.verify.problems

    # A tour that ran nothing must not pass.
    assert len(result.steps) == _NARRATED_STEP_COUNT
    assert result.seed.session_count > 0
    assert result.seed.message_count > 0
    for step in result.steps:
        assert step.exit_code == 0, (step.name, step.command)
        assert step.bytes_written > 0, step.name
        assert step.output_path.is_file()

    assert result.report_json_path.is_file()
    assert result.report_markdown_path.is_file()
    assert result.transcript_path.read_text(encoding="utf-8").strip()

    with ArchiveStore.open_existing(result.archive_root) as archive:
        costs = archive.list_session_cost_insights(session_id=DEMO_CLAUDE_CODE_SESSION_ID)
    assert len(costs) == 1
    estimate = costs[0].estimate
    assert estimate.status != "unavailable", estimate.unavailable_reason
    assert estimate.total_usd is not None and estimate.total_usd > 0
