from __future__ import annotations

import time
from pathlib import Path

import pytest

import polylogue.demo.tour as tour_module
from polylogue.demo.seed import seed_demo_archive as real_seed_demo_archive
from polylogue.demo.tour import run_demo_tour

_INJECTED_SETUP_DELAY_S = 6.0
_TEST_FIRST_RESULT_BUDGET_S = 6.0


@pytest.mark.slow
def test_first_result_budget_excludes_one_time_archive_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``first_result_s`` must measure query latency, not cold-start seeding.

    Reproduces polylogue-3ycw: a cold environment measured
    ``first_result_s = 70.9s`` against the 30s budget even though each of
    the four narrated CLI steps individually took only 4.6-5.3s -- the
    overrun was one-time fixture seed/verify setup time being charged
    against a budget meant to catch a slow *query*.

    This test injects a real, artificial delay around the actual
    ``seed_demo_archive`` call and shrinks the first-result budget to a
    value that comfortably covers one real narrated step's own cold-start
    subprocess overhead (observed ~2.5s in this environment) but not that
    overhead plus the injected delay. That isolates the exact bug: whether
    the injected setup delay leaks into the budgeted timer.

    ANTI-VACUITY: the production entry point exercised is
    ``polylogue.demo.tour.run_demo_tour``. Reverting the fix (rebasing
    ``first_result_s`` back to the tour's overall ``start`` instead of the
    post-setup ``query_phase_start``) makes this test fail: the injected
    6s setup delay would then be included in ``first_result_s``, pushing it
    past the 6s budget and past the assertion below, and the tour's own
    ``problems`` list would report a budget overrun.
    """

    async def slow_seed_demo_archive(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        time.sleep(_INJECTED_SETUP_DELAY_S)
        return await real_seed_demo_archive(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tour_module, "seed_demo_archive", slow_seed_demo_archive)
    monkeypatch.setattr(tour_module, "FIRST_RESULT_BUDGET_S", _TEST_FIRST_RESULT_BUDGET_S)

    result = run_demo_tour(output_dir=tmp_path / "tour", force=True)

    # The injected delay must show up in the full tour duration...
    assert result.total_duration_s >= _INJECTED_SETUP_DELAY_S
    # ...but must NOT be charged against the first-result query budget.
    assert result.first_result_s < _INJECTED_SETUP_DELAY_S
    assert not any("first result exceeded" in problem for problem in result.problems)
    assert result.ok, result.problems
