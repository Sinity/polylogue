"""Coverage-gated tool-outcome aggregates (polylogue-cuxz.4 AC4).

Anti-vacuity: every assertion here is keyed to the declared floor and the
refusal it drives. Lowering ``TOOL_OUTCOME_COVERAGE_FLOOR`` below the live
measured coverage (~28%) makes ``test_declared_floor_refuses_live_measured_
coverage`` red; deleting the refusal in ``ToolOutcomeAggregate.success_rate``
or the ``COVERAGE_BELOW_FLOOR`` gap in ``.outcome`` makes the low-coverage
tests red because the bare scalar reappears and the state falls back to ok.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from polylogue.analysis.measurement.outcome_coverage import (
    COVERAGE_BELOW_FLOOR,
    TOOL_OUTCOME_COVERAGE_FLOOR,
    build_tool_outcome_aggregate,
)
from polylogue.core.enums import ToolOutcome

# Live 2026-07-28 shape: the classified minority is overwhelmingly ok, so an
# ungated rate would read as a confident 92% success claim.
LIVE_SHAPED_COUNTS: Mapping[ToolOutcome | str, int] = {
    ToolOutcome.OK: 240,
    ToolOutcome.ERROR: 20,
    ToolOutcome.UNKNOWN: 700,
    ToolOutcome.NO_RESULT: 40,
}


def test_low_coverage_refuses_the_bare_success_rate_scalar() -> None:
    aggregate = build_tool_outcome_aggregate(LIVE_SHAPED_COUNTS, scope="origin:codex-session")

    assert aggregate.success_rate is None
    # The arithmetic still exists, but only behind a name that states its
    # population -- it is never the number a reader gets by default.
    assert aggregate.classified_success_rate == pytest.approx(240 / 260)
    assert aggregate.coverage == pytest.approx(260 / 1000)
    assert not aggregate.meets_floor


def test_low_coverage_outcome_is_degraded_with_the_named_gap() -> None:
    aggregate = build_tool_outcome_aggregate(LIVE_SHAPED_COUNTS, scope="origin:codex-session")
    outcome = aggregate.outcome

    assert outcome.state == "degraded"
    assert outcome.reason == COVERAGE_BELOW_FLOOR
    assert outcome.detail["coverage_floor"] == TOOL_OUTCOME_COVERAGE_FLOOR
    assert outcome.detail["unknown_n"] == 700
    assert outcome.detail["no_result_n"] == 40
    assert not outcome.rows_are_authoritative


def test_declared_floor_refuses_live_measured_coverage() -> None:
    """The floor must sit above the coverage actually measured on the archive.

    A floor at or below ~28% would let the live archive publish the very
    scalar this gate exists to refuse.
    """

    assert TOOL_OUTCOME_COVERAGE_FLOOR > 0.28
    live = build_tool_outcome_aggregate({ToolOutcome.OK: 28, ToolOutcome.UNKNOWN: 72})
    assert live.success_rate is None


def test_refused_render_never_prints_a_rate() -> None:
    rendered = build_tool_outcome_aggregate(LIVE_SHAPED_COUNTS, scope="origin:codex-session").render()

    assert "withheld" in rendered
    assert "26.0%" in rendered  # the coverage, stated
    assert "92.3%" not in rendered  # the refused rate, absent
    assert "below the declared floor" in rendered


def test_sufficient_coverage_presents_the_rate_and_stays_ok() -> None:
    aggregate = build_tool_outcome_aggregate(
        {ToolOutcome.OK: 90, ToolOutcome.ERROR: 5, ToolOutcome.UNKNOWN: 5},
        scope="origin:claude-ai-export",
    )

    assert aggregate.meets_floor
    assert aggregate.success_rate == pytest.approx(90 / 95)
    assert aggregate.outcome.state == "ok"
    assert "success over" in aggregate.render()


def test_zero_observations_is_empty_not_degraded() -> None:
    aggregate = build_tool_outcome_aggregate({})

    assert aggregate.coverage is None
    assert aggregate.success_rate is None
    assert aggregate.outcome.state == "empty"


def test_explicit_floor_is_honoured_and_serialized() -> None:
    aggregate = build_tool_outcome_aggregate(LIVE_SHAPED_COUNTS, floor=0.2, scope="scoped")
    payload = aggregate.to_dict()

    assert payload["coverage_floor"] == 0.2
    assert payload["meets_coverage_floor"] is True
    assert payload["success_rate"] == pytest.approx(240 / 260)
    outcome = payload["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "ok"


def test_wire_value_keys_and_unrecognised_keys() -> None:
    assert build_tool_outcome_aggregate({"ok": 8, "unknown": 2}).unknown_n == 2
    with pytest.raises(ValueError, match="unrecognised tool_outcome keys: pending"):
        build_tool_outcome_aggregate({"pending": 1})
    with pytest.raises(ValueError, match="coverage floor"):
        build_tool_outcome_aggregate({}, floor=1.5)
