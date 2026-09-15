"""Coverage-gated tool-outcome aggregates (polylogue-cuxz.4 AC4).

``blocks.tool_outcome`` is the canonical structural outcome, and ``unknown``
is a deliberate retained refusal -- never a success. A tool success rate
computed over a population where a large share of invocations carry
``unknown`` (or were never paired with a result at all) is therefore a
misleading scalar: it silently reports the rate over the classified minority
while reading as a statement about the whole population.

This module is the read-side gate. Every aggregate it builds carries its
coverage alongside the rate, and below a declared coverage floor it refuses
the bare scalar: :attr:`ToolOutcomeAggregate.success_rate` is ``None`` and the
aggregate's :class:`~polylogue.surfaces.outcome.OutcomeEnvelope` is
``degraded`` with the coverage gap named. Per the terminal-outcome contract
``degraded`` outranks ``empty``, so an aggregate behind a named coverage gap
is never presented as a confident rate nor as an empty scope.

The module is deliberately storage-agnostic and lives outside the derived
schema-identity closure: it consumes counts a caller already has and returns
a pure receipt, so wiring a new read surface to it does not move the archive
schema identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.core.enums import ToolOutcome
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome

TOOL_OUTCOME_COVERAGE_FLOOR = 0.8
"""Declared floor: the share of invocations that must carry a trusted
pass/fail outcome before a bare success-rate scalar may be presented.

What the number buys, stated so it is reviewable rather than arbitrary. At
coverage ``c`` the unmeasured share is ``1 - c``, and the true whole-population
success rate can differ from the classified rate by at most that much -- every
uncounted invocation could have gone the other way. So the floor is a declared
ceiling on how wrong the published scalar may be:

* at 0.81 the reader sees a rate that can be off by at most 19 points. A
  published "92% success" is then somewhere in 73-100%: the claim survives
  in shape, and the uncounted bucket is reported beside it.
* at 0.79 the error bound exceeds one fifth of the scale. "92% success" is
  then compatible with 73% and with 100% and the reader has no way to tell
  which, so the scalar has stopped carrying its own caveat.

0.8 is where that bound crosses 20%. It is a judgment about publishable
error, not a measured property of the archive; moving it is a one-line change
whose consequence is exactly the bound above.

For scale, measured live on 2026-07-28 whole-archive outcome coverage sat near
28%, with per-origin coverage from 0% (``chatgpt-export``) to 100%
(``claude-ai-export``). Any floor below the worst-covered origin would let that
origin publish exactly the misleading scalar this gate exists to refuse.
"""

COVERAGE_BELOW_FLOOR = "tool_outcome_coverage_below_floor"
"""Named gap recorded when coverage falls under the declared floor."""


def _count(counts: Mapping[ToolOutcome | str, int], outcome: ToolOutcome) -> int:
    value = counts.get(outcome)
    if value is None:
        value = counts.get(outcome.value, 0)
    return int(value or 0)


@dataclass(frozen=True, slots=True)
class ToolOutcomeAggregate:
    """A tool-outcome aggregate that always answers 'a rate over WHAT'.

    ``classified_n`` counts invocations whose structural outcome is trusted
    (``ok`` or ``error``). ``unknown_n`` and ``no_result_n`` are the two
    distinct uncounted buckets and are preserved separately -- a retained
    unknown-outcome reason is not the same fact as an unpaired invocation,
    and neither is ever folded into the numerator as a success.
    """

    ok_n: int
    error_n: int
    unknown_n: int
    no_result_n: int
    floor: float = TOOL_OUTCOME_COVERAGE_FLOOR
    scope: str = ""

    @property
    def total_n(self) -> int:
        """Every observed invocation, classified or not."""

        return self.ok_n + self.error_n + self.unknown_n + self.no_result_n

    @property
    def classified_n(self) -> int:
        """Invocations carrying a trusted structural pass/fail outcome."""

        return self.ok_n + self.error_n

    @property
    def uncounted_n(self) -> int:
        """Invocations excluded from the rate: unknown plus unpaired."""

        return self.unknown_n + self.no_result_n

    @property
    def coverage(self) -> float | None:
        """Share of invocations that carry a trusted outcome.

        ``None`` when nothing was observed -- an undefined coverage, never a
        zero that reads as measured absence.
        """

        total = self.total_n
        if total <= 0:
            return None
        return self.classified_n / total

    @property
    def meets_floor(self) -> bool:
        coverage = self.coverage
        return coverage is not None and coverage >= self.floor

    @property
    def success_rate(self) -> float | None:
        """The bare success-rate scalar, or ``None`` when it may not be shown.

        Refused below the declared coverage floor. Callers that need the
        underlying arithmetic regardless must ask for
        :attr:`classified_success_rate` and say what population it covers.
        """

        if not self.meets_floor:
            return None
        return self.classified_success_rate

    @property
    def classified_success_rate(self) -> float | None:
        """Success rate over the classified population only, ungated."""

        if self.classified_n <= 0:
            return None
        return self.ok_n / self.classified_n

    @property
    def outcome(self) -> OutcomeEnvelope:
        """Terminal outcome for an envelope carrying this aggregate."""

        detail: dict[str, object] = {
            "ok_n": self.ok_n,
            "error_n": self.error_n,
            "unknown_n": self.unknown_n,
            "no_result_n": self.no_result_n,
            "classified_n": self.classified_n,
            "total_n": self.total_n,
            "coverage": self.coverage,
            "coverage_floor": self.floor,
        }
        if self.scope:
            detail["scope"] = self.scope
        gaps = () if self.meets_floor or self.total_n <= 0 else (COVERAGE_BELOW_FLOOR,)
        return decide_outcome(matched=self.total_n, degraded=gaps, detail=detail)

    def render(self) -> str:
        """One line that never presents a refused rate as a number."""

        scope_label = self.scope or "declared scope"
        coverage = self.coverage
        coverage_label = "undefined" if coverage is None else f"{coverage * 100:.1f}%"
        floor_label = f"{self.floor * 100:.1f}%"
        tail = (
            f"(ok={self.ok_n}, error={self.error_n}, unknown={self.unknown_n}, "
            f"no_result={self.no_result_n}, coverage={coverage_label}, floor={floor_label})"
        )
        if self.total_n <= 0:
            return f"no tool invocations in {scope_label} {tail}"
        if not self.meets_floor:
            return (
                f"success rate withheld for {scope_label}: outcome coverage "
                f"{coverage_label} is below the declared floor {floor_label} {tail}"
            )
        rate = self.success_rate
        pct = "n/a" if rate is None else f"{rate * 100:.1f}%"
        return f"{pct} success over {scope_label} {tail}"

    def to_dict(self) -> dict[str, object]:
        """Serializable projection shared by plaintext and JSON readers."""

        return {
            "scope": self.scope or None,
            "ok_n": self.ok_n,
            "error_n": self.error_n,
            "unknown_n": self.unknown_n,
            "no_result_n": self.no_result_n,
            "classified_n": self.classified_n,
            "uncounted_n": self.uncounted_n,
            "total_n": self.total_n,
            "coverage": self.coverage,
            "coverage_floor": self.floor,
            "meets_coverage_floor": self.meets_floor,
            "success_rate": self.success_rate,
            "classified_success_rate": self.classified_success_rate,
            "summary": self.render(),
            "outcome": self.outcome.to_dict(),
        }


def build_tool_outcome_aggregate(
    counts: Mapping[ToolOutcome | str, int],
    *,
    floor: float = TOOL_OUTCOME_COVERAGE_FLOOR,
    scope: str = "",
) -> ToolOutcomeAggregate:
    """Build a coverage-gated aggregate from ``tool_outcome`` counts.

    ``counts`` is keyed by :class:`ToolOutcome` (or its wire value). Absent
    members count zero; unrecognised keys are rejected rather than silently
    dropped into an unnamed bucket.
    """

    known = {member.value for member in ToolOutcome}
    unknown_keys = sorted(
        str(key.value if isinstance(key, ToolOutcome) else key)
        for key in counts
        if str(key.value if isinstance(key, ToolOutcome) else key) not in known
    )
    if unknown_keys:
        raise ValueError(f"unrecognised tool_outcome keys: {', '.join(unknown_keys)}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"coverage floor must lie in [0, 1], got {floor!r}")
    return ToolOutcomeAggregate(
        ok_n=_count(counts, ToolOutcome.OK),
        error_n=_count(counts, ToolOutcome.ERROR),
        unknown_n=_count(counts, ToolOutcome.UNKNOWN),
        no_result_n=_count(counts, ToolOutcome.NO_RESULT),
        floor=floor,
        scope=scope,
    )


__all__ = [
    "COVERAGE_BELOW_FLOOR",
    "TOOL_OUTCOME_COVERAGE_FLOOR",
    "ToolOutcomeAggregate",
    "build_tool_outcome_aggregate",
]
