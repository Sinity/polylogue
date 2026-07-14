"""Standing-query alert budget: cooldowns, magnitude floors, deviation-first
ordering (rxdo.9.5).

rxdo.5 re-tests many watched queries every convergence pass. Naive
per-query alerting on any nonzero deviation is a guaranteed false-discovery
machine even though every underlying count is an exact enumeration with
zero sampling error -- the multiplicity problem here is purely operational
(too many looks, too often), not statistical, so the guard is an
operational budget, not a p-value correction (binding anti-goal: "no
inferential significance appears for exact enumeration").

This module is the pure decision function: given a batch of candidate
deviations and a caller-owned, serializable :class:`AlertBudgetState`, it
returns one :class:`AlertDecision` per candidate -- fired or suppressed,
always with a human-legible receipt, never silently dropped. Candidates are
ordered largest-standardized-deviation-first so a larger valid deviation
always outranks a smaller one for the shared budget. Frame degradation
never lets a candidate fire (a degraded frame cannot be trusted to have
produced a real deviation) and it is never silently absent from the
decision list either -- "frame degradation cannot silently trigger/clear an
alert" cuts both ways.

Wiring this into the actual rxdo.5 standing-query daemon loop (persisting
:class:`AlertBudgetState` across restarts, resolving `watch_ref`/frame
degradation from live query-run telemetry) is deferred to whichever lane
lands rxdo.5's scheduler; :meth:`AlertBudgetState.to_dict`/`from_dict`
exist specifically so that wiring is a thin persistence shim over this
pure core, not a rewrite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

AlertOutcome = Literal[
    "fired",
    "suppressed-cooldown",
    "suppressed-magnitude-floor",
    "suppressed-budget",
    "suppressed-frame-degraded",
]


@dataclass(frozen=True, slots=True)
class AlertBudgetPolicy:
    """Declared per-program alert policy. No field here is a p-value."""

    cooldown_ms: int
    magnitude_floor: float
    global_budget_per_window: int
    window_ms: int

    def __post_init__(self) -> None:
        if self.cooldown_ms < 0:
            raise ValueError("cooldown_ms cannot be negative")
        if self.magnitude_floor < 0:
            raise ValueError("magnitude_floor cannot be negative")
        if self.global_budget_per_window < 0:
            raise ValueError("global_budget_per_window cannot be negative")
        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")


@dataclass(frozen=True, slots=True)
class AlertCandidate:
    """One standing query's evaluated deviation this convergence pass."""

    watch_ref: str
    standardized_deviation: float
    detected_at_ms: int
    frame_degraded: bool = False
    frame_degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AlertDecision:
    """What happened to one candidate, and why -- always explicit, never silent."""

    watch_ref: str
    outcome: AlertOutcome
    standardized_deviation: float
    receipt: str


@dataclass(slots=True)
class AlertBudgetState:
    """Mutable, JSON-serializable operational state a caller persists.

    Not a statistical accumulator -- just cooldown timestamps and a spent
    counter for the current window, so a restart resumes the same budget
    posture instead of forgetting recent alerts and re-firing them.
    """

    last_alert_at_ms: dict[str, int] = field(default_factory=dict)
    window_start_ms: int | None = None
    spent_in_window: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "last_alert_at_ms": dict(self.last_alert_at_ms),
            "window_start_ms": self.window_start_ms,
            "spent_in_window": self.spent_in_window,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AlertBudgetState:
        raw_last_alert = payload.get("last_alert_at_ms") or {}
        if not isinstance(raw_last_alert, Mapping):
            raise ValueError("last_alert_at_ms must be a mapping")
        last_alert_at_ms: dict[str, int] = {}
        for key, value in raw_last_alert.items():
            if not isinstance(value, int):
                raise ValueError(f"last_alert_at_ms[{key!r}] must be an integer")
            last_alert_at_ms[str(key)] = value
        window_start = payload.get("window_start_ms")
        if window_start is not None and not isinstance(window_start, int):
            raise ValueError("window_start_ms must be an integer or None")
        spent_in_window = payload.get("spent_in_window", 0)
        if not isinstance(spent_in_window, int):
            raise ValueError("spent_in_window must be an integer")
        return cls(
            last_alert_at_ms=last_alert_at_ms,
            window_start_ms=window_start,
            spent_in_window=spent_in_window,
        )


def evaluate_alert_candidates(
    candidates: Sequence[AlertCandidate],
    *,
    policy: AlertBudgetPolicy,
    state: AlertBudgetState,
    now_ms: int,
) -> tuple[AlertDecision, ...]:
    """Decide fire/suppress for a batch of candidates, mutating ``state`` in place.

    Ordering is largest-|standardized_deviation|-first: within one budget
    window, a bigger deviation always has first claim on the shared budget
    over a smaller one, regardless of candidate input order.
    """

    if state.window_start_ms is None or now_ms - state.window_start_ms >= policy.window_ms:
        state.window_start_ms = now_ms
        state.spent_in_window = 0

    ordered = sorted(candidates, key=lambda candidate: abs(candidate.standardized_deviation), reverse=True)
    decisions: list[AlertDecision] = []
    for candidate in ordered:
        magnitude = abs(candidate.standardized_deviation)
        if magnitude < policy.magnitude_floor:
            decisions.append(
                AlertDecision(
                    watch_ref=candidate.watch_ref,
                    outcome="suppressed-magnitude-floor",
                    standardized_deviation=candidate.standardized_deviation,
                    receipt=f"|deviation|={magnitude:g} below floor {policy.magnitude_floor:g}",
                )
            )
            continue

        last_alert_ms = state.last_alert_at_ms.get(candidate.watch_ref)
        if last_alert_ms is not None and now_ms - last_alert_ms < policy.cooldown_ms:
            decisions.append(
                AlertDecision(
                    watch_ref=candidate.watch_ref,
                    outcome="suppressed-cooldown",
                    standardized_deviation=candidate.standardized_deviation,
                    receipt=(
                        f"cooldown active: {now_ms - last_alert_ms}ms since last alert "
                        f"< {policy.cooldown_ms}ms required"
                    ),
                )
            )
            continue

        # Frame degradation is checked after cooldown/floor (those are cheap,
        # policy-declared thresholds) but always before budget spend -- a
        # degraded frame must never consume budget that a trustworthy
        # candidate could use, and it must never be silently omitted from
        # the decision list either.
        if candidate.frame_degraded:
            decisions.append(
                AlertDecision(
                    watch_ref=candidate.watch_ref,
                    outcome="suppressed-frame-degraded",
                    standardized_deviation=candidate.standardized_deviation,
                    receipt=candidate.frame_degraded_reason
                    or "frame degraded: deviation cannot be trusted, alert suppressed pending full coverage",
                )
            )
            continue

        if state.spent_in_window >= policy.global_budget_per_window:
            decisions.append(
                AlertDecision(
                    watch_ref=candidate.watch_ref,
                    outcome="suppressed-budget",
                    standardized_deviation=candidate.standardized_deviation,
                    receipt=(
                        f"global alert budget ({policy.global_budget_per_window} per {policy.window_ms}ms) "
                        "exhausted by higher-ranked candidates this window"
                    ),
                )
            )
            continue

        state.spent_in_window += 1
        state.last_alert_at_ms[candidate.watch_ref] = now_ms
        decisions.append(
            AlertDecision(
                watch_ref=candidate.watch_ref,
                outcome="fired",
                standardized_deviation=candidate.standardized_deviation,
                receipt=f"fired: |deviation|={magnitude:g}, ranked within budget",
            )
        )
    return tuple(decisions)


__all__ = [
    "AlertBudgetPolicy",
    "AlertBudgetState",
    "AlertCandidate",
    "AlertDecision",
    "AlertOutcome",
    "evaluate_alert_candidates",
]
