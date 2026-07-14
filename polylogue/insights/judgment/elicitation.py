"""Active elicitation sessions: the resorter loop (rxdo.9.14, mechanism N).

An elicitation session = (target set, dimension, judge, blinding policy,
budget). The selection engine picks the next comparison to maximize
information -- closest current latent estimates, weighted toward
fewest-observed items -- so a useful ranking of ~50 items costs ~100
comparisons, not 1225 (all pairs).

Exploration quotas guarantee uncertain, minority, disconnected, or
low-coverage regions cannot starve behind a larger high-score region:
disabling the quota (``fraction=0.0``) measurably changes selection.
Every pick is receipted (candidate pool, quota state, mode) so a batch is
reconstructible without exposing hidden labels -- this module only ever
handles item refs, never provenance (mechanism F applies at the surface that
renders the picked pair, not here).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from polylogue.core.hashing import hash_payload


@dataclass(frozen=True, slots=True)
class ExplorationQuota:
    """Reserves a fraction of picks for under-covered items regardless of estimate spread."""

    fraction: float
    min_coverage_threshold: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("exploration quota fraction must be within [0, 1]")
        if self.min_coverage_threshold < 0:
            raise ValueError("min_coverage_threshold cannot be negative")


@dataclass(frozen=True, slots=True)
class SelectionReceipt:
    left_ref: str
    right_ref: str
    mode: str
    """``"exploration"`` or ``"exploitation"``."""
    candidate_pool_size: int
    quota_fraction: float
    picks_so_far: int
    exploration_picks_so_far: int
    receipt_hash: str


@dataclass
class ElicitationSession:
    """Stateful resorter loop: one outstanding selection at a time.

    Callers must ``record_verdict`` the previously returned pair before
    calling ``select_next`` again; the session does not track "outstanding"
    selections itself (kept minimal -- this is a pure selection engine, not a
    durable store; a CLI/MCP surface owns persistence of the session id).
    """

    session_id: str
    item_refs: tuple[str, ...]
    dimension: str
    budget_total: int
    quota: ExplorationQuota
    rng_seed: int = 0
    observations: dict[str, int] = field(default_factory=dict)
    estimates: dict[str, float] = field(default_factory=dict)
    picks_so_far: int = 0
    exploration_picks_so_far: int = 0
    emitted_judgment_ids: list[str] = field(default_factory=list)
    _compared_pairs: set[frozenset[str]] = field(default_factory=set, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.item_refs) < 2:
            raise ValueError("elicitation session needs at least 2 items")
        if len(set(self.item_refs)) != len(self.item_refs):
            raise ValueError("elicitation session item refs must be distinct")
        for ref in self.item_refs:
            self.observations.setdefault(ref, 0)
            self.estimates.setdefault(ref, 0.0)
        self._rng = random.Random(self.rng_seed)

    def record_verdict(self, judgment_id: str, left_ref: str, right_ref: str, *, left_won: bool | None) -> None:
        """Fold one verdict into session state. ``left_won=None`` means tie/incomparable/abstain."""

        self.observations[left_ref] = self.observations.get(left_ref, 0) + 1
        self.observations[right_ref] = self.observations.get(right_ref, 0) + 1
        self._compared_pairs.add(frozenset((left_ref, right_ref)))
        self.picks_so_far += 1
        self.emitted_judgment_ids.append(judgment_id)
        if left_won is True:
            self.estimates[left_ref] = self.estimates.get(left_ref, 0.0) + 1.0
        elif left_won is False:
            self.estimates[right_ref] = self.estimates.get(right_ref, 0.0) + 1.0

    def select_next(self) -> SelectionReceipt | None:
        """Return the next pair to elicit, or ``None`` when all pairs are exhausted."""

        remaining_pairs = [
            (a, b)
            for a, b in itertools.combinations(self.item_refs, 2)
            if frozenset((a, b)) not in self._compared_pairs
        ]
        if not remaining_pairs:
            return None
        if self.budget_total and self.picks_so_far >= self.budget_total:
            return None

        under_covered = {
            ref for ref in self.item_refs if self.observations.get(ref, 0) < self.quota.min_coverage_threshold
        }
        exploration_pairs = [pair for pair in remaining_pairs if under_covered & set(pair)]

        current_rate = self.exploration_picks_so_far / self.picks_so_far if self.picks_so_far else 0.0
        use_exploration = bool(exploration_pairs) and self.quota.fraction > 0.0 and current_rate < self.quota.fraction

        if use_exploration:
            left_ref, right_ref = self._rng.choice(exploration_pairs)
            mode = "exploration"
        else:
            # Maximize information: prefer the pair whose current latent
            # estimates are closest together (most uncertain outcome), tie-
            # broken deterministically by the session's seeded RNG.
            ranked = sorted(
                remaining_pairs,
                key=lambda pair: (
                    abs(self.estimates.get(pair[0], 0.0) - self.estimates.get(pair[1], 0.0)),
                    self._rng.random(),
                ),
            )
            left_ref, right_ref = ranked[0]
            mode = "exploitation"

        if mode == "exploration":
            self.exploration_picks_so_far += 1

        receipt_hash = hash_payload(
            {
                "session_id": self.session_id,
                "left_ref": left_ref,
                "right_ref": right_ref,
                "mode": mode,
                "picks_so_far": self.picks_so_far,
            }
        )
        return SelectionReceipt(
            left_ref=left_ref,
            right_ref=right_ref,
            mode=mode,
            candidate_pool_size=len(remaining_pairs),
            quota_fraction=self.quota.fraction,
            picks_so_far=self.picks_so_far,
            exploration_picks_so_far=self.exploration_picks_so_far,
            receipt_hash=receipt_hash,
        )


__all__ = ["ElicitationSession", "ExplorationQuota", "SelectionReceipt"]
