"""Shared types for the comparative-judgment mechanisms (rxdo.9.11, part I/K).

``JudgeIdentity`` is an interim, scoped stand-in for the not-yet-built
:mod:`polylogue.core` ``ActorRef``/``ExecutionContextRef`` pair (polylogue-h6r).
It carries the same two-part shape h6r specifies -- a durable actor identity
plus a separate content-addressed execution-context fingerprint -- so that
when h6r lands, judge identity here can be re-pointed at it without a shape
change. Do not add a third "vibes" field; execution-context drift must always
be visible as a different ``execution_context_id``, never folded into
``actor_ref``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from polylogue.core.enums import ComparativeVerdict

Dimension: TypeAlias = str
Ordering: TypeAlias = tuple[str, ...]
VerdictValue: TypeAlias = "ComparativeVerdict | Ordering"

#: Verdicts that carry no directed preference and must never be treated as a
#: weak preference by aggregation (rxdo.9.11 AC).
NON_DIRECTED_VERDICTS: frozenset[ComparativeVerdict] = frozenset(
    {
        ComparativeVerdict.TIE,
        ComparativeVerdict.INCOMPARABLE,
        ComparativeVerdict.ABSTAIN,
        ComparativeVerdict.INSUFFICIENT_EVIDENCE,
    }
)


@dataclass(frozen=True, slots=True)
class JudgeIdentity:
    """WHO judged: an operator or an agent (model + prompt/config fingerprint).

    ``actor_ref`` identifies the durable actor/model family (e.g.
    ``user:local`` or ``agent:claude-sonnet-5``); ``execution_context_id`` is
    a content-addressed fingerprint of the exact prompt/tools/runtime/config
    that produced the verdict. The same actor under two execution contexts is
    a *different* calibration stratum (rxdo.9.12) -- never collapse the two.
    """

    actor_ref: str
    execution_context_id: str

    def __post_init__(self) -> None:
        if not self.actor_ref.strip():
            raise ValueError("actor_ref cannot be empty")
        if not self.execution_context_id.strip():
            raise ValueError("execution_context_id cannot be empty")


@dataclass(frozen=True, slots=True)
class ComparativeJudgment:
    """One ``compare(items, dimension, verdict, judge_ref, blinded, ...)`` row.

    Pairwise (``len(items) == 2``) is the base case. N-wise orderings decompose
    Plackett-Luce-style (:func:`decompose_to_pairwise`) so one shape covers
    both -- there is no separate n-wise storage form. ``verdict`` is either a
    :class:`~polylogue.core.enums.ComparativeVerdict` (pairwise choice/tie/
    incomparable/abstain/insufficient-evidence) or a full ordering (a
    permutation of ``items``, best-to-worst) for n-wise elicitation.
    """

    judgment_id: str
    items: Ordering
    dimension: Dimension
    verdict: VerdictValue
    judge: JudgeIdentity
    blinded: bool
    rubric_id: str
    rubric_version: int
    evidence_refs: Ordering = ()
    elicitation_ref: str | None = None
    rationale: str | None = None
    rationale_visible: bool = False
    decided_at_ms: int = 0

    def __post_init__(self) -> None:
        if len(self.items) < 2:
            raise ValueError("comparative judgment needs at least 2 items")
        if len(set(self.items)) != len(self.items):
            raise ValueError("comparative judgment items must be distinct")
        if not self.dimension.strip():
            raise ValueError("comparative judgment dimension cannot be empty")
        if isinstance(self.verdict, ComparativeVerdict):
            if (
                self.verdict in (ComparativeVerdict.PREFER_LEFT, ComparativeVerdict.PREFER_RIGHT)
                and len(self.items) != 2
            ):
                raise ValueError("prefer_left/prefer_right verdicts require exactly 2 items")
        else:
            ordering = tuple(self.verdict)
            if sorted(ordering) != sorted(self.items):
                raise ValueError("an ordering verdict must be a permutation of items")

    @property
    def is_ordering(self) -> bool:
        """``True`` for an n-wise full-ordering verdict, ``False`` for a pairwise choice."""

        return not isinstance(self.verdict, ComparativeVerdict)


@dataclass(frozen=True, slots=True)
class PairwiseComponent:
    """One implied directed preference edge, either elicited or PL-decomposed."""

    winner_ref: str
    loser_ref: str
    dimension: Dimension
    source_judgment_id: str
    weight: float = 1.0


def decompose_to_pairwise(judgment: ComparativeJudgment) -> tuple[PairwiseComponent, ...]:
    """Lower one judgment into its implied directed preference edges.

    Pairwise ``prefer_left``/``prefer_right`` yields exactly one edge. A
    non-directed verdict (tie/incomparable/abstain/insufficient-evidence)
    yields ZERO edges -- it must never be silently treated as a weak
    preference. An n-wise ordering decomposes Plackett-Luce-style into every
    implied pair (each earlier item beats every later item in the ordering).
    """

    if isinstance(judgment.verdict, ComparativeVerdict):
        if judgment.verdict is ComparativeVerdict.PREFER_LEFT:
            return (
                PairwiseComponent(
                    winner_ref=judgment.items[0],
                    loser_ref=judgment.items[1],
                    dimension=judgment.dimension,
                    source_judgment_id=judgment.judgment_id,
                ),
            )
        if judgment.verdict is ComparativeVerdict.PREFER_RIGHT:
            return (
                PairwiseComponent(
                    winner_ref=judgment.items[1],
                    loser_ref=judgment.items[0],
                    dimension=judgment.dimension,
                    source_judgment_id=judgment.judgment_id,
                ),
            )
        return ()

    ordering = judgment.verdict
    components: list[PairwiseComponent] = []
    for i in range(len(ordering)):
        for j in range(i + 1, len(ordering)):
            components.append(
                PairwiseComponent(
                    winner_ref=ordering[i],
                    loser_ref=ordering[j],
                    dimension=judgment.dimension,
                    source_judgment_id=judgment.judgment_id,
                )
            )
    return tuple(components)


def undirected_pair_kind(judgment: ComparativeJudgment) -> tuple[str, str] | None:
    """Return the sorted item pair for a pairwise TIE/INCOMPARABLE verdict, else ``None``."""

    if (
        isinstance(judgment.verdict, ComparativeVerdict)
        and judgment.verdict in (ComparativeVerdict.TIE, ComparativeVerdict.INCOMPARABLE)
        and len(judgment.items) == 2
    ):
        a, b = judgment.items
        return (a, b) if a <= b else (b, a)
    return None


def all_items(judgments: Sequence[ComparativeJudgment]) -> tuple[str, ...]:
    """Return the deduplicated, sorted item-ref universe across a judgment set."""

    seen: set[str] = set()
    for judgment in judgments:
        seen.update(judgment.items)
    return tuple(sorted(seen))


__all__ = [
    "NON_DIRECTED_VERDICTS",
    "ComparativeJudgment",
    "Dimension",
    "JudgeIdentity",
    "Ordering",
    "PairwiseComponent",
    "VerdictValue",
    "all_items",
    "decompose_to_pairwise",
    "undirected_pair_kind",
]
