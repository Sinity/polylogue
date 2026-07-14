"""Judges as actors (human or agent) with measured calibration (rxdo.9.12, mechanism L).

``judge_ref`` identifies WHO judged: the operator, or an agent (model +
prompt/config fingerprint -- a judge is a program). Nothing here assumes
human. Calibration is a REPORT computed fresh from the judgment substrate,
stratified by exact ``(actor_ref, execution_context_id, dimension)`` -- never
a stored ``JudgeSpec`` table, and never pooled across execution contexts: an
unseen context has unknown calibration, not inherited confidence from a
sibling context under the same actor.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from polylogue.core.enums import ComparativeVerdict
from polylogue.insights.judgment.types import ComparativeJudgment, decompose_to_pairwise


@dataclass(frozen=True, slots=True)
class CalibrationKey:
    actor_ref: str
    execution_context_id: str
    dimension: str


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Per-stratum calibration: agreement with gold plus verdict-mix visibility."""

    key: CalibrationKey
    n_gold_overlap: int
    agreement_rate: float | None
    """``None`` means unknown -- zero gold overlap in this exact stratum."""
    tie_rate: float
    incomparable_rate: float
    abstain_rate: float
    insufficient_evidence_rate: float
    n_total_verdicts: int


def _overlap_key(judgment: ComparativeJudgment) -> tuple[str, ...]:
    """Two judgments are "the same comparison" iff same dimension + item set."""

    return (judgment.dimension, *sorted(judgment.items))


def _winner_identity(judgment: ComparativeJudgment) -> frozenset[tuple[str, str]] | ComparativeVerdict:
    """Order-independent, representation-independent identity for one verdict.

    Blinding randomizes which item lands at ``items[0]`` (left) vs
    ``items[1]`` (right) per judgment, so ``PREFER_LEFT``/``PREFER_RIGHT``
    are only meaningful relative to *this judgment's own* ``items`` order --
    the same real-world winner can be recorded as ``PREFER_LEFT`` in one
    judgment and ``PREFER_RIGHT`` in another that happened to draw the
    opposite left/right placement. A directed verdict -- pairwise
    ``PREFER_LEFT``/``PREFER_RIGHT`` *or* an n-wise ordering -- is resolved
    to the full set of implied ``(winner_ref, loser_ref)`` edges via
    :func:`decompose_to_pairwise` before comparing two judgments on the same
    (order-independent, see :func:`_overlap_key`) comparison. This is what
    makes a 2-item n-wise ordering and a pairwise ``PREFER_LEFT``/
    ``PREFER_RIGHT`` verdict on the same item pair comparable: both name the
    same real-world winner via the same edge-set representation, not two
    incompatible shapes (a raw ordering tuple vs a bare item ref) that can
    never compare equal. A non-directed verdict (tie/incomparable/abstain/
    insufficient-evidence) carries no winner and is compared by its own
    label.
    """

    verdict = judgment.verdict
    if isinstance(verdict, ComparativeVerdict) and verdict not in (
        ComparativeVerdict.PREFER_LEFT,
        ComparativeVerdict.PREFER_RIGHT,
    ):
        return verdict
    return frozenset((component.winner_ref, component.loser_ref) for component in decompose_to_pairwise(judgment))


def _verdicts_agree(candidate: ComparativeJudgment, gold: ComparativeJudgment) -> bool:
    return _winner_identity(candidate) == _winner_identity(gold)


def compute_calibration(
    candidate_judgments: Sequence[ComparativeJudgment],
    gold_judgments: Sequence[ComparativeJudgment],
) -> dict[CalibrationKey, CalibrationReport]:
    """Compute per-``(actor, execution-context, dimension)`` calibration.

    ``gold_judgments`` are operator (or otherwise trusted) verdicts. A
    candidate judge's calibration is its agreement rate with gold on
    overlapping comparisons only.
    """

    gold_by_key: dict[tuple[str, ...], ComparativeJudgment] = {}
    for gold in gold_judgments:
        gold_by_key[_overlap_key(gold)] = gold

    grouped: dict[CalibrationKey, list[ComparativeJudgment]] = defaultdict(list)
    for judgment in candidate_judgments:
        key = CalibrationKey(
            actor_ref=judgment.judge.actor_ref,
            execution_context_id=judgment.judge.execution_context_id,
            dimension=judgment.dimension,
        )
        grouped[key].append(judgment)

    reports: dict[CalibrationKey, CalibrationReport] = {}
    for key, judgments in grouped.items():
        n_total = len(judgments)
        tie = sum(1 for j in judgments if j.verdict is ComparativeVerdict.TIE)
        incomparable = sum(1 for j in judgments if j.verdict is ComparativeVerdict.INCOMPARABLE)
        abstain = sum(1 for j in judgments if j.verdict is ComparativeVerdict.ABSTAIN)
        insufficient = sum(1 for j in judgments if j.verdict is ComparativeVerdict.INSUFFICIENT_EVIDENCE)

        overlap_agree = 0
        overlap_total = 0
        for judgment in judgments:
            gold_judgment = gold_by_key.get(_overlap_key(judgment))
            if gold_judgment is None:
                continue
            overlap_total += 1
            if _verdicts_agree(judgment, gold_judgment):
                overlap_agree += 1

        reports[key] = CalibrationReport(
            key=key,
            n_gold_overlap=overlap_total,
            agreement_rate=(overlap_agree / overlap_total) if overlap_total > 0 else None,
            tie_rate=tie / n_total if n_total else 0.0,
            incomparable_rate=incomparable / n_total if n_total else 0.0,
            abstain_rate=abstain / n_total if n_total else 0.0,
            insufficient_evidence_rate=insufficient / n_total if n_total else 0.0,
            n_total_verdicts=n_total,
        )
    return reports


__all__ = ["CalibrationKey", "CalibrationReport", "compute_calibration"]
