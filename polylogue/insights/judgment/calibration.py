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
from polylogue.insights.judgment.types import ComparativeJudgment, VerdictValue


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


def _verdicts_agree(a: VerdictValue, b: VerdictValue) -> bool:
    if isinstance(a, ComparativeVerdict) and isinstance(b, ComparativeVerdict):
        return a is b
    if isinstance(a, ComparativeVerdict) or isinstance(b, ComparativeVerdict):
        return False
    return tuple(a) == tuple(b)


def compute_calibration(
    candidate_judgments: Sequence[ComparativeJudgment],
    gold_judgments: Sequence[ComparativeJudgment],
) -> dict[CalibrationKey, CalibrationReport]:
    """Compute per-``(actor, execution-context, dimension)`` calibration.

    ``gold_judgments`` are operator (or otherwise trusted) verdicts. A
    candidate judge's calibration is its agreement rate with gold on
    overlapping comparisons only.
    """

    gold_by_key: dict[tuple[str, ...], VerdictValue] = {}
    for gold in gold_judgments:
        gold_by_key[_overlap_key(gold)] = gold.verdict

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
            gold_verdict = gold_by_key.get(_overlap_key(judgment))
            if gold_verdict is None:
                continue
            overlap_total += 1
            if _verdicts_agree(judgment.verdict, gold_verdict):
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
