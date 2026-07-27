"""Like-for-like comparison of two workload receipts (polylogue-b054.1.1.3).

``devtools verify`` and ``devtools test`` already emit a real ``WorkloadReceipt``
(``polylogue/scenarios/workload.py``) for every managed pytest step, embedded
under ``metadata["workload_receipt"]`` and persisted to each step's
``postmortem.json``. PR #2934/#2980 extended that receipt with per-process-tree
and cgroup RSS/PSS/swap/read/write accounting, but nothing yet *used* two such
receipts together to answer the question the resource accounting exists to
answer: "is this run clean, comparable to a named baseline, and within its
declared physical envelope?"

This module closes that gap. It does not invent a new sampling mechanism —
it reads the receipts PR #2934/#2980 already produce and:

1. Confirms the two runs are actually comparable ("like-for-like"): the same
   declared workload/family/input identity, and both terminated
   ``succeeded`` — a partial, failed, or cancelled run cannot anchor or be
   judged against a physical-envelope target.
2. Evaluates a small set of declared comparison targets (wall-time speedup
   relative to the baseline, an absolute peak-PSS ceiling) against the
   observed ``execute``-phase measures, using the same
   :class:`~polylogue.scenarios.workload.BudgetVerdict` vocabulary the rest of
   the workload-receipt system already uses — ``pass``, ``exceeded``, or
   ``measurement-unavailable`` (never a fabricated pass when a measure was not
   captured).
3. Names an explicit blocker and a linked follow-up reference for every
   unmet target, so an unmet 2x/peak-memory target is durable evidence
   instead of a silently-dropped aspiration (polylogue-b054.1.1 AC5/AC8).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from polylogue.scenarios.workload import BudgetMeasure, BudgetVerdict

#: The follow-up that owns closing the still-outstanding memory-amplification
#: gap identified while proving polylogue-b054.1.1's own 2x/<3GiB target
#: (see that bead's notes, 2026-07-16).
DEFAULT_FOLLOW_UP_REF = "polylogue-b054.1.1.2"

_GIB = 1024**3


class ComparisonTargetKind(str, Enum):
    """How a target's ``limit`` relates the candidate to the baseline."""

    #: candidate <= baseline * limit (e.g. limit=0.5 asserts a 2x speedup).
    MAX_RATIO_OF_BASELINE = "max-ratio-of-baseline"
    #: candidate <= limit, independent of the baseline's own value.
    MAX_ABSOLUTE = "max-absolute"


@dataclass(frozen=True, slots=True)
class ReceiptComparisonTarget:
    """One declared comparison budget over a workload receipt's phase measure."""

    measure: BudgetMeasure
    kind: ComparisonTargetKind
    limit: float
    phase: str = "execute"
    follow_up_ref: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "measure": self.measure.value,
            "kind": self.kind.value,
            "limit": self.limit,
            "phase": self.phase,
            "follow_up_ref": self.follow_up_ref,
        }


@dataclass(frozen=True, slots=True)
class ReceiptComparisonResult:
    """Verdict for one declared target after comparing baseline vs candidate."""

    measure: BudgetMeasure
    kind: ComparisonTargetKind
    phase: str
    limit: float
    baseline_value: float | None
    candidate_value: float | None
    ratio: float | None
    verdict: BudgetVerdict
    blocker: str | None
    follow_up_ref: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "measure": self.measure.value,
            "kind": self.kind.value,
            "phase": self.phase,
            "limit": self.limit,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "ratio": self.ratio,
            "verdict": self.verdict.value,
            "blocker": self.blocker,
            "follow_up_ref": self.follow_up_ref,
        }


def default_seed_comparison_targets(
    *,
    follow_up_ref: str = DEFAULT_FOLLOW_UP_REF,
    speedup_ratio: float = 0.5,
    peak_pss_ceiling_bytes: float = 3 * _GIB,
) -> tuple[ReceiptComparisonTarget, ...]:
    """Return the polylogue-b054.1.1 seed-comparison targets (2x wall, <3GiB peak PSS).

    ``speedup_ratio=0.5`` means the candidate's wall time must be at most half
    the baseline's — i.e. at least a 2x speedup, matching the target recorded
    in polylogue-b054.1.1's own acceptance criteria.
    """
    return (
        ReceiptComparisonTarget(
            measure=BudgetMeasure.WALL_MS,
            kind=ComparisonTargetKind.MAX_RATIO_OF_BASELINE,
            limit=speedup_ratio,
            phase="execute",
            follow_up_ref=follow_up_ref,
        ),
        ReceiptComparisonTarget(
            measure=BudgetMeasure.PEAK_PSS_BYTES,
            kind=ComparisonTargetKind.MAX_ABSOLUTE,
            limit=peak_pss_ceiling_bytes,
            phase="execute",
            follow_up_ref=follow_up_ref,
        ),
    )


def _phase(receipt: Mapping[str, Any], phase: str) -> Mapping[str, Any] | None:
    phases = receipt.get("phases")
    if not isinstance(phases, list):
        return None
    for candidate_phase in phases:
        if isinstance(candidate_phase, dict) and candidate_phase.get("name") == phase:
            return candidate_phase
    return None


def _measure_value(receipt: Mapping[str, Any], *, phase: str, measure: BudgetMeasure) -> float | None:
    observed_phase = _phase(receipt, phase)
    if observed_phase is None:
        return None
    value = observed_phase.get(measure.value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def identity_mismatches(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every field that keeps two receipts from being like-for-like.

    Comparable receipts must declare the identical workload/family identity
    and the identical input digest (``spec.inputs[*].input_id`` is a content
    hash of the workload's command/selection) — otherwise a ratio between the
    two measures nothing.
    """
    raw_baseline_spec = baseline.get("spec")
    raw_candidate_spec = candidate.get("spec")
    baseline_spec: dict[str, Any] = raw_baseline_spec if isinstance(raw_baseline_spec, dict) else {}
    candidate_spec: dict[str, Any] = raw_candidate_spec if isinstance(raw_candidate_spec, dict) else {}
    mismatches: list[str] = []
    for field in ("workload_id", "family_id", "measurement_scope"):
        baseline_value = baseline_spec.get(field)
        candidate_value = candidate_spec.get(field)
        if baseline_value != candidate_value:
            mismatches.append(f"{field}: baseline={baseline_value!r} candidate={candidate_value!r}")
    baseline_inputs = tuple(item.get("input_id") for item in baseline_spec.get("inputs", []) if isinstance(item, dict))
    candidate_inputs = tuple(
        item.get("input_id") for item in candidate_spec.get("inputs", []) if isinstance(item, dict)
    )
    if baseline_inputs != candidate_inputs:
        mismatches.append(f"inputs: baseline={baseline_inputs!r} candidate={candidate_inputs!r}")
    return tuple(mismatches)


def evaluate_target(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    target: ReceiptComparisonTarget,
) -> ReceiptComparisonResult:
    """Score one declared target without inventing a verdict for missing data."""
    baseline_value = _measure_value(baseline, phase=target.phase, measure=target.measure)
    candidate_value = _measure_value(candidate, phase=target.phase, measure=target.measure)

    ratio = None
    if baseline_value is not None and candidate_value is not None and baseline_value != 0:
        ratio = candidate_value / baseline_value

    if baseline_value is None or candidate_value is None:
        verdict = BudgetVerdict.MEASUREMENT_UNAVAILABLE
    elif target.kind is ComparisonTargetKind.MAX_RATIO_OF_BASELINE:
        effective_limit = baseline_value * target.limit
        verdict = BudgetVerdict.PASS if candidate_value <= effective_limit else BudgetVerdict.EXCEEDED
    else:
        verdict = BudgetVerdict.PASS if candidate_value <= target.limit else BudgetVerdict.EXCEEDED

    blocker: str | None = None
    if verdict is BudgetVerdict.EXCEEDED:
        if target.kind is ComparisonTargetKind.MAX_RATIO_OF_BASELINE:
            blocker = (
                f"{target.measure.value} on phase {target.phase!r}: candidate={candidate_value!r} "
                f"baseline={baseline_value!r} ratio={ratio!r} exceeds the declared "
                f"max-ratio-of-baseline target {target.limit!r}"
            )
        else:
            blocker = (
                f"{target.measure.value} on phase {target.phase!r}: candidate={candidate_value!r} "
                f"exceeds the declared absolute ceiling {target.limit!r}"
            )

    return ReceiptComparisonResult(
        measure=target.measure,
        kind=target.kind,
        phase=target.phase,
        limit=target.limit,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        ratio=ratio,
        verdict=verdict,
        blocker=blocker,
        follow_up_ref=target.follow_up_ref if verdict is BudgetVerdict.EXCEEDED else None,
    )


def compare_receipts(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    targets: Sequence[ReceiptComparisonTarget] | None = None,
) -> dict[str, Any]:
    """Compare two workload receipts and score every declared target.

    Returns a JSON-serializable dict; ``like_for_like`` is ``False`` whenever
    the two receipts do not share workload identity or either run did not
    terminate ``succeeded`` — a dirty or mismatched pair cannot license any
    budget verdict below, regardless of what the individual measures say.
    """
    resolved_targets = tuple(targets) if targets is not None else default_seed_comparison_targets()
    mismatches = identity_mismatches(baseline, candidate)
    baseline_status = baseline.get("status")
    candidate_status = candidate.get("status")
    clean = baseline_status == "succeeded" and candidate_status == "succeeded"
    like_for_like = not mismatches and clean

    results = tuple(evaluate_target(baseline, candidate, target) for target in resolved_targets)
    exceeded = tuple(result for result in results if result.verdict is BudgetVerdict.EXCEEDED)

    return {
        "like_for_like": like_for_like,
        "identity_mismatches": list(mismatches),
        "baseline_status": baseline_status,
        "candidate_status": candidate_status,
        "targets_met": like_for_like and not exceeded,
        "results": [result.to_payload() for result in results],
    }


def _unwrap_receipt(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a bare workload receipt or a wrapping postmortem.json."""
    workload_receipt = payload.get("workload_receipt")
    if isinstance(workload_receipt, dict):
        return workload_receipt
    return payload


def load_receipt(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    return _unwrap_receipt(payload)


def _render_text(comparison: Mapping[str, Any]) -> str:
    lines: list[str] = []
    if comparison["like_for_like"]:
        lines.append("like-for-like: yes")
    else:
        lines.append("like-for-like: NO")
        for mismatch in comparison["identity_mismatches"]:
            lines.append(f"  identity mismatch: {mismatch}")
        if comparison["baseline_status"] != "succeeded":
            lines.append(f"  baseline run did not succeed: status={comparison['baseline_status']!r}")
        if comparison["candidate_status"] != "succeeded":
            lines.append(f"  candidate run did not succeed: status={comparison['candidate_status']!r}")
    for result in comparison["results"]:
        marker = {"pass": "PASS", "exceeded": "EXCEEDED", "measurement-unavailable": "N/A"}[result["verdict"]]
        lines.append(
            f"[{marker}] {result['measure']} ({result['phase']}): "
            f"baseline={result['baseline_value']!r} candidate={result['candidate_value']!r} "
            f"ratio={result['ratio']!r} limit={result['limit']!r} kind={result['kind']}"
        )
        if result["blocker"]:
            lines.append(f"    blocker: {result['blocker']}")
            lines.append(f"    follow-up: {result['follow_up_ref']}")
    lines.append(f"targets_met: {comparison['targets_met']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a baseline and candidate workload receipt (devtools verify/test "
            "postmortem.json or a bare workload_receipt payload) for a clean, "
            "like-for-like seed/incident comparison (polylogue-b054.1.1.3)."
        )
    )
    parser.add_argument("--baseline", required=True, type=Path, help="Path to the baseline receipt JSON.")
    parser.add_argument("--candidate", required=True, type=Path, help="Path to the candidate receipt JSON.")
    parser.add_argument(
        "--follow-up-ref",
        default=DEFAULT_FOLLOW_UP_REF,
        help="Tracking-item reference cited on any unmet target (default: %(default)s).",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=0.5,
        help="Maximum candidate/baseline wall-time ratio to pass (default 0.5 = 2x speedup).",
    )
    parser.add_argument(
        "--peak-pss-ceiling-mb",
        type=float,
        default=3 * 1024.0,
        help="Absolute peak-PSS ceiling in MiB (default 3072 MiB = 3 GiB).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the complete comparison as JSON.")
    args = parser.parse_args(argv)

    baseline = load_receipt(args.baseline)
    candidate = load_receipt(args.candidate)
    targets = default_seed_comparison_targets(
        follow_up_ref=args.follow_up_ref,
        speedup_ratio=args.speedup_ratio,
        peak_pss_ceiling_bytes=args.peak_pss_ceiling_mb * 1024 * 1024,
    )
    comparison = compare_receipts(baseline, candidate, targets)

    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        print(_render_text(comparison))

    if not comparison["like_for_like"]:
        return 2
    if not comparison["targets_met"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
