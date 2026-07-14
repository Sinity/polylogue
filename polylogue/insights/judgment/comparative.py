"""Comparative judgment objects: pairwise + n-wise, per-dimension (rxdo.9.11).

Builds validated :class:`~polylogue.insights.judgment.types.ComparativeJudgment`
records and lowers them to/from the assertion-row ``value_json`` shape stored
under :class:`~polylogue.core.enums.AssertionKind.COMPARATIVE_JUDGMENT`
(storage in ``polylogue.storage.sqlite.archive_tiers.user_write``). This
module owns validation and the value shape only, so it stays testable without
a database -- the graph already has the right bones (assertion rows), this
just defines what one comparison row means.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from polylogue.core.enums import ComparativeVerdict
from polylogue.core.hashing import hash_payload
from polylogue.insights.judgment.types import ComparativeJudgment, JudgeIdentity, Ordering

COMPARATIVE_JUDGMENT_PROTOCOL_VERSION = "polylogue.comparative-judgment.v1"


def build_comparative_judgment(
    *,
    items: Sequence[str],
    dimension: str,
    verdict: ComparativeVerdict | Sequence[str],
    judge: JudgeIdentity,
    blinded: bool,
    rubric_id: str,
    rubric_version: int,
    decided_at_ms: int,
    evidence_refs: Sequence[str] = (),
    elicitation_ref: str | None = None,
    rationale: str | None = None,
    rationale_visible: bool = False,
    judgment_id: str | None = None,
) -> ComparativeJudgment:
    """Validate and construct one comparative judgment row.

    ``judgment_id`` defaults to a deterministic content-hash id so recording
    the identical verdict twice (e.g. a retried elicitation write) is
    idempotent rather than duplicative.
    """

    resolved_verdict: ComparativeVerdict | Ordering = (
        verdict if isinstance(verdict, ComparativeVerdict) else tuple(verdict)
    )
    resolved_id = judgment_id or _deterministic_judgment_id(
        items=items,
        dimension=dimension,
        verdict=resolved_verdict,
        judge=judge,
        rubric_id=rubric_id,
        rubric_version=rubric_version,
        decided_at_ms=decided_at_ms,
    )
    return ComparativeJudgment(
        judgment_id=resolved_id,
        items=tuple(items),
        dimension=dimension,
        verdict=resolved_verdict,
        judge=judge,
        blinded=blinded,
        rubric_id=rubric_id,
        rubric_version=rubric_version,
        evidence_refs=tuple(evidence_refs),
        elicitation_ref=elicitation_ref,
        rationale=rationale,
        rationale_visible=rationale_visible,
        decided_at_ms=decided_at_ms,
    )


def _deterministic_judgment_id(
    *,
    items: Sequence[str],
    dimension: str,
    verdict: ComparativeVerdict | Ordering,
    judge: JudgeIdentity,
    rubric_id: str,
    rubric_version: int,
    decided_at_ms: int,
) -> str:
    verdict_repr: str | list[str] = verdict.value if isinstance(verdict, ComparativeVerdict) else list(verdict)
    digest = hash_payload(
        {
            "items": list(items),
            "dimension": dimension,
            "verdict": verdict_repr,
            "actor_ref": judge.actor_ref,
            "execution_context_id": judge.execution_context_id,
            "rubric_id": rubric_id,
            "rubric_version": rubric_version,
            "decided_at_ms": decided_at_ms,
        }
    )
    return f"comparative-judgment-{digest[:32]}"


def comparative_judgment_to_value(judgment: ComparativeJudgment) -> dict[str, Any]:
    """Serialize a judgment to the ``value_json`` document stored on the assertion row."""

    return {
        "protocol_version": COMPARATIVE_JUDGMENT_PROTOCOL_VERSION,
        "items": list(judgment.items),
        "dimension": judgment.dimension,
        "verdict": judgment.verdict.value
        if isinstance(judgment.verdict, ComparativeVerdict)
        else list(judgment.verdict),
        "actor_ref": judgment.judge.actor_ref,
        "execution_context_id": judgment.judge.execution_context_id,
        "blinded": judgment.blinded,
        "rubric_id": judgment.rubric_id,
        "rubric_version": judgment.rubric_version,
        "elicitation_ref": judgment.elicitation_ref,
        "rationale": judgment.rationale if judgment.rationale_visible else None,
        "rationale_visible": judgment.rationale_visible,
        "decided_at_ms": judgment.decided_at_ms,
    }


def comparative_judgment_from_value(
    judgment_id: str,
    value: Mapping[str, Any],
    *,
    evidence_refs: Sequence[str] = (),
) -> ComparativeJudgment:
    """Reconstruct a judgment from its stored ``value_json`` document."""

    raw_verdict = value["verdict"]
    verdict: ComparativeVerdict | Ordering = (
        ComparativeVerdict.from_string(raw_verdict)
        if isinstance(raw_verdict, str)
        else tuple(cast(Sequence[str], raw_verdict))
    )
    return ComparativeJudgment(
        judgment_id=judgment_id,
        items=tuple(cast(Sequence[str], value["items"])),
        dimension=cast(str, value["dimension"]),
        verdict=verdict,
        judge=JudgeIdentity(
            actor_ref=cast(str, value["actor_ref"]),
            execution_context_id=cast(str, value["execution_context_id"]),
        ),
        blinded=bool(value.get("blinded", False)),
        rubric_id=cast(str, value["rubric_id"]),
        rubric_version=int(cast(int, value["rubric_version"])),
        evidence_refs=tuple(evidence_refs),
        elicitation_ref=cast("str | None", value.get("elicitation_ref")),
        rationale=cast("str | None", value.get("rationale")),
        rationale_visible=bool(value.get("rationale_visible", False)),
        decided_at_ms=int(cast(int, value.get("decided_at_ms", 0))),
    )


__all__ = [
    "COMPARATIVE_JUDGMENT_PROTOCOL_VERSION",
    "build_comparative_judgment",
    "comparative_judgment_from_value",
    "comparative_judgment_to_value",
]
