"""The JSON body of a comparative judgment on the ``mutation.judgment.record`` wire.

Deliberately not in :mod:`polylogue.analysis.judgment.types`: that module is IN
the derived-schema identity closure, so a transport shape added there would move
the archive's schema identity and force a reconvergence for a CLI change
(CLAUDE.md, "the identity moves on ordinary code edits"). ``polylogue/operations``
is outside the closure, which is where a transport contract belongs anyway.

:func:`comparative_judgment_wire_form` and
:func:`comparative_judgment_from_wire_form` are exact inverses; the round-trip
law is what keeps a judgment recorded through the daemon identical to one an
in-process caller would have written.
"""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.analysis.judgment.types import ComparativeJudgment, JudgeIdentity, VerdictValue
from polylogue.core.enums import ComparativeVerdict


def comparative_judgment_wire_form(judgment: ComparativeJudgment) -> dict[str, object]:
    """Render one judgment as the JSON body of ``mutation.judgment.record``."""

    verdict = judgment.verdict
    return {
        "judgment_id": judgment.judgment_id,
        "items": list(judgment.items),
        "dimension": judgment.dimension,
        "verdict": verdict.value if isinstance(verdict, ComparativeVerdict) else list(verdict),
        "judge": {
            "actor_ref": judgment.judge.actor_ref,
            "execution_context_id": judgment.judge.execution_context_id,
            "role": judgment.judge.role,
        },
        "blinded": judgment.blinded,
        "rubric_id": judgment.rubric_id,
        "rubric_version": judgment.rubric_version,
        "evidence_refs": list(judgment.evidence_refs),
        "elicitation_ref": judgment.elicitation_ref,
        "rationale": judgment.rationale,
        "rationale_visible": judgment.rationale_visible,
        "decided_at_ms": judgment.decided_at_ms,
    }


def comparative_judgment_from_wire_form(body: Mapping[str, object]) -> ComparativeJudgment:
    """Rebuild one judgment from :func:`comparative_judgment_wire_form`.

    A verdict arrives either as one :class:`ComparativeVerdict` token or as a
    full ordering; a list is always an ordering, because every declared verdict
    token is a scalar string. ``ComparativeJudgment.__post_init__`` re-checks
    every invariant, so a malformed body fails here rather than reaching the
    durable row.
    """

    raw_verdict = body["verdict"]
    verdict: VerdictValue = (
        tuple(str(item) for item in raw_verdict)
        if isinstance(raw_verdict, list | tuple)
        else ComparativeVerdict.from_string(str(raw_verdict))
    )
    judge_body = body["judge"]
    if not isinstance(judge_body, Mapping):
        raise ValueError("judgment judge must be an object")
    evidence_refs = body.get("evidence_refs") or ()
    if not isinstance(evidence_refs, list | tuple):
        raise ValueError("judgment evidence_refs must be a list")
    items = body["items"]
    if not isinstance(items, list | tuple):
        raise ValueError("judgment items must be a list")
    elicitation_ref = body.get("elicitation_ref")
    rationale = body.get("rationale")
    return ComparativeJudgment(
        judgment_id=str(body["judgment_id"]),
        items=tuple(str(item) for item in items),
        dimension=str(body["dimension"]),
        verdict=verdict,
        judge=JudgeIdentity(
            actor_ref=str(judge_body["actor_ref"]),
            execution_context_id=str(judge_body["execution_context_id"]),
            role=str(judge_body.get("role") or "judge"),
        ),
        blinded=bool(body["blinded"]),
        rubric_id=str(body["rubric_id"]),
        rubric_version=int(str(body["rubric_version"])),
        evidence_refs=tuple(str(ref) for ref in evidence_refs),
        elicitation_ref=None if elicitation_ref is None else str(elicitation_ref),
        rationale=None if rationale is None else str(rationale),
        rationale_visible=bool(body.get("rationale_visible", False)),
        decided_at_ms=int(str(body.get("decided_at_ms", 0))),
    )


__all__ = ["comparative_judgment_from_wire_form", "comparative_judgment_wire_form"]
