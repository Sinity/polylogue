"""Carry the Source/Index assessment from the reader that supplied the rows."""

from __future__ import annotations

from polylogue.operations.read_contracts import ReadMaterialization
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.outcome import OutcomeEnvelope

MATERIALIZATION_DETAIL = "archive_materialization"


def pinned_read_materialization(archive: ArchiveStore) -> ReadMaterialization:
    assessment = archive.read_materialization_assessment()
    return ReadMaterialization.model_validate(
        {**assessment.to_dict(), "raw_state_revision": archive.raw_state_read_revision()}
    )


def materialization_outcome(outcome: OutcomeEnvelope, materialization: ReadMaterialization) -> OutcomeEnvelope:
    detail = {**outcome.detail, MATERIALIZATION_DETAIL: materialization.model_dump(mode="json")}
    gap = materialization.gap_reason
    if gap is None or outcome.state == "error":
        return outcome.model_copy(update={"detail": detail})
    prior = detail.get("gaps")
    gaps = list(prior) if isinstance(prior, list) else []
    if outcome.state == "degraded" and outcome.reason:
        gaps.insert(0, outcome.reason)
    gaps.append(gap)
    gaps = list(dict.fromkeys(gaps))
    return OutcomeEnvelope(state="degraded", reason=gaps[0], detail={**detail, "gaps": gaps})


def decorate_materialized_read(result: dict[str, object], materialization: ReadMaterialization) -> dict[str, object]:
    """Copy the result so a cached base verdict remains independent of Source."""
    result = dict(result)
    raw = result.get("outcome")
    if isinstance(raw, dict):
        result["outcome"] = materialization_outcome(OutcomeEnvelope.model_validate(raw), materialization).to_dict()
    else:
        result["materialization"] = materialization.model_dump(mode="json")
        if materialization.gap_reason is not None:
            result["outcome"] = OutcomeEnvelope(
                state="degraded",
                reason=materialization.gap_reason,
                detail={MATERIALIZATION_DETAIL: materialization.model_dump(mode="json")},
            ).to_dict()
        payload = result.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("outcome"), dict):
            result["payload"] = {
                **payload,
                "outcome": materialization_outcome(
                    OutcomeEnvelope.model_validate(payload["outcome"]), materialization
                ).to_dict(),
            }
    return result
