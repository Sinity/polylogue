"""Deterministic selection of replay-eligible raw revision evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionKind


class ApplicationDecision(StrEnum):
    SELECTED_BASELINE = "selected_baseline"
    APPLIED_APPEND = "applied_append"
    SUPERSEDED = "superseded"
    AMBIGUOUS = "ambiguous"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class RevisionCandidate:
    raw_id: str
    logical_source_key: str
    kind: RawRevisionKind
    source_revision: str
    acquisition_generation: int
    authority: RawRevisionAuthority
    blob_size: int
    predecessor_raw_id: str | None = None
    baseline_raw_id: str | None = None
    append_start_offset: int | None = None
    append_end_offset: int | None = None


@dataclass(frozen=True, slots=True)
class RevisionApplication:
    raw_id: str
    decision: ApplicationDecision
    accepted_raw_id: str | None
    accepted_source_revision: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class RevisionReplayPlan:
    logical_source_key: str
    applications: tuple[RevisionApplication, ...]
    accepted_chain: tuple[str, ...] = ()

    @property
    def accepted_raw_ids(self) -> tuple[str, ...]:
        return self.accepted_chain


def plan_revision_replay(candidates: list[RevisionCandidate]) -> RevisionReplayPlan:
    """Choose a unique proven full baseline and its exact append chain.

    No enumeration order, timestamp, raw-id ordering, or provider timestamp can
    promote evidence. A tie or branch stops replay at the last unique head.
    """
    if not candidates:
        raise ValueError("revision replay requires at least one candidate")
    logical_keys = {candidate.logical_source_key for candidate in candidates}
    if len(logical_keys) != 1:
        raise ValueError("revision replay candidates must share one logical source key")
    logical_key = next(iter(logical_keys))
    by_id = {candidate.raw_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("revision replay candidate raw ids must be unique")

    proven_full = [
        candidate
        for candidate in candidates
        if candidate.kind is RawRevisionKind.FULL and candidate.authority is RawRevisionAuthority.BYTE_PROVEN
    ]
    applications: dict[str, RevisionApplication] = {}
    if not proven_full:
        return RevisionReplayPlan(
            logical_source_key=logical_key,
            applications=tuple(
                RevisionApplication(
                    candidate.raw_id,
                    (
                        ApplicationDecision.AMBIGUOUS
                        if candidate.authority is RawRevisionAuthority.QUARANTINED
                        else ApplicationDecision.DEFERRED
                    ),
                    None,
                    None,
                    "no unique byte-proven full baseline",
                )
                for candidate in sorted(candidates, key=lambda item: item.raw_id)
            ),
        )

    newest_generation = max(candidate.acquisition_generation for candidate in proven_full)
    newest = [candidate for candidate in proven_full if candidate.acquisition_generation == newest_generation]
    if len(newest) != 1:
        tied_ids = {candidate.raw_id for candidate in newest}
        for candidate in candidates:
            decision = ApplicationDecision.AMBIGUOUS if candidate.raw_id in tied_ids else ApplicationDecision.DEFERRED
            applications[candidate.raw_id] = RevisionApplication(
                candidate.raw_id,
                decision,
                None,
                None,
                "multiple byte-proven full baselines share the newest generation",
            )
        return _ordered_plan(logical_key, applications)

    baseline = newest[0]
    accepted = baseline
    applications[baseline.raw_id] = RevisionApplication(
        baseline.raw_id,
        ApplicationDecision.SELECTED_BASELINE,
        baseline.raw_id,
        baseline.source_revision,
        "newest unique byte-proven full baseline",
    )
    accepted_chain = [baseline.raw_id]
    for candidate in proven_full:
        if candidate.raw_id == baseline.raw_id:
            continue
        applications[candidate.raw_id] = RevisionApplication(
            candidate.raw_id,
            ApplicationDecision.SUPERSEDED,
            baseline.raw_id,
            baseline.source_revision,
            "older byte-proven full baseline",
        )

    while True:
        children = [
            candidate
            for candidate in candidates
            if candidate.kind is RawRevisionKind.APPEND
            and candidate.authority is RawRevisionAuthority.BYTE_PROVEN
            and candidate.baseline_raw_id == baseline.raw_id
            and candidate.predecessor_raw_id == accepted.raw_id
            and candidate.append_start_offset
            == (accepted.blob_size if accepted.kind is RawRevisionKind.FULL else accepted.append_end_offset)
        ]
        if not children:
            break
        if len(children) > 1:
            for child in children:
                applications[child.raw_id] = RevisionApplication(
                    child.raw_id,
                    ApplicationDecision.AMBIGUOUS,
                    accepted.raw_id,
                    accepted.source_revision,
                    "multiple byte-proven appends branch from the accepted head",
                )
            break
        child = children[0]
        applications[child.raw_id] = RevisionApplication(
            child.raw_id,
            ApplicationDecision.APPLIED_APPEND,
            child.raw_id,
            child.source_revision,
            "unique contiguous byte-proven append",
        )
        accepted = child
        accepted_chain.append(child.raw_id)

    for candidate in candidates:
        if candidate.raw_id in applications:
            continue
        decision = (
            ApplicationDecision.AMBIGUOUS
            if candidate.authority is RawRevisionAuthority.QUARANTINED
            else ApplicationDecision.DEFERRED
        )
        applications[candidate.raw_id] = RevisionApplication(
            candidate.raw_id,
            decision,
            accepted.raw_id,
            accepted.source_revision,
            "evidence is not a unique contiguous successor of the accepted head",
        )
    return _ordered_plan(logical_key, applications, accepted_chain=tuple(accepted_chain))


def _ordered_plan(
    logical_source_key: str,
    applications: dict[str, RevisionApplication],
    *,
    accepted_chain: tuple[str, ...] = (),
) -> RevisionReplayPlan:
    return RevisionReplayPlan(
        logical_source_key=logical_source_key,
        applications=tuple(applications[raw_id] for raw_id in sorted(applications)),
        accepted_chain=accepted_chain,
    )


__all__ = [
    "ApplicationDecision",
    "RevisionApplication",
    "RevisionCandidate",
    "RevisionReplayPlan",
    "plan_revision_replay",
]
