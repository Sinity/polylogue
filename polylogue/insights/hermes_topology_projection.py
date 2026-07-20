"""Provenance-preserving Hermes topology projection (fs1.14).

ATIF and (future) ATOF materialization must enrich one logical Hermes session
revision without merging raw artifacts destructively or double-counting
evidence. Every Hermes artifact class -- the state-db conversational session
(``hermes_state.py``), the ATIF trajectory export and ATOF event stream
(``hermes_spans.py``), and the verification ledger (``hermes_verification.py``)
-- is retained as its own independent, non-colliding archive session (fs1.14
fixed a prior identity collision where ATIF and ATOF shared one
``observer:<id>`` provider_session_id and silently clobbered each other, see
``hermes_spans.py``'s module docstring). This module owns the typed,
read-side composition across those independently-acquired artifacts for one
raw Hermes session id: it states, per artifact, whether evidence exists and
what fidelity it carries, without inferring parentage from proximity or
copying any message/event body across artifact boundaries.

Design discipline (mirrors ``hermes_verification_coverage.py`` /
``context.hermes_delivery_correlation``): a pure aggregator over
already-fetched ``SessionEventRecord`` rows, no I/O, unit-testable in
isolation. Callers resolve each artifact's expected session id (via
``hermes_state._qualified_session_id`` / ``hermes_spans.atif_session_provider_id``
/ ``hermes_spans.atof_session_provider_id`` /
``hermes_verification.observer_session_provider_id``) and fetch its events
(``get_session_events``) before calling in; a resolved id with ``events=None``
means that artifact's session does not exist in the archive -- the common
case for a Hermes session that only ever produced some of its possible
evidence classes, not an error.

Provenance rules this module enforces:

- **Never merge.** The projection only ever references session ids and
  counts; it never returns message/event bodies from one artifact class
  labelled as another's.
- **Unpaired traces are visible debt.** An ATIF session with no sibling ATOF
  session (or vice versa) for the same raw Hermes session id is a partial
  acquisition, not silently treated as "no observer evidence at all" --
  surfaced in ``unpaired_artifacts`` plus an explicit caveat.
- **Conflicting parent identifiers fail closed.** A subagent trajectory that
  names its own parent session as its child (a structurally impossible
  self-reference), or an ATIF-reported subagent-session-id set that shares no
  member with the independently-acquired ATOF-reported set for the same
  parent, is rendered as an explicit ``HermesTopologyConflict`` rather than
  silently trusting one source over the other.
- **Deterministic composition.** Output ordering never depends on input
  ordering (event lists are scanned but every collection in the result is
  sorted before being returned), so rebuilding this projection from the same
  retained raw evidence reproduces byte-identical output (AC: "replay and
  rebuild are idempotent").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.sources.parsers.hermes_state import HermesFidelityStatus

if TYPE_CHECKING:
    from polylogue.storage.runtime.archive.records import SessionEventRecord

HermesArtifactKind = Literal["conversational", "atif", "atof", "verification"]
_ARTIFACT_ORDER: tuple[HermesArtifactKind, ...] = ("conversational", "atif", "atof", "verification")


@dataclass(frozen=True, slots=True)
class HermesArtifactInput:
    """One artifact's expected session id and its already-fetched events, if any.

    ``session_id`` is always the caller-resolved, deterministic identity this
    artifact class would carry for the raw Hermes session id being projected
    (a producer-positive join key, never inferred from proximity/timing).
    ``events is None`` means no session was found at that id -- the artifact
    is simply absent from the archive, not an error.
    """

    session_id: str
    events: Sequence[SessionEventRecord] | None = None


class HermesArtifactObservation(ArchiveInsightModel):
    """Whether one Hermes artifact class has retained evidence, and its fidelity."""

    artifact: HermesArtifactKind
    session_id: str
    available: bool
    event_count: int = 0
    fidelity_status: HermesFidelityStatus | None = None
    caveats: tuple[str, ...] = ()


class HermesSubagentEvidenceRef(ArchiveInsightModel):
    """One producer-reported subagent-session reference, never physically linked.

    ``source_artifact`` states exactly which independently-acquired artifact
    reported this reference (ATIF's ``subagent_trajectories`` or ATOF's
    ``hermes.subagent.*`` marks) -- field-level provenance, per the design.
    """

    source_artifact: Literal["atif", "atof"]
    subagent_session_id: str
    agent_name: str | None = None
    status: str | None = None


class HermesTopologyConflict(ArchiveInsightModel):
    """An explicit, fail-closed conflict -- never silently resolved by picking a side."""

    kind: Literal["self_referential_subagent", "atif_atof_subagent_identity_mismatch"]
    detail: str
    evidence_refs: tuple[str, ...]


class HermesTopologyProjection(ArchiveInsightModel):
    """One logical Hermes session's topology, composed read-side from every retained artifact."""

    hermes_session_id: str
    artifacts: tuple[HermesArtifactObservation, ...]
    subagent_evidence: tuple[HermesSubagentEvidenceRef, ...] = ()
    unpaired_artifacts: tuple[HermesArtifactKind, ...] = ()
    conflicts: tuple[HermesTopologyConflict, ...] = ()
    caveats: tuple[str, ...] = ()


def _observe_artifact(artifact: HermesArtifactKind, artifact_input: HermesArtifactInput) -> HermesArtifactObservation:
    events = artifact_input.events
    if events is None:
        return HermesArtifactObservation(artifact=artifact, session_id=artifact_input.session_id, available=False)

    caveats: list[str] = []
    fidelity_status: HermesFidelityStatus = "exact" if events else "absent"
    if artifact == "atof":
        unpaired_scopes = sum(1 for event in events if event.event_type == "hermes_atof_unpaired_scope")
        if unpaired_scopes:
            fidelity_status = "degraded"
            caveats.append(f"{unpaired_scopes} ATOF scope(s) never observed both their start and end phase.")
    if artifact == "atif":
        # Only a genuinely unrecognized step shape (hermes_spans._events_for_step's
        # "unrecognized" branch) is fidelity debt. "observation_only" is a
        # deliberately-recognized shape (a step with an observation but no
        # tool_calls/message) -- see _events_for_step -- and must not be
        # conflated with unrecognized input just because both happen to reuse
        # the generic hermes_observer_span event type.
        unrecognized_steps = sum(
            1
            for event in events
            if event.event_type == "hermes_observer_span" and event.payload.get("shape") == "unrecognized"
        )
        if unrecognized_steps:
            fidelity_status = "degraded"
            caveats.append(f"{unrecognized_steps} ATIF step(s) matched none of the documented shapes.")
    if artifact == "verification":
        # Mirror hermes_verification.import_fidelity_declaration()'s own
        # capabilities rather than defaulting to "exact": that declaration's
        # retention_completeness capability is unconditionally "degraded"
        # whenever any hermes_verification_event evidence exists (the
        # producer prunes events older than 30 days and caps retained
        # events), and its correlation capability is "degraded" whenever any
        # event/state row carries Hermes's own session_id="default" fallback
        # (ambiguous_correlation). Upgrading either back to "exact" here
        # would silently discard fidelity debt the source parser already
        # declared.
        verification_events = sum(1 for event in events if event.event_type == "hermes_verification_event")
        if verification_events:
            fidelity_status = "degraded"
            caveats.append(
                f"{verification_events} verification event(s) reflect only what Hermes's producer "
                "currently retains (prunes events older than 30 days, caps retained events) -- not a "
                "complete historical ledger (hermes_verification.import_fidelity_declaration's "
                "retention_completeness capability)."
            )
        ambiguous_events = sum(1 for event in events if event.payload.get("ambiguous_correlation"))
        if ambiguous_events:
            fidelity_status = "degraded"
            caveats.append(
                f"{ambiguous_events} verification event(s)/state row(s) carry Hermes's own "
                "session_id='default' fallback -- ambiguous correlation, never silently trusted "
                "(hermes_verification.import_fidelity_declaration's correlation capability)."
            )

    return HermesArtifactObservation(
        artifact=artifact,
        session_id=artifact_input.session_id,
        available=True,
        event_count=len(events),
        fidelity_status=fidelity_status,
        caveats=tuple(caveats),
    )


def _atif_subagent_refs(events: Sequence[SessionEventRecord]) -> list[HermesSubagentEvidenceRef]:
    refs: list[HermesSubagentEvidenceRef] = []
    for event in events:
        if event.event_type != "hermes_subagent_span":
            continue
        subagent_session_id = event.payload.get("subagent_session_id")
        if not isinstance(subagent_session_id, str) or not subagent_session_id:
            continue
        agent_name = event.payload.get("subagent_agent_name")
        refs.append(
            HermesSubagentEvidenceRef(
                source_artifact="atif",
                subagent_session_id=subagent_session_id,
                agent_name=agent_name if isinstance(agent_name, str) and agent_name else None,
            )
        )
    return refs


def _atof_subagent_refs(events: Sequence[SessionEventRecord]) -> list[HermesSubagentEvidenceRef]:
    refs: list[HermesSubagentEvidenceRef] = []
    for event in events:
        if event.event_type != "hermes_subagent_span":
            continue
        child_session_id = event.payload.get("child_session_id")
        if not isinstance(child_session_id, str) or not child_session_id:
            continue
        status = event.payload.get("status")
        refs.append(
            HermesSubagentEvidenceRef(
                source_artifact="atof",
                subagent_session_id=child_session_id,
                status=status if isinstance(status, str) and status else None,
            )
        )
    return refs


def _detect_conflicts(
    hermes_session_id: str,
    subagent_evidence: Sequence[HermesSubagentEvidenceRef],
) -> list[HermesTopologyConflict]:
    conflicts: list[HermesTopologyConflict] = []

    self_referential = sorted(
        {ref.subagent_session_id for ref in subagent_evidence if ref.subagent_session_id == hermes_session_id}
    )
    if self_referential:
        conflicts.append(
            HermesTopologyConflict(
                kind="self_referential_subagent",
                detail=(
                    f"Hermes session {hermes_session_id} was reported as its own subagent -- a "
                    "structurally impossible parent/child identity, rendered explicit rather than "
                    "silently linked."
                ),
                evidence_refs=tuple(self_referential),
            )
        )

    atif_ids = {ref.subagent_session_id for ref in subagent_evidence if ref.source_artifact == "atif"}
    atof_ids = {ref.subagent_session_id for ref in subagent_evidence if ref.source_artifact == "atof"}
    if atif_ids and atof_ids and atif_ids.isdisjoint(atof_ids):
        conflicts.append(
            HermesTopologyConflict(
                kind="atif_atof_subagent_identity_mismatch",
                detail=(
                    f"ATIF and ATOF independently reported non-overlapping subagent-session identities "
                    f"for Hermes session {hermes_session_id} -- neither source is trusted over the other; "
                    "both evidence sets are retained as-is."
                ),
                evidence_refs=tuple(sorted(atif_ids | atof_ids)),
            )
        )

    return conflicts


def project_hermes_topology(
    hermes_session_id: str,
    *,
    conversational: HermesArtifactInput,
    atif: HermesArtifactInput,
    atof: HermesArtifactInput,
    verification: HermesArtifactInput,
) -> HermesTopologyProjection:
    """Compose every retained Hermes artifact for one raw session id, without merging.

    Every input's ``session_id`` is the caller-resolved, producer-positive
    join key for that artifact class (see ``HermesArtifactInput``); this
    function only reads structural event evidence already fetched by the
    caller (never re-queries, never infers a parent from timing/proximity).
    """
    observations = {
        "conversational": _observe_artifact("conversational", conversational),
        "atif": _observe_artifact("atif", atif),
        "atof": _observe_artifact("atof", atof),
        "verification": _observe_artifact("verification", verification),
    }

    subagent_evidence: list[HermesSubagentEvidenceRef] = []
    if atif.events is not None:
        subagent_evidence.extend(_atif_subagent_refs(atif.events))
    if atof.events is not None:
        subagent_evidence.extend(_atof_subagent_refs(atof.events))
    subagent_evidence.sort(key=lambda ref: (ref.source_artifact, ref.subagent_session_id))

    unpaired: list[HermesArtifactKind] = []
    caveats: list[str] = []
    atif_available = observations["atif"].available
    atof_available = observations["atof"].available
    if atif_available and not atof_available:
        unpaired.append("atof")
        caveats.append(
            f"Hermes session {hermes_session_id}: ATIF trajectory evidence exists with no sibling ATOF "
            "event stream -- partial artifact-family acquisition, surfaced as visible debt, not silently "
            "treated as complete."
        )
    if atof_available and not atif_available:
        unpaired.append("atif")
        caveats.append(
            f"Hermes session {hermes_session_id}: ATOF event-stream evidence exists with no sibling ATIF "
            "trajectory export -- partial artifact-family acquisition, surfaced as visible debt, not "
            "silently treated as complete."
        )

    conflicts = _detect_conflicts(hermes_session_id, subagent_evidence)
    caveats.extend(conflict.detail for conflict in conflicts)
    for observation in observations.values():
        caveats.extend(observation.caveats)

    return HermesTopologyProjection(
        hermes_session_id=hermes_session_id,
        artifacts=tuple(observations[artifact] for artifact in _ARTIFACT_ORDER),
        subagent_evidence=tuple(subagent_evidence),
        unpaired_artifacts=tuple(sorted(unpaired)),
        conflicts=tuple(conflicts),
        caveats=tuple(caveats),
    )


__all__ = [
    "HermesArtifactInput",
    "HermesArtifactKind",
    "HermesArtifactObservation",
    "HermesSubagentEvidenceRef",
    "HermesTopologyConflict",
    "HermesTopologyProjection",
    "project_hermes_topology",
]
