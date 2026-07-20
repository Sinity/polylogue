"""Tests for the Hermes topology projection (fs1.14).

Constructs real ``SessionEventRecord`` rows (the same typed row every other
Hermes read model consumes via ``get_session_events``) rather than a
hand-rolled stub, and drives ``project_hermes_topology`` -- the pure
aggregator under test -- directly, mirroring
``tests/unit/insights/test_hermes_verification_coverage.py``'s design.
"""

from __future__ import annotations

from polylogue.core.types import SessionEventId, SessionId
from polylogue.insights.hermes_topology_projection import (
    HermesArtifactInput,
    project_hermes_topology,
)
from polylogue.sources.parsers import hermes_spans
from polylogue.storage.runtime.archive.records import SessionEventRecord

_HERMES_SESSION_ID = "topology-session-1"


def _event(event_type: str, payload: dict[str, object], *, index: int = 0) -> SessionEventRecord:
    return SessionEventRecord(
        event_id=SessionEventId(f"evt-{event_type}-{index}"),
        session_id=SessionId("hermes-session:observer:atif:" + _HERMES_SESSION_ID),
        origin="hermes-session",
        event_index=index,
        event_type=event_type,
        payload=payload,
    )


def _artifact(session_id: str, events: list[SessionEventRecord] | None) -> HermesArtifactInput:
    return HermesArtifactInput(session_id=session_id, events=events)


def test_projection_is_absent_for_every_artifact_when_nothing_was_ingested() -> None:
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), None),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), None),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    assert projection.hermes_session_id == _HERMES_SESSION_ID
    assert all(not observation.available for observation in projection.artifacts)
    assert projection.unpaired_artifacts == ()
    assert projection.conflicts == ()
    assert projection.subagent_evidence == ()


def test_atif_present_without_atof_sibling_becomes_visible_unpaired_debt() -> None:
    """AC: 'an unpaired trace becomes visible debt.'

    Mutation: drop the unpaired-artifact detection (always return
    ``unpaired_artifacts=()``) and this test's non-empty assertion fails.
    """
    atif_events = [_event("hermes_llm_request_span", {"model": "example"})]
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), atif_events),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), None),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    assert projection.unpaired_artifacts == ("atof",)
    assert any("ATIF trajectory evidence exists with no sibling ATOF" in caveat for caveat in projection.caveats)

    atif_obs = next(obs for obs in projection.artifacts if obs.artifact == "atif")
    assert atif_obs.available is True
    assert atif_obs.event_count == 1
    atof_obs = next(obs for obs in projection.artifacts if obs.artifact == "atof")
    assert atof_obs.available is False


def test_atof_present_without_atif_sibling_becomes_visible_unpaired_debt() -> None:
    atof_events = [_event("hermes_context_span", {"context_event": "hermes.session.start"})]
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), None),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), atof_events),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    assert projection.unpaired_artifacts == ("atif",)
    assert any("ATOF event-stream evidence exists with no sibling ATIF" in caveat for caveat in projection.caveats)


def test_both_atif_and_atof_present_is_not_unpaired() -> None:
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(
            hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), [_event("hermes_llm_request_span", {})]
        ),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), [_event("hermes_context_span", {})]),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    assert projection.unpaired_artifacts == ()


def test_atof_unpaired_scope_events_degrade_atof_fidelity_without_affecting_atif() -> None:
    atof_events = [
        _event("hermes_llm_request_span", {}),
        _event("hermes_atof_unpaired_scope", {"event_uuid": "x"}, index=1),
    ]
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), None),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), atof_events),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    atof_obs = next(obs for obs in projection.artifacts if obs.artifact == "atof")
    assert atof_obs.fidelity_status == "degraded"
    assert any("never observed both their start and end phase" in caveat for caveat in atof_obs.caveats)


def test_subagent_evidence_is_surfaced_from_both_artifacts_with_source_provenance() -> None:
    """AC: 'derive session links and subagent topology edges from producer-positive
    IDs only' -- surfaced as evidence refs, never a physical session merge."""
    atif_events = [
        _event(
            "hermes_subagent_span",
            {"subagent_session_id": "child-a", "subagent_agent_name": "Docs Agent"},
        )
    ]
    atof_events = [
        _event(
            "hermes_subagent_span",
            {"child_session_id": "child-a", "status": "completed"},
            index=1,
        )
    ]
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), atif_events),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), atof_events),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    assert len(projection.subagent_evidence) == 2
    by_source = {ref.source_artifact: ref for ref in projection.subagent_evidence}
    assert by_source["atif"].subagent_session_id == "child-a"
    assert by_source["atif"].agent_name == "Docs Agent"
    assert by_source["atof"].subagent_session_id == "child-a"
    assert by_source["atof"].status == "completed"
    # Corroborating (same id from both independently-acquired sources) is not a conflict.
    assert projection.conflicts == ()


def test_self_referential_subagent_fails_closed_as_explicit_conflict() -> None:
    """AC: 'a conflicting parent identifier fails closed or renders an explicit conflict.'

    Mutation: drop the self-reference check and this test's non-empty
    ``conflicts`` assertion fails while the bogus link would otherwise be
    silently treated as ordinary subagent evidence.
    """
    atif_events = [_event("hermes_subagent_span", {"subagent_session_id": _HERMES_SESSION_ID})]
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), atif_events),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), None),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    assert len(projection.conflicts) == 1
    conflict = projection.conflicts[0]
    assert conflict.kind == "self_referential_subagent"
    assert conflict.evidence_refs == (_HERMES_SESSION_ID,)
    assert conflict.detail in projection.caveats


def test_disjoint_atif_atof_subagent_identities_fail_closed_as_explicit_conflict() -> None:
    atif_events = [_event("hermes_subagent_span", {"subagent_session_id": "child-atif-only"})]
    atof_events = [_event("hermes_subagent_span", {"child_session_id": "child-atof-only"}, index=1)]
    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", None),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), atif_events),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), atof_events),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
    )

    mismatch = [c for c in projection.conflicts if c.kind == "atif_atof_subagent_identity_mismatch"]
    assert len(mismatch) == 1
    assert set(mismatch[0].evidence_refs) == {"child-atif-only", "child-atof-only"}


def test_projection_is_idempotent_and_deterministic_across_repeated_calls() -> None:
    """AC: 'replay and rebuild are idempotent.'

    Same retained raw evidence, called twice, must produce byte-identical
    output -- including stable ordering that does not depend on how the
    caller happened to order its event lists.
    """
    atif_events = [
        _event("hermes_subagent_span", {"subagent_session_id": "child-b"}, index=0),
        _event("hermes_subagent_span", {"subagent_session_id": "child-a"}, index=1),
        _event("hermes_llm_request_span", {"model": "x"}, index=2),
    ]
    atof_events = [_event("hermes_context_span", {"context_event": "hermes.session.start"})]

    def build() -> object:
        return project_hermes_topology(
            _HERMES_SESSION_ID,
            conversational=_artifact("conv-id", None),
            atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), list(atif_events)),
            atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), list(atof_events)),
            verification=_artifact("verification:" + _HERMES_SESSION_ID, None),
        )

    first = build()
    second = build()
    assert first.model_dump(mode="json") == second.model_dump(mode="json")  # type: ignore[attr-defined]
    # Subagent evidence ordering is stable regardless of source-event order.
    assert [ref.subagent_session_id for ref in first.subagent_evidence] == ["child-a", "child-b"]  # type: ignore[attr-defined]


def test_per_artifact_event_counts_never_cross_contaminate() -> None:
    """AC: '...no duplicate messages/actions.'

    Each artifact's ``event_count`` must equal exactly the length of the
    event list the caller supplied for THAT artifact -- never the union or
    sum across artifacts. This is the concrete form "no duplication" takes
    for a read-side composition that holds four independently-sized event
    lists: nothing from ATIF's event list may leak into ATOF's count (or
    vice versa) even though both describe the same raw Hermes session id.

    Mutation: swap which observation two artifacts are assigned to (or sum
    counts across artifacts instead of keeping them independent) and the
    distinct-length assertions below fail.
    """
    conversational_events = [_event("hermes_identity", {"raw_session_id": _HERMES_SESSION_ID})]
    atif_events = [
        _event("hermes_llm_request_span", {"message_char_len": 42}, index=0),
        _event("hermes_tool_execution_span", {"function_name": "read_file"}, index=1),
    ]
    atof_events = [_event("hermes_context_span", {}, index=0)]
    verification_events = [
        _event("hermes_verification_event", {"status": "passed"}, index=0),
        _event("hermes_verification_event", {"status": "failed"}, index=1),
        _event("hermes_verification_state", {"changed_paths": ["a.py"]}, index=2),
    ]

    projection = project_hermes_topology(
        _HERMES_SESSION_ID,
        conversational=_artifact("conv-id", conversational_events),
        atif=_artifact(hermes_spans.atif_session_provider_id(_HERMES_SESSION_ID), atif_events),
        atof=_artifact(hermes_spans.atof_session_provider_id(_HERMES_SESSION_ID), atof_events),
        verification=_artifact("verification:" + _HERMES_SESSION_ID, verification_events),
    )

    counts_by_artifact = {observation.artifact: observation.event_count for observation in projection.artifacts}
    assert counts_by_artifact == {
        "conversational": 1,
        "atif": 2,
        "atof": 1,
        "verification": 3,
    }
    # Every artifact reports its own resolved session id, not a shared one --
    # a physical merge would collapse these onto a single identity.
    session_ids_by_artifact = {observation.artifact: observation.session_id for observation in projection.artifacts}
    assert len(set(session_ids_by_artifact.values())) == 4
