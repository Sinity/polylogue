"""Tests for the deterministic agent-workflow pathology detectors (#2383)."""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.pathology import (
    PATHOLOGY_DETECTOR_VERSION,
    compile_pathology_report,
    detect_session_pathologies,
)
from polylogue.insights.run_projection import (
    ContextBoundary,
    ContextInheritanceMode,
    ContextSnapshot,
    ObservedEvent,
    ObservedEventKind,
    ProjectedRun,
    RunProjection,
)


def _ev(session_id: str = "s1") -> tuple[EvidenceRef, ...]:
    return (EvidenceRef(session_id=session_id),)


def _run(session_id: str = "s1") -> ProjectedRun:
    return ProjectedRun(run_ref=ObjectRef.parse(f"run:{session_id}"), evidence_refs=_ev(session_id))


def _event(kind: ObservedEventKind, *, session_id: str = "s1", delivery: str = "observed") -> ObservedEvent:
    return ObservedEvent(
        event_ref=ObjectRef.parse(f"observed-event:{session_id}:{kind}"),
        kind=kind,
        run_ref=ObjectRef.parse(f"run:{session_id}"),
        summary=f"{kind} event",
        delivery_state=delivery,  # type: ignore[arg-type]
        evidence_refs=_ev(session_id),
    )


def _snapshot(boundary: ContextBoundary, mode: ContextInheritanceMode, *, session_id: str = "s1") -> ContextSnapshot:
    return ContextSnapshot(
        snapshot_ref=ObjectRef.parse(f"context-snapshot:{session_id}:{boundary}"),
        run_ref=ObjectRef.parse(f"run:{session_id}"),
        boundary=boundary,
        inheritance_mode=mode,
        evidence_refs=_ev(session_id),
    )


def _projection(
    session_id: str = "s1",
    *,
    events: Sequence[ObservedEvent] = (),
    snapshots: Sequence[ContextSnapshot] = (),
) -> RunProjection:
    return RunProjection(
        session_id=session_id,
        runs=(_run(session_id),),
        events=tuple(events),
        context_snapshots=tuple(snapshots),
    )


# ── wasted_loop ──────────────────────────────────────────────────────


def test_wasted_loop_detected_on_repeated_failures() -> None:
    proj = _projection(events=[_event("test_failed"), _event("check_failed"), _event("test_failed")])
    findings = [f for f in detect_session_pathologies(proj) if f.kind == "wasted_loop"]
    assert len(findings) == 1
    assert findings[0].occurrence_count == 3
    assert findings[0].evidence_refs


def test_wasted_loop_not_flagged_on_single_failure() -> None:
    proj = _projection(events=[_event("test_failed"), _event("test_passed")])
    assert [f for f in detect_session_pathologies(proj) if f.kind == "wasted_loop"] == []


def test_wasted_loop_severity_scales_with_count() -> None:
    low = _projection(events=[_event("test_failed")] * 2)
    high = _projection(events=[_event("check_failed")] * 8)
    assert next(f for f in detect_session_pathologies(low) if f.kind == "wasted_loop").severity == "low"
    assert next(f for f in detect_session_pathologies(high) if f.kind == "wasted_loop").severity == "high"


# ── missed_review ────────────────────────────────────────────────────


def test_missed_review_detected_when_unaddressed() -> None:
    proj = _projection(events=[_event("review_posted"), _event("review_seen_by_tool")])
    findings = [f for f in detect_session_pathologies(proj) if f.kind == "missed_review"]
    assert len(findings) == 1
    assert findings[0].severity == "high"


def test_missed_review_cleared_by_explicit_action_event() -> None:
    proj = _projection(events=[_event("review_posted"), _event("review_acted_on")])
    assert [f for f in detect_session_pathologies(proj) if f.kind == "missed_review"] == []


def test_missed_review_cleared_by_delivery_state() -> None:
    proj = _projection(events=[_event("review_posted", delivery="acted_on")])
    assert [f for f in detect_session_pathologies(proj) if f.kind == "missed_review"] == []


# ── stale_context ────────────────────────────────────────────────────


def test_stale_context_detected_on_lossy_resume() -> None:
    proj = _projection(snapshots=[_snapshot("resume", "summary")])
    findings = [f for f in detect_session_pathologies(proj) if f.kind == "stale_context"]
    assert len(findings) == 1
    assert "summary" in findings[0].detail


def test_stale_context_not_flagged_on_clean_resume() -> None:
    proj = _projection(snapshots=[_snapshot("resume", "clean")])
    assert [f for f in detect_session_pathologies(proj) if f.kind == "stale_context"] == []


def test_stale_context_excludes_normal_subagent_dispatch() -> None:
    # A subagent receiving a focused summary/prefix is by design, not a pathology.
    proj = _projection(snapshots=[_snapshot("subagent_start", "summary")])
    assert [f for f in detect_session_pathologies(proj) if f.kind == "stale_context"] == []


# ── aggregation / determinism ────────────────────────────────────────


def test_compile_report_counts_and_versions() -> None:
    a = _projection("s1", events=[_event("test_failed", session_id="s1")] * 2)
    b = _projection("s2", events=[_event("review_posted", session_id="s2")])
    report = compile_pathology_report([a, b])
    assert report.session_count == 2
    assert report.counts_by_kind == {"missed_review": 1, "wasted_loop": 1}
    assert report.detector_version == PATHOLOGY_DETECTOR_VERSION


def test_compile_report_is_deterministic() -> None:
    projs = [
        _projection("s2", events=[_event("review_posted", session_id="s2")]),
        _projection("s1", events=[_event("test_failed", session_id="s1")] * 3),
    ]
    first = compile_pathology_report(projs).model_dump(mode="json")
    second = compile_pathology_report(projs).model_dump(mode="json")
    assert first == second
    # findings are sorted by (kind, session_id, detail) regardless of input order
    kinds = [f["kind"] for f in first["findings"]]
    assert kinds == sorted(kinds)


def test_clean_projection_yields_no_findings() -> None:
    proj = _projection(events=[_event("test_passed"), _event("session_started")])
    assert detect_session_pathologies(proj) == []
