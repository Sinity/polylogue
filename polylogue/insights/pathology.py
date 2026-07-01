"""Deterministic agent-workflow pathology detectors (#2383).

Rule-based, versioned detectors that mine recurring failure-mode pathologies from
the typed run projection (:class:`polylogue.insights.run_projection.RunProjection`)
that the session digest already carries. No I/O, no LLM-as-judge — every finding
is reproducible from the same evidence and drillable through ``EvidenceRef``s.

Detectors operate over structured run-projection evidence (#2482):

- ``wasted_loop`` — repeated ``test_failed``/``command_failed`` outcome events in
  a run, the edit→test-fail→edit cycle that burns turns without converging.
- ``stale_context`` — a continuation/subagent context boundary inherited context
  in a lossy mode (``summary``/``prefix``), the stale-context-after-compaction
  pattern.

The former ``missed_review`` detector was removed (#2482): its entire basis was
the prose-mined ``review_*`` events, which fabricated review lifecycle state from
unverified text patterns. Review/PR/CI state is external truth owned by
git/GitHub, not synthesizable from transcript prose.

The detectors feed the postmortem report's pathology fields. Emitting
``Assertion(kind=pathology)`` candidate rows and the cross-archive distribution
query are a follow-up slice (they require a new ``AssertionKind`` member, a
schema-CHECK change, and the assertion candidate lifecycle).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from polylogue.core.refs import EvidenceRef
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.run_projection import ObservedEvent, RunProjection

# Bump when a detector's rule changes so cached/rebuilt output is comparable.
PATHOLOGY_DETECTOR_VERSION = 4

PathologyKind = Literal["wasted_loop", "stale_context"]
PathologySeverity = Literal["low", "medium", "high"]

# Event kinds that prove the failure streak converged before later failures.
_SUCCESS_EVENT_KINDS = frozenset({"test_passed", "command_succeeded"})
# Command-like handler kinds whose failure can represent an edit/test/debug
# loop. File reads, edits, todos, and other generic tool failures are execution
# noise for this detector unless old payloads only expose a concrete command.
_DIAGNOSTIC_FAILURE_HANDLER_KINDS = frozenset({"shell", "git", "github", "test"})
# Continuation boundaries where a *resumed* run re-inherits prior context. A
# ``subagent_start`` boundary is intentionally excluded: a subagent receiving a
# focused summary/prefix is by design, not a stale-context pathology — only a
# resume-after-compaction that drops context counts.
_CONTINUATION_BOUNDARIES = frozenset({"resume"})
# Inheritance modes that carry only a lossy slice of the prior context.
_LOSSY_INHERITANCE = frozenset({"summary", "prefix"})

_MAX_FINDING_EVIDENCE = 5


class PathologyFinding(ArchiveInsightModel):
    """One detected pathology occurrence, drillable through its evidence."""

    kind: PathologyKind
    session_id: str
    severity: PathologySeverity
    detail: str
    occurrence_count: int = 1
    evidence_refs: tuple[EvidenceRef, ...] = ()
    detector_version: int = PATHOLOGY_DETECTOR_VERSION


class PathologyReport(ArchiveInsightModel):
    """Aggregate pathology distribution across a set of run projections."""

    findings: tuple[PathologyFinding, ...] = ()
    counts_by_kind: dict[str, int] = {}
    session_count: int = 0
    detector_version: int = PATHOLOGY_DETECTOR_VERSION


def _evidence_from_events(events: Sequence[ObservedEvent]) -> tuple[EvidenceRef, ...]:
    """Bounded, de-duplicated evidence-ref sample from triggering events."""
    seen: set[tuple[str, str | None]] = set()
    refs: list[EvidenceRef] = []
    for event in events:
        for ref in event.evidence_refs:
            key = (ref.session_id, ref.message_id)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
            if len(refs) >= _MAX_FINDING_EVIDENCE:
                return tuple(refs)
    return tuple(refs)


def _wasted_loop_severity(count: int) -> PathologySeverity:
    if count >= 7:
        return "high"
    if count >= 4:
        return "medium"
    return "low"


def _detect_wasted_loops(projection: RunProjection) -> list[PathologyFinding]:
    """Repeated failed test/command turns with no structured success between."""
    longest_streak: list[ObservedEvent] = []
    current_streak: list[ObservedEvent] = []
    current_signature: str | None = None
    for event in projection.events:
        if _is_diagnostic_failure_event(event):
            signature = _diagnostic_failure_signature(event)
            if current_signature is not None and signature != current_signature:
                current_streak = []
            current_streak.append(event)
            current_signature = signature
            if len(current_streak) > len(longest_streak):
                longest_streak = list(current_streak)
        elif event.kind in _SUCCESS_EVENT_KINDS:
            current_streak = []
            current_signature = None
    if len(longest_streak) < 2:
        return []
    return [
        PathologyFinding(
            kind="wasted_loop",
            session_id=projection.session_id,
            severity=_wasted_loop_severity(len(longest_streak)),
            detail=(
                f"{len(longest_streak)} repeated failed diagnostic turn(s) for "
                f"{_diagnostic_failure_signature(longest_streak[0])!r} without a structured success"
            ),
            occurrence_count=len(longest_streak),
            evidence_refs=_evidence_from_events(longest_streak),
        )
    ]


def _diagnostic_failure_signature(event: ObservedEvent) -> str:
    if event.command and event.command.strip():
        return " ".join(event.command.split())
    if event.tool_name and event.tool_name.strip():
        return f"{event.kind}:{event.handler_kind or ''}:{event.tool_name.strip()}"
    return f"{event.kind}:{event.handler_kind or ''}"


def _is_diagnostic_failure_event(event: ObservedEvent) -> bool:
    if event.kind == "test_failed":
        return True
    if event.kind != "command_failed":
        return False
    if event.handler_kind in _DIAGNOSTIC_FAILURE_HANDLER_KINDS:
        return True
    # Backward-compatible path for already-materialized v3 payloads that predate
    # handler metadata. A concrete command is still a shell-like diagnostic
    # signal; a bare "Read failed" / "Edit failed" summary is not.
    return bool(event.command and event.command.strip())


def _detect_stale_context(projection: RunProjection) -> list[PathologyFinding]:
    """Resume (post-compaction) boundary that re-inherited context lossily."""
    lossy = [
        snapshot
        for snapshot in projection.context_snapshots
        if snapshot.boundary in _CONTINUATION_BOUNDARIES and snapshot.inheritance_mode in _LOSSY_INHERITANCE
    ]
    if not lossy:
        return []
    seen: set[tuple[str, str | None]] = set()
    refs: list[EvidenceRef] = []
    for snapshot in lossy:
        for ref in snapshot.evidence_refs:
            key = (ref.session_id, ref.message_id)
            if key not in seen:
                seen.add(key)
                refs.append(ref)
            if len(refs) >= _MAX_FINDING_EVIDENCE:
                break
    modes = sorted({snapshot.inheritance_mode for snapshot in lossy})
    return [
        PathologyFinding(
            kind="stale_context",
            session_id=projection.session_id,
            severity="medium",
            detail=(f"{len(lossy)} resume boundary(ies) re-inherited context in lossy mode(s): {', '.join(modes)}"),
            occurrence_count=len(lossy),
            evidence_refs=tuple(refs),
        )
    ]


def detect_session_pathologies(projection: RunProjection) -> list[PathologyFinding]:
    """Run every v1 detector against a single session's run projection."""
    findings: list[PathologyFinding] = []
    findings.extend(_detect_wasted_loops(projection))
    findings.extend(_detect_stale_context(projection))
    return findings


def _finding_sort_key(finding: PathologyFinding) -> tuple[str, str, str]:
    return (finding.kind, finding.session_id, finding.detail)


def compile_pathology_report(projections: Iterable[RunProjection]) -> PathologyReport:
    """Aggregate pathology findings across run projections into a report.

    Deterministic: findings are sorted by ``(kind, session_id, detail)`` so a
    rebuild over identical evidence produces byte-identical output.
    """
    findings: list[PathologyFinding] = []
    session_count = 0
    for projection in projections:
        session_count += 1
        findings.extend(detect_session_pathologies(projection))
    findings.sort(key=_finding_sort_key)
    counts_by_kind: dict[str, int] = {}
    for finding in findings:
        counts_by_kind[finding.kind] = counts_by_kind.get(finding.kind, 0) + 1
    return PathologyReport(
        findings=tuple(findings),
        counts_by_kind=counts_by_kind,
        session_count=session_count,
        detector_version=PATHOLOGY_DETECTOR_VERSION,
    )


__all__ = [
    "PATHOLOGY_DETECTOR_VERSION",
    "PathologyFinding",
    "PathologyKind",
    "PathologyReport",
    "PathologySeverity",
    "compile_pathology_report",
    "detect_session_pathologies",
]
