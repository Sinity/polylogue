"""Distilled agent-session postmortem bundle (#2380).

One compact, shareable artifact over a matched session scope. Every headline
metric carries at least one drillable :class:`EvidenceRef`. The ``failure_mode``
and ``wasted_loop`` fields are populated by the #2383 pathology detectors
(:mod:`polylogue.insights.pathology`) run over the recovery-digest run
projections; they report ``detected``/``clean``/``unavailable`` rather than
fabricating a signal. ``longest_tool_gap`` still degrades in v0 because the
session profile and recovery digest do not carry per-tool-call timestamps.

The aggregator :func:`compile_postmortem_bundle` is pure: it consumes
already-fetched profiles and digests and performs no I/O, so it is unit-testable
in isolation. The owning API method (``Polylogue.postmortem_bundle``) does the
fetching and constructs the :class:`PostmortemScope`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from polylogue.core.refs import EvidenceRef
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.pathology import PathologyFinding, compile_pathology_report

if TYPE_CHECKING:
    from polylogue.archive.session.models import SessionProfile
    from polylogue.insights.transforms import RecoveryDigest

POSTMORTEM_SCHEMA_VERSION = 2

# Bounded number of evidence refs attached to an aggregate metric so a
# whole-archive scope does not emit an unbounded ref list.
_MAX_AGGREGATE_EVIDENCE = 5

# Reasons used when a field degrades honestly instead of fabricating a value.
_PATHOLOGY_UNAVAILABLE_REASON = (
    "no run projection available in scope; pathology detection needs recovery-digest evidence"
)
_TOOL_GAP_REASON = (
    "per-tool-call timestamps are not present in the session profile or recovery "
    "digest; longest_tool_gap is not cheaply derivable in v0"
)


class PostmortemScope(ArchiveInsightModel):
    """The matched window the bundle was computed over."""

    since: str | None = None
    until: str | None = None
    query: str | None = None
    matched_session_count: int = 0
    analyzed_session_count: int = 0
    truncated: bool = False
    dropped_session_count: int = 0


class SessionCountMetric(ArchiveInsightModel):
    """Number of analyzed sessions, with a bounded sample of refs."""

    count: int
    evidence_refs: tuple[EvidenceRef, ...]


class TokenLanes(ArchiveInsightModel):
    """Differentiated token lanes — never an undifferentiated ``tokens`` total."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class CostMetric(ArchiveInsightModel):
    """Aggregate spend with labelled token lanes (#2380 AC4)."""

    total_cost_usd: float
    cost_is_estimated: bool
    tokens: TokenLanes
    evidence_refs: tuple[EvidenceRef, ...]


class WallclockSpanMetric(ArchiveInsightModel):
    """Real per-session wall time, summed and spanned.

    ``span_ms`` is the calendar span from the earliest ``first_message_at`` to
    the latest ``last_message_at``. ``summed_wall_duration_ms`` sums the real
    per-session ``wall_duration_ms`` and is distinct from the calendar span
    (sessions may overlap or have idle gaps between them).
    """

    earliest_first_message_at: str | None
    latest_last_message_at: str | None
    span_ms: int | None
    summed_wall_duration_ms: int
    evidence_refs: tuple[EvidenceRef, ...]


class RepoTouchedMetric(ArchiveInsightModel):
    """One repo seen in scope and how many sessions touched it."""

    repo: str
    session_count: int
    evidence_refs: tuple[EvidenceRef, ...]


class TopExpensiveSessionMetric(ArchiveInsightModel):
    """The single most expensive session in scope."""

    status: Literal["ok", "no_signal"]
    session_id: str | None = None
    title: str | None = None
    total_cost_usd: float = 0.0
    cost_is_estimated: bool = False
    evidence_refs: tuple[EvidenceRef, ...] = ()
    reason: str | None = None


class SubagentBranchMetric(ArchiveInsightModel):
    """Count of subagent runs across analyzed sessions."""

    count: int
    evidence_refs: tuple[EvidenceRef, ...]


class DegradedField(ArchiveInsightModel):
    """A headline field with no signal — null value, explicit status, reason.

    Used for ``longest_tool_gap`` (no per-tool-call timing in v0).
    """

    value: None = None
    status: Literal["no_signal", "unavailable"]
    reason: str


class PathologyField(ArchiveInsightModel):
    """A pathology headline field populated by the #2383 detectors.

    ``status`` is ``detected`` when one or more findings exist, ``clean`` when the
    detectors ran but found nothing, and ``unavailable`` when there was no run
    projection to analyze. ``count`` is the number of findings; ``by_kind`` and
    ``detail`` describe the distribution; ``evidence_refs`` drill into examples.
    """

    status: Literal["detected", "clean", "unavailable"]
    count: int = 0
    detail: str = ""
    by_kind: dict[str, int] = {}
    evidence_refs: tuple[EvidenceRef, ...] = ()


class PostmortemBundle(ArchiveInsightModel):
    """The distilled, shareable postmortem artifact for a session scope."""

    schema_version: int = POSTMORTEM_SCHEMA_VERSION
    scope: PostmortemScope
    session_count: SessionCountMetric
    wallclock_span: WallclockSpanMetric
    estimated_cost: CostMetric
    repos_touched: tuple[RepoTouchedMetric, ...]
    top_expensive_session: TopExpensiveSessionMetric
    subagent_branch_count: SubagentBranchMetric
    longest_tool_gap: DegradedField
    wasted_loop: PathologyField
    failure_mode: PathologyField


def _pathology_field(findings: Sequence[PathologyFinding]) -> PathologyField:
    """Build a postmortem pathology field from detector findings."""
    if not findings:
        return PathologyField(status="clean", detail="detectors ran; no pathology found")
    by_kind: dict[str, int] = {}
    refs: list[EvidenceRef] = []
    seen: set[tuple[str, str | None]] = set()
    for finding in findings:
        by_kind[finding.kind] = by_kind.get(finding.kind, 0) + 1
        for ref in finding.evidence_refs:
            key = (ref.session_id, ref.message_id)
            if key not in seen and len(refs) < _MAX_AGGREGATE_EVIDENCE:
                seen.add(key)
                refs.append(ref)
    detail = ", ".join(f"{kind}={count}" for kind, count in sorted(by_kind.items()))
    return PathologyField(
        status="detected",
        count=len(findings),
        detail=detail,
        by_kind=by_kind,
        evidence_refs=tuple(refs),
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def compile_postmortem_bundle(
    profiles: Sequence[SessionProfile],
    digests: Mapping[str, RecoveryDigest],
    *,
    scope: PostmortemScope,
) -> PostmortemBundle:
    """Pure aggregator: build a :class:`PostmortemBundle` from fetched data.

    No I/O. ``profiles`` are the hydrated session profiles in scope; ``digests``
    maps ``session_id`` to its recovery digest (a subset of profiles is fine —
    digests may be missing for sessions without one).
    """

    session_refs = tuple(EvidenceRef(session_id=p.session_id) for p in profiles)

    # --- session_count -------------------------------------------------------
    session_count = SessionCountMetric(
        count=len(profiles),
        evidence_refs=session_refs[:_MAX_AGGREGATE_EVIDENCE],
    )

    # --- wallclock_span ------------------------------------------------------
    earliest_profile: SessionProfile | None = None
    latest_profile: SessionProfile | None = None
    summed_wall_ms = 0
    for profile in profiles:
        summed_wall_ms += max(int(profile.wall_duration_ms), 0)
        if profile.first_message_at is not None and (
            earliest_profile is None
            or earliest_profile.first_message_at is None
            or profile.first_message_at < earliest_profile.first_message_at
        ):
            earliest_profile = profile
        if profile.last_message_at is not None and (
            latest_profile is None
            or latest_profile.last_message_at is None
            or profile.last_message_at > latest_profile.last_message_at
        ):
            latest_profile = profile

    earliest_first = earliest_profile.first_message_at if earliest_profile is not None else None
    latest_last = latest_profile.last_message_at if latest_profile is not None else None
    span_ms: int | None = None
    if earliest_first is not None and latest_last is not None:
        span_ms = max(int((latest_last - earliest_first).total_seconds() * 1000), 0)
    span_refs: list[EvidenceRef] = []
    for boundary in (earliest_profile, latest_profile):
        if boundary is not None:
            ref = EvidenceRef(session_id=boundary.session_id)
            if ref not in span_refs:
                span_refs.append(ref)
    wallclock_span = WallclockSpanMetric(
        earliest_first_message_at=_iso(earliest_first),
        latest_last_message_at=_iso(latest_last),
        span_ms=span_ms,
        summed_wall_duration_ms=summed_wall_ms,
        # Fall back to a session ref so a non-zero summed_wall_duration_ms stays
        # drillable even when no boundary timestamp was available.
        evidence_refs=tuple(span_refs) if span_refs else session_refs[:1],
    )

    # --- estimated_cost + token lanes ---------------------------------------
    total_cost = 0.0
    any_estimated = False
    input_tokens = output_tokens = cache_read = cache_write = 0
    cost_bearing_refs: list[EvidenceRef] = []
    for profile in profiles:
        total_cost += float(profile.total_cost_usd)
        any_estimated = any_estimated or bool(profile.cost_is_estimated)
        input_tokens += int(profile.total_input_tokens)
        output_tokens += int(profile.total_output_tokens)
        cache_read += int(profile.total_cache_read_tokens)
        cache_write += int(profile.total_cache_write_tokens)
        if profile.total_cost_usd > 0 and len(cost_bearing_refs) < _MAX_AGGREGATE_EVIDENCE:
            cost_bearing_refs.append(EvidenceRef(session_id=profile.session_id))
    cost_refs = tuple(cost_bearing_refs) if cost_bearing_refs else session_refs[:1]
    estimated_cost = CostMetric(
        total_cost_usd=round(total_cost, 6),
        cost_is_estimated=any_estimated,
        tokens=TokenLanes(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
        evidence_refs=cost_refs,
    )

    # --- repos_touched -------------------------------------------------------
    repo_sessions: dict[str, list[str]] = {}
    for profile in profiles:
        for repo in profile.repo_names:
            repo_sessions.setdefault(repo, []).append(profile.session_id)
    repos_touched = tuple(
        RepoTouchedMetric(
            repo=repo,
            session_count=len(sessions),
            evidence_refs=tuple(EvidenceRef(session_id=sid) for sid in sessions[:_MAX_AGGREGATE_EVIDENCE]),
        )
        for repo, sessions in sorted(repo_sessions.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    )

    # --- top_expensive_session ----------------------------------------------
    cost_ranked = [p for p in profiles if p.total_cost_usd > 0]
    if cost_ranked:
        top = max(cost_ranked, key=lambda p: (p.total_cost_usd, p.session_id))
        top_expensive_session = TopExpensiveSessionMetric(
            status="ok",
            session_id=top.session_id,
            title=top.title,
            total_cost_usd=round(float(top.total_cost_usd), 6),
            cost_is_estimated=bool(top.cost_is_estimated),
            evidence_refs=(EvidenceRef(session_id=top.session_id),),
        )
    else:
        top_expensive_session = TopExpensiveSessionMetric(
            status="no_signal",
            reason="no session in scope carries a positive cost figure",
            evidence_refs=session_refs[:1],
        )

    # --- subagent_branch_count ----------------------------------------------
    subagent_total = 0
    subagent_refs: list[EvidenceRef] = []
    for profile in profiles:
        digest = digests.get(profile.session_id)
        if digest is None:
            continue
        branch_count = sum(1 for run in digest.run_projection.runs if run.role == "subagent")
        if branch_count:
            subagent_total += branch_count
            if len(subagent_refs) < _MAX_AGGREGATE_EVIDENCE:
                subagent_refs.append(EvidenceRef(session_id=profile.session_id))
    subagent_branch_count = SubagentBranchMetric(
        count=subagent_total,
        evidence_refs=tuple(subagent_refs) if subagent_refs else session_refs[:1],
    )

    # --- pathology detection (#2383) ----------------------------------------
    # The recovery digests carry the typed run projection the detectors need.
    projections = [digests[p.session_id].run_projection for p in profiles if p.session_id in digests]
    if projections:
        report = compile_pathology_report(projections)
        wasted_loop = _pathology_field([f for f in report.findings if f.kind == "wasted_loop"])
        failure_mode = _pathology_field([f for f in report.findings if f.kind == "stale_context"])
    else:
        wasted_loop = PathologyField(status="unavailable", detail=_PATHOLOGY_UNAVAILABLE_REASON)
        failure_mode = PathologyField(status="unavailable", detail=_PATHOLOGY_UNAVAILABLE_REASON)

    return PostmortemBundle(
        scope=scope,
        session_count=session_count,
        wallclock_span=wallclock_span,
        estimated_cost=estimated_cost,
        repos_touched=repos_touched,
        top_expensive_session=top_expensive_session,
        subagent_branch_count=subagent_branch_count,
        longest_tool_gap=DegradedField(status="unavailable", reason=_TOOL_GAP_REASON),
        wasted_loop=wasted_loop,
        failure_mode=failure_mode,
    )


def _fmt_degraded(field: DegradedField) -> str:
    return f"{field.status} ({field.reason})"


def _fmt_pathology(field: PathologyField) -> str:
    if field.status == "detected":
        return f"detected ({field.count}: {field.detail})"
    if field.status == "clean":
        return "clean (no pathology found)"
    return f"unavailable ({field.detail})"


def _format_ref(ref: EvidenceRef) -> str:
    parts = [ref.session_id]
    if ref.message_id is not None:
        parts.append(ref.message_id)
        if ref.block_index is not None:
            parts.append(str(ref.block_index))
    return "::".join(parts)


def _evidence_index(bundle: PostmortemBundle) -> list[tuple[str, tuple[EvidenceRef, ...]]]:
    """Collect (field, refs) pairs so the shared artifacts stay drillable."""
    pairs: list[tuple[str, tuple[EvidenceRef, ...]]] = [
        ("session_count", bundle.session_count.evidence_refs),
        ("wallclock_span", bundle.wallclock_span.evidence_refs),
        ("estimated_cost", bundle.estimated_cost.evidence_refs),
        ("top_expensive_session", bundle.top_expensive_session.evidence_refs),
        ("subagent_branch_count", bundle.subagent_branch_count.evidence_refs),
        *((f"repos_touched[{repo.repo}]", repo.evidence_refs) for repo in bundle.repos_touched),
    ]
    return [(name, refs) for name, refs in pairs if refs]


def render_postmortem_plain(bundle: PostmortemBundle) -> str:
    """Render the bundle as a compact plain-text artifact.

    Pure: derives entirely from ``bundle`` so plain and markdown render from the
    same payload object.
    """

    scope = bundle.scope
    cost = bundle.estimated_cost
    span = bundle.wallclock_span
    top = bundle.top_expensive_session
    lines: list[str] = []
    lines.append("Postmortem bundle")
    scope_bits = [f"matched={scope.matched_session_count}", f"analyzed={scope.analyzed_session_count}"]
    if scope.since:
        scope_bits.append(f"since={scope.since}")
    if scope.until:
        scope_bits.append(f"until={scope.until}")
    if scope.query:
        scope_bits.append(f"query={scope.query!r}")
    if scope.truncated:
        scope_bits.append(f"truncated (dropped={scope.dropped_session_count})")
    lines.append("  scope: " + ", ".join(scope_bits))
    lines.append(f"  sessions: {bundle.session_count.count}")
    lines.append(
        f"  wallclock: span_ms={span.span_ms} summed_wall_ms={span.summed_wall_duration_ms} "
        f"({span.earliest_first_message_at} -> {span.latest_last_message_at})"
    )
    cost_label = "estimated" if cost.cost_is_estimated else "exact"
    lines.append(f"  cost: ${cost.total_cost_usd:.6f} ({cost_label})")
    lines.append(
        f"    tokens: input={cost.tokens.input_tokens} output={cost.tokens.output_tokens} "
        f"cache_read={cost.tokens.cache_read_tokens} cache_write={cost.tokens.cache_write_tokens}"
    )
    if top.status == "ok":
        lines.append(
            f"  top_expensive_session: {top.session_id} (${top.total_cost_usd:.6f}) {top.title or ''}".rstrip()
        )
    else:
        lines.append(f"  top_expensive_session: {top.status} ({top.reason})")
    if bundle.repos_touched:
        repo_bits = ", ".join(f"{r.repo}({r.session_count})" for r in bundle.repos_touched)
        lines.append(f"  repos_touched: {repo_bits}")
    else:
        lines.append("  repos_touched: (none)")
    lines.append(f"  subagent_branch_count: {bundle.subagent_branch_count.count}")
    lines.append(f"  longest_tool_gap: {_fmt_degraded(bundle.longest_tool_gap)}")
    lines.append(f"  wasted_loop: {_fmt_pathology(bundle.wasted_loop)}")
    lines.append(f"  failure_mode: {_fmt_pathology(bundle.failure_mode)}")
    evidence = _evidence_index(bundle)
    if evidence:
        lines.append("  evidence:")
        for name, refs in evidence:
            lines.append(f"    {name}: {', '.join(_format_ref(r) for r in refs)}")
    return "\n".join(lines)


def render_postmortem_markdown(bundle: PostmortemBundle) -> str:
    """Render the bundle as Markdown from the same payload object."""

    scope = bundle.scope
    cost = bundle.estimated_cost
    span = bundle.wallclock_span
    top = bundle.top_expensive_session
    cost_label = "estimated" if cost.cost_is_estimated else "exact"
    lines: list[str] = []
    lines.append("# Postmortem Bundle")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Matched sessions: {scope.matched_session_count}")
    lines.append(f"- Analyzed sessions: {scope.analyzed_session_count}")
    lines.append(f"- Since: {scope.since or '(unbounded)'}")
    lines.append(f"- Until: {scope.until or '(unbounded)'}")
    lines.append(f"- Query: {scope.query or '(none)'}")
    if scope.truncated:
        lines.append(f"- Truncated: dropped {scope.dropped_session_count} sessions beyond the analysis cap")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| session_count | {bundle.session_count.count} |")
    lines.append(f"| wallclock_span | span_ms={span.span_ms}, summed_wall_ms={span.summed_wall_duration_ms} |")
    lines.append(f"| estimated_cost_usd | ${cost.total_cost_usd:.6f} ({cost_label}) |")
    lines.append(
        f"| token_lanes | input={cost.tokens.input_tokens}, output={cost.tokens.output_tokens}, "
        f"cache_read={cost.tokens.cache_read_tokens}, cache_write={cost.tokens.cache_write_tokens} |"
    )
    if top.status == "ok":
        lines.append(f"| top_expensive_session | {top.session_id} (${top.total_cost_usd:.6f}) |")
    else:
        lines.append(f"| top_expensive_session | {top.status}: {top.reason} |")
    repos = ", ".join(f"{r.repo} ({r.session_count})" for r in bundle.repos_touched) or "(none)"
    lines.append(f"| repos_touched | {repos} |")
    lines.append(f"| subagent_branch_count | {bundle.subagent_branch_count.count} |")
    lines.append(f"| longest_tool_gap | {_fmt_degraded(bundle.longest_tool_gap)} |")
    lines.append(f"| wasted_loop | {_fmt_pathology(bundle.wasted_loop)} |")
    lines.append(f"| failure_mode | {_fmt_pathology(bundle.failure_mode)} |")
    lines.append("")
    evidence = _evidence_index(bundle)
    if evidence:
        lines.append("## Evidence")
        lines.append("")
        lines.append("| Field | Evidence refs |")
        lines.append("| --- | --- |")
        for name, refs in evidence:
            lines.append(f"| {name} | {', '.join(_format_ref(r) for r in refs)} |")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "POSTMORTEM_SCHEMA_VERSION",
    "CostMetric",
    "DegradedField",
    "PostmortemBundle",
    "PostmortemScope",
    "RepoTouchedMetric",
    "SessionCountMetric",
    "SubagentBranchMetric",
    "TokenLanes",
    "TopExpensiveSessionMetric",
    "WallclockSpanMetric",
    "compile_postmortem_bundle",
    "render_postmortem_markdown",
    "render_postmortem_plain",
]
