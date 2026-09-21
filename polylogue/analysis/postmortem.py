"""Distilled agent-session postmortem bundle (#2380).

One compact, shareable artifact over a matched session scope. Every headline
metric carries at least one drillable :class:`EvidenceRef`. The ``failure_mode``
and ``wasted_loop`` fields are populated by the #2383 pathology detectors
(:mod:`polylogue.analysis.pathology`) run over the session-digest run
projections; they report ``detected``/``clean``/``partial``/``unavailable``
rather than fabricating a signal, and every pathology field carries the
coverage it is a statement about (:class:`PathologyCoverage`) so a sweep over
part of the scope can never read as a swept-clean scope.
``longest_tool_gap`` still degrades in v0 because the session profile and
session digest do not carry per-tool-call timestamps.

The aggregator :func:`compile_postmortem_bundle` is pure: it consumes
already-fetched profiles and digests and performs no I/O, so it is unit-testable
in isolation. The owning API method (``Polylogue.postmortem_bundle``) does the
fetching and constructs the :class:`PostmortemScope`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from polylogue.analysis.archive_models import ArchiveInsightModel
from polylogue.analysis.pathology import PathologyFinding, compile_pathology_report
from polylogue.core.refs import EvidenceRef

if TYPE_CHECKING:
    from polylogue.analysis.transforms import SessionDigest
    from polylogue.archive.session.models import SessionProfile

POSTMORTEM_SCHEMA_VERSION = 3

# Bounded number of evidence refs attached to an aggregate metric so a
# whole-archive scope does not emit an unbounded ref list.
_MAX_AGGREGATE_EVIDENCE = 5

# Reasons used when a field degrades honestly instead of fabricating a value.
_PATHOLOGY_UNAVAILABLE_REASON = (
    "no run projection available in scope; pathology detection needs session-digest evidence"
)
_TOOL_GAP_REASON = (
    "per-tool-call timestamps are not present in the session profile or session "
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
    """Aggregate spend with labelled token lanes (#2380 AC4).

    ``unpriced_session_count`` is how many of the analyzed sessions carried no
    usage evidence to price (``cost_provenance == "unknown"``). Those sessions
    contribute 0.0 to ``total_cost_usd`` because absent evidence is not a bill,
    so without this count "$12.40 across 200 sessions" reads identically
    whether none or most of them were unpriced. It is required, not defaulted:
    a caller that cannot say how much of its total is unpriced would be
    reporting the absence as a known zero.
    """

    total_cost_usd: float
    cost_is_estimated: bool
    unpriced_session_count: int
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


@dataclass(frozen=True, slots=True)
class PathologyCoverage:
    """What population the detectors actually swept, bucket by bucket.

    A pathology sweep answers an *existential* question -- "does this scope
    contain a pathology?" -- and an existential answer does not survive
    partial coverage the way a rate does. The coverage floor in
    :mod:`polylogue.analysis.measurement.outcome_coverage` works because at
    coverage ``c`` a published rate can be wrong by at most ``1 - c``; a single
    unswept session, by contrast, flips ``no pathology found`` outright no
    matter how small a share of the scope it is. So there is no floor below
    ``1.0`` at which ``clean`` is honest, and this type declares strict
    coverage rather than a second tunable threshold.

    Following ``outcome_coverage``'s discipline, the unswept sessions are kept
    in two distinct buckets rather than folded into one number: a session whose
    digest is missing was fetched and had no run projection to give, which is
    not the same fact as a session the analysis cap never reached at all.
    """

    swept_session_count: int
    missing_digest_count: int
    unanalyzed_session_count: int

    @property
    def scope_session_count(self) -> int:
        """Every session in the declared scope, swept or not."""

        return self.swept_session_count + self.missing_digest_count + self.unanalyzed_session_count

    @property
    def unswept_session_count(self) -> int:
        """Sessions the detectors never saw: missing digest plus unanalyzed."""

        return self.missing_digest_count + self.unanalyzed_session_count

    @property
    def is_complete(self) -> bool:
        """True only when every session in scope supplied a run projection."""

        return self.swept_session_count > 0 and self.unswept_session_count == 0

    @property
    def coverage(self) -> float | None:
        """Share of the scope the detectors swept, or ``None`` when undefined."""

        total = self.scope_session_count
        if total <= 0:
            return None
        return self.swept_session_count / total

    def gap_detail(self) -> str:
        """Name the unswept portion, bucket by bucket, for the payload."""

        buckets = []
        if self.missing_digest_count:
            buckets.append(f"{self.missing_digest_count} without a session digest")
        if self.unanalyzed_session_count:
            buckets.append(f"{self.unanalyzed_session_count} never analyzed")
        gap = "; ".join(buckets) if buckets else f"{self.unswept_session_count} unswept"
        return f"swept {self.swept_session_count} of {self.scope_session_count} sessions in scope ({gap})"


def _coverage_for(
    profiles: Sequence[SessionProfile],
    digests: Mapping[str, SessionDigest],
    *,
    scope: PostmortemScope,
) -> PathologyCoverage:
    """Split the declared scope into swept and the two unswept buckets."""

    swept = sum(1 for profile in profiles if profile.session_id in digests)
    missing_digest = len(profiles) - swept
    # Sessions the scope matched but that never reached this aggregator at all:
    # the analysis cap dropped them, or their profile failed to hydrate.
    unanalyzed = max(int(scope.matched_session_count) - len(profiles), 0)
    return PathologyCoverage(
        swept_session_count=swept,
        missing_digest_count=missing_digest,
        unanalyzed_session_count=unanalyzed,
    )


class PathologyField(ArchiveInsightModel):
    """A pathology headline field populated by the #2383 detectors.

    ``status`` always answers "over WHAT population":

    * ``detected`` -- one or more findings exist. True under any coverage, but
      ``count`` is a *lower bound* whenever the coverage counts below show an
      unswept remainder.
    * ``clean`` -- the detectors ran over **every** session in the declared
      scope and found nothing. This is the only status that licenses "the scope
      contains no pathology".
    * ``partial`` -- the detectors ran over a strict subset and found nothing
      *in the part they saw*. The unswept remainder is unexamined, not clean.
    * ``unavailable`` -- no run projection was available anywhere in scope, so
      the detectors never ran.

    ``partial`` is a genuine vocabulary addition because none of the other
    three can carry the fact. ``clean``'s own contract is "the detectors ran
    but found nothing", read by every consumer as a statement about the scope;
    ``detected`` is false when there are no findings; and ``unavailable`` means
    the detectors never ran, which discards the real evidence that a subset was
    in fact swept and came back empty. Collapsing a partial sweep into any of
    them either overclaims or throws away measurement.

    ``count`` is the number of findings; ``by_kind`` and ``detail`` describe the
    distribution; ``evidence_refs`` drill into examples. The
    ``*_session_count`` fields are the coverage receipt described by
    :class:`PathologyCoverage`.
    """

    status: Literal["detected", "clean", "partial", "unavailable"]
    count: int = 0
    detail: str = ""
    by_kind: dict[str, int] = {}
    # Coverage receipt: which population the status above is a statement about.
    swept_session_count: int = 0
    scope_session_count: int = 0
    missing_digest_count: int = 0
    unanalyzed_session_count: int = 0
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @property
    def unswept_session_count(self) -> int:
        return self.missing_digest_count + self.unanalyzed_session_count

    @property
    def covers_whole_scope(self) -> bool:
        """Whether the status is a claim about the entire declared scope."""

        return self.scope_session_count > 0 and self.unswept_session_count == 0


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


def _coverage_field(
    status: Literal["detected", "clean", "partial", "unavailable"],
    *,
    detail: str,
    coverage: PathologyCoverage,
    count: int = 0,
    by_kind: dict[str, int] | None = None,
    evidence_refs: tuple[EvidenceRef, ...] = (),
) -> PathologyField:
    """Attach the coverage receipt to every field, whatever its status."""

    return PathologyField(
        status=status,
        count=count,
        detail=detail,
        by_kind=by_kind if by_kind is not None else {},
        swept_session_count=coverage.swept_session_count,
        scope_session_count=coverage.scope_session_count,
        missing_digest_count=coverage.missing_digest_count,
        unanalyzed_session_count=coverage.unanalyzed_session_count,
        evidence_refs=evidence_refs,
    )


def _pathology_field(
    findings: Sequence[PathologyFinding],
    *,
    coverage: PathologyCoverage,
) -> PathologyField:
    """Build a postmortem pathology field from detector findings.

    ``coverage`` is required: a findings list alone cannot distinguish "nothing
    exists in this scope" from "nothing exists in the fraction of it that was
    examined", and the caller is the only party that knows which.
    """
    if not findings:
        if coverage.swept_session_count <= 0:
            return _coverage_field("unavailable", detail=_PATHOLOGY_UNAVAILABLE_REASON, coverage=coverage)
        if coverage.is_complete:
            return _coverage_field(
                "clean",
                detail=f"detectors ran over all {coverage.scope_session_count} sessions in scope; no pathology found",
                coverage=coverage,
            )
        return _coverage_field(
            "partial",
            detail=(
                f"no pathology found in the swept subset; {coverage.gap_detail()}. "
                "The unswept remainder is unexamined, not clean"
            ),
            coverage=coverage,
        )
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
    if not coverage.is_complete:
        # The count is a floor, not a total: the unswept remainder may hold more.
        detail = f"{detail} (lower bound: {coverage.gap_detail()})"
    return _coverage_field(
        "detected",
        detail=detail,
        coverage=coverage,
        count=len(findings),
        by_kind=by_kind,
        evidence_refs=tuple(refs),
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def compile_postmortem_bundle(
    profiles: Sequence[SessionProfile],
    digests: Mapping[str, SessionDigest],
    *,
    scope: PostmortemScope,
) -> PostmortemBundle:
    """Pure aggregator: build a :class:`PostmortemBundle` from fetched data.

    No I/O. ``profiles`` are the hydrated session profiles in scope; ``digests``
    maps ``session_id`` to its session digest (a subset of profiles is fine —
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
    unpriced = 0
    for profile in profiles:
        total_cost += float(profile.total_cost_usd)
        any_estimated = any_estimated or bool(profile.cost_is_estimated)
        if profile.cost_provenance == "unknown":
            unpriced += 1
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
        unpriced_session_count=unpriced,
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
    # The session digests carry the typed run projection the detectors need.
    # The detectors see only the sessions that supplied a run projection, so the
    # coverage receipt travels with each field: "no findings" is a claim about
    # the swept population, never about the declared scope by default.
    coverage = _coverage_for(profiles, digests, scope=scope)
    projections = [digests[p.session_id].run_projection for p in profiles if p.session_id in digests]
    report = compile_pathology_report(projections) if projections else None
    findings = report.findings if report is not None else ()
    wasted_loop = _pathology_field([f for f in findings if f.kind == "wasted_loop"], coverage=coverage)
    failure_mode = _pathology_field([f for f in findings if f.kind == "stale_context"], coverage=coverage)

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
        return f"clean (no pathology found across all {field.scope_session_count} sessions in scope)"
    if field.status == "partial":
        return (
            f"partial (no pathology found in {field.swept_session_count} of "
            f"{field.scope_session_count} sessions; {field.unswept_session_count} unswept)"
        )
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
    "PathologyCoverage",
    "PathologyField",
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
