"""Corpus-wide sanitized agent-usage portfolio report (#2437).

One rendered narrative over the *whole* archive (or a ``--since``/``--origin``
bounded slice): aggregate session/repo counts, cost and wall-clock
distributions, the top recurring failure-mode pathologies, and the top
context-loss patterns. It is the corpus-scope sibling of the per-session
postmortem bundle (#2380) and is meant to be sanitized and attached to a
portfolio.

This module is a thin **composition** over existing transforms — it owns no new
durable noun (per the #1807 anti-ceremony rule). It reuses the postmortem
metric models (:mod:`polylogue.insights.postmortem`), the pathology detectors
(:mod:`polylogue.insights.pathology`), and is routed through the #2381
fail-closed sanitizer by its owning API method.

The aggregator :func:`compile_portfolio_bundle` is pure: it consumes
already-fetched profiles and session digests and performs no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from polylogue.core.refs import EvidenceRef
from polylogue.core.stats import percentile
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.pathology import PathologyFinding, compile_pathology_report
from polylogue.insights.postmortem import (
    _MAX_AGGREGATE_EVIDENCE,
    CostMetric,
    PathologyField,
    PostmortemScope,
    RepoTouchedMetric,
    SessionCountMetric,
    TokenLanes,
    WallclockSpanMetric,
    _pathology_field,
)

if TYPE_CHECKING:
    from polylogue.archive.session.models import SessionProfile
    from polylogue.insights.transforms import SessionDigest

PORTFOLIO_SCHEMA_VERSION = 1

# Bounded number of repos / origins / pathology examples rendered so a
# whole-archive scope produces a compact, skimmable artifact.
_DEFAULT_TOP_N = 10

# Severity rank for ordering the most-serious pathologies first.
_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}

# The context-loss patterns the portfolio surfaces are exactly the
# stale-context detector's findings (#2383).
_CONTEXT_LOSS_KINDS = frozenset({"stale_context"})


class DistributionStat(ArchiveInsightModel):
    """A quartile distribution over a per-session numeric metric.

    ``count`` is the number of non-null samples; ``total`` their sum. The
    percentile fields use nearest-rank over the sorted samples so the same input
    always yields the same output (snapshot-stable). All fields are ``0`` when
    there is no signal — honest degradation, never a fabricated quartile.
    """

    count: int = 0
    total: float = 0.0
    minimum: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    maximum: float = 0.0
    mean: float = 0.0


class OriginCountMetric(ArchiveInsightModel):
    """Number of sessions seen for one public origin token."""

    origin: str
    session_count: int


class PortfolioBundle(ArchiveInsightModel):
    """The corpus-wide, sanitizable portfolio artifact (#2437).

    Composes the postmortem metric vocabulary (#2380) and the pathology
    distribution (#2383) over a whole-archive scope, adding per-session cost and
    wall-clock distributions. No-signal fields degrade honestly.
    """

    schema_version: int = PORTFOLIO_SCHEMA_VERSION
    scope: PostmortemScope
    session_count: SessionCountMetric
    origins: tuple[OriginCountMetric, ...] = ()
    repos_touched: tuple[RepoTouchedMetric, ...] = ()
    estimated_cost: CostMetric
    cost_distribution: DistributionStat
    wallclock_span: WallclockSpanMetric
    wallclock_distribution_s: DistributionStat
    pathologies: PathologyField
    context_loss: PathologyField
    top_pathologies: tuple[PathologyFinding, ...] = ()


def _distribution(values: Sequence[float]) -> DistributionStat:
    if not values:
        return DistributionStat()
    ordered = sorted(values)
    total = float(sum(ordered))
    return DistributionStat(
        count=len(ordered),
        total=round(total, 6),
        minimum=round(float(ordered[0]), 6),
        p50=round(percentile(ordered, 0.5, method="nearest"), 6),
        p90=round(percentile(ordered, 0.9, method="nearest"), 6),
        maximum=round(float(ordered[-1]), 6),
        mean=round(total / len(ordered), 6),
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _finding_sort_key(finding: PathologyFinding) -> tuple[int, int, str, str]:
    # Most severe, then most frequent, then deterministic by id/detail.
    return (
        -_SEVERITY_RANK.get(finding.severity, 0),
        -finding.occurrence_count,
        finding.session_id,
        finding.detail,
    )


def compile_portfolio_bundle(
    profiles: Sequence[SessionProfile],
    digests: Mapping[str, SessionDigest],
    *,
    scope: PostmortemScope,
    top_n: int = _DEFAULT_TOP_N,
) -> PortfolioBundle:
    """Pure aggregator: build a :class:`PortfolioBundle` from fetched data.

    No I/O. ``profiles`` are the hydrated session profiles in scope; ``digests``
    maps ``session_id`` to its session digest (a subset is fine). Reuses the
    postmortem aggregation idioms for session_count / cost / wallclock /
    repos_touched and adds per-session distributions plus the pathology
    distribution (#2383).
    """

    session_refs = tuple(EvidenceRef(session_id=p.session_id) for p in profiles)

    session_count = SessionCountMetric(
        count=len(profiles),
        evidence_refs=session_refs[:_MAX_AGGREGATE_EVIDENCE],
    )

    # --- origins -------------------------------------------------------------
    origin_counts: dict[str, int] = {}
    for profile in profiles:
        origin_counts[profile.origin] = origin_counts.get(profile.origin, 0) + 1
    origins = tuple(
        OriginCountMetric(origin=origin, session_count=count)
        for origin, count in sorted(origin_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    # --- estimated_cost + token lanes + cost distribution --------------------
    total_cost = 0.0
    any_estimated = False
    input_tokens = output_tokens = cache_read = cache_write = 0
    cost_bearing_refs: list[EvidenceRef] = []
    per_session_cost: list[float] = []
    for profile in profiles:
        cost = float(profile.total_cost_usd)
        total_cost += cost
        per_session_cost.append(cost)
        any_estimated = any_estimated or bool(profile.cost_is_estimated)
        input_tokens += int(profile.total_input_tokens)
        output_tokens += int(profile.total_output_tokens)
        cache_read += int(profile.total_cache_read_tokens)
        cache_write += int(profile.total_cache_write_tokens)
        if cost > 0 and len(cost_bearing_refs) < _MAX_AGGREGATE_EVIDENCE:
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
    # Only sessions that carry a positive cost contribute to the spend
    # distribution; a corpus of mostly $0 sessions would otherwise crush every
    # quartile to zero and hide the real spread.
    cost_distribution = _distribution([c for c in per_session_cost if c > 0])

    # --- wallclock span + distribution --------------------------------------
    earliest_profile: SessionProfile | None = None
    latest_profile: SessionProfile | None = None
    summed_wall_ms = 0
    per_session_wall_s: list[float] = []
    for profile in profiles:
        wall_ms = max(int(profile.wall_duration_ms), 0)
        summed_wall_ms += wall_ms
        if wall_ms > 0:
            per_session_wall_s.append(wall_ms / 1000.0)
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
        evidence_refs=tuple(span_refs) if span_refs else session_refs[:1],
    )
    wallclock_distribution_s = _distribution(per_session_wall_s)

    # --- repos_touched (top N) ----------------------------------------------
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
        for repo, sessions in sorted(repo_sessions.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top_n]
    )

    # --- pathology distribution (#2383) -------------------------------------
    projections = [digests[p.session_id].run_projection for p in profiles if p.session_id in digests]
    if not projections:
        pathologies = PathologyField(
            status="unavailable",
            detail="no run projection available in scope; pathology detection needs session-digest evidence",
        )
        context_loss = PathologyField(
            status="unavailable",
            detail="no run projection available in scope; context-loss detection needs session-digest evidence",
        )
        top_pathologies: tuple[PathologyFinding, ...] = ()
    else:
        report = compile_pathology_report(projections)
        findings = list(report.findings)
        pathologies = _pathology_field(findings)
        context_loss = _pathology_field([f for f in findings if f.kind in _CONTEXT_LOSS_KINDS])
        top_pathologies = tuple(sorted(findings, key=_finding_sort_key)[:top_n])

    return PortfolioBundle(
        scope=scope,
        session_count=session_count,
        origins=origins,
        repos_touched=repos_touched,
        estimated_cost=estimated_cost,
        cost_distribution=cost_distribution,
        wallclock_span=wallclock_span,
        wallclock_distribution_s=wallclock_distribution_s,
        pathologies=pathologies,
        context_loss=context_loss,
        top_pathologies=top_pathologies,
    )


def _fmt_usd(value: float) -> str:
    return f"${value:.2f}"


def _fmt_distribution(label: str, dist: DistributionStat, *, unit: str) -> str:
    if dist.count == 0:
        return f"{label}: no signal"
    return (
        f"{label}: n={dist.count} total={dist.total:g}{unit} "
        f"min={dist.minimum:g} p50={dist.p50:g} p90={dist.p90:g} "
        f"max={dist.maximum:g} mean={dist.mean:g}"
    )


def render_portfolio_plain(bundle: PortfolioBundle) -> str:
    """Render the bundle as a compact plain-text artifact.

    Pure: derives entirely from ``bundle`` so plain and markdown render from the
    same payload object.
    """
    lines: list[str] = []
    scope = bundle.scope
    lines.append("Polylogue Portfolio Report")
    window = f"{scope.since or 'archive-start'} → {scope.until or 'now'}"
    lines.append(f"scope: {window}")
    lines.append(
        f"sessions: analyzed={scope.analyzed_session_count} matched={scope.matched_session_count}"
        + (f" (truncated, dropped={scope.dropped_session_count})" if scope.truncated else "")
    )
    if bundle.origins:
        origin_text = ", ".join(f"{o.origin}={o.session_count}" for o in bundle.origins)
        lines.append(f"origins: {origin_text}")
    cost = bundle.estimated_cost
    estimated = " (estimated)" if cost.cost_is_estimated else ""
    lines.append(f"total_cost: {_fmt_usd(cost.total_cost_usd)}{estimated}")
    lines.append(
        "tokens: "
        f"in={cost.tokens.input_tokens} out={cost.tokens.output_tokens} "
        f"cache_read={cost.tokens.cache_read_tokens} cache_write={cost.tokens.cache_write_tokens}"
    )
    lines.append(_fmt_distribution("cost_per_session_usd", bundle.cost_distribution, unit=""))
    lines.append(_fmt_distribution("wall_per_session_s", bundle.wallclock_distribution_s, unit="s"))
    if bundle.wallclock_span.span_ms is not None:
        lines.append(f"wallclock_span_ms: {bundle.wallclock_span.span_ms}")
    if bundle.repos_touched:
        lines.append("repos_touched:")
        for repo in bundle.repos_touched:
            lines.append(f"  - {repo.repo}: {repo.session_count}")
    lines.append(f"pathologies: {bundle.pathologies.status} ({bundle.pathologies.detail})")
    lines.append(f"context_loss: {bundle.context_loss.status} ({bundle.context_loss.detail})")
    if bundle.top_pathologies:
        lines.append("top_pathologies:")
        for finding in bundle.top_pathologies:
            lines.append(f"  - [{finding.severity}] {finding.kind}: {finding.detail}")
    return "\n".join(lines)


def render_portfolio_markdown(bundle: PortfolioBundle) -> str:
    """Render the bundle as Markdown from the same payload object."""
    scope = bundle.scope
    cost = bundle.estimated_cost
    window = f"{scope.since or 'archive-start'} → {scope.until or 'now'}"
    out: list[str] = ["# Portfolio Report", ""]
    out.append(f"- **Scope:** {window}")
    out.append(
        f"- **Sessions:** {scope.analyzed_session_count} analyzed / {scope.matched_session_count} matched"
        + (f" (truncated, dropped {scope.dropped_session_count})" if scope.truncated else "")
    )
    if bundle.origins:
        out.append("- **Origins:** " + ", ".join(f"{o.origin} ({o.session_count})" for o in bundle.origins))
    estimated = " _(estimated)_" if cost.cost_is_estimated else ""
    out.append(f"- **Total cost:** {_fmt_usd(cost.total_cost_usd)}{estimated}")
    out.append(
        "- **Tokens:** "
        f"in {cost.tokens.input_tokens}, out {cost.tokens.output_tokens}, "
        f"cache-read {cost.tokens.cache_read_tokens}, cache-write {cost.tokens.cache_write_tokens}"
    )
    out.append("")
    out.append("## Distributions")
    out.append("")
    out.append(f"- {_fmt_distribution('Cost per session (USD)', bundle.cost_distribution, unit='')}")
    out.append(f"- {_fmt_distribution('Wall time per session', bundle.wallclock_distribution_s, unit='s')}")
    if bundle.repos_touched:
        out.append("")
        out.append("## Repos touched")
        out.append("")
        for repo in bundle.repos_touched:
            out.append(f"- `{repo.repo}` — {repo.session_count} session(s)")
    out.append("")
    out.append("## Failure-mode pathologies")
    out.append("")
    out.append(f"- **Overall:** {bundle.pathologies.status} — {bundle.pathologies.detail}")
    out.append(f"- **Context loss:** {bundle.context_loss.status} — {bundle.context_loss.detail}")
    if bundle.top_pathologies:
        out.append("")
        out.append("### Top patterns")
        out.append("")
        for finding in bundle.top_pathologies:
            out.append(f"- **[{finding.severity}] {finding.kind}** — {finding.detail}")
    out.append("")
    return "\n".join(out)


__all__ = [
    "PORTFOLIO_SCHEMA_VERSION",
    "DistributionStat",
    "OriginCountMetric",
    "PortfolioBundle",
    "compile_portfolio_bundle",
    "render_portfolio_markdown",
    "render_portfolio_plain",
]
