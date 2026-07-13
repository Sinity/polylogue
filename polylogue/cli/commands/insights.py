"""Archive insight inspection commands — registry-driven.

Insight commands inherit ``--origin``, ``--since``, and ``--until`` from
the root CLI context so that ``polylogue --origin codex-session analyze insights
profiles`` works without re-specifying the filter on the subcommand.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.insight_command_contracts import (
    InsightCommandInputError,
    InsightCommandRequest,
    normalize_insight_query_kwargs,
    query_model_field_names,
)
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv
from polylogue.insights.archive import ArchiveInsightUnavailableError
from polylogue.insights.audit import (
    DEFAULT_AUDIT_SAMPLE_LIMIT,
    InsightRigorAuditQuery,
    InsightRigorAuditReport,
    build_insight_rigor_audit_report,
)
from polylogue.insights.export_bundles import (
    InsightExportBundleError,
    InsightExportBundleRequest,
    InsightExportBundleResult,
    InsightExportFormat,
)
from polylogue.insights.readiness import InsightReadinessQuery, InsightReadinessReport, known_insight_readiness_names
from polylogue.insights.registry import (
    INSIGHT_REGISTRY,
    InsightQueryError,
    InsightType,
    fetch_insights,
    project_origin_payload,
    render_insight_items,
)
from polylogue.insights.timeline_renderer import (
    build_session_timeline,
    render_markdown,
    render_plain,
)

_ROOT_FILTER_KEYS = ("origin", "since", "until")


def _build_click_params(pt: InsightType) -> list[click.Parameter]:
    """Build Click Option parameters from an insight type's cli_options."""
    params: list[click.Parameter] = []

    for opt in pt.cli_options:
        params.append(
            click.Option(
                opt.flags,
                help=opt.help,
                type=opt.type,
                default=opt.default,
                show_default=opt.show_default,
                is_flag=opt.is_flag,
            )
        )

    # Standard options on every insight command
    params.append(
        click.Option(
            ("--limit", "-l"),
            type=int,
            default=pt.mcp_default_limit,
            show_default=True,
            help="Maximum rows",
        )
    )
    params.append(
        click.Option(
            ("--offset",),
            type=int,
            default=0,
            show_default=True,
            help="Start offset",
        )
    )
    params.append(
        click.Option(
            ("--json", "output_format"),
            flag_value="json",
            default=None,
            help="Shortcut for --format json.",
        )
    )
    params.append(
        click.Option(
            ("--format", "-f", "output_format"),
            type=click.Choice(["json"]),
            default=None,
            help="Output format",
        )
    )

    return params


def _make_callback(pt: InsightType) -> Callable[..., None]:
    """Create the Click callback for an insight type command.

    Inherits ``origin``, ``since``, and ``until`` from the root CLI
    context when the insight's query class accepts them.
    """
    # Pre-resolve accepted fields so we only inject keys the query class understands.
    accepted_query_fields = query_model_field_names(pt)
    accepted_root_keys = tuple(key for key in _ROOT_FILTER_KEYS if key in accepted_query_fields)

    @click.pass_context
    def callback(
        ctx: click.Context,
        /,
        output_format: str | None = None,
        **kwargs: object,
    ) -> None:
        env: AppEnv = ctx.obj
        try:
            request = InsightCommandRequest.from_context(
                ctx,
                pt,
                output_format=output_format,
                kwargs=kwargs,
                inherited_root_keys=accepted_root_keys,
            )
            items = fetch_insights(pt, env.polylogue, **request.query_kwargs)
        except (ArchiveInsightUnavailableError, InsightCommandInputError, InsightQueryError) as exc:
            fail(f"insights {pt.resolved_cli_command_name}", str(exc))
        render_insight_items(items, pt, json_mode=request.wants_json)

    return callback


def _build_insight_command(pt: InsightType) -> click.Command:
    """Build a Click command for a registered insight type."""
    return click.Command(
        name=pt.resolved_cli_command_name,
        callback=_make_callback(pt),
        params=_build_click_params(pt),
        help=pt.cli_help or f"List {pt.display_name.lower()}.",
    )


class _AnalyzeInsightsGroup(click.Group):
    """Click group with section headers for command listing."""

    _SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Session-level", ("profiles", "work-events", "phases", "timeline")),
        ("Aggregate", ("threads", "tag-rollups", "coverage", "tags")),
        ("Analytics", ("tool-usage", "costs", "cost-rollups", "usage-timeline", "debt", "latency")),
    )

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        section_commands: dict[str, list[tuple[str, str]]] = {sec[0]: [] for sec in self._SECTIONS}
        other: list[tuple[str, str]] = []

        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None:
                continue
            help_text = cmd.short_help or ""
            placed = False
            for section_title, cmd_names in self._SECTIONS:
                if name in cmd_names:
                    section_commands[section_title].append((name, help_text))
                    placed = True
                    break
            if not placed:
                other.append((name, help_text))

        limit = formatter.width
        for section_title, _ in self._SECTIONS:
            cmds = section_commands[section_title]
            if not cmds:
                continue
            with formatter.section(section_title):
                formatter.write_dl(sorted(cmds), col_max=limit)

        if other:
            with formatter.section("Other"):
                formatter.write_dl(sorted(other), col_max=limit)


@click.group("insights", cls=_AnalyzeInsightsGroup)
def analyze_insights_command() -> None:
    """Inspect durable archive insight read models."""


@click.group("insights")
def ops_insights_command() -> None:
    """Operate durable archive insight materialization."""


def _status_wants_json(ctx: click.Context, *, output_format: str | None) -> bool:
    if output_format == "json":
        return True
    root_output = ctx.find_root().params.get("output_format")
    return root_output == "json"


def _render_status_plain(report: InsightReadinessReport) -> None:
    def origin_label(value: str | None) -> str:
        if not value:
            return "-"
        projected = project_origin_payload({"origin": value})
        if isinstance(projected, dict):
            return str(projected.get("origin") or "-")
        return "-"

    click.echo(f"Insight Readiness: {report.aggregate_verdict}")
    click.echo(f"Total sessions: {report.total_sessions}")
    if report.origin or report.since or report.until:
        click.echo(
            f"Scope: origin={origin_label(report.origin)} since={report.since or '-'} until={report.until or '-'}"
        )
    click.echo("")
    for insight in report.insights:
        expected = f" expected={insight.expected_row_count}" if insight.expected_row_count is not None else ""
        click.echo(f"{insight.insight_name}: {insight.verdict} rows={insight.row_count}{expected}")
        if insight.missing_count or insight.stale_count or insight.orphan_count or insight.incompatible_count:
            click.echo(
                "  "
                f"missing={insight.missing_count} stale={insight.stale_count} "
                f"orphan={insight.orphan_count} incompatible={insight.incompatible_count}"
            )
        if insight.ready_flags:
            flags = ", ".join(f"{key}={value}" for key, value in sorted(insight.ready_flags.items()))
            click.echo(f"  flags: {flags}")
        if insight.provider_coverage:
            origins = ", ".join(
                f"{origin_label(coverage.source_name)}={coverage.row_count}" for coverage in insight.provider_coverage
            )
            click.echo(f"  origins: {origins}")
        if insight.version_coverage:
            versions = ", ".join(f"{coverage.field}={dict(coverage.versions)}" for coverage in insight.version_coverage)
            click.echo(f"  versions: {versions}")
        if insight.schema_contract_issues:
            click.echo(f"  schema: {', '.join(insight.schema_contract_issues)}")


def _render_export_plain(result: InsightExportBundleResult) -> None:
    click.echo(f"Insight export bundle: {result.output_path}")
    click.echo(f"Manifest: {result.manifest_path}")
    click.echo(f"Coverage: {result.coverage_path}")
    click.echo("")
    for insight in result.manifest.insights:
        click.echo(f"{insight.insight_name}: rows={insight.row_count} readiness={insight.readiness_verdict or '-'}")
        for warning in insight.warnings:
            click.echo(f"  warning: {warning}")
        for error in insight.errors:
            click.echo(f"  error: {error}")


@ops_insights_command.command("status")
@click.option("--insight", "insights", multiple=True, help="Insight readiness target. May be repeated.")
@click.option("--origin", "-o", default=None, help="Limit origin coverage details to one origin.")
@click.option("--since", default=None, help="Limit coverage details to rows at/after this timestamp or date.")
@click.option("--until", default=None, help="Limit coverage details to rows at/before this timestamp or date.")
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.pass_context
def insights_status_command(
    ctx: click.Context,
    insights: tuple[str, ...],
    origin: str | None,
    since: str | None,
    until: str | None,
    output_format: str | None,
) -> None:
    """Report insight materialization coverage and readiness."""
    env: AppEnv = ctx.obj
    root_params = ctx.find_root().params
    inherited_origin = origin if origin is not None else root_params.get("origin")
    inherited_since = since if since is not None else root_params.get("since")
    inherited_until = until if until is not None else root_params.get("until")
    try:
        filters = normalize_insight_query_kwargs(
            {
                "origin": inherited_origin,
                "since": inherited_since,
                "until": inherited_until,
            }
        )
        query = InsightReadinessQuery(
            insights=insights,
            origin=filters["origin"] if isinstance(filters["origin"], str) else None,
            since=filters["since"] if isinstance(filters["since"], str) else None,
            until=filters["until"] if isinstance(filters["until"], str) else None,
        )
        report = run_coroutine_sync(env.polylogue.insight_readiness_report(query))
    except (InsightCommandInputError, ValueError) as exc:
        valid = ", ".join(known_insight_readiness_names())
        fail("insights status", f"{exc}. Known insights: {valid}")
    if _status_wants_json(ctx, output_format=output_format):
        emit_success(cast(dict[str, object], project_origin_payload(report.model_dump(mode="json"))))
        return
    _render_status_plain(report)


@ops_insights_command.command("export")
@click.option("--out", "output_path", required=True, type=click.Path(path_type=Path), help="Output bundle directory.")
@click.option("--insight", "insights", multiple=True, help="Insight to include. Defaults to all exportable insights.")
@click.option("--origin", "-o", default=None, help="Limit supported insights to one origin.")
@click.option("--since", default=None, help="Limit supported insights to rows at/after this timestamp or date.")
@click.option("--until", default=None, help="Limit supported insights to rows at/before this timestamp or date.")
@click.option("--bundle-format", type=click.Choice(["jsonl"]), default="jsonl", show_default=True)
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.option(
    "--overwrite", is_flag=True, help="Replace an existing bundle directory after writing a complete new one."
)
@click.pass_context
def insights_export_command(
    ctx: click.Context,
    output_path: Path,
    insights: tuple[str, ...],
    origin: str | None,
    since: str | None,
    until: str | None,
    bundle_format: str,
    output_format: str | None,
    overwrite: bool,
) -> None:
    """Export versioned archive-insight bundles."""
    env: AppEnv = ctx.obj
    root_params = ctx.find_root().params
    inherited_origin = origin if origin is not None else root_params.get("origin")
    inherited_since = since if since is not None else root_params.get("since")
    inherited_until = until if until is not None else root_params.get("until")
    try:
        export_format: InsightExportFormat = "jsonl"
        if bundle_format != "jsonl":
            fail("insights export", f"unsupported export format: {bundle_format}")
        filters = normalize_insight_query_kwargs(
            {
                "origin": inherited_origin,
                "since": inherited_since,
                "until": inherited_until,
            }
        )
        request = InsightExportBundleRequest(
            output_path=output_path,
            insights=insights,
            origin=filters["origin"] if isinstance(filters["origin"], str) else None,
            since=filters["since"] if isinstance(filters["since"], str) else None,
            until=filters["until"] if isinstance(filters["until"], str) else None,
            output_format=export_format,
            overwrite=overwrite,
        )
        result = run_coroutine_sync(env.polylogue.export_insight_bundle(request))
    except (InsightCommandInputError, InsightExportBundleError) as exc:
        fail("insights export", str(exc))
    if output_format == "json" or ctx.find_root().params.get("output_format") == "json":
        emit_success(cast(dict[str, object], project_origin_payload(result.model_dump(mode="json"))))
        return
    _render_export_plain(result)


def _format_pct(count: int, sample: int) -> str:
    if sample <= 0:
        return "-"
    return f"{(count * 100) // sample}%"


def _render_audit_plain(report: InsightRigorAuditReport) -> None:
    click.echo(f"Insight Rigor Audit (sample_limit={report.sample_limit})")
    click.echo("")
    for entry in report.entries:
        sample = entry.sample_size
        click.echo(f"{entry.insight_name} ({entry.display_name})")
        if entry.coverage_status == "uncovered":
            click.echo("  UNCOVERED: no rigor contract declared for this registered product")
            continue
        if entry.coverage_status == "exempt":
            click.echo("  exempt: no rigor contract needed")
            for note in entry.notes:
                click.echo(f"  reason: {note}")
            continue
        if entry.error is not None:
            click.echo(f"  error: {entry.error}")
            continue
        if sample == 0:
            click.echo("  sample=0 (no rows materialized)")
            continue

        click.echo(f"  sample={sample}")
        if entry.has_evidence_payload:
            click.echo(f"  evidence:  {entry.evidence_count} ({_format_pct(entry.evidence_count, sample)})")
        if entry.has_inference_payload:
            click.echo(f"  inference: {entry.inference_count} ({_format_pct(entry.inference_count, sample)})")
        if entry.has_fallback_markers:
            click.echo(f"  fallback:  {entry.fallback_count} ({_format_pct(entry.fallback_count, sample)})")
        click.echo(f"  stale-version rows: {entry.stale_version_count}")
        if entry.has_confidence_field:
            dist = entry.confidence_distribution
            click.echo(f"  confidence: low={dist.low} mid={dist.mid} high={dist.high} unknown={dist.unknown}")
        if entry.version_targets:
            versions = ", ".join(f"{name}={value}" for name, value in sorted(entry.version_targets.items()))
            click.echo(f"  version targets: {versions}")
        for note in entry.notes:
            click.echo(f"  note: {note}")


@ops_insights_command.command("audit")
@click.option(
    "--insight",
    "insights",
    multiple=True,
    help="Limit the audit to one or more registered insight names. Default: every registered product.",
)
@click.option(
    "--sample-limit",
    type=int,
    default=DEFAULT_AUDIT_SAMPLE_LIMIT,
    show_default=True,
    help="Maximum rows per product to sample for the rigor profile.",
)
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format.",
)
@click.pass_context
def insights_audit_command(
    ctx: click.Context,
    insights: tuple[str, ...],
    sample_limit: int,
    output_format: str | None,
) -> None:
    """Report per-product rigor profile across materialized insights (#1275).

    Every registered insight product appears, not just contracted ones
    (9e5.28): a product with a contract reports the share of rows that
    carry an evidence payload, an inference payload, and a fallback
    marker, plus the stale-version row count and a confidence-bucket
    distribution; a product with no contract shows as uncovered unless
    it is explicitly listed as exempt.
    """

    env: AppEnv = ctx.obj
    try:
        query = InsightRigorAuditQuery(insights=insights, sample_limit=sample_limit)
        report = run_coroutine_sync(build_insight_rigor_audit_report(env.polylogue, query))
    except ArchiveInsightUnavailableError as exc:
        fail("insights audit", str(exc))
    wants_json = output_format == "json" or ctx.find_root().params.get("output_format") == "json"
    if wants_json:
        emit_success(report.model_dump(mode="json"))
        return
    _render_audit_plain(report)


@analyze_insights_command.command("timeline")
@click.argument("session_id")
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=None,
    help="Shortcut for --format json.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "markdown", "json"]),
    default=None,
    help="Output format (default: plain, inherits root --format).",
)
@click.pass_context
def insights_timeline_command(
    ctx: click.Context,
    session_id: str,
    output_format: str | None,
) -> None:
    """Render a per-session timeline with hook-vs-sort-key fidelity tags.

    Merges materialized work events and session phases for one session
    into a chronological timeline. Each entry carries an explicit fidelity
    tag: ``hook`` for entries whose timing came from a recorded timestamped
    range, ``sort_key`` for entries reconstructed from message sort-key
    ordering.
    """
    env: AppEnv = ctx.obj
    resolved_format = output_format or ctx.find_root().params.get("output_format") or "plain"
    try:
        work_events = run_coroutine_sync(env.polylogue.get_session_work_event_insights(session_id))
        phases = run_coroutine_sync(env.polylogue.get_session_phase_insights(session_id))
    except ArchiveInsightUnavailableError as exc:
        fail("insights timeline", str(exc))
    timeline = build_session_timeline(session_id, work_events, phases)
    if resolved_format == "json":
        emit_success(timeline.to_dict())
        return
    if resolved_format == "markdown":
        click.echo(render_markdown(timeline))
        return
    click.echo(render_plain(timeline))


# Register all insight types as subcommands
for _pt in INSIGHT_REGISTRY.values():
    if _pt.query_model is not None and _pt.operations_method_name:
        analyze_insights_command.add_command(_build_insight_command(_pt))


__all__ = ["analyze_insights_command", "ops_insights_command"]
