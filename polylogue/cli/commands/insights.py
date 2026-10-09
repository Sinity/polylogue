"""Archive insight inspection commands — registry-driven.

Insight commands inherit ``--origin``, ``--since``, and ``--until`` from
the root CLI context so that ``polylogue --origin codex-session analyze insights
profiles`` works without re-specifying the filter on the subcommand.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click

from polylogue.analysis.archive import ArchiveInsightUnavailableError
from polylogue.analysis.audit import (
    DEFAULT_AUDIT_SAMPLE_LIMIT,
    InsightRigorAuditQuery,
    InsightRigorAuditReport,
)
from polylogue.analysis.export_bundle_contracts import (
    InsightExportBundleError,
    InsightExportBundleRequest,
    InsightExportBundleResult,
    InsightExportFormat,
)
from polylogue.analysis.readiness import (
    InsightReadinessQuery,
    InsightReadinessReport,
    known_insight_readiness_names,
    normalize_insight_readiness_name,
)
from polylogue.analysis.registry import (
    INSIGHT_REGISTRY,
    InsightQueryError,
    InsightType,
    build_insight_query,
    render_insight_items,
)
from polylogue.cli.operation_kernel import OperationRequest
from polylogue.cli.read_dispatch import dispatch_read
from polylogue.cli.shared.helper_support import fail
from polylogue.cli.shared.insight_command_contracts import (
    InsightCommandInputError,
    InsightCommandRequest,
    normalize_insight_query_kwargs,
    query_model_field_names,
)
from polylogue.cli.shared.machine_errors import emit_success
from polylogue.cli.shared.types import AppEnv

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
            ("--json", "output_format"),
            flag_value="json",
            default=None,
            help="Alias for --format json.",
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
            build_insight_query(pt, **request.query_kwargs)
            result, _served_by = dispatch_read(
                env.config,
                OperationRequest("insights.list", {"page": {"insight": pt.name, "query": request.query_kwargs}}),
            )
            from polylogue.operations.insight_contracts import InsightListResult

            parsed = InsightListResult.model_validate(result)
            page = parsed.page
            if page.insight != pt.name:
                raise click.ClickException("resident insight page belongs to a different query type")
            items = page.items
        except (ArchiveInsightUnavailableError, InsightCommandInputError, InsightQueryError) as exc:
            fail(f"insights {pt.resolved_cli_command_name}", str(exc))
        render_insight_items(items, pt, json_mode=request.wants_json, outcome=parsed.outcome)

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
        ("Session-level", ("profiles",)),
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
        return value or "-"

    click.echo("Readiness: ready" if report.converged else "Readiness: derived domains incomplete")
    if report.debt_stages:
        click.echo(f"Operation debt: {', '.join(report.debt_stages)}")
    click.echo(f"Total sessions: {report.total_sessions}")
    if report.origin or report.since or report.until:
        click.echo(
            f"Scope: origin={origin_label(report.origin)} since={report.since or '-'} until={report.until or '-'}"
        )
    click.echo("")
    for insight in report.insights:
        expected = f" expected={insight.expected_row_count}" if insight.expected_row_count is not None else ""
        presence = "" if insight.table_present else " (table absent)"
        click.echo(f"{insight.insight_name}: rows={insight.row_count}{expected}{presence}")
        if insight.degraded_count or insight.fallback_reason_counts:
            reasons = ", ".join(f"{name}={count}" for name, count in sorted(insight.fallback_reason_counts.items()))
            detail = f" fallback_reasons={reasons}" if reasons else ""
            click.echo(f"  degraded={insight.degraded_count}{detail}")
        if insight.missing_count or insight.stale_count or insight.orphan_count or insight.incompatible_count:
            click.echo(
                "  "
                f"missing={insight.missing_count} stale={insight.stale_count} "
                f"orphan={insight.orphan_count} incompatible={insight.incompatible_count}"
            )
        if insight.origin_coverage:
            origins = ", ".join(
                f"{origin_label(coverage.origin)}={coverage.row_count}" for coverage in insight.origin_coverage
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
        withheld = f" withheld={insight.withheld_reason}" if insight.withheld_reason else ""
        click.echo(f"{insight.insight_name}: rows={insight.row_count}{withheld}")
        for warning in insight.warnings:
            click.echo(f"  warning: {warning}")
        for error in insight.errors:
            click.echo(f"  error: {error}")


@ops_insights_command.command("status")
@click.option("--insight", "insights", multiple=True, help="Insight readiness target. May be repeated.")
@click.option("--origin", "-o", default=None, help="Limit origin coverage details to one origin.")
@click.option("--since", default=None, help="Limit coverage details to rows at/after this timestamp or date.")
@click.option("--until", default=None, help="Limit coverage details to rows at/before this timestamp or date.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.option("--json", "output_format", flag_value="json", default=None, help="Alias for --format json.")
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
        for name in query.insights:
            normalize_insight_readiness_name(name)
        result, _served_by = dispatch_read(
            env.config, OperationRequest("insights.readiness", {"query": query.model_dump(mode="json")})
        )
        from polylogue.operations.insight_contracts import InsightReadinessResult

        selected = InsightReadinessResult.model_validate(result)
        report = selected.report
    except (InsightCommandInputError, ValueError) as exc:
        valid = ", ".join(known_insight_readiness_names())
        fail("insights status", f"{exc}. Known insights: {valid}")
    if _status_wants_json(ctx, output_format=output_format):
        emit_success({**report.model_dump(mode="json"), "outcome": selected.outcome.to_dict()})
        return
    _render_status_plain(report)


@ops_insights_command.command("hermes-health")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.option("--json", "output_format", flag_value="json", default=None, help="Alias for --format json.")
@click.pass_context
def insights_hermes_health_command(ctx: click.Context, output_format: str | None) -> None:
    """Report the bounded Hermes-to-Polylogue integration health rollup (fs1.15).

    Composes existing per-source freshness, dry-run parser/fidelity,
    convergence debt, lifecycle-event, and context-delivery-correlation
    evidence into one read-only view. Reports an explicit
    disabled/unavailable/degraded/healthy verdict rather than a silent zero.
    """
    env: AppEnv = ctx.obj
    from polylogue.operations.hermes_health_contracts import decode_hermes_health_result

    result_payload, _served_by = dispatch_read(
        env.config, OperationRequest(operation="insights.hermes_health", payload={})
    )
    result = decode_hermes_health_result(result_payload)
    health = result.report
    if _status_wants_json(ctx, output_format=output_format):
        emit_success({**health.to_dict(), "outcome": result.outcome.to_dict()})
        return
    _render_hermes_health_plain(health)


def _render_hermes_health_plain(health: object) -> None:
    from polylogue.analysis.hermes_health_contracts import HermesIntegrationHealth

    assert isinstance(health, HermesIntegrationHealth)
    click.echo(f"Hermes integration: {health.verdict} (enabled={health.enabled})")
    click.echo(f"  {health.enabled_reason}")
    if not health.enabled:
        return
    click.echo("")
    click.echo(f"Sources ({len(health.sources)}):")
    for source in health.sources:
        click.echo(
            f"  {source.source_ref} [{source.source_class}] stage={source.stage} "
            f"state={source.operational_state} parse={source.parse_state}"
        )
    if health.parser_failures:
        click.echo("")
        click.echo(f"Parser failures ({len(health.parser_failures)}):")
        for failure in health.parser_failures:
            click.echo(f"  {failure.source_ref}: {failure.reason}")
    click.echo("")
    click.echo(
        f"Convergence debt: failed={health.convergence_debt_failed_count} "
        f"retry_due={health.convergence_debt_retry_due_count}"
    )
    click.echo(
        f"Lifecycle debt: sessions_checked={health.lifecycle_debt.sessions_checked} "
        f"unpaired={health.lifecycle_debt.unpaired_event_count} "
        f"unknown_refs={health.lifecycle_debt.unknown_message_reference_count}"
    )
    click.echo(
        f"Delivery correlation: sessions_checked={health.delivery_correlation.sessions_checked} "
        f"available={health.delivery_correlation.available_count} "
        f"unavailable={health.delivery_correlation.unavailable_count}"
    )
    if health.caveats:
        click.echo("")
        click.echo("Caveats:")
        for caveat in health.caveats:
            click.echo(f"  - {caveat}")


@ops_insights_command.command("export")
@click.option("--out", "output_path", required=True, type=click.Path(path_type=Path), help="Output bundle directory.")
@click.option("--insight", "insights", multiple=True, help="Insight to include. Defaults to all exportable insights.")
@click.option("--origin", "-o", default=None, help="Limit supported insights to one origin.")
@click.option("--since", default=None, help="Limit supported insights to rows at/after this timestamp or date.")
@click.option("--until", default=None, help="Limit supported insights to rows at/before this timestamp or date.")
@click.option("--bundle-format", type=click.Choice(["jsonl"]), default="jsonl", show_default=True)
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.option("--json", "output_format", flag_value="json", default=None, help="Alias for --format json.")
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
            output_path=output_path.absolute(),
            insights=insights,
            origin=filters["origin"] if isinstance(filters["origin"], str) else None,
            since=filters["since"] if isinstance(filters["since"], str) else None,
            until=filters["until"] if isinstance(filters["until"], str) else None,
            output_format=export_format,
            overwrite=overwrite,
        )
        from polylogue.operations.insight_export_contracts import decode_insight_export_result

        payload, _served_by = dispatch_read(
            env.config, OperationRequest("insights.export_bundle", {"request": request.model_dump(mode="json")})
        )
        selected = decode_insight_export_result(payload)
        result = selected.bundle
    except (InsightCommandInputError, InsightExportBundleError) as exc:
        fail("insights export", str(exc))
    if output_format == "json" or ctx.find_root().params.get("output_format") == "json":
        emit_success({**result.model_dump(mode="json"), "outcome": selected.outcome.to_dict()})
    else:
        _render_export_plain(result)
    from polylogue.cli.render.outcome import finish_supplied_outcome

    finish_supplied_outcome(selected.outcome)


@ops_insights_command.command("fable-packet")
@click.option("--seed", required=True, help="Deterministic cohort seed.")
@click.option("--requested-size", type=click.IntRange(min=0), required=True, help="Maximum sampled attempts.")
@click.option("--schema-id", default="delegation.discourse", show_default=True)
@click.option("--schema-version", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--exact-template-cap", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("-f", "--format", "output_format", type=click.Choice(["json"]), default=None)
@click.option("--json", "output_format", flag_value="json", default=None, help="Alias for --format json.")
@click.pass_context
def insights_fable_packet_command(
    ctx: click.Context,
    seed: str,
    requested_size: int,
    schema_id: str,
    schema_version: int,
    exact_template_cap: int,
    output_format: str | None,
) -> None:
    """Cold-regenerate the private, descriptive Fable delegation packet."""
    env: AppEnv = ctx.obj
    from polylogue.operations.fable_packet_contracts import FablePacketRequest, decode_fable_packet_result

    request = FablePacketRequest(
        seed=seed,
        requested_size=requested_size,
        schema_id=schema_id,
        schema_version=schema_version,
        exact_template_cap=exact_template_cap,
    )
    result_payload, _served_by = dispatch_read(
        env.config, OperationRequest(operation="insights.fable_packet", payload=request.model_dump(mode="json"))
    )
    result = decode_fable_packet_result(result_payload)
    packet = result.packet
    payload = {**result.model_dump(mode="json")["packet"], "outcome": result.outcome.to_dict()}
    if output_format == "json" or ctx.find_root().params.get("output_format") == "json":
        emit_success(payload)
        return
    click.echo(f"Fable packet: {packet.status}")
    click.echo(
        f"Population: {packet.population_count} "
        f"(action={packet.action_observed_count}, edge_only={packet.edge_only_count}, unresolved={packet.unresolved_count})"
    )
    click.echo(f"Selected: {len(packet.selected_refs)}; manifest={packet.manifest_id}")
    if packet.not_supported_reasons:
        click.echo(f"Not supported: {', '.join(packet.not_supported_reasons)}")
    else:
        click.echo(f"Distributions: {len(packet.distributions)}; disagreements={packet.disagreement_count}")
    click.echo("Limits: " + ", ".join(packet.limits))


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
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format.",
)
@click.option("--json", "output_format", flag_value="json", default=None, help="Alias for --format json.")
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
        from polylogue.operations.insight_contracts import InsightRigorResult

        result_payload, _served_by = dispatch_read(
            env.config, OperationRequest("insights.rigor", {"query": query.model_dump(mode="json")})
        )
        result = InsightRigorResult.model_validate(result_payload)
        report = result.report
    except (ArchiveInsightUnavailableError, ValueError, InsightQueryError) as exc:
        fail("insights audit", str(exc))
    wants_json = output_format == "json" or ctx.find_root().params.get("output_format") == "json"
    if wants_json:
        emit_success({**report.model_dump(mode="json"), "outcome": result.outcome.to_dict()})
        return
    _render_audit_plain(report)


# Register all insight types as subcommands
for _pt in INSIGHT_REGISTRY.values():
    if _pt.query_model is not None and _pt.operations_method_name:
        analyze_insights_command.add_command(_build_insight_command(_pt))


__all__ = ["analyze_insights_command", "ops_insights_command"]
