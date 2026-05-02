"""Archive insight inspection commands — registry-driven.

Insight commands inherit ``--provider``, ``--since``, and ``--until`` from
the root CLI context so that ``polylogue --provider codex insights profiles``
works without re-specifying the filter on the subcommand.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
    render_insight_items,
)

_ROOT_FILTER_KEYS = ("provider", "since", "until")


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
            ("--limit",),
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
            ("--format", "output_format"),
            type=click.Choice(["json"]),
            default=None,
            help="Output format",
        )
    )

    return params


def _make_callback(pt: InsightType) -> Callable[..., None]:
    """Create the Click callback for an insight type command.

    Inherits ``provider``, ``since``, and ``until`` from the root CLI
    context when the insight's query class accepts them.
    """
    # Pre-resolve accepted fields so we only inject keys the query class understands.
    accepted_root_keys = tuple(key for key in _ROOT_FILTER_KEYS if key in query_model_field_names(pt))

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
            items = fetch_insights(pt, env.operations, **request.query_kwargs)
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


@click.group("insights")
def insights_command() -> None:
    """Inspect durable archive insights."""


def _status_wants_json(ctx: click.Context, *, output_format: str | None) -> bool:
    if output_format == "json":
        return True
    root_output = ctx.find_root().params.get("output_format")
    return root_output == "json"


def _render_status_plain(report: InsightReadinessReport) -> None:
    click.echo(f"Insight Readiness: {report.aggregate_verdict}")
    click.echo(f"Total conversations: {report.total_conversations}")
    if report.provider or report.since or report.until:
        click.echo(f"Scope: provider={report.provider or '-'} since={report.since or '-'} until={report.until or '-'}")
    click.echo("")
    for insight in report.insights:
        expected = f" expected={insight.expected_row_count}" if insight.expected_row_count is not None else ""
        click.echo(f"{insight.insight_name}: {insight.verdict} rows={insight.row_count}{expected}")
        if insight.missing_count or insight.stale_count or insight.orphan_count or insight.legacy_incompatible_count:
            click.echo(
                "  "
                f"missing={insight.missing_count} stale={insight.stale_count} "
                f"orphan={insight.orphan_count} legacy={insight.legacy_incompatible_count}"
            )
        if insight.ready_flags:
            flags = ", ".join(f"{key}={value}" for key, value in sorted(insight.ready_flags.items()))
            click.echo(f"  flags: {flags}")
        if insight.provider_coverage:
            providers = ", ".join(
                f"{coverage.provider_name}={coverage.row_count}" for coverage in insight.provider_coverage
            )
            click.echo(f"  providers: {providers}")
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


@insights_command.command("status")
@click.option("--insight", "insights", multiple=True, help="Insight readiness target. May be repeated.")
@click.option("--provider", default=None, help="Limit provider coverage details to one provider.")
@click.option("--since", default=None, help="Limit coverage details to rows at/after this timestamp or date.")
@click.option("--until", default=None, help="Limit coverage details to rows at/before this timestamp or date.")
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.pass_context
def insights_status_command(
    ctx: click.Context,
    insights: tuple[str, ...],
    provider: str | None,
    since: str | None,
    until: str | None,
    output_format: str | None,
) -> None:
    """Report insight materialization coverage and readiness."""
    env: AppEnv = ctx.obj
    root_params = ctx.find_root().params
    inherited_provider = provider if provider is not None else root_params.get("provider")
    inherited_since = since if since is not None else root_params.get("since")
    inherited_until = until if until is not None else root_params.get("until")
    try:
        filters = normalize_insight_query_kwargs(
            {
                "provider": inherited_provider,
                "since": inherited_since,
                "until": inherited_until,
            }
        )
        query = InsightReadinessQuery(
            insights=insights,
            provider=filters["provider"] if isinstance(filters["provider"], str) else None,
            since=filters["since"] if isinstance(filters["since"], str) else None,
            until=filters["until"] if isinstance(filters["until"], str) else None,
        )
        report = run_coroutine_sync(env.operations.get_insight_readiness_report(query))
    except (InsightCommandInputError, ValueError) as exc:
        valid = ", ".join(known_insight_readiness_names())
        fail("insights status", f"{exc}. Known insights: {valid}")
    if _status_wants_json(ctx, output_format=output_format):
        emit_success(report.model_dump(mode="json"))
        return
    _render_status_plain(report)


@insights_command.command("export")
@click.option("--out", "output_path", required=True, type=click.Path(path_type=Path), help="Output bundle directory.")
@click.option("--insight", "insights", multiple=True, help="Insight to include. Defaults to all exportable insights.")
@click.option("--provider", default=None, help="Limit supported insights to one provider.")
@click.option("--since", default=None, help="Limit supported insights to rows at/after this timestamp or date.")
@click.option("--until", default=None, help="Limit supported insights to rows at/before this timestamp or date.")
@click.option("--bundle-format", type=click.Choice(["jsonl"]), default="jsonl", show_default=True)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.option(
    "--overwrite", is_flag=True, help="Replace an existing bundle directory after writing a complete new one."
)
@click.pass_context
def insights_export_command(
    ctx: click.Context,
    output_path: Path,
    insights: tuple[str, ...],
    provider: str | None,
    since: str | None,
    until: str | None,
    bundle_format: str,
    output_format: str | None,
    overwrite: bool,
) -> None:
    """Export versioned archive-insight bundles."""
    env: AppEnv = ctx.obj
    root_params = ctx.find_root().params
    inherited_provider = provider if provider is not None else root_params.get("provider")
    inherited_since = since if since is not None else root_params.get("since")
    inherited_until = until if until is not None else root_params.get("until")
    try:
        export_format: InsightExportFormat = "jsonl"
        if bundle_format != "jsonl":
            fail("insights export", f"unsupported export format: {bundle_format}")
        filters = normalize_insight_query_kwargs(
            {
                "provider": inherited_provider,
                "since": inherited_since,
                "until": inherited_until,
            }
        )
        request = InsightExportBundleRequest(
            output_path=output_path,
            insights=insights,
            provider=filters["provider"] if isinstance(filters["provider"], str) else None,
            since=filters["since"] if isinstance(filters["since"], str) else None,
            until=filters["until"] if isinstance(filters["until"], str) else None,
            output_format=export_format,
            overwrite=overwrite,
        )
        result = run_coroutine_sync(env.operations.export_insight_bundle(request))
    except (InsightCommandInputError, InsightExportBundleError) as exc:
        fail("insights export", str(exc))
    if output_format == "json" or ctx.find_root().params.get("output_format") == "json":
        emit_success(result.model_dump(mode="json"))
        return
    _render_export_plain(result)


# Register all insight types as subcommands
for _pt in INSIGHT_REGISTRY.values():
    if _pt.query_model is not None and _pt.operations_method_name:
        insights_command.add_command(_build_insight_command(_pt))


__all__ = ["insights_command"]
