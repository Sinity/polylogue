"""Schema generation, package inspection, comparison, and audit commands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

from polylogue.cli.helpers import fail
from polylogue.cli.schema_command_support import build_schema_privacy_config
from polylogue.cli.schema_rendering import (
    render_schema_audit_result,
    render_schema_compare_result,
    render_schema_explain_result,
    render_schema_generate_result,
    render_schema_list_result,
    render_schema_promote_result,
)
from polylogue.cli.types import AppEnv
from polylogue.schemas.operator_models import (
    SchemaAuditRequest,
    SchemaCompareRequest,
    SchemaExplainRequest,
    SchemaInferRequest,
    SchemaListRequest,
    SchemaPromoteRequest,
)
from polylogue.schemas.operator_workflow import (
    audit_schemas,
    compare_schema_versions,
    explain_schema,
    infer_schema,
    list_schemas,
    promote_schema_cluster,
)

TResult = TypeVar("TResult")


def _run_schema_action(command_name: str, action: Callable[[], TResult]) -> TResult:
    try:
        return action()
    except ValueError as exc:
        fail(command_name, str(exc))


@click.group("schema")
@click.pass_context
def schema_command(ctx: click.Context) -> None:
    """Schema generation, package versioning, and evidence inspection."""
    del ctx


@schema_command.command("generate")
@click.option("--provider", required=True, help="Provider to generate schema for")
@click.option("--cluster", is_flag=True, help="Also cluster observed samples by structural fingerprint")
@click.option("--max-samples", type=int, default=None, help="Limit samples for generation")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option(
    "--privacy",
    type=click.Choice(["strict", "standard", "permissive"], case_sensitive=False),
    default=None,
    help="Privacy preset level (default: standard)",
)
@click.option(
    "--privacy-config",
    "privacy_config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to TOML privacy config overrides",
)
@click.option("--report", is_flag=True, help="Write a redaction report alongside the schema")
@click.option("--full-corpus", is_flag=True, help="Bypass all sample caps for full-corpus schema generation")
@click.pass_obj
def schema_generate(
    env: AppEnv,
    provider: str,
    cluster: bool,
    max_samples: int | None,
    json_output: bool,
    privacy: str | None,
    privacy_config_path: Path | None,
    report: bool,
    full_corpus: bool,
) -> None:
    """Generate provider schema packages and optional evidence clusters."""
    privacy_config = build_schema_privacy_config(
        privacy=privacy,
        privacy_config_path=privacy_config_path,
    )

    result = infer_schema(
        SchemaInferRequest(
            provider=provider,
            db_path=env.config.db_path,
            max_samples=max_samples,
            privacy_config=privacy_config,
            cluster=cluster,
            full_corpus=full_corpus,
        )
    )
    generation = result.generation
    if not generation.success:
        fail("schema generate", generation.error or "Schema generation failed")
    if cluster and result.manifest is None:
        fail("schema generate", "No samples found for clustering")

    render_schema_generate_result(
        provider=provider,
        result=result,
        json_output=json_output,
        report=report,
    )


@schema_command.command("list")
@click.option("--provider", default=None, help="Filter to specific provider")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_list(env: AppEnv, provider: str | None, json_output: bool) -> None:
    """List available schema packages, versions, and evidence manifests."""
    del env
    result = list_schemas(SchemaListRequest(provider=provider))
    render_schema_list_result(provider=provider, result=result, json_output=json_output)


@schema_command.command("compare")
@click.option("--provider", required=True, help="Provider name")
@click.option("--from", "from_version", required=True, help="Source version (e.g., v1)")
@click.option("--to", "to_version", required=True, help="Target version (e.g., v2)")
@click.option("--element", "element_kind", default=None, help="Element kind inside the package")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--markdown", "md_output", is_flag=True, help="Output as Markdown")
@click.pass_obj
def schema_compare(
    env: AppEnv,
    provider: str,
    from_version: str,
    to_version: str,
    element_kind: str | None,
    json_output: bool,
    md_output: bool,
) -> None:
    """Compare two schema package versions for a provider."""
    del env
    result = _run_schema_action(
        "schema compare",
        lambda: compare_schema_versions(
            SchemaCompareRequest(
                provider=provider,
                from_version=from_version,
                to_version=to_version,
                element_kind=element_kind,
            )
        ),
    )

    render_schema_compare_result(result=result, json_output=json_output, md_output=md_output)


@schema_command.command("promote")
@click.option("--provider", required=True, help="Provider name")
@click.option("--cluster", "cluster_id", required=True, help="Evidence cluster ID to promote")
@click.option("--with-samples", is_flag=True, help="Re-load samples for full schema generation")
@click.option("--max-samples", type=int, default=500, help="Max samples when using --with-samples")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_promote(
    env: AppEnv,
    provider: str,
    cluster_id: str,
    with_samples: bool,
    max_samples: int,
    json_output: bool,
) -> None:
    """Promote an evidence cluster to a new registered package version."""
    result = _run_schema_action(
        "schema promote",
        lambda: promote_schema_cluster(
            SchemaPromoteRequest(
                provider=provider,
                cluster_id=cluster_id,
                db_path=env.config.db_path,
                with_samples=with_samples,
                max_samples=max_samples,
            )
        ),
    )

    render_schema_promote_result(result=result, json_output=json_output)


@schema_command.command("explain")
@click.option("--provider", required=True, help="Provider name")
@click.option("--version", default="latest", help="Schema version (default: latest)")
@click.option("--element", "element_kind", default=None, help="Element kind inside the package")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show semantic roles and coverage")
@click.option("--proof", is_flag=True, help="Show proof surface for role assignment decisions")
@click.pass_obj
def schema_explain(
    env: AppEnv,
    provider: str,
    version: str,
    element_kind: str | None,
    json_output: bool,
    verbose: bool,
    proof: bool,
) -> None:
    """Explain a package element schema with evidence and annotations."""
    del env
    result = _run_schema_action(
        "schema explain",
        lambda: explain_schema(
            SchemaExplainRequest(
                provider=provider,
                version=version,
                element_kind=element_kind,
                proof=proof,
            )
        ),
    )

    render_schema_explain_result(result=result, json_output=json_output, verbose=verbose)


@schema_command.command("audit")
@click.option("--provider", default=None, help="Audit a specific provider (default: all)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_audit(
    env: AppEnv,
    provider: str | None,
    json_output: bool,
) -> None:
    """Run automated quality checks on committed schema packages."""
    del env
    report = audit_schemas(SchemaAuditRequest(provider=provider))
    render_schema_audit_result(report=report, json_output=json_output)
    if not report.all_passed:
        raise SystemExit(1)


__all__ = ["schema_command"]
