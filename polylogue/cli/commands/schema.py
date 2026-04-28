"""Schema package inspection and comparison commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import click

from polylogue.cli.schema_rendering import (
    render_schema_compare_result,
    render_schema_explain_result,
    render_schema_list_result,
)
from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.schemas.operator_models import (
    SchemaCompareRequest,
    SchemaExplainRequest,
    SchemaListRequest,
)
from polylogue.schemas.operator_workflow import (
    compare_schema_versions,
    explain_schema,
    list_schemas,
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
    """Inspect schema packages, versions, and evidence."""
    del ctx


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


__all__ = ["schema_command"]
