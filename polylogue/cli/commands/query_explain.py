"""Explain query DSL parsing, AST, and lowering."""

from __future__ import annotations

import json
from typing import cast

import click

from polylogue.archive.query.expression import explain_expression


@click.command("query-explain")
@click.argument("expression", nargs=-1, required=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def query_explain_command(expression: tuple[str, ...], output_format: str) -> None:
    """Explain how a query expression parses and lowers before execution."""

    source = " ".join(expression)
    explanation = explain_expression(source)
    payload = explanation.to_payload()
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"query: {payload['source_text']}")
    click.echo(f"lowerer: {payload['lowerer']}")
    predicate = payload["predicate"]
    clauses = cast(list[object], payload["clauses"])
    plan_description = cast(list[str], payload["plan_description"])
    unsupported_nodes = cast(list[str], payload["unsupported_nodes"])
    if predicate is not None:
        click.echo("predicate:")
        click.echo(json.dumps(predicate, indent=2, sort_keys=True))
    elif clauses:
        click.echo("clauses:")
        for clause in clauses:
            click.echo(f"  - {json.dumps(clause, sort_keys=True)}")
    if plan_description:
        click.echo("plan:")
        for line in plan_description:
            click.echo(f"  - {line}")
    if unsupported_nodes:
        click.echo("unsupported:")
        for node in unsupported_nodes:
            click.echo(f"  - {node}")


__all__ = ["query_explain_command"]
