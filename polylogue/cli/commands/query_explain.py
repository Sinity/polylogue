"""Explain query DSL parsing, AST, and lowering."""

from __future__ import annotations

import json
from typing import cast

import click

from polylogue.archive.query.expression import explain_expression


class QueryExplainCommand(click.Command):
    """Click command that treats query words as opaque DSL input."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if any(arg in {"--help", "-h"} for arg in args):
            return super().parse_args(ctx, args)

        output_format = "plain"
        expression: list[str] = []
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "--":
                expression.extend(args[index + 1 :])
                break
            if arg == "--format":
                index += 1
                if index >= len(args):
                    raise click.UsageError("--format requires an argument")
                output_format = args[index]
            elif arg.startswith("--format="):
                output_format = arg.split("=", 1)[1]
            else:
                expression.append(arg)
            index += 1

        choices = {"plain", "json"}
        if output_format not in choices:
            raise click.BadParameter(
                f"{output_format!r} is not one of {'; '.join(sorted(choices))}",
                param_hint="--format",
            )
        if not expression:
            raise click.UsageError("Missing argument 'EXPRESSION...'.")

        ctx.params["output_format"] = output_format
        ctx.params["expression"] = tuple(expression)
        return []


@click.command("query-explain", cls=QueryExplainCommand)
@click.argument("expression", nargs=-1, required=True, type=click.UNPROCESSED)
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
    selected_units = cast(list[str], payload["selected_units"])
    execution_legs = cast(list[str], payload["execution_legs"])
    plan_description = cast(list[str], payload["plan_description"])
    unsupported_nodes = cast(list[str], payload["unsupported_nodes"])
    if selected_units:
        click.echo("units: " + ", ".join(selected_units))
    if execution_legs:
        click.echo("execution legs: " + ", ".join(execution_legs))
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
