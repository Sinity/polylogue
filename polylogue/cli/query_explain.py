"""Render query DSL parser/lowering explanations for CLI surfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import click

from polylogue.archive.query.expression import explain_expression


def explain_query_expression(
    source: str,
    *,
    output_format: str = "plain",
    terminal_action: Mapping[str, object] | None = None,
) -> None:
    """Render how a query expression parses and lowers before execution."""

    explanation = explain_expression(source)
    payload = explanation.to_payload()
    if terminal_action is not None:
        payload["terminal_action"] = dict(terminal_action)
        lowering_plan = payload.get("lowering_plan")
        if isinstance(lowering_plan, dict):
            lowering_plan["terminal_action"] = dict(terminal_action)
        plan_description = payload.get("plan_description")
        action_name = str(terminal_action.get("action") or "action")
        if isinstance(plan_description, list):
            plan_description.append(f"terminal action: {action_name}")
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
    lowering_plan = cast(dict[str, object] | None, payload["lowering_plan"])
    terminal_action_payload = cast(dict[str, object] | None, payload.get("terminal_action"))
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
    if lowering_plan is not None:
        click.echo("lowering plan:")
        click.echo(json.dumps(lowering_plan, indent=2, sort_keys=True))
    if terminal_action_payload is not None and (lowering_plan is None or "terminal_action" not in lowering_plan):
        click.echo("terminal action:")
        click.echo(json.dumps(terminal_action_payload, indent=2, sort_keys=True))
    if plan_description:
        click.echo("plan:")
        for line in plan_description:
            click.echo(f"  - {line}")
    if unsupported_nodes:
        click.echo("unsupported:")
        for node in unsupported_nodes:
            click.echo(f"  - {node}")


__all__ = ["explain_query_expression"]
