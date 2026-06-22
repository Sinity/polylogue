"""Render the public CLI action-contract report for generated docs."""

from __future__ import annotations

from collections.abc import Iterable

from devtools.render_cli_output_schemas import SCHEMAS, CliOutputSchema
from polylogue.operations.action_contracts import ACTION_CONTRACTS


def _path_cell(path: tuple[str, ...]) -> str:
    return f"`polylogue {' '.join(path)}`"


def _code_cell(value: str) -> str:
    return f"`{value}`"


def _list_cell(values: Iterable[str]) -> str:
    rendered = tuple(_code_cell(value) for value in values)
    return ", ".join(rendered) if rendered else "-"


def _format_set(values: frozenset[str]) -> tuple[str, ...]:
    order = {"human": 0, "json": 1, "ndjson": 2}
    return tuple(sorted(values, key=order.__getitem__))


def _contract_rows() -> list[str]:
    rows = [
        "| Action | Effect | Target | Input | Cardinality | Safety | Formats | Destinations | Confirm | Select | Machine envelope | Guards | Next actions | Completion |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for contract in ACTION_CONTRACTS:
        rows.append(
            " | ".join(
                (
                    _path_cell(contract.path),
                    _code_cell(contract.effect),
                    _code_cell(contract.target),
                    _code_cell(contract.input_unit),
                    _code_cell(contract.cardinality),
                    _code_cell(contract.safety_level),
                    _list_cell(_format_set(contract.formats)),
                    _list_cell(contract.destination_support),
                    _code_cell(contract.confirmation_command) if contract.confirmation_command else "-",
                    _code_cell(contract.selection_command) if contract.selection_command else "-",
                    _code_cell(contract.machine_envelope),
                    _list_cell(contract.guards),
                    _list_cell(contract.next_actions),
                    _code_cell(contract.completion_context) if contract.completion_context else "-",
                )
            ).join(("| ", " |"))
        )
    return rows


def _schema_rows(schemas: tuple[CliOutputSchema, ...] = SCHEMAS) -> list[str]:
    rows = [
        "| Schema | Model | Surfaces |",
        "| --- | --- | --- |",
    ]
    for schema in schemas:
        rows.append(
            " | ".join(
                (
                    f"`{schema.name}`",
                    _code_cell(schema.model.__name__),
                    "<br>".join(_code_cell(surface) for surface in schema.surfaces),
                )
            ).join(("| ", " |"))
        )
    return rows


def render_action_contract_report() -> str:
    """Return the generated CLI action/output contract report."""
    lines = [
        "## Public Action Contracts",
        "",
        "This section is generated from `polylogue.operations.action_contracts.ACTION_CONTRACTS`.",
        "It records the public action floor, not every utility command in the Click tree.",
        "",
        *_contract_rows(),
        "",
        "## Published Machine Output Schemas",
        "",
        "This section is generated from `devtools.render_cli_output_schemas.SCHEMAS`.",
        "The schema files live under `docs/schemas/cli-output/`.",
        "",
        *_schema_rows(),
    ]
    return "\n".join(lines).rstrip()


__all__ = ["render_action_contract_report"]
