"""Shared query-completion metadata for CLI, API, and MCP adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from polylogue.archive.query.metadata import (
    COUNT_QUERY_FIELD_REGISTRY,
    DATE_QUERY_FIELD_REGISTRY,
    EXPRESSION_FIELD_REGISTRY,
    count_query_fields,
    count_query_operators,
    date_query_fields,
    date_query_operators,
    query_unit_descriptor,
    structural_query_field_info,
    structural_query_fields,
    structural_query_units,
    terminal_query_field_info,
    terminal_query_fields,
    terminal_query_sources,
    terminal_query_unit,
)
from polylogue.operations.action_contracts import ACTION_CONTRACTS, CliActionContract

CompletionKind = str
CandidateProvider = Callable[[], list["QueryCompletionCandidate"]]

QUERY_COMPLETION_KINDS: tuple[str, ...] = (
    "field",
    "structural-unit",
    "structural-field",
    "terminal-source",
    "terminal-field",
    "count-operator",
    "date-operator",
    "action",
)


class QueryCompletionError(ValueError):
    """Raised when a completion request lacks required context."""


@dataclass(frozen=True)
class QueryCompletionCandidate:
    """Structured query-completion candidate shared by non-shell adapters."""

    value: str
    insert: str
    display: str
    kind: str
    group: str
    description: str
    source: str
    replace_start: int | None = None
    replace_end: int | None = None
    stale: bool = False
    danger: bool = False
    score: float = 1.0
    payload_model: str | None = None
    unsupported_reason: str | None = None
    preview_command: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the structured candidate payload for machine consumers."""

        return {
            "value": self.value,
            "insert": self.insert,
            "replace_start": self.replace_start,
            "replace_end": self.replace_end,
            "display": self.display,
            "kind": self.kind,
            "group": self.group,
            "description": self.description,
            "score": self.score,
            "source": self.source,
            "stale": self.stale,
            "danger": self.danger,
            "payload_model": self.payload_model,
            "unsupported_reason": self.unsupported_reason,
            "preview_command": self.preview_command,
        }


def query_field_candidates(incomplete: str) -> list[QueryCompletionCandidate]:
    """Return DSL field candidates from the shared expression registry."""

    current = incomplete.strip().lstrip("-").lower()
    if ":" in current:
        return []
    candidates: list[QueryCompletionCandidate] = []
    emitted: set[str] = set()
    for field_name, info in sorted(EXPRESSION_FIELD_REGISTRY.items()):
        if current and not field_name.startswith(current):
            continue
        insert = f"{field_name}:"
        description = info.get("description", "")
        example = info.get("example")
        if example:
            description = f"{description} Example: {example}" if description else f"Example: {example}"
        source = "EXPRESSION_FIELD_REGISTRY"
        count_info = COUNT_QUERY_FIELD_REGISTRY.get(field_name)
        if count_info is not None:
            operators = ", ".join((*count_info.operators, count_info.range_keyword))
            description = (
                f"{description} Readable operators: {operators}. Example: {count_info.example}"
                if description
                else f"Readable operators: {operators}. Example: {count_info.example}"
            )
            source = "EXPRESSION_FIELD_REGISTRY/COUNT_QUERY_FIELD_REGISTRY"
        candidates.append(
            QueryCompletionCandidate(
                value=field_name,
                insert=insert,
                display=insert,
                kind="query-field",
                group="query fields",
                description=description,
                source=source,
            )
        )
        emitted.add(field_name)
    for field_name, date_info in sorted(DATE_QUERY_FIELD_REGISTRY.items()):
        if field_name in emitted:
            continue
        if current and not field_name.startswith(current):
            continue
        operators = ", ".join((*date_info.operators, date_info.range_keyword))
        description = f"{date_info.description} Readable operators: {operators}. Example: {date_info.example}"
        candidates.append(
            QueryCompletionCandidate(
                value=field_name,
                insert=f"{field_name} ",
                display=f"{field_name} ",
                kind="query-date-field",
                group="query readable fields",
                description=description,
                source="DATE_QUERY_FIELD_REGISTRY",
            )
        )
    return candidates


def query_structural_unit_candidates(incomplete: str) -> list[QueryCompletionCandidate]:
    """Return ``exists <unit>(...)`` candidates from the grammar registry."""

    current = incomplete.strip().lower()
    candidates: list[QueryCompletionCandidate] = []
    for unit in structural_query_units():
        if current and not unit.startswith(current):
            continue
        descriptor = query_unit_descriptor(unit)
        description = "Structural query unit."
        if descriptor is not None:
            description = descriptor.description
            if descriptor.example:
                description = (
                    f"{description} Example: {descriptor.example}" if description else f"Example: {descriptor.example}"
                )
        candidates.append(
            QueryCompletionCandidate(
                value=unit,
                insert=f"{unit}(",
                display=f"{unit}(",
                kind="query-structural-unit",
                group="query structural units",
                description=description,
                source="QUERY_UNIT_DESCRIPTORS",
                payload_model=descriptor.payload_model if descriptor is not None else None,
            )
        )
    return candidates


def query_structural_field_candidates(unit: str, incomplete: str) -> list[QueryCompletionCandidate]:
    """Return field candidates accepted inside ``exists <unit>(...)``."""

    current = incomplete.strip().lstrip("-").lower()
    if ":" in current:
        return []
    descriptor = query_unit_descriptor(unit)
    candidates: list[QueryCompletionCandidate] = []
    for field_name in structural_query_fields(unit):
        if current and not field_name.startswith(current):
            continue
        info = structural_query_field_info(unit, field_name)
        description = f"Field accepted inside exists {unit}(...)."
        if info is not None:
            description = info.description
            if info.example:
                description = f"{description} Example: {info.example}"
        candidates.append(
            QueryCompletionCandidate(
                value=field_name,
                insert=f"{field_name}:",
                display=f"{field_name}:",
                kind="query-structural-field",
                group=f"{unit} structural fields",
                description=description,
                source="QUERY_UNIT_DESCRIPTORS",
                payload_model=descriptor.payload_model if descriptor is not None else None,
            )
        )
    return candidates


def query_terminal_source_candidates(incomplete: str) -> list[QueryCompletionCandidate]:
    """Return ``<source> where`` candidates for terminal row queries."""

    current = incomplete.strip().lower()
    candidates: list[QueryCompletionCandidate] = []
    for source in terminal_query_sources():
        if current and not source.startswith(current):
            continue
        unit = terminal_query_unit(source)
        descriptor = query_unit_descriptor(unit or "")
        description = "Terminal query row source."
        if descriptor is not None:
            description = descriptor.description
            if descriptor.example:
                description = f"{description} Example: {descriptor.example}"
        candidates.append(
            QueryCompletionCandidate(
                value=source,
                insert=f"{source} where ",
                display=f"{source} where",
                kind="query-terminal-source",
                group="query terminal sources",
                description=description,
                source="QUERY_UNIT_DESCRIPTORS",
                payload_model=descriptor.payload_model if descriptor is not None else None,
            )
        )
    return candidates


def query_terminal_field_candidates(source: str, incomplete: str) -> list[QueryCompletionCandidate]:
    """Return field candidates accepted after ``<source> where``."""

    current = incomplete.strip().lstrip("-").lower()
    if ":" in current:
        return []
    descriptor = query_unit_descriptor(source)
    candidates: list[QueryCompletionCandidate] = []
    for field_name in terminal_query_fields(source):
        if current and not field_name.startswith(current):
            continue
        info = terminal_query_field_info(source, field_name)
        description = f"Field accepted after {source} where."
        if info is not None:
            description = info.description
            if info.example:
                description = f"{description} Example: {info.example}"
        candidates.append(
            QueryCompletionCandidate(
                value=field_name,
                insert=f"{field_name}:",
                display=f"{field_name}:",
                kind="query-terminal-field",
                group=f"{source} terminal fields",
                description=description,
                source="QUERY_UNIT_DESCRIPTORS",
                payload_model=descriptor.payload_model if descriptor is not None else None,
            )
        )
    return candidates


def query_count_operator_candidates(field: str, incomplete: str) -> list[QueryCompletionCandidate]:
    """Return readable count operators accepted by the query grammar."""

    field_name = field.lower()
    if field_name not in count_query_fields():
        return []
    current = incomplete.strip().lower()
    info = COUNT_QUERY_FIELD_REGISTRY[field_name]
    candidates: list[QueryCompletionCandidate] = []
    for operator in count_query_operators(field_name):
        if current and not operator.startswith(current):
            continue
        insert = f"{operator} " if operator == info.range_keyword else operator
        candidates.append(
            QueryCompletionCandidate(
                value=operator,
                insert=insert,
                display=insert,
                kind="query-count-operator",
                group=f"{field_name} count operators",
                description=f"{info.description} Example: {info.example}",
                source="COUNT_QUERY_FIELD_REGISTRY",
            )
        )
    return candidates


def query_date_operator_candidates(field: str, incomplete: str) -> list[QueryCompletionCandidate]:
    """Return readable date operators accepted by the query grammar."""

    field_name = field.lower()
    if field_name not in date_query_fields():
        return []
    current = incomplete.strip().lower()
    info = DATE_QUERY_FIELD_REGISTRY[field_name]
    candidates: list[QueryCompletionCandidate] = []
    for operator in date_query_operators(field_name):
        if current and not operator.startswith(current):
            continue
        insert = f"{operator} " if operator == info.range_keyword else operator
        candidates.append(
            QueryCompletionCandidate(
                value=operator,
                insert=insert,
                display=insert,
                kind="query-date-operator",
                group=f"{field_name} date operators",
                description=f"{info.description} Example: {info.example}",
                source="DATE_QUERY_FIELD_REGISTRY",
            )
        )
    return candidates


def _action_description(contract: CliActionContract) -> str:
    pieces = [
        f"effect={contract.effect}",
        f"input={contract.input_unit}",
        f"cardinality={contract.cardinality}",
        f"default_format={contract.default_format}",
        f"machine_envelope={contract.machine_envelope}",
    ]
    if contract.guards:
        pieces.append(f"guards={','.join(contract.guards)}")
    return "; ".join(pieces)


def query_action_candidates(incomplete: str) -> list[QueryCompletionCandidate]:
    """Return root query/action candidates from public action contracts."""

    current = incomplete.strip().lower()
    candidates: list[QueryCompletionCandidate] = []
    for contract in ACTION_CONTRACTS:
        if len(contract.path) != 1:
            continue
        name = contract.path[0]
        if current and not name.startswith(current):
            continue
        candidates.append(
            QueryCompletionCandidate(
                value=name,
                insert=name,
                display=name,
                kind="query-action",
                group="query actions",
                description=_action_description(contract),
                source="ACTION_CONTRACTS",
                danger=contract.effect == "destructive",
            )
        )
    return candidates


def query_completion_candidates(
    kind: CompletionKind,
    *,
    incomplete: str = "",
    unit: str | None = None,
    field: str | None = None,
) -> list[QueryCompletionCandidate]:
    """Return completion candidates for a shared query-completion request."""

    if kind == "field":
        return query_field_candidates(incomplete)
    if kind == "structural-unit":
        return query_structural_unit_candidates(incomplete)
    if kind == "structural-field":
        if unit is None:
            raise QueryCompletionError("--unit is required for structural-field completion")
        return query_structural_field_candidates(unit, incomplete)
    if kind == "terminal-source":
        return query_terminal_source_candidates(incomplete)
    if kind == "terminal-field":
        if unit is None:
            raise QueryCompletionError("--unit is required for terminal-field completion")
        return query_terminal_field_candidates(unit, incomplete)
    if kind == "count-operator":
        if field is None:
            raise QueryCompletionError("--field is required for count-operator completion")
        return query_count_operator_candidates(field, incomplete)
    if kind == "date-operator":
        if field is None:
            raise QueryCompletionError("--field is required for date-operator completion")
        return query_date_operator_candidates(field, incomplete)
    if kind == "action":
        return query_action_candidates(incomplete)
    raise QueryCompletionError(f"unsupported completion kind: {kind}")


def query_completion_payload(
    kind: CompletionKind,
    *,
    incomplete: str = "",
    unit: str | None = None,
    field: str | None = None,
) -> dict[str, object]:
    """Return the stable completion payload shared by public adapters."""

    candidates = query_completion_candidates(kind, incomplete=incomplete, unit=unit, field=field)
    return {
        "kind": kind,
        "incomplete": incomplete,
        "unit": unit,
        "field": field,
        "candidates": [candidate.to_payload() for candidate in candidates],
    }


__all__ = [
    "QUERY_COMPLETION_KINDS",
    "QueryCompletionCandidate",
    "QueryCompletionError",
    "query_action_candidates",
    "query_completion_candidates",
    "query_completion_payload",
    "query_count_operator_candidates",
    "query_date_operator_candidates",
    "query_field_candidates",
    "query_structural_field_candidates",
    "query_structural_unit_candidates",
    "query_terminal_field_candidates",
    "query_terminal_source_candidates",
]
