"""Resolve the structured incident coverage contract for the 818fy campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator

from devtools import repo_root

ROOT = repo_root()
LEDGER_PATH = ROOT / "docs" / "plans" / "reindex-incident-coverage.json"
SCHEMA_PATH = ROOT / "docs" / "plans" / "reindex-incident-coverage.schema.json"
CAMPAIGN_GRAPH_PATH = ROOT / "tests" / "fixtures" / "reindex_incident_coverage" / "campaign_graph.json"

JsonObject = dict[str, object]


class IncidentCoverageLedgerError(ValueError):
    """Raised when the ledger or its campaign graph is incomplete."""


@dataclass(frozen=True, slots=True)
class CoverageResolution:
    """The useful summary of a successfully resolved coverage ledger."""

    target_bead_id: str
    forcing_dependency_ids: tuple[str, ...]
    ledger_row_count: int
    closed_implementation_ids: tuple[str, ...]
    successor_backed_ids: tuple[str, ...]


def _load_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncidentCoverageLedgerError(f"cannot load structured artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IncidentCoverageLedgerError(f"structured artifact {path} must contain an object")
    return cast(JsonObject, value)


def load_ledger(
    path: Path = LEDGER_PATH,
    *,
    schema_path: Path = SCHEMA_PATH,
) -> JsonObject:
    """Load and JSON-Schema-validate the versioned ledger document."""

    ledger = _load_json(path)
    schema = _load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(ledger), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "$"
        raise IncidentCoverageLedgerError(f"ledger schema error at {location}: {first.message}")
    return ledger


def load_campaign_graph(path: Path = CAMPAIGN_GRAPH_PATH) -> JsonObject:
    """Load the structured snapshot of the current 818fy forcing graph."""

    return _load_json(path)


def _object(value: object, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise IncidentCoverageLedgerError(f"{context} must be an object")
    return cast(JsonObject, value)


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise IncidentCoverageLedgerError(f"{context} must be a non-empty string")
    return value


def _strings(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise IncidentCoverageLedgerError(f"{context} must be a list of non-empty strings")
    return tuple(cast(str, item) for item in value)


def _catalog(ledger: JsonObject, name: str) -> dict[str, JsonObject]:
    value = ledger.get(name)
    if not isinstance(value, dict):
        raise IncidentCoverageLedgerError(f"ledger catalog {name!r} must be an object")
    return {str(key): _object(item, context=f"ledger catalog {name}.{key}") for key, item in value.items()}


def _graph_dependencies(graph: JsonObject) -> tuple[dict[str, object], ...]:
    target = _string(graph.get("target_bead_id"), context="campaign graph target_bead_id")
    if target != "polylogue-818fy":
        raise IncidentCoverageLedgerError(f"campaign graph target {target!r} is not polylogue-818fy")
    raw_dependencies = graph.get("forcing_dependencies")
    if not isinstance(raw_dependencies, list):
        raise IncidentCoverageLedgerError("campaign graph forcing_dependencies must be a list")
    dependencies: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw_dependency in enumerate(raw_dependencies):
        dependency = _object(raw_dependency, context=f"campaign graph dependency {index}")
        bead_id = _string(dependency.get("bead_id"), context=f"campaign graph dependency {index}.bead_id")
        if bead_id in seen:
            raise IncidentCoverageLedgerError(f"duplicate forcing dependency {bead_id}")
        seen.add(bead_id)
        _string(dependency.get("status"), context=f"campaign graph dependency {bead_id}.status")
        _string(dependency.get("kind"), context=f"campaign graph dependency {bead_id}.kind")
        child_ids = _strings(
            dependency.get("child_bead_ids"),
            context=f"campaign graph dependency {bead_id}.child_bead_ids",
        )
        known_children = _strings(graph.get("known_child_bead_ids"), context="campaign graph known_child_bead_ids")
        unknown_children = sorted(set(child_ids) - set(known_children))
        if unknown_children:
            raise IncidentCoverageLedgerError(
                f"campaign graph dependency {bead_id} names unknown child beads {unknown_children}"
            )
        dependencies.append(dependency)
    return tuple(dependencies)


def resolve_incident_coverage(ledger: JsonObject, graph: JsonObject) -> CoverageResolution:
    """Resolve row completeness and all structured references for one campaign graph."""

    target = _string(ledger.get("target_bead_id"), context="ledger target_bead_id")
    dependencies = _graph_dependencies(graph)
    if target != "polylogue-818fy":
        raise IncidentCoverageLedgerError(f"ledger target {target!r} is not polylogue-818fy")
    if target != _string(graph.get("target_bead_id"), context="campaign graph target_bead_id"):
        raise IncidentCoverageLedgerError("ledger and campaign graph target beads differ")

    fixtures = _catalog(ledger, "fixtures")
    checks = _catalog(ledger, "checks")
    snapshots = _catalog(ledger, "snapshots")
    receipts = _catalog(ledger, "receipts")
    successors = _catalog(ledger, "successors")

    raw_rows = ledger.get("rows")
    if not isinstance(raw_rows, list):
        raise IncidentCoverageLedgerError("ledger rows must be a list")
    rows = tuple(_object(row, context=f"ledger row {index}") for index, row in enumerate(raw_rows))
    row_ids = tuple(_string(row.get("bead_id"), context="ledger row bead_id") for row in rows)
    duplicate_ids = sorted({bead_id for bead_id in row_ids if row_ids.count(bead_id) > 1})
    if duplicate_ids:
        raise IncidentCoverageLedgerError(f"ledger has duplicate rows for {duplicate_ids}")

    dependency_ids = tuple(_string(dep.get("bead_id"), context="forcing dependency bead_id") for dep in dependencies)
    missing = sorted(set(dependency_ids) - set(row_ids))
    extra = sorted(set(row_ids) - set(dependency_ids))
    if missing or extra or len(rows) != len(dependencies):
        raise IncidentCoverageLedgerError(
            f"ledger rows do not match forcing dependencies: missing={missing}, extra={extra}, "
            f"expected={len(dependencies)}, actual={len(rows)}"
        )

    graph_by_id = {str(dep["bead_id"]): dep for dep in dependencies}
    closed_implementation_ids: list[str] = []
    successor_backed_ids: list[str] = []
    orders: list[int] = []
    for row in rows:
        bead_id = _string(row.get("bead_id"), context="ledger row bead_id")
        graph_entry = graph_by_id[bead_id]
        if row.get("bead_status") != graph_entry.get("status"):
            raise IncidentCoverageLedgerError(f"ledger status disagrees with graph for {bead_id}")

        incident = _object(row.get("incident"), context=f"ledger row {bead_id}.incident")
        if _string(incident.get("bead_id"), context=f"ledger row {bead_id}.incident.bead_id") != bead_id:
            raise IncidentCoverageLedgerError(f"incident bead reference disagrees for {bead_id}")

        schedule = _object(row.get("schedule"), context=f"ledger row {bead_id}.schedule")
        order = schedule.get("order")
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise IncidentCoverageLedgerError(f"schedule order is invalid for {bead_id}")
        orders.append(order)

        expected_snapshot = _object(row.get("expected_snapshot"), context=f"ledger row {bead_id}.expected_snapshot")
        snapshot_id = _string(
            expected_snapshot.get("snapshot_id"),
            context=f"ledger row {bead_id}.expected_snapshot.snapshot_id",
        )
        if snapshot_id not in snapshots:
            raise IncidentCoverageLedgerError(f"unknown snapshot {snapshot_id} for {bead_id}")

        red_mutation = _object(row.get("red_mutation"), context=f"ledger row {bead_id}.red_mutation")
        fixture_id = _string(red_mutation.get("fixture_id"), context=f"ledger row {bead_id}.red_mutation.fixture_id")
        if fixture_id not in fixtures:
            raise IncidentCoverageLedgerError(f"unknown fixture {fixture_id} for {bead_id}")

        check_ids = _strings(row.get("registry_checks"), context=f"ledger row {bead_id}.registry_checks")
        unknown_checks = sorted(set(check_ids) - set(checks))
        if unknown_checks:
            raise IncidentCoverageLedgerError(f"unknown checks {unknown_checks} for {bead_id}")

        receipt_ids = _strings(row.get("receipts"), context=f"ledger row {bead_id}.receipts")
        unknown_receipts = sorted(set(receipt_ids) - set(receipts))
        if unknown_receipts:
            raise IncidentCoverageLedgerError(f"unknown receipts {unknown_receipts} for {bead_id}")

        successor = row.get("residual_successor")
        successor_id: str | None = None
        if successor is not None:
            successor_object = _object(successor, context=f"ledger row {bead_id}.residual_successor")
            successor_id = _string(
                successor_object.get("bead_id"), context=f"ledger row {bead_id}.residual_successor.bead_id"
            )
            if successor_id not in successors:
                raise IncidentCoverageLedgerError(f"unknown successor {successor_id} for {bead_id}")
            child_ids = _strings(
                graph_entry.get("child_bead_ids"), context=f"campaign graph dependency {bead_id}.child_bead_ids"
            )
            if successor_id not in child_ids:
                raise IncidentCoverageLedgerError(f"successor {successor_id} is not a named child of {bead_id}")
            successor_backed_ids.append(bead_id)

        if graph_entry.get("status") == "closed" and graph_entry.get("kind") == "implementation":
            closed_implementation_ids.append(bead_id)
            live_proof = any(receipts[receipt_id].get("kind") == "live-proof" for receipt_id in receipt_ids)
            if not live_proof and successor_id is None:
                raise IncidentCoverageLedgerError(
                    f"closed implementation bead {bead_id} has no live proof or named child successor"
                )

    if len(set(orders)) != len(orders) or set(orders) != set(range(1, len(rows) + 1)):
        raise IncidentCoverageLedgerError("ledger schedule orders must be a permutation of 1..row_count")

    return CoverageResolution(
        target_bead_id=target,
        forcing_dependency_ids=dependency_ids,
        ledger_row_count=len(rows),
        closed_implementation_ids=tuple(closed_implementation_ids),
        successor_backed_ids=tuple(successor_backed_ids),
    )


def resolve_default_incident_coverage() -> CoverageResolution:
    """Load and resolve the committed 818fy ledger and graph fixture."""

    return resolve_incident_coverage(load_ledger(), load_campaign_graph())


__all__ = [
    "CAMPAIGN_GRAPH_PATH",
    "CoverageResolution",
    "IncidentCoverageLedgerError",
    "LEDGER_PATH",
    "SCHEMA_PATH",
    "load_campaign_graph",
    "load_ledger",
    "resolve_default_incident_coverage",
    "resolve_incident_coverage",
]
