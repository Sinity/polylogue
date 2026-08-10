"""Resolve the structured incident coverage contract for the 818fy campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from jsonschema import Draft202012Validator

from devtools import repo_root

ROOT = repo_root()
LEDGER_PATH = ROOT / "docs" / "plans" / "reindex-incident-coverage.json"
SCHEMA_PATH = ROOT / "docs" / "plans" / "reindex-incident-coverage.schema.json"
CAMPAIGN_GRAPH_PATH = ROOT / "tests" / "fixtures" / "reindex_incident_coverage" / "campaign_graph.json"
BEADS_PATH = ROOT / ".beads" / "issues.jsonl"

JsonObject = dict[str, object]
DEPENDENCY_KINDS = frozenset({"blocks", "discovered-from", "parent-child", "relates-to", "supersedes"})
GRAPH_KINDS = frozenset({"decision", "design", "implementation", "operation", "verification"})
ROUTE_KINDS = frozenset({"campaign", "canary", "decision", "operation", "registry"})


class IncidentCoverageLedgerError(ValueError):
    """Raised when the ledger or its campaign graph is incomplete."""

    def __init__(self, message: str, *, diagnostic: JsonObject | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic or {"error": "incident_coverage_ledger", "message": message}


@dataclass(frozen=True, slots=True)
class CoverageResolution:
    """The useful summary of a successfully resolved coverage ledger."""

    target_bead_id: str
    forcing_dependency_ids: tuple[str, ...]
    ledger_row_count: int
    closed_implementation_ids: tuple[str, ...]
    successor_backed_ids: tuple[str, ...]


def _fail(code: str, message: str, **fields: object) -> NoReturn:
    raise IncidentCoverageLedgerError(message, diagnostic={"error": code, **fields})


def _load_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("artifact_load_failed", f"cannot load structured artifact {path}: {exc}", path=str(path))
    if not isinstance(value, dict):
        _fail("artifact_shape_invalid", f"structured artifact {path} must contain an object", path=str(path))
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
        _fail("ledger_schema_invalid", f"ledger schema error at {location}: {first.message}", location=location)
    return ledger


def load_campaign_graph(path: Path = CAMPAIGN_GRAPH_PATH) -> JsonObject:
    """Load the committed normalized snapshot of the 818fy forcing graph."""

    return _load_json(path)


def _parse_beads_jsonl(lines: list[str]) -> dict[str, JsonObject]:
    """Parse only structured Beads records and dependency fields.

    Descriptions, notes, close reasons, comments, and PR text are deliberately
    never inspected here. The JSONL is the committed source of dependency
    membership and status.
    """

    records: dict[str, JsonObject] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail("beads_json_invalid", f"invalid Beads JSONL at line {line_number}: {exc}", line=line_number)
        if not isinstance(value, dict):
            _fail("beads_record_invalid", f"Beads record at line {line_number} must be an object", line=line_number)
        record = cast(JsonObject, value)
        bead_id = _string(record.get("id"), context=f"Beads record {line_number}.id")
        if bead_id in records:
            _fail("duplicate_bead_id", f"duplicate Bead record {bead_id}", bead_id=bead_id)
        dependencies = record.get("dependencies", [])
        if not isinstance(dependencies, list):
            _fail("bead_dependencies_invalid", f"Bead {bead_id}.dependencies must be a list", bead_id=bead_id)
        for index, raw_dependency in enumerate(dependencies):
            dependency = _object(raw_dependency, context=f"Bead {bead_id}.dependencies[{index}]")
            dependency_kind = _string(dependency.get("type"), context=f"Bead {bead_id}.dependencies[{index}].type")
            if dependency_kind not in DEPENDENCY_KINDS:
                _fail(
                    "unknown_dependency_kind",
                    f"unknown dependency kind {dependency_kind!r} on {bead_id}",
                    bead_id=bead_id,
                    dependency_kind=dependency_kind,
                    allowed_dependency_kinds=sorted(DEPENDENCY_KINDS),
                )
        records[bead_id] = record
    return records


def load_beads_jsonl(path: Path = BEADS_PATH) -> dict[str, JsonObject]:
    """Load a supplied committed Beads export without invoking ``bd``."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail("beads_load_failed", f"cannot load structured Beads JSONL {path}: {exc}", path=str(path))
    return _parse_beads_jsonl(lines)


def _object(value: object, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        _fail("object_required", f"{context} must be an object", context=context)
    return cast(JsonObject, value)


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("string_required", f"{context} must be a non-empty string", context=context)
    return value


def _load_registered_value(source: str, registry: str, *, context: str) -> object:
    if not source.endswith(".py"):
        _fail("registry_source_invalid", f"{context} source must be a Python module path", source=source)
    module_name = source[:-3].replace("/", ".")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        _fail("registry_import_failed", f"cannot import {context} registry {source}: {exc}", source=source)
    value: object = module
    for component in registry.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                _fail(
                    "registry_entry_missing",
                    f"{context} registry {registry} is absent from {source}",
                    source=source,
                    registry=registry,
                )
            value = value[component]
            continue
        if not hasattr(value, component):
            _fail(
                "registry_entry_missing",
                f"{context} registry {registry} is absent from {source}",
                source=source,
                registry=registry,
            )
        value = getattr(value, component)
    return value


def _strings(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        _fail("strings_required", f"{context} must be a list of non-empty strings", context=context)
    return tuple(cast(str, item) for item in value)


def _catalog(ledger: JsonObject, name: str) -> dict[str, JsonObject]:
    value = ledger.get(name)
    if not isinstance(value, dict):
        _fail("catalog_invalid", f"ledger catalog {name!r} must be an object", catalog=name)
    return {str(key): _object(item, context=f"ledger catalog {name}.{key}") for key, item in value.items()}


def _derive_forcing_dependencies(records: dict[str, JsonObject], target: str) -> tuple[JsonObject, ...]:
    target_record = records.get(target)
    if target_record is None:
        _fail("target_bead_missing", f"Beads JSONL has no target bead {target}", target_bead_id=target)
    raw_dependencies = target_record.get("dependencies", [])
    if not isinstance(raw_dependencies, list):
        _fail("bead_dependencies_invalid", f"Bead {target}.dependencies must be a list", bead_id=target)
    dependencies: list[JsonObject] = []
    seen: set[str] = set()
    queued: list[tuple[str, int]] = []
    for index, raw_dependency in enumerate(raw_dependencies):
        dependency = _object(raw_dependency, context=f"Bead {target}.dependencies[{index}]")
        issue_id = _string(dependency.get("issue_id"), context=f"Bead {target}.dependencies[{index}].issue_id")
        if issue_id != target:
            _fail(
                "dependency_owner_mismatch",
                f"dependency record {index} on {target} names issue {issue_id}",
                target_bead_id=target,
                dependency_index=index,
                issue_id=issue_id,
            )
        dependency_kind = _string(dependency.get("type"), context=f"Bead {target}.dependencies[{index}].type")
        if dependency_kind == "blocks":
            queued.append(
                (
                    _string(
                        dependency.get("depends_on_id"), context=f"Bead {target}.dependencies[{index}].depends_on_id"
                    ),
                    1,
                )
            )
    while queued:
        bead_id, depth = queued.pop(0)
        if bead_id in seen:
            continue
        seen.add(bead_id)
        record = records.get(bead_id)
        if record is None:
            _fail("forcing_bead_missing", f"forcing dependency {bead_id} has no Beads record", bead_id=bead_id)
        status = _string(record.get("status"), context=f"Beads record {bead_id}.status")
        children = record.get("dependencies", [])
        if not isinstance(children, list):
            _fail("bead_dependencies_invalid", f"Bead {bead_id}.dependencies must be a list", bead_id=bead_id)
        child_ids: list[str] = []
        for index, raw_child in enumerate(children):
            child = _object(raw_child, context=f"Bead {bead_id}.dependencies[{index}]")
            if _string(child.get("issue_id"), context=f"Bead {bead_id}.dependencies[{index}].issue_id") != bead_id:
                _fail(
                    "dependency_owner_mismatch",
                    f"dependency record {index} on {bead_id} names another issue",
                    bead_id=bead_id,
                    dependency_index=index,
                )
            if _string(child.get("type"), context=f"Bead {bead_id}.dependencies[{index}].type") == "blocks":
                child_id = _string(
                    child.get("depends_on_id"), context=f"Bead {bead_id}.dependencies[{index}].depends_on_id"
                )
                child_ids.append(child_id)
                queued.append(
                    (
                        child_id,
                        depth + 1,
                    )
                )
        dependencies.append(
            {
                "bead_id": bead_id,
                "status": status,
                "dependency_kind": "blocks",
                "depth": depth,
                "child_bead_ids": child_ids,
                "priority": record.get("priority"),
                "issue_type": record.get("issue_type"),
            }
        )
    return tuple(dependencies)


def _graph_dependencies(graph: JsonObject, *, bead_records: dict[str, JsonObject]) -> tuple[JsonObject, ...]:
    target = _string(graph.get("target_bead_id"), context="campaign graph target_bead_id")
    if target != "polylogue-818fy":
        _fail("target_mismatch", f"campaign graph target {target!r} is not polylogue-818fy", target_bead_id=target)
    raw_dependencies = graph.get("forcing_dependencies")
    if not isinstance(raw_dependencies, list):
        _fail("graph_dependencies_invalid", "campaign graph forcing_dependencies must be a list")
    known_children = _strings(graph.get("known_child_bead_ids"), context="campaign graph known_child_bead_ids")
    unknown_known_children = sorted(set(known_children) - set(bead_records))
    if unknown_known_children:
        _fail(
            "unknown_successor_id",
            f"campaign graph names unknown child beads {unknown_known_children}",
            unknown_ids=unknown_known_children,
        )
    dependencies: list[JsonObject] = []
    seen: set[str] = set()
    for index, raw_dependency in enumerate(raw_dependencies):
        dependency = _object(raw_dependency, context=f"campaign graph dependency {index}")
        bead_id = _string(dependency.get("bead_id"), context=f"campaign graph dependency {index}.bead_id")
        if bead_id in seen:
            _fail("duplicate_graph_dependency", f"duplicate forcing dependency {bead_id}", duplicate_ids=[bead_id])
        seen.add(bead_id)
        status = _string(dependency.get("status"), context=f"campaign graph dependency {bead_id}.status")
        graph_kind = _string(dependency.get("kind"), context=f"campaign graph dependency {bead_id}.kind")
        if graph_kind not in GRAPH_KINDS:
            _fail(
                "unknown_graph_kind",
                f"unknown campaign graph kind {graph_kind!r} for {bead_id}",
                bead_id=bead_id,
                dependency_kind=graph_kind,
                allowed_dependency_kinds=sorted(GRAPH_KINDS),
            )
        dependency_kind = _string(
            dependency.get("dependency_kind"),
            context=f"campaign graph dependency {bead_id}.dependency_kind",
        )
        if dependency_kind not in DEPENDENCY_KINDS:
            _fail(
                "unknown_dependency_kind",
                f"unknown dependency kind {dependency_kind!r} for {bead_id}",
                bead_id=bead_id,
                dependency_kind=dependency_kind,
                allowed_dependency_kinds=sorted(DEPENDENCY_KINDS),
            )
        child_ids = _strings(
            dependency.get("child_bead_ids"),
            context=f"campaign graph dependency {bead_id}.child_bead_ids",
        )
        unknown_children = sorted(set(child_ids) - set(bead_records))
        if unknown_children:
            _fail(
                "unknown_successor_id",
                f"campaign graph dependency {bead_id} names unknown child beads {unknown_children}",
                bead_id=bead_id,
                unknown_ids=unknown_children,
            )
        dependencies.append(
            {
                **dependency,
                "status": status,
                "dependency_kind": dependency_kind,
            }
        )
    return tuple(dependencies)


def _set_diagnostic(
    *,
    expected_ids: tuple[str, ...],
    actual_ids: tuple[str, ...],
    stale_ids: list[str] | None = None,
    duplicate_ids: list[str] | None = None,
) -> JsonObject:
    expected = set(expected_ids)
    actual = set(actual_ids)
    return {
        "missing_ids": sorted(expected - actual),
        "extra_ids": sorted(actual - expected),
        "stale_ids": sorted(stale_ids or []),
        "duplicate_ids": sorted(duplicate_ids or []),
        "expected_count": len(expected_ids),
        "actual_count": len(actual_ids),
    }


def _assert_same_forcing_graph(derived: tuple[JsonObject, ...], fixture: tuple[JsonObject, ...]) -> None:
    expected_ids = tuple(_string(item.get("bead_id"), context="derived forcing dependency bead_id") for item in derived)
    fixture_ids = tuple(_string(item.get("bead_id"), context="campaign graph dependency bead_id") for item in fixture)
    duplicate_ids = sorted({bead_id for bead_id in fixture_ids if fixture_ids.count(bead_id) > 1})
    expected_by_id = {str(item["bead_id"]): item for item in derived}
    fixture_by_id = {str(item["bead_id"]): item for item in fixture}
    stale_ids = sorted(
        bead_id
        for bead_id in set(expected_by_id) & set(fixture_by_id)
        if expected_by_id[bead_id].get("status") != fixture_by_id[bead_id].get("status")
        or fixture_by_id[bead_id].get("dependency_kind", "blocks") != "blocks"
        or expected_by_id[bead_id].get("child_bead_ids", []) != fixture_by_id[bead_id].get("child_bead_ids", [])
    )
    diagnostic = _set_diagnostic(
        expected_ids=expected_ids,
        actual_ids=fixture_ids,
        stale_ids=stale_ids,
        duplicate_ids=duplicate_ids,
    )
    if any(diagnostic[key] for key in ("missing_ids", "extra_ids", "stale_ids", "duplicate_ids")):
        _fail(
            "campaign_graph_mismatch",
            "campaign graph does not match current Beads forcing dependencies",
            **diagnostic,
        )


def _committed_paths() -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail("git_files_unavailable", f"cannot inspect committed source files: {exc}")
    return {raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw}


def _validate_graph_provenance(graph: JsonObject, *, beads_path: Path) -> None:
    source_commit = _string(graph.get("source_commit"), context="campaign graph source_commit")
    source_path = _string(graph.get("source_path"), context="campaign graph source_path")
    if source_path != ".beads/issues.jsonl":
        _fail("graph_source_path_invalid", f"campaign graph source path must be .beads/issues.jsonl, got {source_path}")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=10,
        )
        source_bytes = subprocess.run(
            ["git", "show", f"{source_commit}:{source_path}"], cwd=ROOT, check=True, capture_output=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(
            "graph_source_commit_missing",
            f"campaign graph source commit cannot be read: {exc}",
            source_commit=source_commit,
        )
    actual_bytes = beads_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != hashlib.sha256(actual_bytes).hexdigest():
        _fail(
            "graph_source_snapshot_mismatch",
            "campaign graph source commit does not contain the current Beads snapshot",
            source_commit=source_commit,
            source_path=source_path,
        )


def _validate_sources(
    catalogs: dict[str, dict[str, JsonObject]],
    *,
    bead_records: dict[str, JsonObject],
) -> None:
    committed = _committed_paths()
    for catalog_name, catalog in catalogs.items():
        for item_id, entry in catalog.items():
            source = _string(entry.get("source"), context=f"ledger catalog {catalog_name}.{item_id}.source")
            if source in committed:
                continue
            if source in bead_records:
                continue
            _fail(
                "unresolved_source_reference",
                f"{catalog_name}.{item_id} source {source!r} is not a committed file or current Bead",
                catalog=catalog_name,
                item_id=item_id,
                source=source,
            )


def resolve_incident_coverage(
    ledger: JsonObject,
    graph: JsonObject,
    *,
    beads_path: Path | None = None,
) -> CoverageResolution:
    """Resolve row completeness and all structured references for one campaign graph."""

    target = _string(ledger.get("target_bead_id"), context="ledger target_bead_id")
    if target != "polylogue-818fy":
        _fail("target_mismatch", f"ledger target {target!r} is not polylogue-818fy", target_bead_id=target)
    if target != _string(graph.get("target_bead_id"), context="campaign graph target_bead_id"):
        _fail("target_mismatch", "ledger and campaign graph target beads differ")

    declared_dependency_kinds = _strings(ledger.get("dependency_kinds"), context="ledger dependency_kinds")
    if set(declared_dependency_kinds) != DEPENDENCY_KINDS:
        _fail(
            "dependency_kind_vocabulary_mismatch",
            "ledger dependency kind vocabulary must equal the closed validator vocabulary",
            declared_dependency_kinds=sorted(declared_dependency_kinds),
            allowed_dependency_kinds=sorted(DEPENDENCY_KINDS),
        )

    bead_records = load_beads_jsonl(beads_path or BEADS_PATH)
    if beads_path is None or beads_path == BEADS_PATH:
        _validate_graph_provenance(graph, beads_path=beads_path or BEADS_PATH)
    derived_dependencies = _derive_forcing_dependencies(bead_records, target)
    graph_dependencies = _graph_dependencies(graph, bead_records=bead_records)
    _assert_same_forcing_graph(derived_dependencies, graph_dependencies)

    catalogs = {
        name: _catalog(ledger, name) for name in ("fixtures", "checks", "snapshots", "receipts", "successors", "routes")
    }
    known_successors = set(_strings(graph.get("known_child_bead_ids"), context="campaign graph known_child_bead_ids"))
    missing_successors = sorted(known_successors - set(catalogs["successors"]))
    if missing_successors:
        _fail("unknown_successor", f"unknown successors {missing_successors}", unknown_ids=missing_successors)
    receipts = catalogs["receipts"]
    for receipt_id, receipt in receipts.items():
        owner = _string(receipt.get("owner_bead_id"), context=f"ledger receipt {receipt_id}.owner_bead_id")
        if owner not in bead_records:
            _fail("receipt_owner_missing", f"receipt {receipt_id} names unknown owner {owner}", receipt_id=receipt_id)
        registry_source = _string(
            receipt.get("registry_source"), context=f"ledger receipt {receipt_id}.registry_source"
        )
        registry_name = _string(receipt.get("registry"), context=f"ledger receipt {receipt_id}.registry")
        producer_registry = _object(
            _load_registered_value(registry_source, registry_name, context=f"ledger receipt {receipt_id}"),
            context=f"ledger receipt {receipt_id}.registry",
        )
        if producer_registry.get(receipt_id) != owner:
            _fail(
                "receipt_registry_mismatch",
                f"receipt {receipt_id} is not bound to owner {owner}",
                receipt_id=receipt_id,
                owner_bead_id=owner,
            )
        source = _string(receipt.get("registry_source"), context=f"ledger receipt {receipt_id}.registry_source")
        registry = _string(receipt.get("registry"), context=f"ledger receipt {receipt_id}.registry")
        registered = _object(
            _load_registered_value(source, registry, context=f"ledger receipt {receipt_id}"),
            context=f"ledger receipt {receipt_id}.registry",
        )
        if registered.get(receipt_id) != owner:
            _fail(
                "receipt_registry_mismatch",
                f"receipt {receipt_id} is not bound to owner {owner}",
                receipt_id=receipt_id,
                owner_bead_id=owner,
            )
    _validate_sources(catalogs, bead_records=bead_records)

    raw_rows = ledger.get("rows")
    if not isinstance(raw_rows, list):
        _fail("rows_invalid", "ledger rows must be a list")
    rows = tuple(_object(row, context=f"ledger row {index}") for index, row in enumerate(raw_rows))
    row_ids = tuple(_string(row.get("bead_id"), context="ledger row bead_id") for row in rows)
    duplicate_ids = sorted({bead_id for bead_id in row_ids if row_ids.count(bead_id) > 1})
    dependency_ids = tuple(
        _string(dep.get("bead_id"), context="forcing dependency bead_id") for dep in derived_dependencies
    )
    diagnostic = _set_diagnostic(
        expected_ids=dependency_ids,
        actual_ids=row_ids,
        duplicate_ids=duplicate_ids,
    )
    if duplicate_ids:
        _fail(
            "duplicate_ledger_row",
            f"ledger has duplicate rows for {duplicate_ids}",
            **diagnostic,
        )
    if any(diagnostic[key] for key in ("missing_ids", "extra_ids", "duplicate_ids")) or len(rows) != len(
        derived_dependencies
    ):
        _fail(
            "forcing_set_mismatch",
            f"ledger rows do not match forcing dependencies: missing={diagnostic['missing_ids']}, "
            f"extra={diagnostic['extra_ids']}, expected={len(derived_dependencies)}, actual={len(rows)}",
            **diagnostic,
        )

    graph_by_id = {str(dep["bead_id"]): dep for dep in graph_dependencies}
    derived_by_id = {str(dep["bead_id"]): dep for dep in derived_dependencies}
    closed_implementation_ids: list[str] = []
    successor_backed_ids: list[str] = []
    orders: list[int] = []
    for row in rows:
        bead_id = _string(row.get("bead_id"), context="ledger row bead_id")
        graph_entry = graph_by_id[bead_id]
        derived_entry = derived_by_id[bead_id]
        if row.get("bead_status") != derived_entry.get("status") or row.get("bead_status") != graph_entry.get("status"):
            _fail("stale_row", f"ledger status disagrees with current Beads for {bead_id}", stale_ids=[bead_id])
        if row.get("dependency_kind", "blocks") != derived_entry.get("dependency_kind"):
            _fail("stale_row", f"ledger dependency kind disagrees for {bead_id}", stale_ids=[bead_id])

        incident = _object(row.get("incident"), context=f"ledger row {bead_id}.incident")
        if _string(incident.get("bead_id"), context=f"ledger row {bead_id}.incident.bead_id") != bead_id:
            _fail("row_reference_mismatch", f"incident bead reference disagrees for {bead_id}")

        route = _object(row.get("route"), context=f"ledger row {bead_id}.route")
        route_kind = _string(route.get("kind"), context=f"ledger row {bead_id}.route.kind")
        if route_kind not in ROUTE_KINDS:
            _fail("unknown_route_kind", f"unknown route kind {route_kind!r} for {bead_id}", bead_id=bead_id)
        entrypoint = _string(route.get("entrypoint"), context=f"ledger row {bead_id}.route.entrypoint")
        if entrypoint not in catalogs["routes"]:
            _fail(
                "unknown_route_entrypoint",
                f"route entrypoint {entrypoint} is not registered for {bead_id}",
                bead_id=bead_id,
                entrypoint=entrypoint,
            )
        route_catalog = catalogs["routes"][entrypoint]
        route_source = _string(route_catalog.get("source"), context=f"ledger route {entrypoint}.source")
        route_registry = _string(route_catalog.get("registry"), context=f"ledger route {entrypoint}.registry")
        registered_routes = _object(
            _load_registered_value(route_source, route_registry, context=f"ledger route {entrypoint}"),
            context=f"ledger route {entrypoint}.registry",
        )
        if entrypoint not in registered_routes:
            _fail(
                "route_registry_mismatch",
                f"route {entrypoint} is absent from its executable registry",
                bead_id=bead_id,
                entrypoint=entrypoint,
            )

        schedule = _object(row.get("schedule"), context=f"ledger row {bead_id}.schedule")
        order = schedule.get("order")
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            _fail("schedule_invalid", f"schedule order is invalid for {bead_id}")
        orders.append(order)

        expected_snapshot = _object(row.get("expected_snapshot"), context=f"ledger row {bead_id}.expected_snapshot")
        snapshot_id = _string(
            expected_snapshot.get("snapshot_id"), context=f"ledger row {bead_id}.expected_snapshot.snapshot_id"
        )
        if snapshot_id not in catalogs["snapshots"]:
            _fail("unknown_snapshot", f"unknown snapshot {snapshot_id} for {bead_id}")

        red_mutation = _object(row.get("red_mutation"), context=f"ledger row {bead_id}.red_mutation")
        fixture_id = _string(red_mutation.get("fixture_id"), context=f"ledger row {bead_id}.red_mutation.fixture_id")
        if fixture_id not in catalogs["fixtures"]:
            _fail("unknown_fixture", f"unknown fixture {fixture_id} for {bead_id}")
        mutation_id = _string(red_mutation.get("mutation_id"), context=f"ledger row {bead_id}.red_mutation.mutation_id")
        mutation_ids = _strings(
            catalogs["fixtures"][fixture_id].get("mutation_ids"), context=f"ledger fixture {fixture_id}.mutation_ids"
        )
        if mutation_id not in mutation_ids:
            _fail(
                "unknown_mutation",
                f"mutation {mutation_id} is not declared by fixture {fixture_id}",
                bead_id=bead_id,
                fixture_id=fixture_id,
                mutation_id=mutation_id,
            )
        fixture_catalog = catalogs["fixtures"][fixture_id]
        mutation_source = _string(
            fixture_catalog.get("mutation_source"), context=f"ledger fixture {fixture_id}.mutation_source"
        )
        mutation_registry = _string(
            fixture_catalog.get("mutation_registry"), context=f"ledger fixture {fixture_id}.mutation_registry"
        )
        registered_mutations = _object(
            _load_registered_value(mutation_source, mutation_registry, context=f"ledger fixture {fixture_id}"),
            context=f"ledger fixture {fixture_id}.mutation_registry",
        )
        if mutation_id not in registered_mutations:
            _fail(
                "mutation_registry_mismatch",
                f"mutation {mutation_id} is absent from its fixture registry",
                bead_id=bead_id,
                fixture_id=fixture_id,
                mutation_id=mutation_id,
            )
        fixture_source = _string(fixture_catalog.get("source"), context=f"ledger fixture {fixture_id}.source")
        if fixture_source.endswith(".json"):
            fixture_payload = _load_json(ROOT / fixture_source)
            source_mutations = _strings(
                fixture_payload.get("mutation_ids"), context=f"fixture source {fixture_source}.mutation_ids"
            )
            if mutation_id not in source_mutations:
                _fail(
                    "fixture_mutation_missing",
                    f"mutation {mutation_id} is absent from fixture source {fixture_source}",
                    bead_id=bead_id,
                    fixture_id=fixture_id,
                    mutation_id=mutation_id,
                )

        check_ids = _strings(row.get("registry_checks"), context=f"ledger row {bead_id}.registry_checks")
        unknown_checks = sorted(set(check_ids) - set(catalogs["checks"]))
        if unknown_checks:
            _fail("unknown_checks", f"unknown checks {unknown_checks} for {bead_id}", unknown_ids=unknown_checks)

        receipt_ids = _strings(row.get("receipts"), context=f"ledger row {bead_id}.receipts")
        unknown_receipts = sorted(set(receipt_ids) - set(receipts))
        if unknown_receipts:
            _fail(
                "unknown_receipts", f"unknown receipts {unknown_receipts} for {bead_id}", unknown_ids=unknown_receipts
            )
        for receipt_id in receipt_ids:
            owner = _string(
                receipts[receipt_id].get("owner_bead_id"), context=f"ledger receipt {receipt_id}.owner_bead_id"
            )
            if owner != bead_id:
                _fail(
                    "receipt_owner_mismatch",
                    f"receipt {receipt_id} is owned by {owner}, not {bead_id}",
                    receipt_id=receipt_id,
                    expected_owner=bead_id,
                    actual_owner=owner,
                )

        successor = row.get("residual_successor")
        successor_id: str | None = None
        if successor is not None:
            successor_object = _object(successor, context=f"ledger row {bead_id}.residual_successor")
            successor_id = _string(
                successor_object.get("bead_id"), context=f"ledger row {bead_id}.residual_successor.bead_id"
            )
            if successor_id not in catalogs["successors"]:
                _fail("unknown_successor", f"unknown successor {successor_id} for {bead_id}")
            child_ids = _strings(
                graph_entry.get("child_bead_ids"), context=f"campaign graph dependency {bead_id}.child_bead_ids"
            )
            if successor_id not in child_ids:
                _fail("successor_parent_mismatch", f"successor {successor_id} is not a named child of {bead_id}")
            successor_backed_ids.append(bead_id)

        if graph_entry.get("status") == "closed" and graph_entry.get("kind") == "implementation":
            closed_implementation_ids.append(bead_id)
            implementation_proof = any(
                receipts[receipt_id].get("kind") in {"live-proof", "implementation-proof"} for receipt_id in receipt_ids
            )
            if not implementation_proof and successor_id is None:
                _fail(
                    "closed_implementation_unproven",
                    f"closed implementation bead {bead_id} has no live proof or named child successor",
                )

    if len(set(orders)) != len(orders) or set(orders) != set(range(1, len(rows) + 1)):
        _fail("schedule_invalid", "ledger schedule orders must be a permutation of 1..row_count")

    return CoverageResolution(
        target_bead_id=target,
        forcing_dependency_ids=dependency_ids,
        ledger_row_count=len(rows),
        closed_implementation_ids=tuple(closed_implementation_ids),
        successor_backed_ids=tuple(successor_backed_ids),
    )


def resolve_default_incident_coverage(*, beads_path: Path = BEADS_PATH) -> CoverageResolution:
    """Load and resolve the ledger against the supplied current Beads export."""

    return resolve_incident_coverage(load_ledger(), load_campaign_graph(), beads_path=beads_path)


def main(argv: list[str] | None = None) -> int:
    """Run the unconditional static verification entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beads-export", type=Path, default=BEADS_PATH)
    args = parser.parse_args(argv)

    try:
        result = resolve_default_incident_coverage(beads_path=args.beads_export)
    except IncidentCoverageLedgerError as exc:
        print(json.dumps(exc.diagnostic, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "target_bead_id": result.target_bead_id,
                "forcing_dependency_count": len(result.forcing_dependency_ids),
                "ledger_row_count": result.ledger_row_count,
            },
            sort_keys=True,
        )
    )
    return 0


ROUTE_REGISTRY: dict[str, object] = {
    "reindex-campaign": resolve_default_incident_coverage,
    "reindex-final-proof": main,
}


__all__ = [
    "BEADS_PATH",
    "CAMPAIGN_GRAPH_PATH",
    "CoverageResolution",
    "DEPENDENCY_KINDS",
    "IncidentCoverageLedgerError",
    "LEDGER_PATH",
    "ROUTE_KINDS",
    "SCHEMA_PATH",
    "load_beads_jsonl",
    "load_campaign_graph",
    "load_ledger",
    "resolve_default_incident_coverage",
    "resolve_incident_coverage",
]


if __name__ == "__main__":
    sys.exit(main())
