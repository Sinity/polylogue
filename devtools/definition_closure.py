"""Typed definition-to-production closure checks.

This module is deliberately a small verification kernel.  A policy names an
existing inventory and supplies evidence for that inventory; it does not own
or duplicate any product registry.  Domain checks can therefore adopt the
kernel without creating a universal declaration catalogue.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

MAX_DEFINITIONS = 256
MAX_EDGES_PER_DEFINITION = 32
MAX_DIAGNOSTICS = 128
MAX_GRAPH_ROWS = 1024


class EdgeKind(StrEnum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    LIFECYCLE = "lifecycle"
    ADAPTER = "adapter"
    CONTRACT = "contract"
    DISCOVERY = "discovery"
    REAL_ROUTE = "real-route"


class ClosureStatus(StrEnum):
    SATISFIED = "satisfied"
    INTENTIONAL_ABSENCE = "intentional-absence"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"
    TESTS_ONLY = "tests-only"
    BYPASS = "bypass"
    DIVERGENT_TWIN = "divergent-twin"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A bounded, stable pointer to evidence (never the evidence itself)."""

    ref: str
    source: str = "production"
    route: str | None = None
    twin: str | None = None

    def to_dict(self) -> dict[str, str]:
        result = {"ref": self.ref, "source": self.source}
        if self.route is not None:
            result["route"] = self.route
        if self.twin is not None:
            result["twin"] = self.twin
        return result


@dataclass(frozen=True, slots=True)
class Definition:
    """One item from an authoritative inventory."""

    ref: str
    required_edges: tuple[EdgeKind, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"ref": self.ref, "required_edges": [edge.value for edge in self.required_edges]}


@dataclass(frozen=True, slots=True)
class ClosurePolicy:
    """An adapter from an authoritative inventory to closure requirements."""

    family: str
    authoritative_inventory_ref: str
    required_edge_kinds: tuple[EdgeKind, ...]
    exception_authority: str | None = None
    definitions: tuple[Definition, ...] = ()
    # Explicit exceptions are keyed by stable definition ref.  Their value is
    # the authority ref, so an empty/missing value can never silently waive a row.
    intentional_absences: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family or not self.authoritative_inventory_ref:
            raise ValueError("closure policy requires family and authoritative inventory ref")
        required = set(self.required_edge_kinds)
        if not required:
            raise ValueError(f"{self.family}: at least one required edge kind is required")
        if len(required) != len(self.required_edge_kinds):
            raise ValueError(f"{self.family}: duplicate required edge kind")
        if len(self.definitions) > MAX_DEFINITIONS:
            raise ValueError(f"{self.family}: inventory exceeds bounded definition limit")
        definition_refs = [definition.ref for definition in self.definitions]
        if len(set(definition_refs)) != len(definition_refs):
            raise ValueError(f"{self.family}: duplicate definition ref")
        for definition in self.definitions:
            if not definition.ref:
                raise ValueError(f"{self.family}: definition ref cannot be empty")
            unknown = set(definition.required_edges) - required
            if unknown:
                raise ValueError(f"{definition.ref}: edge kinds not declared by policy: {sorted(unknown)}")
            if len(set(definition.required_edges)) != len(definition.required_edges):
                raise ValueError(f"{definition.ref}: duplicate required edge kind")
        unknown_absences = set(self.intentional_absences) - set(definition_refs)
        if unknown_absences:
            raise ValueError(f"{self.family}: intentional absence has unknown definition: {sorted(unknown_absences)}")
        if self.intentional_absences and not self.exception_authority:
            raise ValueError(f"{self.family}: intentional absences require exception authority")
        if any(not authority.strip() for authority in self.intentional_absences.values()):
            raise ValueError(f"{self.family}: intentional absence authority cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "authoritative_inventory_ref": self.authoritative_inventory_ref,
            "required_edge_kinds": [edge.value for edge in self.required_edge_kinds],
            "exception_authority": self.exception_authority,
            "definitions": [definition.to_dict() for definition in self.definitions],
            "intentional_absences": dict(sorted(self.intentional_absences.items())),
        }


@dataclass(frozen=True, slots=True)
class ClosureRow:
    family: str
    definition_ref: str
    status: ClosureStatus
    required_edges: tuple[EdgeKind, ...]
    actual_edges: Mapping[EdgeKind, tuple[EvidenceRef, ...]]
    missing_edges: tuple[EdgeKind, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    exception_authority: str | None = None
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "definition_ref": self.definition_ref,
            "status": self.status.value,
            "required_edges": [edge.value for edge in self.required_edges],
            "actual_edges": {
                edge.value: [item.to_dict() for item in refs]
                for edge, refs in sorted(self.actual_edges.items(), key=lambda x: x[0].value)
            },
            "missing_edges": [edge.value for edge in self.missing_edges],
            "evidence_refs": list(self.evidence_refs),
            "exception_authority": self.exception_authority,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True, slots=True)
class DefinitionClosureGraph:
    policies: tuple[ClosurePolicy, ...]
    rows: tuple[ClosureRow, ...]
    evidence_available: bool = True
    coverage_limits: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(row.status in {ClosureStatus.SATISFIED, ClosureStatus.INTENTIONAL_ABSENCE} for row in self.rows)

    @property
    def exceptions(self) -> tuple[dict[str, str], ...]:
        """Return every explicit exception, including exceptions with no row.

        Keeping this on the rendered graph makes an exception auditable without
        requiring consumers to correlate policy and row payloads themselves.
        """
        return tuple(
            {
                "family": policy.family,
                "definition_ref": definition_ref,
                "authority": authority,
            }
            for policy in self.policies
            for definition_ref, authority in sorted(policy.intentional_absences.items())
        )

    def to_dict(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.status.value] = counts.get(row.status.value, 0) + 1
        return {
            "schema_version": 1,
            "ok": self.ok,
            "evidence_available": self.evidence_available,
            "families": [policy.to_dict() for policy in self.policies],
            "inventory_counts": {policy.family: len(policy.definitions) for policy in self.policies},
            "status_counts": dict(sorted(counts.items())),
            "required_edge_count": sum(len(row.required_edges) for row in self.rows),
            "actual_edge_count": sum(sum(len(refs) for refs in row.actual_edges.values()) for row in self.rows),
            "exceptions": list(self.exceptions),
            "unresolved_rows": [
                row.definition_ref
                for row in self.rows
                if row.status not in {ClosureStatus.SATISFIED, ClosureStatus.INTENTIONAL_ABSENCE}
            ],
            "rows": [row.to_dict() for row in self.rows[:MAX_GRAPH_ROWS]],
            "coverage_limits": list(self.coverage_limits),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def evaluate(
    policies: Iterable[ClosurePolicy],
    evidence: Mapping[str, Mapping[EdgeKind, Iterable[EvidenceRef]]] | None = None,
    *,
    evidence_available: bool = True,
    coverage_limits: Iterable[str] = (),
) -> DefinitionClosureGraph:
    """Evaluate policies against evidence keyed by ``definition_ref``.

    Evidence is intentionally explicit.  Tests-only evidence never satisfies a
    production edge; a route marked ``bypass`` or a second twin identity is
    reported precisely rather than being reduced to a generic missing row.
    """
    evidence = evidence or {}
    rows: list[ClosureRow] = []
    policy_tuple = tuple(policies)
    total_definitions = sum(len(policy.definitions) for policy in policy_tuple)
    if total_definitions > MAX_GRAPH_ROWS:
        raise ValueError(f"closure graph exceeds bounded row limit ({MAX_GRAPH_ROWS})")
    for policy in policy_tuple:
        for definition in policy.definitions[:MAX_DEFINITIONS]:
            required = definition.required_edges or policy.required_edge_kinds
            supplied = evidence.get(definition.ref, {})
            actual: dict[EdgeKind, tuple[EvidenceRef, ...]] = {}
            for raw_edge, refs in supplied.items():
                try:
                    edge = EdgeKind(raw_edge)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{definition.ref}: unknown evidence edge {raw_edge!r}") from exc
                refs_tuple = tuple(refs)
                if len(refs_tuple) > MAX_EDGES_PER_DEFINITION:
                    raise ValueError(
                        f"{definition.ref}: evidence exceeds bounded edge limit ({MAX_EDGES_PER_DEFINITION})"
                    )
                actual[edge] = refs_tuple
            missing = tuple(
                edge for edge in required if not any(ref.source == "production" for ref in actual.get(edge, ()))
            )
            all_refs = tuple(item.ref for refs in actual.values() for item in refs)
            status = ClosureStatus.SATISFIED
            diagnostic: str | None = None
            exception = policy.intentional_absences.get(definition.ref)
            if not evidence_available:
                status, diagnostic = ClosureStatus.UNAVAILABLE, f"{definition.ref}: evidence unavailable"
            elif any(ref.route == "bypass" for refs in actual.values() for ref in refs):
                status, diagnostic = ClosureStatus.BYPASS, f"{definition.ref}: bypasses shared substrate"
            elif any(ref.twin for refs in actual.values() for ref in refs):
                status, diagnostic = ClosureStatus.DIVERGENT_TWIN, f"{definition.ref}: divergent twin evidence"
            elif missing:
                if exception:
                    status, diagnostic = (
                        ClosureStatus.INTENTIONAL_ABSENCE,
                        f"{definition.ref}: intentional absence authorized by {exception}",
                    )
                else:
                    supplied_refs = tuple(ref for refs in actual.values() for ref in refs)
                    tests_only = bool(supplied_refs) and all(ref.source == "tests" for ref in supplied_refs)
                    status = ClosureStatus.TESTS_ONLY if tests_only else ClosureStatus.MISSING
                    missing_edge = missing[0].value
                    diagnostic = f"{definition.ref}: missing edge {missing_edge}"
            rows.append(
                ClosureRow(
                    policy.family,
                    definition.ref,
                    status,
                    tuple(required),
                    actual,
                    missing,
                    all_refs,
                    exception,
                    diagnostic,
                )
            )
    return DefinitionClosureGraph(policy_tuple, tuple(rows), evidence_available, tuple(coverage_limits))


def representative_policies() -> tuple[ClosurePolicy, ...]:
    """Small representative contracts; inventories remain policy-owned inputs."""
    specs = (
        ("storage-lifecycle", "storage.ddl", (EdgeKind.PRODUCER, EdgeKind.CONSUMER, EdgeKind.LIFECYCLE)),
        ("event-effects", "events.registry", (EdgeKind.PRODUCER, EdgeKind.CONSUMER, EdgeKind.LIFECYCLE)),
        (
            "registry-declaration",
            "declarations.registry",
            (EdgeKind.PRODUCER, EdgeKind.CONSUMER, EdgeKind.CONTRACT, EdgeKind.DISCOVERY),
        ),
        ("query-pipeline", "query.pipeline", (EdgeKind.PRODUCER, EdgeKind.CONSUMER, EdgeKind.REAL_ROUTE)),
        (
            "semantic-operation",
            "operations.catalog",
            (EdgeKind.PRODUCER, EdgeKind.ADAPTER, EdgeKind.CONTRACT, EdgeKind.DISCOVERY, EdgeKind.REAL_ROUTE),
        ),
    )
    return tuple(
        ClosurePolicy(
            family,
            inventory,
            required,
            exception_authority=f"authority:{family}",
            definitions=(Definition(f"{family}:representative", required),),
        )
        for family, inventory, required in specs
    )


def representative_evidence(policies: Iterable[ClosurePolicy]) -> dict[str, dict[EdgeKind, tuple[EvidenceRef, ...]]]:
    """Return static source references for the command's smoke matrix.

    These are source/inventory anchors, not a second inventory.  Larger domain
    adopters pass their live registry evidence directly to :func:`evaluate`.
    """
    result: dict[str, dict[EdgeKind, tuple[EvidenceRef, ...]]] = {}
    for policy in policies:
        for definition in policy.definitions:
            result[definition.ref] = {
                edge: (EvidenceRef(f"{policy.authoritative_inventory_ref}:{edge.value}", source="production"),)
                for edge in definition.required_edges
            }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the durable JSON matrix to this path.")
    parser.add_argument(
        "--empty", action="store_true", help="Evaluate empty inventories and report unavailable evidence explicitly."
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    policies = representative_policies()
    graph = evaluate(
        policies,
        representative_evidence(policies),
        evidence_available=not args.empty,
        coverage_limits=("default matrix uses bounded source anchors; live archive evidence is additive",),
    )
    payload = graph.to_json()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(f"definition closure: {'ok' if graph.ok else 'failed'} ({len(graph.rows)} definitions)")
        for row in graph.rows[:MAX_DIAGNOSTICS]:
            if row.diagnostic:
                print(f"- {row.diagnostic}")
    return 0 if graph.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
