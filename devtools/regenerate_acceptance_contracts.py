"""Regenerate the exact typed acceptance-contract snapshot without invoking bd."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from devtools import beads_acceptance_contracts as contracts
from devtools.acceptance_route_registry import registry_digest
from polylogue.core.json import dumps as json_dumps

_ROUTE_CLASS_BY_TYPE = {
    "implementation": "ImplementationRoute",
    "live_operation": "LiveOperationRoute",
    "audit": "AuditRoute",
    "decision": "DecisionRoute",
    "epic": "EpicRoute",
    "test_harness": "TestHarnessRoute",
    "process": "ProcessRoute",
    "documentation": "DocumentationRoute",
}


def _span(evidence: str) -> dict[str, Any]:
    encoded = evidence.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "snapshot": evidence,
        "snapshot_digest": digest,
        "range": {"start": 0, "end": len(encoded)},
        "text_digest": digest,
    }


def _route_entry(issue: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    contract_type = contract["contract_type"]
    dispatch = next(iter(contracts._ROUTE_DISPATCH_BY_TYPE[contract_type]))
    identifier = f"acceptance/{issue['id']}"
    return {
        "identifier": identifier,
        "bead_id": issue["id"],
        "contract_type": contract_type,
        "class": _ROUTE_CLASS_BY_TYPE[contract_type],
        "dispatch": dispatch,
        "targets": list(contract["routes"]),
    }


def regenerate(
    rows: list[dict[str, Any]], required_ids: Iterable[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = set(required_ids)
    routes: list[dict[str, Any]] = []
    output: list[dict[str, Any]] = []
    for issue in rows:
        if issue.get("id") not in required:
            output.append(issue)
            continue
        metadata = issue.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("acceptance_contract_v1"), dict):
            raise ValueError(f"{issue.get('id')}: missing acceptance_contract_v1")
        contract = metadata["acceptance_contract_v1"]
        route = _route_entry(issue, contract)
        contract["route_spec"] = {
            "mode": "named",
            "identifier": route["identifier"],
            "class": route["class"],
            "dispatch": route["dispatch"],
        }
        if contract.get("contract_type") in {"implementation", "test_harness"}:
            contract["verification_route"] = {
                "manager": "devtools",
                "focused": "devtools test",
                "default": "devtools verify",
            }
        contract["evidence_spans"] = [_span(value) for value in contract.get("evidence", [])]
        if contract.get("contract_type") == "live_operation":
            contract["receipt"] = {
                "kind": "live-operation",
                "requirement": "required",
                "bindings": sorted(contracts._REQUIRED_RECEIPT_BINDINGS),
            }
        contract["source_digest"] = contracts.source_digest(issue)
        contract["dependency_digest"] = contracts.dependency_digest(issue)
        issue["acceptance_criteria"] = contracts.render(contract)
        routes.append(route)
        output.append(issue)
    if {issue.get("id") for issue in output if issue.get("id") in required} != required:
        raise ValueError("canonical snapshot does not contain every manifest id")
    return output, routes


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json_dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path, default=Path(".beads/issues.jsonl"))
    parser.add_argument("--manifest", type=Path, default=contracts._DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=Path("docs/plans/beads-acceptance-route-registry.json"))
    args = parser.parse_args(argv)
    rows = contracts.load(args.issues)
    required = contracts.load_manifest(args.manifest)
    regenerated, routes = regenerate(rows, required)
    registry_document = {
        "schema_version": 1,
        "manifest_count": len(required),
        "manifest_digest": contracts._EXPECTED_MANIFEST_DIGEST,
        "routes": sorted(routes, key=lambda route: route["identifier"]),
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json_dumps(registry_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(args.issues, regenerated)
    print(
        json_dumps(
            {
                "ok": True,
                "records": len(required),
                "registry_routes": len(routes),
                "registry_digest": registry_digest({route["identifier"]: route for route in routes}),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
