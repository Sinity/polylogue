from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from polylogue.core.json import JSONDecodeError
from polylogue.core.json import dumps as json_dumps
from polylogue.core.json import loads as json_loads

_ALLOWED_TYPES = {
    "implementation",
    "live_operation",
    "audit",
    "decision",
    "epic",
    "test_harness",
    "process",
    "documentation",
}
_ALLOWED_RISKS = {"ordinary", "read-only", "durable-mutation", "semantic-integrity", "resource-concurrency"}
_ALLOWED_CONFIDENCE = {"high", "medium", "planner-review"}
_ALLOWED_CLOSURE_DISPOSITIONS = {"whole-or-explicit-partial"}
_ALLOWED_ROUTE_MODES = {"named"}
_ALLOWED_ROUTE_DISPATCH = {"production", "read-only", "decision", "documentation"}
_ROUTE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._:/-][a-z0-9]+)*$")
_ROUTE_DISPATCH_BY_TYPE = {
    "implementation": frozenset({"production"}),
    "live_operation": frozenset({"production"}),
    "audit": frozenset({"read-only"}),
    "decision": frozenset({"decision"}),
    "epic": frozenset({"production"}),
    "test_harness": frozenset({"production"}),
    "process": frozenset({"production"}),
    "documentation": frozenset({"documentation"}),
}
_EVIDENCE_SPAN_FIELDS = frozenset({"snapshot_digest", "range", "text_digest"})
_EVIDENCE_RANGE_FIELDS = frozenset({"start", "end"})
_ALLOWED_VERIFICATION_MANAGERS = {"devtools"}
_ALLOWED_VERIFICATION_FOCUSED = {"devtools test"}
_ALLOWED_VERIFICATION_DEFAULT = {"devtools verify"}
_ALLOWED_RECEIPT_KINDS = {"live-operation"}
_ALLOWED_RECEIPT_REQUIREMENTS = {"required"}
_REQUIRED_RECEIPT_BINDINGS = frozenset(
    {"archive_identity", "operation", "target", "before_state", "after_state", "result_status"}
)
_PLACEHOLDER = re.compile(
    r"(?:<[^>]+>|\.{3}|\b(?:TBD|TODO|FIXME|as appropriate|figure out|choose an approach|add suitable tests)\b)",
    re.I,
)
_ROUTE_PLACEHOLDER = re.compile(r"\b(?:where applicable|as appropriate)\b", re.I)
_SOURCE_FIELDS = ("id", "title", "description", "design", "notes", "priority", "issue_type")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_ID = re.compile(r"^polylogue-[a-z0-9]+(?:\.[a-z0-9]+)*$")
_DEFAULT_MANIFEST = Path(__file__).parents[1] / "docs" / "plans" / "beads-acceptance-contracts-2026-08-07.txt"
_EXPECTED_MANIFEST_COUNT = 218
_EXPECTED_MANIFEST_DIGEST = "703df11c81dae8af6d7106bc4737502ca8baddc9013916bbb68922696d8206b5"


def _dependency_projection(issue: Mapping[str, Any]) -> list[dict[str, str | None]]:
    """Return a stable, scope-bearing projection of Bead dependencies."""
    raw_dependencies = issue.get("dependencies")
    if raw_dependencies is None:
        return []
    if not isinstance(raw_dependencies, list):
        return [{"invalid_type": type(raw_dependencies).__name__}]
    dependencies: list[dict[str, str | None]] = []
    for dependency in raw_dependencies:
        if isinstance(dependency, dict):
            dependencies.append(
                {
                    "depends_on_id": dependency.get("depends_on_id") or dependency.get("to_id") or dependency.get("id"),
                    "type": dependency.get("type") or dependency.get("dep_type"),
                }
            )
        elif isinstance(dependency, str):
            dependencies.append({"depends_on_id": dependency, "type": None})
        else:
            dependencies.append({"invalid_type": type(dependency).__name__})
    return sorted(
        dependencies,
        key=lambda dependency: (
            dependency.get("depends_on_id") or "",
            dependency.get("type") or "",
            dependency.get("invalid_type") or "",
        ),
    )


def dependency_digest(issue: Mapping[str, Any]) -> str:
    """Return the digest for the canonical dependency projection."""
    return hashlib.sha256(json_dumps(_dependency_projection(issue), sort_keys=True).encode("utf-8")).hexdigest()


def source_digest(issue: dict[str, Any]) -> str:
    """Return the digest bound to scope fields and the stable dependency projection."""
    payload = {key: issue.get(key) for key in _SOURCE_FIELDS}
    payload["dependencies"] = _dependency_projection(issue)
    return hashlib.sha256(json_dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _decode_document(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json_loads(value)
    except JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _require_string(errors: list[str], value: object, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")


def _require_string_list(errors: list[str], contract: dict[str, Any], key: str, *, optional: bool = False) -> bool:
    value = contract.get(key)
    if value == [] and optional:
        return True
    if not isinstance(value, list) or not value:
        if optional:
            errors.append(f"{key} must be a list of strings; use [] when empty")
        else:
            errors.append(f"{key} must be a non-empty list of strings")
        return False
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{key} must contain only non-empty strings")
        return False
    return True


def _require_mapping(errors: list[str], value: object, key: str) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return None
    return value


def _validate_route_spec(errors: list[str], contract: dict[str, Any]) -> bool:
    route_spec = _require_mapping(errors, contract.get("route_spec"), "route_spec")
    if route_spec is None:
        return False
    valid = True
    if route_spec.get("mode") not in _ALLOWED_ROUTE_MODES:
        errors.append("route_spec.mode must be named")
        valid = False
    identifier = route_spec.get("identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        errors.append("route_spec.identifier must be a non-empty named identifier")
        valid = False
    elif not _ROUTE_IDENTIFIER.fullmatch(identifier):
        errors.append("route_spec.identifier must be a structured named identifier")
        valid = False
    if route_spec.get("dispatch") not in _ALLOWED_ROUTE_DISPATCH:
        errors.append("route_spec.dispatch is invalid")
        valid = False
    else:
        contract_type = contract.get("contract_type")
        allowed_dispatch = _ROUTE_DISPATCH_BY_TYPE.get(
            contract_type if isinstance(contract_type, str) else "", frozenset()
        )
        if route_spec.get("dispatch") not in allowed_dispatch:
            errors.append(
                f"route_spec.dispatch {route_spec['dispatch']!r} is incompatible with "
                f"contract_type {contract.get('contract_type')!r}"
            )
            valid = False
    return valid


def _validate_verification_route(errors: list[str], contract: dict[str, Any]) -> bool:
    route = _require_mapping(errors, contract.get("verification_route"), "verification_route")
    if route is None:
        return False
    valid = True
    if route.get("manager") not in _ALLOWED_VERIFICATION_MANAGERS:
        errors.append("verification_route.manager must be devtools")
        valid = False
    if route.get("focused") not in _ALLOWED_VERIFICATION_FOCUSED:
        errors.append("verification_route.focused must be devtools test")
        valid = False
    if route.get("default") not in _ALLOWED_VERIFICATION_DEFAULT:
        errors.append("verification_route.default must be devtools verify")
        valid = False
    return valid


def _validate_receipt(errors: list[str], contract: dict[str, Any]) -> bool:
    receipt = _require_mapping(errors, contract.get("receipt"), "receipt")
    if receipt is None:
        return False
    valid = True
    if receipt.get("kind") not in _ALLOWED_RECEIPT_KINDS:
        errors.append("receipt.kind must be live-operation")
        valid = False
    if receipt.get("requirement") not in _ALLOWED_RECEIPT_REQUIREMENTS:
        errors.append("receipt.requirement must be required")
        valid = False
    bindings = receipt.get("bindings")
    if not isinstance(bindings, list) or any(not isinstance(item, str) for item in bindings):
        errors.append("receipt.bindings must be a list of strings")
        return False
    if set(bindings) != _REQUIRED_RECEIPT_BINDINGS or len(bindings) != len(_REQUIRED_RECEIPT_BINDINGS):
        errors.append("receipt.bindings must include each required live-operation dimension exactly once")
        valid = False
    return valid


def _validate_evidence_spans(errors: list[str], contract: dict[str, Any]) -> None:
    """Validate evidence as snapshot-bound typed ranges, never prose heuristics."""
    evidence = contract.get("evidence")
    spans = contract.get("evidence_spans")
    if not isinstance(evidence, list) or not isinstance(spans, list):
        errors.append("evidence_spans must provide one typed span for every evidence item")
        return
    if len(spans) != len(evidence):
        errors.append("evidence_spans must contain exactly one span for every evidence item")
        return
    for index, (value, span) in enumerate(zip(evidence, spans, strict=True)):
        if not isinstance(span, Mapping):
            errors.append(f"evidence_spans[{index}] must be an object")
            continue
        if set(span) != _EVIDENCE_SPAN_FIELDS:
            errors.append(f"evidence_spans[{index}] fields must be exactly snapshot_digest, range, and text_digest")
        snapshot_digest = span.get("snapshot_digest")
        if not isinstance(snapshot_digest, str) or not _SHA256.fullmatch(snapshot_digest):
            errors.append(f"evidence_spans[{index}].snapshot_digest must be a lowercase SHA-256 digest")
        evidence_range = span.get("range")
        if not isinstance(evidence_range, Mapping) or set(evidence_range) != _EVIDENCE_RANGE_FIELDS:
            errors.append(f"evidence_spans[{index}].range must contain exactly start and end")
        else:
            start = evidence_range.get("start")
            end = evidence_range.get("end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
            ):
                errors.append(f"evidence_spans[{index}].range must be a non-empty half-open integer range")
        text_digest = span.get("text_digest")
        if not isinstance(text_digest, str) or not _SHA256.fullmatch(text_digest):
            errors.append(f"evidence_spans[{index}].text_digest must be a lowercase SHA-256 digest")
        elif isinstance(value, str):
            expected_text_digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if text_digest != expected_text_digest:
                errors.append(f"evidence_spans[{index}].text_digest does not match the evidence item")


def load_manifest(path: Path) -> tuple[str, ...]:
    """Load and ratchet the committed required-contract inventory."""
    if not path.is_file():
        raise SystemExit(f"{path}: acceptance-contract manifest is missing")
    required_values = path.read_text(encoding="utf-8").split()
    if (
        len(required_values) != _EXPECTED_MANIFEST_COUNT
        or len(set(required_values)) != _EXPECTED_MANIFEST_COUNT
        or any(not _MANIFEST_ID.fullmatch(value) for value in required_values)
    ):
        raise SystemExit(f"{path}: acceptance-contract manifest inventory is invalid")
    manifest_digest = hashlib.sha256(("\n".join(sorted(required_values)) + "\n").encode("utf-8")).hexdigest()
    if manifest_digest != _EXPECTED_MANIFEST_DIGEST:
        raise SystemExit(f"{path}: acceptance-contract manifest inventory changed")
    return tuple(sorted(required_values))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def render(contract: dict[str, Any]) -> str:
    rows: list[str] = []
    n = 1

    def add(label: str, value: str) -> None:
        nonlocal n
        rows.append(f"{n}. {label}: {value.strip()}")
        n += 1

    add("Outcome", contract["outcome"])
    if contract.get("confidence") == "planner-review":
        add("Dispatch gate", "Planner review is required before implementation dispatch.")
    route_spec = contract["route_spec"]
    add(
        "Route authority",
        f"{route_spec['mode']} {route_spec['identifier']} {route_spec['dispatch']} route coverage is required.",
    )
    for value in contract.get("retained_scope", []):
        add("Existing scope retained", value)
    for value in contract.get("routes", []):
        add("Production route", value)
    for value in contract.get("evidence", []):
        add("Evidence", value)
    for value in contract.get("verification", []):
        add("Verification", value)
    for value in contract.get("anti_vacuity", []):
        add("Anti-vacuity", value)
    for value in contract.get("safety", []):
        add("Safety", value)
    if contract.get("contract_type") in {"implementation", "test_harness"}:
        verification_route = contract["verification_route"]
        add(
            "Managed verification route",
            f"focused={verification_route['focused']}; default={verification_route['default']}",
        )
    if contract.get("contract_type") == "live_operation":
        receipt = contract["receipt"]
        add(
            "Receipt requirement",
            f"{receipt['kind']} result={receipt['requirement']} bindings={','.join(receipt['bindings'])}",
        )
    add("Closure disposition", contract["closure"]["disposition"])
    add(
        "Partial closure successor",
        "required when the closure disposition is whole-or-explicit-partial.",
    )
    add("Closure", contract["closure"]["rule"])
    return "\n".join(rows)


def validate(issue: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    metadata = _decode_document(issue.get("metadata"))
    if metadata is None:
        return ["metadata is not a JSON object"]
    contract = metadata.get("acceptance_contract_v1")
    if not isinstance(contract, dict):
        return ["missing metadata.acceptance_contract_v1"]
    if contract.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if contract.get("bead_id") != issue.get("id"):
        errors.append("bead_id does not match issue id")
    if contract.get("contract_type") not in _ALLOWED_TYPES:
        errors.append("invalid contract_type")
    if contract.get("risk") not in _ALLOWED_RISKS:
        errors.append("invalid risk")
    if contract.get("confidence") not in _ALLOWED_CONFIDENCE:
        errors.append("confidence must be high, medium, or planner-review")
    _require_string(errors, contract.get("outcome"), "outcome")
    for key in ("routes", "evidence", "verification", "anti_vacuity"):
        _require_string_list(errors, contract, key)
    _require_string_list(errors, contract, "retained_scope", optional=True)
    _require_string_list(errors, contract, "safety", optional=True)
    _validate_route_spec(errors, contract)
    if contract.get("contract_type") in {"implementation", "test_harness"}:
        _validate_verification_route(errors, contract)
    closure = contract.get("closure")
    if not isinstance(closure, Mapping):
        errors.append("closure must be an object")
    else:
        _require_string(errors, closure.get("rule"), "closure.rule")
        if closure.get("disposition") not in _ALLOWED_CLOSURE_DISPOSITIONS:
            errors.append("closure.disposition must be whole-or-explicit-partial")
        if not isinstance(closure.get("successor_required_for_partial"), bool):
            errors.append("closure.successor_required_for_partial must be boolean")
        elif closure.get("disposition") == "whole-or-explicit-partial" and not closure.get(
            "successor_required_for_partial"
        ):
            errors.append("whole-or-explicit-partial requires successor_required_for_partial=true")
    digest = contract.get("source_digest")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        errors.append("source_digest must be a lowercase SHA-256 digest")
    elif digest != source_digest(issue):
        errors.append("source_digest does not match the Bead source snapshot")
    dependency_digest_value = contract.get("dependency_digest")
    if not isinstance(dependency_digest_value, str) or not _SHA256.fullmatch(dependency_digest_value):
        errors.append("dependency_digest must be a lowercase SHA-256 digest")
    elif dependency_digest_value != dependency_digest(issue):
        errors.append("dependency_digest does not match the Bead dependency projection")
    if issue.get("dependencies") is not None and not isinstance(issue.get("dependencies"), list):
        errors.append("dependencies must be a list")
    if contract.get("contract_type") == "live_operation" and not contract.get("safety"):
        errors.append("live_operation requires safety clauses")
    if contract.get("contract_type") == "live_operation":
        _validate_receipt(errors, contract)
    if contract.get("risk") == "durable-mutation" and not contract.get("safety"):
        errors.append("durable-mutation requires safety clauses")
    for value in contract.get("routes", []) if isinstance(contract.get("routes"), list) else []:
        if isinstance(value, str) and _ROUTE_PLACEHOLDER.search(value):
            errors.append("routes contains a generic placeholder; use named route fields")
    for value in _strings(contract):
        if _PLACEHOLDER.search(value):
            errors.append(f"placeholder in contract: {value[:80]}")
    _validate_evidence_spans(errors, contract)
    if not errors:
        expected = render(contract)
        if issue.get("acceptance_criteria") != expected:
            errors.append("acceptance_criteria drifted from structured contract")
    return sorted(set(errors))


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json_loads(line)
        except JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{n}: expected a JSON object, got {type(row).__name__}")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate structured Beads acceptance contracts without guessing from prose."
    )
    parser.add_argument("issues", type=Path, nargs="?", default=Path(".beads/issues.jsonl"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="newline-separated Bead ids that must carry a valid contract",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    required_values = load_manifest(args.manifest)
    required = set(required_values)
    failures: dict[str, list[str]] = {}
    seen = set()
    for issue in load(args.issues):
        bid = issue.get("id")
        if not isinstance(bid, str) or bid not in required:
            continue
        seen.add(bid)
        errors = validate(issue)
        if errors:
            failures[bid] = errors
    for missing in sorted(required - seen):
        failures[missing] = ["manifest id missing from issues or contract"]
    regeneration_required = [{"id": bead_id, "reasons": failures[bead_id]} for bead_id in sorted(failures)]
    report = {
        "ok": not failures,
        "dispatch_blocked": bool(failures),
        "manifest": {
            "expected_count": _EXPECTED_MANIFEST_COUNT,
            "digest": _EXPECTED_MANIFEST_DIGEST,
        },
        "validated": len(seen),
        "failures": failures,
        "regeneration_required": regeneration_required,
    }
    if args.json:
        print(json_dumps(report, indent=2, sort_keys=True))
    else:
        for bid, errors in sorted(failures.items()):
            for error in errors:
                print(f"{bid}: {error}")
        print(f"validated={len(seen)} failures={len(failures)} regeneration_required={len(regeneration_required)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
