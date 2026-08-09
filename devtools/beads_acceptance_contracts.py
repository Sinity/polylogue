from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from devtools.acceptance_route_registry import (
    AcceptanceRouteRegistryError,
    load_registry,
    registry_digest,
    resolve_route,
)
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
_EVIDENCE_SPAN_FIELDS = frozenset({"snapshot", "snapshot_digest", "range", "text_digest"})
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


class DependencyProjectionError(ValueError):
    """Raised when a structured dependency projection contains an invalid scalar."""


def _dependency_projection(issue: Mapping[str, Any]) -> list[dict[str, str | None]]:
    """Return a stable, scope-bearing projection of Bead dependencies."""
    raw_dependencies = issue.get("dependencies")
    if raw_dependencies is None:
        return []
    if not isinstance(raw_dependencies, list):
        return [{"invalid_type": type(raw_dependencies).__name__}]
    dependencies: list[dict[str, str | None]] = []
    for index, dependency in enumerate(raw_dependencies):
        if isinstance(dependency, dict):
            depends_on_id = _dependency_scalar(dependency, index, ("depends_on_id", "to_id", "id"), "depends_on_id")
            dependency_type = _dependency_scalar(dependency, index, ("type", "dep_type"), "type")
            dependencies.append(
                {
                    "depends_on_id": depends_on_id,
                    "type": dependency_type,
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


def _dependency_scalar(dependency: Mapping[str, Any], index: int, keys: tuple[str, ...], label: str) -> str | None:
    for key in keys:
        value = dependency.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise DependencyProjectionError(
                f"dependencies[{index}].{label} must be a string or null (got {type(value).__name__})"
            )
        return value
    return None


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
    if set(route_spec) != {"mode", "identifier", "class", "dispatch"}:
        errors.append("route_spec fields must be exactly mode, identifier, class, and dispatch")
        valid = False
    mode = route_spec.get("mode")
    if not isinstance(mode, str) or mode not in _ALLOWED_ROUTE_MODES:
        errors.append("route_spec.mode must be named")
        valid = False
    identifier = route_spec.get("identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        errors.append("route_spec.identifier must be a non-empty named identifier")
        valid = False
    elif not _ROUTE_IDENTIFIER.fullmatch(identifier):
        errors.append("route_spec.identifier must be a structured named identifier")
        valid = False
    dispatch = route_spec.get("dispatch")
    if not isinstance(dispatch, str) or dispatch not in _ALLOWED_ROUTE_DISPATCH:
        errors.append("route_spec.dispatch is invalid")
        valid = False
    else:
        contract_type = contract.get("contract_type")
        allowed_dispatch = _ROUTE_DISPATCH_BY_TYPE.get(
            contract_type if isinstance(contract_type, str) else "", frozenset()
        )
        if dispatch not in allowed_dispatch:
            errors.append(
                f"route_spec.dispatch {dispatch!r} is incompatible with contract_type {contract.get('contract_type')!r}"
            )
            valid = False
    identifier = route_spec.get("identifier")
    try:
        registered = resolve_route(identifier)
    except AcceptanceRouteRegistryError as exc:
        errors.append(str(exc))
        return False
    if registered is None:
        errors.append(f"route_spec.identifier {identifier!r} is not registered")
        return False
    registered_bead_id = registered.get("bead_id")
    if not isinstance(registered_bead_id, str) or not registered_bead_id:
        errors.append("registered route authority must bind a non-empty Bead id")
        valid = False
    if not isinstance(contract.get("bead_id"), str) or registered_bead_id != contract.get("bead_id"):
        errors.append("route_spec.identifier is registered for a different Bead")
        valid = False
    registered_contract_type = registered.get("contract_type")
    if not isinstance(registered_contract_type, str) or registered_contract_type != contract.get("contract_type"):
        errors.append("route_spec.identifier is registered for a different contract_type")
        valid = False
    registered_dispatch = registered.get("dispatch")
    if not isinstance(registered_dispatch, str) or registered_dispatch != dispatch:
        errors.append("route_spec.dispatch does not match the registered route class")
        valid = False
    contract_type = contract.get("contract_type")
    expected_class = _ROUTE_CLASS_BY_TYPE.get(contract_type if isinstance(contract_type, str) else "")
    route_class = route_spec.get("class")
    if not isinstance(route_class, str) or route_class != expected_class:
        errors.append("route_spec.class does not match the contract_type")
        valid = False
    registered_class = registered.get("class")
    if not isinstance(registered_class, str) or registered_class != route_class:
        errors.append("route_spec.class does not match the registered route class")
        valid = False
    targets = registered.get("targets")
    if not isinstance(targets, list) or any(not isinstance(target, str) for target in targets):
        errors.append("registered route authority has no target list")
        valid = False
    elif targets != contract.get("routes"):
        errors.append("route_spec targets do not match the registered route authority")
        valid = False
    return valid


def _validate_verification_route(errors: list[str], contract: dict[str, Any]) -> bool:
    route = _require_mapping(errors, contract.get("verification_route"), "verification_route")
    if route is None:
        return False
    valid = True
    manager = route.get("manager")
    if not isinstance(manager, str) or manager not in _ALLOWED_VERIFICATION_MANAGERS:
        errors.append("verification_route.manager must be devtools")
        valid = False
    focused = route.get("focused")
    if not isinstance(focused, str) or focused not in _ALLOWED_VERIFICATION_FOCUSED:
        errors.append("verification_route.focused must be devtools test")
        valid = False
    default = route.get("default")
    if not isinstance(default, str) or default not in _ALLOWED_VERIFICATION_DEFAULT:
        errors.append("verification_route.default must be devtools verify")
        valid = False
    return valid


def _validate_receipt(errors: list[str], contract: dict[str, Any]) -> bool:
    receipt = _require_mapping(errors, contract.get("receipt"), "receipt")
    if receipt is None:
        return False
    valid = True
    kind = receipt.get("kind")
    if not isinstance(kind, str) or kind not in _ALLOWED_RECEIPT_KINDS:
        errors.append("receipt.kind must be live-operation")
        valid = False
    requirement = receipt.get("requirement")
    if not isinstance(requirement, str) or requirement not in _ALLOWED_RECEIPT_REQUIREMENTS:
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
    """Validate evidence as byte ranges over digest-bound UTF-8 snapshots."""
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
            errors.append(
                f"evidence_spans[{index}] fields must be exactly snapshot, snapshot_digest, range, and text_digest"
            )
        snapshot = span.get("snapshot")
        snapshot_bytes: bytes | None = None
        if not isinstance(snapshot, str):
            errors.append(f"evidence_spans[{index}].snapshot must be a UTF-8 snapshot string")
        else:
            snapshot_bytes = snapshot.encode("utf-8")
        snapshot_digest = span.get("snapshot_digest")
        if not isinstance(snapshot_digest, str) or not _SHA256.fullmatch(snapshot_digest):
            errors.append(f"evidence_spans[{index}].snapshot_digest must be a lowercase SHA-256 digest")
        elif snapshot_bytes is not None and snapshot_digest != hashlib.sha256(snapshot_bytes).hexdigest():
            errors.append(f"evidence_spans[{index}].snapshot_digest does not match the snapshot")
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
            elif snapshot_bytes is not None and end > len(snapshot_bytes):
                errors.append(f"evidence_spans[{index}].range exceeds the snapshot byte length")
        text_digest = span.get("text_digest")
        if not isinstance(text_digest, str) or not _SHA256.fullmatch(text_digest):
            errors.append(f"evidence_spans[{index}].text_digest must be a lowercase SHA-256 digest")
        elif snapshot_bytes is not None and isinstance(evidence_range, Mapping):
            start = evidence_range.get("start")
            end = evidence_range.get("end")
            if (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end <= len(snapshot_bytes)
            ):
                span_bytes = snapshot_bytes[start:end]
                try:
                    span_text = span_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    errors.append(f"evidence_spans[{index}].range must align to UTF-8 boundaries")
                else:
                    expected_text_digest = hashlib.sha256(span_bytes).hexdigest()
                    if text_digest != expected_text_digest:
                        errors.append(f"evidence_spans[{index}].text_digest does not match the snapshot range")
                    if value != span_text:
                        errors.append(f"evidence_spans[{index}].range text does not match the evidence item")


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
    contract_type = contract.get("contract_type")
    if isinstance(contract_type, str) and contract_type in {"implementation", "test_harness"}:
        verification_route = contract["verification_route"]
        add(
            "Managed verification route",
            f"focused={verification_route['focused']}; default={verification_route['default']}",
        )
    if contract_type == "live_operation":
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
    contract_type = contract.get("contract_type")
    if not isinstance(contract_type, str) or contract_type not in _ALLOWED_TYPES:
        errors.append("invalid contract_type")
    risk = contract.get("risk")
    if not isinstance(risk, str) or risk not in _ALLOWED_RISKS:
        errors.append("invalid risk")
    confidence = contract.get("confidence")
    if not isinstance(confidence, str) or confidence not in _ALLOWED_CONFIDENCE:
        errors.append("confidence must be high, medium, or planner-review")
    _require_string(errors, contract.get("bead_id"), "bead_id")
    _require_string(errors, contract.get("outcome"), "outcome")
    for key in ("routes", "evidence", "verification", "anti_vacuity"):
        _require_string_list(errors, contract, key)
    _require_string_list(errors, contract, "retained_scope", optional=True)
    _require_string_list(errors, contract, "safety", optional=True)
    _validate_route_spec(errors, contract)
    if isinstance(contract_type, str) and contract_type in {"implementation", "test_harness"}:
        _validate_verification_route(errors, contract)
    closure = contract.get("closure")
    if not isinstance(closure, Mapping):
        errors.append("closure must be an object")
    else:
        _require_string(errors, closure.get("rule"), "closure.rule")
        disposition = closure.get("disposition")
        if not isinstance(disposition, str) or disposition not in _ALLOWED_CLOSURE_DISPOSITIONS:
            errors.append("closure.disposition must be whole-or-explicit-partial")
        if not isinstance(closure.get("successor_required_for_partial"), bool):
            errors.append("closure.successor_required_for_partial must be boolean")
        elif disposition == "whole-or-explicit-partial" and not closure.get("successor_required_for_partial"):
            errors.append("whole-or-explicit-partial requires successor_required_for_partial=true")
    try:
        computed_source_digest = source_digest(issue)
        computed_dependency_digest = dependency_digest(issue)
    except DependencyProjectionError as exc:
        errors.append(str(exc))
        computed_source_digest = None
        computed_dependency_digest = None
    digest = contract.get("source_digest")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        errors.append("source_digest must be a lowercase SHA-256 digest")
    elif computed_source_digest is not None and digest != computed_source_digest:
        errors.append("source_digest does not match the Bead source snapshot")
    dependency_digest_value = contract.get("dependency_digest")
    if not isinstance(dependency_digest_value, str) or not _SHA256.fullmatch(dependency_digest_value):
        errors.append("dependency_digest must be a lowercase SHA-256 digest")
    elif computed_dependency_digest is not None and dependency_digest_value != computed_dependency_digest:
        errors.append("dependency_digest does not match the Bead dependency projection")
    if issue.get("dependencies") is not None and not isinstance(issue.get("dependencies"), list):
        errors.append("dependencies must be a list")
    if contract_type == "live_operation" and not contract.get("safety"):
        errors.append("live_operation requires safety clauses")
    if contract_type == "live_operation":
        _validate_receipt(errors, contract)
    if risk == "durable-mutation" and not contract.get("safety"):
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


def validate_route_registry(required_ids: Iterable[str]) -> list[str]:
    """Check that the committed route authority covers exactly the manifest IDs."""
    try:
        registry = load_registry()
    except AcceptanceRouteRegistryError as exc:
        return [str(exc)]
    required = set(required_ids)
    errors: list[str] = []
    entries = list(registry.items())
    if len(entries) != len(required):
        errors.append(f"route registry entry count mismatch: expected {len(required)}, found {len(entries)}")
    bound_ids: list[str] = []
    for identifier, entry in entries:
        bead_id = entry.get("bead_id")
        if not isinstance(bead_id, str) or not bead_id:
            errors.append(f"route registry entry {identifier!r} must bind one non-empty manifest Bead id")
        else:
            bound_ids.append(bead_id)
            if bead_id not in required:
                errors.append(f"route registry entry {identifier!r} binds unlisted Bead {bead_id!r}")
        for field in ("class", "contract_type", "dispatch"):
            value = entry.get(field)
            if not isinstance(value, str) or not value or value == "*":
                errors.append(f"route registry entry {identifier!r} has invalid {field} authority")
        targets = entry.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, str) or not target for target in targets)
        ):
            errors.append(f"route registry entry {identifier!r} must have a non-empty string target list")
    if len(bound_ids) != len(set(bound_ids)):
        errors.append("route registry contains duplicate Bead bindings")
    if set(bound_ids) != required:
        errors.append(f"route registry Bead population mismatch: expected {len(required)}, found {len(set(bound_ids))}")
    return errors


def route_registry_digest() -> str:
    return registry_digest()


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
    registry_errors = validate_route_registry(required_values)
    failures: dict[str, list[str]] = {}
    required = set(required_values)
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
        "ok": not failures and not registry_errors,
        "dispatch_blocked": bool(failures or registry_errors),
        "manifest": {
            "expected_count": _EXPECTED_MANIFEST_COUNT,
            "digest": _EXPECTED_MANIFEST_DIGEST,
        },
        "validated": len(seen),
        "route_registry_errors": registry_errors,
        "failures": failures,
        "regeneration_required": regeneration_required,
    }
    if args.json:
        print(json_dumps(report, indent=2, sort_keys=True))
    else:
        for error in registry_errors:
            print(f"route_registry: {error}")
        for bid, errors in sorted(failures.items()):
            for error in errors:
                print(f"{bid}: {error}")
        print(f"validated={len(seen)} failures={len(failures)} regeneration_required={len(regeneration_required)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
