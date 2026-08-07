from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
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
_PLACEHOLDER = re.compile(
    r"(?:<[^>]+>|\.{3}|\b(?:TBD|TODO|FIXME|as appropriate|where applicable|figure out|choose an approach|add suitable tests)\b)",
    re.I,
)
_SOURCE_FIELDS = ("id", "title", "description", "design", "notes", "priority", "issue_type")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_ID = re.compile(r"^polylogue-[a-z0-9]+(?:\.[a-z0-9]+)*$")
_DEFAULT_MANIFEST = Path(__file__).parents[1] / "docs" / "plans" / "beads-acceptance-contracts-2026-08-07.txt"
_EXPECTED_MANIFEST_COUNT = 218
_EXPECTED_MANIFEST_DIGEST = "703df11c81dae8af6d7106bc4737502ca8baddc9013916bbb68922696d8206b5"


def source_digest(issue: dict[str, Any]) -> str:
    """Return the digest used to bind a contract to its source Bead snapshot."""
    payload = {key: issue.get(key) for key in _SOURCE_FIELDS}
    dependencies: list[dict[str, str | None]] = []
    for dependency in issue.get("dependencies") or []:
        if isinstance(dependency, dict):
            dependencies.append(
                {
                    "depends_on_id": dependency.get("depends_on_id") or dependency.get("to_id") or dependency.get("id"),
                    "type": dependency.get("type") or dependency.get("dep_type"),
                }
            )
        elif isinstance(dependency, str):
            dependencies.append({"depends_on_id": dependency, "type": None})
    payload["dependencies"] = sorted(
        dependencies,
        key=lambda dependency: (dependency["depends_on_id"] or "", dependency["type"] or ""),
    )
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


def _require_string_list(errors: list[str], contract: dict[str, Any], key: str, *, optional: bool = False) -> None:
    value = contract.get(key)
    if value is None and optional:
        return
    if value == [] and optional:
        return
    if not isinstance(value, list) or not value:
        errors.append(f"{key} must be a non-empty list of strings")
        return
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{key} must contain only non-empty strings")


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
    add("Closure disposition", contract["closure"]["disposition"])
    add("Closure", contract["closure"]["rule"])
    return "\n".join(rows)


def validate(issue: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    metadata = _decode_document(issue.get("metadata") or {})
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
    closure = contract.get("closure")
    if not isinstance(closure, dict):
        errors.append("closure must be an object")
    else:
        _require_string(errors, closure.get("rule"), "closure.rule")
        if closure.get("disposition") not in _ALLOWED_CLOSURE_DISPOSITIONS:
            errors.append("closure.disposition must be whole-or-explicit-partial")
        if not isinstance(closure.get("successor_required_for_partial"), bool):
            errors.append("closure.successor_required_for_partial must be boolean")
    digest = contract.get("source_digest")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        errors.append("source_digest must be a lowercase SHA-256 digest")
    elif digest != source_digest(issue):
        errors.append("source_digest does not match the Bead source snapshot")
    if contract.get("contract_type") == "live_operation" and not contract.get("safety"):
        errors.append("live_operation requires safety clauses")
    verification = contract.get("verification")
    if contract.get("contract_type") in {"implementation", "test_harness"} and not (
        isinstance(verification, list) and any("`devtools verify`" in value for value in verification)
    ):
        errors.append(f"{contract['contract_type']} requires the affected-test `devtools verify` baseline")
    if contract.get("contract_type") == "live_operation" and not (
        isinstance(verification, list)
        and any(
            "receipt" in value.casefold()
            and not re.search(r"\b(?:no|not|without|never)\b[^.\n]{0,40}\breceipt\b", value.casefold())
            for value in verification
        )
    ):
        errors.append("live_operation requires typed receipt verification")
    if contract.get("risk") == "durable-mutation" and not contract.get("safety"):
        errors.append("durable-mutation requires safety clauses")
    for value in _strings(contract):
        if _PLACEHOLDER.search(value):
            errors.append(f"placeholder in contract: {value[:80]}")
    if isinstance(contract.get("evidence"), list) and any(
        isinstance(value, str) and value[:1].islower() for value in contract["evidence"]
    ):
        errors.append("evidence contains a lowercase fragment")
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
    if not args.manifest.is_file():
        raise SystemExit(f"{args.manifest}: acceptance-contract manifest is missing")
    required_values = args.manifest.read_text(encoding="utf-8").split()
    if (
        len(required_values) != _EXPECTED_MANIFEST_COUNT
        or len(set(required_values)) != _EXPECTED_MANIFEST_COUNT
        or any(not _MANIFEST_ID.fullmatch(value) for value in required_values)
    ):
        raise SystemExit(f"{args.manifest}: acceptance-contract manifest inventory is invalid")
    manifest_digest = hashlib.sha256(("\n".join(sorted(required_values)) + "\n").encode("utf-8")).hexdigest()
    if manifest_digest != _EXPECTED_MANIFEST_DIGEST:
        raise SystemExit(f"{args.manifest}: acceptance-contract manifest inventory changed")
    required = set(required_values)
    failures = {}
    seen = set()
    for issue in load(args.issues):
        bid = issue.get("id")
        if not bid or (required is not None and bid not in required):
            continue
        metadata = _decode_document(issue.get("metadata") or {}) or {}
        carries = "acceptance_contract_v1" in metadata
        if required is None and not carries:
            continue
        seen.add(bid)
        errors = validate(issue)
        if errors:
            failures[bid] = errors
    if required is not None:
        for missing in sorted(required - seen):
            failures[missing] = ["manifest id missing from issues or contract"]
    if args.json:
        print(json.dumps({"ok": not failures, "failures": failures}, indent=2, sort_keys=True))
    else:
        for bid, errors in sorted(failures.items()):
            for error in errors:
                print(f"{bid}: {error}")
        print(f"validated={len(seen)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
