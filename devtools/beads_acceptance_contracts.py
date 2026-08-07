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
_PLACEHOLDER = re.compile(
    r"(?:<[^>]+>|\.{3}|\b(?:TBD|TODO|FIXME|as appropriate|where applicable|figure out|choose an approach|add suitable tests)\b)",
    re.I,
)
_SOURCE_FIELDS = ("id", "title", "description", "design", "notes", "status", "priority", "issue_type", "updated_at")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_digest(issue: dict[str, Any]) -> str:
    """Return the digest used to bind a contract to its source Bead snapshot."""
    payload = {key: issue.get(key) for key in _SOURCE_FIELDS}
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
        if not isinstance(closure.get("successor_required_for_partial"), bool):
            errors.append("closure.successor_required_for_partial must be boolean")
    digest = contract.get("source_digest")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        errors.append("source_digest must be a lowercase SHA-256 digest")
    elif digest != source_digest(issue):
        errors.append("source_digest does not match the Bead source snapshot")
    if contract.get("contract_type") == "live_operation" and not contract.get("safety"):
        errors.append("live_operation requires safety clauses")
    if contract.get("risk") == "durable-mutation" and not contract.get("safety"):
        errors.append("durable-mutation requires safety clauses")
    for value in _strings(contract):
        if _PLACEHOLDER.search(value):
            errors.append(f"placeholder in contract: {value[:80]}")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate structured Beads acceptance contracts without guessing from prose."
    )
    parser.add_argument("issues", type=Path, nargs="?", default=Path(".beads/issues.jsonl"))
    parser.add_argument(
        "--manifest", type=Path, help="optional newline-separated Bead ids that must carry a valid contract"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    required = set(args.manifest.read_text().split()) if args.manifest else None
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
