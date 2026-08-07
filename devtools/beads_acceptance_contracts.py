from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
    r"(?:<[^>]+>|\.{3}|\b(?:TBD|TODO|FIXME|as appropriate|figure out|choose an approach|add suitable tests)\b)", re.I
)


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
    metadata = issue.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return ["metadata is not valid JSON"]
    contract = metadata.get("acceptance_contract_v1") if isinstance(metadata, dict) else None
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
    for key in ("outcome", "routes", "evidence", "verification", "anti_vacuity", "closure", "source_digest"):
        if not contract.get(key):
            errors.append(f"missing {key}")
    if contract.get("contract_type") == "live_operation" and not contract.get("safety"):
        errors.append("live_operation requires safety clauses")
    if contract.get("risk") == "durable-mutation" and not contract.get("safety"):
        errors.append("durable-mutation requires safety clauses")
    for value in _strings(contract):
        if _PLACEHOLDER.search(value):
            errors.append(f"placeholder in contract: {value[:80]}")
    try:
        expected = render(contract)
        if issue.get("acceptance_criteria") != expected:
            errors.append("acceptance_criteria drifted from structured contract")
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot render contract: {exc}")
    return sorted(set(errors))


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: {exc}") from exc
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
        metadata = issue.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        carries = isinstance(metadata, dict) and "acceptance_contract_v1" in metadata
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
