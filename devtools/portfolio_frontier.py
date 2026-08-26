"""Read-only portfolio views for the external Beads task authority.

The policy deliberately keeps ambition, admission, and execution focus as
separate projections.  This module accepts a complete iterable or a bounded
page callback so callers cannot accidentally turn a large export into an
unbounded ``bd list`` materialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

HORIZONS = frozenset({"frontier", "mid", "vision"})


class PortfolioPolicyError(ValueError):
    """The planning input is incomplete or semantically unsafe."""


@dataclass(frozen=True)
class ActivePolicy:
    target: int = 30
    warning: int = 50
    stale_claim_days: int = 7
    focus_limit: int = 4


def _labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("labels", ())
    return tuple(str(item) for item in value) if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _meta(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata", {})
    if not isinstance(value, Mapping):
        raise PortfolioPolicyError(f"{row.get('id', '<unknown>')}: metadata must be an object")
    return value


def _horizon(row: Mapping[str, Any]) -> str | None:
    values = {label.removeprefix("horizon:") for label in _labels(row) if label.startswith("horizon:")}
    if len(values) != 1 or next(iter(values), None) not in HORIZONS:
        return None
    return next(iter(values))


def _blocked_by(row: Mapping[str, Any]) -> tuple[str, ...]:
    deps = row.get("dependencies", ())
    if not isinstance(deps, Sequence) or isinstance(deps, str):
        return ()
    return tuple(
        str(dep.get("depends_on_id")) for dep in deps if isinstance(dep, Mapping) and dep.get("type") == "blocks"
    )


def _parent_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    if row.get("parent"):
        values.append(str(row["parent"]))
    deps = row.get("dependencies", ())
    if isinstance(deps, Sequence) and not isinstance(deps, str):
        values.extend(
            str(dep.get("depends_on_id"))
            for dep in deps
            if isinstance(dep, Mapping) and dep.get("type") == "parent-child"
        )
    return tuple(dict.fromkeys(values))


def _keys(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(item.strip() for item in value.split(";") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, str):
        return frozenset(str(item).strip() for item in value if str(item).strip())
    return frozenset()


def enumerate_complete(
    pages: Callable[[str | None, int], tuple[Sequence[Mapping[str, Any]], str | None]],
    *,
    page_size: int = 500,
) -> tuple[dict[str, Any], ...]:
    """Read a complete bounded page stream, rejecting truncation/repetition."""
    if page_size < 1:
        raise PortfolioPolicyError("page size must be positive")
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen: set[str | None] = set()
    while True:
        if cursor in seen:
            raise PortfolioPolicyError(f"planning-surface-incomplete: repeating page cursor {cursor!r}")
        seen.add(cursor)
        page, next_cursor = pages(cursor, page_size)
        if len(page) > page_size:
            raise PortfolioPolicyError("planning-surface-incomplete: page exceeded requested bound")
        rows.extend(dict(row) for row in page)
        if next_cursor is None:
            return tuple(rows)
        if next_cursor == cursor:
            raise PortfolioPolicyError("planning-surface-incomplete: non-progressing page cursor")
        cursor = str(next_cursor)


def _validate_receipt(receipt: Mapping[str, Any] | None) -> None:
    if receipt is None:
        return
    required = {"schema", "complete", "source_fingerprint", "rows"}
    if not required <= set(receipt) or receipt.get("complete") is not True or not isinstance(receipt.get("rows"), int):
        raise PortfolioPolicyError("planning-surface-corrupt: sync receipt is missing completeness or row evidence")
    if (
        receipt["rows"] < 0
        or not isinstance(receipt.get("source_fingerprint"), str)
        or not receipt["source_fingerprint"]
    ):
        raise PortfolioPolicyError("planning-surface-corrupt: sync receipt has invalid source identity")


def build_views(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: ActivePolicy = ActivePolicy(),
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build full ambition, active admission, and derived execution focus."""
    _validate_receipt(receipt)
    records = [dict(row) for row in rows]
    by_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in records:
        bead_id = str(row.get("id", ""))
        if not bead_id or bead_id in by_id:
            errors.append(f"duplicate or missing issue id: {bead_id or '<missing>'}")
        by_id[bead_id] = row
    for row in records:
        bead_id = str(row.get("id"))
        if row.get("status") == "closed":
            continue
        if _horizon(row) is None:
            errors.append(f"{bead_id}: exactly one horizon:frontier, horizon:mid, or horizon:vision is required")
        parents = _parent_ids(row)
        if len(parents) > 1:
            errors.append(f"{bead_id}: multiple canonical parents: {', '.join(parents)}")
        if parents and parents[0] not in by_id:
            errors.append(f"{bead_id}: missing canonical parent {parents[0]}")
        for dep in _blocked_by(row):
            if dep not in by_id:
                errors.append(f"{bead_id}: missing dependency {dep}")

    programs = {
        bead_id
        for bead_id, row in by_id.items()
        if row.get("issue_type") == "epic" and _meta(row).get("frontier_program") == "active"
    }
    active: list[dict[str, Any]] = []
    for row in records:
        bead_id = str(row.get("id"))
        meta = _meta(row)
        if row.get("status") == "closed" or meta.get("frontier") != "active":
            continue
        if row.get("issue_type") == "epic":
            errors.append(f"{bead_id}: active epics cannot be leaves")
            continue
        program = meta.get("frontier_program_ref")
        if not isinstance(program, str) or program not in programs:
            errors.append(f"{bead_id}: invalid active program ref {program!r}")
        missing = [
            field
            for field in ("design", "acceptance_criteria", "area")
            if not row.get(field) and not (field == "area" and any(label.startswith("area:") for label in _labels(row)))
        ]
        if missing:
            errors.append(f"{bead_id}: missing execution contract: {', '.join(missing)}")
        assignee = (
            row.get("assignee") or row.get("owner") if row.get("status") == "in_progress" else row.get("assignee")
        )
        if row.get("status") == "open" and assignee:
            errors.append(f"{bead_id}: stale ownership on open issue")
        active.append(
            {
                "id": bead_id,
                "program": program,
                "status": row.get("status"),
                "horizon": _horizon(row),
                "priority": row.get("priority", 4),
                "blocked_by": list(_blocked_by(row)),
                "claims": bool(assignee),
                "conflict_keys": sorted(_keys(meta.get("conflict_keys", ()))),
            }
        )

    blockers = {item["id"]: item["blocked_by"] for item in active if item["blocked_by"]}
    unlocks: dict[str, list[str]] = defaultdict(list)
    for bead_id, deps in blockers.items():
        for dep in deps:
            unlocks[dep].append(bead_id)
    focus_candidates = [item for item in active if item["status"] == "in_progress" or not item["blocked_by"]]
    focus_candidates.sort(
        key=lambda item: (
            not item["claims"],
            int(item["priority"] or 4),
            -len(unlocks.get(item["id"], [])),
            item["id"],
        )
    )
    focus: list[dict[str, Any]] = []
    used_conflicts: set[str] = set()
    for item in focus_candidates:
        conflicts = set(item["conflict_keys"])
        if conflicts & used_conflicts:
            continue
        focus.append(item)
        used_conflicts.update(conflicts)
        if len(focus) >= policy.focus_limit:
            break
    diagnostics = []
    if len(active) > policy.warning:
        diagnostics.append(
            f"active set has {len(active)} leaves; investigate unexplained growth beyond {policy.warning}"
        )
    elif len(active) > policy.target:
        diagnostics.append(f"active set has {len(active)} leaves; above soft target {policy.target}")
    if errors:
        raise PortfolioPolicyError("portfolio policy failed: " + "; ".join(errors))
    return {
        "ambition": {
            horizon: sorted(
                item["id"] for item in records if item.get("status") != "closed" and _horizon(item) == horizon
            )
            for horizon in sorted(HORIZONS)
        },
        "active": {
            "count": len(active),
            "target": policy.target,
            "warning": policy.warning,
            "leaves": active,
            "programs": {
                program: sorted(item["id"] for item in active if item["program"] == program)
                for program in sorted(programs)
            },
            "blockers": blockers,
            "unlocks": dict(unlocks),
        },
        "execution_focus": focus,
        "diagnostics": diagnostics,
        "ok": True,
    }


def main(argv: list[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", help="complete JSON or JSONL export path")
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--warning", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    try:
        with open(args.export, encoding="utf-8") as handle:
            rows = (
                json.load(handle)
                if args.export.endswith(".json")
                else [json.loads(line) for line in handle if line.strip()]
            )
        report = build_views(rows, policy=ActivePolicy(target=args.target, warning=args.warning))
    except (OSError, json.JSONDecodeError, PortfolioPolicyError) as exc:
        print(f"portfolio-frontier: {exc}", file=output)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True), file=output)
    else:
        print(
            f"active leaves: {report['active']['count']} (soft target {args.target}, warning {args.warning})",
            file=output,
        )
        print(f"execution focus: {', '.join(item['id'] for item in report['execution_focus']) or 'none'}", file=output)
        for diagnostic in report["diagnostics"]:
            print(f"WARNING: {diagnostic}", file=output)
    return 0
