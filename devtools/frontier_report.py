"""Derive the execution focus from the complete Beads frontier.

The report keeps three deliberately separate sets:

* active ambition: every open or in-progress Bead;
* active set: Beads explicitly admitted through structured frontier metadata;
* execution focus: ready, unclaimed work selected after declared resource and
  footprint-conflict constraints.

It only reports those derivations.  It never claims, releases, truncates, or
otherwise mutates Beads admission state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from devtools import bead_cluster

ROOT = Path(__file__).resolve().parents[1]

# This is a focus-selection policy, not an active-set cap.  It is visible in
# every JSON report, so a coordinator can see exactly why a ready Bead waits.
RESOURCE_POLICY: dict[str, dict[str, Any]] = {
    "schema-lane": {"max_parallel": 1, "reason": "schema changes serialize through the active schema owner"},
    "live-state": {"max_parallel": 1, "reason": "live-state work requires one operator-controlled lane"},
    "ordinary": {"max_parallel": None, "reason": "no shared resource limit declared"},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace frontier",
        description="Derive a complete, non-mutating execution focus from live Beads state.",
    )
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository root containing the Beads workspace.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Write the report to this path.")
    return parser


def _run_bd(repo: Path, args: list[str]) -> list[dict[str, Any]]:
    command = ["bd", "--readonly", *args, "--json"]
    try:
        completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"failed to run {' '.join(command)}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{' '.join(command)} returned non-JSON output") from exc
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise RuntimeError(f"{' '.join(command)} returned {type(payload).__name__}, expected list")


def _labels(issue: dict[str, Any]) -> set[str]:
    labels = issue.get("labels")
    return {label for label in labels if isinstance(label, str)} if isinstance(labels, list) else set()


def _metadata(issue: dict[str, Any]) -> dict[str, Any]:
    metadata = issue.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _priority(issue: dict[str, Any]) -> int:
    try:
        return int(issue.get("priority", 2))
    except (TypeError, ValueError):
        return 2


def _is_open(issue: dict[str, Any]) -> bool:
    return issue.get("status") in {"open", "in_progress"}


def _resource_class(issue: dict[str, Any]) -> str:
    footprint = bead_cluster._extract_footprint(issue)
    if footprint.migration_slots or "area:schema" in _labels(issue):
        return "schema-lane"
    labels = _labels(issue)
    if "resource:live-state" in labels or "risk:live-state" in labels:
        return "live-state"
    return "ordinary"


def _active_set(issues: list[dict[str, Any]]) -> list[str]:
    return sorted(
        str(issue["id"])
        for issue in issues
        if _is_open(issue) and _metadata(issue).get("frontier") == "active" and issue.get("issue_type") != "epic"
    )


def _critical_path_leverage(issues: list[dict[str, Any]]) -> dict[str, int]:
    """Count every open downstream Bead a candidate can unblock via blocks edges."""
    blocked_by: dict[str, set[str]] = defaultdict(set)
    known = {str(issue["id"]): issue for issue in issues if isinstance(issue.get("id"), str)}
    for issue in known.values():
        if not _is_open(issue):
            continue
        dependencies = issue.get("dependencies")
        for dependency in dependencies if isinstance(dependencies, list) else []:
            if not isinstance(dependency, dict) or dependency.get("type") != "blocks":
                continue
            target = dependency.get("depends_on_id")
            if isinstance(target, str) and target in known and _is_open(known[target]):
                blocked_by[target].add(str(issue["id"]))
    leverage: dict[str, int] = {}
    for blocker in blocked_by:
        reachable: set[str] = set()
        pending = list(blocked_by[blocker])
        while pending:
            child = pending.pop()
            if child in reachable:
                continue
            reachable.add(child)
            pending.extend(blocked_by.get(child, ()))
        leverage[blocker] = len(reachable)
    return leverage


def _footprint_conflicts(issue: dict[str, Any], occupied: list[dict[str, Any]]) -> list[str]:
    footprint = bead_cluster._extract_footprint(issue)
    keys = footprint.overlap_keys() | footprint.contention_keys()
    conflicts: list[str] = []
    for other in occupied:
        other_footprint = bead_cluster._extract_footprint(other)
        other_keys = other_footprint.overlap_keys() | other_footprint.contention_keys()
        if keys & other_keys:
            conflicts.append(str(other["id"]))
    return sorted(conflicts)


def _candidate_row(
    issue: dict[str, Any], *, leverage: int, occupied: list[dict[str, Any]], ready_ids: set[str]
) -> dict[str, Any]:
    resource_class = _resource_class(issue)
    return {
        "id": str(issue["id"]),
        "title": str(issue.get("title", "")),
        "status": str(issue.get("status", "unknown")),
        "priority": _priority(issue),
        "dependency_ready": str(issue["id"]) in ready_ids,
        "critical_path_leverage": leverage,
        "resource_class": resource_class,
        "conflicts_with_claims": _footprint_conflicts(issue, occupied),
        "frontier_program_ref": _metadata(issue).get("frontier_program_ref"),
    }


def derive_execution_focus(issues: list[dict[str, Any]], ready_ids: set[str]) -> dict[str, Any]:
    """Select every executable candidate permitted by the declared policy.

    Claims occupy resource classes and footprint keys.  Candidates remain in
    the report whether selected or deferred, making this a transparent focus
    derivation rather than a hidden admission or count-pruning mechanism.
    """
    open_issues = [issue for issue in issues if _is_open(issue)]
    claims = sorted(
        (issue for issue in open_issues if issue.get("status") == "in_progress"), key=lambda item: str(item["id"])
    )
    leverage = _critical_path_leverage(issues)
    candidates = [
        _candidate_row(issue, leverage=leverage.get(str(issue["id"]), 0), occupied=claims, ready_ids=ready_ids)
        for issue in open_issues
        if issue.get("status") == "open" and str(issue["id"]) in ready_ids and issue.get("issue_type") != "epic"
    ]
    candidates.sort(key=lambda row: (row["priority"], -row["critical_path_leverage"], row["id"]))

    occupied_by_resource: dict[str, list[str]] = defaultdict(list)
    for claim in claims:
        occupied_by_resource[_resource_class(claim)].append(str(claim["id"]))
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    selected_issues: list[dict[str, Any]] = []
    by_id = {str(issue["id"]): issue for issue in open_issues}
    for candidate in candidates:
        resource_class = str(candidate["resource_class"])
        policy = RESOURCE_POLICY[resource_class]
        occupied_ids = sorted(occupied_by_resource[resource_class])
        conflicts = list(candidate["conflicts_with_claims"])
        selected_conflicts = _footprint_conflicts(by_id[candidate["id"]], selected_issues)
        if conflicts:
            candidate["focus_state"] = "deferred"
            candidate["reason"] = f"footprint conflict with active claim(s): {', '.join(conflicts)}"
            deferred.append(candidate)
        elif selected_conflicts:
            candidate["focus_state"] = "deferred"
            candidate["reason"] = f"footprint conflict with selected focus: {', '.join(selected_conflicts)}"
            deferred.append(candidate)
        elif policy["max_parallel"] is not None and occupied_ids:
            candidate["focus_state"] = "deferred"
            candidate["reason"] = f"{resource_class} occupied by claim(s): {', '.join(occupied_ids)}"
            deferred.append(candidate)
        elif policy["max_parallel"] is not None and any(item["resource_class"] == resource_class for item in selected):
            candidate["focus_state"] = "deferred"
            candidate["reason"] = f"{resource_class} already selected under declared max_parallel policy"
            deferred.append(candidate)
        else:
            candidate["focus_state"] = "focus"
            candidate["reason"] = "ready, unclaimed, and permitted by declared resource policy"
            selected.append(candidate)
            selected_issues.append(by_id[candidate["id"]])
    return {
        "resource_policy": RESOURCE_POLICY,
        "occupied_claims": [str(issue["id"]) for issue in claims],
        "candidates": candidates,
        "focus": selected,
        "deferred": deferred,
    }


def build_report(issues: list[dict[str, Any]], ready: list[dict[str, Any]], *, repo: Path) -> dict[str, Any]:
    ready_ids = {str(issue["id"]) for issue in ready if isinstance(issue.get("id"), str)}
    execution_focus = derive_execution_focus(issues, ready_ids)
    open_issues = [issue for issue in issues if _is_open(issue)]
    claims = [issue for issue in open_issues if issue.get("status") == "in_progress"]
    return {
        "report_version": 2,
        "command": "devtools workspace frontier",
        "repo": str(repo),
        "counts": {
            "ambition": len(open_issues),
            "active_set": len(_active_set(issues)),
            "claims": len(claims),
            "dependency_ready": len(ready_ids),
            "execution_focus": len(execution_focus["focus"]),
            "deferred": len(execution_focus["deferred"]),
        },
        "active_set": _active_set(issues),
        "execution_focus": execution_focus,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Execution Focus",
        "",
        f"repo: `{report['repo']}`",
        (
            "counts: "
            f"ambition={counts['ambition']} active_set={counts['active_set']} claims={counts['claims']} "
            f"dependency_ready={counts['dependency_ready']} execution_focus={counts['execution_focus']} "
            f"deferred={counts['deferred']}"
        ),
        "",
        "## Focus",
        "",
    ]
    for item in report["execution_focus"]["focus"]:
        lines.append(f"- `{item['id']}` P{item['priority']} leverage={item['critical_path_leverage']} {item['title']}")
    lines.extend(["", "## Deferred", ""])
    for item in report["execution_focus"]["deferred"]:
        lines.append(f"- `{item['id']}` P{item['priority']} {item['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        issues = _run_bd(args.repo, ["list", "--all", "--limit", "0"])
        ready = _run_bd(args.repo, ["ready", "--limit", "0"])
        report = build_report(issues, ready, repo=args.repo)
    except RuntimeError as exc:
        print(f"frontier report failed: {exc}", file=sys.stderr)
        return 1
    output = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.json else _render_markdown(report) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
