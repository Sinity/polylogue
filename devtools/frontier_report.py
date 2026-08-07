"""Derive the execution focus from the complete Beads frontier.

The report keeps three deliberately separate sets:

* full ambition: every open or in-progress Bead;
* active set: Beads explicitly admitted through structured frontier metadata;
* execution focus: ready, unclaimed admitted active leaves selected after
  declared resource and footprint-conflict constraints.

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

_MAX_FRONTIER_RECORDS = 100_000
_SCHEMA_TEXT_MARKERS = (
    "index_schema_version",
    "source_schema_version",
    "user_schema_version",
    "canonical ddl",
    "schema migration",
    "schema change",
)
_CANONICAL_TIER_DDL_SUFFIXES = (
    "storage/sqlite/archive_tiers/index.py",
    "storage/sqlite/archive_tiers/source.py",
    "storage/sqlite/archive_tiers/user.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace frontier",
        description="Derive a complete, non-mutating execution focus from live Beads state.",
    )
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository root containing the Beads workspace.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Write the report to this path.")
    return parser


def _run_bd(repo: Path, args: list[str]) -> list[Any]:
    command = [
        "bd",
        "--readonly",
        *args,
        "--limit",
        str(_MAX_FRONTIER_RECORDS + 1),
        "--max-rows",
        str(_MAX_FRONTIER_RECORDS + 1),
        "--json",
    ]
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
        if len(payload) > _MAX_FRONTIER_RECORDS:
            raise RuntimeError(
                f"{' '.join(command)} exceeded the {_MAX_FRONTIER_RECORDS:,}-record complete-report bound"
            )
        return payload
    raise RuntimeError(f"{' '.join(command)} returned {type(payload).__name__}, expected list")


def _normalize_issues(records: list[Any], *, source: str) -> list[dict[str, Any]]:
    """Validate the live Beads records once before report derivation."""
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"{source} record {index} is {type(record).__name__}, expected object with string id")
        bead_id = record.get("id")
        if not isinstance(bead_id, str) or not bead_id:
            raise RuntimeError(f"{source} record {index} has no non-empty string id")
        if bead_id in seen_ids:
            raise RuntimeError(f"{source} record {index} duplicates id {bead_id!r}")
        seen_ids.add(bead_id)
        normalized.append(record)
    return normalized


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


def _is_executable_leaf(issue: dict[str, Any]) -> bool:
    """Whether a record represents work that can occupy an execution lane."""
    return issue.get("issue_type") not in {"epic", "program"}


def _resource_classes(issue: dict[str, Any], footprint: bead_cluster.Footprint) -> tuple[str, ...]:
    labels = _labels(issue)
    text = " ".join(
        str(issue.get(field, "")) for field in ("design", "notes", "acceptance_criteria", "description")
    ).lower()
    resources: list[str] = []
    if (
        footprint.migration_slots
        or "area:schema" in labels
        or any(marker in text for marker in _SCHEMA_TEXT_MARKERS)
        or any("schema" in path.lower() and "storage" in path.lower() for path in footprint.files)
        or any(path.lower().endswith(suffix) for path in footprint.files for suffix in _CANONICAL_TIER_DDL_SUFFIXES)
    ):
        resources.append("schema-lane")
    if "resource:live-state" in labels or "risk:live-state" in labels:
        resources.append("live-state")
    return tuple(resources) or ("ordinary",)


def _active_set(issues: list[dict[str, Any]]) -> list[str]:
    return sorted(
        str(issue["id"])
        for issue in issues
        if _is_open(issue) and _is_executable_leaf(issue) and _metadata(issue).get("frontier") == "active"
    )


def _active_set_rows(issues: list[dict[str, Any]], ready_ids: set[str]) -> list[dict[str, Any]]:
    """Expose readiness, blockers, and program grouping for admitted leaves."""
    active_ids = set(_active_set(issues))
    by_id = {str(issue["id"]): issue for issue in issues}
    rows: list[dict[str, Any]] = []
    for issue_id in sorted(active_ids):
        issue = by_id[issue_id]
        blockers = sorted(
            (
                str(dependency["depends_on_id"])
                if str(dependency.get("depends_on_id", "")) in by_id
                else f"{str(dependency.get('depends_on_id', '')) or '<missing-id>'} (missing)"
            )
            for dependency in issue.get("dependencies", [])
            if isinstance(dependency, dict)
            and dependency.get("type") == "blocks"
            and (
                str(dependency.get("depends_on_id", "")) not in by_id
                or by_id[str(dependency["depends_on_id"])].get("status") != "closed"
            )
        )
        rows.append(
            {
                "id": issue_id,
                "title": str(issue.get("title", "")),
                "status": str(issue.get("status", "unknown")),
                "priority": _priority(issue),
                "dependency_ready": issue_id in ready_ids,
                "blocked_by": blockers,
                "frontier_program_ref": _metadata(issue).get("frontier_program_ref"),
            }
        )
    return rows


def _full_ambition(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every open Bead with the fields needed to audit its admission."""
    rows: list[dict[str, Any]] = []
    for issue in issues:
        if not _is_open(issue):
            continue
        metadata = _metadata(issue)
        rows.append(
            {
                "id": str(issue["id"]),
                "title": str(issue.get("title", "")),
                "status": str(issue.get("status", "unknown")),
                "priority": _priority(issue),
                "issue_type": str(issue.get("issue_type", "unknown")),
                "frontier": metadata.get("frontier"),
                "frontier_program_ref": metadata.get("frontier_program_ref"),
                "horizons": sorted(label for label in _labels(issue) if label.startswith("horizon:")),
            }
        )
    return sorted(rows, key=lambda row: str(row["id"]))


def _critical_path_leverage(issues: list[dict[str, Any]]) -> dict[str, int]:
    """Count downstream work unblocked through exclusively single-blocker paths."""
    blocked_by: dict[str, set[str]] = defaultdict(set)
    known = {issue["id"]: issue for issue in issues}
    for child_id, issue in known.items():
        if not _is_open(issue):
            continue
        dependencies = issue.get("dependencies")
        remaining_blockers: set[str] = set()
        for dependency in dependencies if isinstance(dependencies, list) else []:
            if not isinstance(dependency, dict) or dependency.get("type") != "blocks":
                continue
            target = dependency.get("depends_on_id")
            if not isinstance(target, str) or not target:
                continue
            blocker = known.get(target)
            if blocker is None or blocker.get("status") != "closed":
                remaining_blockers.add(target)
        if len(remaining_blockers) == 1:
            blocked_by[next(iter(remaining_blockers))].add(child_id)
    leverage: dict[str, int] = {}
    for blocker_id in blocked_by:
        reachable: set[str] = set()
        pending = list(blocked_by[blocker_id])
        while pending:
            child = pending.pop()
            if child in reachable:
                continue
            reachable.add(child)
            pending.extend(blocked_by.get(child, ()))
        leverage[blocker_id] = len(reachable)
    return leverage


def _footprint_conflicts(issue_id: str, occupied_ids: list[str], footprint_keys: dict[str, set[str]]) -> list[str]:
    keys = footprint_keys[issue_id]
    return sorted(other_id for other_id in occupied_ids if keys & footprint_keys[other_id])


def _candidate_row(
    issue: dict[str, Any],
    *,
    footprint: bead_cluster.Footprint,
    footprint_keys: dict[str, set[str]],
    leverage: int,
    occupied_ids: list[str],
    ready_ids: set[str],
) -> dict[str, Any]:
    issue_id = issue["id"]
    resource_classes = _resource_classes(issue, footprint)
    return {
        "id": issue_id,
        "title": str(issue.get("title", "")),
        "status": str(issue.get("status", "unknown")),
        "priority": _priority(issue),
        "dependency_ready": issue_id in ready_ids,
        "critical_path_leverage": leverage,
        "resource_class": resource_classes[0],
        "resource_classes": list(resource_classes),
        "conflicts_with_claims": _footprint_conflicts(issue_id, occupied_ids, footprint_keys),
        "frontier_program_ref": _metadata(issue).get("frontier_program_ref"),
    }


def derive_execution_focus(issues: list[dict[str, Any]], ready_ids: set[str]) -> dict[str, Any]:
    """Select every executable candidate permitted by the declared policy.

    Claims occupy resource classes and footprint keys.  Candidates remain in
    the report whether selected or deferred, making this a transparent focus
    derivation rather than a hidden admission or count-pruning mechanism.
    """
    open_issues = [issue for issue in issues if _is_open(issue)]
    footprints = {issue["id"]: bead_cluster.extract_footprint(issue) for issue in open_issues}
    footprint_keys = {
        issue_id: footprint.overlap_keys() | footprint.contention_keys() for issue_id, footprint in footprints.items()
    }
    claims = sorted(
        (issue for issue in open_issues if issue.get("status") == "in_progress" and _is_executable_leaf(issue)),
        key=lambda item: item["id"],
    )
    claim_ids = [issue["id"] for issue in claims]
    ambiguous_claim_ids = sorted(issue_id for issue_id in claim_ids if not footprint_keys[issue_id])
    leverage = _critical_path_leverage(issues)
    active_ids = set(_active_set(issues))
    candidates = [
        _candidate_row(
            issue,
            footprint=footprints[issue["id"]],
            footprint_keys=footprint_keys,
            leverage=leverage.get(issue["id"], 0),
            occupied_ids=claim_ids,
            ready_ids=ready_ids,
        )
        for issue in open_issues
        if issue.get("status") == "open" and issue["id"] in ready_ids and issue["id"] in active_ids
    ]
    candidates.sort(key=lambda row: (row["priority"], -row["critical_path_leverage"], row["id"]))

    occupied_by_resource: dict[str, list[str]] = defaultdict(list)
    for claim in claims:
        for resource_class in _resource_classes(claim, footprints[claim["id"]]):
            occupied_by_resource[resource_class].append(claim["id"])
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    selected_by_resource: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        resource_classes = [str(value) for value in candidate["resource_classes"]]
        conflicts = list(candidate["conflicts_with_claims"])
        selected_conflicts = _footprint_conflicts(candidate["id"], selected_by_resource["all"], footprint_keys)
        saturated_resources = [
            required_resource
            for required_resource in resource_classes
            if (
                (max_parallel := RESOURCE_POLICY[required_resource]["max_parallel"]) is not None
                and (
                    len(occupied_by_resource[required_resource]) + len(selected_by_resource[required_resource])
                    >= max_parallel
                )
            )
        ]
        if not footprint_keys[candidate["id"]]:
            candidate["focus_state"] = "deferred"
            candidate["reason"] = "footprint is ambiguous; confirm ownership before parallel focus"
            deferred.append(candidate)
        elif ambiguous_claim_ids:
            candidate["focus_state"] = "deferred"
            candidate["reason"] = (
                "active claim footprint is ambiguous; confirm ownership before parallel focus: "
                + ", ".join(ambiguous_claim_ids)
            )
            deferred.append(candidate)
        elif conflicts:
            candidate["focus_state"] = "deferred"
            candidate["reason"] = f"footprint conflict with active claim(s): {', '.join(conflicts)}"
            deferred.append(candidate)
        elif selected_conflicts:
            candidate["focus_state"] = "deferred"
            candidate["reason"] = f"footprint conflict with selected focus: {', '.join(selected_conflicts)}"
            deferred.append(candidate)
        elif saturated_resources:
            saturated_resource = saturated_resources[0]
            saturated_occupied_ids = sorted(occupied_by_resource[saturated_resource])
            candidate["focus_state"] = "deferred"
            if saturated_occupied_ids:
                candidate["reason"] = f"{saturated_resource} occupied by claim(s): {', '.join(saturated_occupied_ids)}"
            else:
                candidate["reason"] = (
                    f"{saturated_resource} occupied by selected focus: "
                    f"{', '.join(selected_by_resource[saturated_resource])}"
                )
            deferred.append(candidate)
        else:
            candidate["focus_state"] = "focus"
            candidate["reason"] = "ready, unclaimed, and permitted by declared resource policy"
            selected.append(candidate)
            for required_resource in resource_classes:
                selected_by_resource[required_resource].append(candidate["id"])
            selected_by_resource["all"].append(candidate["id"])
    return {
        "resource_policy": RESOURCE_POLICY,
        "occupied_claims": claim_ids,
        "candidates": candidates,
        "focus": selected,
        "deferred": deferred,
    }


def build_report(issues: list[Any], ready: list[Any], *, repo: Path) -> dict[str, Any]:
    issues = _normalize_issues(issues, source="bd list")
    ready = _normalize_issues(ready, source="bd ready")
    ready_ids = {issue["id"] for issue in ready}
    issue_ids = {issue["id"] for issue in issues}
    missing_ready_ids = sorted(ready_ids - issue_ids)
    if missing_ready_ids:
        raise RuntimeError(
            "bd ready snapshot contains IDs absent from bd list snapshot: " + ", ".join(missing_ready_ids)
        )
    execution_focus = derive_execution_focus(issues, ready_ids)
    ambition = _full_ambition(issues)
    claims = [
        issue
        for issue in issues
        if _is_open(issue) and issue.get("status") == "in_progress" and _is_executable_leaf(issue)
    ]
    return {
        "report_version": 3,
        "command": "devtools workspace frontier",
        "repo": str(repo),
        "counts": {
            "ambition": len(ambition),
            "active_set": len(_active_set(issues)),
            "claims": len(claims),
            "dependency_ready": len(ready_ids),
            "execution_focus": len(execution_focus["focus"]),
            "deferred": len(execution_focus["deferred"]),
        },
        "ambition": ambition,
        "active_set": _active_set(issues),
        "active_set_rows": _active_set_rows(issues, ready_ids),
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
        "## Active Set",
        "",
    ]
    for item in report.get("active_set_rows", []):
        blockers = ", ".join(item["blocked_by"]) or "none"
        program = item["frontier_program_ref"] or "no program"
        lines.append(
            f"- `{item['id']}` P{item['priority']} ready={item['dependency_ready']} "
            f"blocked_by={blockers} program={program} {item['title']}"
        )
    lines.extend(
        [
            "",
            "## Focus",
            "",
        ]
    )
    for item in report["execution_focus"]["focus"]:
        lines.append(f"- `{item['id']}` P{item['priority']} leverage={item['critical_path_leverage']} {item['title']}")
    lines.extend(["", "## Deferred", ""])
    for item in report["execution_focus"]["deferred"]:
        lines.append(f"- `{item['id']}` P{item['priority']} {item['reason']}")
    lines.extend(["", "## Full Ambition", ""])
    for item in report["ambition"]:
        horizons = ", ".join(item["horizons"]) or "no horizon"
        program = item["frontier_program_ref"] or "no program"
        lines.append(
            f"- `{item['id']}` P{item['priority']} {item['status']} {item['issue_type']} "
            f"program={program} horizons={horizons} {item['title']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        issues = _run_bd(args.repo, ["list", "--all"])
        ready = _run_bd(args.repo, ["ready"])
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
