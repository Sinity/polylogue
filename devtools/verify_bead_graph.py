"""Bead-graph invariant lint and full missing-AC census.

Run before shipping Beads state.  The command reads live structured ``bd``
state through ``bd dep cycles`` and unbounded ``bd list --all --json``.  It
never treats prose fields, titles, or labels as acceptance criteria.

Besides dependency-cycle and wave checks, the report validates that each
Bead has zero or one canonical parent derived solely from ``parent-child``
dependency records.  The JSON output is deliberately complete and stable so
the coordinator can batch the real missing-AC population without weakening
the fail-closed lint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from devtools.beads_acceptance_contracts import validate as validate_acceptance_contract


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    bead_id: str
    detail: str


def _wave(issue: dict[str, Any]) -> tuple[int | None, Finding | None]:
    """Parse the issue's ``wave:`` label."""
    for label in issue.get("labels") or []:
        if isinstance(label, str) and label.startswith("wave:"):
            raw = label[len("wave:") :]
            try:
                return int(raw), None
            except ValueError:
                return None, Finding("malformed-wave", str(issue["id"]), f"non-numeric wave label: {label!r}")
    return None, None


def _run_bd_dep_cycles() -> tuple[bool, str]:
    result = subprocess.run(["bd", "dep", "cycles"], capture_output=True, text=True, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def _run_bd_list_all() -> list[dict[str, Any]]:
    """Load the complete live population; ``-n 0`` is never a display page."""
    result = subprocess.run(
        ["bd", "list", "--all", "-n", "0", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise RuntimeError(f"bd list returned {type(payload).__name__}, expected list")
    issues: list[dict[str, Any]] = []
    for index, issue in enumerate(payload):
        if not isinstance(issue, dict):
            raise RuntimeError(
                f"bd list record {index} is {type(issue).__name__}, expected object with non-empty string id"
            )
        bead_id = issue.get("id")
        if not isinstance(bead_id, str) or not bead_id:
            raise RuntimeError(f"bd list record {index} has no non-empty string id")
        issues.append(issue)
    return issues


def _metadata(issue: dict[str, Any]) -> dict[str, Any]:
    value = issue.get("metadata")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _labels(issue: dict[str, Any]) -> list[str]:
    value = issue.get("labels")
    return sorted(label for label in value if isinstance(label, str)) if isinstance(value, list) else []


def _priority(issue: dict[str, Any]) -> int:
    value = issue.get("priority", 2)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2


def _parent_targets(issue: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    dependencies = issue.get("dependencies")
    if not isinstance(dependencies, list):
        return targets
    for dependency in dependencies:
        if not isinstance(dependency, dict) or dependency.get("type") != "parent-child":
            continue
        target = dependency.get("depends_on_id")
        if isinstance(target, str) and target:
            targets.append(target)
    return targets


def canonical_parent_map(issues: list[dict[str, Any]]) -> dict[str, str | None]:
    """Return the structured canonical parent for every known issue.

    A zero-parent issue maps to ``None``.  Multiple parent-child targets are
    intentionally represented as ``None`` because there is no canonical
    choice until the graph validator reports and an operator repairs it.
    """
    parents: dict[str, str | None] = {}
    for issue in issues:
        bead_id = str(issue.get("id", ""))
        targets = _parent_targets(issue)
        parents[bead_id] = targets[0] if len(targets) == 1 else None
    return parents


def _parent_findings(issues: list[dict[str, Any]]) -> list[Finding]:
    """Validate structured parent-child edges, independent of Bead prose."""
    by_id = {str(issue["id"]): issue for issue in issues if isinstance(issue.get("id"), str)}
    findings: list[Finding] = []
    canonical: dict[str, str] = {}
    for bead_id, issue in sorted(by_id.items()):
        for dependency in issue.get("dependencies", []) if isinstance(issue.get("dependencies"), list) else []:
            if not isinstance(dependency, dict) or dependency.get("type") != "parent-child":
                continue
            target = dependency.get("depends_on_id")
            if not isinstance(target, str) or not target:
                findings.append(Finding("malformed-parent", bead_id, f"invalid parent-child target: {target!r}"))
        targets = _parent_targets(issue)
        if len(targets) > 1:
            findings.append(Finding("multiple-parents", bead_id, f"parent-child targets={sorted(targets)}"))
            continue
        if not targets:
            continue
        parent_id = next(iter(targets))
        if parent_id not in by_id:
            findings.append(Finding("missing-parent", bead_id, f"parent-child target {parent_id!r} does not exist"))
            continue
        if parent_id == bead_id:
            findings.append(Finding("parent-self-cycle", bead_id, "parent-child target is the child itself"))
            continue
        canonical[bead_id] = parent_id

    visited: set[str] = set()
    for start in sorted(canonical):
        if start in visited:
            continue
        path: list[str] = []
        index: dict[str, int] = {}
        node = start
        while node in canonical and node not in visited:
            if node in index:
                cycle = path[index[node] :] + [node]
                findings.append(Finding("parent-cycle", node, "parent-child cycle: " + " -> ".join(cycle)))
                break
            index[node] = len(path)
            path.append(node)
            node = canonical[node]
        visited.update(path)
    return findings


def collect_findings(issues: list[dict[str, Any]]) -> list[Finding]:
    by_id = {str(issue["id"]): issue for issue in issues if isinstance(issue.get("id"), str)}
    findings: list[Finding] = _parent_findings(issues)

    for issue_id, issue in sorted(by_id.items()):
        if "acceptance_contract_v1" not in _metadata(issue):
            continue
        for error in validate_acceptance_contract(issue):
            findings.append(Finding("invalid-acceptance-contract", issue_id, error))

    waves: dict[str, int | None] = {}
    for issue_id, issue in sorted(by_id.items()):
        if issue.get("status") == "closed":
            continue
        wave_value, malformed = _wave(issue)
        waves[issue_id] = wave_value
        if malformed is not None:
            findings.append(malformed)

    for issue_id, issue in sorted(by_id.items()):
        if issue.get("status") == "closed":
            continue
        wave_labels = [label for label in _labels(issue) if label.startswith("wave:")]
        if len(wave_labels) > 1:
            findings.append(Finding("duplicate-wave", issue_id, f"labels={wave_labels}"))
        acceptance_criteria = issue.get("acceptance_criteria")
        if not isinstance(acceptance_criteria, str) or not acceptance_criteria.strip():
            detail = (
                str(issue.get("title", ""))[:60]
                if isinstance(acceptance_criteria, str)
                else "acceptance_criteria must be a non-empty string"
            )
            findings.append(Finding("missing-ac", issue_id, detail))
        wave_value = waves.get(issue_id)
        dependencies = issue.get("dependencies")
        for dependency in dependencies if isinstance(dependencies, list) else []:
            if not isinstance(dependency, dict) or dependency.get("type") != "blocks":
                continue
            blocker_id = dependency.get("depends_on_id")
            blocker = by_id.get(str(blocker_id))
            if blocker is None or blocker.get("status") == "closed":
                continue
            blocker_wave = waves.get(str(blocker_id))
            if wave_value is not None and blocker_wave is not None and blocker_wave > wave_value:
                findings.append(
                    Finding("wave-inversion", issue_id, f"(wave:{wave_value}) <- {blocker_id} (wave:{blocker_wave})")
                )
    return sorted(findings, key=lambda finding: (finding.kind, finding.bead_id, finding.detail))


def _campaigns(issue: dict[str, Any]) -> list[str]:
    """Read campaign declarations from structured metadata or labels only."""
    values: set[str] = set()
    metadata_campaign = _metadata(issue).get("campaign")
    if isinstance(metadata_campaign, str) and metadata_campaign:
        values.add(metadata_campaign)
    elif isinstance(metadata_campaign, list):
        values.update(value for value in metadata_campaign if isinstance(value, str) and value)
    values.update(label.removeprefix("campaign:") for label in _labels(issue) if label.startswith("campaign:"))
    if "campaign" in _labels(issue):
        values.add(str(issue.get("id", "")))
    return sorted(values)


def _partition(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        groups[str(item[key])].append(str(item["id"]))
    return {name: {"count": len(ids), "ids": sorted(ids)} for name, ids in sorted(groups.items())}


def missing_ac_census(issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce a complete, deterministic census of fail-closed missing ACs."""
    parents = canonical_parent_map(issues)
    missing_ids = {finding.bead_id for finding in collect_findings(issues) if finding.kind == "missing-ac"}
    rows: list[dict[str, Any]] = []
    for issue in issues:
        bead_id = str(issue.get("id", ""))
        if bead_id not in missing_ids:
            continue
        program = _metadata(issue).get("frontier_program_ref")
        program_or_parent = (
            str(program) if isinstance(program, str) and program else parents.get(bead_id) or "unparented"
        )
        campaigns = _campaigns(issue)
        rows.append(
            {
                "id": bead_id,
                "status": str(issue.get("status", "unknown")),
                "priority": _priority(issue),
                "program_or_parent": program_or_parent,
                "campaign_relevance": "declared" if campaigns else "none",
                "campaigns": campaigns,
            }
        )
    rows.sort(key=lambda row: (row["status"], row["priority"], row["program_or_parent"], row["id"]))
    return {
        "report_version": 1,
        "total": len(rows),
        "by_status": _partition(rows, "status"),
        "by_priority": _partition(rows, "priority"),
        "by_program_or_parent": _partition(rows, "program_or_parent"),
        "by_campaign_relevance": _partition(rows, "campaign_relevance"),
        "items": rows,
    }


def build_report(issues: list[dict[str, Any]], *, cycles_ok: bool, cycles_output: str) -> dict[str, Any]:
    findings = collect_findings(issues)
    by_kind: dict[str, int] = defaultdict(int)
    for finding in findings:
        by_kind[finding.kind] += 1
    return {
        "report_version": 1,
        "cycles": {"ok": cycles_ok, "output": cycles_output},
        "issues_scanned": len(issues),
        "findings": [{"kind": f.kind, "id": f.bead_id, "detail": f.detail} for f in findings],
        "counts": dict(sorted(by_kind.items())),
        "missing_ac_census": missing_ac_census(issues),
    }


def _format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    cycle_output = report["cycles"]["output"]
    if cycle_output:
        lines.append(str(cycle_output))
    for finding in report["findings"]:
        lines.append(f"{finding['kind']}: {finding['id']} {finding['detail']}")
    counts = report["counts"]
    lines.append(
        "violations: "
        f"dup_labels={counts.get('duplicate-wave', 0)} "
        f"inversions={counts.get('wave-inversion', 0)} "
        f"missing_ac={counts.get('missing-ac', 0)} "
        f"invalid_contracts={counts.get('invalid-acceptance-contract', 0)} "
        f"malformed_wave={counts.get('malformed-wave', 0)} "
        f"parent_integrity={sum(value for key, value in counts.items() if key.startswith('parent-') or key in {'multiple-parents', 'missing-parent'})}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete machine-readable graph report")
    args = parser.parse_args(argv)

    try:
        cycles_ok, cycles_output = _run_bd_dep_cycles()
        if not cycles_ok:
            if args.json:
                print(
                    json.dumps(
                        {
                            "report_version": 1,
                            "error": "dependency cycle check failed",
                            "cycles_output": cycles_output,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"bead-graph: dependency cycle check failed: {cycles_output}", file=sys.stderr)
            return 1
        issues = _run_bd_list_all()
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"report_version": 1, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"bead-graph: failed to load live Beads state: {exc}", file=sys.stderr)
        return 1
    report = build_report(issues, cycles_ok=cycles_ok, cycles_output=cycles_output)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))
    return 0 if not report["findings"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
