"""Validate structural integrity of the Beads dependency graph.

The gate inspects typed dependency records: endpoint existence, duplicate
edges, parent cardinality, and cycles.  It deliberately does not interpret
titles, labels, descriptions, acceptance prose, campaign snapshots, or a
hard-coded list of project-specific edges.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    bead_id: str
    detail: str


def _validated_issues(payload: object, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise RuntimeError(f"{source} returned {type(payload).__name__}, expected list")
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, issue in enumerate(payload):
        if not isinstance(issue, dict):
            raise RuntimeError(f"{source} record {index} is {type(issue).__name__}, expected object")
        bead_id = issue.get("id")
        if not isinstance(bead_id, str) or not bead_id:
            raise RuntimeError(f"{source} record {index} has no non-empty string id")
        if bead_id in seen:
            raise RuntimeError(f"{source} contains duplicate issue id {bead_id!r}")
        seen.add(bead_id)
        issues.append(issue)
    return issues


def _run_bd_dep_cycles() -> tuple[bool, str]:
    result = subprocess.run(["bd", "dep", "cycles"], capture_output=True, text=True, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def _run_bd_list_all() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["bd", "list", "--all", "-n", "0", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return _validated_issues(json.loads(result.stdout), source="bd list")


def _load_export(path: Path) -> list[dict[str, Any]]:
    records: list[object] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return _validated_issues(records, source=str(path))


def _dependency_records(issue: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies = issue.get("dependencies")
    return (
        [dependency for dependency in dependencies if isinstance(dependency, dict)]
        if isinstance(dependencies, list)
        else []
    )


def _parent_targets(issue: dict[str, Any]) -> list[str]:
    return [
        target
        for dependency in _dependency_records(issue)
        if dependency.get("type") == "parent-child"
        and isinstance((target := dependency.get("depends_on_id")), str)
        and target
    ]


def canonical_parent_map(issues: list[dict[str, Any]]) -> dict[str, str | None]:
    parents: dict[str, str | None] = {}
    for issue in issues:
        targets = _parent_targets(issue)
        parents[str(issue["id"])] = targets[0] if len(targets) == 1 else None
    return parents


def _cycle_findings(edges: dict[str, set[str]], *, kind: str, label: str) -> list[Finding]:
    findings: list[Finding] = []
    state: dict[str, int] = {}
    path: list[str] = []
    positions: dict[str, int] = {}
    reported: set[frozenset[str]] = set()

    def visit(node: str) -> None:
        state[node] = 1
        positions[node] = len(path)
        path.append(node)
        for target in sorted(edges.get(node, set())):
            if state.get(target, 0) == 0:
                visit(target)
            elif state.get(target) == 1:
                cycle = path[positions[target] :] + [target]
                identity = frozenset(cycle)
                if identity not in reported:
                    findings.append(Finding(kind, target, f"{label}: " + " -> ".join(cycle)))
                    reported.add(identity)
        path.pop()
        positions.pop(node)
        state[node] = 2

    for start in sorted(edges):
        if state.get(start, 0) == 0:
            visit(start)
    return findings


def collect_findings(issues: list[dict[str, Any]]) -> list[Finding]:
    by_id = {str(issue["id"]): issue for issue in issues}
    findings: list[Finding] = []
    parent_edges: dict[str, set[str]] = {}
    block_edges: dict[str, set[str]] = defaultdict(set)

    for bead_id, issue in sorted(by_id.items()):
        raw_dependencies = issue.get("dependencies")
        if raw_dependencies is not None and not isinstance(raw_dependencies, list):
            findings.append(Finding("malformed-dependencies", bead_id, "dependencies must be a list"))
            continue
        seen_edges: set[tuple[str, str]] = set()
        parents: list[str] = []
        for index, dependency in enumerate(raw_dependencies or []):
            if not isinstance(dependency, dict):
                findings.append(Finding("malformed-dependency", bead_id, f"dependency {index} is not an object"))
                continue
            dep_type = dependency.get("type")
            target = dependency.get("depends_on_id")
            if not isinstance(dep_type, str) or not dep_type or not isinstance(target, str) or not target:
                findings.append(Finding("malformed-dependency", bead_id, f"dependency {index} lacks type or target"))
                continue
            edge = (dep_type, target)
            if edge in seen_edges:
                findings.append(Finding("duplicate-dependency", bead_id, f"duplicate {dep_type} edge to {target}"))
            seen_edges.add(edge)
            if target not in by_id:
                findings.append(
                    Finding("missing-dependency-target", bead_id, f"{dep_type} target {target!r} does not exist")
                )
            if target == bead_id:
                findings.append(Finding("self-dependency", bead_id, f"{dep_type} edge targets itself"))
            if dep_type == "parent-child":
                parents.append(target)
            elif dep_type == "blocks":
                block_edges[bead_id].add(target)
        if len(parents) > 1:
            findings.append(Finding("multiple-parents", bead_id, f"parent-child targets={sorted(parents)}"))
        elif parents:
            parent_edges[bead_id] = {parents[0]}

    findings.extend(_cycle_findings(parent_edges, kind="parent-cycle", label="parent-child cycle"))
    findings.extend(_cycle_findings(block_edges, kind="blocks-cycle", label="blocks cycle"))
    return sorted(findings, key=lambda finding: (finding.kind, finding.bead_id, finding.detail))


def build_report(issues: list[dict[str, Any]], *, cycles_ok: bool, cycles_output: str) -> dict[str, Any]:
    findings = collect_findings(issues)
    structured_cycles_ok = not any(finding.kind in {"parent-cycle", "blocks-cycle"} for finding in findings)
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        counts[finding.kind] += 1
    return {
        "report_version": 2,
        "cycles": {"ok": cycles_ok and structured_cycles_ok, "output": cycles_output},
        "issues_scanned": len(issues),
        "findings": [{"kind": f.kind, "id": f.bead_id, "detail": f.detail} for f in findings],
        "counts": dict(sorted(counts.items())),
    }


def _format_report(report: dict[str, Any]) -> str:
    lines = [str(report["cycles"]["output"])] if report["cycles"]["output"] else []
    lines.extend(f"{item['kind']}: {item['id']} {item['detail']}" for item in report["findings"])
    lines.append(f"bead-graph: {report['issues_scanned']} issues, {len(report['findings'])} structural violations")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable structural report")
    parser.add_argument(
        "--export", type=Path, help="validate a JSONL export without touching the shared live Beads database"
    )
    args = parser.parse_args(argv)

    try:
        if args.export is not None:
            issues = _load_export(args.export)
            cycles_ok, cycles_output = True, ""
        else:
            cycles_ok, cycles_output = _run_bd_dep_cycles()
            if not cycles_ok:
                raise RuntimeError(f"dependency cycle check failed: {cycles_output}")
            issues = _run_bd_list_all()
        report = build_report(issues, cycles_ok=cycles_ok, cycles_output=cycles_output)
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        payload = {"report_version": 2, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"bead-graph: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _format_report(report))
    return 0 if not report["findings"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
