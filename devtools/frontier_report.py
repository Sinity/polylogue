"""Report Beads frontier batches for the Polylogue devloop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ClassifiedIssue:
    issue: dict[str, Any]
    subsystem: str
    proof_cost: str
    runtime_risk: str
    subagent_suitability: str
    schema_lane: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.issue.get("id"),
            "title": self.issue.get("title"),
            "status": self.issue.get("status"),
            "priority": self.issue.get("priority"),
            "type": self.issue.get("issue_type"),
            "labels": self.issue.get("labels", []),
            "subsystem": self.subsystem,
            "proof_cost": self.proof_cost,
            "runtime_risk": self.runtime_risk,
            "subagent_suitability": self.subagent_suitability,
            "schema_lane": self.schema_lane,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace frontier",
        description="Classify ready and in-progress Beads into devloop batches.",
    )
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository root containing the Beads workspace.")
    parser.add_argument("--limit", type=int, default=40, help="Maximum ready Beads to inspect.")
    parser.add_argument(
        "--include-in-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include in-progress Beads as occupied lanes.",
    )
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


def _text(issue: dict[str, Any]) -> str:
    fields = [
        issue.get("id"),
        issue.get("title"),
        issue.get("description"),
        issue.get("design"),
        issue.get("acceptance_criteria"),
        issue.get("notes"),
        " ".join(str(label) for label in issue.get("labels", []) if isinstance(label, str)),
    ]
    return "\n".join(str(field).lower() for field in fields if field)


def _labels(issue: dict[str, Any]) -> list[str]:
    labels = issue.get("labels", [])
    return [str(label) for label in labels] if isinstance(labels, list) else []


def _subsystem(issue: dict[str, Any]) -> str:
    for label in _labels(issue):
        if label.startswith("area:"):
            return label.removeprefix("area:")
    haystack = _text(issue)
    keyword_map = (
        ("devloop", ("devloop", "dev-loop", "frontier", "wait-ahead", "subagent")),
        ("schema", ("migration", "archive_tiers", "ddl", "user_version", "schema_version", "index schema")),
        ("storage", ("storage", "sqlite", "archive", "blob", "fts", "rebuild")),
        ("context", ("context", "recall", "memory", "injection", "handoff")),
        ("demos", ("demo", "artifact", "finding", "cold-reader", "uplift")),
        ("ops", ("daemon", "ops", "backup", "restore", "hooks", "install")),
        ("web", ("web", "reader", "ui", "vite", "preact", "browser")),
        ("mcp", ("mcp", "tool", "server")),
        ("docs", ("readme", "docs", "doctrine", "positioning")),
    )
    for subsystem, keywords in keyword_map:
        if any(keyword in haystack for keyword in keywords):
            return subsystem
    return "unclassified"


def _proof_cost(issue: dict[str, Any]) -> str:
    labels = set(_labels(issue))
    haystack = _text(issue)
    if "size:L" in labels or any(term in haystack for term in ("live archive", "rebuild", "verify --all")):
        return "high"
    if "size:M" in labels or any(term in haystack for term in ("devtools verify", "integration", "daemon")):
        return "medium"
    if "size:S" in labels or any(term in haystack for term in ("focused", "read-only", "docs")):
        return "low"
    return "unknown"


def _runtime_risk(issue: dict[str, Any]) -> str:
    haystack = _text(issue)
    if any(term in haystack for term in ("live archive", "rebuild", "reset", "backup", "restore", "daemon")):
        return "live-state"
    if any(term in haystack for term in ("migration", "archive_tiers", "storage", "sqlite", "blob", "fts")):
        return "stateful"
    if any(term in haystack for term in ("docs", "readme", "report", "audit", "plan")):
        return "read-only"
    return "normal"


def _subagent_suitability(issue: dict[str, Any]) -> str:
    haystack = _text(issue)
    if any(term in haystack for term in ("audit", "classify", "investigate", "research", "plan", "read-only")):
        return "read-only-audit"
    if any(term in haystack for term in ("docs", "readme", "report", "artifact", "catalog")):
        return "draft-or-artifact"
    if any(term in haystack for term in ("implementation", "wire", "scaffold", "add", "fix")):
        return "worker-with-file-ownership"
    return "conductor-owned"


def _schema_lane(issue: dict[str, Any]) -> bool:
    haystack = _text(issue)
    return any(
        term in haystack
        for term in (
            "migration",
            "archive_tiers",
            "user_version",
            "schema_version",
            "index schema",
            "schema migration",
            "canonical ddl",
            " ddl",
        )
    )


def _classify(issue: dict[str, Any]) -> ClassifiedIssue:
    return ClassifiedIssue(
        issue=issue,
        subsystem=_subsystem(issue),
        proof_cost=_proof_cost(issue),
        runtime_risk=_runtime_risk(issue),
        subagent_suitability=_subagent_suitability(issue),
        schema_lane=_schema_lane(issue),
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    ready = _run_bd(args.repo, ["ready", "--limit", str(args.limit)])
    in_progress = (
        _run_bd(args.repo, ["list", "--status", "in_progress", "--limit", "0"]) if args.include_in_progress else []
    )
    classified = [_classify(issue) for issue in [*ready, *in_progress]]
    groups: dict[str, list[ClassifiedIssue]] = defaultdict(list)
    for item in classified:
        groups[item.subsystem].append(item)
    return {
        "report_version": 1,
        "command": "devtools workspace frontier",
        "repo": str(args.repo),
        "counts": {
            "ready": len(ready),
            "in_progress": len(in_progress),
            "total": len(classified),
        },
        "groups": {
            name: {
                "count": len(items),
                "ready": sum(1 for item in items if item.issue.get("status") == "open"),
                "in_progress": sum(1 for item in items if item.issue.get("status") == "in_progress"),
                "high_proof_cost": sum(1 for item in items if item.proof_cost == "high"),
                "read_only_audit": sum(1 for item in items if item.subagent_suitability == "read-only-audit"),
                "schema_lane": sum(1 for item in items if item.schema_lane),
                "items": [
                    item.to_dict()
                    for item in sorted(
                        items, key=lambda item: (item.issue.get("priority", 9), str(item.issue.get("id")))
                    )
                ],
            }
            for name, items in sorted(groups.items())
        },
        "recommendations": _recommendations(classified),
    }


def _recommendations(items: list[ClassifiedIssue]) -> list[str]:
    recommendations: list[str] = []
    active_subsystems = {item.subsystem for item in items if item.issue.get("status") == "in_progress"}
    for subsystem in sorted(active_subsystems):
        ready_same = [item for item in items if item.issue.get("status") == "open" and item.subsystem == subsystem]
        if ready_same:
            recommendations.append(
                f"Batch opportunity: {subsystem} has {len(ready_same)} ready item(s) near an active lane."
            )
    audits = [
        item for item in items if item.issue.get("status") == "open" and item.subagent_suitability == "read-only-audit"
    ]
    if audits:
        ids = ", ".join(str(item.issue.get("id")) for item in audits[:5])
        recommendations.append(f"Subagent candidates: delegate read-only audit/research for {ids}.")
    high_live = [
        item
        for item in items
        if item.issue.get("status") == "open" and item.proof_cost == "high" and item.runtime_risk == "live-state"
    ]
    if high_live:
        ids = ", ".join(str(item.issue.get("id")) for item in high_live[:5])
        recommendations.append(f"Wait-ahead candidates: record devloop-wait before high-cost live proof for {ids}.")
    if any(item.schema_lane for item in items):
        recommendations.append("Schema-lane items are present; serialize with the active schema PR owner.")
    recommendations.append(
        "Velocity/Meta is mandatory: record a no-op reason, batch grouping, delegation, friction fix, or follow-up."
    )
    return recommendations


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Devloop Frontier",
        "",
        f"repo: `{report['repo']}`",
        (
            "counts: "
            f"ready={report['counts']['ready']} "
            f"in_progress={report['counts']['in_progress']} "
            f"total={report['counts']['total']}"
        ),
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines.append("")
    for name, group in report["groups"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                (
                    f"count={group['count']} ready={group['ready']} in_progress={group['in_progress']} "
                    f"high_proof_cost={group['high_proof_cost']} read_only_audit={group['read_only_audit']} "
                    f"schema_lane={group['schema_lane']}"
                ),
                "",
            ]
        )
        for item in group["items"]:
            markers = [
                f"proof:{item['proof_cost']}",
                f"risk:{item['runtime_risk']}",
                f"subagent:{item['subagent_suitability']}",
            ]
            if item["schema_lane"]:
                markers.append("schema-lane")
            lines.append(
                f"- `{item['id']}` P{item['priority']} {item['status']} - {item['title']} ({', '.join(markers)})"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_report(args)
    except RuntimeError as exc:
        print(f"frontier report failed: {exc}", file=sys.stderr)
        return 1
    output = json.dumps(report, indent=2) + "\n" if args.json else _render_markdown(report) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
