"""Evidence dashboard and changed-path traceability.

Reads real artifacts emitted by the verification pipeline and presents them
as a single dashboard for operators and PR consumers. Unlike
``devtools evidence-report`` (which focuses on verify-history + suppressions),
this command aggregates the full pytest-first evidence surface introduced by
PRs #1083/#1086/#1087/#1088:

- pytest health from ``.cache/verify/last-pytest.json``;
- coverage from ``.coverage`` / ``coverage.xml`` when present;
- benchmark/SLO catalog rows and their required-artifact coverage;
- static gate status from ``.cache/verify-history.jsonl``;
- witness lifecycle counts;
- mutation/benchmark campaign freshness.

The ``trace`` subcommand walks changed paths through the same artifact sources
to produce a change -> claim -> evidence -> gate -> merge view suitable for
PR comments and agent consumption.

All sections are backed by real artifacts. Sections with no underlying file
are reported as ``"available": false`` with the reason — no aspirational rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devtools import repo_root as _get_root

ROOT = _get_root()

# Artifact paths (relative to repo root).
PYTEST_REPORT_REL = Path(".cache/verify/last-pytest.json")
VERIFY_HISTORY_REL = Path(".cache/verify-history.jsonl")
LAST_VERIFY_RESULT_REL = Path(".cache/last-verify-result.json")
COVERAGE_DATA_REL = Path(".coverage")
COVERAGE_XML_REL = Path("coverage.xml")
WITNESSES_COMMITTED_REL = Path("tests/witnesses")
WITNESSES_LOCAL_REL = Path(".local/witnesses")
BENCHMARK_CAMPAIGNS_REL = Path(".local/benchmark-campaigns")
MUTATION_CAMPAIGNS_REL = Path(".local/mutation-campaigns")
SLO_CATALOG_REL = Path("docs/plans/slo-catalog.yaml")

DEFAULT_STALE_DAYS = 7


# ──────────────────────────────────────────────────────────────────────
# Section: pytest health
# ──────────────────────────────────────────────────────────────────────


def _pytest_health(root: Path, *, now: datetime) -> dict[str, Any]:
    report_path = root / PYTEST_REPORT_REL
    if not report_path.exists():
        return {
            "available": False,
            "reason": f"missing {PYTEST_REPORT_REL} — run devtools verify",
            "path": str(PYTEST_REPORT_REL),
        }
    try:
        raw = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "reason": f"unreadable {PYTEST_REPORT_REL}: {exc}",
            "path": str(PYTEST_REPORT_REL),
        }
    if not isinstance(raw, dict):
        return {
            "available": False,
            "reason": f"{PYTEST_REPORT_REL} not a JSON object",
            "path": str(PYTEST_REPORT_REL),
        }
    raw_summary = raw.get("summary")
    summary: dict[str, Any] = raw_summary if isinstance(raw_summary, dict) else {}
    mtime = datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc)
    age_days = (now - mtime).days
    counts: dict[str, int] = {}
    for key in ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "total"):
        value = summary.get(key)
        if isinstance(value, int):
            counts[key] = value
    failed = counts.get("failed", 0) + counts.get("error", 0)
    status = "ok" if failed == 0 else "fail"
    return {
        "available": True,
        "path": str(PYTEST_REPORT_REL),
        "status": status,
        "counts": counts,
        "duration_s": (round(float(raw["duration"]), 2) if isinstance(raw.get("duration"), (int, float)) else None),
        "last_run": mtime.isoformat(),
        "age_days": age_days,
        "stale": age_days > DEFAULT_STALE_DAYS,
    }


# ──────────────────────────────────────────────────────────────────────
# Section: coverage
# ──────────────────────────────────────────────────────────────────────


def _coverage(root: Path) -> dict[str, Any]:
    xml_path = root / COVERAGE_XML_REL
    data_path = root / COVERAGE_DATA_REL
    if xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            attrib = tree.getroot().attrib
            line_rate = float(attrib.get("line-rate", "0"))
            return {
                "available": True,
                "path": str(COVERAGE_XML_REL),
                "line_rate": round(line_rate, 4),
                "percent": round(line_rate * 100, 2),
                "lines_covered": int(attrib["lines-covered"]) if "lines-covered" in attrib else None,
                "lines_valid": int(attrib["lines-valid"]) if "lines-valid" in attrib else None,
            }
        except (OSError, ET.ParseError, ValueError) as exc:
            return {
                "available": False,
                "reason": f"unreadable {COVERAGE_XML_REL}: {exc}",
                "path": str(COVERAGE_XML_REL),
            }
    if data_path.exists():
        mtime = datetime.fromtimestamp(data_path.stat().st_mtime, tz=timezone.utc)
        return {
            "available": True,
            "path": str(COVERAGE_DATA_REL),
            "note": "binary .coverage present — run coverage report/xml for percent",
            "mtime": mtime.isoformat(),
        }
    return {
        "available": False,
        "reason": "no coverage.xml or .coverage found — run devtools coverage-gate",
        "path": str(COVERAGE_XML_REL),
    }


# ──────────────────────────────────────────────────────────────────────
# Section: benchmark / SLO catalog
# ──────────────────────────────────────────────────────────────────────


def _benchmark_slo(root: Path, *, now: datetime) -> dict[str, Any]:
    out: dict[str, Any] = {}
    campaigns_dir = root / BENCHMARK_CAMPAIGNS_REL
    if campaigns_dir.exists():
        runs: list[dict[str, Any]] = []
        for json_file in sorted(campaigns_dir.glob("**/*.json")):
            mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
            runs.append(
                {
                    "name": json_file.stem,
                    "path": str(json_file.relative_to(root)),
                    "mtime": mtime.isoformat(),
                    "age_days": (now - mtime).days,
                }
            )
        out["benchmark_campaigns"] = {
            "available": True,
            "path": str(BENCHMARK_CAMPAIGNS_REL),
            "total_runs": len(runs),
            "runs": runs,
        }
    else:
        out["benchmark_campaigns"] = {
            "available": False,
            "reason": f"missing {BENCHMARK_CAMPAIGNS_REL} — run devtools benchmark-campaign run <name>",
            "path": str(BENCHMARK_CAMPAIGNS_REL),
        }
    slo_path = root / SLO_CATALOG_REL
    if slo_path.exists():
        try:
            from devtools.verify_slos import _parse_slo_catalog

            surfaces = _parse_slo_catalog(slo_path.read_text())
            required = [
                name
                for name, cfg in surfaces.items()
                if isinstance(cfg.get("gate"), str) and cfg.get("gate") == "required"
            ]
            informational = [
                name
                for name, cfg in surfaces.items()
                if isinstance(cfg.get("gate"), str) and cfg.get("gate") == "informational"
            ]
            out["slo_catalog"] = {
                "available": True,
                "path": str(SLO_CATALOG_REL),
                "total_surfaces": len(surfaces),
                "required_surfaces": sorted(required),
                "informational_surfaces": sorted(informational),
            }
        except (OSError, ValueError) as exc:
            out["slo_catalog"] = {
                "available": False,
                "reason": f"unreadable {SLO_CATALOG_REL}: {exc}",
                "path": str(SLO_CATALOG_REL),
            }
    else:
        out["slo_catalog"] = {
            "available": False,
            "reason": f"missing {SLO_CATALOG_REL}",
            "path": str(SLO_CATALOG_REL),
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Section: static gates (from verify history)
# ──────────────────────────────────────────────────────────────────────


_STATIC_GATE_NAMES: tuple[str, ...] = (
    "ruff format",
    "ruff check",
    "mypy",
    "render-all",
    "verify-topology",
    "verify-layering",
    "verify-schema-roundtrip",
    "verify-manifests",
    "verify-lane-assertions",
)


def _static_gates(root: Path, *, now: datetime) -> dict[str, Any]:
    history_path = root / VERIFY_HISTORY_REL
    last_result_path = root / LAST_VERIFY_RESULT_REL

    # Prefer last-verify-result.json (the most recent run) then walk back through
    # history to find the last status for each gate.
    last_steps: dict[str, dict[str, Any]] = {}
    last_result_mtime: str | None = None
    if last_result_path.exists():
        try:
            data = json.loads(last_result_path.read_text())
            result = data.get("result") if isinstance(data, dict) else None
            if isinstance(result, dict):
                for step in result.get("steps", []):
                    if isinstance(step, dict) and isinstance(step.get("name"), str):
                        last_steps[step["name"]] = step
                last_result_mtime = result.get("timestamp")
        except (OSError, json.JSONDecodeError):
            pass

    history_entries: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            with history_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        history_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    # Fill in any gates missing from last-verify-result with the most recent
    # appearance in history.
    if history_entries:
        for entry in reversed(history_entries):
            steps = entry.get("steps", [])
            for step in steps:
                if not isinstance(step, dict):
                    continue
                name = step.get("name")
                if isinstance(name, str) and name not in last_steps:
                    last_steps[name] = {**step, "_source": "history", "_run_timestamp": entry.get("timestamp")}

    gates: list[dict[str, Any]] = []
    for gate_name in _STATIC_GATE_NAMES:
        step = last_steps.get(gate_name)
        if step is None:
            gates.append({"name": gate_name, "available": False, "reason": "no run observed in cached history"})
            continue
        exit_code = step.get("exit", -1)
        gates.append(
            {
                "name": gate_name,
                "available": True,
                "status": "ok" if exit_code == 0 else "fail",
                "exit_code": exit_code,
                "duration_s": step.get("duration_s"),
                "last_run": step.get("_run_timestamp", last_result_mtime),
            }
        )
    failing = [g for g in gates if g.get("status") == "fail"]
    return {
        "available": last_result_path.exists() or history_path.exists(),
        "history_path": str(VERIFY_HISTORY_REL),
        "last_result_path": str(LAST_VERIFY_RESULT_REL),
        "total_gates_tracked": len(_STATIC_GATE_NAMES),
        "gates_with_status": sum(1 for g in gates if g.get("available")),
        "failing": [g["name"] for g in failing],
        "gates": gates,
        "history_runs": len(history_entries),
    }


# ──────────────────────────────────────────────────────────────────────
# Section: witnesses
# ──────────────────────────────────────────────────────────────────────


def _witnesses(root: Path) -> dict[str, Any]:
    committed = root / WITNESSES_COMMITTED_REL
    local = root / WITNESSES_LOCAL_REL
    committed_files = sorted(committed.glob("*.witness.json")) if committed.exists() else []
    local_files = sorted(local.glob("**/*.witness.json")) if local.exists() else []
    return {
        "committed": {
            "available": committed.exists(),
            "path": str(WITNESSES_COMMITTED_REL),
            "count": len(committed_files),
        },
        "local": {
            "available": local.exists(),
            "path": str(WITNESSES_LOCAL_REL),
            "count": len(local_files),
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Section: mutation campaigns
# ──────────────────────────────────────────────────────────────────────


def _mutation_campaigns(root: Path, *, now: datetime) -> dict[str, Any]:
    mut_dir = root / MUTATION_CAMPAIGNS_REL
    if not mut_dir.exists():
        return {
            "available": False,
            "reason": f"missing {MUTATION_CAMPAIGNS_REL} — run devtools mutmut-campaign run <name>",
            "path": str(MUTATION_CAMPAIGNS_REL),
        }
    campaigns: dict[str, dict[str, Any]] = {}
    for child in sorted(mut_dir.iterdir()):
        if not child.is_dir():
            continue
        latest_mtime: float | None = None
        files = 0
        for f in child.rglob("*"):
            if f.is_file():
                files += 1
                m = f.stat().st_mtime
                if latest_mtime is None or m > latest_mtime:
                    latest_mtime = m
        entry: dict[str, Any] = {"files": files}
        if latest_mtime is not None:
            mtime = datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
            entry["latest_mtime"] = mtime.isoformat()
            entry["age_days"] = (now - mtime).days
        campaigns[child.name] = entry
    return {
        "available": True,
        "path": str(MUTATION_CAMPAIGNS_REL),
        "total_campaigns": len(campaigns),
        "campaigns": campaigns,
    }


# ──────────────────────────────────────────────────────────────────────
# Dashboard assembly
# ──────────────────────────────────────────────────────────────────────


def build_dashboard(root: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Build the complete evidence-dashboard payload from real artifacts."""
    when = now or datetime.now(timezone.utc)
    benchmark_section = _benchmark_slo(root, now=when)
    return {
        "schema_version": 1,
        "generated_at": when.isoformat(),
        "root": str(root),
        "pytest": _pytest_health(root, now=when),
        "coverage": _coverage(root),
        "benchmark_campaigns": benchmark_section["benchmark_campaigns"],
        "slo_catalog": benchmark_section["slo_catalog"],
        "static_gates": _static_gates(root, now=when),
        "witnesses": _witnesses(root),
        "mutation_campaigns": _mutation_campaigns(root, now=when),
    }


# ──────────────────────────────────────────────────────────────────────
# Change traceability
# ──────────────────────────────────────────────────────────────────────


def build_trace(
    root: Path,
    *,
    base_ref: str = "origin/master",
    head_ref: str = "HEAD",
    changed_paths: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a change → claim → evidence → gate trace (verification_impact removed)."""
    when = now or datetime.now(timezone.utc)
    # verification_impact module was deleted as part of #1737.
    # Return an empty trace envelope with a deprecation note.
    return {
        "trace": {
            "available": False,
            "reason": "verification_impact module removed (#1737)",
            "generated_at": when.isoformat(),
        }
    }


# ──────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────


def _availability_marker(available: bool) -> str:
    return "yes" if available else "no"


def render_markdown(dashboard: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Evidence Dashboard ({dashboard.get('generated_at', 'unknown')})")
    lines.append("")

    pytest_data = dashboard["pytest"]
    lines.append("## Pytest health")
    if pytest_data.get("available"):
        counts = pytest_data.get("counts", {})
        stale = " (stale)" if pytest_data.get("stale") else ""
        lines.append(
            f"- status: **{pytest_data.get('status')}** — passed={counts.get('passed', 0)}, "
            f"failed={counts.get('failed', 0)}, xfailed={counts.get('xfailed', 0)}, "
            f"skipped={counts.get('skipped', 0)}"
        )
        lines.append(
            f"- duration: {pytest_data.get('duration_s', '?')}s, last_run: {pytest_data.get('last_run')}{stale}"
        )
    else:
        lines.append(f"- unavailable: {pytest_data.get('reason')}")
    lines.append("")

    coverage = dashboard["coverage"]
    lines.append("## Coverage")
    if coverage.get("available"):
        if "percent" in coverage:
            lines.append(f"- line coverage: {coverage['percent']}% (from {coverage['path']})")
        else:
            lines.append(f"- {coverage.get('note', 'present')} ({coverage['path']})")
    else:
        lines.append(f"- unavailable: {coverage.get('reason')}")
    lines.append("")

    bench = dashboard["benchmark_campaigns"]
    lines.append("## Benchmark campaigns")
    if bench.get("available"):
        lines.append(f"- total_runs: {bench.get('total_runs')}")
    else:
        lines.append(f"- unavailable: {bench.get('reason')}")
    lines.append("")

    slo = dashboard["slo_catalog"]
    lines.append("## SLO catalog")
    if slo.get("available"):
        lines.append(
            f"- surfaces: {slo.get('total_surfaces')} "
            f"(required={len(slo.get('required_surfaces', []))}, "
            f"informational={len(slo.get('informational_surfaces', []))})"
        )
    else:
        lines.append(f"- unavailable: {slo.get('reason')}")
    lines.append("")

    gates = dashboard["static_gates"]
    lines.append("## Static gates")
    if gates.get("available"):
        lines.append(
            f"- gates with cached status: {gates['gates_with_status']}/{gates['total_gates_tracked']}, "
            f"failing: {len(gates['failing'])}"
        )
        if gates["failing"]:
            lines.append(f"  - failing: {', '.join(gates['failing'])}")
    else:
        lines.append("- unavailable: no verify history found")
    lines.append("")

    witnesses = dashboard["witnesses"]
    lines.append("## Witnesses")
    lines.append(
        f"- committed: {witnesses['committed']['count']} ({_availability_marker(witnesses['committed']['available'])}), "
        f"local: {witnesses['local']['count']} ({_availability_marker(witnesses['local']['available'])})"
    )
    lines.append("")

    mut = dashboard["mutation_campaigns"]
    lines.append("## Mutation campaigns")
    if mut.get("available"):
        lines.append(f"- total_campaigns: {mut.get('total_campaigns')}")
    else:
        lines.append(f"- unavailable: {mut.get('reason')}")
    lines.append("")

    return "\n".join(lines)


def render_trace_markdown(trace: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Change Trace {trace['base_ref']}..{trace['head_ref']}")
    lines.append("")
    lines.append(f"- changed paths: {trace['changed_path_count']}")
    lines.append(f"- required PR gates: {len(trace.get('required_gates', []))}")
    lines.append("")
    if not trace["changes"]:
        lines.append("(no recognised change subjects)")
        return "\n".join(lines)
    lines.append("## Changes")
    for row in trace["changes"]:
        lines.append(f"### {row['path']}")
        lines.append(f"- kind: {row['kind']}")
        if row["reason"]:
            lines.append(f"- reason: {row['reason']}")
        if row["subject_ids"]:
            lines.append(f"- subjects: {', '.join(row['subject_ids'])}")
        if row["surface_names"]:
            lines.append(f"- surfaces: {', '.join(row['surface_names'])}")
        if row["checks"]:
            lines.append("- recommended checks:")
            for check in row["checks"][:5]:
                cmd = check.get("command")
                if isinstance(cmd, list):
                    lines.append(f"  - `{' '.join(str(c) for c in cmd)}` — {check.get('reason', '')}")
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _emit_dashboard(args: argparse.Namespace) -> int:
    dashboard = build_dashboard(ROOT)
    if args.json or not args.markdown:
        json.dump(dashboard, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    if args.markdown:
        sys.stdout.write(render_markdown(dashboard))
        sys.stdout.write("\n")
    return 0


def _emit_trace(args: argparse.Namespace) -> int:
    paths: list[str] | None = list(args.path) if args.path else None
    trace = build_trace(ROOT, base_ref=args.base, head_ref=args.head, changed_paths=paths)
    if args.json or not args.markdown:
        json.dump(trace, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    if args.markdown:
        sys.stdout.write(render_trace_markdown(trace))
        sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON output (default).")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (combinable with --json).")
    sub = parser.add_subparsers(dest="cmd")

    trace = sub.add_parser("trace", help="Change → claim → evidence → gate trace for the changed paths.")
    trace.add_argument("--base", default="origin/master", help="Base git ref (default: origin/master).")
    trace.add_argument("--head", default="HEAD", help="Head git ref (default: HEAD).")
    trace.add_argument(
        "--path",
        action="append",
        default=[],
        help="Explicit changed path. Repeat to bypass git diff discovery.",
    )
    trace.add_argument("--json", action="store_true", help="Emit JSON output (default).")
    trace.add_argument("--markdown", action="store_true", help="Emit Markdown output (combinable with --json).")

    args = parser.parse_args(argv)
    if args.cmd == "trace":
        return _emit_trace(args)
    return _emit_dashboard(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
