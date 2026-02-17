"""Report generation for showcase results."""

from __future__ import annotations

import json

from polylogue.showcase.exercises import GROUPS
from polylogue.showcase.runner import ShowcaseResult


def generate_summary(result: ShowcaseResult) -> str:
    """Generate a human-readable summary table."""
    lines: list[str] = []
    total_ms = result.total_duration_ms
    total_s = total_ms / 1000

    total = len(result.results)
    lines.append(f"Showcase Results ({total} exercises, {total_s:.1f}s)")
    lines.append("")
    lines.append(f"  {'Group':<20s}  {'Pass':>4s}  {'Fail':>4s}  {'Skip':>4s}")
    lines.append(f"  {'─' * 20}  {'─' * 4}  {'─' * 4}  {'─' * 4}")

    group_counts = result.group_counts()
    for group in GROUPS:
        counts = group_counts.get(group, {"pass": 0, "fail": 0, "skip": 0})
        lines.append(
            f"  {group:<20s}  {counts['pass']:>4d}  {counts['fail']:>4d}  {counts['skip']:>4d}"
        )

    lines.append(f"  {'─' * 20}  {'─' * 4}  {'─' * 4}  {'─' * 4}")
    lines.append(
        f"  {'TOTAL':<20s}  {result.passed:>4d}  {result.failed:>4d}  {result.skipped:>4d}"
    )
    lines.append("")

    if result.output_dir:
        lines.append(f"Output: {result.output_dir}")
        cookbook_path = result.output_dir / "showcase-cookbook.md"
        if cookbook_path.exists():
            lines.append(f"Cookbook: {cookbook_path}")

    # List failures
    failures = [r for r in result.results if not r.passed and not r.skipped]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for f in failures:
            lines.append(f"  ✗ {f.exercise.name}: {f.error}")

    return "\n".join(lines)


def generate_json_report(result: ShowcaseResult) -> str:
    """Generate a machine-readable JSON report."""
    exercises: list[dict] = []
    for r in result.results:
        entry: dict = {
            "name": r.exercise.name,
            "group": r.exercise.group,
            "description": r.exercise.description,
            "passed": r.passed,
            "exit_code": r.exit_code,
            "duration_ms": round(r.duration_ms, 1),
        }
        if r.skipped:
            entry["skipped"] = True
            entry["skip_reason"] = r.skip_reason
        if r.error:
            entry["error"] = r.error
        exercises.append(entry)

    report = {
        "total": len(result.results),
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "total_duration_ms": round(result.total_duration_ms, 1),
        "exercises": exercises,
    }
    return json.dumps(report, indent=2)


def generate_cookbook(result: ShowcaseResult) -> str:
    """Generate a Markdown cookbook from exercise outputs."""
    lines: list[str] = []
    lines.append("# Polylogue CLI Cookbook")
    lines.append("")
    lines.append("Auto-generated reference of CLI commands and their outputs.")
    lines.append("")

    current_group = ""
    group_titles = {
        "structural": "Structural (Help & Version)",
        "sources": "Sources & Completions",
        "pipeline": "Pipeline Execution",
        "query-read": "Query Mode (Read-Only)",
        "query-write": "Query Mode (Write Operations)",
        "subcommands": "Subcommands",
        "advanced": "Advanced Queries",
    }

    for r in result.results:
        if r.skipped:
            continue

        # Group heading
        if r.exercise.group != current_group:
            current_group = r.exercise.group
            title = group_titles.get(current_group, current_group.title())
            lines.append(f"## {title}")
            lines.append("")

        # Exercise
        args_str = " ".join(r.exercise.args) if r.exercise.args else "(default stats)"
        lines.append(f"### {r.exercise.description}")
        lines.append("")
        lines.append("```console")
        lines.append(f"$ polylogue {args_str}")

        # Truncate very long output
        output = r.output.rstrip()
        output_lines = output.splitlines()
        if len(output_lines) > 40:
            lines.extend(output_lines[:35])
            lines.append(f"... ({len(output_lines) - 35} more lines)")
        else:
            lines.append(output)

        lines.append("```")
        lines.append("")

        if not r.passed:
            lines.append(f"> **Note**: This exercise failed: {r.error}")
            lines.append("")

    return "\n".join(lines)


def save_reports(result: ShowcaseResult) -> None:
    """Save all report artifacts to the output directory."""
    if not result.output_dir:
        return

    out = result.output_dir

    # Summary text
    summary = generate_summary(result)
    (out / "showcase-summary.txt").write_text(summary)

    # JSON report
    json_report = generate_json_report(result)
    (out / "showcase-report.json").write_text(json_report)

    # Markdown cookbook
    cookbook = generate_cookbook(result)
    (out / "showcase-cookbook.md").write_text(cookbook)
