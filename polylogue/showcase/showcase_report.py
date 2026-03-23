"""Showcase-specific report rendering and session payloads."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.showcase.exercises import GROUPS
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult


def serialize_showcase_exercise(
    result: ExerciseResult,
    *,
    include_description: bool = True,
    include_tier: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": result.exercise.name,
        "group": result.exercise.group,
        "passed": result.passed,
        "exit_code": result.exit_code,
        "duration_ms": round(result.duration_ms, 1),
    }
    if include_description:
        entry["description"] = result.exercise.description
    if include_tier:
        entry["tier"] = result.exercise.tier
    if result.skipped:
        entry["skipped"] = True
        entry["skip_reason"] = result.skip_reason
    if result.error:
        entry["error"] = result.error
    return entry


def showcase_summary_payload(result: ShowcaseResult) -> dict[str, int | float]:
    return {
        "total": len(result.results),
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "total_duration_ms": round(result.total_duration_ms, 1),
    }


def build_showcase_session_payload(
    result: ShowcaseResult,
    *,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": timestamp,
        "summary": showcase_summary_payload(result),
        "group_counts": result.group_counts(),
        "exercises": [
            serialize_showcase_exercise(
                report,
                include_description=False,
                include_tier=True,
            )
            for report in result.results
        ],
    }


def generate_showcase_session(result: ShowcaseResult) -> dict[str, Any]:
    """Generate a structured showcase session record."""
    return build_showcase_session_payload(
        result,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def write_showcase_session(result: ShowcaseResult, audit_dir: Path) -> Path:
    """Write a showcase session record to ``audit_dir`` and return the path."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = audit_dir / f"showcase-{ts}.json"
    out_path.write_text(json.dumps(generate_showcase_session(result), indent=2))
    return out_path


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

    failures = [report for report in result.results if not report.passed and not report.skipped]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for failure in failures:
            lines.append(f"  ✗ {failure.exercise.name}: {failure.error}")

    return "\n".join(lines)


def generate_json_report(result: ShowcaseResult) -> str:
    """Generate a machine-readable JSON report."""
    report = {
        **showcase_summary_payload(result),
        "exercises": [serialize_showcase_exercise(entry) for entry in result.results],
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

    for result_entry in result.results:
        if result_entry.skipped:
            continue

        if result_entry.exercise.group != current_group:
            current_group = result_entry.exercise.group
            title = group_titles.get(current_group, current_group.title())
            lines.append(f"## {title}")
            lines.append("")

        args_str = (
            " ".join(result_entry.exercise.args)
            if result_entry.exercise.args
            else "(default stats)"
        )
        lines.append(f"### {result_entry.exercise.description}")
        lines.append("")
        lines.append("```console")
        lines.append(f"$ polylogue {args_str}")

        output = result_entry.output.rstrip()
        output_lines = output.splitlines()
        if len(output_lines) > 40:
            lines.extend(output_lines[:35])
            lines.append(f"... ({len(output_lines) - 35} more lines)")
        else:
            lines.append(output)

        lines.append("```")
        lines.append("")

        if not result_entry.passed:
            lines.append(f"> **Note**: This exercise failed: {result_entry.error}")
            lines.append("")

    return "\n".join(lines)


def generate_showcase_markdown(
    result: ShowcaseResult,
    *,
    git_sha: str | None = None,
) -> str:
    """Generate a stable, diffable Markdown showcase session report."""
    lines: list[str] = []
    lines.append("# Showcase QA Session")
    lines.append("")
    if git_sha:
        lines.append(f"**Git SHA**: `{git_sha}`")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Total | {len(result.results)} |")
    lines.append(f"| Passed | {result.passed} |")
    lines.append(f"| Failed | {result.failed} |")
    lines.append(f"| Skipped | {result.skipped} |")
    lines.append("")

    lines.append("## Results by Group")
    lines.append("")
    lines.append("| Group | Pass | Fail | Skip |")
    lines.append("| --- | ---: | ---: | ---: |")

    group_counts = result.group_counts()
    for group in GROUPS:
        counts = group_counts.get(group, {"pass": 0, "fail": 0, "skip": 0})
        lines.append(
            f"| {group} | {counts['pass']} | {counts['fail']} | {counts['skip']} |"
        )
    lines.append("")

    lines.append("## Exercises")
    lines.append("")

    current_group = ""
    for result_entry in result.results:
        if result_entry.exercise.group != current_group:
            current_group = result_entry.exercise.group
            lines.append(f"### {current_group}")
            lines.append("")

        if result_entry.skipped:
            status = "SKIP"
        elif result_entry.passed:
            status = "PASS"
        else:
            status = "FAIL"

        lines.append(
            f"- **[{status}]** `{result_entry.exercise.name}` — {result_entry.exercise.description}"
        )

        if not result_entry.passed and not result_entry.skipped and result_entry.error:
            lines.append(f"  - Error: {result_entry.error}")
            if result_entry.output:
                output_lines = result_entry.output.strip().splitlines()
                snippet = output_lines[:10]
                lines.append("  ```")
                for output_line in snippet:
                    lines.append(f"  {output_line}")
                if len(output_lines) > 10:
                    lines.append(f"  ... ({len(output_lines) - 10} more lines)")
                lines.append("  ```")

    lines.append("")
    return "\n".join(lines)
