"""Human-readable showcase report rendering."""

from __future__ import annotations

from polylogue.showcase.exercises import GROUPS
from polylogue.showcase.runner import ShowcaseResult


def generate_summary(result: ShowcaseResult) -> str:
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


def generate_cookbook(result: ShowcaseResult) -> str:
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
    def _status_marker(result_entry) -> str:
        if result_entry.skipped:
            return "[SKIP]"
        return "[PASS]" if result_entry.passed else "[FAIL]"

    def _truncated_output(output: str, *, max_lines: int = 10) -> list[str]:
        output_lines = output.rstrip().splitlines()
        if len(output_lines) <= max_lines:
            return output_lines
        remaining = len(output_lines) - max_lines
        return [*output_lines[:max_lines], f"... ({remaining} more lines)"]

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

        lines.append(
            f"- {_status_marker(result_entry)} `{result_entry.exercise.name}`: {result_entry.exercise.description}"
        )
        if result_entry.error:
            lines.append(f"  Error: {result_entry.error}")
        if result_entry.skip_reason:
            lines.append(f"  Reason: {result_entry.skip_reason}")
        if result_entry.output and not result_entry.passed and not result_entry.skipped:
            for output_line in _truncated_output(result_entry.output):
                lines.append(f"  {output_line}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["generate_cookbook", "generate_showcase_markdown", "generate_summary"]
