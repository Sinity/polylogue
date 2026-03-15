"""Report generation for showcase results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

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


def generate_qa_session(result: ShowcaseResult) -> dict:
    """Generate a structured QA session record from a showcase result.

    Returns a dict suitable for writing to an audit trail directory.
    Fields are stable across runs so diffs are meaningful.
    """
    exercises: list[dict] = []
    for r in result.results:
        entry: dict = {
            "name": r.exercise.name,
            "group": r.exercise.group,
            "tier": r.exercise.tier,
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

    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(result.results),
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "total_duration_ms": round(result.total_duration_ms, 1),
        },
        "group_counts": result.group_counts(),
        "exercises": exercises,
    }


def write_qa_session(result: ShowcaseResult, audit_dir: Path) -> Path:
    """Write a QA session record to audit_dir and return the written path."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = audit_dir / f"showcase-{ts}.json"
    session = generate_qa_session(result)
    out_path.write_text(json.dumps(session, indent=2))
    return out_path


def generate_qa_markdown(result: ShowcaseResult, *, git_sha: str | None = None) -> str:
    """Generate a stable, diffable Markdown QA session report.

    The report contains no run-specific timestamps in the body, making it
    suitable for committing to the repo and diffing across runs.
    """
    lines: list[str] = []
    lines.append("# Showcase QA Session")
    lines.append("")
    if git_sha:
        lines.append(f"**Git SHA**: `{git_sha}`")
        lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | ---: |")
    lines.append(f"| Total | {len(result.results)} |")
    lines.append(f"| Passed | {result.passed} |")
    lines.append(f"| Failed | {result.failed} |")
    lines.append(f"| Skipped | {result.skipped} |")
    lines.append("")

    # Group breakdown
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

    # Exercises grouped by group name
    lines.append("## Exercises")
    lines.append("")

    current_group = ""
    for r in result.results:
        if r.exercise.group != current_group:
            current_group = r.exercise.group
            lines.append(f"### {current_group}")
            lines.append("")

        if r.skipped:
            status = "SKIP"
        elif r.passed:
            status = "PASS"
        else:
            status = "FAIL"

        lines.append(f"- **[{status}]** `{r.exercise.name}` — {r.exercise.description}")

        # Include truncated output for failures
        if not r.passed and not r.skipped and r.error:
            lines.append(f"  - Error: {r.error}")
            if r.output:
                output_lines = r.output.strip().splitlines()
                snippet = output_lines[:10]
                lines.append("  ```")
                for ol in snippet:
                    lines.append(f"  {ol}")
                if len(output_lines) > 10:
                    lines.append(f"  ... ({len(output_lines) - 10} more lines)")
                lines.append("  ```")

    lines.append("")
    return "\n".join(lines)


def generate_manifest(
    result: ShowcaseResult,
    *,
    include_hashes: bool = True,
) -> dict:
    """Produce a manifest with file hashes for all generated artifacts.

    Scans the output directory (if set) for files and records their
    relative paths and optional SHA-256 hashes.
    """
    entries: list[dict] = []

    if result.output_dir and result.output_dir.exists():
        for path in sorted(result.output_dir.rglob("*")):
            if not path.is_file():
                continue
            entry: dict = {
                "relative_path": str(path.relative_to(result.output_dir)),
                "size_bytes": path.stat().st_size,
            }
            if include_hashes:
                digest = hashlib.sha256()
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        digest.update(chunk)
                entry["sha256"] = digest.hexdigest()
            entries.append(entry)

    return {
        "schema_version": 1,
        "entry_count": len(entries),
        "entries": entries,
    }


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

    # Stable QA markdown
    qa_md = generate_qa_markdown(result)
    (out / "showcase-session.md").write_text(qa_md)

    # Artifact manifest (written last so it captures all other files)
    manifest = generate_manifest(result, include_hashes=True)
    (out / "showcase-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
