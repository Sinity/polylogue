"""Report generation for showcase results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.exercises import GROUPS
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult

if TYPE_CHECKING:
    from polylogue.showcase.qa_runner import QAResult


def _serialize_showcase_exercise(
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


def _showcase_summary_payload(result: ShowcaseResult) -> dict[str, int | float]:
    return {
        "total": len(result.results),
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "total_duration_ms": round(result.total_duration_ms, 1),
    }


def _serialize_invariant_result(result: InvariantResult) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "invariant": result.invariant_name,
        "exercise": result.exercise_name,
        "status": result.status.value,
    }
    if result.error:
        entry["error"] = result.error
    return entry


def _summarize_invariants(results: list[InvariantResult]) -> dict[str, int]:
    return {
        "passed": sum(1 for result in results if result.status is OutcomeStatus.OK),
        "failed": sum(1 for result in results if result.status is OutcomeStatus.ERROR),
        "skipped": sum(1 for result in results if result.status is OutcomeStatus.SKIP),
    }


def _status_label(status: OutcomeStatus) -> str:
    return {
        OutcomeStatus.OK: "PASS",
        OutcomeStatus.WARNING: "WARN",
        OutcomeStatus.ERROR: "FAIL",
        OutcomeStatus.SKIP: "SKIPPED",
    }[status]


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
    report = {
        **_showcase_summary_payload(result),
        "exercises": [_serialize_showcase_exercise(r) for r in result.results],
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


def generate_showcase_session(result: ShowcaseResult) -> dict[str, Any]:
    """Generate a structured showcase session record.

    Returns a dict suitable for writing to an audit trail directory.
    Fields are stable across runs so diffs are meaningful.
    """
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": _showcase_summary_payload(result),
        "group_counts": result.group_counts(),
        "exercises": [
            _serialize_showcase_exercise(r, include_description=False, include_tier=True)
            for r in result.results
        ],
    }


def write_showcase_session(result: ShowcaseResult, audit_dir: Path) -> Path:
    """Write a showcase session record to audit_dir and return the written path."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = audit_dir / f"showcase-{ts}.json"
    session = generate_showcase_session(result)
    out_path.write_text(json.dumps(session, indent=2))
    return out_path


def generate_showcase_markdown(result: ShowcaseResult, *, git_sha: str | None = None) -> str:
    """Generate a stable, diffable Markdown showcase session report.

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
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
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


def generate_qa_session(result: QAResult) -> dict[str, Any]:
    """Generate a structured full QA session record."""
    showcase_session = (
        generate_showcase_session(result.showcase_result)
        if result.showcase_result is not None
        else None
    )

    audit_payload: dict[str, Any] = {
        "status": result.audit_status.value,
        "skipped": result.audit_skipped,
    }
    if result.audit_report is not None:
        audit_payload["report"] = result.audit_report.to_json()
    if result.audit_error is not None:
        audit_payload["error"] = result.audit_error

    invariant_checks = [
        _serialize_invariant_result(invariant_result)
        for invariant_result in result.invariant_results
    ]
    invariant_summary = _summarize_invariants(result.invariant_results)

    showcase_payload: dict[str, Any] = {
        "status": result.showcase_status.value,
        "skipped": result.exercises_skipped,
        "summary": showcase_session["summary"] if showcase_session else None,
        "group_counts": showcase_session["group_counts"] if showcase_session else {},
        "exercises": showcase_session["exercises"] if showcase_session else [],
    }

    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit": audit_payload,
        "showcase": showcase_payload,
        "invariants": {
            "status": result.invariant_status.value,
            "skipped": result.invariants_skipped or result.showcase_result is None,
            "summary": invariant_summary,
            "checks": invariant_checks,
        },
        "overall_status": result.overall_status.value,
        "overall_passed": result.all_passed,
        "report_dir": str(result.report_dir) if result.report_dir else None,
    }


def generate_qa_summary(result: QAResult) -> str:
    """Generate a human-readable summary for a full QA run."""
    session = generate_qa_session(result)
    lines: list[str] = []

    lines.append(f"Schema Audit: {_status_label(result.audit_status)}")
    if result.audit_status is OutcomeStatus.ERROR:
        if result.audit_error:
            lines.append(f"  Error: {result.audit_error}")
        elif result.audit_report is not None:
            summary = result.audit_report.to_json()["summary"]
            lines.append(
                f"  Checks: {summary['passed']} pass, {summary['warned']} warn, {summary['failed']} fail"
            )
        lines.append("Overall: FAIL")
        if result.report_dir:
            lines.append(f"Reports: {result.report_dir}")
        return "\n".join(lines)

    showcase_summary = session["showcase"]["summary"]
    if result.exercises_skipped:
        lines.append("Exercises: SKIPPED")
    elif showcase_summary is not None:
        lines.append(
            "Exercises: "
            f"{showcase_summary['passed']}/{showcase_summary['total']} passed, "
            f"{showcase_summary['failed']} failed, {showcase_summary['skipped']} skipped "
            f"({showcase_summary['total_duration_ms'] / 1000:.1f}s)"
        )

    invariant_summary = session["invariants"]["summary"]
    if session["invariants"]["skipped"]:
        lines.append("Invariants: SKIPPED")
    else:
        lines.append(
            "Invariants: "
            f"{invariant_summary['passed']} pass, "
            f"{invariant_summary['failed']} fail, "
            f"{invariant_summary['skipped']} skip"
        )

    lines.append("")
    lines.append(f"Overall: {_status_label(result.overall_status)}")
    if result.report_dir:
        lines.append(f"Reports: {result.report_dir}")
    return "\n".join(lines)


def generate_qa_markdown(result: QAResult, *, git_sha: str | None = None) -> str:
    """Generate a stable, diffable Markdown report for a full QA run."""
    session = generate_qa_session(result)
    lines: list[str] = ["# QA Session", ""]
    if git_sha:
        lines.append(f"**Git SHA**: `{git_sha}`")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Status |")
    lines.append("| --- | --- |")
    lines.append(f"| Schema Audit | {_status_label(result.audit_status)} |")
    lines.append(f"| Exercises | {_status_label(result.showcase_status)} |")
    lines.append(f"| Invariants | {_status_label(result.invariant_status)} |")
    lines.append(f"| Overall | {_status_label(result.overall_status)} |")
    lines.append("")

    if result.audit_report is not None:
        audit_summary = result.audit_report.to_json()["summary"]
        lines.append("## Schema Audit")
        lines.append("")
        lines.append(
            f"- Passed: {audit_summary['passed']}"
        )
        lines.append(
            f"- Warned: {audit_summary['warned']}"
        )
        lines.append(
            f"- Failed: {audit_summary['failed']}"
        )
        lines.append("")
    elif result.audit_error:
        lines.append("## Schema Audit")
        lines.append("")
        lines.append(f"- Error: {result.audit_error}")
        lines.append("")

    showcase_summary = session["showcase"]["summary"]
    if showcase_summary is not None:
        lines.append("## Exercises")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Total | {showcase_summary['total']} |")
        lines.append(f"| Passed | {showcase_summary['passed']} |")
        lines.append(f"| Failed | {showcase_summary['failed']} |")
        lines.append(f"| Skipped | {showcase_summary['skipped']} |")
        lines.append("")
        lines.append("### Results by Group")
        lines.append("")
        lines.append("| Group | Pass | Fail | Skip |")
        lines.append("| --- | ---: | ---: | ---: |")
        for group in GROUPS:
            counts = session["showcase"]["group_counts"].get(group, {"pass": 0, "fail": 0, "skip": 0})
            lines.append(
                f"| {group} | {counts['pass']} | {counts['fail']} | {counts['skip']} |"
            )
        lines.append("")

    invariant_summary = session["invariants"]["summary"]
    if not session["invariants"]["skipped"]:
        lines.append("## Invariants")
        lines.append("")
        lines.append(f"- Passed: {invariant_summary['passed']}")
        lines.append(f"- Failed: {invariant_summary['failed']}")
        lines.append(f"- Skipped: {invariant_summary['skipped']}")
        failures = [
            check for check in session["invariants"]["checks"] if check["status"] == OutcomeStatus.ERROR.value
        ]
        if failures:
            lines.append("")
            lines.append("### Failures")
            lines.append("")
            for failure in failures:
                lines.append(
                    f"- `{failure['invariant']}` @ `{failure['exercise']}`: {failure.get('error', '')}"
                )
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
    qa_md = generate_showcase_markdown(result)
    (out / "showcase-session.md").write_text(qa_md)

    # Artifact manifest (written last so it captures all other files)
    manifest = generate_manifest(result, include_hashes=True)
    (out / "showcase-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
