"""Health/runtime sections for plain check output."""

from __future__ import annotations

from polylogue.cli.check_rendering_plain_support import status_icon
from polylogue.cli.check_workflow import CheckCommandOptions, CheckCommandResult
from polylogue.cli.types import AppEnv
from polylogue.health import VerifyStatus


def build_health_lines(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> list[str]:
    lines: list[str] = []
    for check in result.report.checks:
        lines.append(f"{status_icon(check.status, plain=env.ui.plain)} {check.name}: {check.detail}")
        if check.breakdown and (options.verbose or check.status in (VerifyStatus.WARNING, VerifyStatus.ERROR)):
            for provider, count in sorted(check.breakdown.items(), key=lambda item: -item[1]):
                lines.append(f"    {provider}: {count:,}")

    summary = result.report.summary
    provenance = result.report.provenance
    source_val = getattr(provenance.source, "value", provenance.source) if hasattr(provenance, "source") else "live"
    lines.extend(
        [
            "",
            (
                f"Summary: {summary.get('ok', 0)} ok, {summary.get('warning', 0)} warnings, "
                f"{summary.get('error', 0)} errors (source={source_val})"
            ),
        ]
    )
    return lines


def append_derived_model_lines(lines: list[str], result: CheckCommandResult) -> None:
    if not result.report.derived_models:
        return
    lines.extend(["", "Derived Models:"])
    for name, status in sorted(result.report.derived_models.items()):
        state = "ready" if status.ready else "pending"
        lines.append(
            f"  {name}: {state}; docs {status.materialized_documents:,}/{status.source_documents:,}; "
            f"rows {status.materialized_rows:,}/{status.source_rows:,}; "
            f"pending_docs={status.pending_documents:,}; pending_rows={status.pending_rows:,}; "
            f"stale={status.stale_rows:,}; orphan={status.orphan_rows:,}; "
            f"missing_provenance={status.missing_provenance_rows:,}"
        )


def append_runtime_lines(lines: list[str], result: CheckCommandResult, *, plain: bool) -> None:
    if result.runtime_report is None:
        return
    lines.extend(["", "Runtime Environment:"])
    for check in result.runtime_report.checks:
        lines.append(
            f"  {status_icon(check.status, plain=plain)} {check.name}: {check.detail}"
        )
    rt_summary = result.runtime_report.summary
    lines.append(
        f"  Runtime: {rt_summary.get('ok', 0)} ok, {rt_summary.get('warning', 0)} warnings, "
        f"{rt_summary.get('error', 0)} errors"
    )


__all__ = ["append_derived_model_lines", "append_runtime_lines", "build_health_lines"]
