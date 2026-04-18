"""Post-run QA output, capture, and archival helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click

from polylogue.cli.qa_capture import run_vhs_capture
from polylogue.cli.qa_requests import QACaptureMode, QAFinalizationPlan
from polylogue.cli.qa_snapshot import execute_snapshot_plan

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.showcase.qa_runner_models import QAResult


def render_qa_session(result: QAResult) -> dict[str, object]:
    from polylogue.showcase.qa_report import generate_qa_session

    return cast(dict[str, object], generate_qa_session(result))


def render_qa_summary(result: QAResult) -> str:
    from polylogue.showcase.qa_runner import format_qa_summary

    return format_qa_summary(result)


def finalize_qa_run(
    result: QAResult,
    *,
    plan: QAFinalizationPlan,
    archive_root: Path,
    env: AppEnv,
) -> None:
    """Emit all post-run QA side effects from the normalized finalization plan."""
    if (
        plan.capture_mode is QACaptureMode.VHS
        and result.showcase_result is not None
        and result.showcase_result.output_dir is not None
    ):
        run_vhs_capture(env, result.showcase_result, plan.json_output)

    if plan.json_output:
        click.echo(json.dumps(render_qa_session(result), indent=2))
    else:
        env.ui.console.print(render_qa_summary(result))

    if plan.snapshot_plan and not plan.snapshot_plan.skips_qa and result.report_dir:
        execute_snapshot_plan(
            plan.snapshot_plan,
            fallback_source_dir=result.report_dir,
            output_root=archive_root / "qa" / "snapshots",
            json_output=plan.json_output,
            env=env,
        )


__all__ = ["finalize_qa_run"]
