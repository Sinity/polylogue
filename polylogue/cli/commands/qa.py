"""Composable QA command: audit, exercises, invariants, and archival."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.cli.commands.generate import generate_command
from polylogue.cli.helpers import complete_configured_source_names, load_effective_config, resolve_sources
from polylogue.cli.qa_finalization import finalize_qa_run
from polylogue.cli.qa_requests import (
    QACaptureMode,
    build_qa_finalization_plan,
    build_qa_snapshot_plan,
)
from polylogue.cli.qa_snapshot import execute_snapshot_plan
from polylogue.cli.types import AppEnv
from polylogue.showcase.qa_runner_request import QAStage, build_qa_session_request

_STAGE_CHOICES = click.Choice([stage.value for stage in QAStage])
_CAPTURE_CHOICES = click.Choice([mode.value for mode in QACaptureMode])


@click.group("audit", invoke_without_command=True)
@click.option("--synthetic/--live", default=True, help="Data source: synthetic (default) or live real data")
@click.option(
    "--source",
    "source_names",
    multiple=True,
    shell_complete=complete_configured_source_names,
    help="Configured real source name (repeatable, implies --fresh)",
)
@click.option("--fresh", is_flag=True, default=None, help="Run in an isolated temp workspace (default for synthetic)")
@click.option("--workspace", type=click.Path(path_type=Path), help="Reuse a specific workspace directory")
@click.option(
    "--ingest", is_flag=True, default=None, help="Run ingestion pipeline (auto for synthetic and fresh-with-sources)"
)
@click.option("--schemas", "regenerate_schemas", is_flag=True, help="Regenerate schemas during pipeline")
@click.option("--only", "only_stage", type=_STAGE_CHOICES, default=None, help="Run only this stage")
@click.option("--skip", "skip_stages", multiple=True, type=_STAGE_CHOICES, help="Skip this stage (repeatable)")
@click.option("--tier", "tier_filter", type=int, default=None, help="Only run exercises at this tier (0/1/2)")
@click.option("--fail-fast", is_flag=True, help="Stop on first exercise failure")
@click.option(
    "--capture",
    type=_CAPTURE_CHOICES,
    default="none",
    show_default=True,
    help="Capture mode for exercises",
)
@click.option(
    "--report-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for QA artifacts (auto-generated if omitted)",
)
@click.option("--json", "json_output", is_flag=True, help="Machine-readable output")
@click.option("--verbose", is_flag=True, help="Print exercise outputs")
@click.option(
    "--snapshot",
    "snapshot_label",
    default=None,
    is_flag=False,
    flag_value="snapshot",
    help="Archive results after QA completes (optional label)",
)
@click.option(
    "--snapshot-from",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Archive an existing output directory (skips QA execution)",
)
@click.pass_obj
def qa_command(
    env: AppEnv,
    synthetic: bool,
    source_names: tuple[str, ...],
    fresh: bool | None,
    workspace: Path | None,
    ingest: bool | None,
    regenerate_schemas: bool,
    only_stage: str | None,
    skip_stages: tuple[str, ...],
    tier_filter: int | None,
    fail_fast: bool,
    capture: str,
    report_dir: Path | None,
    json_output: bool,
    verbose: bool,
    snapshot_label: str | None,
    snapshot_from: Path | None,
) -> None:
    """Run composable QA: schema audit, exercises, and invariant checks.

    \b
    By default, creates a fresh workspace with synthetic data and runs
    all stages: audit → exercises → invariants.

    \b
    Examples:
      polylogue audit                              # Full synthetic QA
      polylogue audit --live                       # Exercises against real data
      polylogue audit --source inbox               # Fresh workspace from inbox
      polylogue audit --only audit                 # Schema audit only
      polylogue audit --only exercises --tier 0    # Tier-0 smoke test
      polylogue audit --skip invariants            # Skip invariant checks
      polylogue audit --snapshot release-v3        # QA + archive results
      polylogue audit --snapshot-from ./qa_outputs # Archive existing directory
      polylogue audit generate                     # Generate synthetic data
      polylogue audit generate --seed              # Full demo environment
    """
    # If a subcommand (e.g. "generate") was invoked, let it run instead.
    ctx = click.get_current_context()
    if ctx.invoked_subcommand is not None:
        return
    capture_mode = QACaptureMode(capture.lower())
    snapshot_plan = build_qa_snapshot_plan(snapshot_label=snapshot_label, snapshot_from=snapshot_from)
    finalization_plan = build_qa_finalization_plan(
        capture_mode=capture_mode,
        json_output=json_output,
        snapshot_plan=snapshot_plan,
    )

    # --- Snapshot-from: archive only, no QA ---
    if snapshot_plan and snapshot_plan.skips_qa and snapshot_plan.source_dir is not None:
        config = load_effective_config(env)
        root = report_dir or (config.archive_root / "qa" / "snapshots")
        execute_snapshot_plan(
            snapshot_plan,
            fallback_source_dir=None,
            output_root=root,
            json_output=json_output,
            env=env,
        )
        return

    config = load_effective_config(env)
    selected_source_names = resolve_sources(config, source_names, "audit") if source_names else None
    try:
        request = build_qa_session_request(
            synthetic=synthetic,
            source_names=tuple(selected_source_names) if selected_source_names else None,
            fresh=fresh,
            ingest=ingest,
            regenerate_schemas=regenerate_schemas,
            only_stage=QAStage(only_stage) if only_stage else None,
            skip_stages=tuple(QAStage(stage) for stage in skip_stages),
            workspace=workspace,
            report_dir=report_dir,
            verbose=verbose,
            fail_fast=fail_fast,
            tier_filter=tier_filter,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    # --- Execute QA session ---
    from polylogue.showcase.qa_runner import run_qa_session

    result = run_qa_session(request)

    finalize_qa_run(
        result,
        plan=finalization_plan,
        archive_root=config.archive_root,
        env=env,
    )

    if not result.all_passed:
        raise SystemExit(1)


qa_command.add_command(generate_command)

__all__ = ["qa_command"]
