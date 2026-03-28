"""Composable QA command: audit, exercises, invariants, and archival."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.cli.helpers import load_effective_config
from polylogue.cli.qa_capture import run_vhs_capture as _run_vhs_capture
from polylogue.cli.qa_snapshot import snapshot_results
from polylogue.cli.types import AppEnv

_STAGE_CHOICES = click.Choice(["audit", "exercises", "invariants"])


@click.command("qa")
@click.option("--synthetic/--live", default=True,
              help="Data source: synthetic (default) or live real data")
@click.option("--source", "source_names", multiple=True,
              help="Specific real source(s) in fresh workspace (repeatable, implies --fresh)")
@click.option("--fresh", is_flag=True, default=None,
              help="Run in an isolated temp workspace (default for synthetic)")
@click.option("--workspace", type=click.Path(path_type=Path),
              help="Reuse a specific workspace directory")
@click.option("--ingest", is_flag=True, default=None,
              help="Run ingestion pipeline (auto for synthetic and fresh-with-sources)")
@click.option("--schemas", "regenerate_schemas", is_flag=True,
              help="Regenerate schemas during pipeline")
@click.option("--only", "only_stage", type=_STAGE_CHOICES, default=None,
              help="Run only this stage")
@click.option("--skip", "skip_stages", multiple=True, type=_STAGE_CHOICES,
              help="Skip this stage (repeatable)")
@click.option("--tier", "tier_filter", type=int, default=None,
              help="Only run exercises at this tier (0/1/2)")
@click.option("--fail-fast", is_flag=True, help="Stop on first exercise failure")
@click.option(
    "--capture",
    type=click.Choice(["none", "vhs"], case_sensitive=False),
    default="none",
    show_default=True,
    help="Capture mode for exercises",
)
@click.option("--report-dir", type=click.Path(path_type=Path), default=None,
              help="Directory for QA artifacts (auto-generated if omitted)")
@click.option("--json", "json_output", is_flag=True, help="Machine-readable output")
@click.option("--verbose", is_flag=True, help="Print exercise outputs")
@click.option("--snapshot", "snapshot_label", default=None, is_flag=False,
              flag_value="snapshot",
              help="Archive results after QA completes (optional label)")
@click.option("--snapshot-from", type=click.Path(path_type=Path, exists=True, file_okay=False),
              default=None,
              help="Archive an existing output directory (skips QA execution)")
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
      polylogue qa                              # Full synthetic QA
      polylogue qa --live                       # Exercises against real data
      polylogue qa --source inbox               # Fresh workspace from inbox
      polylogue qa --only audit                 # Schema audit only
      polylogue qa --only exercises --tier 0    # Tier-0 smoke test
      polylogue qa --skip invariants            # Skip invariant checks
      polylogue qa --snapshot release-v3        # QA + archive results
      polylogue qa --snapshot-from ./qa_outputs # Archive existing directory
    """
    # --- Snapshot-from: archive only, no QA ---
    if snapshot_from is not None:
        config = load_effective_config(env)
        root = report_dir or (config.archive_root / "qa" / "snapshots")
        snapshot_results(
            snapshot_from,
            label=snapshot_label or "snapshot",
            output_root=root,
            json_output=json_output,
            env=env,
        )
        return

    # --- Validate flag combinations ---
    if only_stage and skip_stages:
        raise click.UsageError("--only and --skip are mutually exclusive")

    live = not synthetic

    # --source implies fresh + live
    if source_names:
        live = True
        if fresh is None:
            fresh = True

    # Determine freshness
    if fresh is None:
        fresh = not live  # synthetic → fresh, live → existing DB

    # Determine ingestion
    if ingest is None:
        ingest = fresh  # fresh workspaces need ingestion

    # Resolve which stages to run
    run_audit = True
    run_exercises = True
    run_invariants = True

    if only_stage:
        run_audit = only_stage == "audit"
        run_exercises = only_stage == "exercises"
        run_invariants = only_stage == "invariants"
    else:
        if "audit" in skip_stages:
            run_audit = False
        if "exercises" in skip_stages:
            run_exercises = False
        if "invariants" in skip_stages:
            run_invariants = False

    # --- Execute QA session ---
    from polylogue.showcase.qa_report import generate_qa_session
    from polylogue.showcase.qa_runner import (
        format_qa_summary,
        run_qa_session,
    )

    result = run_qa_session(
        live=live,
        fresh=fresh,
        ingest=ingest,
        source_names=list(source_names) if source_names else None,
        regenerate_schemas=regenerate_schemas,
        skip_audit=not run_audit,
        skip_exercises=not run_exercises,
        skip_invariants=not run_invariants,
        workspace_dir=workspace,
        report_dir=report_dir,
        verbose=verbose,
        fail_fast=fail_fast,
        tier_filter=tier_filter,
    )

    # --- VHS capture ---
    if capture.lower() == "vhs" and result.showcase_result and result.showcase_result.output_dir:
        _run_vhs_capture(env, result.showcase_result, json_output)

    # --- Output ---
    if json_output:
        click.echo(json.dumps(generate_qa_session(result), indent=2))
    else:
        env.ui.console.print(format_qa_summary(result))

    # --- Snapshot ---
    if snapshot_label is not None and result.report_dir:
        config = load_effective_config(env)
        root = config.archive_root / "qa" / "snapshots"
        snapshot_results(
            result.report_dir,
            label=snapshot_label,
            output_root=root,
            json_output=json_output,
            env=env,
        )

    if not result.all_passed:
        raise SystemExit(1)
__all__ = ["qa_command"]
