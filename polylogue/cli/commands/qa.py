"""Archive QA command: schema audit, artifact proof, and archival."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.cli.shared.helpers import complete_configured_source_names, load_effective_config, resolve_sources
from polylogue.cli.shared.qa_finalization import finalize_qa_run
from polylogue.cli.shared.qa_requests import (
    build_qa_invocation_plan,
)
from polylogue.cli.shared.qa_snapshot import execute_snapshot_plan
from polylogue.cli.shared.types import AppEnv
from polylogue.showcase.qa_runner_request import QAStage

_PRODUCT_STAGE_CHOICES = click.Choice([QAStage.AUDIT.value])


@click.command("audit")
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
@click.option("--only", "only_stage", type=_PRODUCT_STAGE_CHOICES, default=None, help="Run only this stage")
@click.option("--skip", "skip_stages", multiple=True, type=_PRODUCT_STAGE_CHOICES, help="Skip this stage (repeatable)")
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
    report_dir: Path | None,
    json_output: bool,
    verbose: bool,
    snapshot_label: str | None,
    snapshot_from: Path | None,
) -> None:
    """Run archive QA: schema audit and artifact proof checks.

    \b
    By default, audits packaged schemas and checks raw artifact proof against the active archive.

    \b
    Examples:
      polylogue audit                         # Schema audit + artifact proof
      polylogue audit --only audit            # Schema audit only
      polylogue audit --snapshot release-v3   # QA + archive results
      polylogue audit --snapshot-from ./qa_outputs

    \b
    Verification-lab corpus and scenario commands live under devtools:
      devtools lab-corpus seed --env-only
      devtools lab-scenario run archive-smoke --tier 0
    """
    config = load_effective_config(env)
    selected_source_names = resolve_sources(config, source_names, "audit") if source_names else None
    product_skip_stages = (
        skip_stages
        if only_stage is not None
        else (
            *skip_stages,
            QAStage.EXERCISES.value,
            QAStage.INVARIANTS.value,
        )
    )
    try:
        invocation_plan = build_qa_invocation_plan(
            synthetic=synthetic,
            source_names=tuple(selected_source_names) if selected_source_names else None,
            fresh=fresh,
            ingest=ingest,
            regenerate_schemas=regenerate_schemas,
            only_stage=only_stage,
            skip_stages=product_skip_stages,
            workspace=workspace,
            report_dir=report_dir,
            verbose=verbose,
            fail_fast=False,
            tier_filter=None,
            capture="none",
            json_output=json_output,
            snapshot_label=snapshot_label,
            snapshot_from=snapshot_from,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    snapshot_plan = invocation_plan.snapshot_plan

    # --- Snapshot-from: archive only, no QA ---
    if invocation_plan.snapshot_only and snapshot_plan and snapshot_plan.source_dir is not None:
        root = report_dir or (config.archive_root / "qa" / "snapshots")
        execute_snapshot_plan(
            snapshot_plan,
            fallback_source_dir=None,
            output_root=root,
            json_output=json_output,
            env=env,
        )
        return

    # --- Execute QA session ---
    from polylogue.showcase.qa_runner import run_qa_session

    request = invocation_plan.session_request
    if request is None:
        raise click.UsageError("QA invocation plan did not produce an executable session request.")

    result = run_qa_session(request)

    finalize_qa_run(
        result,
        plan=invocation_plan.finalization_plan,
        archive_root=config.archive_root,
        env=env,
    )

    if not result.all_passed:
        raise SystemExit(1)


__all__ = ["qa_command"]
