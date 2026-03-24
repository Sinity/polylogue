"""Canonical QA session workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from polylogue.showcase.invariants import check_invariants
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.qa_runner_reporting import save_qa_reports
from polylogue.showcase.qa_runner_stages import (
    generate_extra_exercises,
    populate_proof,
    populate_roundtrip_proof,
    populate_semantic_proof,
)
from polylogue.showcase.runner import ShowcaseRunner
from polylogue.showcase.workspace import (
    create_verification_workspace,
    ensure_report_dir,
    generate_synthetic_fixtures,
    run_pipeline_for_configured_sources,
    run_pipeline_for_fixture_workspace,
)
from polylogue.sync_bridge import run_coroutine_sync


def run_qa_session(
    *,
    live: bool = False,
    fresh: bool = True,
    ingest: bool = False,
    source_names: list[str] | None = None,
    regenerate_schemas: bool = False,
    skip_audit: bool = False,
    skip_exercises: bool = False,
    skip_invariants: bool = False,
    workspace_dir: Path | None = None,
    workspace_env: dict[str, str] | None = None,
    report_dir: Path | None = None,
    provider: str | None = None,
    verbose: bool = False,
    fail_fast: bool = False,
    tier_filter: int | None = None,
    synthetic_count: int = 3,
) -> QAResult:
    """Execute a composable QA session."""
    result = QAResult(report_dir=report_dir)
    workspace_env_for_runner: dict[str, str] | None = workspace_env
    needs_workspace = not skip_exercises

    if needs_workspace and fresh and not live:
        workspace = create_verification_workspace(workspace_dir)
        generate_synthetic_fixtures(workspace.fixture_dir, count=synthetic_count, style="showcase")
        run_pipeline_for_fixture_workspace(
            workspace,
            regenerate_schemas=regenerate_schemas,
        )
        workspace_env_for_runner = dict(workspace.env_vars)
        report_dir = ensure_report_dir(workspace, report_dir)
        result.report_dir = report_dir
    elif needs_workspace and fresh and live and source_names:
        workspace = create_verification_workspace(workspace_dir)
        run_pipeline_for_configured_sources(
            workspace,
            source_names=source_names,
            regenerate_schemas=regenerate_schemas,
        )
        workspace_env_for_runner = dict(workspace.env_vars)
        report_dir = ensure_report_dir(workspace, report_dir)
        result.report_dir = report_dir
    elif needs_workspace and fresh and live:
        workspace = create_verification_workspace(workspace_dir)
        run_pipeline_for_configured_sources(
            workspace,
            regenerate_schemas=regenerate_schemas,
        )
        workspace_env_for_runner = dict(workspace.env_vars)
        report_dir = ensure_report_dir(workspace, report_dir)
        result.report_dir = report_dir
    elif needs_workspace and live and ingest:
        from polylogue.config import get_config
        from polylogue.pipeline.runner import run_sources

        config = get_config()
        names = source_names if source_names else None
        run_coroutine_sync(run_sources(
            config=config,
            stage="all",
            plan=None,
            ui=None,
            source_names=names,
        ))

    if skip_audit:
        result.audit_skipped = True
    else:
        try:
            from polylogue.schemas.audit import audit_all_providers, audit_provider

            result.audit_report = audit_provider(provider) if provider else audit_all_providers()
            if not result.audit_report.all_passed:
                populate_proof(result, workspace_env=workspace_env_for_runner)
                populate_semantic_proof(result, workspace_env=workspace_env_for_runner)
                populate_roundtrip_proof(result, provider=provider)
                result.exercises_skipped = True
                result.invariants_skipped = True
                if verbose:
                    print(result.audit_report.format_text(), file=sys.stderr)
                if report_dir:
                    result.report_dir = report_dir
                    save_qa_reports(result, report_dir)
                return result
        except Exception as exc:
            result.audit_error = str(exc)
            populate_proof(result, workspace_env=workspace_env_for_runner)
            populate_semantic_proof(result, workspace_env=workspace_env_for_runner)
            populate_roundtrip_proof(result, provider=provider)
            result.exercises_skipped = True
            result.invariants_skipped = True
            if report_dir:
                result.report_dir = report_dir
                save_qa_reports(result, report_dir)
            return result

    populate_proof(result, workspace_env=workspace_env_for_runner)
    populate_semantic_proof(result, workspace_env=workspace_env_for_runner)
    populate_roundtrip_proof(result, provider=provider)

    if skip_exercises:
        result.exercises_skipped = True
    else:
        runner = ShowcaseRunner(
            live=live and not fresh,
            output_dir=report_dir,
            fail_fast=fail_fast,
            verbose=verbose,
            tier_filter=tier_filter,
            extra_exercises=generate_extra_exercises(),
            workspace_env=workspace_env_for_runner,
        )
        result.showcase_result = runner.run()

    if skip_invariants:
        result.invariants_skipped = True
    elif result.showcase_result:
        result.invariant_results = check_invariants(result.showcase_result.results)

    if report_dir:
        result.report_dir = report_dir
        save_qa_reports(result, report_dir)
    return result


__all__ = ["run_qa_session"]
