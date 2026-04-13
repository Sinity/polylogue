"""Canonical QA session workflow."""

from __future__ import annotations

import sys

from polylogue.showcase.invariants import check_invariants
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.qa_runner_reporting import save_qa_reports
from polylogue.showcase.qa_runner_request import QASessionRequest
from polylogue.showcase.qa_runner_stages import (
    generate_extra_exercises,
    populate_proof,
)
from polylogue.showcase.runner import ShowcaseRunner
from polylogue.showcase.workspace import (
    create_verification_workspace,
    ensure_report_dir,
    run_pipeline_for_configured_sources,
    seed_workspace_from_corpus_request,
)
from polylogue.sync_bridge import run_coroutine_sync


def run_qa_session(
    request: QASessionRequest,
) -> QAResult:
    """Execute a composable QA session."""
    result = QAResult(report_dir=request.report_dir)
    result.proof_skipped = request.skip_proof
    workspace_env_for_runner: dict[str, str] | None = request.workspace_env
    needs_workspace = request.needs_workspace
    report_dir = request.report_dir

    if needs_workspace and request.fresh and not request.live:
        workspace = create_verification_workspace(request.workspace_dir)
        seed_workspace_from_corpus_request(
            workspace,
            request=request.corpus_request,
            regenerate_schemas=request.regenerate_schemas,
        )
        workspace_env_for_runner = dict(workspace.env_vars)
        report_dir = ensure_report_dir(workspace, request.report_dir)
        result.report_dir = report_dir
    elif needs_workspace and request.fresh and request.live and request.source_names:
        workspace = create_verification_workspace(request.workspace_dir)
        run_pipeline_for_configured_sources(
            workspace,
            source_names=list(request.source_names),
            regenerate_schemas=request.regenerate_schemas,
        )
        workspace_env_for_runner = dict(workspace.env_vars)
        report_dir = ensure_report_dir(workspace, request.report_dir)
        result.report_dir = report_dir
    elif needs_workspace and request.fresh and request.live:
        workspace = create_verification_workspace(request.workspace_dir)
        run_pipeline_for_configured_sources(
            workspace,
            regenerate_schemas=request.regenerate_schemas,
        )
        workspace_env_for_runner = dict(workspace.env_vars)
        report_dir = ensure_report_dir(workspace, request.report_dir)
        result.report_dir = report_dir
    elif needs_workspace and request.live and request.ingest:
        from polylogue.config import get_config
        from polylogue.pipeline.runner import run_sources

        config = get_config()
        names = list(request.source_names) if request.source_names else None
        run_coroutine_sync(
            run_sources(
                config=config,
                stage="all",
                plan=None,
                ui=None,
                source_names=names,
            )
        )

    if request.skip_audit:
        result.audit_skipped = True
    else:
        try:
            from polylogue.schemas.audit_workflow import audit_all_providers, audit_provider

            result.audit_report = audit_provider(request.provider) if request.provider else audit_all_providers()
            if not result.audit_report.all_passed:
                if not request.skip_proof:
                    populate_proof(result, workspace_env=workspace_env_for_runner)
                result.exercises_skipped = True
                result.invariants_skipped = True
                if request.verbose:
                    print(result.audit_report.format_text(), file=sys.stderr)
                if report_dir:
                    result.report_dir = report_dir
                    save_qa_reports(result, report_dir)
                return result
        except Exception as exc:
            result.audit_error = str(exc)
            if not request.skip_proof:
                populate_proof(result, workspace_env=workspace_env_for_runner)
            result.exercises_skipped = True
            result.invariants_skipped = True
            if report_dir:
                result.report_dir = report_dir
                save_qa_reports(result, report_dir)
            return result

    if not request.skip_proof:
        populate_proof(result, workspace_env=workspace_env_for_runner)

    if request.skip_exercises:
        result.exercises_skipped = True
    else:
        runner = ShowcaseRunner(
            live=request.live and not request.fresh,
            output_dir=report_dir,
            fail_fast=request.fail_fast,
            verbose=request.verbose,
            tier_filter=request.tier_filter,
            extra_exercises=generate_extra_exercises(),
            workspace_env=workspace_env_for_runner,
            corpus_request=request.corpus_request,
        )
        result.showcase_result = runner.run()

    if request.skip_invariants:
        result.invariants_skipped = True
    elif result.showcase_result:
        result.invariant_results = check_invariants(result.showcase_result.results)

    if report_dir:
        result.report_dir = report_dir
        save_qa_reports(result, report_dir)
    return result


__all__ = ["run_qa_session"]
