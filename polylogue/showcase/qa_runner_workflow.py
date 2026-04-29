"""Canonical QA session workflow."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.showcase.invariants import check_invariants
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.qa_runner_reporting import save_qa_reports
from polylogue.showcase.qa_runner_request import QASessionRequest, QAWorkspaceMode
from polylogue.showcase.qa_runner_stages import generate_extra_exercises, populate_proof
from polylogue.showcase.runner import ShowcaseRunner
from polylogue.showcase.workspace import (
    create_verification_workspace,
    ensure_report_dir,
    run_pipeline_for_configured_sources,
    seed_workspace_from_corpus_request,
)


@dataclass(frozen=True, slots=True)
class PreparedQARuntime:
    """Runtime environment resolved before the staged QA workflow begins."""

    workspace_env: dict[str, str] | None = None
    report_dir: Path | None = None


def _persist_and_return(result: QAResult, *, report_dir: Path | None) -> QAResult:
    if report_dir is not None:
        result.report_dir = report_dir
        save_qa_reports(result, report_dir)
    return result


def _prepare_runtime(request: QASessionRequest) -> PreparedQARuntime:
    """Resolve workspace/report wiring for the requested QA run."""
    plan = request.execution_plan
    if not plan.needs_workspace:
        return PreparedQARuntime(
            workspace_env=request.workspace_env,
            report_dir=request.report_dir,
        )

    workspace = create_verification_workspace(request.workspace_dir)
    if plan.workspace_mode is QAWorkspaceMode.SEEDED_CORPUS:
        seed_workspace_from_corpus_request(
            workspace,
            request=request.corpus_request,
            regenerate_schemas=request.regenerate_schemas,
        )
    elif plan.workspace_mode is QAWorkspaceMode.CONFIGURED_SOURCES:
        run_pipeline_for_configured_sources(
            workspace,
            source_names=list(request.source_names) if request.source_names else None,
            regenerate_schemas=request.regenerate_schemas,
        )
    else:
        raise RuntimeError(f"unexpected workspace mode: {plan.workspace_mode}")

    return PreparedQARuntime(
        workspace_env=dict(workspace.env_vars),
        report_dir=ensure_report_dir(workspace, request.report_dir),
    )


def _run_live_ingest(request: QASessionRequest) -> None:
    """Run the live archive ingest pass requested by the QA session."""
    from polylogue.config import get_config
    from polylogue.pipeline.run_support import RUN_STAGE_SEQUENCES
    from polylogue.pipeline.runner import run_sources

    names = list(request.source_names) if request.source_names else None
    stage_sequence = None
    if request.regenerate_schemas:
        stage_sequence = ("schema", *RUN_STAGE_SEQUENCES["all"])
    run_coroutine_sync(
        run_sources(
            config=get_config(),
            stage="all",
            stage_sequence=stage_sequence,
            plan=None,
            ui=None,
            source_names=names,
        )
    )


def _run_audit_stage(
    result: QAResult,
    *,
    request: QASessionRequest,
    workspace_env: dict[str, str] | None,
    report_dir: Path | None,
) -> QAResult | None:
    """Execute the schema-audit stage, returning an early-finished result when needed."""
    if request.skip_audit:
        result.audit_skipped = True
        return None

    try:
        from polylogue.schemas.audit.workflow import audit_all_providers, audit_provider

        result.audit_report = audit_provider(request.provider) if request.provider else audit_all_providers()
    except Exception as exc:
        result.audit_error = str(exc)
        if not request.skip_proof:
            populate_proof(result, workspace_env=workspace_env)
        result.exercises_skipped = True
        result.invariants_skipped = True
        return _persist_and_return(result, report_dir=report_dir)

    if result.audit_report.all_passed:
        return None

    if not request.skip_proof:
        populate_proof(result, workspace_env=workspace_env)
    result.exercises_skipped = True
    result.invariants_skipped = True
    if request.verbose:
        print(result.audit_report.format_text(), file=sys.stderr)
    return _persist_and_return(result, report_dir=report_dir)


def run_qa_session(request: QASessionRequest) -> QAResult:
    """Execute a composable QA session."""
    result = QAResult(report_dir=request.report_dir)
    result.proof_skipped = request.skip_proof
    runtime = _prepare_runtime(request)
    plan = request.execution_plan

    if plan.workspace_mode is QAWorkspaceMode.LIVE_INGEST:
        _run_live_ingest(request)

    early_result = _run_audit_stage(
        result,
        request=request,
        workspace_env=runtime.workspace_env,
        report_dir=runtime.report_dir,
    )
    if early_result is not None:
        return early_result

    if not request.skip_proof:
        populate_proof(result, workspace_env=runtime.workspace_env)

    if request.skip_exercises:
        result.exercises_skipped = True
    else:
        runner = ShowcaseRunner(
            live=request.live and not request.fresh,
            output_dir=runtime.report_dir,
            fail_fast=request.fail_fast,
            verbose=request.verbose,
            tier_filter=request.tier_filter,
            extra_exercises=generate_extra_exercises(),
            workspace_env=runtime.workspace_env,
            corpus_request=request.corpus_request,
        )
        result.showcase_result = runner.run()

    if request.skip_invariants:
        result.invariants_skipped = True
    elif result.showcase_result is not None:
        result.invariant_results = check_invariants(result.showcase_result.results)

    return _persist_and_return(result, report_dir=runtime.report_dir)


__all__ = ["run_qa_session"]
