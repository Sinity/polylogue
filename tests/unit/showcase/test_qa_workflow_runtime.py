from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.pipeline.run_support import RUN_STAGE_SEQUENCES
from polylogue.scenarios import polylogue_execution
from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.qa_runner_request import (
    QASessionPlan,
    QASessionRequest,
    QAWorkspaceMode,
)
from polylogue.showcase.qa_runner_workflow import (
    PreparedQARuntime,
    _prepare_runtime,
    _run_audit_stage,
    _run_live_ingest,
    run_qa_session,
)
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult


def _make_audit_report(*, all_passed: bool) -> SimpleNamespace:
    return SimpleNamespace(all_passed=all_passed, format_text=lambda: "audit report")


def _make_showcase_result() -> ShowcaseResult:
    exercise = Exercise(
        name="stats-default",
        group="query-read",
        description="stats",
        execution=polylogue_execution("stats"),
    )
    result = ShowcaseResult()
    result.results = [
        ExerciseResult(
            exercise=exercise,
            passed=True,
            exit_code=0,
            output="stats output",
            duration_ms=12.0,
        )
    ]
    return result


def test_prepare_runtime_returns_existing_context_when_workspace_not_needed(tmp_path: Path) -> None:
    request = QASessionRequest(
        skip_exercises=True,
        workspace_env={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive")},
        report_dir=tmp_path / "reports",
    )

    runtime = _prepare_runtime(request)

    assert runtime == PreparedQARuntime(
        workspace_env={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive")},
        report_dir=tmp_path / "reports",
    )


def test_prepare_runtime_seeds_corpus_workspace_and_report_dir(tmp_path: Path) -> None:
    request = QASessionRequest(
        workspace_dir=tmp_path / "workspace",
        report_dir=tmp_path / "reports",
        regenerate_schemas=True,
    )
    workspace = SimpleNamespace(env_vars={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "workspace" / "archive")})

    with (
        patch(
            "polylogue.showcase.qa_runner_workflow.create_verification_workspace", return_value=workspace
        ) as mock_create,
        patch("polylogue.showcase.qa_runner_workflow.seed_workspace_from_corpus_request") as mock_seed,
        patch(
            "polylogue.showcase.qa_runner_workflow.ensure_report_dir", return_value=tmp_path / "final-report"
        ) as mock_report,
    ):
        runtime = _prepare_runtime(request)

    mock_create.assert_called_once_with(request.workspace_dir)
    mock_seed.assert_called_once_with(
        workspace,
        request=request.corpus_request,
        regenerate_schemas=True,
    )
    mock_report.assert_called_once_with(workspace, request.report_dir)
    assert runtime.workspace_env == workspace.env_vars
    assert runtime.report_dir == tmp_path / "final-report"


def test_prepare_runtime_uses_configured_sources_workspace_mode(tmp_path: Path) -> None:
    request = QASessionRequest(
        live=True,
        fresh=True,
        source_names=("codex",),
        workspace_dir=tmp_path / "workspace",
        report_dir=tmp_path / "reports",
    )
    workspace = SimpleNamespace(env_vars={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "workspace" / "archive")})

    with (
        patch("polylogue.showcase.qa_runner_workflow.create_verification_workspace", return_value=workspace),
        patch("polylogue.showcase.qa_runner_workflow.run_pipeline_for_configured_sources") as mock_run,
        patch("polylogue.showcase.qa_runner_workflow.ensure_report_dir", return_value=tmp_path / "configured-report"),
    ):
        runtime = _prepare_runtime(request)

    mock_run.assert_called_once_with(
        workspace,
        source_names=["codex"],
        regenerate_schemas=False,
    )
    assert runtime.workspace_env == workspace.env_vars
    assert runtime.report_dir == tmp_path / "configured-report"


def test_prepare_runtime_rejects_unexpected_workspace_mode(tmp_path: Path) -> None:
    request = QASessionRequest(workspace_dir=tmp_path / "workspace")

    with (
        patch.object(
            QASessionRequest,
            "execution_plan",
            new_callable=PropertyMock,
            return_value=QASessionPlan(
                workspace_mode=QAWorkspaceMode.LIVE_INGEST,
                run_audit=True,
                run_proof=True,
                run_exercises=True,
                run_invariants=True,
            ),
        ),
        patch(
            "polylogue.showcase.qa_runner_workflow.create_verification_workspace",
            return_value=SimpleNamespace(env_vars={}),
        ),
    ):
        with pytest.raises(RuntimeError, match="unexpected workspace mode"):
            _prepare_runtime(request)


def test_run_live_ingest_uses_schema_stage_sequence_when_regenerating() -> None:
    request = QASessionRequest(
        live=True,
        ingest=True,
        regenerate_schemas=True,
        source_names=("chatgpt", "codex"),
    )

    with (
        patch("polylogue.pipeline.runner.run_sources", new=MagicMock(return_value="coro")) as mock_run_sources,
        patch("polylogue.showcase.qa_runner_workflow.run_coroutine_sync") as mock_sync,
        patch("polylogue.config.get_config", return_value=SimpleNamespace(name="config")) as mock_get_config,
    ):
        _run_live_ingest(request)

    mock_get_config.assert_called_once_with()
    mock_run_sources.assert_called_once()
    assert mock_run_sources.call_args.kwargs["config"].name == "config"
    assert mock_run_sources.call_args.kwargs["stage"] == "all"
    assert mock_run_sources.call_args.kwargs["stage_sequence"] == ("schema", *RUN_STAGE_SEQUENCES["all"])
    assert mock_run_sources.call_args.kwargs["source_names"] == ["chatgpt", "codex"]
    mock_sync.assert_called_once_with("coro")


def test_run_audit_stage_marks_skip_without_running_audit() -> None:
    result = QAResult()
    request = QASessionRequest(skip_audit=True)

    returned = _run_audit_stage(
        result,
        request=request,
        workspace_env=None,
        report_dir=None,
    )

    assert returned is None
    assert result.audit_skipped is True


def test_run_audit_stage_records_exception_and_returns_early_result(tmp_path: Path) -> None:
    result = QAResult()
    request = QASessionRequest()

    with (
        patch("polylogue.schemas.audit_workflow.audit_all_providers", side_effect=RuntimeError("boom")),
        patch("polylogue.showcase.qa_runner_workflow.populate_proof") as mock_proof,
        patch("polylogue.showcase.qa_runner_workflow.save_qa_reports") as mock_save,
    ):
        returned = _run_audit_stage(
            result,
            request=request,
            workspace_env={"XDG_CONFIG_HOME": str(tmp_path / "config")},
            report_dir=tmp_path / "report",
        )

    assert returned is result
    assert result.audit_error == "boom"
    assert result.exercises_skipped is True
    assert result.invariants_skipped is True
    assert result.report_dir == tmp_path / "report"
    mock_proof.assert_called_once_with(result, workspace_env={"XDG_CONFIG_HOME": str(tmp_path / "config")})
    mock_save.assert_called_once_with(result, tmp_path / "report")


def test_run_audit_stage_returns_none_when_audit_passes() -> None:
    result = QAResult()
    request = QASessionRequest(provider="chatgpt")

    with patch(
        "polylogue.schemas.audit_workflow.audit_provider", return_value=_make_audit_report(all_passed=True)
    ) as mock_audit:
        returned = _run_audit_stage(
            result,
            request=request,
            workspace_env=None,
            report_dir=None,
        )

    assert returned is None
    assert result.audit_report is not None
    mock_audit.assert_called_once_with("chatgpt")


def test_run_audit_stage_persists_failed_audit_and_prints_verbose_output(capsys: pytest.CaptureFixture[str]) -> None:
    result = QAResult()
    request = QASessionRequest(verbose=True)

    with (
        patch(
            "polylogue.schemas.audit_workflow.audit_all_providers", return_value=_make_audit_report(all_passed=False)
        ),
        patch("polylogue.showcase.qa_runner_workflow.populate_proof") as mock_proof,
    ):
        returned = _run_audit_stage(
            result,
            request=request,
            workspace_env=None,
            report_dir=None,
        )

    captured = capsys.readouterr()
    assert returned is result
    assert result.exercises_skipped is True
    assert result.invariants_skipped is True
    assert "audit report" in captured.err
    mock_proof.assert_called_once_with(result, workspace_env=None)


def test_run_qa_session_returns_early_audit_result(tmp_path: Path) -> None:
    request = QASessionRequest(fresh=False)
    early = QAResult(report_dir=tmp_path / "report")

    with (
        patch(
            "polylogue.showcase.qa_runner_workflow._prepare_runtime", return_value=PreparedQARuntime()
        ) as mock_prepare,
        patch("polylogue.showcase.qa_runner_workflow._run_audit_stage", return_value=early) as mock_audit,
    ):
        returned = run_qa_session(request)

    assert returned is early
    mock_prepare.assert_called_once_with(request)
    mock_audit.assert_called_once()


def test_run_qa_session_executes_full_success_path(tmp_path: Path) -> None:
    request = QASessionRequest(fresh=False, report_dir=tmp_path / "report")
    runtime = PreparedQARuntime(
        workspace_env={"POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "archive")}, report_dir=tmp_path / "report"
    )
    showcase_result = _make_showcase_result()
    invariant_result = SimpleNamespace(status=OutcomeStatus.OK)
    runner = MagicMock()
    runner.run.return_value = showcase_result

    with (
        patch("polylogue.showcase.qa_runner_workflow._prepare_runtime", return_value=runtime),
        patch("polylogue.showcase.qa_runner_workflow._run_audit_stage", return_value=None),
        patch("polylogue.showcase.qa_runner_workflow.populate_proof") as mock_proof,
        patch("polylogue.showcase.qa_runner_workflow.ShowcaseRunner", return_value=runner) as mock_runner_class,
        patch(
            "polylogue.showcase.qa_runner_workflow.check_invariants", return_value=[invariant_result]
        ) as mock_invariants,
        patch("polylogue.showcase.qa_runner_workflow.save_qa_reports") as mock_save,
    ):
        result = run_qa_session(request)

    mock_proof.assert_called_once_with(result, workspace_env=runtime.workspace_env)
    mock_runner_class.assert_called_once_with(
        live=False,
        output_dir=runtime.report_dir,
        fail_fast=False,
        verbose=False,
        tier_filter=None,
        extra_exercises=mock_runner_class.call_args.kwargs["extra_exercises"],
        workspace_env=runtime.workspace_env,
        corpus_request=request.corpus_request,
    )
    assert isinstance(mock_runner_class.call_args.kwargs["extra_exercises"], list)
    mock_invariants.assert_called_once_with(showcase_result.results)
    mock_save.assert_called_once_with(result, tmp_path / "report")
    assert result.showcase_result is showcase_result
    assert result.invariant_results == [invariant_result]


def test_run_qa_session_respects_skip_flags_and_live_ingest() -> None:
    request = QASessionRequest(
        live=True,
        ingest=True,
        skip_proof=True,
        skip_exercises=True,
        skip_invariants=True,
    )

    with (
        patch("polylogue.showcase.qa_runner_workflow._prepare_runtime", return_value=PreparedQARuntime()),
        patch("polylogue.showcase.qa_runner_workflow._run_live_ingest") as mock_ingest,
        patch("polylogue.showcase.qa_runner_workflow._run_audit_stage", return_value=None),
        patch("polylogue.showcase.qa_runner_workflow.populate_proof") as mock_proof,
        patch("polylogue.showcase.qa_runner_workflow.ShowcaseRunner") as mock_runner_class,
        patch("polylogue.showcase.qa_runner_workflow.check_invariants") as mock_invariants,
    ):
        result = run_qa_session(request)

    mock_ingest.assert_called_once_with(request)
    mock_proof.assert_not_called()
    mock_runner_class.assert_not_called()
    mock_invariants.assert_not_called()
    assert result.proof_skipped is True
    assert result.exercises_skipped is True
    assert result.invariants_skipped is True
