# mypy: disable-error-code="comparison-overlap"

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from polylogue.showcase.exercises import QA_EXTRA_EXERCISES
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.qa_runner_stages import generate_extra_exercises, populate_proof


def test_generate_extra_exercises_returns_list_copy() -> None:
    exercises = generate_extra_exercises()

    assert exercises == list(QA_EXTRA_EXERCISES)
    assert exercises is not QA_EXTRA_EXERCISES


def test_populate_proof_sets_report_without_workspace_override(tmp_path: Path) -> None:
    result = QAResult()
    proof_report = SimpleNamespace(is_clean=True)

    with (
        patch("polylogue.paths.db_path", return_value=tmp_path / "polylogue.db") as mock_db_path,
        patch(
            "polylogue.schemas.validation.artifacts.prove_raw_artifact_coverage", return_value=proof_report
        ) as mock_prove,
    ):
        populate_proof(result, workspace_env=None)

    mock_db_path.assert_called_once_with()
    mock_prove.assert_called_once()
    assert mock_prove.call_args.kwargs["db_path"] == tmp_path / "polylogue.db"
    assert result.proof_report is proof_report
    assert result.proof_error is None


def test_populate_proof_uses_workspace_override_context(tmp_path: Path) -> None:
    result = QAResult()
    proof_report = SimpleNamespace(is_clean=True)

    with (
        patch("polylogue.paths.db_path", return_value=tmp_path / "polylogue.db"),
        patch(
            "polylogue.showcase.qa_runner_stages.override_workspace_env", return_value=nullcontext()
        ) as mock_override,
        patch(
            "polylogue.schemas.validation.artifacts.prove_raw_artifact_coverage", return_value=proof_report
        ) as mock_prove,
    ):
        populate_proof(result, workspace_env={"XDG_CONFIG_HOME": str(tmp_path / "config")})

    mock_override.assert_called_once_with({"XDG_CONFIG_HOME": str(tmp_path / "config")})
    mock_prove.assert_called_once()
    assert result.proof_report is proof_report


def test_populate_proof_captures_exceptions() -> None:
    result = QAResult()

    with patch(
        "polylogue.schemas.validation.artifacts.prove_raw_artifact_coverage", side_effect=RuntimeError("broken")
    ):
        populate_proof(result, workspace_env=None)

    assert result.proof_report is None
    assert result.proof_error == "broken"
