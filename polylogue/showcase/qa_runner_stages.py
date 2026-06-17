"""Stage helpers for QA orchestration."""

from __future__ import annotations

from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.workspace import override_workspace_env


def generate_extra_exercises() -> list[Exercise]:
    """Generate dynamic exercises from CLI introspection and schema catalog."""
    from polylogue.showcase.exercises import QA_EXTRA_EXERCISES

    return list(QA_EXTRA_EXERCISES)


def populate_artifact_coverage(result: QAResult, *, workspace_env: dict[str, str] | None) -> None:
    """Populate the artifact coverage stage against the active archive."""
    from polylogue.paths import db_path
    from polylogue.schemas.validation.artifacts import inspect_raw_artifact_coverage
    from polylogue.schemas.validation.requests import ArtifactCoverageRequest

    request = ArtifactCoverageRequest()
    try:
        if workspace_env:
            with override_workspace_env(workspace_env):
                result.coverage_report = inspect_raw_artifact_coverage(
                    db_path=db_path(),
                    request=request,
                )
        else:
            result.coverage_report = inspect_raw_artifact_coverage(
                db_path=db_path(),
                request=request,
            )
    except Exception as exc:
        result.coverage_error = str(exc)


__all__ = [
    "generate_extra_exercises",
    "populate_artifact_coverage",
]
