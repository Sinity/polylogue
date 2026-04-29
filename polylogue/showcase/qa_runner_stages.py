"""Stage helpers for QA orchestration."""

from __future__ import annotations

from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.workspace import override_workspace_env


def generate_extra_exercises() -> list[Exercise]:
    """Generate dynamic exercises from CLI introspection and schema catalog."""
    from polylogue.showcase.exercises import QA_EXTRA_EXERCISES

    return list(QA_EXTRA_EXERCISES)


def populate_proof(result: QAResult, *, workspace_env: dict[str, str] | None) -> None:
    """Populate the artifact proof stage against the active archive."""
    from polylogue.paths import db_path
    from polylogue.schemas.validation.artifacts import prove_raw_artifact_coverage
    from polylogue.schemas.validation.requests import ArtifactProofRequest

    request = ArtifactProofRequest()
    try:
        if workspace_env:
            with override_workspace_env(workspace_env):
                result.proof_report = prove_raw_artifact_coverage(
                    db_path=db_path(),
                    request=request,
                )
        else:
            result.proof_report = prove_raw_artifact_coverage(
                db_path=db_path(),
                request=request,
            )
    except Exception as exc:
        result.proof_error = str(exc)


__all__ = [
    "generate_extra_exercises",
    "populate_proof",
]
