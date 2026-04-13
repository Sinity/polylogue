"""Stage helpers for QA orchestration."""

from __future__ import annotations

from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.workspace import override_workspace_env


def generate_extra_exercises() -> list:
    """Generate dynamic exercises from CLI introspection and schema catalog."""
    from polylogue.showcase.generators import (
        generate_format_scenarios,
        generate_schema_scenarios,
    )
    from polylogue.showcase.scenario_models import compile_exercise_scenarios

    scenarios = generate_schema_scenarios() + generate_format_scenarios()
    return list(compile_exercise_scenarios(scenarios))


def populate_proof(result: QAResult, *, workspace_env: dict[str, str] | None) -> None:
    """Populate the artifact proof stage against the active archive."""
    from polylogue.paths import db_path as default_db_path
    from polylogue.schemas.verification_artifacts import prove_raw_artifact_coverage
    from polylogue.schemas.verification_requests import ArtifactProofRequest

    try:
        if workspace_env:
            with override_workspace_env(workspace_env):
                result.proof_report = prove_raw_artifact_coverage(
                    db_path=default_db_path(),
                    request=ArtifactProofRequest(),
                )
        else:
            result.proof_report = prove_raw_artifact_coverage(
                db_path=default_db_path(),
                request=ArtifactProofRequest(),
            )
    except Exception as exc:
        result.proof_error = str(exc)


__all__ = [
    "generate_extra_exercises",
    "populate_proof",
]
