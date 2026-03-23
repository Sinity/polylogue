"""QA orchestration: composable audit → exercises → invariants pipeline.

Supports both synthetic (fresh workspace) and live (existing DB) modes,
with optional real-data ingestion into isolated workspaces.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.invariants import (
    InvariantResult,
    check_invariants,
)
from polylogue.showcase.runner import ShowcaseResult, ShowcaseRunner
from polylogue.showcase.workspace import (
    create_verification_workspace,
    ensure_report_dir,
    generate_synthetic_fixtures,
    override_workspace_env,
    run_pipeline_for_configured_sources,
    run_pipeline_for_fixture_workspace,
)
from polylogue.sync_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.rendering.semantic_proof import SemanticProofSuiteReport
    from polylogue.schemas.audit import AuditReport
    from polylogue.schemas.roundtrip_proof import RoundtripProofSuiteReport
    from polylogue.schemas.verification_models import ArtifactProofReport


@dataclass
class QAResult:
    """Complete QA session result."""

    audit_report: AuditReport | None = None
    audit_error: str | None = None
    audit_skipped: bool = False
    proof_report: ArtifactProofReport | None = None
    proof_error: str | None = None
    semantic_proof_report: SemanticProofSuiteReport | None = None
    semantic_proof_error: str | None = None
    roundtrip_proof_report: RoundtripProofSuiteReport | None = None
    roundtrip_proof_error: str | None = None
    showcase_result: ShowcaseResult | None = None
    exercises_skipped: bool = False
    invariant_results: list[InvariantResult] = field(default_factory=list)
    invariants_skipped: bool = False
    report_dir: Path | None = None

    @property
    def audit_status(self) -> OutcomeStatus:
        """Status for the schema audit stage."""
        if self.audit_skipped:
            return OutcomeStatus.SKIP
        if self.audit_error is not None:
            return OutcomeStatus.ERROR
        if self.audit_report is None:
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK if self.audit_report.all_passed else OutcomeStatus.ERROR

    @property
    def audit_passed(self) -> bool:
        """True if the schema audit stage passed."""
        return self.audit_status is OutcomeStatus.OK

    @property
    def proof_status(self) -> OutcomeStatus:
        """Status for the artifact proof stage."""
        if self.proof_error is not None:
            return OutcomeStatus.ERROR
        if self.proof_report is None:
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK if self.proof_report.is_clean else OutcomeStatus.ERROR

    @property
    def semantic_proof_status(self) -> OutcomeStatus:
        """Status for the semantic proof stage."""
        if self.semantic_proof_error is not None:
            return OutcomeStatus.ERROR
        if self.semantic_proof_report is None:
            return OutcomeStatus.SKIP
        return OutcomeStatus.OK if self.semantic_proof_report.is_clean else OutcomeStatus.ERROR

    @property
    def roundtrip_proof_status(self) -> OutcomeStatus:
        """Status for the schema/evidence roundtrip proof stage."""
        if self.roundtrip_proof_error is not None:
            return OutcomeStatus.ERROR
        if self.roundtrip_proof_report is None:
            return OutcomeStatus.SKIP
        return OutcomeStatus.OK if self.roundtrip_proof_report.is_clean else OutcomeStatus.ERROR

    @property
    def showcase_status(self) -> OutcomeStatus:
        """Status for the exercise stage."""
        if self.exercises_skipped:
            return OutcomeStatus.SKIP
        if self.showcase_result is None:
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK if self.showcase_result.failed == 0 else OutcomeStatus.ERROR

    @property
    def invariant_status(self) -> OutcomeStatus:
        """Status for the invariant stage."""
        if self.invariants_skipped or self.showcase_result is None:
            return OutcomeStatus.SKIP
        if any(r.status is OutcomeStatus.ERROR for r in self.invariant_results):
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK

    @property
    def overall_status(self) -> OutcomeStatus:
        """Aggregate status across all executed QA stages."""
        if any(
            stage is OutcomeStatus.ERROR
            for stage in (
                self.audit_status,
                self.proof_status,
                self.semantic_proof_status,
                self.roundtrip_proof_status,
                self.showcase_status,
                self.invariant_status,
            )
        ):
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK

    @property
    def all_passed(self) -> bool:
        """True if all executed stages passed."""
        return self.overall_status is OutcomeStatus.OK


def _generate_extra_exercises() -> list:
    """Generate dynamic exercises from CLI introspection and schema catalog."""
    from polylogue.showcase.generators import (
        generate_format_exercises,
        generate_schema_exercises,
    )

    exercises = []
    exercises.extend(generate_schema_exercises())
    exercises.extend(generate_format_exercises())
    return exercises


def _populate_proof(
    result: QAResult,
    *,
    workspace_env: dict[str, str] | None,
) -> None:
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


def _populate_semantic_proof(
    result: QAResult,
    *,
    workspace_env: dict[str, str] | None,
) -> None:
    """Populate the semantic proof stage against the active archive."""
    from polylogue.rendering.semantic_proof import prove_semantic_surface_suite

    try:
        if workspace_env:
            with override_workspace_env(workspace_env):
                result.semantic_proof_report = prove_semantic_surface_suite()
        else:
            result.semantic_proof_report = prove_semantic_surface_suite()
    except Exception as exc:
        result.semantic_proof_error = str(exc)


def _populate_roundtrip_proof(
    result: QAResult,
    *,
    provider: str | None,
) -> None:
    """Populate the synthetic schema/evidence roundtrip proof stage."""
    from polylogue.schemas.roundtrip_proof import prove_schema_evidence_roundtrip_suite

    try:
        result.roundtrip_proof_report = prove_schema_evidence_roundtrip_suite(
            providers=[provider] if provider else None,
            count=1,
        )
    except Exception as exc:
        result.roundtrip_proof_error = str(exc)


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
    """Execute a composable QA session.

    Stages (in order, each skippable):
      1. Schema audit
      2. Artifact proof
      3. Semantic proof
      4. Roundtrip proof
      5. Exercises (showcase)
      6. Invariant checks
      7. Report generation
    """
    result = QAResult(report_dir=report_dir)
    workspace_env_for_runner: dict[str, str] | None = workspace_env

    # Skip workspace setup if only running audit (no data needed)
    needs_workspace = not skip_exercises

    # --- Workspace setup ---
    if needs_workspace and fresh and not live:
        # Synthetic mode: create workspace, generate fixtures, ingest
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
        # Real data in fresh workspace
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
        # Fresh workspace with all real sources
        workspace = create_verification_workspace(workspace_dir)
        run_pipeline_for_configured_sources(
            workspace,
            regenerate_schemas=regenerate_schemas,
        )
        workspace_env_for_runner = dict(workspace.env_vars)
        report_dir = ensure_report_dir(workspace, report_dir)
        result.report_dir = report_dir

    elif needs_workspace and live and ingest:
        # Live mode with ingestion on existing DB
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

    # --- Step 1: Schema audit ---
    if skip_audit:
        result.audit_skipped = True
    else:
        try:
            from polylogue.schemas.audit import audit_all_providers, audit_provider

            audit_report = audit_provider(provider) if provider else audit_all_providers()

            result.audit_report = audit_report

            if not audit_report.all_passed:
                _populate_proof(result, workspace_env=workspace_env_for_runner)
                _populate_semantic_proof(result, workspace_env=workspace_env_for_runner)
                _populate_roundtrip_proof(result, provider=provider)
                result.exercises_skipped = True
                result.invariants_skipped = True
                if verbose:
                    print(audit_report.format_text(), file=sys.stderr)
                if report_dir:
                    result.report_dir = report_dir
                    _save_qa_reports(result, report_dir)
                return result
        except Exception as e:
            result.audit_error = str(e)
            _populate_proof(result, workspace_env=workspace_env_for_runner)
            _populate_semantic_proof(result, workspace_env=workspace_env_for_runner)
            _populate_roundtrip_proof(result, provider=provider)
            result.exercises_skipped = True
            result.invariants_skipped = True
            if report_dir:
                result.report_dir = report_dir
                _save_qa_reports(result, report_dir)
            return result

    # --- Step 2: Artifact proof ---
    _populate_proof(result, workspace_env=workspace_env_for_runner)

    # --- Step 3: Semantic proof ---
    _populate_semantic_proof(result, workspace_env=workspace_env_for_runner)

    # --- Step 4: Roundtrip proof ---
    _populate_roundtrip_proof(result, provider=provider)

    # --- Step 5: Exercises ---
    if skip_exercises:
        result.exercises_skipped = True
    else:
        extra = _generate_extra_exercises()
        runner = ShowcaseRunner(
            live=live and not fresh,
            output_dir=report_dir,
            fail_fast=fail_fast,
            verbose=verbose,
            tier_filter=tier_filter,
            extra_exercises=extra,
            workspace_env=workspace_env_for_runner,
        )
        showcase_result = runner.run()
        result.showcase_result = showcase_result

    # --- Step 6: Invariant checks ---
    if skip_invariants:
        result.invariants_skipped = True
    elif result.showcase_result:
        result.invariant_results = check_invariants(result.showcase_result.results)

    # --- Step 7: Save reports ---
    if report_dir:
        result.report_dir = report_dir
        _save_qa_reports(result, report_dir)

    return result


def _save_qa_reports(result: QAResult, report_dir: Path) -> None:
    """Save all QA artifacts to the report directory."""
    import json

    report_dir.mkdir(parents=True, exist_ok=True)

    if result.audit_report:
        (report_dir / "schema-audit.json").write_text(
            json.dumps(result.audit_report.to_json(), indent=2)
        )
    elif result.audit_error:
        (report_dir / "schema-audit.json").write_text(
            json.dumps({"error": result.audit_error}, indent=2)
        )

    if result.proof_report is not None:
        (report_dir / "artifact-proof.json").write_text(
            json.dumps(result.proof_report.to_dict(), indent=2, sort_keys=True)
        )
    elif result.proof_error is not None:
        (report_dir / "artifact-proof.json").write_text(
            json.dumps({"error": result.proof_error}, indent=2, sort_keys=True)
        )

    if result.semantic_proof_report is not None:
        (report_dir / "semantic-proof.json").write_text(
            json.dumps(result.semantic_proof_report.to_dict(), indent=2, sort_keys=True)
        )
    elif result.semantic_proof_error is not None:
        (report_dir / "semantic-proof.json").write_text(
            json.dumps({"error": result.semantic_proof_error}, indent=2, sort_keys=True)
        )

    if result.roundtrip_proof_report is not None:
        (report_dir / "roundtrip-proof.json").write_text(
            json.dumps(result.roundtrip_proof_report.to_dict(), indent=2, sort_keys=True)
        )
    elif result.roundtrip_proof_error is not None:
        (report_dir / "roundtrip-proof.json").write_text(
            json.dumps({"error": result.roundtrip_proof_error}, indent=2, sort_keys=True)
        )

    from polylogue.showcase.qa_report import (
        generate_qa_markdown,
        generate_qa_session,
    )
    from polylogue.showcase.report_files import save_reports

    if result.showcase_result:
        save_reports(result.showcase_result)

    qa_session = generate_qa_session(result)
    (report_dir / "qa-session.json").write_text(
        json.dumps(qa_session, indent=2, sort_keys=True)
    )
    (report_dir / "invariant-checks.json").write_text(
        json.dumps(qa_session["invariants"]["checks"], indent=2, sort_keys=True)
    )
    (report_dir / "qa-session.md").write_text(generate_qa_markdown(result))


def format_qa_summary(result: QAResult) -> str:
    """Format a human-readable QA session summary."""
    from polylogue.showcase.qa_report import generate_qa_summary

    return generate_qa_summary(result)


__all__ = [
    "QAResult",
    "format_qa_summary",
    "run_qa_session",
]
