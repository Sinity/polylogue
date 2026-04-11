"""Typed QA orchestration result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.runner import ShowcaseResult

if TYPE_CHECKING:
    from polylogue.schemas.audit_models import AuditReport
    from polylogue.schemas.verification_models import ArtifactProofReport


@dataclass
class QAResult:
    """Complete QA session result."""

    audit_report: AuditReport | None = None
    audit_error: str | None = None
    audit_skipped: bool = False
    proof_report: ArtifactProofReport | None = None
    proof_error: str | None = None
    proof_skipped: bool = False
    showcase_result: ShowcaseResult | None = None
    exercises_skipped: bool = False
    invariant_results: list[InvariantResult] = field(default_factory=list)
    invariants_skipped: bool = False
    report_dir: Path | None = None

    @property
    def audit_status(self) -> OutcomeStatus:
        if self.audit_skipped:
            return OutcomeStatus.SKIP
        if self.audit_error is not None or self.audit_report is None:
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK if self.audit_report.all_passed else OutcomeStatus.ERROR

    @property
    def audit_passed(self) -> bool:
        return self.audit_status is OutcomeStatus.OK

    @property
    def proof_status(self) -> OutcomeStatus:
        if self.proof_skipped:
            return OutcomeStatus.SKIP
        if self.proof_error is not None or self.proof_report is None:
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK if self.proof_report.is_clean else OutcomeStatus.ERROR

    @property
    def showcase_status(self) -> OutcomeStatus:
        if self.exercises_skipped:
            return OutcomeStatus.SKIP
        if self.showcase_result is None:
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK if self.showcase_result.failed == 0 else OutcomeStatus.ERROR

    @property
    def invariant_status(self) -> OutcomeStatus:
        if self.invariants_skipped or self.showcase_result is None:
            return OutcomeStatus.SKIP
        if any(result.status is OutcomeStatus.ERROR for result in self.invariant_results):
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK

    @property
    def overall_status(self) -> OutcomeStatus:
        if any(
            stage is OutcomeStatus.ERROR
            for stage in (
                self.audit_status,
                self.proof_status,
                self.showcase_status,
                self.invariant_status,
            )
        ):
            return OutcomeStatus.ERROR
        return OutcomeStatus.OK

    @property
    def all_passed(self) -> bool:
        return self.overall_status is OutcomeStatus.OK


__all__ = ["QAResult"]
