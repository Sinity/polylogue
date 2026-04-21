"""Structured showcase and QA report records.

These records sit between the runtime models and the JSON/Markdown surfaces so
the reporting layer can work with one typed contract instead of nested ad-hoc
payload dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.scenarios import PayloadDict, ScenarioMetadata
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.showcase_runner_models import (
    ExerciseResult,
    ShowcaseGroupCounts,
    ShowcaseResult,
    ShowcaseSummary,
)


@dataclass(frozen=True, slots=True)
class InvariantCheckRecord:
    """Serializable view of one invariant outcome."""

    invariant: str
    exercise: str
    status: str
    error: str | None = None

    @classmethod
    def from_result(cls, result: InvariantResult) -> InvariantCheckRecord:
        return cls(
            invariant=result.invariant_name,
            exercise=result.exercise_name,
            status=result.status.value,
            error=result.error,
        )

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {
            "invariant": self.invariant,
            "exercise": self.exercise,
            "status": self.status,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class InvariantSummary:
    """Aggregate QA invariant counts."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0

    @classmethod
    def from_results(cls, results: list[InvariantResult]) -> InvariantSummary:
        passed = 0
        failed = 0
        skipped = 0
        for result in results:
            if result.status.value == "ok":
                passed += 1
            elif result.status.value == "error":
                failed += 1
            else:
                skipped += 1
        return cls(passed=passed, failed=failed, skipped=skipped)

    def to_payload(self) -> dict[str, int]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
        }


@dataclass(frozen=True, slots=True)
class ShowcaseExerciseRecord:
    """Serializable view of one showcase exercise result."""

    name: str
    group: str
    passed: bool
    exit_code: int
    duration_ms: float
    metadata: ScenarioMetadata
    corpus_specs: tuple[PayloadDict, ...] = ()
    description: str | None = None
    tier: int | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None

    @classmethod
    def from_result(
        cls,
        result: ExerciseResult,
        *,
        include_description: bool = True,
        include_tier: bool = False,
    ) -> ShowcaseExerciseRecord:
        return cls(
            name=result.exercise.name,
            group=result.exercise.group,
            passed=result.passed,
            exit_code=result.exit_code,
            duration_ms=round(result.duration_ms, 1),
            metadata=ScenarioMetadata.from_object(result.exercise),
            corpus_specs=tuple(spec.to_payload() for spec in result.exercise.corpus_specs),
            description=result.exercise.description if include_description else None,
            tier=result.exercise.tier if include_tier else None,
            skipped=result.skipped,
            skip_reason=result.skip_reason,
            error=result.error,
        )

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {
            "name": self.name,
            "group": self.group,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.tier is not None:
            payload["tier"] = self.tier
        payload.update(self.metadata.to_payload())
        if self.corpus_specs:
            payload["corpus_specs"] = [dict(spec) for spec in self.corpus_specs]
        if self.skipped:
            payload["skipped"] = True
            payload["skip_reason"] = self.skip_reason
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class ShowcaseSessionRecord:
    """Serializable showcase run session."""

    timestamp: str
    summary: ShowcaseSummary
    group_counts: dict[str, ShowcaseGroupCounts]
    exercises: tuple[ShowcaseExerciseRecord, ...]
    schema_version: int = 1

    @classmethod
    def from_result(cls, result: ShowcaseResult, *, timestamp: str) -> ShowcaseSessionRecord:
        return cls(
            timestamp=timestamp,
            summary=result.summary(),
            group_counts=result.group_counts(),
            exercises=tuple(
                ShowcaseExerciseRecord.from_result(report, include_description=False, include_tier=True)
                for report in result.results
            ),
        )

    def to_payload(self) -> PayloadDict:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "summary": self.summary.to_payload(),
            "group_counts": {group: counts.to_payload() for group, counts in self.group_counts.items()},
            "exercises": [exercise.to_payload() for exercise in self.exercises],
        }


@dataclass(frozen=True, slots=True)
class QAAuditRecord:
    status: str
    skipped: bool
    report: Mapping[str, object] | None = None
    error: str | None = None

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {"status": self.status, "skipped": self.skipped}
        if self.report is not None:
            payload["report"] = dict(self.report)
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class QAProofRecord:
    status: str
    skipped: bool
    report: Mapping[str, object] | None = None
    error: str | None = None

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {"status": self.status, "skipped": self.skipped}
        if self.report is not None:
            payload["report"] = dict(self.report)
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class QAShowcaseRecord:
    status: str
    skipped: bool
    summary: ShowcaseSummary | None
    group_counts: dict[str, ShowcaseGroupCounts]
    exercises: tuple[ShowcaseExerciseRecord, ...]

    def to_payload(self) -> PayloadDict:
        return {
            "status": self.status,
            "skipped": self.skipped,
            "summary": self.summary.to_payload() if self.summary is not None else None,
            "group_counts": {group: counts.to_payload() for group, counts in self.group_counts.items()},
            "exercises": [exercise.to_payload() for exercise in self.exercises],
        }


@dataclass(frozen=True, slots=True)
class QAInvariantRecord:
    status: str
    skipped: bool
    summary: InvariantSummary
    checks: tuple[InvariantCheckRecord, ...]

    def to_payload(self) -> PayloadDict:
        return {
            "status": self.status,
            "skipped": self.skipped,
            "summary": self.summary.to_payload(),
            "checks": [check.to_payload() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class QASessionRecord:
    """Serializable QA run session."""

    timestamp: str
    audit: QAAuditRecord
    proof: QAProofRecord
    showcase: QAShowcaseRecord
    invariants: QAInvariantRecord
    overall_status: str
    overall_passed: bool
    report_dir: str | None = None
    schema_version: int = 1

    @classmethod
    def from_result(
        cls,
        result: QAResult,
        *,
        timestamp: str,
        showcase_session: ShowcaseSessionRecord | None,
    ) -> QASessionRecord:
        invariant_checks = tuple(InvariantCheckRecord.from_result(item) for item in result.invariant_results)
        invariant_summary = InvariantSummary.from_results(result.invariant_results)
        return cls(
            timestamp=timestamp,
            audit=QAAuditRecord(
                status=result.audit_status.value,
                skipped=result.audit_skipped,
                report=result.audit_report.to_json() if result.audit_report is not None else None,
                error=result.audit_error,
            ),
            proof=QAProofRecord(
                status=result.proof_status.value,
                skipped=result.proof_skipped,
                report=result.proof_report.to_dict() if result.proof_report is not None else None,
                error=result.proof_error,
            ),
            showcase=QAShowcaseRecord(
                status=result.showcase_status.value,
                skipped=result.exercises_skipped,
                summary=showcase_session.summary if showcase_session is not None else None,
                group_counts=showcase_session.group_counts if showcase_session is not None else {},
                exercises=showcase_session.exercises if showcase_session is not None else (),
            ),
            invariants=QAInvariantRecord(
                status=result.invariant_status.value,
                skipped=result.invariants_skipped or result.showcase_result is None,
                summary=invariant_summary,
                checks=invariant_checks,
            ),
            overall_status=result.overall_status.value,
            overall_passed=result.all_passed,
            report_dir=str(result.report_dir) if result.report_dir is not None else None,
        )

    def to_payload(self) -> PayloadDict:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "audit": self.audit.to_payload(),
            "proof": self.proof.to_payload(),
            "showcase": self.showcase.to_payload(),
            "invariants": self.invariants.to_payload(),
            "overall_status": self.overall_status,
            "overall_passed": self.overall_passed,
            "report_dir": self.report_dir,
        }


def canonical_showcase_session(result: ShowcaseResult, *, timestamp: str) -> ShowcaseSessionRecord:
    """Build the typed showcase session record for one result set."""
    return ShowcaseSessionRecord.from_result(result, timestamp=timestamp)


def canonical_qa_session(
    result: QAResult,
    *,
    timestamp: str,
    showcase_session: ShowcaseSessionRecord | None,
) -> QASessionRecord:
    """Build the typed QA session record for one result set."""
    return QASessionRecord.from_result(result, timestamp=timestamp, showcase_session=showcase_session)


__all__ = [
    "QAAuditRecord",
    "QAInvariantRecord",
    "QAProofRecord",
    "QASessionRecord",
    "QAShowcaseRecord",
    "ShowcaseExerciseRecord",
    "ShowcaseSessionRecord",
    "canonical_qa_session",
    "canonical_showcase_session",
    "InvariantCheckRecord",
    "InvariantSummary",
]
