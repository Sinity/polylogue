"""Property-based tests for showcase report generation.

Laws that hold regardless of exercise content:
- generate_showcase_session always returns stable schema fields
- JSON output is always valid JSON
- summary counts are internally consistent
- write_showcase_session writes exactly one file with predictable name prefix
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.insights.authored_payloads import require_payload_mapping
from polylogue.scenarios import CorpusSpec, polylogue_execution
from polylogue.schemas.audit.models import AuditReport
from polylogue.schemas.validation.models import ArtifactCoverageReport, ProviderArtifactCoverage
from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.qa_runner import QAResult
from polylogue.showcase.qa_session_payload import build_qa_session_record
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult
from polylogue.showcase.showcase_report_payloads import (
    build_showcase_session_record,
    generate_json_report,
    generate_showcase_session,
    write_showcase_session,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_exercise(name: str = "ex", group: str = "structural", tier: int = 1) -> Exercise:
    return Exercise(name=name, group=group, tier=tier, description="A test exercise", output_ext=".txt")


def _make_result(passed: bool = True, skipped: bool = False, error: str | None = None) -> ExerciseResult:
    return ExerciseResult(
        exercise=_make_exercise(),
        passed=passed,
        exit_code=0 if passed else 1,
        output="output text",
        error=error,
        duration_ms=42.0,
        skipped=skipped,
        skip_reason="dep missing" if skipped else None,
    )


def _make_showcase(results: list[ExerciseResult]) -> ShowcaseResult:
    sr = ShowcaseResult()
    sr.results = results
    sr.total_duration_ms = sum(r.duration_ms for r in results)
    return sr


# ---------------------------------------------------------------------------
# Law 1: generate_showcase_session always has required top-level keys
# ---------------------------------------------------------------------------


@given(
    n_passed=st.integers(min_value=0, max_value=20),
    n_failed=st.integers(min_value=0, max_value=20),
    n_skipped=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=50, deadline=None)
def test_qa_session_always_has_required_keys(n_passed: int, n_failed: int, n_skipped: int) -> None:
    """generate_showcase_session always returns the mandatory schema fields."""
    results = (
        [_make_result(passed=True) for _ in range(n_passed)]
        + [_make_result(passed=False, error="fail") for _ in range(n_failed)]
        + [_make_result(skipped=True) for _ in range(n_skipped)]
    )
    session = generate_showcase_session(_make_showcase(results))

    assert "schema_version" in session
    assert "timestamp" in session
    assert "summary" in session
    assert "exercises" in session
    assert "group_counts" in session


# ---------------------------------------------------------------------------
# Law 2: summary counts equal exercise list lengths
# ---------------------------------------------------------------------------


@given(
    n_passed=st.integers(min_value=0, max_value=15),
    n_failed=st.integers(min_value=0, max_value=15),
    n_skipped=st.integers(min_value=0, max_value=15),
)
@settings(max_examples=50, deadline=None)
def test_qa_session_summary_counts_consistent(n_passed: int, n_failed: int, n_skipped: int) -> None:
    """Summary passed+failed+skipped must equal total and match exercise list."""
    results = (
        [_make_result(passed=True) for _ in range(n_passed)]
        + [_make_result(passed=False, error="err") for _ in range(n_failed)]
        + [_make_result(skipped=True) for _ in range(n_skipped)]
    )
    session = build_showcase_session_record(_make_showcase(results), timestamp="2026-01-01T00:00:00Z")
    s = session.summary

    total = n_passed + n_failed + n_skipped
    assert s.total == total
    assert s.passed == n_passed
    assert s.failed == n_failed
    assert s.skipped == n_skipped
    assert len(session.exercises) == total


# ---------------------------------------------------------------------------
# Law 3: generate_showcase_session output is valid JSON when serialized
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=0, max_value=30))
@settings(max_examples=40, deadline=None)
def test_qa_session_serializes_to_valid_json(n: int) -> None:
    """generate_showcase_session output round-trips through JSON without error."""
    results = [_make_result(passed=i % 3 != 0) for i in range(n)]
    session = generate_showcase_session(_make_showcase(results))
    raw = json.dumps(session)
    loaded = json.loads(raw)
    assert loaded["summary"]["total"] == n


# ---------------------------------------------------------------------------
# Law 4: schema_version is always 1 (stable across runs)
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=0, max_value=10))
@settings(max_examples=20, deadline=None)
def test_qa_session_schema_version_is_always_one(n: int) -> None:
    """schema_version field is always 1 regardless of result set."""
    results = [_make_result() for _ in range(n)]
    session = build_showcase_session_record(_make_showcase(results), timestamp="2026-01-01T00:00:00Z")
    assert session.schema_version == 1


# ---------------------------------------------------------------------------
# Law 5: generate_json_report output is valid JSON
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=0, max_value=20))
@settings(max_examples=40, deadline=None)
def test_generate_json_report_always_valid_json(n: int) -> None:
    """generate_json_report always returns a valid JSON string."""
    results = [_make_result(passed=i % 2 == 0) for i in range(n)]
    raw = generate_json_report(_make_showcase(results))
    data = json.loads(raw)
    assert isinstance(data, dict)
    assert "total" in data
    assert "exercises" in data


# ---------------------------------------------------------------------------
# Law 6: write_qa_session writes exactly one file, JSON-parseable
# ---------------------------------------------------------------------------


def test_write_qa_session_creates_one_file(tmp_path: Path) -> None:
    """write_showcase_session writes exactly one JSON file to the audit dir."""
    results = [_make_result(passed=True), _make_result(passed=False, error="boom")]
    sr = _make_showcase(results)

    written = write_showcase_session(sr, tmp_path / "audit")
    assert written.exists()
    assert written.suffix == ".json"
    data = json.loads(written.read_text())
    assert data["summary"]["total"] == 2
    assert data["summary"]["failed"] == 1


def test_write_qa_session_creates_audit_dir_if_missing(tmp_path: Path) -> None:
    """write_showcase_session creates the audit directory if it does not exist."""
    audit_dir = tmp_path / "deep" / "nested" / "audit"
    assert not audit_dir.exists()
    write_showcase_session(_make_showcase([_make_result()]), audit_dir)
    assert audit_dir.exists()
    assert any(audit_dir.iterdir())


# ---------------------------------------------------------------------------
# Law 7: tier field is preserved in each exercise entry
# ---------------------------------------------------------------------------


def test_qa_session_exercise_tier_preserved() -> None:
    """Each exercise entry in qa_session carries the tier from the Exercise."""
    ex1 = _make_exercise(name="e1", tier=1)
    ex2 = _make_exercise(name="e2", tier=2)
    ex3 = _make_exercise(name="e3", tier=3)

    results = [
        ExerciseResult(exercise=ex1, passed=True, exit_code=0, output="", duration_ms=1.0),
        ExerciseResult(exercise=ex2, passed=True, exit_code=0, output="", duration_ms=2.0),
        ExerciseResult(exercise=ex3, passed=False, exit_code=1, output="", error="bad", duration_ms=3.0),
    ]
    session = build_showcase_session_record(_make_showcase(results), timestamp="2026-01-01T00:00:00Z")
    tiers = {exercise.name: exercise.tier for exercise in session.exercises}
    assert tiers == {"e1": 1, "e2": 2, "e3": 3}


def test_generate_json_report_preserves_exercise_corpus_specs() -> None:
    exercise = Exercise(
        name="seeded-query",
        group="query-read",
        description="Seeded query",
        execution=polylogue_execution("list", "-n", "1"),
        corpus_specs=(CorpusSpec.for_provider("chatgpt", count=2),),
    )
    result = ShowcaseResult()
    result.results = [
        ExerciseResult(
            exercise=exercise,
            passed=True,
            exit_code=0,
            output="[]",
            duration_ms=12.0,
        )
    ]

    payload = json.loads(generate_json_report(result))

    assert payload["exercises"][0]["corpus_specs"][0]["provider"] == "chatgpt"


def test_full_qa_session_contains_composed_stage_payloads() -> None:
    """generate_qa_session preserves audit/showcase/invariant truth in one payload."""
    showcase = _make_showcase([_make_result(passed=True), _make_result(passed=False, error="boom")])
    audit = AuditReport(
        checks=[
            OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
        ]
    )
    qa_result = QAResult(
        audit_report=audit,
        coverage_report=ArtifactCoverageReport(
            providers={
                "chatgpt": ProviderArtifactCoverage(
                    provider="chatgpt",
                    total_records=2,
                    contract_backed_records=1,
                    unsupported_parseable_records=1,
                    package_versions={"v1": 1},
                    element_kinds={"session_document": 1},
                    resolution_reasons={"exact_structure": 1},
                )
            },
            total_records=2,
        ),
        showcase_result=showcase,
        invariant_results=[
            InvariantResult("json_valid", "ex", OutcomeStatus.OK),
            InvariantResult("exit_code", "ex", OutcomeStatus.ERROR, error="bad exit"),
        ],
    )

    session = build_qa_session_record(
        qa_result,
        timestamp="2026-01-01T00:00:00Z",
        showcase_session=build_showcase_session_record(showcase, timestamp="2026-01-01T00:00:00Z"),
    )
    audit_report = require_payload_mapping(session.audit.report, context="audit.report")
    audit_summary = require_payload_mapping(audit_report["summary"], context="audit.summary")
    coverage_report = require_payload_mapping(session.artifact_coverage.report, context="artifact_coverage.report")
    coverage_summary = require_payload_mapping(coverage_report["summary"], context="artifact_coverage.summary")
    showcase_summary = session.showcase.summary

    assert session.audit.status == "ok"
    assert audit_summary == {"passed": 1, "warned": 0, "failed": 0}
    assert session.artifact_coverage.status == "error"
    assert coverage_summary["contract_backed_records"] == 1
    assert coverage_summary["unsupported_parseable_records"] == 1
    assert coverage_summary["package_versions"] == {"v1": 1}
    assert coverage_summary["element_kinds"] == {"session_document": 1}
    assert showcase_summary is not None
    assert showcase_summary.to_payload() == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "total_duration_ms": 84.0,
    }
    assert session.invariants.summary.to_payload() == {"passed": 1, "failed": 1, "skipped": 0}
    assert session.overall_status == "error"
