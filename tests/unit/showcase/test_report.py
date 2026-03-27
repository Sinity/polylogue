"""Property-based tests for showcase report generation.

Laws that hold regardless of exercise content:
- generate_showcase_session always returns stable schema fields
- JSON output is always valid JSON
- summary counts are internally consistent
- write_showcase_session writes exactly one file with predictable name prefix
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.schemas.audit import AuditReport
from polylogue.schemas.verification_models import ArtifactProofReport, ProviderArtifactProof
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.qa_report import (
    generate_qa_markdown,
    generate_qa_session,
    generate_qa_summary,
)
from polylogue.showcase.qa_runner import QAResult
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult
from polylogue.showcase.showcase_report import (
    generate_json_report,
    generate_showcase_session,
    write_showcase_session,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_exercise(name: str = "ex", group: str = "structural", tier: int = 1):
    ex = MagicMock()
    ex.name = name
    ex.group = group
    ex.tier = tier
    ex.description = "A test exercise"
    ex.args = []
    ex.output_ext = ".txt"
    return ex


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
@settings(max_examples=50)
def test_qa_session_always_has_required_keys(n_passed, n_failed, n_skipped):
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
@settings(max_examples=50)
def test_qa_session_summary_counts_consistent(n_passed, n_failed, n_skipped):
    """Summary passed+failed+skipped must equal total and match exercise list."""
    results = (
        [_make_result(passed=True) for _ in range(n_passed)]
        + [_make_result(passed=False, error="err") for _ in range(n_failed)]
        + [_make_result(skipped=True) for _ in range(n_skipped)]
    )
    session = generate_showcase_session(_make_showcase(results))
    s = session["summary"]

    total = n_passed + n_failed + n_skipped
    assert s["total"] == total
    assert s["passed"] == n_passed
    assert s["failed"] == n_failed
    assert s["skipped"] == n_skipped
    assert len(session["exercises"]) == total


# ---------------------------------------------------------------------------
# Law 3: generate_showcase_session output is valid JSON when serialized
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=0, max_value=30))
@settings(max_examples=40)
def test_qa_session_serializes_to_valid_json(n):
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
@settings(max_examples=20)
def test_qa_session_schema_version_is_always_one(n):
    """schema_version field is always 1 regardless of result set."""
    results = [_make_result() for _ in range(n)]
    session = generate_showcase_session(_make_showcase(results))
    assert session["schema_version"] == 1


# ---------------------------------------------------------------------------
# Law 5: generate_json_report output is valid JSON
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=0, max_value=20))
@settings(max_examples=40)
def test_generate_json_report_always_valid_json(n):
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


def test_write_qa_session_creates_one_file(tmp_path):
    """write_showcase_session writes exactly one JSON file to the audit dir."""
    results = [_make_result(passed=True), _make_result(passed=False, error="boom")]
    sr = _make_showcase(results)

    written = write_showcase_session(sr, tmp_path / "audit")
    assert written.exists()
    assert written.suffix == ".json"
    data = json.loads(written.read_text())
    assert data["summary"]["total"] == 2
    assert data["summary"]["failed"] == 1


def test_write_qa_session_creates_audit_dir_if_missing(tmp_path):
    """write_showcase_session creates the audit directory if it does not exist."""
    audit_dir = tmp_path / "deep" / "nested" / "audit"
    assert not audit_dir.exists()
    write_showcase_session(_make_showcase([_make_result()]), audit_dir)
    assert audit_dir.exists()
    assert any(audit_dir.iterdir())


# ---------------------------------------------------------------------------
# Law 7: tier field is preserved in each exercise entry
# ---------------------------------------------------------------------------


def test_qa_session_exercise_tier_preserved():
    """Each exercise entry in qa_session carries the tier from the Exercise."""
    ex1 = _make_exercise(name="e1", tier=1)
    ex2 = _make_exercise(name="e2", tier=2)
    ex3 = _make_exercise(name="e3", tier=3)

    results = [
        ExerciseResult(exercise=ex1, passed=True, exit_code=0, output="", duration_ms=1.0),
        ExerciseResult(exercise=ex2, passed=True, exit_code=0, output="", duration_ms=2.0),
        ExerciseResult(exercise=ex3, passed=False, exit_code=1, output="", error="bad", duration_ms=3.0),
    ]
    session = generate_showcase_session(_make_showcase(results))
    tiers = {e["name"]: e["tier"] for e in session["exercises"]}
    assert tiers == {"e1": 1, "e2": 2, "e3": 3}


def test_full_qa_session_contains_composed_stage_payloads():
    """generate_qa_session preserves audit/showcase/invariant truth in one payload."""
    showcase = _make_showcase([_make_result(passed=True), _make_result(passed=False, error="boom")])
    audit = AuditReport(checks=[
        OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
    ])
    qa_result = QAResult(
        audit_report=audit,
        proof_report=ArtifactProofReport(
            providers={
                "chatgpt": ProviderArtifactProof(
                    provider="chatgpt",
                    total_records=2,
                    contract_backed_records=1,
                    unsupported_parseable_records=1,
                    package_versions={"v1": 1},
                    element_kinds={"conversation_document": 1},
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

    session = generate_qa_session(qa_result)

    assert session["audit"]["status"] == "ok"
    assert session["audit"]["report"]["summary"] == {"passed": 1, "warned": 0, "failed": 0}
    assert session["proof"]["status"] == "error"
    assert session["proof"]["report"]["summary"]["contract_backed_records"] == 1
    assert session["proof"]["report"]["summary"]["unsupported_parseable_records"] == 1
    assert session["proof"]["report"]["summary"]["package_versions"] == {"v1": 1}
    assert session["proof"]["report"]["summary"]["element_kinds"] == {"conversation_document": 1}
    assert session["showcase"]["summary"] == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "total_duration_ms": 84.0,
    }
    assert session["invariants"]["summary"] == {"passed": 1, "failed": 1, "skipped": 0}
    assert session["overall_status"] == "error"


def test_generate_qa_summary_reports_stage_statuses():
    """generate_qa_summary renders the composed session instead of hand-built counts."""
    qa_result = QAResult(
        audit_report=AuditReport(checks=[
            OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
        ]),
        proof_report=ArtifactProofReport(
            providers={
                "chatgpt": ProviderArtifactProof(
                    provider="chatgpt",
                    total_records=1,
                    contract_backed_records=1,
                    package_versions={"v1": 1},
                    element_kinds={"conversation_document": 1},
                    resolution_reasons={"exact_structure": 1},
                )
            },
            total_records=1,
        ),
        exercises_skipped=True,
        invariants_skipped=True,
    )

    summary = generate_qa_summary(qa_result)

    assert "Schema Audit: PASS" in summary
    assert "Artifact Proof: contract_backed=1" in summary
    assert "Packages: v1=1" in summary
    assert "Elements: conversation_document=1" in summary
    assert "Exercises: SKIPPED" in summary
    assert "Invariants: SKIPPED" in summary


def test_generate_qa_markdown_includes_artifact_proof_section():
    qa_result = QAResult(
        audit_report=AuditReport(checks=[
            OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
        ]),
        proof_report=ArtifactProofReport(
            providers={
                "claude-code": ProviderArtifactProof(
                    provider="claude-code",
                    total_records=2,
                    recognized_non_parseable_records=1,
                    unsupported_parseable_records=1,
                    package_versions={"v4": 1},
                    element_kinds={"subagent_conversation_stream": 1},
                    resolution_reasons={"bundle_scope": 1},
                    linked_sidecars=1,
                    subagent_streams=1,
                    streams_with_sidecars=1,
                )
            },
            total_records=2,
        ),
        exercises_skipped=True,
        invariants_skipped=True,
    )

    markdown = generate_qa_markdown(qa_result)

    assert "## Artifact Proof" in markdown
    assert "| Unsupported parseable | 1 |" in markdown
    assert "| v4 | 1 |" in markdown
    assert "| subagent_conversation_stream | 1 |" in markdown
    assert "| bundle_scope | 1 |" in markdown
    assert "| claude-code | 2 | 0 | 1 | 1 | 0 | 0 |" in markdown
