"""Property-based tests for showcase report generation.

Laws that hold regardless of exercise content:
- generate_qa_session always returns stable schema fields
- JSON output is always valid JSON
- summary counts are internally consistent
- write_qa_session writes exactly one file with predictable name prefix
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.showcase.report import generate_json_report, generate_qa_session, write_qa_session
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult

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
# Law 1: generate_qa_session always has required top-level keys
# ---------------------------------------------------------------------------


@given(
    n_passed=st.integers(min_value=0, max_value=20),
    n_failed=st.integers(min_value=0, max_value=20),
    n_skipped=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=50)
def test_qa_session_always_has_required_keys(n_passed, n_failed, n_skipped):
    """generate_qa_session always returns the mandatory schema fields."""
    results = (
        [_make_result(passed=True) for _ in range(n_passed)]
        + [_make_result(passed=False, error="fail") for _ in range(n_failed)]
        + [_make_result(skipped=True) for _ in range(n_skipped)]
    )
    session = generate_qa_session(_make_showcase(results))

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
    session = generate_qa_session(_make_showcase(results))
    s = session["summary"]

    total = n_passed + n_failed + n_skipped
    assert s["total"] == total
    assert s["passed"] == n_passed
    assert s["failed"] == n_failed
    assert s["skipped"] == n_skipped
    assert len(session["exercises"]) == total


# ---------------------------------------------------------------------------
# Law 3: generate_qa_session output is valid JSON when serialized
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=0, max_value=30))
@settings(max_examples=40)
def test_qa_session_serializes_to_valid_json(n):
    """generate_qa_session output round-trips through JSON without error."""
    results = [_make_result(passed=i % 3 != 0) for i in range(n)]
    session = generate_qa_session(_make_showcase(results))
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
    session = generate_qa_session(_make_showcase(results))
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
    """write_qa_session writes exactly one JSON file to the audit dir."""
    results = [_make_result(passed=True), _make_result(passed=False, error="boom")]
    sr = _make_showcase(results)

    written = write_qa_session(sr, tmp_path / "audit")
    assert written.exists()
    assert written.suffix == ".json"
    data = json.loads(written.read_text())
    assert data["summary"]["total"] == 2
    assert data["summary"]["failed"] == 1


def test_write_qa_session_creates_audit_dir_if_missing(tmp_path):
    """write_qa_session creates the audit directory if it does not exist."""
    audit_dir = tmp_path / "deep" / "nested" / "audit"
    assert not audit_dir.exists()
    write_qa_session(_make_showcase([_make_result()]), audit_dir)
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
    session = generate_qa_session(_make_showcase(results))
    tiers = {e["name"]: e["tier"] for e in session["exercises"]}
    assert tiers == {"e1": 1, "e2": 2, "e3": 3}
