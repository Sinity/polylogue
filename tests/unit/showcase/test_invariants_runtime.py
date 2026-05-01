# mypy: disable-error-code="union-attr"

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from polylogue.core.outcomes import OutcomeStatus
from polylogue.scenarios import AssertionSpec, polylogue_execution
from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.invariants import (
    SHOWCASE_INVARIANTS,
    SKIP,
    Invariant,
    InvariantResult,
    _check_clean_stderr,
    _check_exit_code,
    _check_json_valid,
    _check_nonempty_output,
    check_invariants,
    format_invariant_summary,
)
from polylogue.showcase.runner import ExerciseResult


def _make_exercise(
    *args: str,
    output_ext: str = ".txt",
    writes: bool = False,
    assertion: AssertionSpec | None = None,
) -> Exercise:
    return Exercise(
        name="exercise",
        group="query-read",
        description="exercise",
        execution=polylogue_execution(*args),
        output_ext=output_ext,
        writes=writes,
        assertion=assertion or AssertionSpec(),
    )


def _make_result(
    *,
    args: tuple[str, ...] = ("stats",),
    output: str = "ok",
    exit_code: int = 0,
    output_ext: str = ".txt",
    writes: bool = False,
    assertion: AssertionSpec | None = None,
    skipped: bool = False,
) -> ExerciseResult:
    return ExerciseResult(
        exercise=_make_exercise(*args, output_ext=output_ext, writes=writes, assertion=assertion),
        passed=not skipped,
        exit_code=exit_code,
        output=output,
        duration_ms=5.0,
        skipped=skipped,
        skip_reason="skip" if skipped else None,
    )


def test_json_valid_skips_non_json_and_validates_json_documents() -> None:
    assert _check_json_valid(_make_result(args=("stats",), output="plain")) == SKIP
    assert _check_json_valid(_make_result(args=("stats", "--json"), output='{"ok": true}', output_ext=".json")) is None
    assert _check_json_valid(_make_result(args=("stats", "--json"), output="{", output_ext=".json")).startswith(
        "Invalid JSON:"
    )


def test_json_valid_handles_jsonl_line_errors() -> None:
    ok_result = _make_result(args=("query", "-f", "json"), output='{"a": 1}\n\n{"b": 2}\n', output_ext=".jsonl")
    bad_result = _make_result(args=("query", "-f", "json"), output='{"a": 1}\nnot-json\n', output_ext=".jsonl")

    assert _check_json_valid(ok_result) is None
    assert _check_json_valid(bad_result).startswith("Invalid JSON on line 2:")


def test_exit_code_invariant_uses_assertion_spec() -> None:
    delegated = _make_result(args=("stats",), assertion=AssertionSpec(exit_code=None))
    matching = _make_result(args=("stats",), exit_code=2, assertion=AssertionSpec(exit_code=2))
    failing = _make_result(args=("stats",), exit_code=1, assertion=AssertionSpec(exit_code=0))

    assert _check_exit_code(delegated) == SKIP
    assert _check_exit_code(matching) is None
    assert _check_exit_code(failing) == "exit code 1, expected 0"


def test_clean_stderr_and_nonempty_output_skip_expected_cases() -> None:
    assert _check_clean_stderr(_make_result(args=("run",), writes=True)) == SKIP
    assert _check_clean_stderr(_make_result(args=("stats",))) is None

    assert _check_nonempty_output(_make_result(args=("stats", "--count"), output="")) == SKIP
    assert _check_nonempty_output(_make_result(args=("--help",), output="")) == SKIP
    assert _check_nonempty_output(_make_result(args=("--version",), output="")) == SKIP
    assert _check_nonempty_output(_make_result(args=("stats",), writes=True, output="")) == SKIP
    assert _check_nonempty_output(_make_result(args=("stats",), output="")) == "Empty output for read command"
    assert _check_nonempty_output(_make_result(args=("stats",), output="not empty")) is None


def test_invariant_to_claim_emits_showcase_claim_metadata() -> None:
    claim = SHOWCASE_INVARIANTS[0].to_claim()

    assert claim.id == "showcase.invariant.json_valid"
    assert claim.breaker is not None
    assert claim.breaker.issue == "#192"
    assert claim.breaker.command == ("devtools", "lab-scenario", "run", "archive-smoke", "--tier", "0")


def test_check_invariants_collects_ok_skip_error_and_crash() -> None:
    result = _make_result(args=("stats",), output="payload")
    skipped_result = replace(result, skipped=True, skip_reason="skip")
    invariants = [
        Invariant(name="ok", description="ok", check=lambda _result: None),
        Invariant(name="skip", description="skip", check=lambda _result: SKIP),
        Invariant(name="error", description="error", check=lambda _result: "broken"),
        Invariant(name="crash", description="crash", check=lambda _result: (_ for _ in ()).throw(RuntimeError("boom"))),
    ]

    with patch("polylogue.showcase.invariants.SHOWCASE_INVARIANTS", invariants):
        results = check_invariants([result, skipped_result])

    assert [(item.invariant_name, item.status) for item in results] == [
        ("ok", OutcomeStatus.OK),
        ("skip", OutcomeStatus.SKIP),
        ("error", OutcomeStatus.ERROR),
        ("crash", OutcomeStatus.ERROR),
    ]
    assert results[-1].error == "invariant check crashed: boom"


def test_format_invariant_summary_lists_failures() -> None:
    summary = format_invariant_summary(
        [
            InvariantResult("json_valid", "stats", OutcomeStatus.OK),
            InvariantResult("nonempty_output", "stats", OutcomeStatus.ERROR, error="empty"),
            InvariantResult("exit_code", "stats", OutcomeStatus.SKIP),
        ]
    )

    assert "Invariant Checks: 1 pass, 1 fail, 1 skip" in summary
    assert "Failures:" in summary
    assert "nonempty_output @ stats: empty" in summary
