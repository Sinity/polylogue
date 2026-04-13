from __future__ import annotations

from polylogue.scenarios import AssertionSpec


def test_assertion_spec_validates_process_output() -> None:
    assertion = AssertionSpec(
        exit_code=0,
        stdout_contains=("polylogue",),
        stdout_not_contains=("traceback",),
        stdout_min_lines=1,
    )

    assert assertion.validate_process("polylogue ok\n", 0) is None
    assert assertion.validate_process("nope\n", 0) == "output missing 'polylogue'"


def test_assertion_spec_validates_json_contracts() -> None:
    assertion = AssertionSpec(stdout_is_valid_json=True)

    assert assertion.validate_process('{"ok": true}', 0) is None
    assert assertion.validate_process("not-json", 0).startswith("invalid JSON:")


def test_assertion_spec_resolves_benchmark_thresholds() -> None:
    assertion = AssertionSpec(benchmark_warn_pct=12.5, benchmark_fail_pct=25.0)

    assert assertion.resolved_benchmark_warn_pct() == 12.5
    assert assertion.resolved_benchmark_fail_pct() == 25.0
