from __future__ import annotations

from polylogue.scenarios import AssertionClass, AssertionSpec


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
    error = assertion.validate_process("not-json", 0)
    assert error is not None
    assert error.startswith("invalid JSON:")


def test_assertion_spec_resolves_benchmark_thresholds() -> None:
    assertion = AssertionSpec(benchmark_warn_pct=12.5, benchmark_fail_pct=25.0)
    assert assertion.resolved_benchmark_warn_pct() == 12.5
    assert assertion.resolved_benchmark_fail_pct() == 25.0


def test_default_assertion_classifies_as_smoke_process() -> None:
    assert AssertionSpec().classification is AssertionClass.SMOKE_PROCESS
    assert AssertionSpec(exit_code=0).classification is AssertionClass.SMOKE_PROCESS


def test_stdout_checks_auto_classify_as_semantic_output() -> None:
    assert AssertionSpec(stdout_contains=("ok",)).classification is AssertionClass.SEMANTIC_OUTPUT
    assert AssertionSpec(stdout_not_contains=("err",)).classification is AssertionClass.SEMANTIC_OUTPUT
    assert AssertionSpec(stdout_is_valid_json=True).classification is AssertionClass.SEMANTIC_OUTPUT
    assert AssertionSpec(stdout_min_lines=5).classification is AssertionClass.SMOKE_PROCESS


def test_benchmark_thresholds_auto_classify_as_runtime_budget() -> None:
    assert AssertionSpec(benchmark_warn_pct=10.0).classification is AssertionClass.RUNTIME_BUDGET
    assert AssertionSpec(benchmark_fail_pct=25.0).classification is AssertionClass.RUNTIME_BUDGET
    assert (
        AssertionSpec(benchmark_warn_pct=10.0, stdout_contains=("ok",)).classification is AssertionClass.RUNTIME_BUDGET
    )


def test_classification_override_preserved() -> None:
    assertion = AssertionSpec(
        stdout_contains=("semantic",),
        classification_override=AssertionClass.METADATA_SPEC,
    )
    assert assertion.classification is AssertionClass.METADATA_SPEC
    assert assertion.validate_process("semantic check\n", 0) is None


def test_classification_roundtrips_through_payload() -> None:
    original = AssertionSpec(
        stdout_contains=("ok",),
        classification_override=AssertionClass.LIVE_OBSERVABILITY,
    )
    payload = original.to_payload()
    assert payload["classification"] == "live-observability"
    rehydrated = AssertionSpec.from_payload(payload)
    assert rehydrated.classification is AssertionClass.LIVE_OBSERVABILITY
    assert rehydrated.stdout_contains == ("ok",)


def test_smoke_process_does_not_serialize_classification() -> None:
    assertion = AssertionSpec()
    payload = assertion.to_payload()
    assert "classification" not in payload
    assert AssertionSpec.from_payload(payload).classification is AssertionClass.SMOKE_PROCESS
