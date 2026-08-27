from __future__ import annotations

from typing import cast

from devtools.falsification_program import _query_oracle, build_report


def test_query_oracle_has_independent_reference_and_mutation_control() -> None:
    result = _query_oracle()
    assert result["passed"] is True
    assert result["mutation_caught"] is True
    assert result["rows"] == 11


def test_falsification_report_has_four_gated_slices() -> None:
    report = build_report()
    slices = cast(dict[str, dict[str, object]], report["slices"])
    gate = cast(dict[str, object], report["gate"])
    resources = cast(dict[str, object], report["resource_measurements"])
    assert report["schema_version"] == 1
    assert set(slices) == {"safety", "semantics", "query", "interaction"}
    assert isinstance(gate["passed"], bool)
    assert slices["safety"]["status"] == "not_run"
    assert report["blind_spots"]
    assert resources["network"] == "none"
