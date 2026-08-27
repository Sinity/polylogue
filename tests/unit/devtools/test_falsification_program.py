from __future__ import annotations

from typing import cast

import pytest

from devtools.falsification_program import _query_oracle, build_report, main


def test_query_oracle_has_independent_reference_and_mutation_control() -> None:
    result = _query_oracle()
    mutation = _query_oracle(suppress_next_cursor=True)
    assert result["passed"] is True
    assert result["rows"] == 11
    assert mutation["passed"] is False
    assert mutation["failure"] == "cursor_missing_before_end"


def test_falsification_report_has_four_gated_slices() -> None:
    report = build_report()
    slices = cast(dict[str, dict[str, object]], report["slices"])
    gate = cast(dict[str, object], report["gate"])
    resources = cast(dict[str, object], report["resource_measurements"])
    assert report["schema_version"] == 1
    assert set(slices) == {"safety", "semantics", "query", "interaction"}
    assert gate["passed"] is False
    assert slices["safety"]["status"] == "not_run"
    assert slices["interaction"]["status"] == "blocked"
    assert report["blind_spots"]
    assert resources["network"] == "none"


def test_cli_requires_safety_execution() -> None:
    with pytest.raises(SystemExit, match="2"):
        main([])
