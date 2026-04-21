from __future__ import annotations

from pathlib import Path

from devtools.regression_cases import RegressionCase, RegressionCaseStore
from polylogue.lib.json import JSONDocument


def test_regression_case_roundtrips_probe_summary(tmp_path: Path) -> None:
    summary: JSONDocument = {
        "probe": {"stage": "parse"},
        "provenance": {"git_commit": "abc123"},
        "result": {"ok": False},
        "db_stats": {"raw_conversations": 1},
        "ignored": "not captured",
    }

    case = RegressionCase.from_probe_summary(name="Parse Failure", summary=summary, tags=("probe",))
    path = RegressionCaseStore(tmp_path).write(case)
    loaded = RegressionCase.read(path)

    assert loaded == case
    assert loaded.case_id.startswith("parse-failure-")
    assert loaded.tags == ("probe",)
    assert "ignored" not in loaded.summary


def test_regression_case_requires_probe_and_result_keys() -> None:
    summary: JSONDocument = {"probe": {}}
    try:
        RegressionCase.from_probe_summary(name="bad", summary=summary)
    except ValueError as exc:
        assert "probe" in str(exc)
        assert "result" in str(exc)
    else:
        raise AssertionError("expected missing result to fail")
