from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import regression_capture
from devtools.regression_cases import RegressionCase
from polylogue.core.json import JSONDocument


def _probe_summary() -> JSONDocument:
    return {
        "probe": {"input_mode": "archive-subset", "stage": "parse"},
        "paths": {"workdir": "/tmp/polylogue-probe"},
        "provenance": {"git_commit": "abc123", "worktree_dirty": False},
        "result": {"ok": False, "error": "parse drift"},
        "run_payload": {"metrics": {"total_duration_ms": 12.5}},
        "db_stats": {"conversations": 0, "raw_conversations": 1},
        "raw_fanout": [{"raw_id": "raw-1", "parse_error": "boom"}],
        "budgets": {"ok": False, "violations": ["rss"]},
    }


def test_regression_capture_writes_probe_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "probe.json"
    output_dir = tmp_path / "cases"
    input_path.write_text(json.dumps(_probe_summary()), encoding="utf-8")

    assert (
        regression_capture.main(
            [
                "--input",
                str(input_path),
                "--name",
                "Parse Drift",
                "--output-dir",
                str(output_dir),
                "--tag",
                "live",
                "--note",
                "captured from probe",
            ]
        )
        == 0
    )

    output_path = Path(capsys.readouterr().out.strip())
    case = RegressionCase.read(output_path)
    assert output_path.parent == output_dir
    assert case.name == "Parse Drift"
    assert case.source == "pipeline-probe"
    assert case.tags == ("live",)
    assert case.notes == ("captured from probe",)
    assert case.summary["result"] == {"ok": False, "error": "parse drift"}


def test_regression_capture_json_output_includes_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "probe.json"
    output_dir = tmp_path / "cases"
    input_path.write_text(json.dumps(_probe_summary()), encoding="utf-8")

    assert (
        regression_capture.main(
            [
                "--input",
                str(input_path),
                "--name",
                "Parse Drift",
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "pipeline-probe"
    assert Path(payload["path"]).exists()
    assert payload["summary"]["db_stats"] == {"conversations": 0, "raw_conversations": 1}
