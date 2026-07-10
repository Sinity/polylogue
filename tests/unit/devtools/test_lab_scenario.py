"""Tests for the executable lab smoke command surface."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_module_imports() -> None:
    assert importlib.import_module("devtools.lab_scenario") is not None


def test_get_archive_smoke_checks_returns_direct_cli_cases() -> None:
    from devtools.lab_scenario import get_archive_smoke_checks

    checks = get_archive_smoke_checks()

    assert [check.name for check in checks] == [
        "help-main",
        "help-mark-candidates",
        "completions-bash",
    ]
    assert checks[0].execution.polylogue_args == ("--help",)
    assert checks[1].execution.polylogue_args == ("mark", "candidates", "--help")
    assert checks[2].execution.polylogue_args == ("config", "completions", "--shell", "bash")


def test_list_scenarios_reports_live_paths_without_baseline_counts(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import list_scenarios

    assert list_scenarios(as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    archive = next(entry for entry in payload["scenarios"] if entry["name"] == "archive-smoke")
    visual = next(entry for entry in payload["scenarios"] if entry["name"] == "reader-visual-smoke")
    storage = next(entry for entry in payload["scenarios"] if entry["name"] == "storage-correctness")
    assert archive == {
        "name": "archive-smoke",
        "kind": "cli-smoke",
        "tier_0_check_count": archive["tier_0_check_count"],
    }
    assert archive["tier_0_check_count"] > 0
    assert visual["command"]
    assert visual["artifact_count"] > 0
    assert storage == {
        "name": "storage-correctness",
        "kind": "archive-storage",
        "check_count": 3,
    }


def test_main_prints_direct_stage_summary(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import main

    def _invoke(execution: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(output=f"{execution}\npolylogue candidates complete", exit_code=0)

    with patch("devtools.lab_scenario.invoke_polylogue_cli", side_effect=_invoke):
        assert main(["run", "archive-smoke", "--tier", "0"]) == 0

    out = capsys.readouterr().out
    assert "Smoke stages:" in out
    assert "cli: ok" in out
    assert "Failed stages: none" in out


def test_main_json_reports_direct_scenario_payload(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import main

    def _invoke(_execution: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(output="polylogue candidates complete", exit_code=0)

    with patch("devtools.lab_scenario.invoke_polylogue_cli", side_effect=_invoke):
        assert main(["run", "archive-smoke", "--tier", "0", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "scenario": "archive-smoke",
        "stages": {
            "cli": "ok",
        },
        "failed_stages": [],
        "ok": True,
        "report_dir": None,
    }


def test_main_json_reports_direct_check_failures(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import main

    def _invoke(_execution: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(output="", exit_code=3)

    with patch("devtools.lab_scenario.invoke_polylogue_cli", side_effect=_invoke):
        assert main(["run", "archive-smoke", "--tier", "0", "--json", "--fail-fast"]) == 1

    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{") :])
    assert payload == {
        "scenario": "archive-smoke",
        "stages": {
            "cli": "error",
        },
        "failed_stages": ["cli"],
        "ok": False,
        "report_dir": None,
    }


def test_main_writes_direct_archive_smoke_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import main

    def _invoke(_execution: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(output="polylogue candidates complete", exit_code=0)

    report_dir = tmp_path / "reports"
    with patch("devtools.lab_scenario.invoke_polylogue_cli", side_effect=_invoke):
        assert main(["run", "archive-smoke", "--tier", "0", "--report-dir", str(report_dir), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["report_dir"] == str(report_dir)
    report_payload = json.loads((report_dir / "archive-smoke.json").read_text(encoding="utf-8"))
    assert report_payload["scenario"] == "archive-smoke"
    assert [check["name"] for check in report_payload["checks"]] == [
        "help-main",
        "help-mark-candidates",
        "completions-bash",
    ]
    assert all(check["passed"] is True for check in report_payload["checks"])


def test_reader_visual_smoke_json_reports_artifact_inventory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools.lab_scenario import main

    report_dir = tmp_path / "reports"
    completed = SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")
    with patch("devtools.lab_scenario.subprocess.run", return_value=completed) as run:
        assert main(["run", "reader-visual-smoke", "--json", "--report-dir", str(report_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads((report_dir / "reader-visual-smoke.json").read_text(encoding="utf-8"))
    assert payload == report_payload
    assert payload["scenario"] == "reader-visual-smoke"
    assert payload["exit_code"] == 0
    assert payload["artifact_report"] == str(report_dir / "reader-visual-smoke.json")
    assert {artifact["artifact_id"] for artifact in payload["artifact_inventory"]} >= {
        "polylogue.local_reader.search",
        "polylogue.local_reader.session",
        "polylogue.local_reader.workspace.stack",
    }
    assert run.call_args.kwargs["capture_output"] is True


def test_storage_correctness_json_runs_archive_backed_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools.lab_scenario import main

    report_dir = tmp_path / "reports"

    assert main(["run", "storage-correctness", "--json", "--report-dir", str(report_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads((report_dir / "storage-correctness.json").read_text(encoding="utf-8"))
    assert payload["scenario"] == "storage-correctness"
    assert payload["ok"] is True
    assert payload["failed_stages"] == []
    assert set(payload["stages"]) == {"idempotent-reingest", "fts-trigger-drift", "lineage-composition"}
    assert payload["checks"] == report_payload["checks"]
    checks = {entry["name"]: entry for entry in payload["checks"]}
    assert checks["idempotent-reingest"]["details"]["repeat_counts"]["skipped_sessions"] == 1
    assert checks["idempotent-reingest"]["details"]["derived_counts"] == {
        "sessions": 1,
        "messages": 2,
        "blocks": 2,
        "message_fts": 2,
    }
    assert checks["fts-trigger-drift"]["details"]["drifted_fts_rows"] == 0
    assert checks["fts-trigger-drift"]["details"]["after_fts_rows"] == 1
    assert checks["fts-trigger-drift"]["details"]["repeat_counts"]["_fts_repair"] == 1
    assert checks["lineage-composition"]["details"]["stored_child_positions"] == [2, 3]
    assert checks["lineage-composition"]["details"]["composed_texts"] == [
        "hello",
        "hi there",
        "child diverges here",
        "child reply",
    ]
    assert checks["lineage-composition"]["details"]["lineage"]["inheritance"] == "prefix-sharing"


def test_main_reports_unsupported_archive_smoke_tier(capsys: pytest.CaptureFixture[str]) -> None:
    from devtools.lab_scenario import main

    assert main(["run", "archive-smoke", "--tier", "2", "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == {"cli": "error"}
    assert payload["failed_stages"] == ["cli"]
