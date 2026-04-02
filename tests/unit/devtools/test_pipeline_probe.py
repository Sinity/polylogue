"""Tests for devtools.pipeline_probe."""

from __future__ import annotations

import json
from pathlib import Path

from devtools.pipeline_probe import main, run_probe


class _Args:
    provider = "chatgpt"
    count = 1
    messages_min = 3
    messages_max = 4
    seed = 7
    stage = "parse"
    json_out = None
    max_total_ms = None
    max_peak_rss_mb = None

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir


async def test_run_probe_emits_real_pipeline_summary(tmp_path) -> None:
    summary = await run_probe(_Args(tmp_path / "probe"))

    assert summary["probe"]["provider"] == "chatgpt"
    assert summary["result"]["run_path"] is not None
    assert summary["run_payload"]["metrics"]["peak_rss_mb"] is not None
    assert "index" in summary["run_payload"]["metrics"]["stages"]
    assert summary["db_stats"]["raw_conversations_count"] >= 1


def test_main_writes_json_summary(tmp_path, capsys) -> None:
    workdir = tmp_path / "probe-main"
    json_out = tmp_path / "probe-summary.json"

    exit_code = main([
        "--provider",
        "chatgpt",
        "--count",
        "1",
        "--messages-min",
        "3",
        "--messages-max",
        "4",
        "--stage",
        "parse",
        "--workdir",
        str(workdir),
        "--json-out",
        str(json_out),
    ])

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(json_out.read_text())

    assert exit_code == 0
    assert printed["paths"]["workdir"] == str(workdir.resolve())
    assert written["result"]["run_path"] is not None


def test_main_uses_ephemeral_workdir_when_omitted(capsys) -> None:
    exit_code = main([
        "--provider",
        "chatgpt",
        "--count",
        "1",
        "--messages-min",
        "3",
        "--messages-max",
        "4",
        "--stage",
        "parse",
    ])

    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert "polylogue-pipeline-probe-" in printed["paths"]["workdir"]
    assert printed["result"]["run_path"] is not None


def test_main_returns_nonzero_when_budget_is_exceeded(capsys) -> None:
    exit_code = main([
        "--provider",
        "chatgpt",
        "--count",
        "1",
        "--messages-min",
        "3",
        "--messages-max",
        "4",
        "--stage",
        "parse",
        "--max-total-ms",
        "0.0",
        "--max-peak-rss-mb",
        "0.0",
    ])

    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert printed["budgets"]["ok"] is False
    assert len(printed["budgets"]["violations"]) >= 1
