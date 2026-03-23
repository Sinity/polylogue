"""Tests for devtools.query_memory_budget."""

from __future__ import annotations

import json
import sys

from devtools.query_memory_budget import _read_vm_rss_kb, main, run_memory_budget


def test_read_vm_rss_kb_missing_pid_returns_zero() -> None:
    assert _read_vm_rss_kb(999_999_999) == 0


def test_run_memory_budget_reports_success_for_small_command() -> None:
    result = run_memory_budget([sys.executable, "-c", "print('ok')"], max_rss_mb=512)

    assert result["exit_code"] == 0
    assert result["within_budget"] is True
    assert result["peak_rss_mb"] >= 0


def test_main_emits_json_summary(capsys) -> None:
    exit_code = main(["--max-rss-mb", "512", "--", sys.executable, "-c", "print('ok')"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["exit_code"] == 0
    assert payload["within_budget"] is True
    assert payload["max_rss_mb"] == 512
