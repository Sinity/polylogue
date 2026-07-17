"""Tests for devtools.query_memory_budget."""

from __future__ import annotations

import json
import sys

import pytest

from devtools.query_memory_budget import _read_process_tree_rss_kb, _read_vm_rss_kb, main, run_memory_budget
from polylogue.core.json import JSONDocument, JSONValue, json_document


def _require_json_object(value: JSONValue) -> JSONDocument:
    assert isinstance(value, dict)
    return value


def _require_json_array(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def test_read_vm_rss_kb_missing_pid_returns_zero() -> None:
    assert _read_vm_rss_kb(999_999_999) == 0


def test_read_process_tree_rss_kb_missing_pid_returns_zero() -> None:
    assert _read_process_tree_rss_kb(999_999_999) == 0


def test_run_memory_budget_reports_success_for_small_command() -> None:
    result = run_memory_budget([sys.executable, "-c", "print('ok')"], max_rss_mb=512)
    peak_rss_mb = result["peak_rss_mb"]
    peak_parent_rss_mb = result["peak_parent_rss_mb"]

    assert result["exit_code"] == 0
    assert result["within_budget"] is True
    assert isinstance(peak_parent_rss_mb, (int, float))
    assert isinstance(peak_rss_mb, (int, float))
    assert peak_parent_rss_mb >= 0
    assert peak_rss_mb >= 0
    assert peak_rss_mb >= peak_parent_rss_mb
    receipt = result["workload_receipt"]
    assert receipt["status"] == "succeeded"
    assert _require_json_object(receipt["spec"])["measurement_scope"] == "process-tree"
    first_phase = _require_json_object(_require_json_array(receipt["phases"])[0])
    assert first_phase["unavailable"]


def test_main_emits_json_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--max-rss-mb", "512", "--", sys.executable, "-c", "print('ok')"])
    captured = capsys.readouterr()
    payload = json_document(json.loads(captured.out))

    assert exit_code == 0
    assert payload["exit_code"] == 0
    assert payload["within_budget"] is True
    assert payload["max_rss_mb"] == 512
    peak_rss_mb = payload["peak_rss_mb"]
    peak_parent_rss_mb = payload["peak_parent_rss_mb"]
    assert isinstance(peak_rss_mb, (int, float))
    assert isinstance(peak_parent_rss_mb, (int, float))
    assert peak_rss_mb >= peak_parent_rss_mb
    receipt = _require_json_object(payload["workload_receipt"])
    spec = _require_json_object(receipt["spec"])
    assert spec["semantic_result"] == "complete"
