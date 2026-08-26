from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools import verify_patterns


def _rule(tmp_path: Path, *, status: str = "enforcing") -> verify_patterns.Rule:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("polylogue/existing.py:10\n", encoding="utf-8")
    return verify_patterns.Rule("synthetic", tmp_path / "rule.yml", baseline, "bead-test", status)


def test_current_pattern_gate_is_seeded_and_reports_pending_rules() -> None:
    payload = verify_patterns._payload(Path(__file__).parents[3])

    assert payload["blocking"] is False
    assert payload["new_matches"] == []
    details = payload["required_gate"]["details"]
    assert any("sqlite-error-default: enforcing" in item for item in details)
    assert any("connection-lifecycle: pending" in item for item in details)


def test_synthetic_new_match_makes_the_ratchet_red(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: {("polylogue/existing.py", 10), ("polylogue/new.py", 20)},
    )

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    assert payload["new_matches"] == ["synthetic polylogue/new.py:20 (owner bead-test)"]
    assert payload["required_gate"]["diagnosis"] == "gate_semantic_violation"


def test_stale_baseline_is_reported_as_shrinkable_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rule = _rule(tmp_path)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(verify_patterns, "_scan", lambda _root, _rule: set())

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is False
    assert payload["stale_matches"] == ["synthetic polylogue/existing.py:10"]


def test_missing_ast_grep_is_typed_and_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    gate = payload["required_gate"]
    assert gate["diagnosis"] == "gate_missing_executable"
    assert "uv sync --group audit" in gate["details"][0]


def test_scan_converts_ast_grep_zero_based_lines_to_one_based(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    completed = SimpleNamespace(
        returncode=0, stdout='[{"file":"polylogue/example.py","range":{"start":{"line":41}}}]', stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    assert verify_patterns._scan(tmp_path, rule) == {("polylogue/example.py", 42)}
