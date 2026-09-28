from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools import verify_patterns


def _rule(tmp_path: Path, *, status: str = "enforcing") -> verify_patterns.Rule:
    baseline = tmp_path / "baseline.txt"
    digest = hashlib.sha1(b"return None").hexdigest()
    baseline.write_text(f"polylogue/existing.py:{digest}\n", encoding="utf-8")
    return verify_patterns.Rule("synthetic", tmp_path / "rule.yml", baseline, "bead-test", status)


def test_current_pattern_gate_is_seeded_and_reports_pending_rules() -> None:
    payload = verify_patterns._payload(Path(__file__).parents[3])

    assert payload["blocking"] is False
    assert payload["new_matches"] == []
    details = payload["required_gate"]["details"]
    assert any("sqlite-error-default: enforcing" in item for item in details)
    assert any("connection-lifecycle: pending" in item for item in details)


@pytest.mark.skipif(shutil.which("ast-grep") is None, reason="ast-grep is required to execute structural rules")
def test_exclusive_create_flags_bound_before_open_are_detected(tmp_path: Path) -> None:
    (tmp_path / "polylogue").mkdir()
    (tmp_path / "polylogue" / "writer.py").write_text(
        "import os\n"
        "def write(path):\n"
        "    flags = os.O_CREAT | os.O_EXCL\n"
        "    fd = os.open(path, flags)\n"
        "    with os.fdopen(fd, 'w') as stream:\n"
        "        stream.write('x')\n",
        encoding="utf-8",
    )
    rule = verify_patterns.Rule(
        "exclusive-create-write",
        Path(__file__).parents[3] / "devtools/patterns/exclusive-create-write.yml",
        tmp_path / "baseline.txt",
        "owner",
        "enforcing",
    )

    assert verify_patterns._scan(tmp_path, rule)


@pytest.mark.skipif(shutil.which("ast-grep") is None, reason="ast-grep is required to execute structural rules")
def test_sqlite_error_rule_ignores_an_explicit_fail_closed_false(tmp_path: Path) -> None:
    (tmp_path / "polylogue").mkdir()
    (tmp_path / "polylogue" / "fallback.py").write_text(
        "import sqlite3\n"
        "def check():\n"
        "    try:\n"
        "        return True\n"
        "    except sqlite3.Error:\n"
        "        return False\n",
        encoding="utf-8",
    )
    rule = verify_patterns.Rule(
        "sqlite-error-default",
        Path(__file__).parents[3] / "devtools/patterns/sqlite-error-default.yml",
        tmp_path / "baseline.txt",
        "owner",
        "enforcing",
    )

    assert not verify_patterns._scan(tmp_path, rule)


def test_synthetic_new_match_makes_the_ratchet_red(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter(
            {
                ("polylogue/existing.py", hashlib.sha1(b"return None").hexdigest()): 1,
                ("polylogue/new.py", hashlib.sha1(b"return False").hexdigest()): 1,
            }
        ),
    )

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    digest = hashlib.sha1(b"return False").hexdigest()
    assert payload["new_matches"] == [f"synthetic polylogue/new.py:{digest} (owner bead-test)"]
    assert payload["required_gate"]["diagnosis"] == "gate_semantic_violation"


def test_stale_baseline_is_reported_as_shrinkable_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rule = _rule(tmp_path)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(verify_patterns, "_scan", lambda _root, _rule: Counter())

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is False
    digest = hashlib.sha1(b"return None").hexdigest()
    assert payload["stale_matches"] == [f"synthetic polylogue/existing.py:{digest}"]


def test_missing_ast_grep_is_typed_and_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    gate = payload["required_gate"]
    assert gate["diagnosis"] == "gate_missing_executable"
    # The remedy must be the command that actually provisions the group:
    # `uv sync --group audit` alone prunes the dev extra back out.
    assert "uv sync --extra dev --group audit --frozen" in gate["details"][0]


def test_scan_converts_ast_grep_zero_based_lines_to_one_based(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    matched_file = tmp_path / "polylogue/example.py"
    matched_file.parent.mkdir()
    matched_file.write_text("\n" * 41 + "    return None\n", encoding="utf-8")
    completed = SimpleNamespace(
        returncode=0, stdout='[{"file":"polylogue/example.py","range":{"start":{"line":41}}}]', stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    digest = hashlib.sha1(b"return None").hexdigest()
    assert verify_patterns._scan(tmp_path, rule) == Counter({("polylogue/example.py", digest): 1})


def test_displacing_a_baselined_match_does_not_trip_the_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    matched_file = tmp_path / "polylogue/existing.py"
    matched_file.parent.mkdir()
    matched_file.write_text("# inserted line\n" * 9 + "    return None\n", encoding="utf-8")
    completed = SimpleNamespace(
        returncode=0, stdout='[{"file":"polylogue/existing.py","range":{"start":{"line":9}}}]', stderr=""
    )
    monkeypatch.setattr(shutil, "which", lambda _name: "/bin/ast-grep")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is False
    assert payload["new_matches"] == []
    assert payload["stale_matches"] == []


def test_duplicate_content_anchors_are_compared_as_a_multiset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    digest = hashlib.sha1(b"return None").hexdigest()
    rule.baseline_path.write_text(f"polylogue/existing.py:{digest}:2\n", encoding="utf-8")
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))

    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter({("polylogue/existing.py", digest): 3}),
    )
    payload = verify_patterns._payload(tmp_path)
    assert payload["blocking"] is True
    assert payload["new_matches"] == [f"synthetic polylogue/existing.py:{digest} (owner bead-test)"]

    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter({("polylogue/existing.py", digest): 1}),
    )
    payload = verify_patterns._payload(tmp_path)
    assert payload["blocking"] is False
    assert payload["stale_matches"] == [f"synthetic polylogue/existing.py:{digest}"]
