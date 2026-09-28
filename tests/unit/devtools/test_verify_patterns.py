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
    context = "a" * 40
    baseline.write_text(f"polylogue/existing.py:{digest}:{context}\n", encoding="utf-8")
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
        lambda _root, _rule: Counter(
            {
                ("polylogue/existing.py", hashlib.sha1(b"return None").hexdigest(), "a" * 40): 1,
                ("polylogue/new.py", hashlib.sha1(b"return False").hexdigest(), "b" * 40): 1,
            }
        ),
    )

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    digest = hashlib.sha1(b"return False").hexdigest()
    assert payload["new_matches"] == [f"synthetic polylogue/new.py:{digest}:{'b' * 40} (owner bead-test)"]
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
    assert payload["stale_matches"] == [f"synthetic polylogue/existing.py:{digest}:{'a' * 40}"]


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
    matched_file.write_text("\n" * 40 + "def example():\n    return None\n", encoding="utf-8")
    completed = SimpleNamespace(
        returncode=0, stdout='[{"file":"polylogue/example.py","range":{"start":{"line":41}}}]', stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    anchor = verify_patterns._match_anchor(
        tmp_path,
        {"file": "polylogue/example.py", "range": {"start": {"line": 41}}},
        {},
    )
    assert verify_patterns._scan(tmp_path, rule) == Counter({anchor: 1})


def test_displacing_a_baselined_match_does_not_trip_the_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rule = _rule(tmp_path)
    matched_file = tmp_path / "polylogue/existing.py"
    matched_file.parent.mkdir()
    matched_file.write_text("# inserted line\n" * 9 + "    return None\n", encoding="utf-8")
    anchor = verify_patterns._match_anchor(
        tmp_path,
        {"file": "polylogue/existing.py", "range": {"start": {"line": 9}}},
        {},
    )
    rule.baseline_path.write_text(verify_patterns._anchor_text(anchor) + "\n", encoding="utf-8")
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
    context = "a" * 40
    rule.baseline_path.write_text(f"polylogue/existing.py:{digest}:{context}:2\n", encoding="utf-8")
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))

    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter({("polylogue/existing.py", digest, context): 3}),
    )
    payload = verify_patterns._payload(tmp_path)
    assert payload["blocking"] is True
    assert payload["new_matches"] == [f"synthetic polylogue/existing.py:{digest}:{context} (owner bead-test)"]

    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter({("polylogue/existing.py", digest, context): 1}),
    )
    payload = verify_patterns._payload(tmp_path)
    assert payload["blocking"] is False
    assert payload["stale_matches"] == [f"synthetic polylogue/existing.py:{digest}:{context}"]


def test_equal_text_match_moving_to_another_ast_context_is_new_debt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same-line-text replacement cannot inherit an unrelated exemption.

    Anti-vacuity: reducing anchors back to (file, line digest) makes the
    baseline Counter equal and incorrectly leaves the gate green.
    """
    rule = _rule(tmp_path)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter(
            {("polylogue/existing.py", hashlib.sha1(b"return None").hexdigest(), "b" * 40): 1}
        ),
    )

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    assert "b" * 40 in payload["new_matches"][0]


def test_committed_baseline_cannot_grow_with_a_new_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Adding a finding and exemption together is rejected against parent Git.

    Anti-vacuity: comparing only current matches with the candidate baseline
    accepts the appended exemption and makes this test green.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Pattern Test"], check=True)
    rule = _rule(tmp_path)
    rule.baseline_path.parent.mkdir(parents=True, exist_ok=True)
    rule.baseline_path.write_text(
        "polylogue/existing.py:" + hashlib.sha1(b"return None").hexdigest() + ":" + "a" * 40 + "\n"
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "seed trusted baseline"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-qm", "candidate parent"], check=True)
    added_digest = hashlib.sha1(b"return False").hexdigest()
    rule.baseline_path.write_text(
        rule.baseline_path.read_text() + f"polylogue/new.py:{added_digest}:{'b' * 40}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter(
            {
                ("polylogue/existing.py", hashlib.sha1(b"return None").hexdigest(), "a" * 40): 1,
                ("polylogue/new.py", added_digest, "b" * 40): 1,
            }
        ),
    )

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    assert any("committed baseline grew" in error for error in payload["required_gate"]["details"])
