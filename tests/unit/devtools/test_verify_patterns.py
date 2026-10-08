from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def test_malformed_pattern_registry_fails_closed(tmp_path: Path) -> None:
    registry = tmp_path / "devtools/patterns/registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text("rules:\n  - id: incomplete\n", encoding="utf-8")

    payload = verify_patterns._payload(tmp_path)

    assert payload["blocking"] is True
    assert any("malformed pattern registry" in detail for detail in payload["required_gate"]["details"])


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
    matched_file.write_text("# inserted line\n" * 8 + "def existing():\n    return None\n", encoding="utf-8")
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


def test_rewriting_a_baseline_entry_into_a_new_ast_context_is_growth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Moving a match to a new context and re-pointing its exemption is caught.

    Anti-vacuity: reduce trusted anchors to ``(file, digest)`` and the parent
    count equals the candidate count, so the gate passes although context
    ``b`` was never exempted by the trusted revision.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Pattern Test"], check=True)
    rule = _rule(tmp_path)
    digest = hashlib.sha1(b"return None").hexdigest()
    rule.baseline_path.parent.mkdir(parents=True, exist_ok=True)
    rule.baseline_path.write_text(f"polylogue/existing.py:{digest}:{'a' * 40}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "seed trusted baseline"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-qm", "candidate parent"], check=True)
    rule.baseline_path.write_text(f"polylogue/existing.py:{digest}:{'b' * 40}\n", encoding="utf-8")
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter({("polylogue/existing.py", digest, "b" * 40): 1}),
    )

    payload = verify_patterns._payload(tmp_path)

    assert payload["new_matches"] == []
    assert payload["blocking"] is True
    assert any(
        "committed baseline grew" in error and "b" * 40 in error for error in payload["required_gate"]["details"]
    )


def test_merging_the_base_branch_does_not_count_its_exemptions_as_growth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An exemption the merged-in base branch added is trusted, not growth.

    Anti-vacuity: trust only the merge commit's first parent and the base
    branch's own new entry is reported as ``committed baseline grew``, failing
    every PR that merged its base.
    """
    git = ["git", "-C", str(tmp_path)]
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run([*git, "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run([*git, "config", "user.name", "Pattern Test"], check=True)
    rule = _rule(tmp_path)
    existing = "polylogue/existing.py:" + hashlib.sha1(b"return None").hexdigest() + ":" + "a" * 40 + "\n"
    added_digest = hashlib.sha1(b"return False").hexdigest()
    added = f"polylogue/base.py:{added_digest}:{'b' * 40}\n"
    rule.baseline_path.write_text(existing, encoding="utf-8")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "seed"], check=True)
    subprocess.run([*git, "checkout", "-qb", "feature"], check=True)
    (tmp_path / "feature.txt").write_text("x", encoding="utf-8")
    subprocess.run([*git, "add", "feature.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "feature work"], check=True)
    subprocess.run([*git, "checkout", "-q", "main"], check=True)
    rule.baseline_path.write_text(existing + added, encoding="utf-8")
    subprocess.run([*git, "commit", "-qam", "base adds an exemption"], check=True)
    subprocess.run([*git, "checkout", "-q", "feature"], check=True)
    subprocess.run([*git, "merge", "-q", "--no-edit", "main"], check=True)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter(
            {
                ("polylogue/existing.py", hashlib.sha1(b"return None").hexdigest(), "a" * 40): 1,
                ("polylogue/base.py", added_digest, "b" * 40): 1,
            }
        ),
    )

    payload = verify_patterns._payload(tmp_path)

    assert not any("committed baseline grew" in error for error in payload["required_gate"]["details"])


def test_a_merge_parent_without_the_baseline_file_contributes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A base branch that predates a feature's new baseline file does not fail the gate.

    Anti-vacuity (Codex P2, #5755): require the file at every parent and the
    merge of that older base is a blocking input error.
    """
    git = ["git", "-C", str(tmp_path)]
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run([*git, "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run([*git, "config", "user.name", "Pattern Test"], check=True)
    (tmp_path / "seed.txt").write_text("seed", encoding="utf-8")
    subprocess.run([*git, "add", "seed.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "seed"], check=True)
    subprocess.run([*git, "checkout", "-qb", "feature"], check=True)
    rule = _rule(tmp_path)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "feature adds the baseline"], check=True)
    subprocess.run([*git, "checkout", "-q", "main"], check=True)
    (tmp_path / "base.txt").write_text("base", encoding="utf-8")
    subprocess.run([*git, "add", "base.txt"], check=True)
    subprocess.run([*git, "commit", "-qm", "base moves on"], check=True)
    subprocess.run([*git, "checkout", "-q", "feature"], check=True)
    subprocess.run([*git, "merge", "-q", "--no-edit", "main"], check=True)
    monkeypatch.setattr(verify_patterns, "_rules", lambda _root: (rule,))
    monkeypatch.setattr(
        verify_patterns,
        "_scan",
        lambda _root, _rule: Counter(
            {("polylogue/existing.py", hashlib.sha1(b"return None").hexdigest(), "a" * 40): 1}
        ),
    )

    payload = verify_patterns._payload(tmp_path)

    assert payload["required_gate"]["error_count"] == 0, payload["required_gate"]["details"]


def test_pattern_entrypoint_marks_sha1_as_non_security_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A FIPS-style SHA-1 that refuses security use must still scan: the anchors are not security digests."""
    directory = tmp_path / "devtools/patterns"
    directory.mkdir(parents=True)
    (directory / "registry.yaml").write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "fixture",
                        "rule": "rule.yml",
                        "baseline": "baseline.txt",
                        "owner": "bead-test",
                        "status": "enforcing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "polylogue/example.py"
    source.parent.mkdir()
    source.write_text("def example():\n    return None\n", encoding="utf-8")
    match = {"file": "polylogue/example.py", "range": {"start": {"line": 1}}}
    anchor = verify_patterns._match_anchor(tmp_path, match, {})
    (directory / "baseline.txt").write_text(verify_patterns._anchor_text(anchor) + "\n", encoding="utf-8")
    real_sha1 = hashlib.sha1
    observations: list[bool] = []

    def restricted_sha1(data: bytes = b"", *, usedforsecurity: bool = True) -> Any:
        observations.append(usedforsecurity)
        if usedforsecurity:
            raise ValueError("SHA-1 is disabled for security use")
        return real_sha1(data, usedforsecurity=False)

    def process(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[0] == "git":
            raise subprocess.CalledProcessError(128, argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps([match]), "")

    monkeypatch.setattr(verify_patterns, "hashlib", SimpleNamespace(sha1=restricted_sha1))
    monkeypatch.setattr(verify_patterns, "repo_root", lambda: tmp_path)
    monkeypatch.setattr("devtools.verify_patterns.shutil.which", lambda _name: "/fixture/ast-grep")
    monkeypatch.setattr("devtools.verify_patterns.subprocess.run", process)
    assert verify_patterns.main(["--json"]) == 0
    assert observations and not any(observations)
    assert json.loads(capsys.readouterr().out)["blocking"] is False


def test_replace_rule_recognizes_public_directory_barrier_and_refuses_missing_barrier(tmp_path: Path) -> None:
    """The actual AST rule must distinguish a shipped barrier from an unsealed replace."""
    package = tmp_path / "polylogue"
    package.mkdir()
    (package / "sealed.py").write_text(
        "import os\n"
        "from polylogue.core.durable_fs import sync_directory\n"
        "def publish(source, destination):\n"
        "    os.replace(source, destination)\n"
        "    sync_directory(destination.parent)\n",
        encoding="utf-8",
    )
    (package / "unsealed.py").write_text(
        "import os\ndef publish(source, destination):\n    os.replace(source, destination)\n",
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[3]
    rule = verify_patterns.Rule(
        "replace-without-parent-fsync",
        repository / "devtools/patterns/replace-without-parent-fsync.yml",
        tmp_path / "unused-baseline.txt",
        "fhikb",
        "enforcing",
    )
    matches = verify_patterns._scan(tmp_path, rule)
    assert sum(matches.values()) == 1
    assert {anchor[0] for anchor in matches} == {"polylogue/unsealed.py"}


@pytest.mark.parametrize("barrier", ["sync_directory", "_fsync_directory", "_fsync_dir", None])
def test_parent_sync_rule_recognizes_canonical_barriers(tmp_path: Path, barrier: str | None) -> None:
    source = tmp_path / "polylogue/example.py"
    source.parent.mkdir()
    suffix = "" if barrier is None else f"    {barrier}(target.parent)\n"
    source.write_text("def publish(source, target):\n    os.replace(source, target)\n" + suffix, encoding="utf-8")
    rule = verify_patterns.Rule(
        "replace-without-parent-fsync",
        Path(__file__).parents[3] / "devtools/patterns/replace-without-parent-fsync.yml",
        tmp_path / "unused-baseline.txt",
        "fhikb",
        "enforcing",
    )
    assert sum(verify_patterns._scan(tmp_path, rule).values()) == (1 if barrier is None else 0)
