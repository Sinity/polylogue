"""Tests for the staged-content secret-candidate pre-commit scan (polylogue-t9xd).

Leak-surfaces audit L5/L11: the pre-commit gate ran ``ruff format``/``ruff
check`` on staged ``*.py`` only -- nothing scanned staged content, of any
file type, for credential-shaped spans. These tests exercise the real
production ``scan_staged_paths``/``main`` against a real temporary git
repository's *index* (not the working tree), so they fail if the wiring
regresses to reading the working tree instead, or stops scanning non-``.py``
paths.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from polylogue.security.precommit_scan import main, scan_staged_paths


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    return repo


def _stage(repo: Path, relpath: str, content: str) -> None:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", relpath], cwd=repo, check=True)


class TestScanStagedPaths:
    def test_flags_staged_file_with_secret_shape(self, git_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A plausible secret pattern staged in a non-.py file is caught.

        Anti-vacuity: the old gate only ever looked at staged ``*.py``, so a
        secret pasted into a staged ``.md``/``.json``/``.txt`` file went
        through unscanned -- exactly the class this fix closes.
        """
        _stage(git_repo, "notes/handoff.md", "some context\nANTHROPIC_API_KEY=sk-ant-api03-" + "a" * 60 + "\n")

        total = scan_staged_paths(["notes/handoff.md"])

        assert total >= 1
        err = capsys.readouterr().err
        assert "notes/handoff.md" in err
        assert "anthropic-api-key" in err
        # Never logs the matched literal.
        assert "sk-ant-api03-" not in err

    def test_clean_content_produces_no_findings(self, git_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _stage(git_repo, "notes/handoff.md", "just some ordinary engineering notes, nothing sensitive here.\n")

        total = scan_staged_paths(["notes/handoff.md"])

        assert total == 0
        assert capsys.readouterr().err == ""

    def test_skips_scanner_test_fixture_paths(self, git_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The scanner's own test fixtures carry deliberate secret-shaped
        literals; scanning them would nag on every touch of the scanner's
        own tests."""
        _stage(
            git_repo,
            "tests/unit/security/test_secret_scan.py",
            "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP\n",
        )

        total = scan_staged_paths(["tests/unit/security/test_secret_scan.py"])

        assert total == 0
        assert capsys.readouterr().err == ""

    def test_reads_the_staged_index_not_the_working_tree(
        self, git_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Must scan ``git show :path`` (the index blob), not the working
        tree -- a secret staged then edited away in the working copy is
        still what will land in the commit."""
        _stage(git_repo, "notes/handoff.md", "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP\n")
        (git_repo / "notes" / "handoff.md").write_text("clean now, but only in the working tree\n", encoding="utf-8")

        total = scan_staged_paths(["notes/handoff.md"])

        assert total >= 1


class TestMain:
    def test_warns_but_does_not_block_by_default(
        self, git_repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POLYLOGUE_SECRET_SCAN_BLOCK", raising=False)
        _stage(git_repo, "notes/handoff.md", "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP\n")

        exit_code = main(["notes/handoff.md"])

        assert exit_code == 0
        assert "not blocking" in capsys.readouterr().err

    def test_blocks_when_opted_in(
        self, git_repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYLOGUE_SECRET_SCAN_BLOCK", "1")
        _stage(git_repo, "notes/handoff.md", "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP\n")

        exit_code = main(["notes/handoff.md"])

        assert exit_code == 1

    def test_no_arguments_is_a_clean_noop(self) -> None:
        assert main([]) == 0
