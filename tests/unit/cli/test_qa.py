"""Tests for the `polylogue qa` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_qa_listed_in_main_help(runner: CliRunner) -> None:
    result = runner.invoke(click_cli, ["--help"])
    assert result.exit_code == 0
    assert "qa" in result.output


def test_qa_fails_without_sources(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(click_cli, ["qa"])
        assert result.exit_code != 0
        assert "No QA sources found" in result.output


def test_qa_snapshots_default_sources(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        cwd = Path.cwd()
        (cwd / "qa_outputs").mkdir()
        (cwd / "qa_outputs" / "Q01.txt").write_text("hello", encoding="utf-8")
        (cwd / "qa_archive").mkdir()
        (cwd / "qa_archive" / "A01.txt").write_text("world", encoding="utf-8")

        archive_root = cwd / ".archive"
        result = runner.invoke(
            click_cli,
            ["qa", "--name", "nightly"],
            env={
                "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
                "POLYLOGUE_FORCE_PLAIN": "1",
            },
        )
        assert result.exit_code == 0
        snapshots_root = archive_root / "qa" / "snapshots"
        snapshots = [p for p in snapshots_root.iterdir() if p.is_dir() and p.name != "latest"]
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert (snapshot / "manifest.json").exists()
        assert (snapshot / "INDEX.md").exists()
        assert (snapshot / "qa_outputs" / "Q01.txt").read_text(encoding="utf-8") == "hello"
        assert (snapshot / "qa_archive" / "A01.txt").read_text(encoding="utf-8") == "world"
        assert (snapshots_root / "latest").exists()


def test_qa_json_output(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        cwd = Path.cwd()
        custom = cwd / "custom-qa"
        custom.mkdir()
        (custom / "result.log").write_text("ok", encoding="utf-8")

        archive_root = cwd / ".archive"
        result = runner.invoke(
            click_cli,
            ["qa", "--source", str(custom), "--json"],
            env={
                "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
                "POLYLOGUE_FORCE_PLAIN": "1",
            },
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["entry_count"] == 1
        assert "snapshot_dir" in payload
