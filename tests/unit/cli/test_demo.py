"""Tests for the ``polylogue demo`` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli


@pytest.fixture
def runner():
    return CliRunner()


class TestDemoHelp:
    """Demo command is registered and shows help."""

    def test_help_exits_zero(self, runner):
        result = runner.invoke(click_cli, ["demo", "--help"])
        assert result.exit_code == 0
        assert "--seed" in result.output
        assert "--corpus" in result.output

    def test_listed_in_main_help(self, runner):
        result = runner.invoke(click_cli, ["--help"])
        assert "demo" in result.output


class TestDemoCorpus:
    """``polylogue demo --corpus`` generates raw wire-format files."""

    def test_corpus_default_all_providers(self, runner, tmp_path):
        result = runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-n", "1"])
        assert result.exit_code == 0
        # Should create subdirectories for each provider
        provider_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(provider_dirs) >= 3  # chatgpt, claude-code, claude-ai, codex, gemini

    def test_corpus_single_provider(self, runner, tmp_path):
        result = runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "chatgpt", "-n", "2"])
        assert result.exit_code == 0
        chatgpt_dir = tmp_path / "chatgpt"
        assert chatgpt_dir.is_dir()
        files = list(chatgpt_dir.glob("*.json"))
        assert len(files) == 2

    def test_corpus_multiple_providers(self, runner, tmp_path):
        result = runner.invoke(click_cli, [
            "demo", "--corpus", "-o", str(tmp_path),
            "-p", "chatgpt", "-p", "codex", "-n", "1",
        ])
        assert result.exit_code == 0
        assert (tmp_path / "chatgpt").is_dir()
        assert (tmp_path / "codex").is_dir()
        # Only requested providers
        provider_dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
        assert set(provider_dirs) == {"chatgpt", "codex"}

    def test_corpus_invalid_provider(self, runner, tmp_path):
        result = runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown provider" in result.output

    def test_corpus_generates_valid_files(self, runner, tmp_path):
        """Generated corpus files contain valid JSON."""
        import json

        result = runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "chatgpt", "-n", "1"])
        assert result.exit_code == 0

        json_files = list((tmp_path / "chatgpt").glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert isinstance(data, (list, dict))

    def test_corpus_jsonl_providers(self, runner, tmp_path):
        """JSONL providers produce .jsonl files."""
        result = runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "claude-code", "-n", "1"])
        assert result.exit_code == 0
        jsonl_files = list((tmp_path / "claude-code").glob("*.jsonl"))
        assert len(jsonl_files) == 1


class TestDemoSeed:
    """``polylogue demo --seed`` creates a full demo environment."""

    def test_seed_env_only(self, runner, tmp_path):
        result = runner.invoke(click_cli, [
            "demo", "--seed", "--env-only", "-o", str(tmp_path),
            "-n", "1", "-p", "chatgpt",
        ])
        assert result.exit_code == 0
        assert "export XDG_DATA_HOME=" in result.output
        assert "export POLYLOGUE_ARCHIVE_ROOT=" in result.output

    def test_seed_creates_database(self, runner, tmp_path):
        result = runner.invoke(click_cli, [
            "demo", "--seed", "-o", str(tmp_path),
            "-n", "1", "-p", "chatgpt",
        ])
        assert result.exit_code == 0
        # Database should exist
        db_path = tmp_path / "data" / "polylogue" / "polylogue.db"
        assert db_path.exists()


class TestDemoDefault:
    """Default mode (no --seed/--corpus) defaults to corpus."""

    def test_default_generates_corpus(self, runner, tmp_path):
        result = runner.invoke(click_cli, ["demo", "-o", str(tmp_path), "-p", "chatgpt", "-n", "1"])
        assert result.exit_code == 0
        assert (tmp_path / "chatgpt").is_dir()
