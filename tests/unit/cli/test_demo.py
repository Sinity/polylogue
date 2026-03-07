"""Tests for the ``polylogue demo`` CLI command."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

from polylogue.cli.click_app import cli as click_cli


class TestDemoHelp:
    """Demo command is registered and shows help."""

    def test_help_exits_zero(self, cli_runner):
        result = cli_runner.invoke(click_cli, ["demo", "--help"])
        assert result.exit_code == 0
        assert "--seed" in result.output
        assert "--corpus" in result.output

    def test_listed_in_main_help(self, cli_runner):
        result = cli_runner.invoke(click_cli, ["--help"])
        assert "demo" in result.output


class TestDemoCorpus:
    """``polylogue demo --corpus`` generates raw wire-format files."""

    def test_corpus_default_all_providers(self, cli_runner, tmp_path):
        result = cli_runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-n", "1"])
        assert result.exit_code == 0
        # Should create subdirectories for each provider
        provider_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(provider_dirs) >= 3  # chatgpt, claude-code, claude-ai, codex, gemini

    def test_corpus_single_provider(self, cli_runner, tmp_path):
        result = cli_runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "chatgpt", "-n", "2"])
        assert result.exit_code == 0
        chatgpt_dir = tmp_path / "chatgpt"
        assert chatgpt_dir.is_dir()
        files = list(chatgpt_dir.glob("*.json"))
        assert len(files) == 2

    def test_corpus_multiple_providers(self, cli_runner, tmp_path):
        result = cli_runner.invoke(click_cli, [
            "demo", "--corpus", "-o", str(tmp_path),
            "-p", "chatgpt", "-p", "codex", "-n", "1",
        ])
        assert result.exit_code == 0
        assert (tmp_path / "chatgpt").is_dir()
        assert (tmp_path / "codex").is_dir()
        # Only requested providers
        provider_dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
        assert set(provider_dirs) == {"chatgpt", "codex"}

    def test_corpus_invalid_provider(self, cli_runner, tmp_path):
        result = cli_runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown provider" in result.output

    def test_corpus_generates_valid_files(self, cli_runner, tmp_path):
        """Generated corpus files contain valid JSON."""
        import json

        result = cli_runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "chatgpt", "-n", "1"])
        assert result.exit_code == 0

        json_files = list((tmp_path / "chatgpt").glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert isinstance(data, (list, dict))

    def test_corpus_jsonl_providers(self, cli_runner, tmp_path):
        """JSONL providers produce .jsonl files."""
        result = cli_runner.invoke(click_cli, ["demo", "--corpus", "-o", str(tmp_path), "-p", "claude-code", "-n", "1"])
        assert result.exit_code == 0
        jsonl_files = list((tmp_path / "claude-code").glob("*.jsonl"))
        assert len(jsonl_files) == 1


class TestDemoSeed:
    """``polylogue demo --seed`` creates a full demo environment."""

    def test_seed_env_only(self, cli_runner, tmp_path):
        result = cli_runner.invoke(click_cli, [
            "demo", "--seed", "--env-only", "-o", str(tmp_path),
            "-n", "1", "-p", "chatgpt",
        ])
        assert result.exit_code == 0
        assert "export XDG_DATA_HOME=" in result.output
        assert "export POLYLOGUE_ARCHIVE_ROOT=" in result.output

    def test_seed_creates_database(self, cli_runner, tmp_path):
        result = cli_runner.invoke(click_cli, [
            "demo", "--seed", "-o", str(tmp_path),
            "-n", "1", "-p", "chatgpt",
        ])
        assert result.exit_code == 0
        # Database should exist
        db_path = tmp_path / "data" / "polylogue" / "polylogue.db"
        assert db_path.exists()

    def test_seed_restores_environment(self, cli_runner, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", "/tmp/original-data")
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", "/tmp/original-archive")

        with patch("polylogue.pipeline.runner.run_sources", new=AsyncMock(return_value=type(
            "_Result",
            (),
            {"counts": {"conversations": 0, "messages": 0}},
        )())):
            result = cli_runner.invoke(
                click_cli,
                ["demo", "--seed", "-o", str(tmp_path), "-n", "1", "-p", "chatgpt"],
            )

        assert result.exit_code == 0
        assert os.environ["XDG_DATA_HOME"] == "/tmp/original-data"
        assert os.environ["POLYLOGUE_ARCHIVE_ROOT"] == "/tmp/original-archive"


class TestDemoDefault:
    """Default mode (no --seed/--corpus) defaults to corpus."""

    def test_default_generates_corpus(self, cli_runner, tmp_path):
        result = cli_runner.invoke(click_cli, ["demo", "-o", str(tmp_path), "-p", "chatgpt", "-n", "1"])
        assert result.exit_code == 0
        assert (tmp_path / "chatgpt").is_dir()


class TestDemoShowcase:
    """``polylogue demo --showcase`` wiring and argument handling."""

    def test_showcase_data_passed_to_runner(self, cli_runner, tmp_path):
        with patch("polylogue.cli.commands.demo._do_showcase") as mock_showcase:
            result = cli_runner.invoke(
                click_cli,
                [
                    "demo",
                    "--showcase",
                    "--showcase-data",
                    "synthetic",
                    "-n",
                    "7",
                    "-o",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0
        assert mock_showcase.call_count == 1
        args = mock_showcase.call_args.args
        assert args[6] == "synthetic"
        assert args[7] == 7

    def test_showcase_data_requires_showcase_mode(self, cli_runner, tmp_path):
        result = cli_runner.invoke(
            click_cli,
            ["demo", "--corpus", "--showcase-data", "synthetic", "-o", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert "--showcase-data requires --showcase" in result.output
