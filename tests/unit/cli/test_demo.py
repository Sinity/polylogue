"""Tests for the ``polylogue demo`` CLI command."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

from polylogue.cli.click_app import cli as click_cli


class TestDemoSeed:
    """``polylogue demo --seed`` creates a full demo environment."""

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


class TestDemoTierFilter:
    """``polylogue demo --showcase --tier`` wiring."""

    def test_tier_option_passed_to_do_showcase(self, cli_runner, tmp_path):
        with patch("polylogue.cli.commands.demo._do_showcase") as mock_showcase:
            result = cli_runner.invoke(
                click_cli,
                ["demo", "--showcase", "--tier", "2", "-o", str(tmp_path)],
            )
        assert result.exit_code == 0
        assert mock_showcase.call_count == 1
        # tier_filter is the second-to-last positional arg; audit_dir is last
        args = mock_showcase.call_args.args
        assert args[-2] == 2   # tier_filter
        assert args[-1] is None  # audit_dir

    def test_no_tier_defaults_none(self, cli_runner, tmp_path):
        with patch("polylogue.cli.commands.demo._do_showcase") as mock_showcase:
            result = cli_runner.invoke(
                click_cli,
                ["demo", "--showcase", "-o", str(tmp_path)],
            )
        assert result.exit_code == 0
        args = mock_showcase.call_args.args
        assert args[-2] is None  # tier_filter
        assert args[-1] is None  # audit_dir

    def test_showcase_data_requires_showcase_mode(self, cli_runner, tmp_path):
        result = cli_runner.invoke(
            click_cli,
            ["demo", "--corpus", "--showcase-data", "synthetic", "-o", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert "--showcase-data requires --showcase" in result.output
