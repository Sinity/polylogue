"""Tests for first-run status UX."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.commands.status import (
    _show_daemon_status,
    _show_direct_json,
    _show_direct_status,
)
from polylogue.cli.shared.types import AppEnv


class _CapturingConsole:
    """Console mock that captures print calls in a list."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.calls.append(" ".join(str(a) for a in args))


def _make_app_env() -> AppEnv:
    """Create a minimal AppEnv for testing."""
    ui: Any = MagicMock()
    ui.plain = True
    ui.console = _CapturingConsole()
    return AppEnv(ui=ui)


def _combined_calls(env: AppEnv) -> str:
    """Get combined output from the capturing console."""
    console: Any = env.ui.console
    return " ".join(console.calls)


class TestNoArchiveStatus:
    """First-run UX when no archive exists."""

    def test_direct_status_no_archive(self) -> None:
        """_show_direct_status when DB does not exist shows actionable steps."""
        env = _make_app_env()
        fake_db = Path("/tmp/nonexistent.db")
        fake_root = Path("/tmp/nonexistent")

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                _show_direct_status(env)

        combined = _combined_calls(env)
        assert "polylogued run" in combined

    def test_direct_status_empty_archive(self) -> None:
        """_show_direct_status when DB exists but is empty shows guidance."""
        env = _make_app_env()
        fake_root = Path("/tmp/empty-root")

        fake_db = MagicMock()
        fake_db.exists.return_value = True

        # Each execute().fetchone() returns [0] (0 rows)
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = [0]

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                with patch("polylogue.storage.sqlite.connection_profile.open_connection", return_value=mock_conn):
                    _show_direct_status(env)

        combined = _combined_calls(env)
        # Empty-archive guidance now mentions "Conversations: 0" rather than the
        # earlier hand-written sentence.
        assert "Conversations: 0" in combined
        assert "polylogued run" in combined

    def test_direct_json_no_archive(self) -> None:
        """_show_direct_json when DB does not exist produces valid JSON."""
        env = _make_app_env()
        fake_db = Path("/tmp/nonexistent.db")
        fake_root = Path("/tmp/test-archive")

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                _show_direct_json(env)

        # The first call's first argument is the JSON string
        console: Any = env.ui.console
        output = console.calls[0]
        payload = json.loads(output)
        assert payload["daemon_liveness"] is False
        assert payload["db_exists"] is False
        assert payload["archive_root"] == str(fake_root)

    @pytest.mark.integration
    def test_status_subprocess_no_archive(self, tmp_path: Path) -> None:
        """polylogue status on fresh XDG paths shows actionable message."""
        from tests.infra.cli_subprocess import run_cli

        archive_root = tmp_path / "polylogue"
        env = {
            **os.environ,
            "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "HOME": str(tmp_path),
        }
        result = run_cli(["--plain", "status"], env=env)
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower
        assert "no archive" in output_lower or "polylogued" in output_lower


class TestDaemonStatus:
    """Daemon status rendering tests."""

    def test_empty_archive_with_sources(self) -> None:
        """When daemon runs but archive is empty, source discovery is shown."""
        env = _make_app_env()

        status_payload: dict[str, object] = {
            "daemon_liveness": True,
            "component_state": {
                "watcher": {"state": "running", "description": "watching 2 sources"},
            },
            "insight_freshness": {"total_sessions": 0, "sessions_with_profiles": 0},
            "live": {
                "sources": [
                    {"name": "claude-code", "root": "/tmp/claude", "exists": True},
                    {"name": "codex", "root": "/tmp/codex", "exists": False},
                ]
            },
            "watcher_roots": ["/tmp/claude", "/tmp/codex"],
            "live_ingest_attempts": {},
            "fts_readiness": {},
            "db_size_bytes": 0,
            "checked_at": "",
        }
        _show_daemon_status(env, status_payload)
        combined = _combined_calls(env)
        # Daemon-status output should report watching sources.
        assert "watching" in combined.lower(), f"expected 'watching' in status output, got: {combined[:200]}"

    def test_running_daemon_with_data(self) -> None:
        """When daemon runs with data, normal status is shown without first-run hints."""
        env = _make_app_env()

        status_payload: dict[str, object] = {
            "daemon_liveness": True,
            "component_state": {
                "watcher": {"state": "running", "description": "watching 2 sources"},
            },
            "insight_freshness": {"total_sessions": 42, "sessions_with_profiles": 40},
            "live": {
                "sources": [
                    {"name": "claude-code", "root": "/tmp/claude", "exists": True},
                ]
            },
            "live_ingest_attempts": {
                "completed_count": 10,
                "total_count": 10,
            },
            "fts_readiness": {"coverage_pct": 98.5},
            "db_size_bytes": 1_048_576,
            "disk_free_bytes": 107_374_182_400,
            "checked_at": "2026-05-07T12:00:00",
        }
        _show_daemon_status(env, status_payload)
        combined = _combined_calls(env)
        assert "no conversations" not in combined.lower()
        assert "running" in combined.lower()
