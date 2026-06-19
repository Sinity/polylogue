"""Tests for the ``polylogue ops tutorial`` walkthrough (#1263)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.tutorial import STAGES, tutorial_command
from polylogue.cli.shared.types import AppEnv
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION


class _CapturingConsole:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.calls.append(" ".join(str(a) for a in args))


def _make_env() -> AppEnv:
    ui: Any = MagicMock()
    ui.plain = True
    ui.console = _CapturingConsole()
    return AppEnv(ui=ui)


def _set_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Inherit XDG paths from the autouse fixture; set HOME to a sandbox."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def test_stage_count() -> None:
    assert len(STAGES) == 5
    assert tuple(s.number for s in STAGES) == (1, 2, 3, 4, 5)


def test_non_interactive_runs_to_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--non-interactive`` must not prompt and must exit cleanly."""
    _set_xdg(monkeypatch, tmp_path)
    runner = CliRunner()
    env = _make_env()
    result = runner.invoke(tutorial_command, ["--non-interactive"], obj=env)
    # The console.print outputs go to env.ui.console, not Click's stdout, so
    # the cleanest signal is exit code.
    assert result.exit_code == 0, result.output
    console: Any = env.ui.console
    combined = " ".join(console.calls)
    # Every stage title appears.
    for stage in STAGES:
        assert stage.title in combined


def test_stage_detect_sources_no_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_detect_sources

    _set_xdg(monkeypatch, tmp_path)
    satisfied, message = _stage_detect_sources()
    assert satisfied is False
    assert "No chat-source" in message


def test_stage_starter_config_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_starter_config

    _set_xdg(monkeypatch, tmp_path)
    config_dir = tmp_path / "xdg-config" / "polylogue"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "polylogue.toml").write_text("[sources]\n")
    satisfied, _ = _stage_starter_config()
    assert satisfied is True


def test_stage_first_search_empty_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    data_dir = tmp_path / "xdg-data" / "polylogue"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "index.db"
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    satisfied, message = _stage_first_search()
    assert satisfied is False
    assert "empty" in message.lower()


def test_stage_first_search_reads_archive_file_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    data_dir = tmp_path / "xdg-data" / "polylogue"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "index.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO sessions VALUES ('codex-session:one')")
    conn.commit()
    conn.close()
    satisfied, message = _stage_first_search()
    assert satisfied is True
    assert "1" in message


def test_stage_first_search_ignores_retired_single_file_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    data_dir = tmp_path / "xdg-data" / "polylogue"
    data_dir.mkdir(parents=True, exist_ok=True)
    retired = sqlite3.connect(data_dir / "retired.sqlite")
    retired.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
    retired.commit()
    retired.close()
    archive = sqlite3.connect(data_dir / "index.db")
    archive.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
    archive.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    archive.execute("INSERT INTO sessions VALUES ('codex-session:one')")
    archive.commit()
    archive.close()

    satisfied, message = _stage_first_search()

    assert satisfied is True
    assert "1" in message


def test_stage_first_search_no_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    satisfied, message = _stage_first_search()
    assert satisfied is False
    assert "ingest" in message.lower() or "archive" in message.lower()
