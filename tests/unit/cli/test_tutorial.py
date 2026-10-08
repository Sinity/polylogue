"""Tests for the ``polylogue tutorial`` walkthrough."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.tutorial import STAGES, tutorial_command
from polylogue.cli.shared.types import AppEnv


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


def _starter_toml() -> str:
    """The config ``polylogue init`` actually writes (no sources detected)."""
    from polylogue.cli.commands.init import render_starter_toml

    return render_starter_toml(())


def _tutorial_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, sessions: int) -> Path:
    """Bootstrap a real archive at the tutorial's archive root with ``sessions`` sessions."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.live_ingest import write_session_sync

    root = tmp_path / "tutorial-archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    bootstrap_archive_root(root)
    for index in range(sessions):
        write_session_sync(
            root / "index.db",
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=f"tutorial-{index}",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
            ),
            archive_root=root,
        )
    return root


def _resident_search_count(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    from polylogue.cli import operation_kernel

    def read(_config: object, operation: str, payload: dict[str, object]) -> Any:
        assert operation == "query.aggregate"
        assert payload == {"mode": "count", "params": {}}
        return SimpleNamespace(value={"count": count})

    monkeypatch.setattr(operation_kernel, "configured_read_operation", read)


def test_stage_count() -> None:
    assert len(STAGES) == 4
    assert tuple(s.number for s in STAGES) == (1, 2, 3, 4)
    assert "Open reader" not in {stage.title for stage in STAGES}
    assert any("polylogue find 'hello' then read" in stage.action_text for stage in STAGES)


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
    assert "Checklist incomplete" in combined
    assert "polylogue find 'hello' then read" in combined
    assert "Done" not in combined
    assert "polylogue manual" in combined


def test_guided_path_shown_on_brand_new_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A totally fresh machine (no config, no archive) gets the shared guided path.

    polylogue-jnj.8: the same numbered steps bare `polylogue` prints on an
    absent archive, so there is one route, not a divergent tutorial-only copy.
    """
    from polylogue.cli.onboarding import GUIDED_PATH_STEPS

    _set_xdg(monkeypatch, tmp_path)
    runner = CliRunner()
    env = _make_env()
    result = runner.invoke(tutorial_command, ["--non-interactive"], obj=env)
    assert result.exit_code == 0, result.output
    console: Any = env.ui.console
    combined = " ".join(console.calls)
    assert "Guided path" in combined
    for step in GUIDED_PATH_STEPS:
        assert step.command_text in combined


def test_guided_path_hidden_once_config_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Partial setup (a starter config already written) skips the from-scratch banner.

    The per-stage checklist below is the right next action on its own —
    re-showing "seed the demo" to someone already wiring in real sources
    would be redundant, not helpful.
    """
    _set_xdg(monkeypatch, tmp_path)
    config_dir = tmp_path / "xdg-config" / "polylogue"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "polylogue.toml").write_text(_starter_toml(), encoding="utf-8")

    runner = CliRunner()
    env = _make_env()
    result = runner.invoke(tutorial_command, ["--non-interactive"], obj=env)
    assert result.exit_code == 0, result.output
    console: Any = env.ui.console
    combined = " ".join(console.calls)
    assert "Guided path" not in combined


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
    (config_dir / "polylogue.toml").write_text(_starter_toml(), encoding="utf-8")
    satisfied, _ = _stage_starter_config()
    assert satisfied is True


def test_stage_first_search_empty_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    _tutorial_archive(monkeypatch, tmp_path, sessions=0)
    _resident_search_count(monkeypatch, 0)
    satisfied, message = _stage_first_search()
    assert satisfied is False
    assert "empty" in message.lower()


def test_stage_first_search_reads_archive_file_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    _tutorial_archive(monkeypatch, tmp_path, sessions=1)
    _resident_search_count(monkeypatch, 1)
    satisfied, message = _stage_first_search()
    assert satisfied is True
    assert "1" in message


def test_stage_first_search_ignores_retired_single_file_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    root = _tutorial_archive(monkeypatch, tmp_path, sessions=1)
    retired = sqlite3.connect(root / "retired.sqlite")
    retired.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
    retired.executemany("INSERT INTO sessions VALUES (?)", [(n,) for n in range(5)])
    retired.commit()
    retired.close()
    _resident_search_count(monkeypatch, 1)

    satisfied, message = _stage_first_search()

    assert satisfied is True
    assert "1" in message and "5" not in message


def test_stage_first_search_no_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.cli.commands.tutorial import _stage_first_search

    _set_xdg(monkeypatch, tmp_path)
    satisfied, message = _stage_first_search()
    assert satisfied is False
    assert "ingest" in message.lower() or "archive" in message.lower()


def test_daemon_alive_rejects_a_non_daemon_status_read() -> None:
    """A non-daemon status result never counts as a running daemon.

    The probe discriminates on the result's authority mode rather than merely
    treating a non-raising operation call as proof that a daemon is alive.

    Anti-vacuity: reading liveness from "the dispatch did not raise" makes
    the non-daemon case red, since that probe succeeds.
    """
    from unittest.mock import patch

    from polylogue.cli.commands.tutorial import _daemon_alive
    from polylogue.cli.operation_kernel import OperationResult

    def _served(mode: str) -> Any:
        def _call(config: Any, operation: str, payload: dict[str, object], **kwargs: Any) -> OperationResult:
            del config, payload, kwargs
            return OperationResult(operation, {}, {"mode": mode, "class": "read"})

        return _call

    with patch("polylogue.cli.operation_kernel.configured_read_operation", new=_served("local")):
        assert _daemon_alive() is False
    with patch("polylogue.cli.operation_kernel.configured_read_operation", new=_served("daemon")):
        assert _daemon_alive() is True


def test_daemon_alive_never_raises() -> None:
    """A probe that explodes is worse than one that says "no".

    Anti-vacuity: letting the dispatch exception propagate makes this red.
    """
    from unittest.mock import patch

    from polylogue.cli.commands.tutorial import _daemon_alive

    def _boom(config: Any, operation: str, payload: dict[str, object], **kwargs: Any) -> Any:
        del config, operation, payload, kwargs
        raise RuntimeError("archive is unreadable")

    with patch("polylogue.cli.operation_kernel.configured_read_operation", new=_boom):
        assert _daemon_alive() is False
