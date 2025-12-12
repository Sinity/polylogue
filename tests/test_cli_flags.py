from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import CommandEnv, run_sync_cli, _resolve_html_settings, _should_use_plain
from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.sync import _run_sync_drive, _run_local_sync
from polylogue.cli.context import resolve_collapse_value
from polylogue.local_sync import LocalSyncResult
from polylogue.settings import Settings
from polylogue.cli.app import _dispatch_sync


class DummyConsole:
    def print(self, *args, **kwargs):
        pass


class DummyUI:
    plain = True
    console = DummyConsole()


def test_html_flag_default_and_overrides_render():
    captured = {}

    def fake_dispatch(args, env):  # noqa: ARG001
        captured["html_mode"] = args.html_mode

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.commands.render.dispatch", fake_dispatch)

    result = runner.invoke(click_cli, ["render", "input.json"])
    assert result.exit_code == 0
    assert captured["html_mode"] == "auto"

    result = runner.invoke(click_cli, ["render", "input.json", "--html"])
    assert result.exit_code == 0
    assert captured["html_mode"] == "on"

    result = runner.invoke(click_cli, ["render", "input.json", "--html", "off"])
    assert result.exit_code == 0
    assert captured["html_mode"] == "off"
    monkeypatch.undo()


def test_html_flag_sync_variants():
    captured = {}

    def fake_dispatch(args, env):  # noqa: ARG001
        captured["html_mode"] = args.html_mode

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.commands.sync.dispatch", fake_dispatch)

    result = runner.invoke(click_cli, ["sync", "codex"])
    assert result.exit_code == 0
    assert captured["html_mode"] == "auto"

    result = runner.invoke(click_cli, ["sync", "codex", "--html", "on"])
    assert result.exit_code == 0
    assert captured["html_mode"] == "on"
    monkeypatch.undo()


def test_html_flag_import_variants():
    captured = {}

    def fake_import(args, env):  # noqa: ARG001
        captured["html_mode"] = args.html_mode

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.imports.run_import_cli", fake_import)

    result = runner.invoke(click_cli, ["import", "chatgpt", "export.zip", "--html", "off"])
    assert result.exit_code == 0
    assert captured["html_mode"] == "off"
    monkeypatch.undo()


def test_resolve_html_settings_modes():
    auto, explicit_auto = _resolve_html_settings(Namespace(html_mode="auto"))
    assert auto in (True, False)  # falls back to global setting
    assert explicit_auto is False

    enabled, explicit_on = _resolve_html_settings(Namespace(html_mode="on"))
    assert enabled is True
    assert explicit_on is True

    disabled, explicit_off = _resolve_html_settings(Namespace(html_mode="off"))
    assert disabled is False
    assert explicit_off is True


def test_run_sync_cli_invalid_provider_raises():
    with pytest.raises(SystemExit):
        run_sync_cli(Namespace(provider="unknown"), CommandEnv(ui=DummyUI()))


def test_run_sync_cli_dispatch(monkeypatch):
    calls = []

    def fake_drive(args, env):  # noqa: ARG001
        calls.append("drive")

    monkeypatch.setattr("polylogue.cli.sync._run_sync_drive", fake_drive)
    run_sync_cli(Namespace(provider="drive"), CommandEnv(ui=DummyUI()))
    assert calls == ["drive"]


def test_sync_parser_supports_selection_flags(tmp_path):
    captured = {}

    def fake_dispatch(args, env):  # noqa: ARG001
        captured["args"] = args

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.commands.sync.dispatch", fake_dispatch)

    result = runner.invoke(click_cli, ["sync", "drive", "--chat-id", "file-a", "--chat-id", "file-b"])
    assert result.exit_code == 0
    assert list(captured["args"].chat_ids) == ["file-a", "file-b"]

    session_one = tmp_path / "one.jsonl"
    session_two = tmp_path / "two.jsonl"
    result = runner.invoke(
        click_cli,
        ["sync", "codex", "--session", str(session_one), "--session", str(session_two)],
    )
    assert result.exit_code == 0
    assert list(captured["args"].sessions) == [session_one, session_two]
    monkeypatch.undo()


def test_print_paths_flags_present():
    captured = {}

    def fake_dispatch(args, env):  # noqa: ARG001
        captured["args"] = args

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.commands.render.dispatch", fake_dispatch)
    result = runner.invoke(click_cli, ["render", "input.json", "--print-paths"])
    assert result.exit_code == 0
    assert captured["args"].print_paths is True

    monkeypatch.setattr("polylogue.cli.commands.sync.dispatch", fake_dispatch)
    result = runner.invoke(click_cli, ["sync", "drive", "--print-paths"])
    assert result.exit_code == 0
    assert captured["args"].print_paths is True

    monkeypatch.setattr("polylogue.cli.imports.run_import_cli", fake_dispatch)
    result = runner.invoke(click_cli, ["import", "chatgpt", "export.zip", "--print-paths"])
    assert result.exit_code == 0
    assert captured["args"].print_paths is True
    monkeypatch.undo()


def test_run_sync_drive_respects_selected_ids(monkeypatch, tmp_path):
    captured = {}

    def fake_sync_command(options, env):  # noqa: ARG001
        captured["options"] = options

        class Result:
            count = 0
            output_dir = tmp_path
            folder_name = "AI Studio"
            folder_id = "folder-id"
            items = []
            total_stats = {}

        return Result()

    monkeypatch.setattr("polylogue.cli.sync.sync_command", fake_sync_command)

    class StubDrive:
        def __init__(self):
            self.calls = []

        def resolve_folder_id(self, folder_name, folder_id):  # noqa: ARG002
            self.calls.append("resolve")
            return "folder-id"

        def list_chats(self, folder_name, folder_id):  # noqa: ARG002
            self.calls.append("list")
            return [{"id": "chat-123", "name": "Test Chat"}]

    args = Namespace(
        links_only=True,
        chat_ids=["chat-123"],
        name_filter=None,
        since=None,
        until=None,
        folder_name="AI Studio",
        folder_id=None,
        list_only=False,
        dry_run=False,
        force=False,
        prune=False,
        collapse_threshold=None,
        out=None,
        html_mode="off",
        diff=False,
        json=False,
        links=None,
        all=False,
    )
    env = CommandEnv(ui=DummyUI())
    env.drive = StubDrive()

    _run_sync_drive(args, env)

    assert captured["options"].selected_ids == ["chat-123"]
    assert captured["options"].prefetched_chats == [{"id": "chat-123", "name": "Test Chat"}]


def test_run_local_sync_passes_sessions(monkeypatch, tmp_path):
    captured_sessions = {}

    class StubProvider:
        name = "stub"
        title = "Stub Provider"
        default_base = tmp_path
        default_output = tmp_path / "out"
        supports_diff = True
        create_base_dir = False
        supports_watch = False

        def list_sessions(self, base_dir):  # noqa: ARG002
            return []

        def sync_fn(self, *, sessions=None, **kwargs):  # noqa: ANN001
            captured_sessions["sessions"] = sessions
            return LocalSyncResult(written=[], skipped=0, pruned=0, output_dir=kwargs["output_dir"])

    monkeypatch.setattr("polylogue.cli.sync.get_local_provider", lambda _name: StubProvider())

    args = Namespace(
        provider="stub",
        base_dir=str(tmp_path),
        out=None,
        collapse_threshold=None,
        html_mode="off",
        force=False,
        prune=False,
        diff=False,
        sessions=[tmp_path / "session-a.jsonl", tmp_path / "session-b.jsonl"],
        all=False,
        json=False,
    )
    env = CommandEnv(ui=DummyUI())

    _run_local_sync("stub", args, env)

    assert captured_sessions["sessions"] == [Path(tmp_path / "session-a.jsonl"), Path(tmp_path / "session-b.jsonl")]


def test_allow_dirty_requires_force(monkeypatch, capsys):
    """Verify --allow-dirty without --force raises error."""
    import sys
    from polylogue.cli.app import main

    # Simulate command line args
    monkeypatch.setattr(sys, "argv", ["polylogue", "render", "input.json", "--allow-dirty"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "--allow-dirty requires --force" in captured.out


def test_drive_watch_is_rejected():
    env = CommandEnv(ui=DummyUI())
    args = Namespace(provider="drive", watch=True)

    with pytest.raises(SystemExit):
        _dispatch_sync(args, env)


def test_resolve_collapse_value_accepts_zero():
    settings = Settings(html_previews=False, html_theme="light", collapse_threshold=0, preferred_providers=[])

    assert resolve_collapse_value(0, settings) == 0
    assert resolve_collapse_value(None, settings) == 0


def test_should_use_plain_flags_override(monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert _should_use_plain(Namespace(interactive=False, plain=False)) is False

    assert _should_use_plain(Namespace(interactive=False, plain=True)) is True

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    assert _should_use_plain(Namespace(interactive=True, plain=False)) is False

    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    assert _should_use_plain(Namespace(interactive=False, plain=False)) is True
