from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from polylogue.cli import CommandEnv, build_parser, run_sync_cli, _resolve_html_settings
from polylogue.cli.sync import _run_sync_drive, _run_local_sync
from polylogue.local_sync import LocalSyncResult


class DummyConsole:
    def print(self, *args, **kwargs):
        pass


class DummyUI:
    plain = True
    console = DummyConsole()


def test_html_flag_default_and_overrides_render():
    parser = build_parser()
    args = parser.parse_args(["render", "input.json"])
    assert args.html_mode == "auto"

    args_on = parser.parse_args(["render", "input.json", "--html"])
    assert args_on.html_mode == "on"

    args_off = parser.parse_args(["render", "input.json", "--html", "off"])
    assert args_off.html_mode == "off"


def test_html_flag_sync_variants():
    parser = build_parser()
    args_auto = parser.parse_args(["sync", "codex"])
    assert args_auto.html_mode == "auto"

    args_explicit = parser.parse_args(["sync", "codex", "--html", "on"])
    assert args_explicit.html_mode == "on"


def test_html_flag_import_variants():
    parser = build_parser()
    args = parser.parse_args(["import", "chatgpt", "export.zip", "--html", "off"])
    assert args.html_mode == "off"


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
    parser = build_parser()
    drive_args = parser.parse_args(
        ["sync", "drive", "--chat-id", "file-a", "--chat-id", "file-b"]
    )
    assert drive_args.chat_ids == ["file-a", "file-b"]

    session_one = tmp_path / "one.jsonl"
    session_two = tmp_path / "two.jsonl"
    local_args = parser.parse_args(
        ["sync", "codex", "--session", str(session_one), "--session", str(session_two)]
    )
    assert local_args.sessions == [session_one, session_two]


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

    _run_sync_drive(args, env)

    assert captured["options"].selected_ids == ["chat-123"]


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
