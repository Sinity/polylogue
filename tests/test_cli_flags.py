from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import _should_use_plain, cli as click_cli
from polylogue.cli.context import resolve_collapse_value
from polylogue.cli.context import resolve_html_settings
from polylogue.cli.sync import _run_local_sync, _run_sync_drive, run_sync_cli
from polylogue.commands import CommandEnv
from polylogue.local_sync import LocalSyncResult
from polylogue.settings import Settings


class DummyConsole:
    def print(self, *args, **kwargs):
        pass


class DummyUI:
    plain = True
    console = DummyConsole()


def test_html_flag_default_and_overrides_render():
    captured = {}

    def fake_render(args, env, *, json_output=False):  # noqa: ARG001
        captured["html_mode"] = args.html_mode

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.render.run_render_cli", fake_render)

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

    def fake_sync(args, env):  # noqa: ARG001
        captured["html_mode"] = args.html_mode

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.sync.run_sync_cli", fake_sync)

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

    result = runner.invoke(click_cli, ["import", "run", "chatgpt", "export.zip", "--html", "off"])
    assert result.exit_code == 0
    assert captured["html_mode"] == "off"
    monkeypatch.undo()


def test_resolve_html_settings_modes():
    auto, explicit_auto = resolve_html_settings(SimpleNamespace(html_mode="auto"))
    assert auto in (True, False)  # falls back to global setting
    assert explicit_auto is False

    enabled, explicit_on = resolve_html_settings(SimpleNamespace(html_mode="on"))
    assert enabled is True
    assert explicit_on is True

    disabled, explicit_off = resolve_html_settings(SimpleNamespace(html_mode="off"))
    assert disabled is False
    assert explicit_off is True


def test_run_sync_cli_invalid_provider_raises():
    with pytest.raises(SystemExit):
        run_sync_cli(SimpleNamespace(provider="unknown"), CommandEnv(ui=DummyUI()))


def test_run_sync_cli_dispatch(monkeypatch):
    calls = []

    def fake_local(provider_name, args, env):  # noqa: ARG001
        calls.append(provider_name)

    monkeypatch.setattr("polylogue.cli.sync._run_local_sync", fake_local)
    run_sync_cli(SimpleNamespace(provider="codex"), CommandEnv(ui=DummyUI()))
    assert calls == ["codex"]


def test_sync_jobs_flag_passed_to_local_provider(monkeypatch, tmp_path):
    captured = {}

    def fake_sync_fn(**kwargs):
        captured.update(kwargs)

        class Result:
            written = []
            skipped = 0
            pruned = 0
            ignored = 0
            attachments = 0
            attachment_bytes = 0
            tokens = 0
            words = 0
            diffs = 0
            duration = 0.0
            failures = 0
            failed = []
            output_dir = tmp_path

        return Result()

    class StubProvider:
        title = "Stub"
        name = "codex"
        supports_diff = True
        supports_watch = True
        supports_jobs = True
        default_base = tmp_path
        default_output = tmp_path
        sync_fn = staticmethod(fake_sync_fn)

    monkeypatch.setattr("polylogue.cli.sync.get_local_provider", lambda _name: StubProvider())
    args = SimpleNamespace(
        provider="codex",
        out=str(tmp_path),
        base_dir=str(tmp_path),
        sessions=[tmp_path / "a.jsonl"],
        all=True,
        jobs=3,
        dry_run=False,
        force=False,
        prune=False,
        diff=False,
        json=False,
        watch=False,
        offline=False,
        collapse_threshold=None,
        html_mode="off",
        attachment_ocr=True,
        _attachment_ocr_explicit=False,
        sanitize_html=False,
        meta=(),
        max_disk=None,
        resume_from=None,
        prune_snapshot=False,
        root=None,
    )
    env = CommandEnv(ui=DummyUI())
    run_sync_cli(args, env)
    assert captured["jobs"] == 3


def test_sync_chatgpt_base_dir_does_not_require_config_inbox(tmp_path):
    env = CommandEnv(ui=DummyUI())
    # Point configured inbox at a missing path; --base-dir should override the check.
    env.config.exports.chatgpt = tmp_path / "missing-inbox"
    base_dir = tmp_path / "exports"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out"
    args = SimpleNamespace(
        provider="chatgpt",
        out=str(out_dir),
        base_dir=str(base_dir),
        sessions=[],
        all=True,
        jobs=1,
        dry_run=False,
        force=False,
        prune=False,
        diff=False,
        json=True,
        watch=False,
        offline=False,
        collapse_threshold=None,
        html_mode="off",
        attachment_ocr=True,
        _attachment_ocr_explicit=False,
        sanitize_html=False,
        meta=(),
        max_disk=None,
        resume_from=None,
        prune_snapshot=False,
        root=None,
        folder_name=None,
        folder_id=None,
        since=None,
        until=None,
        name_filter=None,
        list_only=False,
        links_only=True,
        attachments_only=False,
        chat_ids=(),
        print_paths=False,
        debounce=0.0,
        stall_seconds=60.0,
        fail_on_stall=False,
        tail=False,
        once=False,
        snapshot=False,
        watch_plan=False,
        drive_retries=None,
        drive_retry_base=None,
    )
    run_sync_cli(args, env)


def test_sync_parser_supports_selection_flags(tmp_path):
    captured = {}

    def fake_sync(args, env):  # noqa: ARG001
        captured["args"] = args

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.sync.run_sync_cli", fake_sync)

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

    def fake_render(args, env, *, json_output=False):  # noqa: ARG001
        captured["args"] = args

    runner = CliRunner()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("polylogue.cli.render.run_render_cli", fake_render)
    result = runner.invoke(click_cli, ["render", "input.json", "--print-paths"])
    assert result.exit_code == 0
    assert captured["args"].print_paths is True

    monkeypatch.setattr("polylogue.cli.sync.run_sync_cli", lambda args, env: captured.__setitem__("args", args))  # noqa: ARG005
    result = runner.invoke(click_cli, ["sync", "drive", "--print-paths"])
    assert result.exit_code == 0
    assert captured["args"].print_paths is True

    monkeypatch.setattr("polylogue.cli.imports.run_import_cli", lambda args, env: captured.__setitem__("args", args))  # noqa: ARG005
    result = runner.invoke(click_cli, ["import", "run", "chatgpt", "export.zip", "--print-paths"])
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
            folder_name = "Google AI Studio"
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

    args = SimpleNamespace(
        links_only=True,
        chat_ids=["chat-123"],
        name_filter=None,
        since=None,
        until=None,
        folder_name="Google AI Studio",
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

    args = SimpleNamespace(
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
    runner = CliRunner()
    result = runner.invoke(click_cli, ["render", "input.json", "--allow-dirty"])
    assert result.exit_code == 1
    assert "--allow-dirty requires --force" in result.output


def test_drive_watch_is_rejected():
    env = CommandEnv(ui=DummyUI())
    args = SimpleNamespace(provider="drive", watch=True)

    with pytest.raises(SystemExit):
        run_sync_cli(args, env)


def test_resolve_collapse_value_accepts_zero():
    settings = Settings(html_previews=False, html_theme="light", collapse_threshold=0, preferred_providers=[])

    assert resolve_collapse_value(0, settings) == 0
    assert resolve_collapse_value(None, settings) == 0


def test_should_use_plain_flags_override(monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    use_plain, reason = _should_use_plain(plain=False, interactive=False)
    assert use_plain is False
    assert reason is None

    use_plain, reason = _should_use_plain(plain=True, interactive=False)
    assert use_plain is True
    assert reason is not None

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    use_plain, reason = _should_use_plain(plain=False, interactive=True)
    assert use_plain is False
    assert reason is None

    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    use_plain, reason = _should_use_plain(plain=False, interactive=False)
    assert use_plain is True
    assert reason and "POLYLOGUE_FORCE_PLAIN" in reason
