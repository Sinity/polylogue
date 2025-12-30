from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli import cli
import polylogue.cli.click_app as click_app
from polylogue.config import load_config


def _write_prompt_file(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def test_cli_config_init_interactive_adds_drive(tmp_path, monkeypatch):
    config_path = tmp_path / "config" / "config.json"
    data_root = tmp_path / "data"
    state_root = tmp_path / "state"
    prompt_file = tmp_path / "prompts.jsonl"
    _write_prompt_file(
        prompt_file,
        [
            {"type": "confirm", "value": True},
        ],
    )

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("POLYLOGUE_TEST_PROMPT_FILE", str(prompt_file))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_root))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))

    runner = CliRunner()
    result = runner.invoke(cli, ["--interactive", "config", "init", "--interactive"])
    assert result.exit_code == 0

    config = load_config(config_path)
    drive_sources = [source for source in config.sources if source.folder]
    assert drive_sources
    assert drive_sources[0].folder == "Google AI Studio"


def test_cli_run_and_export(tmp_path, monkeypatch):
    config_path = tmp_path / "config" / "config.json"
    data_root = tmp_path / "data"
    state_root = tmp_path / "state"
    archive_root = tmp_path / "archive"

    inbox = data_root / "polylogue" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv1",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello"},
            {"id": "m2", "role": "assistant", "content": "world"},
        ],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "version": 2,
        "archive_root": str(archive_root),
        "sources": [
            {"name": "inbox", "path": str(inbox)}
        ],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_root))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

    runner = CliRunner()
    run_result = runner.invoke(cli, ["run", "--stage", "all", "--no-plan"])
    assert run_result.exit_code == 0

    render_root = archive_root / "render"
    assert any(render_root.rglob("conversation.md"))

    search_result = runner.invoke(cli, ["--plain", "search", "hello", "--limit", "1", "--json"])
    assert search_result.exit_code == 0
    payload = json.loads(search_result.output.strip())
    assert payload and isinstance(payload[0].get("conversation_path"), str)

    export_path = tmp_path / "export.jsonl"
    export_result = runner.invoke(cli, ["export", "--out", str(export_path)])
    assert export_result.exit_code == 0
    assert export_path.exists()


def test_cli_search_csv_header(tmp_path, monkeypatch):
    config_path = tmp_path / "config" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    state_root = tmp_path / "state"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))

    runner = CliRunner()
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, ["--plain", "search", "missing", "--csv", str(output)])
    assert result.exit_code != 0

    # create index and retry with no hits
    runner.invoke(cli, ["--plain", "config", "show"])
    runner.invoke(cli, ["--plain", "run", "--stage", "index", "--no-plan"])
    result = runner.invoke(cli, ["--plain", "search", "missing", "--csv", str(output)])
    assert result.exit_code == 0
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("provider,conversation_id,message_id")


def test_cli_open_missing_render(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["open", "--open"])
    assert result.exit_code != 0
    assert "no rendered outputs found" in result.output


def test_cli_search_open_prefers_html(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    archive_root = tmp_path / "archive"
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv-html",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello html"},
        ],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")
    config_payload = {
        "version": 2,
        "archive_root": str(archive_root),
        "sources": [{"name": "inbox", "path": str(inbox)}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    opened = {}

    def fake_open_browser(path):
        opened["path"] = path
        return True

    monkeypatch.setattr(click_app, "open_in_browser", fake_open_browser)
    monkeypatch.setattr(click_app, "open_in_editor", lambda path: False)

    runner = CliRunner()
    run_result = runner.invoke(cli, ["run", "--stage", "all", "--no-plan"])
    assert run_result.exit_code == 0
    search_result = runner.invoke(cli, ["search", "hello", "--limit", "1", "--open"])
    assert search_result.exit_code == 0
    assert opened["path"].suffix == ".html"


def test_cli_config_set_invalid(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["config", "set", "unknown.key", "value"])
    assert result.exit_code != 0
    result = runner.invoke(cli, ["config", "set", "source.missing.type", "auto"])
    assert result.exit_code != 0
