from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli import cli
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
            {"type": "confirm", "use_default": True},
            {"type": "confirm", "value": True},
            {"type": "input", "use_default": True},
            {"type": "input", "use_default": True},
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
    drive_sources = [source for source in config.sources if source.type == "drive"]
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
        "version": 1,
        "archive_root": str(archive_root),
        "sources": [
            {"name": "inbox", "type": "auto", "path": str(inbox)}
        ],
        "profiles": {
            "default": {
                "attachments": "download",
                "html": "auto",
                "index": True,
                "sanitize_html": False,
            }
        },
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
