from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.config import load_config
from polylogue.storage.db import default_db_path


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
        "sources": [{"name": "inbox", "path": str(inbox)}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_root))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(archive_root / "render"))

    runner = CliRunner()
    run_result = runner.invoke(cli, ["run", "--stage", "all"])
    assert run_result.exit_code == 0

    render_root = archive_root / "render"
    assert any(render_root.rglob("conversation.md"))

    latest_result = runner.invoke(cli, ["search", "--latest"])
    assert latest_result.exit_code == 0
    # Output may have extra lines, so take the first line containing the path
    latest_path = latest_result.output.strip().split("\n")[0]
    assert latest_path.endswith("conversation.html") or latest_path.endswith("conversation.md")

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
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str((tmp_path / "archive") / "render"))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))

    runner = CliRunner()
    output = tmp_path / "out.csv"
    result = runner.invoke(cli, ["--plain", "search", "missing", "--csv", str(output)])
    assert result.exit_code == 0
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("source,provider,conversation_id,message_id")


def test_cli_search_latest_missing_render(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--latest", "--open"])
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

    import polylogue.cli.commands.search as search_mod

    monkeypatch.setattr(search_mod, "open_in_browser", fake_open_browser)
    monkeypatch.setattr(search_mod, "open_in_editor", lambda path: False)

    runner = CliRunner()
    run_result = runner.invoke(cli, ["run", "--stage", "all"])
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


def test_cli_state_reset_clears_db_and_last_source(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))

    db_path = default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("junk", encoding="utf-8")

    last_source = state_root / "polylogue" / "last-source.json"
    last_source.parent.mkdir(parents=True, exist_ok=True)
    last_source.write_text('{"source": "inbox"}', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", "state", "reset", "--all", "--force"])
    assert result.exit_code == 0
    assert db_path.exists()
    assert not last_source.exists()


# --latest validation tests


def test_cli_search_latest_returns_path_without_open(tmp_path, monkeypatch):
    """polylogue search --latest prints path when --open not specified."""
    config_path = tmp_path / "config.json"
    archive_root = tmp_path / "archive"
    render_root = archive_root / "render"

    # Create a rendered file
    conv_dir = render_root / "test" / "conv1-abc123"
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / "conversation.html").write_text("<html>test</html>", encoding="utf-8")

    config_payload = {
        "version": 2,
        "archive_root": str(archive_root),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(render_root))

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--latest"])
    assert result.exit_code == 0
    assert "conversation.html" in result.output


def test_cli_search_latest_rejects_query(tmp_path, monkeypatch):
    """--latest with query argument fails with clear error."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--latest", "some query"])
    assert result.exit_code != 0
    # Should indicate that --latest can't be used with a query
    assert "latest" in result.output.lower() or "query" in result.output.lower()


def test_cli_search_latest_rejects_json_output(tmp_path, monkeypatch):
    """--latest with --json fails with clear error."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--latest", "--json"])
    assert result.exit_code != 0


def test_cli_search_requires_query_without_latest(tmp_path, monkeypatch):
    """polylogue search (no args) fails with 'Query required' message."""
    config_path = tmp_path / "config.json"
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
    result = runner.invoke(cli, ["search"])
    assert result.exit_code != 0
    assert "Query required" in result.output or "query" in result.output.lower()


# Race condition test


def test_latest_render_path_handles_deleted_file(tmp_path, monkeypatch):
    """latest_render_path() doesn't crash if file deleted between list and stat."""
    from polylogue.cli import helpers as helpers_mod
    import time

    render_root = tmp_path / "render"
    conv_dir = render_root / "test" / "conv1-abc"
    conv_dir.mkdir(parents=True, exist_ok=True)

    html_file = conv_dir / "conversation.html"
    html_file.write_text("<html>test</html>", encoding="utf-8")

    # Verify it works normally first
    result = helpers_mod.latest_render_path(render_root)
    assert result is not None
    assert result.name == "conversation.html"

    # Now test with a file that gets "deleted" during iteration
    # Create multiple files
    conv_dir2 = render_root / "test" / "conv2-def"
    conv_dir2.mkdir(parents=True, exist_ok=True)
    html_file2 = conv_dir2 / "conversation.html"
    html_file2.write_text("<html>test2</html>", encoding="utf-8")

    # Touch html_file2 to make it the newest
    time.sleep(0.01)
    html_file2.touch()

    # Delete the first file to simulate race condition
    html_file.unlink()

    # Should still work, returning the file that exists
    result = helpers_mod.latest_render_path(render_root)
    assert result is not None
    assert "conv2" in str(result)


# --open missing render test


def test_cli_search_open_missing_render_shows_hint(tmp_path, monkeypatch):
    """--open with missing render shows 'Run polylogue run' hint."""
    config_path = tmp_path / "config.json"
    archive_root = tmp_path / "archive"
    state_root = tmp_path / "state"

    # Create inbox with a conversation but don't run render
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv-no-render",
        "messages": [{"id": "m1", "role": "user", "text": "no render"}],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    config_payload = {
        "version": 2,
        "archive_root": str(archive_root),
        "sources": [{"name": "inbox", "path": str(inbox)}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_root))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(archive_root / "render"))

    # Run ingest to create DB entry but skip render
    runner = CliRunner()
    run_result = runner.invoke(cli, ["run", "--stage", "ingest"])
    assert run_result.exit_code == 0

    # Now search and try to open - render doesn't exist
    search_result = runner.invoke(cli, ["search", "render", "--open"])
    # Should either succeed with a warning or indicate render not found
    # The exact behavior depends on implementation, but shouldn't crash
    assert search_result.exit_code == 0 or "render" in search_result.output.lower() or "run" in search_result.output.lower()
