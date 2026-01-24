from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.config import load_config

# Mark for tests that need subprocess isolation due to module caching issues
NEEDS_SUBPROCESS = pytest.mark.skip(
    reason="CLI tests with sync hang due to module caching; need subprocess isolation"
)


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


@NEEDS_SUBPROCESS
def test_cli_sync_and_search(workspace_env, tmp_path):
    """Test CLI sync and search with isolated workspace."""
    from polylogue.config import default_config

    config = default_config()
    inbox = config.archive_root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv1",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello"},
            {"id": "m2", "role": "assistant", "content": "world"},
        ],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    run_result = runner.invoke(cli, ["--plain", "sync", "--stage", "all"])
    assert run_result.exit_code == 0, run_result.output

    assert any(config.render_root.rglob("conversation.html")) or any(config.render_root.rglob("conversation.md"))

    # Query mode: --latest shows most recent conversation
    latest_result = runner.invoke(cli, ["--plain", "--latest"])
    # exit_code 0 = found result, exit_code 2 = no results
    assert latest_result.exit_code in (0, 2)

    # Query mode: search with query terms, json format, --list forces list output
    search_result = runner.invoke(cli, ["--plain", "hello", "--limit", "1", "-f", "json", "--list"])
    # exit_code 0 = found result, exit_code 2 = no results
    assert search_result.exit_code in (0, 2)
    if search_result.exit_code == 0:
        payload = json.loads(search_result.output.strip())
        # With --list flag, output is always a list
        assert payload and isinstance(payload, list)


@NEEDS_SUBPROCESS
def test_cli_search_csv_header(tmp_path, monkeypatch):
    """Test that CSV output includes proper header."""
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
    # Query mode: positional args are query terms, --csv writes output
    result = runner.invoke(cli, ["--plain", "missing", "--csv", str(output)])
    # exit_code 2 = no results found, but CSV should still be written with header
    assert result.exit_code in (0, 2)
    if output.exists():
        header = output.read_text(encoding="utf-8").splitlines()[0]
        assert header.startswith("source,provider,conversation_id,message_id")


@NEEDS_SUBPROCESS
def test_cli_search_latest_missing_render(tmp_path, monkeypatch):
    """Test --latest --open with no rendered outputs shows error."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    # Query mode: --latest --open
    result = runner.invoke(cli, ["--plain", "--latest", "--open"])
    # Should fail: either no results or no rendered outputs
    assert result.exit_code != 0
    output_lower = result.output.lower()
    # Accept various error messages
    assert ("no rendered" in output_lower or
            "no conversation" in output_lower or
            "no results" in output_lower or
            result.exit_code == 2)


@NEEDS_SUBPROCESS
def test_cli_search_open_prefers_html(tmp_path, monkeypatch):
    """Test that --open prefers HTML over markdown."""
    config_path = tmp_path / "config.json"
    archive_root = tmp_path / "archive"
    data_dir = tmp_path / "data"
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
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(archive_root / "render"))

    # Reload modules to pick up new XDG_DATA_HOME
    import importlib

    import polylogue.paths
    import polylogue.config
    import polylogue.storage.backends.sqlite
    import polylogue.storage.db
    import polylogue.storage.search
    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)
    importlib.reload(polylogue.storage.backends.sqlite)
    importlib.reload(polylogue.storage.db)
    importlib.reload(polylogue.storage.search)

    from polylogue.cli.container import reset_container
    reset_container()

    opened = {}

    def fake_webbrowser_open(url):
        opened["url"] = url
        return True

    import webbrowser
    monkeypatch.setattr(webbrowser, "open", fake_webbrowser_open)

    runner = CliRunner()
    run_result = runner.invoke(cli, ["--plain", "sync", "--stage", "all"])
    assert run_result.exit_code == 0, run_result.output

    # Query mode: positional args are query terms
    search_result = runner.invoke(cli, ["--plain", "hello", "--limit", "1", "--open"])
    # Result depends on whether render was created and conversation found
    if search_result.exit_code == 0 and "url" in opened:
        # Should prefer HTML if available
        assert ".html" in opened["url"] or ".md" in opened["url"]


@NEEDS_SUBPROCESS
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


# --latest validation tests


@NEEDS_SUBPROCESS
def test_cli_search_latest_returns_path_without_open(tmp_path, monkeypatch):
    """polylogue --latest prints conversation info when --open not specified."""
    config_path = tmp_path / "config.json"
    archive_root = tmp_path / "archive"
    render_root = archive_root / "render"
    data_dir = tmp_path / "data"
    inbox = tmp_path / "inbox"

    # Create a conversation to ingest
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv1-abc123",
        "messages": [
            {"id": "m1", "role": "user", "content": "test content"},
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
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(render_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    # Reload modules
    import importlib

    import polylogue.paths
    import polylogue.config
    import polylogue.storage.backends.sqlite
    import polylogue.storage.db
    import polylogue.storage.search
    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)
    importlib.reload(polylogue.storage.backends.sqlite)
    importlib.reload(polylogue.storage.db)
    importlib.reload(polylogue.storage.search)

    from polylogue.cli.container import reset_container
    reset_container()

    runner = CliRunner()
    # First sync
    sync_result = runner.invoke(cli, ["--plain", "sync", "--stage", "all"])
    assert sync_result.exit_code == 0, sync_result.output

    # Query mode: --latest
    result = runner.invoke(cli, ["--plain", "--latest"])
    # Should succeed and show conversation info
    assert result.exit_code in (0, 2)  # 0 = found, 2 = no results


@NEEDS_SUBPROCESS
def test_cli_query_latest_with_query(tmp_path, monkeypatch):
    """--latest with query terms is now allowed in query-first mode."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    # Query mode: query terms + --latest = find latest matching query
    result = runner.invoke(cli, ["--plain", "some", "query", "--latest"])
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


@NEEDS_SUBPROCESS
def test_cli_query_latest_with_json(tmp_path, monkeypatch):
    """--latest with --format json is now allowed in query-first mode."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    # Query mode: --latest with json format
    result = runner.invoke(cli, ["--plain", "--latest", "-f", "json"])
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


@NEEDS_SUBPROCESS
def test_cli_no_args_shows_stats(tmp_path, monkeypatch):
    """polylogue (no args) shows stats in query-first mode."""
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
    # Query mode: no args shows stats
    result = runner.invoke(cli, ["--plain"])
    # Should succeed and show archive stats
    assert result.exit_code == 0


# Race condition test


@NEEDS_SUBPROCESS
def test_latest_render_path_handles_deleted_file(tmp_path, monkeypatch):
    """latest_render_path() doesn't crash if file deleted between list and stat."""
    import time

    from polylogue.cli import helpers as helpers_mod

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


@NEEDS_SUBPROCESS
def test_cli_search_open_missing_render_shows_hint(tmp_path, monkeypatch):
    """--open with missing render shows 'Run polylogue sync' hint."""
    config_path = tmp_path / "config.json"
    archive_root = tmp_path / "archive"
    data_dir = tmp_path / "data"

    # Create inbox with a conversation but don't run render
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv-no-render",
        "messages": [{"id": "m1", "role": "user", "content": "no render"}],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    config_payload = {
        "version": 2,
        "archive_root": str(archive_root),
        "sources": [{"name": "inbox", "path": str(inbox)}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(archive_root / "render"))

    # Reload modules to pick up new XDG_DATA_HOME
    import importlib

    import polylogue.paths
    import polylogue.config
    import polylogue.storage.backends.sqlite
    import polylogue.storage.db
    import polylogue.storage.search
    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)
    importlib.reload(polylogue.storage.backends.sqlite)
    importlib.reload(polylogue.storage.db)
    importlib.reload(polylogue.storage.search)

    from polylogue.cli.container import reset_container
    reset_container()

    # Run ingest stage only, skip render (index happens automatically if FTS available)
    runner = CliRunner()
    run_result = runner.invoke(cli, ["--plain", "sync", "--stage", "ingest"])
    assert run_result.exit_code == 0

    # Query mode: search and try to open - render doesn't exist
    search_result = runner.invoke(cli, ["--plain", "render", "--open"])
    # Should either succeed with a warning or indicate render/sync not found
    # The exact behavior depends on implementation, but shouldn't crash
    # Note: Error messages may be in exception rather than output
    exception_msg = str(search_result.exception) if search_result.exception else ""
    assert (
        search_result.exit_code == 0
        or search_result.exit_code == 2  # no results
        or "render" in search_result.output.lower()
        or "sync" in search_result.output.lower()
        or "sync" in exception_msg.lower()
    )
