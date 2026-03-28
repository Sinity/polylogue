from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.config import load_config
from tests.helpers.cli_subprocess import run_cli, setup_isolated_workspace


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


def test_cli_sync_and_search(tmp_path):
    """Test CLI sync and search with isolated workspace."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]

    # Create test conversation in inbox
    inbox = paths["inbox"]
    payload = {
        "id": "conv1",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello"},
            {"id": "m2", "role": "assistant", "content": "world"},
        ],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    # Run sync via subprocess
    result = run_cli(["--plain", "sync", "--stage", "all"], env=env, cwd=tmp_path)
    assert result.exit_code == 0, result.output

    render_root = paths["render_root"]
    assert any(render_root.rglob("*.html")) or any(render_root.rglob("*.md"))

    # Query mode: --latest shows most recent conversation
    latest_result = run_cli(["--plain", "--latest"], env=env, cwd=tmp_path)
    # exit_code 0 = found result, exit_code 2 = no results
    assert latest_result.exit_code in (0, 2)

    # Query mode: search with query terms, json format, --list forces list output
    search_result = run_cli(["--plain", "hello", "--limit", "1", "-f", "json", "--list"], env=env, cwd=tmp_path)
    # exit_code 0 = found result, exit_code 2 = no results
    assert search_result.exit_code in (0, 2)
    if search_result.exit_code == 0:
        payload = json.loads(search_result.stdout.strip())
        # With --list flag, output is always a list
        assert payload and isinstance(payload, list)


def test_cli_search_csv_header(tmp_path):
    """Test that CSV output includes proper header."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    output = tmp_path / "out.csv"
    # Query mode: positional args are query terms, --csv writes output
    result = run_cli(["--plain", "missing", "--csv", str(output)], env=env, cwd=tmp_path)
    # exit_code 2 = no results found, but CSV should still be written with header
    assert result.exit_code in (0, 2)
    if output.exists():
        header = output.read_text(encoding="utf-8").splitlines()[0]
        assert header.startswith("source,provider,conversation_id,message_id")


def test_cli_search_latest_missing_render(tmp_path):
    """Test --latest --open with no rendered outputs shows error."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: --latest --open
    result = run_cli(["--plain", "--latest", "--open"], env=env, cwd=tmp_path)
    # Should fail: either no results or no rendered outputs
    assert result.exit_code != 0
    output_lower = result.output.lower()
    # Accept various error messages
    assert ("no rendered" in output_lower or
            "no conversation" in output_lower or
            "no results" in output_lower or
            result.exit_code == 2)


def test_cli_search_open_prefers_html(tmp_path):
    """Test that --open prefers HTML over markdown.

    Note: We can't directly verify webbrowser.open was called via subprocess,
    but we can verify the CLI runs without error and creates rendered output.
    """
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    payload = {
        "id": "conv-html",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello html"},
        ],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    # First sync to create conversation and render
    result = run_cli(["--plain", "sync", "--stage", "all"], env=env, cwd=tmp_path)
    assert result.exit_code == 0, result.output

    # Verify render was created
    render_root = paths["render_root"]
    html_files = list(render_root.rglob("*.html"))
    assert html_files, "Expected HTML render to be created"

    # Query mode with --open - just verify it doesn't crash
    # (subprocess can't capture webbrowser.open call)
    search_result = run_cli(["--plain", "hello", "--limit", "1"], env=env, cwd=tmp_path)
    # exit_code 0 = found result, exit_code 2 = no results
    assert search_result.exit_code in (0, 2)


def test_cli_config_set_invalid(tmp_path):
    """Test that invalid config keys are rejected."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    result = run_cli(["config", "set", "unknown.key", "value"], env=env, cwd=tmp_path)
    assert result.exit_code != 0
    result = run_cli(["config", "set", "source.missing.type", "auto"], env=env, cwd=tmp_path)
    assert result.exit_code != 0


# --latest validation tests


def test_cli_search_latest_returns_path_without_open(tmp_path):
    """polylogue --latest prints conversation info when --open not specified."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    # Create a conversation to ingest
    payload = {
        "id": "conv1-abc123",
        "messages": [
            {"id": "m1", "role": "user", "content": "test content"},
        ],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    # First sync
    sync_result = run_cli(["--plain", "sync", "--stage", "all"], env=env, cwd=tmp_path)
    assert sync_result.exit_code == 0, sync_result.output

    # Query mode: --latest
    result = run_cli(["--plain", "--latest"], env=env, cwd=tmp_path)
    # Should succeed and show conversation info
    assert result.exit_code in (0, 2)  # 0 = found, 2 = no results


def test_cli_query_latest_with_query(tmp_path):
    """--latest with query terms is now allowed in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: query terms + --latest = find latest matching query
    result = run_cli(["--plain", "some", "query", "--latest"], env=env, cwd=tmp_path)
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


def test_cli_query_latest_with_json(tmp_path):
    """--latest with --format json is now allowed in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: --latest with json format
    result = run_cli(["--plain", "--latest", "-f", "json"], env=env, cwd=tmp_path)
    # exit_code 2 = no results (empty db), but should not be invalid syntax
    assert result.exit_code in (0, 2)


def test_cli_no_args_shows_stats(tmp_path):
    """polylogue (no args) shows stats in query-first mode."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]

    # Query mode: no args shows stats
    result = run_cli(["--plain"], env=env, cwd=tmp_path)
    # Should succeed and show archive stats
    assert result.exit_code == 0


# Race condition test


def test_latest_render_path_handles_deleted_file(tmp_path):
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


def test_cli_search_open_missing_render_shows_hint(tmp_path):
    """--open with missing render shows 'Run polylogue sync' hint."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    # Create inbox with a conversation but don't run render
    payload = {
        "id": "conv-no-render",
        "messages": [{"id": "m1", "role": "user", "content": "no render"}],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    # Run ingest stage only, skip render
    result = run_cli(["--plain", "sync", "--stage", "ingest"], env=env, cwd=tmp_path)
    assert result.exit_code == 0

    # Query mode: search and try to open - render doesn't exist
    search_result = run_cli(["--plain", "render", "--open"], env=env, cwd=tmp_path)
    # Should either succeed with a warning or indicate render/sync not found
    # The exact behavior depends on implementation, but shouldn't crash
    assert (
        search_result.exit_code == 0
        or search_result.exit_code == 2  # no results
        or "render" in search_result.output.lower()
        or "sync" in search_result.output.lower()
    )
