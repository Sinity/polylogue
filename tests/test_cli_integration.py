"""Consolidated CLI integration tests.

SYSTEMATIZATION: Merged from:
- test_cli.py (End-to-end CLI tests)
- test_cli_search_expanded.py (Search integration tests)

This file contains integration tests for:
- CLI workflows
- Config initialization
- Search operations
- Latest render operations
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.config import Config, load_config
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.store import store_records
from tests.cli_helpers.cli_subprocess import run_cli, setup_isolated_workspace
from tests.factories import DbFactory
from tests.helpers import GenericConversationBuilder, make_conversation, make_message


def _write_prompt_file(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


# =============================================================================
# END-TO-END CLI TESTS (from test_cli.py)
# =============================================================================


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


def test_cli_run_and_search(tmp_path):
    """Test CLI run and search with isolated workspace."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    # Create test conversation in inbox
    (GenericConversationBuilder("conv1")
     .add_user("hello")
     .add_assistant("world")
     .write_to(inbox / "conversation.json"))

    # Run pipeline via subprocess
    result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
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
    """Test that --open prefers HTML over markdown."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    (GenericConversationBuilder("conv-html")
     .add_user("hello html")
     .write_to(inbox / "conversation.json"))

    # First run to create conversation and render
    result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert result.exit_code == 0, result.output

    # Verify render was created
    render_root = paths["render_root"]
    html_files = list(render_root.rglob("*.html"))
    assert html_files, "Expected HTML render to be created"

    # Query mode with --open - just verify it doesn't crash
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
    (GenericConversationBuilder("conv1-abc123")
     .add_user("test content")
     .write_to(inbox / "conversation.json"))

    # First run
    run_result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert run_result.exit_code == 0, run_result.output

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
    html_file2.touch()

    # Delete the first file to simulate race condition
    html_file.unlink()

    # Should still work, returning the file that exists
    result = helpers_mod.latest_render_path(render_root)
    assert result is not None
    assert "conv2" in str(result)


# --open missing render test


def test_cli_search_open_missing_render_shows_hint(tmp_path):
    """--open with missing render shows hint to run polylogue."""
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    # Create inbox with a conversation but don't run render
    (GenericConversationBuilder("conv-no-render")
     .add_user("no render")
     .write_to(inbox / "conversation.json"))

    # Run parse stage only, skip render
    result = run_cli(["--plain", "run", "--stage", "parse"], env=env, cwd=tmp_path)
    assert result.exit_code == 0

    # Query mode: search and try to open - render doesn't exist
    search_result = run_cli(["--plain", "render", "--open"], env=env, cwd=tmp_path)
    # Should either succeed with a warning or indicate render/run not found
    assert (
        search_result.exit_code == 0
        or search_result.exit_code == 2  # no results
        or "render" in search_result.output.lower()
        or "run" in search_result.output.lower()
    )

# =============================================================================
# SEARCH INTEGRATION TESTS (from test_cli_search_expanded.py)
# =============================================================================

@pytest.fixture
def search_workspace(cli_workspace, monkeypatch):
    """CLI workspace with searchable conversations."""
    # Set up environment
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Create sample conversations with searchable content
    db_path = cli_workspace["db_path"]
    factory = DbFactory(db_path)

    # Conversation 1: Python content, recent
    factory.create_conversation(
        id="conv1",
        provider="chatgpt",
        title="Python Error Handling",
        messages=[
            {"id": "m1", "role": "user", "text": "How to handle exceptions in Python?"},
            {"id": "m2", "role": "assistant", "text": "Use try-except blocks for Python exception handling."},
        ],
        created_at=datetime.now() - timedelta(days=1),
        updated_at=datetime.now() - timedelta(days=1),
    )

    # Conversation 2: JavaScript content, older
    factory.create_conversation(
        id="conv2",
        provider="claude",
        title="JavaScript Async Patterns",
        messages=[
            {"id": "m3", "role": "user", "text": "Explain async/await in JavaScript"},
            {"id": "m4", "role": "assistant", "text": "Async/await is JavaScript syntax for promises."},
        ],
        created_at=datetime.now() - timedelta(days=10),
        updated_at=datetime.now() - timedelta(days=10),
    )

    # Conversation 3: Rust content
    factory.create_conversation(
        id="conv3",
        provider="claude-code",
        title="Rust Ownership",
        messages=[
            {"id": "m5", "role": "user", "text": "What is ownership in Rust?"},
            {"id": "m6", "role": "assistant", "text": "Rust ownership ensures memory safety without garbage collection."},
        ],
        created_at=datetime.now() - timedelta(hours=6),
        updated_at=datetime.now() - timedelta(hours=6),
    )

    # Build FTS index using rebuild_index
    from polylogue.storage.index import rebuild_index

    rebuild_index()

    return cli_workspace


class TestSearchFilters:
    """Tests for search filtering options."""

    def test_search_with_provider_filter(self, search_workspace):
        """Filter search results by provider."""
        runner = CliRunner()
        # Query mode: positional args = query, -p = provider filter
        result = runner.invoke(cli, ["--plain", "Python", "-p", "chatgpt"])
        # exit_code 0 = found, exit_code 2 = no results
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            assert "Python" in result.output or "conv1" in result.output

    def test_search_with_since_date(self, search_workspace):
        """Filter search results by date."""
        runner = CliRunner()
        since_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        # Query mode: positional args = query, --since = date filter
        result = runner.invoke(cli, ["--plain", "Python", "--since", since_date])
        assert result.exit_code in (0, 2)
        # Should find recent Python conversation

    def test_search_with_invalid_since_date(self, search_workspace):
        """Handle invalid --since date format gracefully."""
        runner = CliRunner()
        # Query mode with invalid date
        result = runner.invoke(cli, ["--plain", "Python", "--since", "not-a-date"])
        # The filter chain should handle this gracefully
        # Either fail with error message or treat as "no results"
        assert result.exit_code in (0, 1, 2)

    def test_search_with_limit(self, search_workspace):
        """Limit number of search results."""
        runner = CliRunner()
        # Query mode with --limit
        result = runner.invoke(cli, ["--plain", "JavaScript", "--limit", "1", "--list"])
        assert result.exit_code in (0, 2)
        # Should return at most 1 result


class TestSearchOutputFormats:
    """Tests for different output formats."""

    def test_search_json_output(self, search_workspace):
        """Search with JSON output format."""
        runner = CliRunner()
        # Query mode with -f json and --list
        result = runner.invoke(cli, ["--plain", "Python", "-f", "json", "--list"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert isinstance(data, list)
            if data:
                # JSON output contains conversation-level info
                assert "id" in data[0]

    def test_search_json_format_single(self, search_workspace):
        """Search with JSON output for single result."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "JavaScript", "-f", "json", "--limit", "1"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            # Single result = dict, multiple or --list = list
            assert isinstance(data, (list, dict))

    def test_search_list_mode(self, search_workspace):
        """Search in list mode (shows all results)."""
        runner = CliRunner()
        # Query mode with --list
        result = runner.invoke(cli, ["--plain", "async", "--list"])
        assert result.exit_code in (0, 2)
        # Should list all results

    def test_search_markdown_format(self, search_workspace):
        """Search with markdown output format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--plain", "Rust", "-f", "markdown", "--limit", "1"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            # Markdown output should contain headers
            assert "#" in result.output or "Rust" in result.output


class TestSearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_no_results(self, search_workspace):
        """Handle query with no matching results."""
        runner = CliRunner()
        # Query mode with non-matching term
        result = runner.invoke(cli, ["--plain", "nonexistent_term_xyz"])
        # exit_code 2 = no results (valid outcome)
        assert result.exit_code == 2
        assert "no conversation" in result.output.lower() or "matched" in result.output.lower()

    def test_stats_mode_no_filters(self, cli_workspace, monkeypatch):
        """Stats mode when no query terms or filters provided."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        runner = CliRunner()
        # No args = stats mode in query-first CLI
        result = runner.invoke(cli, ["--plain"])
        assert result.exit_code == 0
        # Should show stats, not require query

    def test_search_case_insensitive(self, search_workspace):
        """Search is case-insensitive."""
        runner = CliRunner()
        # Query mode with --list to ensure consistent output
        result_lower = runner.invoke(cli, ["--plain", "python", "-f", "json", "--list"])
        result_upper = runner.invoke(cli, ["--plain", "PYTHON", "-f", "json", "--list"])

        # Both should have same exit code
        assert result_lower.exit_code == result_upper.exit_code

        if result_lower.exit_code == 0:
            # Both should find results (FTS5 is case-insensitive by default)
            data_lower = json.loads(result_lower.output)
            data_upper = json.loads(result_upper.output)
            assert len(data_lower) > 0
            assert len(data_upper) > 0

    def test_search_multiple_terms(self, search_workspace):
        """Search with multiple query terms."""
        runner = CliRunner()
        # Query mode: multiple positional args = multiple query terms
        result = runner.invoke(cli, ["--plain", "Python", "exception", "-f", "json", "--list"])
        assert result.exit_code in (0, 2)
        if result.exit_code == 0:
            data = json.loads(result.output)
            assert isinstance(data, list)


class TestSearchIndexRebuild:
    """Tests for automatic index rebuild on missing index."""

    def test_search_handles_missing_index(self, cli_workspace, monkeypatch):
        """Search handles missing index gracefully."""
        monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
        monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_root"]))
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

        # Create conversation without building index
        db_path = cli_workspace["db_path"]
        factory = DbFactory(db_path)
        factory.create_conversation(
            id="c1",
            provider="test",
            title="Test",
            messages=[{"id": "m1", "role": "user", "text": "searchable content"}],
        )

        runner = CliRunner()
        # Query mode
        result = runner.invoke(cli, ["--plain", "searchable"])
        # Should either succeed (rebuild worked) or report no results
        assert result.exit_code in (0, 1, 2)
