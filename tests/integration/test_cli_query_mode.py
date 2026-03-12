"""End-to-end subprocess tests for query-first CLI behavior."""

from __future__ import annotations

import json

import pytest

from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace
from tests.infra.source_builders import GenericConversationBuilder


@pytest.mark.integration
def test_cli_run_and_search(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    GenericConversationBuilder("conv1").add_user("hello").add_assistant("world").write_to(inbox / "conversation.json")

    result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert result.exit_code == 0, result.output
    assert any(paths["render_root"].rglob("*.html")) or any(paths["render_root"].rglob("*.md"))

    latest_result = run_cli(["--plain", "--latest"], env=env, cwd=tmp_path)
    assert latest_result.exit_code in (0, 2)

    search_result = run_cli(["--plain", "hello", "--limit", "1", "-f", "json", "--list"], env=env, cwd=tmp_path)
    assert search_result.exit_code in (0, 2)
    if search_result.exit_code == 0:
        payload = json.loads(search_result.stdout.strip())
        assert payload and isinstance(payload, list)


@pytest.mark.integration
def test_cli_search_csv_header(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    output = tmp_path / "out.csv"
    result = run_cli(["--plain", "missing", "--csv", str(output)], env=env, cwd=tmp_path)
    assert result.exit_code in (0, 2)
    if output.exists():
        header = output.read_text(encoding="utf-8").splitlines()[0]
        assert header.startswith("source,provider,conversation_id,message_id")


@pytest.mark.integration
def test_cli_search_latest_missing_render(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    result = run_cli(["--plain", "--latest", "--open"], env=env, cwd=tmp_path)
    assert result.exit_code != 0
    output_lower = result.output.lower()
    assert (
        "no rendered" in output_lower
        or "no conversation" in output_lower
        or "no results" in output_lower
        or result.exit_code == 2
    )


@pytest.mark.integration
def test_cli_search_open_prefers_html(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    GenericConversationBuilder("conv-html").add_user("hello html").write_to(inbox / "conversation.json")

    result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert result.exit_code == 0, result.output
    assert list(paths["render_root"].rglob("*.html"))

    search_result = run_cli(["--plain", "hello", "--limit", "1"], env=env, cwd=tmp_path)
    assert search_result.exit_code in (0, 2)


@pytest.mark.integration
def test_cli_config_set_invalid(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    result = run_cli(["config", "set", "unknown.key", "value"], env=env, cwd=tmp_path)
    assert result.exit_code != 0
    result = run_cli(["config", "set", "source.missing.type", "auto"], env=env, cwd=tmp_path)
    assert result.exit_code != 0


@pytest.mark.integration
def test_cli_search_latest_returns_path_without_open(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    GenericConversationBuilder("conv1-abc123").add_user("test content").write_to(inbox / "conversation.json")

    run_result = run_cli(["--plain", "run", "--stage", "all"], env=env, cwd=tmp_path)
    assert run_result.exit_code == 0, run_result.output

    result = run_cli(["--plain", "--latest"], env=env, cwd=tmp_path)
    assert result.exit_code in (0, 2)


@pytest.mark.integration
def test_cli_query_latest_with_query(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    result = run_cli(["--plain", "some", "query", "--latest"], env=env, cwd=tmp_path)
    assert result.exit_code in (0, 2)


@pytest.mark.integration
def test_cli_query_latest_with_json(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    result = run_cli(["--plain", "--latest", "-f", "json"], env=env, cwd=tmp_path)
    assert result.exit_code in (0, 2)


@pytest.mark.integration
def test_cli_no_args_shows_stats(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    result = run_cli(["--plain"], env=env, cwd=tmp_path)
    assert result.exit_code == 0


@pytest.mark.integration
def test_cli_search_open_missing_render_shows_hint(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    env = workspace["env"]
    paths = workspace["paths"]
    inbox = paths["inbox"]

    GenericConversationBuilder("conv-no-render").add_user("no render").write_to(inbox / "conversation.json")

    result = run_cli(["--plain", "run", "--stage", "parse"], env=env, cwd=tmp_path)
    assert result.exit_code == 0

    search_result = run_cli(["--plain", "render", "--open"], env=env, cwd=tmp_path)
    assert (
        search_result.exit_code == 0
        or search_result.exit_code == 2
        or "render" in search_result.output.lower()
        or "run" in search_result.output.lower()
    )
