"""End-to-end subprocess proofs for query-first CLI route selection."""

from __future__ import annotations

import json

import pytest

from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace
from tests.infra.source_builders import GenericConversationBuilder, InboxBuilder

pytestmark = [pytest.mark.integration, pytest.mark.query_routing]


def _run_inbox(workspace, *, cwd) -> None:
    result = run_cli(["--plain", "run", "--source", "inbox", "--stage", "all"], env=workspace["env"], cwd=cwd)
    assert result.exit_code == 0, result.output


def test_cli_query_count_route_returns_exact_count(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-count").title("Count Route").add_user("alpha route").add_assistant("beta").write_to(
        inbox / "conversation.json"
    )
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "alpha", "--count"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "1"


def test_cli_query_summary_list_json_route_returns_structured_rows(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-list").title("List Route").add_user("searchable alpha").add_assistant("response").write_to(
        inbox / "conversation.json"
    )
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "searchable", "--list", "-f", "json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert len(rows) == 1
    assert str(rows[0]["id"]).endswith("conv-list")
    assert rows[0]["title"] == "List Route"


def test_cli_query_stream_route_emits_json_lines_header_messages_footer(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-stream").title("Stream Route").add_user("stream alpha").add_assistant("stream beta").write_to(
        inbox / "conversation.json"
    )
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "--latest", "--stream", "-f", "json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert records[0]["type"] == "header"
    assert str(records[0]["conversation_id"]).endswith("conv-stream")
    assert [record["type"] for record in records[1:-1]] == ["message", "message"]
    assert records[-1] == {"type": "footer", "message_count": 2}


def test_cli_query_stats_by_provider_reports_provider_groups(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    (
        InboxBuilder(inbox)
        .add_chatgpt_export("conv-chatgpt", title="ChatGPT Route")
        .add_codex_conversation("conv-codex", title="Codex Route")
        .build()
    )
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "--stats-by", "provider"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    output = result.output.lower()
    assert "matched: 2 conversations" in output
    assert "chatgpt" in output
    assert "unknown" in output


def test_cli_no_args_stats_surface_still_works(tmp_path):
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(["--plain"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert "archive:" in result.output.lower()
    assert "sources:" in result.output.lower()
