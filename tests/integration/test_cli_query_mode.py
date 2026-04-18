"""End-to-end subprocess proofs for query-first CLI route selection."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tests.infra.cli_subprocess import IsolatedWorkspace, run_cli, setup_isolated_workspace
from tests.infra.source_builders import GenericConversationBuilder, InboxBuilder

pytestmark = [pytest.mark.integration, pytest.mark.query_routing]


def _run_inbox(workspace: IsolatedWorkspace, *, cwd: Path) -> None:
    result = run_cli(["--plain", "run", "--source", "inbox"], env=workspace["env"], cwd=cwd)
    assert result.exit_code == 0, result.output


def _run_completion(workspace: IsolatedWorkspace, *, cwd: Path, words: str, cword: int) -> list[dict[str, str]]:
    result = run_cli(
        [],
        env={
            **workspace["env"],
            "_POLYLOGUE_COMPLETE": "zsh_complete",
            "COMP_WORDS": words,
            "COMP_CWORD": str(cword),
        },
        cwd=cwd,
    )
    assert result.exit_code == 0, result.output
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) % 3 == 0, lines
    records: list[dict[str, str]] = []
    for index in range(0, len(lines), 3):
        records.append(
            {
                "type": lines[index],
                "value": lines[index + 1].replace("\\:", ":"),
                "help": lines[index + 2],
            }
        )
    return records


def test_cli_query_count_route_returns_exact_count(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-count").title("Count Route").add_user("alpha route").add_assistant(
        "beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "alpha", "count"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "1"


def test_cli_query_summary_list_json_route_returns_structured_rows(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-list").title("List Route").add_user("searchable alpha").add_assistant(
        "response"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "searchable", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert len(rows) == 1
    assert str(rows[0]["id"]).endswith("conv-list")
    assert rows[0]["title"] == "List Route"


def test_cli_query_summary_list_json_no_results_still_returns_json(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(["--plain", "searchable", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 2, result.output
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == "no_results"
    assert payload["message"] == "No conversations matched."


def test_cli_query_open_print_path_json_no_results_still_returns_json(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(
        ["--plain", "--format", "json", "--latest", "open", "--print-path"], env=workspace["env"], cwd=tmp_path
    )

    assert result.exit_code == 2, result.output
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == "no_results"
    assert payload["message"] == "No conversations matched."


def test_cli_query_stats_json_empty_archive_still_returns_json(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(["--plain", "stats", "-f", "json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 2, result.output
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == "no_results"
    assert payload["message"] == "No conversations in archive."


def test_cli_query_stream_route_emits_json_lines_header_messages_footer(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-stream").title("Stream Route").add_user("stream alpha").add_assistant(
        "stream beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "--latest", "--stream", "-f", "json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert records[0]["type"] == "header"
    assert str(records[0]["conversation_id"]).endswith("conv-stream")
    assert [record["type"] for record in records[1:-1]] == ["message", "message"]
    assert records[-1] == {"type": "footer", "message_count": 2}


def test_cli_query_open_print_path_returns_render_target_without_launching(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-open").title("Open Route").add_user("open alpha").add_assistant(
        "open beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)
    render_result = run_cli(["--plain", "run", "render"], env=workspace["env"], cwd=tmp_path)
    assert render_result.exit_code == 0, render_result.output

    result = run_cli(["--plain", "--latest", "open", "--print-path"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    render_path = result.stdout.strip()
    assert render_path.endswith("conversation.html") or render_path.endswith("conversation.md")


def test_cli_query_open_print_path_accepts_query_after_verb(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-open-by-id").title("Open By Id").add_user("open alpha").add_assistant(
        "open beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)
    render_result = run_cli(["--plain", "run", "render"], env=workspace["env"], cwd=tmp_path)
    assert render_result.exit_code == 0, render_result.output
    list_result = run_cli(["--plain", "--latest", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)
    assert list_result.exit_code == 0, list_result.output
    conv_id = json.loads(list_result.stdout)[0]["id"]

    result = run_cli(
        ["--plain", "open", "--print-path", conv_id],
        env=workspace["env"],
        cwd=tmp_path,
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    render_path = result.stdout.strip()
    assert render_path.endswith("conversation.html") or render_path.endswith("conversation.md")


def test_cli_query_stats_by_provider_reports_provider_groups(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    (
        InboxBuilder(inbox)
        .add_chatgpt_export("conv-chatgpt", title="ChatGPT Route")
        .add_codex_conversation("conv-codex", title="Codex Route")
        .build()
    )
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(["--plain", "stats", "--by", "provider"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    output = result.output.lower()
    assert "matched: 2 conversations (by provider)" in output
    assert "total" in output
    assert "chatgpt" in output
    assert "unknown" in output


def test_cli_query_stats_by_provider_accepts_limit_after_verb(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-stats-limit").title("Stats Limit").add_user("alpha").add_assistant(
        "beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)

    result = run_cli(
        ["--plain", "stats", "--by", "provider", "--limit", "1", "-f", "json"], env=workspace["env"], cwd=tmp_path
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dimension"] == "provider"
    assert payload["summary"]["conversations"] == 1
    assert len(payload["rows"]) == 1


def test_cli_completion_id_offers_recent_conversation_ids_with_titles(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-complete-id").title("Completion Target").add_user("alpha").add_assistant(
        "beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)
    list_result = run_cli(["--plain", "--latest", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)
    assert list_result.exit_code == 0, list_result.output
    conv_id = json.loads(list_result.stdout)[0]["id"]

    records = _run_completion(workspace, cwd=tmp_path, words="polylogue --id conv", cword=2)

    match = next(record for record in records if record["value"] == conv_id)
    assert match["type"] == "plain"
    assert "Completion Target" in match["help"]


def test_cli_completion_open_target_offers_recent_conversation_ids(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-open-complete").title("Open Completion").add_user("alpha").add_assistant(
        "beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)
    list_result = run_cli(["--plain", "--latest", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)
    assert list_result.exit_code == 0, list_result.output
    conv_id = json.loads(list_result.stdout)[0]["id"]

    records = _run_completion(workspace, cwd=tmp_path, words="polylogue open conv", cword=2)

    assert any(record["value"] == conv_id for record in records)


def test_cli_completion_tag_and_tool_values_are_archive_backed(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-complete-meta").title("Completion Metadata").add_user("alpha").add_assistant(
        "beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)
    list_result = run_cli(["--plain", "--latest", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)
    assert list_result.exit_code == 0, list_result.output
    conv_id = json.loads(list_result.stdout)[0]["id"]

    db_path = workspace["paths"]["db_path"]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET metadata = json(?) WHERE conversation_id = ?",
            (json.dumps({"tags": ["review"]}), conv_id),
        )
        message_id = conn.execute(
            "SELECT message_id FROM messages WHERE conversation_id = ? ORDER BY sort_key ASC LIMIT 1",
            (conv_id,),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO action_events (
                event_id,
                conversation_id,
                message_id,
                materializer_version,
                source_block_id,
                timestamp,
                sort_key,
                sequence_index,
                provider_name,
                action_kind,
                tool_name,
                normalized_tool_name,
                tool_id,
                affected_paths_json,
                cwd_path,
                branch_names_json,
                command,
                query_text,
                url,
                output_text,
                search_text
            ) VALUES (?, ?, ?, 1, NULL, NULL, 1.0, 0, 'unknown', 'shell', 'bash', 'bash', NULL, '[]', NULL, '[]', 'bash', NULL, NULL, NULL, 'bash')
            """,
            ("event-complete-bash", conv_id, message_id),
        )
        conn.commit()

    tag_records = _run_completion(workspace, cwd=tmp_path, words="polylogue --tag re", cword=2)
    tool_records = _run_completion(workspace, cwd=tmp_path, words="polylogue --tool ba", cword=2)

    review = next(record for record in tag_records if record["value"] == "review")
    bash = next(record for record in tool_records if record["value"] == "bash")
    assert review["help"] == "1 conversations"
    assert bash["help"] == "1 actions"


def test_cli_completion_provider_values_keep_csv_prefix_and_descriptions(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    records = _run_completion(workspace, cwd=tmp_path, words="polylogue --provider claude-ai,c", cword=2)

    assert any(
        record["value"] == "claude-ai,claude-code" and record["help"] == "Claude Code local sessions"
        for record in records
    )
    assert any(record["value"] == "claude-ai,codex" and record["help"] == "OpenAI Codex sessions" for record in records)


def test_cli_no_args_stats_surface_still_works(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(["--plain"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert "archive:" in result.output.lower()
    assert "sources:" in result.output.lower()
