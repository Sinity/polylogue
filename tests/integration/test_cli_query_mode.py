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
    result = run_cli(
        ["--plain", "run", "--input", str(workspace["paths"]["inbox"])],
        env=workspace["env"],
        cwd=cwd,
    )
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


def _assert_only_optional_open_progress(stderr: str) -> None:
    lines = [line for line in stderr.splitlines() if line.strip()]
    assert all(
        line.startswith("Query still running after ") and "route: open" in line and "retrieval: auto" in line
        for line in lines
    )


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


def test_cli_messages_and_raw_routes_read_conversation_records(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-read-surface").title("Read Surface").add_user("read surface alpha").add_assistant(
        "read surface beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)

    list_result = run_cli(["--plain", "alpha", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)
    assert list_result.exit_code == 0, list_result.output
    conversation_id = json.loads(list_result.stdout)[0]["id"]

    messages_result = run_cli(
        ["--plain", "messages", conversation_id, "--message-role", "user", "--limit", "1", "-f", "json"],
        env=workspace["env"],
        cwd=tmp_path,
    )
    assert messages_result.exit_code == 0, messages_result.output
    messages_payload = json.loads(messages_result.stdout)
    assert messages_payload["conversation_id"] == conversation_id
    assert messages_payload["total"] == 1
    assert len(messages_payload["messages"]) == 1
    assert messages_payload["messages"][0]["role"] == "user"
    assert messages_payload["messages"][0]["text"] == "read surface alpha"

    raw_result = run_cli(["--plain", "raw", conversation_id, "-f", "json"], env=workspace["env"], cwd=tmp_path)
    assert raw_result.exit_code == 0, raw_result.output
    raw_payload = json.loads(raw_result.stdout)
    assert raw_payload["conversation_id"] == conversation_id
    assert raw_payload["total"] == 1
    assert len(raw_payload["artifacts"]) == 1
    assert raw_payload["artifacts"][0]["raw_id"]


def test_cli_query_select_returns_first_matched_conversation_for_pipes(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-select").title("Select Surface").add_user("select alpha").add_assistant(
        "select beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)

    id_result = run_cli(["--plain", "select alpha", "select", "--print", "id"], env=workspace["env"], cwd=tmp_path)
    assert id_result.exit_code == 0, id_result.output
    assert id_result.stdout.strip().endswith("conv-select")

    json_result = run_cli(
        ["--plain", "select alpha", "select", "--print", "json"],
        env=workspace["env"],
        cwd=tmp_path,
    )
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.stdout)
    assert str(payload["id"]).endswith("conv-select")
    assert payload["title"] == "Select Surface"


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
    _assert_only_optional_open_progress(result.stderr)
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
    _assert_only_optional_open_progress(result.stderr)
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


def test_cli_completion_repo_cwd_and_read_ids_are_archive_backed(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    GenericConversationBuilder("conv-complete-insights").title("Completion Insights").add_user("alpha").add_assistant(
        "beta"
    ).write_to(inbox / "conversation.json")
    _run_inbox(workspace, cwd=tmp_path)
    list_result = run_cli(["--plain", "--latest", "list", "-f", "json"], env=workspace["env"], cwd=tmp_path)
    assert list_result.exit_code == 0, list_result.output
    conv_id = json.loads(list_result.stdout)[0]["id"]

    db_path = workspace["paths"]["db_path"]
    cwd_path = "/realm/project/polylogue"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO session_profiles (
                conversation_id,
                materialized_at,
                provider_name,
                title,
                repo_names_json,
                evidence_payload_json,
                search_text
            ) VALUES (?, '2026-01-01T00:00:00Z', 'claude-code', ?, json(?), json(?), ?)
            """,
            (
                conv_id,
                "Completion Insights",
                json.dumps(["polylogue"]),
                json.dumps({"cwd_paths": [cwd_path]}),
                "Completion Insights polylogue",
            ),
        )
        conn.commit()

    repo_records = _run_completion(workspace, cwd=tmp_path, words="polylogue --repo po", cword=2)
    repo_csv_records = _run_completion(workspace, cwd=tmp_path, words="polylogue --repo old,po", cword=2)
    cwd_records = _run_completion(workspace, cwd=tmp_path, words="polylogue --cwd-prefix /realm/project/p", cword=2)
    messages_records = _run_completion(workspace, cwd=tmp_path, words="polylogue messages conv", cword=2)
    raw_records = _run_completion(workspace, cwd=tmp_path, words="polylogue raw conv", cword=2)

    repo = next(record for record in repo_records if record["value"] == "polylogue")
    repo_csv = next(record for record in repo_csv_records if record["value"] == "old,polylogue")
    cwd = next(record for record in cwd_records if record["value"] == cwd_path)
    assert repo["help"] == "1 sessions"
    assert repo_csv["help"] == "1 sessions"
    assert cwd["help"] == "1 sessions"
    assert any(record["value"] == conv_id for record in messages_records)
    assert any(record["value"] == conv_id for record in raw_records)


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
