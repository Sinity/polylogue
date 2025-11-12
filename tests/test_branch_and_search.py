from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.branch_explorer import branch_diff, build_branch_html, format_branch_tree
from polylogue.commands import branches_command, search_command
from polylogue.db import open_connection
from polylogue.importers.chatgpt import import_chatgpt_export
from polylogue.options import BranchExploreOptions, SearchOptions


def _create_branching_conversation(tmp_path: Path) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    conversations = [
        {
            "id": "conv-1",
            "title": "Test Conversation",
            "create_time": 1,
            "update_time": 5,
            "model_slug": "gpt-test",
            "current_node": "node-assistant-main",
            "mapping": {
                "node-user": {
                    "id": "node-user",
                    "message": {
                        "id": "msg-user",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["User root message"]},
                    },
                },
                "node-assistant-main": {
                    "id": "node-assistant-main",
                    "parent": "node-user",
                    "message": {
                        "id": "msg-assistant-main",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Model canonical reply"]},
                    },
                },
                "node-assistant-alt": {
                    "id": "node-assistant-alt",
                    "parent": "node-user",
                    "message": {
                        "id": "msg-assistant-alt",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Model branch alternate"]},
                    },
                },
            },
        }
    ]

    (export_dir / "conversations.json").write_text(json.dumps(conversations), encoding="utf-8")

    output_dir = tmp_path / "out"
    results = import_chatgpt_export(
        export_dir,
        output_dir=output_dir,
        collapse_threshold=16,
        html=False,
        html_theme="light",
    )
    assert results

    with open_connection() as conn:
        conn.execute(
            "UPDATE messages SET attachment_count = 1 WHERE message_id = ?",
            ("msg-assistant-alt",),
        )
        conn.commit()

    return results[0].markdown_path.parent


def test_branch_explorer_outputs(state_env, tmp_path):
    _create_branching_conversation(tmp_path)

    options = BranchExploreOptions(provider="chatgpt", slug=None, conversation_id=None, min_branches=1)
    result = branches_command(options)
    assert result.conversations
    convo = result.conversations[0]

    tree = format_branch_tree(convo, use_color=False)
    assert "branch-000" in tree
    assert any("branch-001" in line for line in tree.splitlines())

    html = build_branch_html(convo, theme="light")
    assert "Branch graph" in html
    assert "branch-001" in html

    diff_text = branch_diff(convo, "branch-001")
    assert diff_text is not None
    assert "branch-001" in diff_text


def test_search_command_filters(state_env, tmp_path):
    _create_branching_conversation(tmp_path)

    base_options = SearchOptions(
        query="branch",
        limit=5,
        provider="chatgpt",
        slug=None,
        conversation_id=None,
        branch_id=None,
        model="gpt-test",
        since=None,
        until=None,
        has_attachments=None,
    )
    result = search_command(base_options)
    assert result.hits
    first = result.hits[0]
    assert first.provider == "chatgpt"
    assert first.model == "gpt-test"

    with_attachments = search_command(
        SearchOptions(
            query="branch",
            limit=5,
            provider="chatgpt",
            slug=None,
            conversation_id=None,
            branch_id=None,
            model=None,
            since=None,
            until=None,
            has_attachments=True,
        )
    )
    assert with_attachments.hits
    assert all(hit.attachment_count > 0 for hit in with_attachments.hits)

    without_attachments = search_command(
        SearchOptions(
            query="canonical",
            limit=5,
            provider="chatgpt",
            slug=None,
            conversation_id=None,
            branch_id=None,
            model=None,
            since=None,
            until=None,
            has_attachments=False,
        )
    )
    assert without_attachments.hits
    assert all(hit.attachment_count == 0 for hit in without_attachments.hits)

