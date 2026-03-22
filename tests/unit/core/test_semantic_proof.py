"""Tests for semantic preservation proofing across render and export surfaces."""

from __future__ import annotations

import asyncio
import json

from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.formatting import format_conversation
from polylogue.rendering.semantic_proof import (
    DEFAULT_SEMANTIC_SURFACES,
    prove_export_surface_semantics,
    prove_markdown_projection_semantics,
    prove_markdown_render_semantics,
    prove_semantic_surface_suite,
    resolve_semantic_surfaces,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from tests.infra.storage_records import ConversationBuilder


async def _load_projection(db_path, conversation_id: str):
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    try:
        return await repository.get_render_projection(conversation_id)
    finally:
        await backend.close()


async def _load_conversation(db_path, conversation_id: str):
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    try:
        return await repository.view(conversation_id)
    finally:
        await backend.close()


def test_markdown_projection_semantics_reports_declared_losses_without_critical_loss(db_path, tmp_path):
    """Tool/thinking flattening is declared loss, not silent critical loss."""
    (
        ConversationBuilder(db_path, "conv-semantic-1")
        .provider("chatgpt")
        .title("Semantic Proof")
        .updated_at("2026-03-22T12:00:00+00:00")
        .add_message("m1", role="user", text="hello")
        .add_message(
            "m2",
            role="assistant",
            text="working on it",
            provider_meta={"content_blocks": [{"type": "thinking", "text": "internal chain"}]},
        )
        .add_message(
            "m3",
            role="tool",
            text="tool output",
            provider_meta={"content_blocks": [{"type": "tool_result", "text": "tool output"}]},
        )
        .add_attachment("att1", message_id="m2", path="/tmp/att1.txt")
        .save()
    )

    projection = asyncio.run(_load_projection(db_path, "conv-semantic-1"))
    assert projection is not None
    formatter = ConversationFormatter(archive_root=tmp_path, db_path=db_path)
    markdown = formatter.format_projection(projection).markdown_text

    report = prove_markdown_projection_semantics(projection, markdown)

    assert report.is_clean is True
    assert report.input_facts["attachment_count"] == 1
    assert {check.metric for check in report.declared_loss_checks} == {
        "thinking_semantics",
        "tool_semantics",
    }
    assert report.metric_summary["renderable_messages"]["preserved"] == 1


def test_markdown_projection_semantics_detects_critical_loss(db_path, tmp_path):
    """Removing a rendered message section is reported as critical loss."""
    (
        ConversationBuilder(db_path, "conv-semantic-2")
        .provider("chatgpt")
        .title("Critical Semantic Loss")
        .updated_at("2026-03-22T12:05:00+00:00")
        .add_message("m1", role="user", text="hello")
        .add_message("m2", role="assistant", text="world")
        .save()
    )

    projection = asyncio.run(_load_projection(db_path, "conv-semantic-2"))
    assert projection is not None
    formatter = ConversationFormatter(archive_root=tmp_path, db_path=db_path)
    markdown = formatter.format_projection(projection).markdown_text
    broken_markdown = markdown.replace("## assistant", "### assistant", 1)

    report = prove_markdown_projection_semantics(projection, broken_markdown)

    assert report.is_clean is False
    assert {check.metric for check in report.critical_loss_checks} == {
        "renderable_messages",
        "role_sections",
    }


def test_export_json_surface_detects_message_loss(db_path):
    """Structured export proof flags silently dropped message rows as critical loss."""
    (
        ConversationBuilder(db_path, "conv-semantic-3")
        .provider("chatgpt")
        .title("JSON Semantic")
        .updated_at("2026-03-22T12:10:00+00:00")
        .add_message("json-m1", role="user", text="hello")
        .add_message("json-m2", role="assistant", text="world")
        .save()
    )

    conversation = asyncio.run(_load_conversation(db_path, "conv-semantic-3"))
    assert conversation is not None
    payload = json.loads(format_conversation(conversation, "json", None))
    payload["messages"] = payload["messages"][:-1]

    report = prove_export_surface_semantics(
        conversation,
        "export_json_v1",
        json.dumps(payload),
    )

    assert report.is_clean is False
    assert {check.metric for check in report.critical_loss_checks} >= {
        "message_entries",
        "message_ids",
        "role_entries",
        "timestamp_values",
    }


def test_semantic_surface_suite_aggregates_export_and_canonical_surfaces(db_path, tmp_path):
    """Suite proof aggregates canonical and export surfaces with explicit loss policies."""
    (
        ConversationBuilder(db_path, "conv-semantic-4")
        .provider("chatgpt")
        .title("ChatGPT Semantic")
        .updated_at("2026-03-22T12:20:00+00:00")
        .add_message("conv4-m1", role="user", text="hello")
        .add_message("conv4-m2", role="assistant", text="branch-root")
        .add_message(
            "conv4-b1",
            role="assistant",
            text="branch alternative",
            parent_message_id="conv4-m1",
            branch_index=1,
            provider_meta={"content_blocks": [{"type": "thinking", "text": "inner"}]},
            has_thinking=1,
        )
        .add_attachment("att1", message_id="conv4-b1", path="/tmp/att1.txt")
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-semantic-5")
        .provider("claude-ai")
        .title("Claude Semantic")
        .updated_at("2026-03-22T12:25:00+00:00")
        .add_message("conv5-m1", role="assistant", text="tool output", has_tool_use=1)
        .save()
    )

    suite = prove_semantic_surface_suite(
        db_path=db_path,
        archive_root=tmp_path,
        surfaces=["canonical", "json", "html", "csv", "obsidian"],
    )

    assert suite.surface_count == 5
    assert suite.total_conversations == 10
    assert set(suite.surfaces) == {
        "canonical_markdown_v1",
        "export_json_v1",
        "export_html_v1",
        "export_csv_v1",
        "export_obsidian_v1",
    }
    assert suite.surfaces["canonical_markdown_v1"].providers["chatgpt"].declared_loss_checks >= 1
    assert suite.surfaces["export_json_v1"].providers["chatgpt"].declared_loss_checks >= 1
    assert suite.surfaces["export_html_v1"].metric_summary["branch_structure"]["preserved"] == 1
    assert suite.surfaces["export_csv_v1"].metric_summary["provider_identity"]["declared_loss"] == 2
    assert suite.to_dict()["summary"]["surface_count"] == 5


def test_markdown_render_semantics_returns_single_surface_report(db_path, tmp_path):
    """Legacy single-surface helper remains a canonical-markdown projection of the suite."""
    (
        ConversationBuilder(db_path, "conv-semantic-6")
        .provider("claude-ai")
        .title("Single Surface")
        .updated_at("2026-03-22T12:30:00+00:00")
        .add_message(
            "conv6-m1",
            role="assistant",
            text="thinking aloud",
            provider_meta={"content_blocks": [{"type": "thinking", "text": "inner"}]},
            has_thinking=1,
        )
        .save()
    )

    report = prove_markdown_render_semantics(
        db_path=db_path,
        archive_root=tmp_path,
    )

    assert report.surface == "canonical_markdown_v1"
    assert report.total_conversations == 1
    assert report.provider_count == 1
    assert report.providers["claude-ai"].declared_loss_checks == 1


def test_resolve_semantic_surfaces_expands_aliases():
    """Surface aliases expand to canonical suite surface names."""
    assert resolve_semantic_surfaces(["canonical", "json", "all"]) == list(DEFAULT_SEMANTIC_SURFACES)
