"""Tests for semantic preservation proofing over canonical markdown rendering."""

from __future__ import annotations

import asyncio

from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.semantic_proof import (
    prove_markdown_projection_semantics,
    prove_markdown_render_semantics,
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


def test_markdown_render_semantics_aggregates_provider_reports(db_path, tmp_path):
    """Conversation-level proofs roll up into provider and report summaries."""
    (
        ConversationBuilder(db_path, "conv-semantic-3")
        .provider("chatgpt")
        .title("ChatGPT Semantic")
        .updated_at("2026-03-22T12:10:00+00:00")
        .add_message("conv3-m1", role="user", text="hello")
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-semantic-4")
        .provider("claude-ai")
        .title("Claude Semantic")
        .updated_at("2026-03-22T12:15:00+00:00")
        .add_message(
            "conv4-m1",
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
    assert report.total_conversations == 2
    assert report.provider_count == 2
    assert report.providers["chatgpt"].clean is True
    assert report.providers["claude-ai"].declared_loss_checks == 1
    assert report.metric_summary["thinking_semantics"]["declared_loss"] == 1
    assert report.to_dict()["summary"]["provider_count"] == 2
