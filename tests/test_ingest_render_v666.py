from __future__ import annotations

from pathlib import Path

from polylogue.ingest import IngestBundle, ingest_bundle
from polylogue.render_v666 import render_conversation
from polylogue.store import AttachmentRecord, ConversationRecord, MessageRecord


def _conversation_record():
    return ConversationRecord(
        conversation_id="conv:hash",
        provider_name="codex",
        provider_conversation_id="conv",
        title="Demo",
        created_at=None,
        updated_at=None,
        content_hash="hash",
        provider_meta=None,
    )


def test_ingest_idempotent(workspace_env):
    bundle = IngestBundle(
        conversation=_conversation_record(),
        messages=[
            MessageRecord(
                message_id="msg:hash",
                conversation_id="conv:hash",
                provider_message_id="msg",
                role="user",
                text="hello",
                timestamp=None,
                content_hash="hash",
                provider_meta=None,
            )
        ],
        attachments=[
            AttachmentRecord(
                attachment_id="att-hash",
                conversation_id="conv:hash",
                message_id="msg:hash",
                mime_type=None,
                size_bytes=None,
                path=None,
                provider_meta=None,
            )
        ],
    )

    first = ingest_bundle(bundle)
    second = ingest_bundle(bundle)

    assert first.conversations == 1
    assert second.skipped_conversations == 1
    assert second.skipped_messages == 1


def test_render_writes_markdown(workspace_env):
    archive_root = workspace_env["archive_root"]
    bundle = IngestBundle(
        conversation=_conversation_record(),
        messages=[
            MessageRecord(
                message_id="msg:hash",
                conversation_id="conv:hash",
                provider_message_id="msg",
                role="user",
                text="hello",
                timestamp=None,
                content_hash="hash",
                provider_meta=None,
            )
        ],
        attachments=[],
    )
    ingest_bundle(bundle)

    result = render_conversation(conversation_id="conv:hash", archive_root=archive_root, html_mode="off")
    assert result.markdown_path.exists()
    assert "hello" in result.markdown_path.read_text(encoding="utf-8")
