from __future__ import annotations

from pathlib import Path

from polylogue.export import export_jsonl
from polylogue.ingest import IngestBundle, ingest_bundle
from polylogue.render import render_conversation
from polylogue.paths import is_within_root
from polylogue.db import open_connection
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

    result = render_conversation(conversation_id="conv:hash", archive_root=archive_root)
    assert result.markdown_path.exists()
    assert result.html_path.exists()
    assert "hello" in result.markdown_path.read_text(encoding="utf-8")


def test_render_escapes_html(workspace_env):
    archive_root = workspace_env["archive_root"]
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv-html",
            provider_name="codex",
            provider_conversation_id="conv-html",
            title="<script>alert(1)</script>",
            created_at=None,
            updated_at=None,
            content_hash="hash",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:hash",
                conversation_id="conv-html",
                provider_message_id="msg",
                role="user",
                text="<script>alert(2)</script>",
                timestamp=None,
                content_hash="hash",
                provider_meta=None,
            )
        ],
        attachments=[],
    )
    ingest_bundle(bundle)

    result = render_conversation(conversation_id="conv-html", archive_root=archive_root)
    html_text = result.html_path.read_text(encoding="utf-8")
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_render_sanitizes_paths(workspace_env):
    archive_root = workspace_env["archive_root"]
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="../escape",
            provider_name="bad/provider",
            provider_conversation_id="conv-escape",
            title="Escape",
            created_at=None,
            updated_at=None,
            content_hash="hash",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:hash",
                conversation_id="../escape",
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

    result = render_conversation(conversation_id="../escape", archive_root=archive_root)
    render_root = archive_root / "render"
    assert is_within_root(result.markdown_path, render_root)


def test_export_includes_attachments(workspace_env, tmp_path):
    bundle = IngestBundle(
        conversation=_conversation_record(),
        messages=[],
        attachments=[
            AttachmentRecord(
                attachment_id="att-1",
                conversation_id="conv:hash",
                message_id=None,
                mime_type="text/plain",
                size_bytes=12,
                path="/tmp/att.txt",
                provider_meta=None,
            )
        ],
    )
    ingest_bundle(bundle)
    output = export_jsonl(archive_root=workspace_env["archive_root"], output_path=tmp_path / "export.jsonl")
    payload = output.read_text(encoding="utf-8").strip().splitlines()[0]
    assert "\"attachments\"" in payload


def test_ingest_updates_metadata(workspace_env):
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv-update",
            provider_name="codex",
            provider_conversation_id="conv-update",
            title="Old",
            created_at=None,
            updated_at=None,
            content_hash="hash-old",
            provider_meta={"source": "inbox"},
        ),
        messages=[
            MessageRecord(
                message_id="msg-update",
                conversation_id="conv-update",
                provider_message_id="msg-update",
                role="user",
                text="hello",
                timestamp="1",
                content_hash="msg-old",
                provider_meta={"k": "v"},
            )
        ],
        attachments=[],
    )
    ingest_bundle(bundle)

    updated = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv-update",
            provider_name="codex",
            provider_conversation_id="conv-update",
            title="New",
            created_at=None,
            updated_at="2",
            content_hash="hash-new",
            provider_meta={"source": "inbox", "updated": True},
        ),
        messages=[
            MessageRecord(
                message_id="msg-update",
                conversation_id="conv-update",
                provider_message_id="msg-update",
                role="assistant",
                text="hello",
                timestamp="2",
                content_hash="msg-new",
                provider_meta={"k": "v2"},
            )
        ],
        attachments=[],
    )
    ingest_bundle(updated)

    with open_connection(None) as conn:
        convo = conn.execute(
            "SELECT title, updated_at, content_hash, provider_meta FROM conversations WHERE conversation_id = ?",
            ("conv-update",),
        ).fetchone()
        msg = conn.execute(
            "SELECT role, timestamp, content_hash, provider_meta FROM messages WHERE message_id = ?",
            ("msg-update",),
        ).fetchone()
    assert convo["title"] == "New"
    assert convo["updated_at"] == "2"
    assert convo["content_hash"] == "hash-new"
    assert msg["role"] == "assistant"
    assert msg["timestamp"] == "2"
    assert msg["content_hash"] == "msg-new"
