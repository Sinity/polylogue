from __future__ import annotations

import json

from polylogue.storage.db import open_connection
from polylogue.export import export_jsonl
from polylogue.ingestion import IngestBundle, ingest_bundle
from polylogue.paths import is_within_root
from polylogue.render import render_conversation
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord


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


def test_ingest_idempotent(workspace_env, storage_repository):
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

    first = ingest_bundle(bundle, repository=storage_repository)
    second = ingest_bundle(bundle, repository=storage_repository)

    assert first.conversations == 1
    assert second.skipped_conversations == 1
    assert second.skipped_messages == 1


def test_render_writes_markdown(workspace_env, storage_repository):
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
    ingest_bundle(bundle, repository=storage_repository)

    result = render_conversation(conversation_id="conv:hash", archive_root=archive_root)
    assert result.markdown_path.exists()
    assert result.html_path.exists()
    assert "hello" in result.markdown_path.read_text(encoding="utf-8")


def test_render_escapes_html(workspace_env, storage_repository):
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
    ingest_bundle(bundle, repository=storage_repository)

    result = render_conversation(conversation_id="conv-html", archive_root=archive_root)
    html_text = result.html_path.read_text(encoding="utf-8")
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_render_sanitizes_paths(workspace_env, storage_repository):
    """Test that render paths are sanitized even with path-like conversation IDs.

    Note: Invalid provider names are now rejected at the validation layer, so we
    test path sanitization through conversation_id alone using a valid provider name.
    """
    archive_root = workspace_env["archive_root"]
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="../escape",
            provider_name="test",  # Valid provider name (path chars now rejected)
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
    ingest_bundle(bundle, repository=storage_repository)

    result = render_conversation(conversation_id="../escape", archive_root=archive_root)
    render_root = archive_root / "render"
    assert is_within_root(result.markdown_path, render_root)


def test_render_includes_orphan_attachments(workspace_env, storage_repository):
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
        attachments=[
            AttachmentRecord(
                attachment_id="att-orphan",
                conversation_id="conv:hash",
                message_id=None,
                mime_type="text/plain",
                size_bytes=12,
                path="/tmp/att.txt",
                provider_meta={"name": "notes.txt"},
            )
        ],
    )
    ingest_bundle(bundle, repository=storage_repository)

    result = render_conversation(conversation_id="conv:hash", archive_root=archive_root)
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "- Attachment: notes.txt" in markdown


def test_export_includes_attachments(workspace_env, tmp_path, storage_repository):
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
    ingest_bundle(bundle, repository=storage_repository)
    output = export_jsonl(archive_root=workspace_env["archive_root"], output_path=tmp_path / "export.jsonl")
    payload = output.read_text(encoding="utf-8").strip().splitlines()[0]
    assert '"attachments"' in payload


def test_ingest_updates_metadata(workspace_env, storage_repository):
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
    ingest_bundle(bundle, repository=storage_repository)

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
    ingest_bundle(updated, repository=storage_repository)

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


def test_ingest_updates_fields_without_hash_changes(workspace_env, storage_repository):
    base_conversation = ConversationRecord(
        conversation_id="conv-hash-stable",
        provider_name="codex",
        provider_conversation_id="conv-hash-stable",
        title="Original",
        created_at=None,
        updated_at="1",
        content_hash="hash-stable",
        provider_meta={"source": "inbox"},
    )
    base_message = MessageRecord(
        message_id="msg-stable",
        conversation_id="conv-hash-stable",
        provider_message_id="msg-stable",
        role="user",
        text="hello",
        timestamp="1",
        content_hash="msg-stable",
        provider_meta={"k": "v1"},
    )
    ingest_bundle(
        IngestBundle(
            conversation=base_conversation,
            messages=[base_message],
            attachments=[],
        ),
        repository=storage_repository,
    )

    updated = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv-hash-stable",
            provider_name="codex",
            provider_conversation_id="conv-hash-stable",
            title="Updated title",
            created_at=None,
            updated_at="2",
            content_hash="hash-stable",
            provider_meta={"source": "inbox", "updated": True},
        ),
        messages=[
            MessageRecord(
                message_id="msg-stable",
                conversation_id="conv-hash-stable",
                provider_message_id="msg-stable",
                role="assistant",
                text="hello",
                timestamp="3",
                content_hash="msg-stable",
                provider_meta={"k": "v2"},
            )
        ],
        attachments=[],
    )
    ingest_bundle(updated, repository=storage_repository)

    with open_connection(None) as conn:
        convo = conn.execute(
            "SELECT title, updated_at, provider_meta FROM conversations WHERE conversation_id = ?",
            ("conv-hash-stable",),
        ).fetchone()
        msg = conn.execute(
            "SELECT role, timestamp, provider_meta FROM messages WHERE message_id = ?",
            ("msg-stable",),
        ).fetchone()
    assert convo["title"] == "Updated title"
    assert convo["updated_at"] == "2"
    convo_meta = json.loads(convo["provider_meta"])
    msg_meta = json.loads(msg["provider_meta"])
    assert convo_meta["updated"] is True
    assert msg["role"] == "assistant"
    assert msg["timestamp"] == "3"
    assert msg_meta["k"] == "v2"


def test_ingest_removes_missing_attachments(workspace_env, storage_repository):
    bundle = IngestBundle(
        conversation=_conversation_record(),
        messages=[
            MessageRecord(
                message_id="msg:att",
                conversation_id="conv:hash",
                provider_message_id="msg:att",
                role="user",
                text="hello",
                timestamp="1",
                content_hash="msg:att",
                provider_meta=None,
            )
        ],
        attachments=[
            AttachmentRecord(
                attachment_id="att-old",
                conversation_id="conv:hash",
                message_id="msg:att",
                mime_type="text/plain",
                size_bytes=10,
                path="/tmp/old.txt",
                provider_meta=None,
            )
        ],
    )
    ingest_bundle(bundle, repository=storage_repository)

    ingest_bundle(
        IngestBundle(
            conversation=_conversation_record(),
            messages=[
                MessageRecord(
                    message_id="msg:att",
                    conversation_id="conv:hash",
                    provider_message_id="msg:att",
                    role="user",
                    text="hello",
                    timestamp="1",
                    content_hash="msg:att",
                    provider_meta=None,
                )
            ],
            attachments=[],
        ),
        repository=storage_repository,
    )

    with open_connection(None) as conn:
        attachment_count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
        ref_count = conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0]
    assert attachment_count == 0
    assert ref_count == 0
