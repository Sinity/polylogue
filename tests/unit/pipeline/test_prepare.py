from __future__ import annotations

import json

from polylogue.export import export_jsonl
from polylogue.paths import is_within_root
from polylogue.rendering.renderers import HTMLRenderer
from polylogue.sources import RecordBundle, save_bundle
from polylogue.storage.backends.sqlite import open_connection
from tests.helpers import make_attachment, make_conversation, make_message


def _conversation_record():
    return make_conversation("conv:hash", provider_name="codex", title="Demo")


def test_ingest_idempotent(workspace_env, storage_repository):
    bundle = RecordBundle(
        conversation=_conversation_record(),
        messages=[make_message("msg:hash", "conv:hash", text="hello")],
        attachments=[make_attachment("att-hash", "conv:hash", "msg:hash", mime_type=None, size_bytes=None)],
    )

    first = save_bundle(bundle, repository=storage_repository)
    second = save_bundle(bundle, repository=storage_repository)

    assert first.conversations == 1
    assert second.skipped_conversations == 1
    assert second.skipped_messages == 1


def test_render_writes_markdown(workspace_env, storage_repository):
    archive_root = workspace_env["archive_root"]
    bundle = RecordBundle(
        conversation=_conversation_record(),
        messages=[make_message("msg:hash", "conv:hash", text="hello")],
        attachments=[],
    )
    save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("conv:hash", output_root)
    md_path = html_path.parent / "conversation.md"

    assert md_path.exists()
    assert html_path.exists()
    assert "hello" in md_path.read_text(encoding="utf-8")


def test_render_escapes_html(workspace_env, storage_repository):
    archive_root = workspace_env["archive_root"]
    bundle = RecordBundle(
        conversation=make_conversation("conv-html", provider_name="codex", title="<script>alert(1)</script>"),
        messages=[make_message("msg:hash", "conv-html", text="<script>alert(2)</script>")],
        attachments=[],
    )
    save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("conv-html", output_root)
    html_text = html_path.read_text(encoding="utf-8")

    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_render_sanitizes_paths(workspace_env, storage_repository):
    """Test that render paths are sanitized even with path-like conversation IDs.

    Note: Invalid provider names are now rejected at the validation layer, so we
    test path sanitization through conversation_id alone using a valid provider name.
    """
    archive_root = workspace_env["archive_root"]
    bundle = RecordBundle(
        conversation=make_conversation("../escape", title="Escape"),
        messages=[make_message("msg:hash", "../escape", text="hello")],
        attachments=[],
    )
    save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("../escape", output_root)
    md_path = html_path.parent / "conversation.md"

    assert is_within_root(md_path, output_root)


def test_render_includes_orphan_attachments(workspace_env, storage_repository):
    archive_root = workspace_env["archive_root"]
    bundle = RecordBundle(
        conversation=_conversation_record(),
        messages=[make_message("msg:hash", "conv:hash", text="hello")],
        attachments=[make_attachment("att-orphan", "conv:hash", None, mime_type="text/plain", size_bytes=12, provider_meta={"name": "notes.txt"})],
    )
    save_bundle(bundle, repository=storage_repository)

    renderer = HTMLRenderer(archive_root)
    output_root = archive_root / "render"
    html_path = renderer.render("conv:hash", output_root)
    md_path = html_path.parent / "conversation.md"
    markdown = md_path.read_text(encoding="utf-8")

    assert "- Attachment: notes.txt" in markdown


def test_export_includes_attachments(workspace_env, tmp_path, storage_repository):
    bundle = RecordBundle(
        conversation=_conversation_record(),
        messages=[],
        attachments=[make_attachment("att-1", "conv:hash", None, mime_type="text/plain", size_bytes=12)],
    )
    save_bundle(bundle, repository=storage_repository)
    output = export_jsonl(archive_root=workspace_env["archive_root"], output_path=tmp_path / "export.jsonl")
    payload = output.read_text(encoding="utf-8").strip().splitlines()[0]
    assert '"attachments"' in payload


def test_ingest_updates_metadata(workspace_env, storage_repository):
    bundle = RecordBundle(
        conversation=make_conversation("conv-update", provider_name="codex", title="Old", content_hash="hash-old", provider_meta={"source": "inbox"}),
        messages=[make_message("msg-update", "conv-update", text="hello", timestamp="1", content_hash="msg-old", provider_meta={"k": "v"})],
        attachments=[],
    )
    save_bundle(bundle, repository=storage_repository)

    updated = RecordBundle(
        conversation=make_conversation("conv-update", provider_name="codex", title="New", updated_at="2", content_hash="hash-new", provider_meta={"source": "inbox", "updated": True}),
        messages=[make_message("msg-update", "conv-update", role="assistant", text="hello", timestamp="2", content_hash="msg-new", provider_meta={"k": "v2"})],
        attachments=[],
    )
    save_bundle(updated, repository=storage_repository)

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
    base_conversation = make_conversation("conv-hash-stable", provider_name="codex", title="Original", updated_at="1", content_hash="hash-stable", provider_meta={"source": "inbox"})
    base_message = make_message("msg-stable", "conv-hash-stable", text="hello", timestamp="1", content_hash="msg-stable", provider_meta={"k": "v1"})
    save_bundle(
        RecordBundle(conversation=base_conversation, messages=[base_message], attachments=[]),
        repository=storage_repository,
    )

    updated = RecordBundle(
        conversation=make_conversation("conv-hash-stable", provider_name="codex", title="Updated title", updated_at="2", content_hash="hash-stable", provider_meta={"source": "inbox", "updated": True}),
        messages=[
            make_message("msg-stable", "conv-hash-stable", role="assistant", text="hello", timestamp="3", content_hash="msg-stable", provider_meta={"k": "v2"})
        ],
        attachments=[],
    )
    save_bundle(updated, repository=storage_repository)

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
    bundle = RecordBundle(
        conversation=_conversation_record(),
        messages=[make_message("msg:att", "conv:hash", text="hello", timestamp="1", content_hash="msg:att")],
        attachments=[make_attachment("att-old", "conv:hash", "msg:att", mime_type="text/plain", size_bytes=10)],
    )
    save_bundle(bundle, repository=storage_repository)

    save_bundle(
        RecordBundle(
            conversation=_conversation_record(),
            messages=[make_message("msg:att", "conv:hash", text="hello", timestamp="1", content_hash="msg:att")],
            attachments=[],
        ),
        repository=storage_repository,
    )

    with open_connection(None) as conn:
        attachment_count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
        ref_count = conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0]
    assert attachment_count == 0
    assert ref_count == 0
