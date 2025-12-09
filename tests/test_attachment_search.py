from __future__ import annotations

import hashlib
from pathlib import Path

from polylogue.archive import Archive
from polylogue.config import CONFIG
from polylogue.conversation import process_conversation
from polylogue.options import SearchOptions
from polylogue.render import AttachmentInfo
from polylogue.branching import MessageRecord
from polylogue.search import execute_search
from polylogue.services.conversation_registrar import ConversationRegistrar
from polylogue.services.conversation_service import ConversationService
from polylogue.persistence.state import ConversationStateRepository
from polylogue.persistence.database import ConversationDatabase


def _build_registrar(root: Path) -> ConversationRegistrar:
    database = ConversationDatabase(path=root / "polylogue.db")
    state_repo = ConversationStateRepository(database=database)
    archive = Archive(CONFIG)
    return ConversationRegistrar(state_repo=state_repo, database=database, archive=archive)


def test_attachment_text_is_indexed(tmp_path):
    registrar = _build_registrar(tmp_path)
    output_dir = tmp_path / "out"
    slug = "conv-1"
    conversation_dir = output_dir / slug
    attachments_dir = conversation_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    attachment_path = attachments_dir / "note.txt"
    attachment_text = "hello from attachment body"
    attachment_path.write_text(attachment_text, encoding="utf-8")

    attachment = AttachmentInfo(
        name="note.txt",
        link="attachments/note.txt",
        local_path=Path("attachments/note.txt"),
        size_bytes=attachment_path.stat().st_size,
        remote=False,
    )
    message = MessageRecord(
        message_id="m1",
        parent_id=None,
        role="user",
        text="Message with attachment",
        token_count=5,
        word_count=3,
        timestamp="2024-01-01T00:00:00Z",
        attachments=1,
        chunk={"role": "user", "text": "Message with attachment"},
        links=[("note.txt", Path("attachments/note.txt"))],
        metadata={"attachments": [{"name": "note.txt", "link": "attachments/note.txt"}]},
        content_hash=hashlib.sha256(b"Message with attachment").hexdigest(),
    )

    process_conversation(
        provider="test",
        conversation_id="conv-1",
        slug=slug,
        title="Attachment Search",
        message_records=[message],
        attachments=[attachment],
        canonical_leaf_id="m1",
        collapse_threshold=16,
        html=False,
        html_theme="light",
        output_dir=output_dir,
        extra_yaml=None,
        extra_state=None,
        source_file_id="conv-1",
        modified_time="2024-01-01T00:00:00Z",
        created_time="2024-01-01T00:00:00Z",
        run_settings=None,
        source_mime="application/json",
        source_size=None,
        attachment_policy=None,
        force=False,
        allow_dirty=False,
        attachment_ocr=False,
        registrar=registrar,
    )

    service = ConversationService(registrar=registrar)
    options = SearchOptions(
        query="attachment",
        limit=5,
        provider="test",
        slug=None,
        conversation_id=None,
        branch_id=None,
        model=None,
        since=None,
        until=None,
        has_attachments=None,
        in_attachments=True,
        attachment_name=None,
    )
    result = execute_search(options, service=service)
    assert result.hits, "Attachment search should return hits"
    hit = result.hits[0]
    assert hit.kind == "attachment"
    assert hit.attachment_name == "note.txt"
    assert "hello from attachment body" in hit.body
