from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.document_store import persist_document
from polylogue.render import AttachmentInfo, MarkdownDocument
from polylogue.util import StateStore, configure_state_store


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path):
    store = StateStore(tmp_path / "state.json")
    configure_state_store(store)
    yield


def _make_document(body: str, attachments: list[AttachmentInfo]) -> MarkdownDocument:
    return MarkdownDocument(
        body=body,
        metadata={"title": "Test Conversation"},
        attachments=attachments,
        stats={"totalTokensApprox": 1},
    )


def test_persist_document_cleans_stale_attachments(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    attachments_dir = output_dir / "conversation" / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    attachment_path = attachments_dir / "note.txt"
    attachment_path.write_text("hello", encoding="utf-8")

    attachment = AttachmentInfo(
        name="note.txt",
        link="attachments/note.txt",
        local_path=Path("attachments/note.txt"),
        size_bytes=5,
        remote=False,
    )

    doc = _make_document("First", [attachment])
    result = persist_document(
        provider="test",
        conversation_id="conv",
        title="Test",
        document=doc,
        output_dir=output_dir,
        collapse_threshold=5,
        attachments=[attachment],
        updated_at="2024-01-01T00:00:00Z",
        created_at="2024-01-01T00:00:00Z",
        html=True,
        html_theme="light",
        attachment_policy=None,
        extra_state=None,
        slug_hint="conversation",
        id_hint=None,
        force=False,
    )

    assert result.attachments_dir and result.attachments_dir.exists()
    assert (result.markdown_path.parent / "conversation.html").exists()

    # Remove attachment reference and disable HTML; expect cleanup.
    doc2 = _make_document("Second", [])
    result2 = persist_document(
        provider="test",
        conversation_id="conv",
        title="Test",
        document=doc2,
        output_dir=output_dir,
        collapse_threshold=5,
        attachments=[],
        updated_at="2024-01-01T00:00:00Z",
        created_at="2024-01-01T00:00:00Z",
        html=False,
        html_theme="light",
        attachment_policy=None,
        extra_state=None,
        slug_hint="conversation",
        id_hint=None,
        force=False,
    )

    assert result2.attachments_dir is None
    assert not attachments_dir.exists()
    assert not (result2.markdown_path.parent / "conversation.html").exists()

