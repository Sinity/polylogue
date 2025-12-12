from __future__ import annotations

from pathlib import Path

from polylogue.document_store import persist_document
from polylogue.redaction import sanitize_text
from polylogue.render import MarkdownDocument
from polylogue.services.conversation_registrar import ConversationRegistrar
from polylogue.persistence.state import ConversationStateRepository
from polylogue.persistence.database import ConversationDatabase
from polylogue.archive import Archive
from polylogue.config import CONFIG


def _registrar(state_home: Path) -> ConversationRegistrar:
    state_home.mkdir(parents=True, exist_ok=True)
    database = ConversationDatabase(path=state_home / "polylogue.db")
    return ConversationRegistrar(
        state_repo=ConversationStateRepository(database=database),
        database=database,
        archive=Archive(CONFIG),
    )


def test_sanitize_text_masks_common_secrets() -> None:
    raw = "Contact me at test@example.com and use sk-abc12345678901234567890 for auth."
    redacted = sanitize_text(raw)
    assert "test@example.com" not in redacted
    assert "[redacted-email]" in redacted
    assert "sk-abc12345678901234567890" not in redacted
    assert "[redacted-key]" in redacted


def test_persist_document_sanitizes_body_and_marks_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    state_home = tmp_path / "state"
    doc = MarkdownDocument(
        body="hello test@example.com",
        metadata={"title": "Test Conversation"},
        attachments=[],
        stats={},
    )
    result = persist_document(
        provider="test",
        conversation_id="conv",
        title="Test",
        document=doc,
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
        registrar=_registrar(state_home),
        sanitize=True,
    )

    assert "test@example.com" not in result.document.body
    assert "[redacted-email]" in result.document.body
    polylogue_meta = result.document.metadata.get("polylogue", {})
    assert polylogue_meta.get("redacted") is True
