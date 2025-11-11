from __future__ import annotations

import json
from pathlib import Path
import hashlib

from polylogue.branching import MessageRecord
from polylogue.conversation import process_conversation
from polylogue.db import open_connection
from polylogue.services.conversation_registrar import ConversationRegistrar
from polylogue.services.conversation_service import ConversationService
from polylogue.persistence.state import ConversationStateRepository
from polylogue.persistence.database import ConversationDatabase
from polylogue.archive import Archive
from polylogue.config import CONFIG
def _build_registrar(root: Path) -> ConversationRegistrar:
    database = ConversationDatabase(path=root / "polylogue.db")
    state_repo = ConversationStateRepository(database=database)
    archive = Archive(CONFIG)
    return ConversationRegistrar(state_repo=state_repo, database=database, archive=archive)


def _build_service(root: Path) -> ConversationService:
    return ConversationService(registrar=_build_registrar(root))


def test_process_conversation_populates_branch_metadata(tmp_path):
    registrar = _build_registrar(tmp_path)
    output_dir = tmp_path / "out"

    user_record = MessageRecord(
        message_id="user-1",
        parent_id=None,
        role="user",
        text="Hello",
        token_count=5,
        word_count=1,
        timestamp="2024-01-01T00:00:00Z",
        attachments=0,
        chunk={"role": "user", "text": "Hello"},
        metadata={
            "tool_calls": [{"type": "debug", "name": "inspector"}],
            "attachments": [],
            "raw": {"role": "user"},
        },
        content_hash=hashlib.sha256(b"Hello").hexdigest(),
    )
    model_record = MessageRecord(
        message_id="model-1",
        parent_id="user-1",
        role="model",
        text="Hi there",
        token_count=7,
        word_count=2,
        timestamp="2024-01-01T00:00:05Z",
        attachments=0,
        chunk={"role": "model", "text": "Hi there"},
        content_hash=hashlib.sha256(b"Hi there").hexdigest(),
    )

    result = process_conversation(
        provider="test",
        conversation_id="conv-1",
        slug="conv-1",
        title="Unit Test Conversation",
        message_records=[user_record, model_record],
        attachments=[],
        canonical_leaf_id=None,
        collapse_threshold=16,
        html=False,
        html_theme="light",
        output_dir=output_dir,
        extra_yaml=None,
        extra_state=None,
        source_file_id="conv-1",
        modified_time="2024-01-01T00:00:05Z",
        created_time="2024-01-01T00:00:00Z",
        run_settings=None,
        source_mime="application/json",
        source_size=None,
        attachment_policy=None,
        force=False,
        registrar=registrar,
    )

    state_entry = registrar.state_repo.get("test", "conv-1")
    assert state_entry is not None
    assert state_entry.get("slug") == result.slug
    assert state_entry.get("dirty") is False

    with open_connection(registrar.database.resolve_path()) as conn:
        row = conn.execute(
            "SELECT slug, current_branch, metadata_json FROM conversations WHERE provider = ? AND conversation_id = ?",
            ("test", "conv-1"),
        ).fetchone()
        assert row is not None
        assert row["slug"] == result.slug
        assert row["current_branch"] == result.canonical_branch_id
        metadata = json.loads(row["metadata_json"] or "{}")
        assert metadata.get("tokens", 0) >= 0
        assert metadata.get("words", 0) >= 0
        message_row = conn.execute(
            "SELECT content_hash, metadata_json FROM messages WHERE provider = ? AND conversation_id = ? ORDER BY position ASC LIMIT 1",
            ("test", "conv-1"),
        ).fetchone()
        assert message_row is not None
        assert message_row["content_hash"] == user_record.content_hash
        message_meta = json.loads(message_row["metadata_json"] or "{}")
        assert message_meta.get("tool_calls")[0]["name"] == "inspector"

        branch_count = conn.execute(
            "SELECT COUNT(*) FROM branches WHERE provider = ? AND conversation_id = ?",
            ("test", "conv-1"),
        ).fetchone()[0]
        assert branch_count >= 1

        message_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE provider = ? AND conversation_id = ?",
            ("test", "conv-1"),
        ).fetchone()[0]
        assert message_count == 2


def test_conversation_service_caches_state(tmp_path, monkeypatch):
    service = _build_service(tmp_path)
    repo = service.state_repo
    original_load = repo.load
    call_count = {"value": 0}

    def wrapped_load():
        call_count["value"] += 1
        return original_load()

    monkeypatch.setattr(repo, "load", wrapped_load)

    monkeypatch.setattr(service, "_current_signature", lambda: (1, 1))

    first = service.load_state()
    second = service.load_state()
    assert first is second
    assert call_count["value"] == 1

    service.invalidate_state_cache()
    monkeypatch.setattr(service, "_current_signature", lambda: (2, 2))
    refreshed = service.load_state()
    assert call_count["value"] == 2
    assert refreshed is service.load_state()
