from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest

from polylogue.config import default_config, load_config, write_config
from polylogue.health import get_health
from polylogue.index import rebuild_index
from polylogue.search import search_messages
from polylogue.ingest import IngestBundle, ingest_bundle
from polylogue.store import ConversationRecord, MessageRecord


def _seed_conversation():
    ingest_bundle(
        IngestBundle(
            conversation=ConversationRecord(
                conversation_id="conv:hash",
                provider_name="codex",
                provider_conversation_id="conv",
                title="Demo",
                created_at=None,
                updated_at=None,
                content_hash="hash",
                provider_meta=None,
            ),
            messages=[
                MessageRecord(
                    message_id="msg:hash",
                    conversation_id="conv:hash",
                    provider_message_id="msg",
                    role="user",
                    text="hello world",
                    timestamp=None,
                    content_hash="hash",
                    provider_meta=None,
                )
            ],
            attachments=[],
        )
    )


def test_search_after_index(workspace_env):
    _seed_conversation()
    rebuild_index()
    results = search_messages("hello", archive_root=workspace_env["archive_root"], limit=5)
    assert results.hits
    assert results.hits[0].conversation_id == "conv:hash"


def test_health_cached(workspace_env):
    config = default_config()
    write_config(config)
    loaded = load_config()
    first = get_health(loaded)
    second = get_health(loaded)
    assert second.get("cached") is True
    assert second.get("age_seconds") is not None


def test_search_invalid_query_reports_error(monkeypatch, workspace_env):
    class StubCursor:
        def __init__(self, row=None):
            self._row = row

        def fetchone(self):
            return self._row

        def fetchall(self):
            return []

    class StubConn:
        def execute(self, sql, params=()):
            if "sqlite_master" in sql:
                return StubCursor(row={"name": "messages_fts"})
            if "MATCH" in sql:
                raise sqlite3.OperationalError("fts5: syntax error")
            return StubCursor()

    @contextmanager
    def stub_open_connection(_):
        yield StubConn()

    monkeypatch.setattr("polylogue.search.open_connection", stub_open_connection)
    with pytest.raises(RuntimeError, match="Invalid search query"):
        search_messages('"unterminated', archive_root=workspace_env["archive_root"], limit=5)


def test_search_prefers_legacy_render_when_present(workspace_env):
    archive_root = workspace_env["archive_root"]
    provider_name = "bad/provider"
    conversation_id = "conv/one"
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id=conversation_id,
            provider_name=provider_name,
            provider_conversation_id=conversation_id,
            title="Legacy",
            created_at=None,
            updated_at=None,
            content_hash="hash",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:legacy",
                conversation_id=conversation_id,
                provider_message_id="msg",
                role="user",
                text="hello legacy",
                timestamp=None,
                content_hash="hash",
                provider_meta=None,
            )
        ],
        attachments=[],
    )
    ingest_bundle(bundle)
    rebuild_index()

    legacy_path = archive_root / "render" / provider_name / conversation_id / "conversation.md"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy", encoding="utf-8")

    results = search_messages("hello", archive_root=archive_root, limit=5)
    assert results.hits
    assert results.hits[0].conversation_path == legacy_path
