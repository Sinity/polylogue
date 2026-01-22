from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pytest

from polylogue.config import default_config, load_config, write_config
from polylogue.health import get_health
from polylogue.index import rebuild_index
from polylogue.ingest import IngestBundle, ingest_bundle
from polylogue.search import search_messages
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
    get_health(loaded)
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
    from polylogue.db import DatabaseError
    with pytest.raises(DatabaseError, match="Invalid search query"):
        search_messages('"unterminated', archive_root=workspace_env["archive_root"], limit=5)


def test_search_prefers_legacy_render_when_present(workspace_env):
    """Test that search returns legacy render paths when they exist.

    Note: Invalid provider names are now rejected at validation, so we use a valid
    provider name but still test legacy path resolution behavior.
    """
    archive_root = workspace_env["archive_root"]
    provider_name = "legacy-provider"  # Valid provider name (path chars now rejected)
    conversation_id = "conv-one"
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

    # Create a legacy-style render path
    legacy_path = archive_root / "render" / provider_name / conversation_id / "conversation.md"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy", encoding="utf-8")

    results = search_messages("hello", archive_root=archive_root, limit=5)
    assert results.hits
    assert results.hits[0].conversation_path == legacy_path


# --since timestamp filtering tests

def test_search_since_filters_by_iso_date(workspace_env):
    """--since with ISO date filters messages correctly."""
    archive_root = workspace_env["archive_root"]
    # Message with ISO timestamp: 2024-01-15T10:00:00
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv:iso",
            provider_name="test",
            provider_conversation_id="iso-conv",
            title="ISO Test",
            created_at=None,
            updated_at=None,
            content_hash="hash-iso",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:old-iso",
                conversation_id="conv:iso",
                provider_message_id="old",
                role="user",
                text="old message iso",
                timestamp="2024-01-10T10:00:00",  # Before cutoff
                content_hash="hash1",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="msg:new-iso",
                conversation_id="conv:iso",
                provider_message_id="new",
                role="user",
                text="new message iso",
                timestamp="2024-01-20T10:00:00",  # After cutoff
                content_hash="hash2",
                provider_meta=None,
            ),
        ],
        attachments=[],
    )
    ingest_bundle(bundle)
    rebuild_index()

    # Filter for messages after 2024-01-15
    results = search_messages(
        "message",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    assert len(results.hits) == 1
    assert results.hits[0].message_id == "msg:new-iso"


def test_search_since_filters_numeric_timestamps(workspace_env):
    """--since works when DB has float timestamps (e.g., 1704067200.0)."""
    archive_root = workspace_env["archive_root"]
    # 1705312800.0 = 2024-01-15T10:00:00 UTC
    # 1704067200.0 = 2024-01-01T00:00:00 UTC
    # 1706227200.0 = 2024-01-26T00:00:00 UTC
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv:numeric",
            provider_name="test",
            provider_conversation_id="numeric-conv",
            title="Numeric Test",
            created_at=None,
            updated_at=None,
            content_hash="hash-numeric",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:old-num",
                conversation_id="conv:numeric",
                provider_message_id="old",
                role="user",
                text="old message numeric",
                timestamp="1704067200.0",  # 2024-01-01 - Before cutoff
                content_hash="hash3",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="msg:new-num",
                conversation_id="conv:numeric",
                provider_message_id="new",
                role="user",
                text="new message numeric",
                timestamp="1706227200.0",  # 2024-01-26 - After cutoff
                content_hash="hash4",
                provider_meta=None,
            ),
        ],
        attachments=[],
    )
    ingest_bundle(bundle)
    rebuild_index()

    # Filter for messages after 2024-01-15
    results = search_messages(
        "numeric",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    assert len(results.hits) == 1
    assert results.hits[0].message_id == "msg:new-num"


def test_search_since_handles_mixed_timestamp_formats(workspace_env):
    """--since works with mix of ISO and numeric timestamps in same DB.

    Note: Search results are deduplicated by conversation, so we create
    separate conversations to verify both ISO and numeric timestamps work.
    """
    archive_root = workspace_env["archive_root"]

    # Create conversation with ISO timestamp (after cutoff)
    bundle_iso = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv:iso-new",
            provider_name="test",
            provider_conversation_id="iso-conv",
            title="ISO Test",
            created_at=None,
            updated_at=None,
            content_hash="hash-iso",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:iso-new",
                conversation_id="conv:iso-new",
                provider_message_id="iso-new",
                role="user",
                text="mixedformat gamma",
                timestamp="2024-01-25T12:00:00",  # ISO - after cutoff
                content_hash="hash7",
                provider_meta=None,
            ),
        ],
        attachments=[],
    )

    # Create conversation with numeric timestamp (after cutoff)
    bundle_num = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv:num-new",
            provider_name="test",
            provider_conversation_id="num-conv",
            title="Numeric Test",
            created_at=None,
            updated_at=None,
            content_hash="hash-num",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:num-new",
                conversation_id="conv:num-new",
                provider_message_id="num-new",
                role="user",
                text="mixedformat delta",
                timestamp="1706400000.0",  # 2024-01-28 - after cutoff
                content_hash="hash8",
                provider_meta=None,
            ),
        ],
        attachments=[],
    )

    # Create conversation with old ISO timestamp (before cutoff)
    bundle_old = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv:old",
            provider_name="test",
            provider_conversation_id="old-conv",
            title="Old Test",
            created_at=None,
            updated_at=None,
            content_hash="hash-old",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:iso-old",
                conversation_id="conv:old",
                provider_message_id="iso-old",
                role="user",
                text="mixedformat alpha",
                timestamp="2024-01-05T12:00:00",  # ISO - before cutoff
                content_hash="hash5",
                provider_meta=None,
            ),
        ],
        attachments=[],
    )

    ingest_bundle(bundle_iso)
    ingest_bundle(bundle_num)
    ingest_bundle(bundle_old)
    rebuild_index()

    results = search_messages(
        "mixedformat",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    # Should get 2 hits: one ISO, one numeric - both after cutoff
    assert len(results.hits) == 2
    hit_conv_ids = {h.conversation_id for h in results.hits}
    assert hit_conv_ids == {"conv:iso-new", "conv:num-new"}


def test_search_since_invalid_date_raises_error(workspace_env):
    """Invalid --since format raises ValueError with helpful message."""
    archive_root = workspace_env["archive_root"]
    _seed_conversation()
    rebuild_index()

    with pytest.raises(ValueError, match="Invalid --since date"):
        search_messages(
            "hello",
            archive_root=archive_root,
            since="not-a-date",
            limit=5,
        )

    with pytest.raises(ValueError, match="ISO format"):
        search_messages(
            "hello",
            archive_root=archive_root,
            since="01/15/2024",  # Wrong format
            limit=5,
        )


def test_search_since_boundary_condition(workspace_env):
    """Messages at or after --since timestamp are included, earlier ones excluded."""
    archive_root = workspace_env["archive_root"]
    # Use dates far enough apart that timezone differences don't matter
    # Filter: 2024-01-15 (any timezone interpretation)
    # Before: 2024-01-10 (definitely before, any timezone)
    # After: 2024-01-20 (definitely after, any timezone)
    bundle = IngestBundle(
        conversation=ConversationRecord(
            conversation_id="conv:boundary",
            provider_name="test",
            provider_conversation_id="boundary-conv",
            title="Boundary Test",
            created_at=None,
            updated_at=None,
            content_hash="hash-boundary",
            provider_meta=None,
        ),
        messages=[
            MessageRecord(
                message_id="msg:after-cutoff",
                conversation_id="conv:boundary",
                provider_message_id="after",
                role="user",
                text="boundary after message",
                timestamp="2024-01-20T12:00:00",  # Clearly after cutoff
                content_hash="hash9",
                provider_meta=None,
            ),
            MessageRecord(
                message_id="msg:before-cutoff",
                conversation_id="conv:boundary",
                provider_message_id="before",
                role="user",
                text="boundary before message",
                timestamp="2024-01-10T12:00:00",  # Clearly before cutoff
                content_hash="hash10",
                provider_meta=None,
            ),
        ],
        attachments=[],
    )
    ingest_bundle(bundle)
    rebuild_index()

    results = search_messages(
        "boundary",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    # Should include after, exclude before
    assert len(results.hits) == 1
    assert results.hits[0].message_id == "msg:after-cutoff"


def test_search_without_fts_table_raises_descriptive_error(workspace_env, db_without_fts, monkeypatch):
    """search() raises DatabaseError mentioning 'polylogue run' when FTS missing."""
    archive_root = workspace_env["archive_root"]

    # Monkey-patch to use the db without FTS
    from polylogue import db
    from polylogue.db import DatabaseError

    monkeypatch.setattr(db, "default_db_path", lambda: db_without_fts)

    with pytest.raises(DatabaseError, match="Search index not built"):
        search_messages("hello", archive_root=archive_root, limit=5)
