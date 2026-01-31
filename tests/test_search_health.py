from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pytest

from polylogue.config import default_config, load_config, write_config
from polylogue.health import get_health
from polylogue.ingestion import IngestBundle, ingest_bundle
from polylogue.storage.index import rebuild_index
from polylogue.storage.search import search_messages
from tests.helpers import make_conversation, make_message


def _seed_conversation(storage_repository):
    ingest_bundle(
        IngestBundle(
            conversation=make_conversation("conv:hash", provider_name="codex", title="Demo"),
            messages=[make_message("msg:hash", "conv:hash", text="hello world")],
            attachments=[],
        ),
        repository=storage_repository,
    )


def test_search_after_index(workspace_env, storage_repository):
    _seed_conversation(storage_repository)
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
    assert second.cached is True
    assert second.age_seconds is not None


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

    monkeypatch.setattr("polylogue.storage.search.open_connection", stub_open_connection)
    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        search_messages('"unterminated', archive_root=workspace_env["archive_root"], limit=5)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Invalid search query" in str(exc_info.value)


def test_search_prefers_legacy_render_when_present(workspace_env, storage_repository):
    """Test that search returns legacy render paths when they exist.

    Note: Invalid provider names are now rejected at validation, so we use a valid
    provider name but still test legacy path resolution behavior.
    """
    archive_root = workspace_env["archive_root"]
    provider_name = "legacy-provider"  # Valid provider name (path chars now rejected)
    conversation_id = "conv-one"
    bundle = IngestBundle(
        conversation=make_conversation(conversation_id, provider_name=provider_name, title="Legacy"),
        messages=[make_message("msg:legacy", conversation_id, text="hello legacy")],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
    rebuild_index()

    # Create a legacy-style render path
    legacy_path = archive_root / "render" / provider_name / conversation_id / "conversation.md"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy", encoding="utf-8")

    results = search_messages("hello", archive_root=archive_root, limit=5)
    assert results.hits
    assert results.hits[0].conversation_path == legacy_path


# --since timestamp filtering tests

def test_search_since_filters_by_iso_date(workspace_env, storage_repository):
    """--since with ISO date filters messages correctly."""
    archive_root = workspace_env["archive_root"]
    # Message with ISO timestamp: 2024-01-15T10:00:00
    bundle = IngestBundle(
        conversation=make_conversation("conv:iso", title="ISO Test"),
        messages=[
            make_message("msg:old-iso", "conv:iso", text="old message iso", timestamp="2024-01-10T10:00:00"),
            make_message("msg:new-iso", "conv:iso", text="new message iso", timestamp="2024-01-20T10:00:00"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
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


def test_search_since_filters_numeric_timestamps(workspace_env, storage_repository):
    """--since works when DB has float timestamps (e.g., 1704067200.0)."""
    archive_root = workspace_env["archive_root"]
    # 1705312800.0 = 2024-01-15T10:00:00 UTC
    # 1704067200.0 = 2024-01-01T00:00:00 UTC
    # 1706227200.0 = 2024-01-26T00:00:00 UTC
    bundle = IngestBundle(
        conversation=make_conversation("conv:numeric", title="Numeric Test"),
        messages=[
            make_message("msg:old-num", "conv:numeric", text="old message numeric", timestamp="1704067200.0"),
            make_message("msg:new-num", "conv:numeric", text="new message numeric", timestamp="1706227200.0"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
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


def test_search_since_handles_mixed_timestamp_formats(workspace_env, storage_repository):
    """--since works with mix of ISO and numeric timestamps in same DB.

    Note: Search results are deduplicated by conversation, so we create
    separate conversations to verify both ISO and numeric timestamps work.
    """
    archive_root = workspace_env["archive_root"]

    # Create conversation with ISO timestamp (after cutoff)
    bundle_iso = IngestBundle(
        conversation=make_conversation("conv:iso-new", title="ISO Test"),
        messages=[make_message("msg:iso-new", "conv:iso-new", text="mixedformat gamma", timestamp="2024-01-25T12:00:00")],
        attachments=[],
    )

    # Create conversation with numeric timestamp (after cutoff)
    bundle_num = IngestBundle(
        conversation=make_conversation("conv:num-new", title="Numeric Test"),
        messages=[make_message("msg:num-new", "conv:num-new", text="mixedformat delta", timestamp="1706400000.0")],
        attachments=[],
    )

    # Create conversation with old ISO timestamp (before cutoff)
    bundle_old = IngestBundle(
        conversation=make_conversation("conv:old", title="Old Test"),
        messages=[make_message("msg:iso-old", "conv:old", text="mixedformat alpha", timestamp="2024-01-05T12:00:00")],
        attachments=[],
    )

    ingest_bundle(bundle_iso, repository=storage_repository)
    ingest_bundle(bundle_num, repository=storage_repository)
    ingest_bundle(bundle_old, repository=storage_repository)
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


def test_search_since_invalid_date_raises_error(workspace_env, storage_repository):
    """Invalid --since format raises ValueError with helpful message."""
    archive_root = workspace_env["archive_root"]
    _seed_conversation(storage_repository)
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


def test_search_since_boundary_condition(workspace_env, storage_repository):
    """Messages at or after --since timestamp are included, earlier ones excluded."""
    archive_root = workspace_env["archive_root"]
    # Use dates far enough apart that timezone differences don't matter
    # Filter: 2024-01-15 (any timezone interpretation)
    # Before: 2024-01-10 (definitely before, any timezone)
    # After: 2024-01-20 (definitely after, any timezone)
    bundle = IngestBundle(
        conversation=make_conversation("conv:boundary", title="Boundary Test"),
        messages=[
            make_message("msg:after-cutoff", "conv:boundary", text="boundary after message", timestamp="2024-01-20T12:00:00"),
            make_message("msg:before-cutoff", "conv:boundary", text="boundary before message", timestamp="2024-01-10T12:00:00"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
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
    from polylogue.storage.backends import sqlite as db

    monkeypatch.setattr(db, "default_db_path", lambda: db_without_fts)

    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        search_messages("hello", archive_root=archive_root, limit=5)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Search index not built" in str(exc_info.value)
