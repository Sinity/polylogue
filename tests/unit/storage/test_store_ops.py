"""Focused roundtrip and validation contracts for storage record helpers."""

from __future__ import annotations

import importlib
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import (
    MAX_ATTACHMENT_SIZE,
    AttachmentRecord,
    ConversationRecord,
    _json_or_none,
)
from tests.infra.storage_records import (
    _make_ref_id,
    _prune_attachment_refs,
    make_attachment,
    make_conversation,
    make_message,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)
from tests.infra.strategies.messages import conversation_strategy
from tests.infra.strategies.storage import (
    TagAssignmentSpec,
    TitleSearchSpec,
    expected_tag_counts,
    literal_title_search_strategy as infra_title_search_strategy,
    seed_conversation_graph,
    tag_assignment_strategy as infra_tag_assignment_strategy,
)


def _conversation_row(conn, conversation_id: str):
    return conn.execute(
        "SELECT * FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()


def _message_count(conn, conversation_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()[0]


def _attachment_row(conn, attachment_id: str):
    return conn.execute(
        "SELECT * FROM attachments WHERE attachment_id = ?",
        (attachment_id,),
    ).fetchone()


def test_store_records_roundtrip_contract(test_conn) -> None:
    """store_records() must insert, skip, update, and handle sparse payloads coherently."""
    initial = make_conversation("conv-create", content_hash="hash-create")
    created = store_records(
        conversation=initial,
        messages=[make_message("msg-create", "conv-create", text="Hello")],
        attachments=[],
        conn=test_conn,
    )
    assert created == {
        "conversations": 1,
        "messages": 1,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    assert _conversation_row(test_conn, "conv-create")["title"] == "Test Conversation"
    assert _message_count(test_conn, "conv-create") == 1

    duplicate = store_records(
        conversation=initial,
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert duplicate["conversations"] == 0
    assert duplicate["skipped_conversations"] == 1

    updated = store_records(
        conversation=make_conversation("conv-create", title="Updated Title", content_hash="hash-updated"),
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert updated["conversations"] == 1
    assert _conversation_row(test_conn, "conv-create")["title"] == "Updated Title"
    assert _conversation_row(test_conn, "conv-create")["content_hash"] == "hash-updated"

    multi = store_records(
        conversation=make_conversation("conv-multi", title="Multi Message"),
        messages=[
            make_message(f"msg-multi-{idx}", "conv-multi", role="user" if idx % 2 == 0 else "assistant", text=f"Message {idx}")
            for idx in range(5)
        ],
        attachments=[],
        conn=test_conn,
    )
    assert multi["messages"] == 5
    assert _message_count(test_conn, "conv-multi") == 5

    sparse = store_records(
        conversation=make_conversation("conv-empty", title="Empty Conversation"),
        messages=[],
        attachments=[
            make_attachment(
                "att-empty",
                "conv-empty",
                message_id=None,
                mime_type="application/pdf",
                size_bytes=5000,
            )
        ],
        conn=test_conn,
    )
    assert sparse["conversations"] == 1
    assert sparse["messages"] == 0
    assert sparse["attachments"] == 1
    assert _attachment_row(test_conn, "att-empty")["ref_count"] == 1


def test_prune_attachment_refs_contract(test_conn) -> None:
    """Pruning refs must keep requested refs, recalculate counts, and delete zero-ref attachments."""
    conv = make_conversation("conv-prune", title="Prune Test")
    msg1 = make_message("msg-prune-1", "conv-prune", provider_message_id="ext-1", text="First")
    msg2 = make_message("msg-prune-2", "conv-prune", provider_message_id="ext-2", text="Second")
    att1 = make_attachment("att-prune-1", "conv-prune", "msg-prune-1", mime_type="image/png")
    att2 = make_attachment("att-prune-2", "conv-prune", "msg-prune-2", mime_type="image/jpeg", size_bytes=2048)
    shared_att_1 = make_attachment("att-shared", "conv-prune", "msg-prune-1", mime_type="image/png")
    shared_att_2 = make_attachment("att-shared", "conv-prune", "msg-prune-2", mime_type="image/png")
    store_records(
        conversation=conv,
        messages=[msg1, msg2],
        attachments=[att1, att2, shared_att_1, shared_att_2],
        conn=test_conn,
    )

    keep_ref = _make_ref_id("att-prune-1", "conv-prune", "msg-prune-1")
    keep_shared = _make_ref_id("att-shared", "conv-prune", "msg-prune-1")
    _prune_attachment_refs(test_conn, "conv-prune", {keep_ref, keep_shared})

    remaining_refs = test_conn.execute(
        "SELECT ref_id FROM attachment_refs WHERE conversation_id = ? ORDER BY ref_id",
        ("conv-prune",),
    ).fetchall()
    assert [row["ref_id"] for row in remaining_refs] == sorted([keep_ref, keep_shared])
    assert _attachment_row(test_conn, "att-prune-1")["ref_count"] == 1
    assert _attachment_row(test_conn, "att-shared")["ref_count"] == 1
    assert _attachment_row(test_conn, "att-prune-2") is None


def test_upsert_optional_and_attachment_contracts(test_conn) -> None:
    """Optional-field upserts and attachment metadata updates must round-trip cleanly."""
    conversation = ConversationRecord(
        conversation_id="conv-optional",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title=None,
        created_at=None,
        updated_at=None,
        content_hash="hash1",
        provider_meta=None,
    )
    assert upsert_conversation(test_conn, conversation) is True
    conv_row = _conversation_row(test_conn, "conv-optional")
    assert conv_row["title"] is None
    assert conv_row["created_at"] is None
    assert conv_row["provider_meta"] is None

    message = make_message(
        "msg-optional",
        "conv-optional",
        role=None,
        text=None,
        timestamp=None,
        provider_message_id=None,
        provider_meta=None,
    )
    assert upsert_message(test_conn, message) is True
    msg_row = test_conn.execute(
        "SELECT * FROM messages WHERE message_id = ?",
        ("msg-optional",),
    ).fetchone()
    assert msg_row["role"] is None
    assert msg_row["text"] is None
    assert msg_row["provider_message_id"] is None

    msg2 = make_message("msg-attachment-2", "conv-optional", provider_message_id="ext-msg-2", text="Second")
    assert upsert_message(test_conn, msg2) is True
    first = make_attachment("att-meta", "conv-optional", "msg-optional", mime_type="image/png")
    second = make_attachment(
        "att-meta",
        "conv-optional",
        "msg-attachment-2",
        mime_type="image/jpeg",
        size_bytes=2048,
        path="/new/path.jpg",
    )
    assert upsert_attachment(test_conn, first) is True
    assert upsert_attachment(test_conn, first) is False
    assert upsert_attachment(test_conn, second) is True
    att_row = _attachment_row(test_conn, "att-meta")
    assert att_row["mime_type"] == "image/jpeg"
    assert att_row["size_bytes"] == 2048
    assert att_row["path"] == "/new/path.jpg"
    assert att_row["ref_count"] == 2


def test_json_or_none_contract() -> None:
    """JSON serialization helper must preserve mappings and None."""
    import json

    payloads = [
        ({"key": "value"}, {"key": "value"}),
        ({"nested": {"key": "value"}, "list": [1, 2, 3]}, {"nested": {"key": "value"}, "list": [1, 2, 3]}),
        (None, None),
    ]
    for input_val, expected in payloads:
        result = _json_or_none(input_val)
        if expected is None:
            assert result is None
        else:
            assert json.loads(result) == expected


def test_make_ref_id_contract() -> None:
    """Attachment ref IDs must be deterministic and sensitive to attachment, conversation, and message."""
    same_1 = _make_ref_id("att1", "conv1", "msg1")
    same_2 = _make_ref_id("att1", "conv1", "msg1")
    different_attachment = _make_ref_id("att2", "conv1", "msg1")
    different_conversation = _make_ref_id("att1", "conv2", "msg1")
    none_message_1 = _make_ref_id("att1", "conv1", None)
    none_message_2 = _make_ref_id("att1", "conv1", None)

    assert same_1 == same_2
    assert same_1 != different_attachment
    assert same_1 != different_conversation
    assert none_message_1 == none_message_2
    assert none_message_1 != same_1
    assert same_1.startswith("ref-")
    assert len(same_1) == len("ref-") + 16


@pytest.mark.slow
def test_write_lock_prevents_concurrent_writes(test_db) -> None:
    """Threaded store_records() calls must complete without corrupting conversation or message counts."""
    results = []
    errors = []

    def write_conversation(conv_id: int) -> None:
        try:
            conv = make_conversation(f"conv{conv_id}", title=f"Conversation {conv_id}")
            messages = [make_message(f"msg{conv_id}-{i}", f"conv{conv_id}", text=f"Message {i}") for i in range(3)]
            with open_connection(test_db) as conn:
                results.append(store_records(conversation=conv, messages=messages, attachments=[], conn=conn))
        except Exception as exc:  # pragma: no cover - failure path assertion target
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_conversation, idx) for idx in range(10)]
        for future in as_completed(futures):
            future.result()

    assert errors == []
    assert len(results) == 10
    with open_connection(test_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 10
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 30


def test_store_records_without_connection_creates_own(test_db, tmp_path, monkeypatch) -> None:
    """store_records() must honor the default DB path when no connection is supplied."""
    import polylogue.paths
    import polylogue.storage.backends.connection as connection_module
    from polylogue.storage.backends.connection import _clear_connection_cache

    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    _clear_connection_cache()
    importlib.reload(polylogue.paths)
    importlib.reload(connection_module)

    default_path = connection_module.default_db_path()
    default_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(test_db), str(default_path))

    counts = store_records(
        conversation=make_conversation("conv-default", title="No Conn Test"),
        messages=[],
        attachments=[],
    )
    assert counts["conversations"] == 1

    with open_connection(default_path) as conn:
        assert _conversation_row(conn, "conv-default") is not None


@pytest.mark.slow
def test_concurrent_upsert_same_attachment_ref_count_correct(test_db) -> None:
    """Concurrent upserts of the same attachment must keep ref_count equal to actual refs."""
    shared_attachment_id = "shared-attachment-race-test"

    def create_conversation(index: int) -> None:
        conv = make_conversation(
            f"race-conv-{index}",
            title=f"Race Test {index}",
            created_at=None,
            updated_at=None,
            content_hash=f"hash-{index}",
        )
        msg = make_message(
            f"race-msg-{index}",
            f"race-conv-{index}",
            text="test",
            timestamp=None,
            provider_meta=None,
        )
        attachment = make_attachment(
            shared_attachment_id,
            f"race-conv-{index}",
            f"race-msg-{index}",
            mime_type="text/plain",
            size_bytes=100,
            provider_meta=None,
        )
        with open_connection(test_db) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[attachment], conn=conn)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(create_conversation, range(10)))

    with open_connection(test_db) as conn:
        stored_ref_count = conn.execute(
            "SELECT ref_count FROM attachments WHERE attachment_id = ?",
            (shared_attachment_id,),
        ).fetchone()[0]
        actual_refs = conn.execute(
            "SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?",
            (shared_attachment_id,),
        ).fetchone()[0]

    assert stored_ref_count == 10
    assert actual_refs == 10
    assert stored_ref_count == actual_refs


@pytest.mark.parametrize(
    ("size_bytes", "valid"),
    [
        (0, True),
        (MAX_ATTACHMENT_SIZE, True),
        (None, True),
        (-100, False),
        (MAX_ATTACHMENT_SIZE + 1, False),
    ],
    ids=["zero", "max", "unknown", "negative", "over-max"],
)
def test_attachment_size_bytes_contract(size_bytes, valid) -> None:
    """Attachment size validation must accept supported bounds and reject invalid sizes."""
    if valid:
        record = AttachmentRecord(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=size_bytes,
            provider_meta=None,
        )
        assert record.size_bytes == size_bytes
    else:
        with pytest.raises(ValidationError):
            AttachmentRecord(
                attachment_id="test",
                conversation_id="conv1",
                message_id="msg1",
                mime_type="text/plain",
                size_bytes=size_bytes,
                provider_meta=None,
            )


@pytest.mark.parametrize("name", ["claude-ai", "claude-code", "Provider123"])
def test_provider_name_accepts_valid(name) -> None:
    """Representative provider-name formats should validate."""
    record = ConversationRecord(
        conversation_id="test",
        provider_name=name,
        provider_conversation_id="ext1",
        title="Test",
        content_hash="hash123",
    )
    assert record.provider_name == name


# ============================================================================
# CRUD Laws (from test_crud_laws.py)
# ============================================================================


class TestCrudLaws:
    """Property-based CRUD round-trip laws."""

    @given(conversation_strategy(min_messages=1, max_messages=5))
    @settings(max_examples=30, deadline=None)
    async def test_save_retrieve_roundtrip(self, conv_data: dict):
        """Saving a strategy-generated conversation and retrieving it preserves identity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "roundtrip.db"
            backend = SQLiteBackend(db_path=db_path)

            conv_id = f"test-{conv_data['id'][:16]}"
            provider = conv_data.get("provider", "test")

            conv = make_conversation(
                conversation_id=conv_id,
                provider_name=provider,
                title=conv_data.get("title", "Generated"),
                created_at=conv_data.get("created_at"),
            )

            messages = []
            for i, msg_data in enumerate(conv_data.get("messages", [])):
                msg = make_message(
                    message_id=f"{conv_id}-m{i}",
                    conversation_id=conv_id,
                    role=msg_data.get("role", "user"),
                    text=msg_data.get("text", ""),
                )
                messages.append(msg)

            await backend.save_conversation_record(conv)
            if messages:
                await backend.save_messages(messages)

            retrieved = await backend.get_conversation(conv_id)
            assert retrieved is not None
            assert retrieved.conversation_id == conv_id
            assert retrieved.provider_name == provider

            retrieved_msgs = await backend.get_messages(conv_id)
            assert len(retrieved_msgs) == len(messages)

            await backend.close()

    @given(conversation_strategy(min_messages=1, max_messages=3))
    @settings(max_examples=20, deadline=None)
    async def test_save_is_idempotent(self, conv_data: dict):
        """Saving the same conversation twice yields the same stored data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "idempotent.db"
            backend = SQLiteBackend(db_path=db_path)

            conv_id = f"idem-{conv_data['id'][:16]}"
            conv = make_conversation(
                conversation_id=conv_id,
                provider_name=conv_data.get("provider", "test"),
                title=conv_data.get("title", "Idempotent"),
            )

            # Save twice
            await backend.save_conversation_record(conv)
            await backend.save_conversation_record(conv)

            # Should still be exactly one conversation
            all_convs = await backend.list_conversations(limit=100)
            matching = [c for c in all_convs if c.conversation_id == conv_id]
            assert len(matching) == 1

            await backend.close()


# ============================================================================
# Repository Laws (from test_repository_laws.py)
# ============================================================================


@st.composite
def simple_tag_spec(draw: st.DrawFn) -> dict:
    """Generate a tag assignment spec: conversation ID + list of tags."""
    conv_suffix = draw(st.text(
        min_size=3, max_size=12,
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-"),
    ).filter(lambda s: s[0].isalpha()))
    tags = draw(st.lists(
        st.text(min_size=1, max_size=15, alphabet=st.characters(
            whitelist_categories=("L", "N"), whitelist_characters="-",
        )),
        min_size=1,
        max_size=4,
        unique=True,
    ))
    return {"conversation_id": f"tag-{conv_suffix}", "tags": tags}


@st.composite
def simple_title_search_spec(draw: st.DrawFn) -> dict:
    """Generate a title search spec: title and search substring."""
    words = draw(st.lists(
        st.text(min_size=3, max_size=12, alphabet=st.characters(whitelist_categories=("L",))),
        min_size=2,
        max_size=5,
    ))
    title = " ".join(words)
    search_word = draw(st.sampled_from(words))
    return {"title": title, "search_term": search_word}


class TestTagAssignmentLaws:
    """Property-based tests for tag operations on repository."""

    @given(simple_tag_spec())
    @settings(max_examples=15, deadline=None)
    async def test_add_tag_is_retrievable(self, spec: dict):
        """Adding a tag to a conversation makes it appear in metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "tags.db"
            conv_id = spec["conversation_id"]

            (ConversationBuilder(db_path, conv_id)
             .provider("test")
             .title("Tag Test")
             .add_message("m1", text="Hello")
             .save())

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            tag = spec["tags"][0]
            await repo.add_tag(conv_id, tag)

            conv = await repo.get(conv_id)
            assert conv is not None
            assert tag in conv.tags

            await backend.close()

    @given(simple_tag_spec())
    @settings(max_examples=15, deadline=None)
    async def test_remove_tag_is_idempotent(self, spec: dict):
        """Removing a tag that doesn't exist doesn't crash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "rmtags.db"
            conv_id = spec["conversation_id"]

            (ConversationBuilder(db_path, conv_id)
             .provider("test")
             .title("Remove Tag Test")
             .add_message("m1", text="Hello")
             .save())

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            tag = spec["tags"][0]
            await repo.remove_tag(conv_id, tag)

            conv = await repo.get(conv_id)
            assert conv is not None
            assert tag not in conv.tags

            await backend.close()


class TestTitleSearchLaws:
    """Property-based tests for title-based search."""

    @given(simple_title_search_spec())
    @settings(max_examples=15, deadline=None)
    async def test_title_search_finds_matching(self, spec: dict):
        """Searching by title substring finds the matching conversation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from polylogue.storage.index import rebuild_index
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "search.db"
            conv_id = "search-conv-1"

            (ConversationBuilder(db_path, conv_id)
             .provider("test")
             .title(spec["title"])
             .add_message("m1", text="Search test content")
             .save())

            with open_connection(db_path) as conn:
                rebuild_index(conn)

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec["search_term"])
            found_ids = [str(c.id) for c in results]
            assert conv_id in found_ids, (
                f"Expected to find '{conv_id}' when searching "
                f"title='{spec['title']}' for term='{spec['search_term']}'"
            )

            await backend.close()

    @given(simple_title_search_spec())
    @settings(max_examples=15, deadline=None)
    async def test_title_search_excludes_non_matching(self, spec: dict):
        """Title search doesn't return conversations with unrelated titles."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "nomatch.db"

            (ConversationBuilder(db_path, "match-conv")
             .provider("test")
             .title(spec["title"])
             .add_message("m1", text="Content")
             .save())

            (ConversationBuilder(db_path, "nomatch-conv")
             .provider("test")
             .title("Zzqxjk Wvpnrl Tmygbs")
             .add_message("m2", text="Other content")
             .save())

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec["search_term"])
            found_ids = {str(c.id) for c in results}
            assert "match-conv" in found_ids

            await backend.close()


# ============================================================================
# Search Cache Tests (from test_cache.py)
# ============================================================================


class TestSearchCacheKey:
    """Tests for SearchCacheKey creation and behavior."""

    def test_create_basic(self, tmp_path):
        """Create a basic cache key."""
        from polylogue.storage.search_cache import SearchCacheKey

        key = SearchCacheKey.create(
            query="hello",
            archive_root=tmp_path,
        )
        assert key.query == "hello"
        assert key.archive_root == str(tmp_path)
        assert key.limit == 20  # default
        assert key.source is None
        assert key.since is None

    def test_create_with_all_params(self, tmp_path):
        """Create a cache key with all parameters."""
        from polylogue.storage.search_cache import SearchCacheKey

        key = SearchCacheKey.create(
            query="test query",
            archive_root=tmp_path / "archive",
            render_root_path=tmp_path / "render",
            db_path=tmp_path / "test.db",
            limit=50,
            source="claude-ai",
            since="2024-01-01",
        )
        assert key.query == "test query"
        assert key.limit == 50
        assert key.source == "claude-ai"
        assert key.since == "2024-01-01"
        assert key.render_root_path == str(tmp_path / "render")
        assert key.db_path == str(tmp_path / "test.db")

    def test_key_is_frozen(self, tmp_path):
        """Cache key is immutable (frozen dataclass)."""
        from polylogue.storage.search_cache import SearchCacheKey

        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        # Frozen dataclass should raise on attribute assignment
        with pytest.raises(AttributeError):
            key.query = "changed"

    def test_same_params_same_key(self, tmp_path):
        """Same parameters produce equal keys (same cache version)."""
        from polylogue.storage.search_cache import SearchCacheKey

        key1 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        key2 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        assert key1 == key2

    def test_different_query_different_key(self, tmp_path):
        """Different queries produce different keys."""
        from polylogue.storage.search_cache import SearchCacheKey

        key1 = SearchCacheKey.create(query="hello", archive_root=tmp_path)
        key2 = SearchCacheKey.create(query="world", archive_root=tmp_path)
        assert key1 != key2

    def test_different_limit_different_key(self, tmp_path):
        """Different limits produce different keys."""
        from polylogue.storage.search_cache import SearchCacheKey

        key1 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=10)
        key2 = SearchCacheKey.create(query="test", archive_root=tmp_path, limit=20)
        assert key1 != key2

    def test_none_render_root(self, tmp_path):
        """None render_root_path stored as None."""
        from polylogue.storage.search_cache import SearchCacheKey

        key = SearchCacheKey.create(
            query="test", archive_root=tmp_path, render_root_path=None
        )
        assert key.render_root_path is None

    def test_key_is_hashable(self, tmp_path):
        """Cache key can be used as dict key (hashable)."""
        from polylogue.storage.search_cache import SearchCacheKey

        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        d = {key: "result"}
        assert d[key] == "result"


class TestInvalidateSearchCache:
    """Tests for cache invalidation."""

    def test_invalidation_increments_version(self, tmp_path):
        """Invalidation changes cache version."""
        from polylogue.storage.search_cache import SearchCacheKey, invalidate_search_cache

        key_before = SearchCacheKey.create(query="test", archive_root=tmp_path)
        invalidate_search_cache()
        key_after = SearchCacheKey.create(query="test", archive_root=tmp_path)

        # Keys should differ due to version change
        assert key_before != key_after
        assert key_before.cache_version < key_after.cache_version

    def test_multiple_invalidations(self, tmp_path):
        """Multiple invalidations increment version each time."""
        from polylogue.storage.search_cache import SearchCacheKey, invalidate_search_cache

        v1 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v2 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version
        invalidate_search_cache()
        v3 = SearchCacheKey.create(query="test", archive_root=tmp_path).cache_version

        assert v1 < v2 < v3


class TestCacheStats:
    """Tests for cache statistics."""

    def test_stats_returns_dict(self):
        """get_cache_stats returns a dictionary."""
        from polylogue.storage.search_cache import get_cache_stats

        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert "cache_version" in stats

    def test_stats_version_matches_current(self, tmp_path):
        """Stats version matches what keys use."""
        from polylogue.storage.search_cache import SearchCacheKey, get_cache_stats

        key = SearchCacheKey.create(query="test", archive_root=tmp_path)
        stats = get_cache_stats()
        assert stats["cache_version"] == key.cache_version


# ============================================================================
# Repository Tests (relocated from test_json.py)
# ============================================================================


class TestRepositoryOperations:
    """ConversationRepository CRUD operations."""

    async def test_repository_basic_operations(self, test_db):
        """Test ConversationRepository basic get/list operations."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        factory.create_conversation(
            id="c1", provider="chatgpt", messages=[{"id": "m1", "role": "user", "text": "hello world"}]
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)

        conv = await repo.get("c1")
        assert conv is not None
        assert conv.id == "c1"
        assert len(conv.messages) == 1
        assert conv.messages[0].text == "hello world"

        lst = await repo.list()
        assert len(lst) == 1
        assert lst[0].id == "c1"

    async def test_get_eager_includes_attachment_conversation_id(self, test_db):
        """ConversationRepository.get_eager() returns attachments with conversation_id field."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        factory.create_conversation(
            id="c-with-att",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "message with attachment",
                    "attachments": [
                        {
                            "id": "att1",
                            "mime_type": "image/png",
                            "size_bytes": 2048,
                            "path": "/path/to/image.png",
                        }
                    ],
                }
            ],
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)
        conv = await repo.get_eager("c-with-att")

        assert conv is not None
        assert len(conv.messages) == 1
        msg = conv.messages[0]
        assert len(msg.attachments) == 1
        att = msg.attachments[0]
        assert att.id == "att1"
        assert att.mime_type == "image/png"

    async def test_get_eager_multiple_attachments(self, test_db):
        """get_eager() correctly groups multiple attachments per message."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        factory.create_conversation(
            id="c-multi-att",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "first message",
                    "attachments": [
                        {"id": "att1", "mime_type": "image/png"},
                        {"id": "att2", "mime_type": "image/jpeg"},
                    ],
                },
                {
                    "id": "m2",
                    "role": "assistant",
                    "text": "second message",
                    "attachments": [
                        {"id": "att3", "mime_type": "application/pdf"},
                    ],
                },
            ],
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)
        conv = await repo.get_eager("c-multi-att")

        assert conv is not None
        assert len(conv.messages) == 2

        m1 = conv.messages[0]
        assert len(m1.attachments) == 2
        m1_att_ids = {a.id for a in m1.attachments}
        assert m1_att_ids == {"att1", "att2"}

        m2 = conv.messages[1]
        assert len(m2.attachments) == 1
        assert m2.attachments[0].id == "att3"

    async def test_get_eager_attachment_metadata_decoded(self, test_db):
        """Attachment provider_meta JSON is properly decoded."""
        from tests.infra.storage_records import DbFactory

        factory = DbFactory(test_db)
        meta = {"original_name": "photo.png", "source": "upload"}
        factory.create_conversation(
            id="c-att-meta",
            provider="test",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "text": "with meta",
                    "attachments": [
                        {
                            "id": "att-meta",
                            "mime_type": "image/png",
                            "meta": meta,
                        }
                    ],
                }
            ],
        )

        backend = SQLiteBackend(db_path=test_db)
        repo = ConversationRepository(backend=backend)
        conv = await repo.get_eager("c-att-meta")

        assert conv is not None
        assert len(conv.messages) == 1
        msg = conv.messages[0]
        assert len(msg.attachments) == 1
        att = msg.attachments[0]
        assert att.provider_meta == meta or att.provider_meta is None


class TestCacheThreadSafety:
    """Thread safety tests for cache invalidation."""

    def test_concurrent_invalidation(self):
        """Concurrent invalidation doesn't corrupt state."""
        import threading
        from polylogue.storage.search_cache import invalidate_search_cache, get_cache_stats

        initial_stats = get_cache_stats()
        initial_version = initial_stats["cache_version"]

        errors: list[Exception] = []
        num_threads = 10
        invalidations_per_thread = 100

        def invalidate_many():
            try:
                for _ in range(invalidations_per_thread):
                    invalidate_search_cache()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=invalidate_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        final_stats = get_cache_stats()
        expected_version = initial_version + (num_threads * invalidations_per_thread)
        assert final_stats["cache_version"] == expected_version


# ============================================================================
# TagAssignmentSpec / TitleSearchSpec — infra strategy activation (B5)
# ============================================================================


class TestInfraTagAssignment:
    """Property-based tests using the full TagAssignmentSpec strategy."""

    @given(infra_tag_assignment_strategy(min_conversations=2, max_conversations=4))
    @settings(max_examples=10, deadline=None)
    async def test_tag_assignment_roundtrip(self, spec: TagAssignmentSpec):
        """Tags assigned via strategy-generated specs are retrievable and consistent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository

            db_path = Path(tmp_dir) / "tags-infra.db"
            seed_conversation_graph(db_path, spec.conversations)

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            # Assign all tags
            for conv, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                for tag in tags:
                    await repo.add_tag(conv.conversation_id, tag)

            # Verify each conversation has the expected tags
            for conv, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                stored = await repo.get(conv.conversation_id)
                assert stored is not None
                stored_tags = set(stored.tags)
                for tag in set(tags):
                    assert tag in stored_tags, (
                        f"Tag '{tag}' missing from conv '{conv.conversation_id}'"
                    )

            await backend.close()

    @given(infra_tag_assignment_strategy(min_conversations=2, max_conversations=4))
    @settings(max_examples=10, deadline=None)
    async def test_tag_counts_match_expected(self, spec: TagAssignmentSpec):
        """Tag counts computed from strategy match actual stored tag counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository

            db_path = Path(tmp_dir) / "tag-counts.db"
            seed_conversation_graph(db_path, spec.conversations)

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            for conv, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                for tag in tags:
                    await repo.add_tag(conv.conversation_id, tag)

            expected = expected_tag_counts(spec)
            actual: dict[str, int] = {}
            for conv, tags in zip(spec.conversations, spec.tag_sequences, strict=True):
                stored = await repo.get(conv.conversation_id)
                if stored:
                    for tag in stored.tags:
                        actual[tag] = actual.get(tag, 0) + 1

            assert actual == expected

            await backend.close()


class TestInfraTitleSearch:
    """Property-based tests using the full TitleSearchSpec strategy."""

    @given(infra_title_search_strategy())
    @settings(max_examples=15, deadline=None)
    async def test_literal_title_search_finds_matching_with_special_chars(self, spec: TitleSearchSpec):
        """Title search with wildcard-sensitive characters finds exact matches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "title-search.db"

            (ConversationBuilder(db_path, "match-conv")
             .provider("test")
             .title(spec.matching_title)
             .add_message("m1", text="Content")
             .save())

            (ConversationBuilder(db_path, "decoy-conv")
             .provider("test")
             .title(spec.decoy_title)
             .add_message("m2", text="Other")
             .save())

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec.needle)
            found_ids = {str(c.id) for c in results}
            assert "match-conv" in found_ids, (
                f"Expected 'match-conv' for needle='{spec.needle}' "
                f"in title='{spec.matching_title}'"
            )

            await backend.close()

    @given(infra_title_search_strategy())
    @settings(max_examples=15, deadline=None)
    async def test_literal_title_search_excludes_decoy(self, spec: TitleSearchSpec):
        """Title search with special characters does not match the decoy title."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from polylogue.storage.repository import ConversationRepository
            from tests.infra.storage_records import ConversationBuilder

            db_path = Path(tmp_dir) / "decoy-search.db"

            (ConversationBuilder(db_path, "decoy-only")
             .provider("test")
             .title(spec.decoy_title)
             .add_message("m1", text="Content")
             .save())

            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            results = await repo.list(title_contains=spec.needle)
            found_ids = {str(c.id) for c in results}
            assert "decoy-only" not in found_ids, (
                f"Decoy '{spec.decoy_title}' should not match "
                f"needle='{spec.needle}'"
            )
