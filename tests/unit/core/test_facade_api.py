"""Unit-level public API contracts for the Polylogue facade."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue import Polylogue
from polylogue.archive_products import (
    DaySessionSummaryProductQuery,
    SessionEnrichmentProductQuery,
    SessionPhaseProductQuery,
    SessionProfileProductQuery,
    SessionTagRollupQuery,
    WeekSessionSummaryProductQuery,
    WorkThreadProductQuery,
)
from polylogue.facade import ArchiveStats
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.storage.store import ConversationRecord, MessageRecord

ARCHIVE_STATS_PARAMS = [
    (10, 50, 1000, {"claude-ai": 7, "chatgpt": 3}, {"test": 2, "work": 3}, None),
    (5, 25, 500, {"claude-ai": 5}, {}, "2025-01-15T12:30:45Z"),
    (0, 0, 0, {}, {}, None),
    (1, 1, 10, {"claude-ai": 1}, {}, None),
    (
        20,
        100,
        2000,
        {"claude-ai": 10, "chatgpt": 5, "gemini": 5},
        {"personal": 1, "work": 2},
        "2025-01-20T10:00:00Z",
    ),
]

LIST_CONV_FILTERS = [
    (0, None, None, None, 0),
    (3, None, None, None, 3),
    (4, "claude-ai", None, None, 2),
    (5, None, "inbox", None, 5),
    (5, None, "inbox", 3, 3),
]


class TestPolylogueInitialization:
    """Simple constructor and repr behavior stays at unit level."""

    def test_init_with_defaults(self):
        archive = Polylogue()
        assert archive is not None
        assert archive.archive_root is not None
        assert archive.config is not None

    def test_init_with_custom_archive_root(self, tmp_path):
        custom_root = tmp_path / "custom_archive"
        archive = Polylogue(archive_root=custom_root)
        assert archive.archive_root == custom_root

    def test_init_with_expanduser(self):
        archive = Polylogue(archive_root="~/test_polylogue")
        assert "~" not in str(archive.archive_root)
        assert archive.archive_root.is_absolute()

    def test_repr(self, tmp_path):
        archive = Polylogue(archive_root=tmp_path / "archive")
        repr_str = repr(archive)
        assert "Polylogue" in repr_str
        assert "archive" in repr_str


class TestArchiveStatsCreation:
    """Test ArchiveStats instantiation and attributes."""

    @pytest.mark.parametrize(
        "conv_count,msg_count,word_count,providers,tags,last_sync",
        ARCHIVE_STATS_PARAMS,
    )
    def test_archive_stats_init(self, conv_count, msg_count, word_count, providers, tags, last_sync):
        """Test ArchiveStats initialization with various parameter combinations."""
        stats = ArchiveStats(
            conversation_count=conv_count,
            message_count=msg_count,
            word_count=word_count,
            providers=providers,
            tags=tags,
            last_sync=last_sync,
            recent=[],
        )
        assert stats.conversation_count == conv_count
        assert stats.message_count == msg_count
        assert stats.word_count == word_count
        assert stats.providers == providers
        assert stats.tags == tags
        assert stats.last_sync == last_sync
        assert stats.recent == []

    def test_archive_stats_with_recent_conversations(self):
        """Test ArchiveStats with recent conversations."""
        recent_msgs = [
            Message(id="m1", role="user", text="Hello", timestamp=datetime.now(tz=timezone.utc)),
        ]
        recent_conv = Conversation(
            id="conv1",
            provider="claude-ai",
            messages=MessageCollection(messages=recent_msgs),
        )
        stats = ArchiveStats(
            conversation_count=1,
            message_count=1,
            word_count=10,
            providers={"claude-ai": 1},
            tags={},
            last_sync=None,
            recent=[recent_conv],
        )
        assert len(stats.recent) == 1
        assert stats.recent[0].id == "conv1"



# ============================================================================
# POLYLOGUE INITIALIZATION TESTS
# ============================================================================


class TestPolylogueInit:
    """Test Polylogue initialization with various configurations."""

    @pytest.mark.parametrize(
        "db_path_type,property_name",
        [
            (":memory:", "archive_root"),
            ("file.db", "config"),
            ("file.db", "archive_root"),
        ],
    )
    def test_polylogue_init_properties(self, tmp_path, db_path_type, property_name):
        """Test Polylogue initialization with various db_path types and property access."""
        db_path = tmp_path / "test.db" if db_path_type == "file.db" else db_path_type

        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        if property_name == "archive_root":
            assert archive.archive_root == tmp_path
        elif property_name == "config":
            cfg = archive.config
            assert type(cfg).__name__ == "Config"
            assert cfg.archive_root is not None


# ============================================================================
# POLYLOGUE CONVERSATION RETRIEVAL TESTS
# ============================================================================


class TestPolylogueGetConversation:
    """Test getting single conversations."""

    @pytest.mark.asyncio
    async def test_get_conversation_empty_db(self, tmp_path):
        """Test getting conversation from empty database."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        conv = await archive.get_conversation("nonexistent_id")
        assert conv is None

    @pytest.mark.asyncio
    async def test_get_conversation_with_seed_data(self, tmp_path):
        """Test retrieving a conversation after adding data."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        repository = archive.repository

        # Create and save a conversation
        conv_record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude-ai",
            provider_conversation_id="provider-1",
            title="Test Conversation",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash-1",
        )

        # Save messages
        msg_records = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T00:00:00Z",
                content_hash="msg-hash-1",
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                role="assistant",
                text="Hi there",
                timestamp="2025-01-01T00:01:00Z",
                content_hash="msg-hash-2",
            ),
        ]

        await repository.save_conversation(conv_record, msg_records, [])

        # Retrieve by ID
        conv = await archive.get_conversation("conv-1")
        assert conv is not None
        assert conv.id == "conv-1"
        assert conv.title == "Test Conversation"


class TestPolylogueGetConversations:
    """Test batch conversation retrieval."""

    @pytest.mark.asyncio
    async def test_get_conversations_empty_list(self, tmp_path):
        """Test get_conversations with empty ID list."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        convs = await archive.get_conversations([])
        assert convs == []

    @pytest.mark.asyncio
    async def test_get_conversations_batch(self, tmp_path):
        """Test batch retrieval of multiple conversations."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        repository = archive.repository

        # Create multiple conversations
        for i in range(3):
            conv_record = ConversationRecord(
                conversation_id=f"conv-{i}",
                provider_name="claude-ai",
                provider_conversation_id=f"provider-{i}",
                title=f"Conversation {i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"hash-{i}",
            )
            await repository.save_conversation(conv_record, [], [])

        # Retrieve batch
        ids = ["conv-0", "conv-1", "conv-2"]
        convs = await archive.get_conversations(ids)
        assert len(convs) == 3
        assert all(c.id in ids for c in convs)

    @pytest.mark.asyncio
    async def test_get_conversations_partial_match(self, tmp_path):
        """Test batch retrieval with some missing IDs."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        repository = archive.repository

        # Create only conv-1
        conv_record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude-ai",
            provider_conversation_id="provider-1",
            title="Conversation 1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash-1",
        )
        await repository.save_conversation(conv_record, [], [])

        # Request multiple IDs but only conv-1 exists
        ids = ["conv-1", "conv-999"]
        convs = await archive.get_conversations(ids)
        assert len(convs) == 1
        assert convs[0].id == "conv-1"


class TestPolylogueArchiveProducts:
    @pytest.mark.asyncio
    async def test_durable_session_products_are_publicly_queryable(self, cli_workspace):
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.session_product_lifecycle import rebuild_session_products_sync
        from tests.infra.storage_records import ConversationBuilder

        db_path = cli_workspace["db_path"]
        (
            ConversationBuilder(db_path, "conv-root")
            .provider("claude-code")
            .title("Root Thread")
            .updated_at("2026-03-01T10:10:00+00:00")
            .add_message(
                "u1",
                role="user",
                text="Plan the refactor",
                timestamp="2026-03-01T10:00:00+00:00",
            )
            .add_message(
                "a1",
                role="assistant",
                text="Editing files",
                timestamp="2026-03-01T10:05:00+00:00",
                provider_meta={
                    "content_blocks": [
                        {
                            "type": "tool_use",
                            "tool_name": "Edit",
                            "semantic_type": "file_edit",
                            "input": {"path": "/realm/project/polylogue/polylogue/facade.py"},
                        }
                    ]
                },
            )
            .save()
        )
        (
            ConversationBuilder(db_path, "conv-child")
            .provider("claude-code")
            .title("Child Thread")
            .parent_conversation("conv-root")
            .branch_type("continuation")
            .updated_at("2026-03-01T11:05:00+00:00")
            .add_message(
                "u2",
                role="user",
                text="Run tests",
                timestamp="2026-03-01T11:00:00+00:00",
            )
            .save()
        )
        with open_connection(db_path) as conn:
            rebuild_session_products_sync(conn)

        archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
        profile = await archive.get_session_profile_product("conv-root")
        profiles = await archive.list_session_profile_products(
            SessionProfileProductQuery(
                provider="claude-code",
                first_message_since="2026-03-01T00:00:00+00:00",
                session_date_since="2026-03-01",
                limit=10,
            )
        )
        enrichments = await archive.list_session_enrichment_products(
            SessionEnrichmentProductQuery(
                provider="claude-code",
                session_date_since="2026-03-01",
                limit=10,
            )
        )
        phases = await archive.list_session_phase_products(
            SessionPhaseProductQuery(provider="claude-code", limit=10)
        )
        threads = await archive.list_work_thread_products(WorkThreadProductQuery(limit=10))

        assert profile is not None
        assert profile.product_kind == "session_profile"
        assert profile.title == "Root Thread"
        assert profile.evidence is not None
        assert profile.evidence.canonical_session_date == "2026-03-01"
        assert profile.inference is not None
        assert profile.inference.engaged_duration_ms >= 0

        evidence_only = await archive.get_session_profile_product("conv-root", tier="evidence")
        inference_only = await archive.get_session_profile_product("conv-root", tier="inference")
        assert evidence_only is not None
        assert evidence_only.semantic_tier == "evidence"
        assert evidence_only.evidence is not None
        assert evidence_only.inference is None
        assert inference_only is not None
        assert inference_only.semantic_tier == "inference"
        assert inference_only.evidence is None
        assert inference_only.inference is not None

        assert any(item.conversation_id == "conv-root" for item in profiles)
        assert any(item.conversation_id == "conv-root" for item in enrichments)
        assert enrichments[0].enrichment.refined_work_kind is not None
        assert any(item.conversation_id == "conv-root" for item in phases)
        assert len(threads) == 1
        assert threads[0].thread["session_count"] == 2

        tag_rollups = await archive.list_session_tag_rollup_products(
            SessionTagRollupQuery(provider="claude-code")
        )
        day_summaries = await archive.list_day_session_summary_products(
            DaySessionSummaryProductQuery(provider="claude-code", limit=10)
        )
        week_summaries = await archive.list_week_session_summary_products(
            WeekSessionSummaryProductQuery(provider="claude-code", limit=10)
        )

        assert any(item.tag == "provider:claude-code" for item in tag_rollups)
        assert len(day_summaries) == 1
        assert day_summaries[0].summary["session_count"] == 2
        assert len(week_summaries) == 1
        assert week_summaries[0].summary["session_count"] == 2


# ============================================================================
# POLYLOGUE LIST CONVERSATIONS TESTS
# ============================================================================


class TestPolylogueListConversations:
    """Test listing conversations with various filters."""

    @pytest.mark.parametrize(
        "setup_count,provider_filter,source_filter,limit,expected_count",
        LIST_CONV_FILTERS,
    )
    @pytest.mark.asyncio
    async def test_list_conversations_with_filters(self, tmp_path, setup_count, provider_filter, source_filter, limit, expected_count):
        """Test listing conversations with various filter combinations."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        repository = archive.repository

        # Setup conversations
        for i in range(setup_count):
            provider = "claude-ai" if i < 2 else "chatgpt"
            conv_record = ConversationRecord(
                conversation_id=f"conv-{i}",
                provider_name=provider,
                provider_conversation_id=f"provider-{i}",
                title=f"Conv {i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"hash-{i}",
                provider_meta={"source": source_filter or "inbox"} if source_filter else {},
            )
            await repository.save_conversation(conv_record, [], [])

        # Retrieve with filters
        convs = await archive.list_conversations(
            provider=provider_filter,
            limit=limit,
        )
        assert len(convs) == expected_count


# ============================================================================
# POLYLOGUE FILTER AND CONTEXT MANAGER TESTS
# ============================================================================


class TestPolylogueFilter:
    """Test filter builder creation."""

    def test_filter_returns_conversation_filter(self, tmp_path):
        """Test that filter() returns a ConversationFilter."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        filter_builder = archive.filter()
        assert filter_builder is not None
        # ConversationFilter has provider method
        assert hasattr(filter_builder, "provider")

    def test_filter_chaining(self, tmp_path):
        """Test filter method chaining."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        filter_builder = archive.filter()
        # Should support chaining
        result = filter_builder.provider("claude-ai")
        assert result is not None


class TestPolylogueContextManager:
    """Test Polylogue as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter_returns_self(self, tmp_path):
        """Test __aenter__ returns self."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        async with archive as ctx:
            assert ctx is archive

    @pytest.mark.asyncio
    async def test_context_manager_exit_calls_close(self, tmp_path):
        """Test __aexit__ calls close."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        # Just verify context manager works without exception
        async with archive:
            pass
        # If we get here without exception, context manager works properly

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self, tmp_path):
        """Test context manager properly closes on exception."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        try:
            async with archive:
                raise ValueError("Test error")
        except ValueError:
            pass
        # If we get here, context manager handled exception properly

    @pytest.mark.asyncio
    async def test_close_method(self, tmp_path):
        """Test close() method."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        # Should not raise
        await archive.close()


# ============================================================================
# POLYLOGUE REBUILD INDEX AND STATS TESTS
# ============================================================================


class TestPolylogueRebuildIndex:
    """Test index rebuilding."""

    @pytest.mark.asyncio
    async def test_rebuild_index_lazy_init(self, tmp_path):
        """Test that rebuild_index can be called."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        # Just verify the method exists and can be called
        assert hasattr(archive, "rebuild_index")
        assert callable(archive.rebuild_index)
        # Can call it (it's async)
        result = await archive.rebuild_index()
        assert isinstance(result, bool)


class TestPolylogueStats:
    """Test statistics generation."""

    @pytest.mark.asyncio
    async def test_stats_empty_db(self, tmp_path):
        """Test stats() on empty database."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        stats = await archive.stats()

        assert isinstance(stats, ArchiveStats)
        assert stats.conversation_count == 0
        assert stats.message_count == 0
        assert stats.word_count == 0
        assert stats.providers == {}
        assert stats.tags == {}

    @pytest.mark.asyncio
    async def test_stats_with_conversations(self, tmp_path):
        """Test stats() with conversations in database."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        repository = archive.repository

        # Create conversations with different providers
        for i in range(2):
            conv_record = ConversationRecord(
                conversation_id=f"claude-{i}",
                provider_name="claude-ai",
                provider_conversation_id=f"p-{i}",
                title=f"Claude Conv {i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"h-{i}",
            )

            # Add messages
            msg_records = [
                MessageRecord(
                    message_id=f"msg-{i}-0",
                    conversation_id=f"claude-{i}",
                    role="user",
                    text="Hello world",
                    timestamp="2025-01-01T00:00:00Z",
                    content_hash=f"mh-{i}-0",
                ),
                MessageRecord(
                    message_id=f"msg-{i}-1",
                    conversation_id=f"claude-{i}",
                    role="assistant",
                    text="Hi there friend",
                    timestamp="2025-01-01T00:01:00Z",
                    content_hash=f"mh-{i}-1",
                ),
            ]
            await repository.save_conversation(conv_record, msg_records, [])

        # Add ChatGPT conversation
        conv_record = ConversationRecord(
            conversation_id="chatgpt-0",
            provider_name="chatgpt",
            provider_conversation_id="p-chatgpt-0",
            title="ChatGPT Conv",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="h-chatgpt",
        )
        msg_records = [
            MessageRecord(
                message_id="msg-chatgpt-0",
                conversation_id="chatgpt-0",
                role="user",
                text="Test",
                timestamp="2025-01-01T00:00:00Z",
                content_hash="mh-chatgpt",
            ),
        ]
        await repository.save_conversation(conv_record, msg_records, [])

        stats = await archive.stats()
        assert stats.conversation_count == 3
        assert stats.message_count == 5
        assert "claude-ai" in stats.providers
        assert "chatgpt" in stats.providers
        assert stats.providers["claude-ai"] == 2
        assert stats.providers["chatgpt"] == 1

    @pytest.mark.asyncio
    async def test_stats_recent_conversations(self, tmp_path):
        """Test that stats includes recent conversations."""
        db_path = tmp_path / "test.db"
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)
        repository = archive.repository

        # Create a single conversation
        conv_record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude-ai",
            provider_conversation_id="p-1",
            title="Test Conv",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-15T12:00:00Z",
            content_hash="h-1",
        )
        await repository.save_conversation(conv_record, [], [])

        stats = await archive.stats()
        assert len(stats.recent) == 1
        assert stats.recent[0].id == "conv-1"


# ============================================================================
# CLI HELPERS TESTS: SOURCE STATE MANAGEMENT
# ============================================================================
