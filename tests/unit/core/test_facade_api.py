"""Unit-level public API contracts for the Polylogue facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.api import ArchiveStats
from polylogue.lib.semantic.content_projection import ContentProjectionSpec
from polylogue.products.archive import (
    ArchiveDebtProductQuery,
    CostRollupProductQuery,
    DaySessionSummaryProductQuery,
    SessionCostProductQuery,
    SessionEnrichmentProductQuery,
    SessionPhaseProductQuery,
    SessionProfileProductQuery,
    SessionTagRollupQuery,
    WeekSessionSummaryProductQuery,
    WorkThreadProductQuery,
)
from tests.infra.builders import make_conv, make_msg
from tests.infra.storage_records import ConversationBuilder, make_conversation, make_message

ArchiveStatsCase = tuple[int, int, int, dict[str, int], dict[str, int], str | None]
ListConversationsCase = tuple[int, str | None, str | None, int | None, int]

ARCHIVE_STATS_PARAMS: list[ArchiveStatsCase] = [
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

LIST_CONV_FILTERS: list[ListConversationsCase] = [
    (0, None, None, None, 0),
    (3, None, None, None, 3),
    (4, "claude-ai", None, None, 2),
    (5, None, "inbox", 3, 3),
]


def _archive(tmp_path: Path, db_name: str = "test.db") -> Polylogue:
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / db_name)


class TestPolylogueInitialization:
    """Simple constructor and repr behavior stays at unit level."""

    def test_init_with_defaults(self: object) -> None:
        archive = Polylogue()
        assert archive.archive_root is not None
        assert archive.config is not None

    def test_init_with_custom_archive_root(self: object, tmp_path: Path) -> None:
        custom_root = tmp_path / "custom_archive"
        archive = Polylogue(archive_root=custom_root)
        assert archive.archive_root == custom_root
        assert archive.config.render_root == custom_root / "render"

    def test_init_with_expanduser(self: object) -> None:
        archive = Polylogue(archive_root="~/test_polylogue")
        assert "~" not in str(archive.archive_root)
        assert archive.archive_root.is_absolute()

    def test_repr(self: object, tmp_path: Path) -> None:
        archive = Polylogue(archive_root=tmp_path / "archive")
        assert "Polylogue" in repr(archive)
        assert "archive" in repr(archive)


class TestArchiveStatsCreation:
    """Test ArchiveStats instantiation and attributes."""

    @pytest.mark.parametrize(
        "conv_count,msg_count,word_count,providers,tags,last_sync",
        ARCHIVE_STATS_PARAMS,
    )
    def test_archive_stats_init(
        self: object,
        conv_count: int,
        msg_count: int,
        word_count: int,
        providers: dict[str, int],
        tags: dict[str, int],
        last_sync: str | None,
    ) -> None:
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

    def test_archive_stats_with_recent_conversations(self: object) -> None:
        recent_conv = make_conv(
            id="conv1",
            provider="claude-ai",
            messages=[make_msg(id="m1", role="user", text="Hello", timestamp="2025-01-01T00:00:00Z")],
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


class TestPolylogueInit:
    @pytest.mark.parametrize(
        "db_path_type,property_name",
        [
            (":memory:", "archive_root"),
            ("file.db", "config"),
            ("file.db", "archive_root"),
        ],
    )
    def test_polylogue_init_properties(
        self: object,
        tmp_path: Path,
        db_path_type: str,
        property_name: str,
    ) -> None:
        db_path: str | Path = tmp_path / "test.db" if db_path_type == "file.db" else db_path_type
        archive = Polylogue(archive_root=tmp_path, db_path=db_path)

        if property_name == "archive_root":
            assert archive.archive_root == tmp_path
        elif property_name == "config":
            assert archive.config.archive_root is not None


class TestPolylogueGetConversation:
    @pytest.mark.asyncio
    async def test_get_conversation_empty_db(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        assert await archive.get_conversation("nonexistent_id") is None

    @pytest.mark.asyncio
    async def test_get_conversation_with_seed_data(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        repository = archive.repository

        conv_record = make_conversation(
            "conv-1",
            provider_name="claude-ai",
            title="Test Conversation",
            provider_conversation_id="provider-1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash-1",
        )
        msg_records = [
            make_message(
                "msg-1",
                "conv-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T00:00:00Z",
                content_hash="msg-hash-1",
            ),
            make_message(
                "msg-2",
                "conv-1",
                role="assistant",
                text="Hi there",
                timestamp="2025-01-01T00:01:00Z",
                content_hash="msg-hash-2",
            ),
        ]
        await repository.save_conversation(conv_record, msg_records, [])

        conv = await archive.get_conversation("conv-1")
        assert conv is not None
        assert conv.id == "conv-1"
        assert conv.title == "Test Conversation"

    @pytest.mark.asyncio
    async def test_get_conversation_applies_content_projection(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        repository = archive.repository

        await repository.save_conversation(
            make_conversation(
                "conv-projection",
                provider_name="claude-ai",
                title="Projected Conversation",
                provider_conversation_id="provider-projection",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash="hash-projection",
            ),
            [
                make_message(
                    "msg-projection",
                    "conv-projection",
                    role="assistant",
                    text="Alpha\n\n```python\nprint('x')\n```\n\nOmega",
                    timestamp="2025-01-01T00:00:00Z",
                    content_hash="msg-hash-projection",
                )
            ],
            [],
        )

        projected = await archive.get_conversation(
            "conv-projection",
            content_projection=ContentProjectionSpec.prose_only(),
        )

        assert projected is not None
        assert len(projected.messages) == 1
        message = next(iter(projected.messages))
        assert message.text == "Alpha\n\nOmega"


class TestPolylogueGetConversations:
    @pytest.mark.asyncio
    async def test_get_conversations_empty_list(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        assert await archive.get_conversations([]) == []

    @pytest.mark.asyncio
    async def test_get_conversations_batch(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        repository = archive.repository

        for i in range(3):
            await repository.save_conversation(
                make_conversation(
                    f"conv-{i}",
                    provider_name="claude-ai",
                    provider_conversation_id=f"provider-{i}",
                    title=f"Conversation {i}",
                    created_at="2025-01-01T00:00:00Z",
                    updated_at="2025-01-01T00:00:00Z",
                    content_hash=f"hash-{i}",
                ),
                [],
                [],
            )

        ids = ["conv-0", "conv-1", "conv-2"]
        convs = await archive.get_conversations(ids)
        assert len(convs) == 3
        assert all(conv.id in ids for conv in convs)

    @pytest.mark.asyncio
    async def test_get_conversations_partial_match(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        repository = archive.repository

        await repository.save_conversation(
            make_conversation(
                "conv-1",
                provider_name="claude-ai",
                provider_conversation_id="provider-1",
                title="Conversation 1",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash="hash-1",
            ),
            [],
            [],
        )

        convs = await archive.get_conversations(["conv-1", "conv-999"])
        assert len(convs) == 1
        assert convs[0].id == "conv-1"


class TestPolylogueArchiveProducts:
    @pytest.mark.asyncio
    async def test_durable_session_products_are_publicly_queryable(
        self: object,
        cli_workspace: dict[str, Path],
    ) -> None:
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.products.session.rebuild import rebuild_session_products_sync

        db_path = cli_workspace["db_path"]
        (
            ConversationBuilder(db_path, "conv-root")
            .provider("claude-code")
            .title("Root Thread")
            .provider_meta({"total_cost_usd": 1.25, "model": "claude-sonnet-4-5"})
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
                            "input": {"path": "/workspace/polylogue/polylogue/api/__init__.py"},
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
        phases = await archive.list_session_phase_products(SessionPhaseProductQuery(provider="claude-code", limit=10))
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
        assert enrichments[0].enrichment.confidence >= 0.0
        assert any(item.conversation_id == "conv-root" for item in phases)
        assert len(threads) == 1
        assert threads[0].thread.session_count == 2
        assert threads[0].thread.support_level == "strong"
        assert threads[0].thread.member_evidence[1].role == "parent_continuation"
        assert threads[0].thread.member_evidence[1].parent_id == "conv-root"

        tag_rollups = await archive.list_session_tag_rollup_products(SessionTagRollupQuery(provider="claude-code"))
        day_summaries = await archive.list_day_session_summary_products(
            DaySessionSummaryProductQuery(provider="claude-code", limit=10)
        )
        week_summaries = await archive.list_week_session_summary_products(
            WeekSessionSummaryProductQuery(provider="claude-code", limit=10)
        )
        archive_debt = await archive.list_archive_debt_products(ArchiveDebtProductQuery(limit=10))
        session_costs = await archive.list_session_cost_products(
            SessionCostProductQuery(provider="claude-code", limit=10)
        )
        cost_rollups = await archive.list_cost_rollup_products(CostRollupProductQuery(provider="claude-code"))

        assert any(item.tag == "provider:claude-code" for item in tag_rollups)
        assert len(day_summaries) == 1
        assert day_summaries[0].summary.session_count == 2
        assert len(week_summaries) == 1
        assert week_summaries[0].summary.session_count == 2
        assert any(item.product_kind == "archive_debt" for item in archive_debt)
        assert any(item.conversation_id == "conv-root" and item.estimate.status == "exact" for item in session_costs)
        assert cost_rollups[0].total_usd == pytest.approx(1.25)

    @pytest.mark.asyncio
    async def test_archive_stats_health_and_rebuild_products_are_public(
        self: object,
        cli_workspace: dict[str, Path],
    ) -> None:
        db_path = cli_workspace["db_path"]
        (
            ConversationBuilder(db_path, "conv-public")
            .provider("codex")
            .title("Public Facade")
            .add_message("u1", role="user", text="Verify facade methods")
            .save()
        )

        archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
        stats = await archive.get_archive_stats()
        health = await archive.health_check()
        counts = await archive.rebuild_products(["conv-public"])

        assert stats.conversation_count == 1
        assert health.summary
        assert counts.profiles == 1
        assert counts.total() >= 1


class TestPolylogueListConversations:
    @pytest.mark.parametrize(
        "setup_count,provider_filter,source_filter,limit,expected_count",
        LIST_CONV_FILTERS,
    )
    @pytest.mark.asyncio
    async def test_list_conversations_with_filters(
        self: object,
        tmp_path: Path,
        setup_count: int,
        provider_filter: str | None,
        source_filter: str | None,
        limit: int | None,
        expected_count: int,
    ) -> None:
        archive = _archive(tmp_path)
        repository = archive.repository

        for i in range(setup_count):
            provider = "claude-ai" if i < 2 else "chatgpt"
            await repository.save_conversation(
                make_conversation(
                    f"conv-{i}",
                    provider_name=provider,
                    provider_conversation_id=f"provider-{i}",
                    title=f"Conv {i}",
                    created_at="2025-01-01T00:00:00Z",
                    updated_at="2025-01-01T00:00:00Z",
                    content_hash=f"hash-{i}",
                    provider_meta={"source": source_filter or "inbox"} if source_filter else {},
                ),
                [],
                [],
            )

        convs = await archive.list_conversations(provider=provider_filter, limit=limit)
        assert len(convs) == expected_count


class TestPolylogueFilter:
    def test_filter_returns_conversation_filter(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        filter_builder = archive.filter()
        assert hasattr(filter_builder, "provider")

    def test_filter_chaining(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        assert archive.filter().provider("claude-ai") is not None


class TestPolylogueContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_enter_returns_self(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        async with archive as ctx:
            assert ctx is archive

    @pytest.mark.asyncio
    async def test_context_manager_exit_calls_close(self: object, tmp_path: Path) -> None:
        async with _archive(tmp_path):
            pass

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        with pytest.raises(ValueError, match="Test error"):
            async with archive:
                raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_close_method(self: object, tmp_path: Path) -> None:
        await _archive(tmp_path).close()


class TestPolylogueRebuildIndex:
    @pytest.mark.asyncio
    async def test_rebuild_index_lazy_init(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        assert callable(archive.rebuild_index)
        assert isinstance(await archive.rebuild_index(), bool)


class TestPolylogueStats:
    @pytest.mark.asyncio
    async def test_stats_empty_db(self: object, tmp_path: Path) -> None:
        stats = await _archive(tmp_path).stats()
        assert isinstance(stats, ArchiveStats)
        assert stats.conversation_count == 0
        assert stats.message_count == 0
        assert stats.word_count == 0
        assert stats.providers == {}
        assert stats.tags == {}

    @pytest.mark.asyncio
    async def test_stats_with_conversations(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        repository = archive.repository

        for i in range(2):
            await repository.save_conversation(
                make_conversation(
                    f"claude-{i}",
                    provider_name="claude-ai",
                    provider_conversation_id=f"p-{i}",
                    title=f"Claude Conv {i}",
                    created_at="2025-01-01T00:00:00Z",
                    updated_at="2025-01-01T00:00:00Z",
                    content_hash=f"h-{i}",
                ),
                [
                    make_message(
                        f"msg-{i}-0",
                        f"claude-{i}",
                        role="user",
                        text="Hello world",
                        timestamp="2025-01-01T00:00:00Z",
                        content_hash=f"mh-{i}-0",
                    ),
                    make_message(
                        f"msg-{i}-1",
                        f"claude-{i}",
                        role="assistant",
                        text="Hi there friend",
                        timestamp="2025-01-01T00:01:00Z",
                        content_hash=f"mh-{i}-1",
                    ),
                ],
                [],
            )

        await repository.save_conversation(
            make_conversation(
                "chatgpt-0",
                provider_name="chatgpt",
                provider_conversation_id="p-chatgpt-0",
                title="ChatGPT Conv",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash="h-chatgpt",
            ),
            [
                make_message(
                    "msg-chatgpt-0",
                    "chatgpt-0",
                    role="user",
                    text="Test",
                    timestamp="2025-01-01T00:00:00Z",
                    content_hash="mh-chatgpt",
                )
            ],
            [],
        )

        stats = await archive.stats()
        assert stats.conversation_count == 3
        assert stats.message_count == 5
        assert stats.providers["claude-ai"] == 2
        assert stats.providers["chatgpt"] == 1

    @pytest.mark.asyncio
    async def test_stats_recent_conversations(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        await archive.repository.save_conversation(
            make_conversation(
                "conv-1",
                provider_name="claude-ai",
                provider_conversation_id="p-1",
                title="Test Conv",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-15T12:00:00Z",
                content_hash="h-1",
            ),
            [],
            [],
        )

        stats = await archive.stats()
        assert len(stats.recent) == 1
        assert stats.recent[0].id == "conv-1"
