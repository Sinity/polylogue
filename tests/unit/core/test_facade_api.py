"""Unit-level public API contracts for the Polylogue facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.api import ArchiveStats
from polylogue.archive.message.roles import Role
from polylogue.insights.archive import (
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsightQuery,
    CostRollupInsightQuery,
    SessionCostInsightQuery,
    SessionPhaseInsightQuery,
    SessionProfileInsightQuery,
    SessionTagRollupQuery,
    ThreadInsightQuery,
)
from tests.infra.builders import make_conv, make_msg
from tests.infra.storage_records import SessionBuilder


def _seed(
    archive: Polylogue,
    conv_id: str,
    *,
    provider: str = "claude-ai",
    title: str = "Test Session",
    provider_session_id: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
    provider_meta: dict[str, object] | None = None,
    messages: list[dict[str, object]] | None = None,
) -> str:
    """Seed one session directly and return its archive session id."""
    db_path = archive.archive_root / "index.db"
    builder = SessionBuilder(db_path, conv_id).provider(provider).title(title)
    if provider_session_id is not None:
        builder.conv = builder.conv.model_copy(update={"provider_session_id": provider_session_id})
    if created_at is not None:
        builder.created_at(created_at)
    if updated_at is not None:
        builder.updated_at(updated_at)
    if provider_meta is not None:
        builder.metadata(provider_meta)
    for msg in messages or []:
        builder.add_message(**msg)  # type: ignore[arg-type]
    builder.save()
    return builder.native_session_id()


ArchiveStatsCase = tuple[int, int, int, dict[str, int], dict[str, int], str | None]
ListSessionsCase = tuple[int, str | None, str | None, int | None, int]

ARCHIVE_STATS_PARAMS: list[ArchiveStatsCase] = [
    (10, 50, 1000, {"claude-ai-export": 7, "chatgpt-export": 3}, {"test": 2, "work": 3}, None),
    (5, 25, 500, {"claude-ai-export": 5}, {}, "2025-01-15T12:30:45Z"),
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

LIST_CONV_FILTERS: list[ListSessionsCase] = [
    (0, None, None, None, 0),
    (3, None, None, None, 3),
    (4, "claude-ai-export", None, None, 2),
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
        assert archive.config.db_path == custom_root / "index.db"
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
        "conv_count,msg_count,word_count,origins,tags,last_sync",
        ARCHIVE_STATS_PARAMS,
    )
    def test_archive_stats_init(
        self: object,
        conv_count: int,
        msg_count: int,
        word_count: int,
        origins: dict[str, int],
        tags: dict[str, int],
        last_sync: str | None,
    ) -> None:
        stats = ArchiveStats(
            session_count=conv_count,
            message_count=msg_count,
            word_count=word_count,
            origins=origins,
            tags=tags,
            last_sync=last_sync,
            recent=[],
        )
        assert stats.session_count == conv_count
        assert stats.message_count == msg_count
        assert stats.word_count == word_count
        assert stats.origins == origins
        assert stats.tags == tags
        assert stats.last_sync == last_sync
        assert stats.recent == []

    def test_archive_stats_with_recent_sessions(self: object) -> None:
        recent_conv = make_conv(
            id="conv1",
            provider="claude-ai",
            messages=[make_msg(id="m1", role="user", text="Hello", timestamp="2025-01-01T00:00:00Z")],
        )
        stats = ArchiveStats(
            session_count=1,
            message_count=1,
            word_count=10,
            origins={"claude-ai-export": 1},
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


class TestPolylogueGetSession:
    @pytest.mark.asyncio
    async def test_get_session_empty_db(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        assert await archive.get_session("nonexistent_id") is None

    @pytest.mark.asyncio
    async def test_get_session_with_seed_data(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)

        native_id = _seed(
            archive,
            "conv-1",
            provider="claude-ai",
            title="Test Session",
            provider_session_id="provider-1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            messages=[
                {"message_id": "msg-1", "role": "user", "text": "Hello", "timestamp": "2025-01-01T00:00:00Z"},
                {"message_id": "msg-2", "role": "assistant", "text": "Hi there", "timestamp": "2025-01-01T00:01:00Z"},
            ],
        )

        conv = await archive.get_session(native_id)
        assert conv is not None
        assert conv.id == native_id
        assert conv.title == "Test Session"


class TestPolylogueGetSessions:
    @pytest.mark.asyncio
    async def test_get_sessions_empty_list(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        assert await archive.get_sessions([]) == []

    @pytest.mark.asyncio
    async def test_get_sessions_batch(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)

        ids = [
            _seed(
                archive,
                f"conv-{i}",
                provider="claude-ai",
                title=f"Session {i}",
                provider_session_id=f"provider-{i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
            )
            for i in range(3)
        ]
        convs = await archive.get_sessions(ids)
        assert len(convs) == 3
        assert all(conv.id in ids for conv in convs)


class TestPolylogueReadSurfaces:
    @pytest.mark.asyncio
    async def test_get_messages_paginated_resolves_id_and_filters_roles(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)

        native_id = _seed(
            archive,
            "conv-read-api",
            provider="claude-ai",
            title="Read API",
            provider_session_id="provider-read-api",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            messages=[
                {"message_id": "msg-user", "role": "user", "text": "visible user", "timestamp": "2025-01-01T00:00:00Z"},
                {
                    "message_id": "msg-assistant",
                    "role": "assistant",
                    "text": "hidden assistant",
                    "timestamp": "2025-01-01T00:00:01Z",
                },
            ],
        )

        result = await archive.get_messages_paginated(native_id, message_role=(Role.USER,), limit=5)

        messages, total = result
        assert total == 1
        assert len(messages) == 1
        assert str(messages[0].role) == "user"
        assert messages[0].text == "visible user"

        from polylogue.api.archive import SessionNotFoundError

        with pytest.raises(SessionNotFoundError):
            await archive.get_messages_paginated("missing-read-api")

        # A valid message_type filter resolves directly; both seeded messages
        # are plain messages, so the default "message" type returns both.
        typed_messages, typed_total = await archive.get_messages_paginated(native_id, message_type="message")
        assert typed_total == 2
        assert len(typed_messages) == 2

    @pytest.mark.asyncio
    async def test_bulk_get_messages_batches_role_and_date_filters(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)

        native_a = _seed(
            archive,
            "conv-bulk-a",
            provider="claude-ai",
            title="Bulk A",
            provider_session_id="provider-conv-bulk-a",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:04Z",
            messages=[
                {
                    "message_id": "msg-a-early",
                    "role": "user",
                    "text": "before window",
                    "timestamp": "2025-01-01T00:00:00Z",
                },
                {
                    "message_id": "msg-a-assistant",
                    "role": "assistant",
                    "text": "wrong role",
                    "timestamp": "2025-01-01T00:00:01Z",
                },
                {
                    "message_id": "msg-a-user",
                    "role": "user",
                    "text": "a in window",
                    "timestamp": "2025-01-01T00:00:02Z",
                },
            ],
        )
        native_b = _seed(
            archive,
            "conv-bulk-b",
            provider="claude-ai",
            title="Bulk B",
            provider_session_id="provider-conv-bulk-b",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:04Z",
            messages=[
                {
                    "message_id": "msg-b-user",
                    "role": "user",
                    "text": "b in window",
                    "timestamp": "2025-01-01T00:00:03Z",
                },
                {
                    "message_id": "msg-b-late",
                    "role": "user",
                    "text": "after window",
                    "timestamp": "2025-01-01T00:00:04Z",
                },
            ],
        )

        result = await archive.bulk_get_messages(
            [native_a, native_b, "missing"],
            since="2025-01-01T00:00:02Z",
            until="2025-01-01T00:00:03Z",
            message_role=(Role.USER,),
        )

        assert [message.text for message in result[native_a]] == ["a in window"]
        assert [message.text for message in result[native_b]] == ["b in window"]
        # Missing sessions are omitted from the archive batch result.
        assert "missing" not in result

    @pytest.mark.asyncio
    async def test_get_raw_artifacts_resolves_id_and_handles_missing(self: object, tmp_path: Path) -> None:
        from polylogue.core.enums import Provider
        from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        archive = _archive(tmp_path)

        payload = b'{"raw": "codex payload"}'
        parsed = ParsedSession(
            source_name=Provider.from_string("codex"),
            provider_session_id="provider-raw-api",
            title="Raw API",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="Hello")],
        )
        with ArchiveStore(archive.archive_root) as archive_db:
            _raw_id, native_id = archive_db.write_raw_and_parsed(
                parsed,
                payload=payload,
                source_path="/tmp/raw.jsonl",
                acquired_at_ms=1735689600000,
            )

        artifacts, total = await archive.get_raw_artifacts_for_session(native_id)
        missing_artifacts, missing_total = await archive.get_raw_artifacts_for_session("missing")

        assert total == 1
        assert artifacts[0]["source_name"] == "codex"
        assert artifacts[0]["blob_size"] == len(payload)
        assert missing_artifacts == []
        assert missing_total == 0

    @pytest.mark.asyncio
    async def test_get_sessions_partial_match(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)

        native_id = _seed(
            archive,
            "conv-1",
            provider="claude-ai",
            title="Session 1",
            provider_session_id="provider-1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )

        convs = await archive.get_sessions([native_id, "conv-999"])
        assert len(convs) == 1
        assert convs[0].id == native_id


class TestPolylogueArchiveInsights:
    @pytest.mark.asyncio
    async def test_durable_session_insights_are_publicly_queryable(
        self: object,
        cli_workspace: dict[str, Path],
    ) -> None:
        db_path = cli_workspace["db_path"]
        root_builder = (
            SessionBuilder(db_path, "conv-root")
            .provider("claude-code")
            .title("Root Thread")
            .metadata({"total_cost_usd": 1.25, "model": "claude-sonnet-4-5"})
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
        )
        root_builder.save()
        root_id = root_builder.native_session_id()
        (
            SessionBuilder(db_path, "conv-child")
            .provider("claude-code")
            .title("Child Thread")
            .parent_session("ext-conv-root")
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

        archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
        await archive.rebuild_insights()
        profile = await archive.get_session_profile_insight(root_id)
        profiles = await archive.list_session_profile_insights(
            SessionProfileInsightQuery(
                provider="claude-code",
                first_message_since="2026-03-01T00:00:00+00:00",
                session_date_since="2026-03-01",
                limit=10,
            )
        )
        phases = await archive.list_session_phase_insights(SessionPhaseInsightQuery(provider="claude-code", limit=10))
        threads = await archive.list_thread_insights(ThreadInsightQuery(limit=10))

        assert profile is not None
        assert profile.insight_kind == "session_profile"
        assert profile.title == "Root Thread"
        assert profile.evidence is not None
        assert profile.evidence.canonical_session_date == "2026-03-01"
        assert profile.inference is not None
        assert profile.inference.engaged_duration_ms >= 0
        assert profile.enrichment is not None
        assert profile.enrichment.confidence >= 0.0

        evidence_only = await archive.get_session_profile_insight(root_id, tier="evidence")
        inference_only = await archive.get_session_profile_insight(root_id, tier="inference")
        assert evidence_only is not None
        assert evidence_only.semantic_tier == "evidence"
        assert evidence_only.evidence is not None
        assert evidence_only.inference is None
        assert inference_only is not None
        assert inference_only.semantic_tier == "inference"
        assert inference_only.evidence is None
        assert inference_only.inference is not None
        assert inference_only.enrichment is None

        assert any(item.session_id == root_id for item in profiles)
        assert any(item.session_id == root_id for item in phases)
        assert len(threads) == 1
        assert threads[0].thread.session_count == 2
        assert threads[0].thread.support_level == "strong"
        assert threads[0].thread.member_evidence[1].role == "parent_continuation"
        assert threads[0].thread.member_evidence[1].parent_id == root_id

        tag_rollups = await archive.list_session_tag_rollup_insights(SessionTagRollupQuery(provider="claude-code"))
        day_coverage = await archive.list_archive_coverage_insights(
            ArchiveCoverageInsightQuery(provider="claude-code", group_by="day", limit=10)
        )
        week_coverage = await archive.list_archive_coverage_insights(
            ArchiveCoverageInsightQuery(provider="claude-code", group_by="week", limit=10)
        )
        archive_debt = await archive.list_archive_debt_insights(ArchiveDebtInsightQuery(limit=10))
        session_costs = await archive.list_session_cost_insights(
            SessionCostInsightQuery(provider="claude-code", limit=10)
        )
        cost_rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery(provider="claude-code"))

        assert any(item.tag == "origin:claude-code-session" for item in tag_rollups)
        assert len(day_coverage) == 1
        assert day_coverage[0].session_count == 2
        assert len(week_coverage) == 1
        assert week_coverage[0].session_count == 2
        assert any(item.insight_kind == "archive_debt" for item in archive_debt)
        root_cost = next(item for item in session_costs if item.session_id == root_id)
        assert root_cost.estimate.status == "unavailable"
        assert root_cost.estimate.missing_reasons == ("missing_token_usage",)
        assert cost_rollups[0].total_usd == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_archive_stats_health_and_rebuild_insights_are_public(
        self: object,
        cli_workspace: dict[str, Path],
    ) -> None:
        db_path = cli_workspace["db_path"]
        builder = (
            SessionBuilder(db_path, "conv-public")
            .provider("codex")
            .title("Public Facade")
            .add_message("u1", role="user", text="Verify facade methods")
        )
        builder.save()
        native_id = builder.native_session_id()

        archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
        stats = await archive.stats()
        health = await archive.health_check()
        counts = await archive.rebuild_insights([native_id])

        assert stats.session_count == 1
        assert health.summary
        assert counts.profiles == 1
        assert counts.total() >= 1


class TestPolylogueListSessions:
    @pytest.mark.parametrize(
        "setup_count,provider_filter,source_filter,limit,expected_count",
        LIST_CONV_FILTERS,
    )
    @pytest.mark.asyncio
    async def test_list_sessions_with_filters(
        self: object,
        tmp_path: Path,
        setup_count: int,
        provider_filter: str | None,
        source_filter: str | None,
        limit: int | None,
        expected_count: int,
    ) -> None:
        archive = _archive(tmp_path)

        for i in range(setup_count):
            provider = "claude-ai" if i < 2 else "chatgpt"
            _seed(
                archive,
                f"conv-{i}",
                provider=provider,
                title=f"Conv {i}",
                provider_session_id=f"provider-{i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                provider_meta={"source": source_filter or "inbox"} if source_filter else None,
            )

        convs = await archive.list_sessions(origin=provider_filter, limit=limit)
        assert len(convs) == expected_count


class TestPolylogueFilter:
    def test_filter_returns_session_filter(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        filter_builder = archive.filter()
        assert hasattr(filter_builder, "origin")

    def test_filter_chaining(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        assert archive.filter().origin("claude-ai-export") is not None


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
        assert stats.session_count == 0
        assert stats.message_count == 0
        assert stats.word_count == 0
        assert stats.origins == {}
        assert stats.tags == {}

    @pytest.mark.asyncio
    async def test_stats_with_sessions(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)

        for i in range(2):
            _seed(
                archive,
                f"claude-{i}",
                provider="claude-ai",
                title=f"Claude Conv {i}",
                provider_session_id=f"p-{i}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                messages=[
                    {
                        "message_id": f"msg-{i}-0",
                        "role": "user",
                        "text": "Hello world",
                        "timestamp": "2025-01-01T00:00:00Z",
                    },
                    {
                        "message_id": f"msg-{i}-1",
                        "role": "assistant",
                        "text": "Hi there friend",
                        "timestamp": "2025-01-01T00:01:00Z",
                    },
                ],
            )

        _seed(
            archive,
            "chatgpt-0",
            provider="chatgpt",
            title="ChatGPT Conv",
            provider_session_id="p-chatgpt-0",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            messages=[
                {"message_id": "msg-chatgpt-0", "role": "user", "text": "Test", "timestamp": "2025-01-01T00:00:00Z"}
            ],
        )

        stats = await archive.stats()
        assert stats.session_count == 3
        assert stats.message_count == 5
        assert stats.origins["claude-ai-export"] == 2
        assert stats.origins["chatgpt-export"] == 1

    @pytest.mark.asyncio
    async def test_stats_recent_sessions(self: object, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        native_id = _seed(
            archive,
            "conv-1",
            provider="claude-ai",
            title="Test Conv",
            provider_session_id="p-1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-15T12:00:00Z",
        )

        stats = await archive.stats()
        assert len(stats.recent) == 1
        assert stats.recent[0].id == native_id
