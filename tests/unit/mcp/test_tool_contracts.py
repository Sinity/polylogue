"""Unit contracts for MCP tool surfaces backed by repository/config mocks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.conversation.neighbor_candidates import ConversationNeighborCandidate, NeighborReason
from polylogue.archive.query.miss_diagnostics import QueryMissDiagnostics, QueryMissReason
from polylogue.archive.query.search_hits import ConversationSearchHit
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.archive.semantic.pricing import CostEstimatePayload, CostUsagePayload
from polylogue.archive.stats import ArchiveStats
from polylogue.lib.models import Conversation, ConversationSummary
from polylogue.products.archive import (
    ArchiveDebtProduct,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveProductProvenance,
    CostRollupProduct,
    DaySessionSummaryProduct,
    ProviderAnalyticsProduct,
    SessionCostProduct,
    SessionEnrichmentPayload,
    SessionEnrichmentProduct,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    SessionPhaseProduct,
    SessionProfileProduct,
    SessionTagRollupProduct,
    SessionWorkEventProduct,
    WeekSessionSummaryProduct,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
    WorkThreadProduct,
)
from polylogue.products.archive_models import (
    DaySessionSummaryPayload,
    WeekSessionSummaryPayload,
    WorkThreadMemberEvidencePayload,
    WorkThreadPayload,
)
from polylogue.storage.products.session.runtime import SessionProductCounts
from polylogue.types import ConversationId, Provider
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import (
    MCPServerUnderTest,
    invoke_surface,
    invoke_surface_async,
    make_query_store_mock,
    make_simple_conversation,
    make_tag_store_mock,
)

STATS_CONFIGS = [
    (100, 5000, {"claude-ai": 50, "chatgpt": 30, "claude-code": 20}, 10, 200, 90, 1048576, 10.0, 1.0),
    (0, 0, {}, 0, 0, 0, 0, 0.0, 0),
    (5, 20, {"test": 5}, 0, 0, 5, 0, 0.0, 0),
]

QUERY_TOOL_CASES = [
    (
        "search",
        {"query": "hello", "limit": 10},
        {
            "contains": ("hello",),
            "limit": (10,),
        },
    ),
    (
        "search",
        {"query": "hello", "provider": "claude-ai", "since": "2024-01-01", "limit": 5},
        {
            "contains": ("hello",),
            "provider": ("claude-ai",),
            "since": ("2024-01-01",),
            "limit": (5,),
        },
    ),
    (
        "search",
        {"query": "hello", "referenced_path": "/workspace/polylogue/README.md", "limit": 5},
        {
            "contains": ("hello",),
            "referenced_path": ("/workspace/polylogue/README.md",),
            "limit": (5,),
        },
    ),
    (
        "search",
        {"query": "hello", "retrieval_lane": "actions", "limit": 5},
        {
            "contains": ("hello",),
            "retrieval_lane": ("actions",),
            "limit": (5,),
        },
    ),
    (
        "search",
        {
            "query": "hello",
            "action": "search",
            "exclude_action": "git",
            "tool": "grep",
            "exclude_tool": "bash",
            "limit": 5,
        },
        {
            "contains": ("hello",),
            "action": ("search",),
            "exclude_action": ("git",),
            "tool": ("grep",),
            "exclude_tool": ("bash",),
            "limit": (5,),
        },
    ),
    (
        "search",
        {"query": "hello", "action_sequence": "file_read,file_edit,shell", "limit": 5},
        {
            "contains": ("hello",),
            "action_sequence": (("file_read", "file_edit", "shell"),),
            "limit": (5,),
        },
    ),
    (
        "search",
        {"query": "hello", "action_text": "pytest -q", "limit": 5},
        {
            "contains": ("hello",),
            "action_text": ("pytest -q",),
            "limit": (5,),
        },
    ),
    (
        "list_conversations",
        {"limit": 10},
        {
            "limit": (10,),
        },
    ),
    (
        "list_conversations",
        {"retrieval_lane": "hybrid", "limit": 2},
        {
            "retrieval_lane": ("hybrid",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"provider": "claude-ai", "since": "2024-01-01", "tag": "bug", "title": "incident", "limit": 2},
        {
            "provider": ("claude-ai",),
            "since": ("2024-01-01",),
            "tag": ("bug",),
            "title": ("incident",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"referenced_path": "/workspace/polylogue/README.md", "limit": 2},
        {
            "referenced_path": ("/workspace/polylogue/README.md",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"action": "file_edit", "exclude_action": "web", "tool": "edit", "exclude_tool": "read", "limit": 2},
        {
            "action": ("file_edit",),
            "exclude_action": ("web",),
            "tool": ("edit",),
            "exclude_tool": ("read",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"action_sequence": "file_read,file_edit,shell", "limit": 2},
        {
            "action_sequence": (("file_read", "file_edit", "shell"),),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"action_text": "pytest -q", "limit": 2},
        {
            "action_text": ("pytest -q",),
            "limit": (2,),
        },
    ),
    (
        "list_conversations",
        {"action": "none", "limit": 2},
        {
            "action": ("none",),
            "limit": (2,),
        },
    ),
]


@pytest.fixture
def simple_conversation() -> Conversation:
    return make_simple_conversation()


def _make_summary(
    conversation_id: str = "test:conv-123",
    *,
    provider: Provider = Provider.CHATGPT,
    title: str | None = "Test Conv",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> ConversationSummary:
    return ConversationSummary(
        id=ConversationId(conversation_id),
        provider=provider,
        title=title,
        message_count=None,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_search_hit(conversation: Conversation) -> ConversationSearchHit:
    messages = conversation.messages.to_list()
    return ConversationSearchHit(
        summary=ConversationSummary(
            id=ConversationId(str(conversation.id)),
            provider=Provider.from_string(str(conversation.provider)),
            title=conversation.display_title,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        ),
        rank=1,
        retrieval_lane="dialogue",
        match_surface="message",
        message_id=str(messages[0].id) if messages else None,
        snippet="hello evidence",
    )


def _make_attachment_search_hit(conversation: Conversation) -> ConversationSearchHit:
    return ConversationSearchHit(
        summary=ConversationSummary(
            id=ConversationId(str(conversation.id)),
            provider=Provider.from_string(str(conversation.provider)),
            title=conversation.display_title,
            message_count=len(conversation.messages),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        ),
        rank=1,
        retrieval_lane="attachment",
        match_surface="attachment",
        message_id="msg-doc",
        snippet='attachment identity provider_meta.fileId=drive-file-1 name="Project Plan"',
    )


def _make_neighbor_candidate() -> ConversationNeighborCandidate:
    return ConversationNeighborCandidate(
        summary=ConversationSummary(
            id=ConversationId("candidate"),
            provider=Provider.CODEX,
            title="Archive Lock Retries",
            message_count=2,
            updated_at=datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc),
        ),
        rank=1,
        score=3.25,
        reasons=(
            NeighborReason(
                kind="nearby_time",
                detail="within 2.0h of source conversation",
                evidence="source=2026-04-22T12:00:00+00:00 candidate=2026-04-22T14:00:00+00:00",
                weight=1.0,
            ),
        ),
        source_conversation_id="target",
    )


def _provenance() -> ArchiveProductProvenance:
    return ArchiveProductProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
    )


def _inference_provenance() -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
        inference_version=1,
        inference_family="heuristic_session_semantics",
    )


def _enrichment_provenance() -> ArchiveEnrichmentProvenance:
    return ArchiveEnrichmentProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
        enrichment_version=1,
        enrichment_family="scored_session_enrichment",
    )


class TestQueryTools:
    @pytest.mark.parametrize(("tool_name", "args", "expected_calls"), QUERY_TOOL_CASES)
    @pytest.mark.asyncio
    async def test_query_tool_filter_contract(
        self,
        simple_conversation: Conversation,
        tool_name: str,
        args: dict[str, object],
        expected_calls: dict[str, tuple[object, ...]],
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.query_conversations = AsyncMock(return_value=[simple_conversation])
            mock_ops.search_conversation_hits = AsyncMock(return_value=[_make_search_hit(simple_conversation)])
            mock_get_archive_ops.return_value = mock_ops

            raw = await invoke_surface_async(mcp_server._tool_manager._tools[tool_name].fn, **args)

        payload = json.loads(raw)
        assert isinstance(payload, list)
        assert len(payload) == 1
        if tool_name == "search":
            assert payload[0]["conversation"]["id"] == str(simple_conversation.id)
            assert payload[0]["match"]["message_id"] == str(simple_conversation.messages.to_list()[0].id)
            mock_ops.search_conversation_hits.assert_awaited_once()
            mock_ops.query_conversations.assert_not_called()
            spec = mock_ops.search_conversation_hits.await_args.args[0]
        else:
            assert payload[0]["id"] == simple_conversation.id
            mock_ops.query_conversations.assert_awaited_once()
            mock_ops.search_conversation_hits.assert_not_called()
            spec = mock_ops.query_conversations.await_args.args[0]
        assert isinstance(spec, ConversationQuerySpec)
        for method_name, method_args in expected_calls.items():
            expected_value = method_args[0] if len(method_args) == 1 else method_args
            if method_name == "contains":
                assert spec.query_terms == (expected_value,)
            elif method_name == "provider":
                assert tuple(str(provider) for provider in spec.providers) == (expected_value,)
            elif method_name == "since":
                assert spec.since == expected_value
            elif method_name == "retrieval_lane":
                assert spec.retrieval_lane == expected_value
            elif method_name == "referenced_path":
                assert spec.referenced_path == (expected_value,)
            elif method_name == "action":
                assert spec.action_terms == (expected_value,)
            elif method_name == "exclude_action":
                assert spec.excluded_action_terms == (expected_value,)
            elif method_name == "tool":
                assert spec.tool_terms == (expected_value,)
            elif method_name == "exclude_tool":
                assert spec.excluded_tool_terms == (expected_value,)
            elif method_name == "action_sequence":
                assert spec.action_sequence == expected_value
            elif method_name == "action_text":
                assert spec.action_text_terms == (expected_value,)
            elif method_name == "tag":
                assert spec.tags == (expected_value,)
            elif method_name == "title":
                assert spec.title == expected_value
            elif method_name == "limit":
                assert spec.limit == expected_value

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, mcp_server: MCPServerUnderTest) -> None:
        diagnostics = QueryMissDiagnostics(
            message="No conversations matched.",
            filters=(),
            reasons=(
                QueryMissReason(
                    code="archive_empty",
                    severity="info",
                    summary="The selected archive scope has no materialized conversations.",
                    count=0,
                ),
            ),
            archive_conversation_count=0,
            raw_conversation_count=0,
        )
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.search_conversation_hits = AsyncMock(return_value=[])
            mock_ops.diagnose_query_miss = AsyncMock(return_value=diagnostics)
            mock_get_archive_ops.return_value = mock_ops

            result = await invoke_surface_async(mcp_server._tool_manager._tools["search"].fn, query="", limit=10)

        parsed = json.loads(result)
        assert parsed["results"] == []
        assert parsed["diagnostics"]["archive_conversation_count"] == 0
        assert parsed["diagnostics"]["reasons"][0]["code"] == "archive_empty"
        mock_ops.diagnose_query_miss.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_conversations_uses_provider_display_label(self, mcp_server: MCPServerUnderTest) -> None:
        gemini_conversation = make_conv(
            id="gemini:gemini-20250422-1234",
            provider=Provider.GEMINI,
            title="gemini-20250422-1234",
            provider_meta={"display_label": "Project Plan: Please review the attached project plan."},
            messages=[make_msg(id="msg-user", role="user", text="Please review the attached project plan.")],
        )
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.query_conversations = AsyncMock(return_value=[gemini_conversation])
            mock_get_archive_ops.return_value = mock_ops

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["list_conversations"].fn,
                provider="gemini",
                limit=10,
            )

        payload = json.loads(raw)
        assert payload[0]["title"] == "Project Plan: Please review the attached project plan."

    @pytest.mark.asyncio
    async def test_search_exposes_attachment_identity_evidence(
        self,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.search_conversation_hits = AsyncMock(
                return_value=[_make_attachment_search_hit(simple_conversation)]
            )
            mock_ops.diagnose_query_miss = AsyncMock()
            mock_get_archive_ops.return_value = mock_ops

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="drive-file-1",
                limit=10,
            )

        payload = json.loads(raw)
        assert payload[0]["match"]["match_surface"] == "attachment"
        assert payload[0]["match"]["retrieval_lane"] == "attachment"
        assert payload[0]["match"]["message_id"] == "msg-doc"
        assert "provider_meta.fileId=drive-file-1" in payload[0]["match"]["snippet"]
        mock_ops.search_conversation_hits.assert_awaited_once()
        mock_ops.diagnose_query_miss.assert_not_called()

    @pytest.mark.asyncio
    async def test_neighbor_candidates_exposes_candidate_reasons(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.neighbor_candidates = AsyncMock(return_value=[_make_neighbor_candidate()])
            mock_get_archive_ops.return_value = mock_ops

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["neighbor_candidates"].fn,
                id="target",
                limit=3,
            )

        payload = json.loads(raw)
        assert payload[0]["conversation"]["id"] == "candidate"
        assert payload[0]["reasons"][0]["kind"] == "nearby_time"
        assert payload[0]["source_conversation_id"] == "target"
        mock_ops.neighbor_candidates.assert_awaited_once_with(
            conversation_id="target",
            query=None,
            provider=None,
            limit=3,
            window_hours=24,
        )


class TestGetConversationTool:
    def test_get_returns_conversation(self, simple_conversation: Conversation, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(
                return_value=_make_summary(
                    str(simple_conversation.id),
                    provider=Provider.from_string(str(simple_conversation.provider)),
                    title=simple_conversation.display_title,
                )
            )
            mock_ops.get_conversation_stats = AsyncMock(
                return_value={"total_messages": len(simple_conversation.messages)}
            )
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation"].fn, id="test:conv-123")

        conv = json.loads(result)
        assert conv["id"] == "test:conv-123"
        assert conv["message_count"] == 2
        assert "messages" not in conv

    def test_get_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(return_value=None)
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation"].fn, id="nonexistent")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_get_messages_returns_full_messages(self, mcp_server: MCPServerUnderTest) -> None:
        long_text = "A" * 2000
        message = make_msg(id="m1", role="assistant", text=long_text)

        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(return_value=_make_summary("test:long"))
            mock_ops.get_messages_paginated = AsyncMock(return_value=([message], 1))
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["get_messages"].fn, conversation_id="test:long")

        assert json.loads(result)["messages"][0]["text"] == long_text

    @pytest.mark.asyncio
    async def test_get_with_nonexistent_id(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(return_value=None)
            mock_get_archive_ops.return_value = mock_ops

            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["get_conversation"].fn, id="nonexistent-id-xyz"
            )

        assert isinstance(json.loads(result), dict)


class TestProductTools:
    @pytest.mark.asyncio
    async def test_session_profile_tool_uses_archive_product_contract(self, mcp_server: MCPServerUnderTest) -> None:
        product = SessionProfileProduct(
            conversation_id="conv-1",
            provider_name="claude-code",
            title="Profiled Session",
            semantic_tier="merged",
            provenance=_provenance(),
            evidence=SessionEvidencePayload(canonical_session_date="2026-03-24", message_count=2),
            inference_provenance=_inference_provenance(),
            inference=SessionInferencePayload(engaged_duration_ms=120000),
        )
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_session_profile_product = AsyncMock(return_value=product)
            mock_get_archive_ops.return_value = mock_ops

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_profile"].fn,
                conversation_id="conv-1",
            )

        payload = json.loads(raw)
        assert payload["product_kind"] == "session_profile"
        assert payload["conversation_id"] == "conv-1"

    @pytest.mark.asyncio
    async def test_product_list_tools_use_archive_queries(self, mcp_server: MCPServerUnderTest) -> None:
        profile = SessionProfileProduct(
            conversation_id="conv-1",
            provider_name="claude-code",
            title="Profiled Session",
            semantic_tier="merged",
            provenance=_provenance(),
            evidence=SessionEvidencePayload(canonical_session_date="2026-03-24", message_count=2),
            inference_provenance=_inference_provenance(),
            inference=SessionInferencePayload(engaged_duration_ms=120000),
        )
        enrichment = SessionEnrichmentProduct(
            conversation_id="conv-1",
            provider_name="claude-code",
            title="Profiled Session",
            provenance=_provenance(),
            enrichment_provenance=_enrichment_provenance(),
            enrichment=SessionEnrichmentPayload(
                intent_summary="Plan the refactor",
                outcome_summary="Ran tests",
                confidence=0.72,
                support_level="moderate",
            ),
        )
        work_event = SessionWorkEventProduct(
            event_id="evt-1",
            conversation_id="conv-1",
            provider_name="claude-code",
            event_index=0,
            provenance=_provenance(),
            inference_provenance=_inference_provenance(),
            evidence=WorkEventEvidencePayload(
                start_index=0, end_index=1, file_paths=("/workspace/polylogue/README.md",)
            ),
            inference=WorkEventInferencePayload(kind="implementation", summary="editing files", confidence=0.8),
        )
        phase = SessionPhaseProduct(
            phase_id="phase-1",
            conversation_id="conv-1",
            provider_name="claude-code",
            phase_index=0,
            provenance=_provenance(),
            inference_provenance=_inference_provenance(),
            evidence=SessionPhaseEvidencePayload(message_range=(0, 2), tool_counts={"edit": 1}),
            inference=SessionPhaseInferencePayload(confidence=0.8),
        )
        thread = WorkThreadProduct(
            thread_id="conv-1",
            root_id="conv-1",
            dominant_repo="polylogue",
            provenance=_provenance(),
            thread=WorkThreadPayload(
                session_ids=("conv-1", "conv-2"),
                session_count=2,
                confidence=1.0,
                support_level="strong",
                support_signals=("explicit_lineage",),
                member_evidence=(
                    WorkThreadMemberEvidencePayload(
                        conversation_id="conv-1",
                        role="root",
                        depth=0,
                        confidence=1.0,
                        support_signals=("root_conversation",),
                        evidence=("conversation has no archived parent inside this thread",),
                    ),
                ),
            ),
        )
        tag_rollup = SessionTagRollupProduct(
            tag="provider:claude-code",
            conversation_count=1,
            explicit_count=0,
            auto_count=1,
            provider_breakdown={"claude-code": 1},
            repo_breakdown={"polylogue": 1},
            provenance=_provenance(),
        )
        day_summary = DaySessionSummaryProduct(
            date="2026-03-24",
            provenance=_provenance(),
            summary=DaySessionSummaryPayload(date="2026-03-24", session_count=1, total_messages=2),
        )
        week_summary = WeekSessionSummaryProduct(
            iso_week="2026-W13",
            provenance=_provenance(),
            summary=WeekSessionSummaryPayload(iso_week="2026-W13", session_count=1, total_messages=2),
        )
        analytics = ProviderAnalyticsProduct(
            provider_name="claude-code",
            conversation_count=1,
            message_count=2,
            user_message_count=1,
            assistant_message_count=1,
            avg_messages_per_conversation=2.0,
            avg_user_words=3.0,
            avg_assistant_words=4.0,
            tool_use_count=1,
            thinking_count=0,
            total_conversations_with_tools=1,
            total_conversations_with_thinking=0,
            tool_use_percentage=100.0,
            thinking_percentage=0.0,
        )
        session_cost = SessionCostProduct(
            conversation_id="conv-root",
            provider_name="claude-code",
            title="Root Thread",
            estimate=CostEstimatePayload(
                provider_name="claude-code",
                conversation_id="conv-root",
                model_name="claude-sonnet-4-5",
                normalized_model="claude-sonnet-4-5",
                status="exact",
                confidence=1.0,
                total_usd=1.25,
                provenance=("archive_provider_reported_cost",),
            ),
            provenance=_provenance(),
        )
        cost_rollup = CostRollupProduct(
            provider_name="claude-code",
            model_name="claude-sonnet-4-5",
            normalized_model="claude-sonnet-4-5",
            session_count=1,
            priced_session_count=1,
            unavailable_session_count=0,
            status_counts={"exact": 1},
            total_usd=1.25,
            usage=CostUsagePayload(),
            confidence=1.0,
            provenance=_provenance(),
        )
        debt = ArchiveDebtProduct(
            debt_name="session_products",
            category="products",
            maintenance_target="session_products",
            destructive=False,
            issue_count=1,
            healthy=False,
            detail="1 pending session-product row",
        )
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.list_session_profile_products = AsyncMock(return_value=[profile])
            mock_ops.list_session_enrichment_products = AsyncMock(return_value=[enrichment])
            mock_ops.list_session_work_event_products = AsyncMock(return_value=[work_event])
            mock_ops.list_session_phase_products = AsyncMock(return_value=[phase])
            mock_ops.list_session_tag_rollup_products = AsyncMock(return_value=[tag_rollup])
            mock_ops.list_work_thread_products = AsyncMock(return_value=[thread])
            mock_ops.list_day_session_summary_products = AsyncMock(return_value=[day_summary])
            mock_ops.list_week_session_summary_products = AsyncMock(return_value=[week_summary])
            mock_ops.list_provider_analytics_products = AsyncMock(return_value=[analytics])
            mock_ops.list_session_cost_products = AsyncMock(return_value=[session_cost])
            mock_ops.list_cost_rollup_products = AsyncMock(return_value=[cost_rollup])
            mock_ops.list_archive_debt_products = AsyncMock(return_value=[debt])
            mock_get_archive_ops.return_value = mock_ops

            profiles_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_profiles"].fn,
                provider="claude-code",
                query="profiled",
                first_message_since="2026-03-24T00:00:00+00:00",
                session_date_since="2026-03-24",
                limit=5,
                offset=2,
            )
            events_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_work_events"].fn,
                kind="implementation",
                limit=5,
            )
            enrichments_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_enrichments"].fn,
                provider="claude-code",
                limit=5,
            )
            phases_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_phases"].fn,
                kind="implementation",
                limit=5,
            )
            threads_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["work_threads"].fn,
                query="polylogue",
                limit=5,
            )
            tags_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_tag_rollups"].fn,
                provider="claude-code",
                limit=5,
            )
            day_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["day_session_summaries"].fn,
                provider="claude-code",
                limit=5,
            )
            week_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["week_session_summaries"].fn,
                provider="claude-code",
                limit=5,
            )
            analytics_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["provider_analytics"].fn,
                provider="claude-code",
                limit=5,
            )
            costs_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_costs"].fn,
                provider="claude-code",
                status="exact",
                limit=5,
            )
            cost_rollups_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["cost_rollups"].fn,
                provider="claude-code",
                model="claude-sonnet-4-5",
                limit=5,
            )
            debt_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["archive_debt"].fn,
                category="products",
                only_actionable=True,
                limit=5,
            )

        profiles_payload = json.loads(profiles_raw)
        events_payload = json.loads(events_raw)
        enrichments_payload = json.loads(enrichments_raw)
        phases_payload = json.loads(phases_raw)
        threads_payload = json.loads(threads_raw)
        tags_payload = json.loads(tags_raw)
        day_payload = json.loads(day_raw)
        week_payload = json.loads(week_raw)
        analytics_payload = json.loads(analytics_raw)
        costs_payload = json.loads(costs_raw)
        cost_rollups_payload = json.loads(cost_rollups_raw)
        debt_payload = json.loads(debt_raw)

        assert profiles_payload["count"] == 1
        assert profiles_payload["items"][0]["product_kind"] == "session_profile"
        assert enrichments_payload["items"][0]["product_kind"] == "session_enrichment"
        assert events_payload["items"][0]["product_kind"] == "session_work_event"
        assert phases_payload["items"][0]["product_kind"] == "session_phase"
        assert tags_payload["items"][0]["product_kind"] == "session_tag_rollup"
        assert threads_payload["items"][0]["product_kind"] == "work_thread"
        assert threads_payload["items"][0]["thread"]["support_level"] == "strong"
        assert threads_payload["items"][0]["thread"]["member_evidence"][0]["role"] == "root"
        assert day_payload["items"][0]["product_kind"] == "day_session_summary"
        assert week_payload["items"][0]["product_kind"] == "week_session_summary"
        assert analytics_payload["items"][0]["product_kind"] == "provider_analytics"
        assert costs_payload["items"][0]["product_kind"] == "session_cost"
        assert costs_payload["items"][0]["estimate"]["status"] == "exact"
        assert cost_rollups_payload["items"][0]["product_kind"] == "cost_rollup"
        assert cost_rollups_payload["items"][0]["total_usd"] == 1.25
        assert debt_payload["items"][0]["product_kind"] == "archive_debt"
        debt_query = mock_ops.list_archive_debt_products.await_args.args[0]
        assert debt_query.category == "products"
        assert debt_query.only_actionable is True

    @pytest.mark.asyncio
    async def test_session_enrichments_tool_rejects_unknown_query_fields(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.list_session_enrichment_products = AsyncMock(return_value=[])
            mock_get_archive_ops.return_value = mock_ops

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_enrichments"].fn,
                provider="claude-code",
                refined_work_kind="planning",
                limit=5,
            )

        payload = json.loads(raw)
        assert payload["tool"] == "session_enrichments"
        assert payload["error"] == "internal MCP tool error"
        assert payload["code"] == "internal_error"
        assert payload["detail"] == "ProductQueryError"
        mock_ops.list_session_enrichment_products.assert_not_awaited()


class TestStatsTool:
    @pytest.mark.parametrize(
        (
            "total_conversations",
            "total_messages",
            "providers",
            "embedded_convs",
            "embedded_msgs",
            "pending_convs",
            "db_size",
            "expected_coverage",
            "expected_mb",
        ),
        STATS_CONFIGS,
    )
    def test_stats_configurations(
        self,
        total_conversations: int,
        total_messages: int,
        providers: dict[str, int],
        embedded_convs: int,
        embedded_msgs: int,
        pending_convs: int,
        db_size: int,
        expected_coverage: float,
        expected_mb: float | int,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.storage_stats = AsyncMock(
                return_value=ArchiveStats(
                    total_conversations=total_conversations,
                    total_messages=total_messages,
                    providers=providers,
                    embedded_conversations=embedded_convs,
                    embedded_messages=embedded_msgs,
                    pending_embedding_conversations=pending_convs,
                    db_size_bytes=db_size,
                )
            )
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["stats"].fn)

        data = json.loads(result)
        assert data["total_conversations"] == total_conversations
        assert data["total_messages"] == total_messages
        assert data["pending_embedding_conversations"] == pending_convs
        assert data["embedding_coverage_percent"] == expected_coverage
        assert data["db_size_mb"] == expected_mb


class TestMutationTools:
    def test_add_tag_success(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store,
        ):
            mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.add_tag.return_value = None
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["add_tag"].fn, conversation_id="test:conv-123", tag="important"
            )

        parsed = json.loads(result)
        assert parsed == {"status": "ok", "conversation_id": "test:conv-123", "tag": "important"}
        mock_tag_store.add_tag.assert_called_once_with("test:conv-123", "important")

    def test_add_tag_error(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store,
        ):
            mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.add_tag.side_effect = ValueError("Invalid tag")
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["add_tag"].fn, conversation_id="test:conv-123", tag="invalid"
            )

        parsed = json.loads(result)
        assert parsed["error"] == "internal MCP tool error"
        assert parsed["code"] == "internal_error"
        assert parsed["detail"] == "ValueError"

    def test_remove_tag_success(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store,
        ):
            mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.remove_tag.return_value = None
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["remove_tag"].fn, conversation_id="test:conv-123", tag="important"
            )

        parsed = json.loads(result)
        assert parsed == {"status": "ok", "conversation_id": "test:conv-123", "tag": "important"}
        mock_tag_store.remove_tag.assert_called_once_with("test:conv-123", "important")

    def test_remove_tag_error(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store,
        ):
            mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.remove_tag.side_effect = RuntimeError("Backend error")
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["remove_tag"].fn, conversation_id="test:conv-123", tag="important"
            )

        assert "error" in json.loads(result)

    def test_bulk_tag_conversations_applies_every_tag_to_every_conversation(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        # The current bulk_tag implementation delegates to bulk_add_tags;
        # the previous N×M add_tag fan-out was replaced when the bulk method
        # landed. The applied_count is whatever bulk_add_tags returns.
        with patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store:
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.bulk_add_tags = AsyncMock(return_value=4)
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["bulk_tag_conversations"].fn,
                conversation_ids=["conv-1", "conv-2"],
                tags=["review", "important"],
            )

        parsed = json.loads(result)
        assert parsed == {
            "status": "ok",
            "conversation_count": 2,
            "tag_count": 2,
            "applied_count": 4,
        }
        mock_tag_store.bulk_add_tags.assert_called_once_with(["conv-1", "conv-2"], ["review", "important"])

    def test_bulk_tag_conversations_rejects_empty_inputs(self, mcp_server: MCPServerUnderTest) -> None:
        result = invoke_surface(
            mcp_server._tool_manager._tools["bulk_tag_conversations"].fn,
            conversation_ids=[],
            tags=["review"],
        )

        assert "requires at least one conversation_id" in json.loads(result)["error"]

    def test_list_tags_returns_counts(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store:
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.list_tags.return_value = {"bug": 3, "feature": 5, "urgent": 1}
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(mcp_server._tool_manager._tools["list_tags"].fn)

        assert json.loads(result) == {"bug": 3, "feature": 5, "urgent": 1}

    def test_list_tags_with_provider(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store:
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.list_tags.return_value = {"claude-ai": 2}
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(mcp_server._tool_manager._tools["list_tags"].fn, provider="claude-ai")

        assert json.loads(result) == {"claude-ai": 2}
        mock_tag_store.list_tags.assert_called_once_with(provider="claude-ai")

    def test_get_metadata_success(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store:
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.get_metadata.return_value = {"key": "value", "count": 42}
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(mcp_server._tool_manager._tools["get_metadata"].fn, conversation_id="test:conv-123")

        assert json.loads(result) == {"key": "value", "count": 42}

    def test_set_metadata_string_value(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store,
        ):
            mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.update_metadata.return_value = None
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["set_metadata"].fn,
                conversation_id="test:conv-123",
                key="author",
                value="john",
            )

        assert json.loads(result)["status"] == "ok"
        mock_tag_store.update_metadata.assert_called_once_with("test:conv-123", "author", "john")

    def test_set_metadata_json_value(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store,
        ):
            mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.update_metadata.return_value = None
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["set_metadata"].fn,
                conversation_id="test:conv-123",
                key="config",
                value='{"nested": true}',
            )

        assert json.loads(result)["status"] == "ok"
        mock_tag_store.update_metadata.assert_called_once_with("test:conv-123", "config", {"nested": True})

    def test_delete_metadata_success(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_query_store") as mock_get_query_store,
            patch("polylogue.mcp.server._get_tag_store") as mock_get_tag_store,
        ):
            mock_get_query_store.return_value = make_query_store_mock(resolved_id="test:conv-123")
            mock_tag_store = make_tag_store_mock()
            mock_tag_store.delete_metadata.return_value = None
            mock_get_tag_store.return_value = mock_tag_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_metadata"].fn,
                conversation_id="test:conv-123",
                key="author",
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["key"] == "author"
        mock_tag_store.delete_metadata.assert_called_once_with("test:conv-123", "author")

    def test_delete_requires_confirm(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_query_store = make_query_store_mock()
            mock_get_query_store.return_value = mock_query_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_conversation"].fn,
                conversation_id="test:conv-123",
                confirm=False,
            )

        parsed = json.loads(result)
        assert "confirm=true" in parsed["error"]
        mock_query_store.delete_conversation.assert_not_called()

    def test_delete_with_confirm(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_query_store = make_query_store_mock(resolved_id="test:conv-123")
            mock_query_store.delete_conversation.return_value = True
            mock_get_query_store.return_value = mock_query_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_conversation"].fn,
                conversation_id="test:conv-123",
                confirm=True,
            )

        assert json.loads(result)["status"] == "deleted"
        mock_query_store.delete_conversation.assert_called_once_with("test:conv-123")

    def test_delete_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        # Two not-found shapes: resolve_id returns None (id never existed) vs.
        # resolve_id succeeds but delete_conversation returns False (race).
        # The current contract returns "conversation not found" error for the
        # former and {status: "not_found"} for the latter; exercise the latter.
        with patch("polylogue.mcp.server._get_query_store") as mock_get_query_store:
            mock_query_store = make_query_store_mock(resolved_id="nonexistent")
            mock_query_store.delete_conversation.return_value = False
            mock_get_query_store.return_value = mock_query_store

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_conversation"].fn,
                conversation_id="nonexistent",
                confirm=True,
            )

        assert json.loads(result)["status"] == "not_found"

    def test_summary_returns_metadata(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_summary = _make_summary(
                "test:conv-123",
                provider=Provider.CHATGPT,
                title="Test Conv",
                created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            )
            mock_ops.get_conversation_summary = AsyncMock(return_value=mock_summary)
            mock_ops.get_conversation_stats = AsyncMock(return_value={"total_messages": 5})
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation_summary"].fn, id="test:conv-123")

        parsed = json.loads(result)
        assert parsed["id"] == "test:conv-123"
        assert parsed["provider"] == "chatgpt"
        assert parsed["title"] == "Test Conv"
        assert parsed["message_count"] == 5

    def test_summary_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_conversation_summary = AsyncMock(return_value=None)
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["get_conversation_summary"].fn, id="nonexistent")

        assert "not found" in json.loads(result)["error"].lower()

    def test_session_tree_returns_list(self, simple_conversation: Conversation, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_session_tree = AsyncMock(return_value=[simple_conversation])
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_session_tree"].fn, conversation_id="test:conv-123"
            )

        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["id"] == "test:conv-123"

    @pytest.mark.parametrize(
        ("group_by", "expected"),
        [
            ("provider", {"chatgpt": 10, "claude-ai": 5}),
            ("month", {"2024-01": 15, "2024-02": 20}),
        ],
    )
    def test_stats_by_group(self, group_by: str, expected: dict[str, int], mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.get_stats_by = AsyncMock(return_value=expected)
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(mcp_server._tool_manager._tools["get_stats_by"].fn, group_by=group_by)

        assert json.loads(result) == expected

    def test_health_check_success(self, mcp_server: MCPServerUnderTest) -> None:
        mock_check = MagicMock()
        mock_check.name = "database"
        mock_check.status.value = "ok"
        mock_check.count = 100
        mock_check.detail = "All good"

        mock_report = MagicMock()
        mock_report.checks = [mock_check]
        mock_report.summary = "Healthy"
        mock_report.provenance = MagicMock(source="live")

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.readiness.get_readiness") as mock_get_readiness,
        ):
            mock_get_config.return_value = MagicMock()
            mock_get_readiness.return_value = mock_report

            result = invoke_surface(mcp_server._tool_manager._tools["readiness_check"].fn)

        parsed = json.loads(result)
        assert parsed["summary"] == "Healthy"
        assert parsed["checks"][0]["name"] == "database"

    def test_rebuild_index_success(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_backend") as mock_get_backend,
            patch("polylogue.pipeline.services.indexing.IndexService") as mock_service_cls,
        ):
            mock_get_config.return_value = MagicMock()
            mock_get_backend.return_value = MagicMock()
            mock_service = MagicMock()
            mock_service.rebuild_index = AsyncMock(return_value=True)
            mock_service.get_index_status = AsyncMock(return_value={"exists": True, "count": 500})
            mock_service_cls.return_value = mock_service

            result = invoke_surface(mcp_server._tool_manager._tools["rebuild_index"].fn)

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["index_exists"] is True
        assert parsed["indexed_messages"] == 500

    def test_update_index_success(self, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_backend") as mock_get_backend,
            patch("polylogue.pipeline.services.indexing.IndexService") as mock_service_cls,
        ):
            mock_get_config.return_value = MagicMock()
            mock_get_backend.return_value = MagicMock()
            mock_service = MagicMock()
            mock_service.update_index = AsyncMock(return_value=True)
            mock_service_cls.return_value = mock_service

            result = invoke_surface(
                mcp_server._tool_manager._tools["update_index"].fn,
                conversation_ids=["test:conv-1", "test:conv-2"],
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["conversation_count"] == 2

    def test_rebuild_session_products_success(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops:
            mock_ops = MagicMock()
            mock_ops.rebuild_session_products = AsyncMock(
                return_value=SessionProductCounts(profiles=2, work_events=3, phases=1)
            )
            mock_get_archive_ops.return_value = mock_ops

            result = invoke_surface(
                mcp_server._tool_manager._tools["rebuild_session_products"].fn,
                conversation_ids=["conv-1", "conv-2"],
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["conversation_count"] == 2
        assert parsed["counts"]["profiles"] == 2
        assert parsed["total"] == 6
        mock_ops.rebuild_session_products.assert_awaited_once_with(conversation_ids=["conv-1", "conv-2"])

    def test_export_query_results_uses_shared_query_contract(
        self,
        simple_conversation: Conversation,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with (
            patch("polylogue.mcp.server._get_archive_ops") as mock_get_archive_ops,
            patch("polylogue.rendering.formatting.format_conversation") as mock_format,
        ):
            mock_ops = MagicMock()
            mock_ops.query_conversations = AsyncMock(return_value=[simple_conversation])
            mock_get_archive_ops.return_value = mock_ops
            mock_format.return_value = '{"exported": true}'

            result = invoke_surface(
                mcp_server._tool_manager._tools["export_query_results"].fn,
                query="hello",
                provider="chatgpt",
                format="json",
                limit=3,
            )

        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert parsed["format"] == "json"
        assert parsed["exports"][0]["conversation_id"] == "test:conv-123"
        assert parsed["exports"][0]["content"] == '{"exported": true}'
        spec = mock_ops.query_conversations.await_args.args[0]
        assert isinstance(spec, ConversationQuerySpec)
        assert spec.query_terms == ("hello",)
        assert tuple(str(provider) for provider in spec.providers) == ("chatgpt",)
        assert spec.limit == 3
