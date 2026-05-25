"""Tests for the tool usage insight contract.

Covers four shapes:

1. Pure aggregation: ``build_tool_usage_insight`` correctly counts,
   filters, derives MCP servers, and reports coverage gaps.
2. End-to-end through the SQLite backend: per-(provider, tool) rollups
   built from canonical action_events.
3. Coverage-gap honesty: providers with conversations but zero
   action_events surface as ``data_available=False`` rather than silent
   zeros.
4. MCP envelope shape: ``list_tool_usage_insights`` returns a single
   ``ToolUsageInsight`` whose ``provider_coverage`` is always exhaustive,
   even when ``entries`` are narrowed by a query filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from polylogue.insights.tool_usage import (
    TOOL_USAGE_INSIGHT_VERSION,
    ToolUsageInsight,
    ToolUsageInsightQuery,
    build_tool_usage_insight,
    extract_mcp_server,
)
from polylogue.operations.archive import list_tool_usage_insights
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import ActionEventRecord
from polylogue.storage.sqlite.queries import action_events as action_events_q
from polylogue.storage.sqlite.queries.tool_usage import (
    ToolUsageProviderCoverageRow,
    ToolUsageRow,
)
from polylogue.types import ConversationId, MessageId
from tests.infra.storage_records import make_conversation, make_message


class TestExtractMcpServer:
    def test_canonical_mcp_prefix(self) -> None:
        assert extract_mcp_server("mcp__github__create_pull_request") == "github"
        assert extract_mcp_server("mcp__polylogue__search") == "polylogue"

    def test_non_mcp_tool(self) -> None:
        assert extract_mcp_server("Read") is None
        assert extract_mcp_server("Bash") is None
        assert extract_mcp_server("") is None

    def test_malformed_mcp_prefix(self) -> None:
        # Prefix present but no second separator => not a valid MCP tool name
        assert extract_mcp_server("mcp__only") is None
        # Prefix present but server segment empty
        assert extract_mcp_server("mcp____tool") is None


def _row(
    *,
    provider: str,
    tool: str,
    action_kind: str = "tool_use",
    calls: int = 1,
    conversations: int = 1,
    messages: int = 1,
    tool_ids: int = 0,
    paths: int = 0,
    outputs: int = 0,
) -> ToolUsageRow:
    return {
        "source_name": provider,
        "normalized_tool_name": tool,
        "action_kind": action_kind,
        "call_count": calls,
        "conversation_count": conversations,
        "message_count": messages,
        "distinct_tool_ids": tool_ids,
        "affected_path_calls": paths,
        "output_text_calls": outputs,
    }


def _coverage(
    *,
    provider: str,
    conversations: int,
    events: int = 0,
    tools: int = 0,
    kinds: int = 0,
    tool_ids: int = 0,
    paths: int = 0,
    outputs: int = 0,
) -> ToolUsageProviderCoverageRow:
    return {
        "source_name": provider,
        "conversation_count": conversations,
        "action_event_count": events,
        "distinct_tool_count": tools,
        "distinct_action_kind_count": kinds,
        "has_tool_id_signal": tool_ids,
        "has_affected_paths_signal": paths,
        "has_output_text_signal": outputs,
    }


class TestBuildToolUsageInsight:
    """Unit test the pure aggregator without a database."""

    def test_basic_aggregation(self) -> None:
        rows = [
            _row(provider="claude-code", tool="Read", calls=10, conversations=3, messages=8),
            _row(provider="claude-code", tool="Bash", calls=4, conversations=2),
        ]
        coverage = [
            _coverage(provider="claude-code", conversations=3, events=14, tools=2, kinds=1, paths=14),
        ]
        insight = build_tool_usage_insight(
            rows=rows,
            coverage_rows=coverage,
            query=ToolUsageInsightQuery(),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.materializer_version == TOOL_USAGE_INSIGHT_VERSION
        # entries are sorted by call count desc
        assert [entry.normalized_tool_name for entry in insight.entries] == ["Read", "Bash"]
        assert insight.total_call_count == 14
        assert insight.total_distinct_tools == 2
        assert insight.providers_with_data == 1
        assert insight.providers_without_data == 0
        assert insight.has_coverage_gaps is False
        assert insight.provider_coverage[0].data_available is True
        assert insight.provider_coverage[0].has_affected_paths_signal is True

    def test_filter_by_provider_does_not_hide_coverage_gaps(self) -> None:
        rows = [
            _row(provider="claude-code", tool="Read", calls=5),
            _row(provider="codex", tool="apply_patch", calls=2),
        ]
        coverage = [
            _coverage(provider="claude-code", conversations=2, events=5, tools=1, kinds=1),
            _coverage(provider="codex", conversations=1, events=2, tools=1, kinds=1),
            # ChatGPT has conversations but no tool data — this MUST stay
            # visible even when the user narrows to a different provider.
            _coverage(provider="chatgpt", conversations=4, events=0),
        ]
        insight = build_tool_usage_insight(
            rows=rows,
            coverage_rows=coverage,
            query=ToolUsageInsightQuery(provider="claude-code"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert [entry.source_name for entry in insight.entries] == ["claude-code"]
        # Coverage is exhaustive even though entries are narrowed.
        assert {entry.source_name for entry in insight.provider_coverage} == {
            "claude-code",
            "codex",
            "chatgpt",
        }
        assert insight.providers_with_data == 2
        assert insight.providers_without_data == 1
        assert insight.has_coverage_gaps is True
        chatgpt = next(entry for entry in insight.provider_coverage if entry.source_name == "chatgpt")
        assert chatgpt.data_available is False
        assert chatgpt.action_event_count == 0

    def test_mcp_server_extracted_for_each_entry(self) -> None:
        rows = [
            _row(provider="claude-code", tool="mcp__github__create_pull_request", calls=3),
            _row(provider="claude-code", tool="mcp__polylogue__search", calls=2),
            _row(provider="claude-code", tool="Read", calls=1),
        ]
        coverage = [_coverage(provider="claude-code", conversations=1, events=6, tools=3, kinds=1)]
        insight = build_tool_usage_insight(
            rows=rows,
            coverage_rows=coverage,
            query=ToolUsageInsightQuery(),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        by_tool = {entry.normalized_tool_name: entry.mcp_server for entry in insight.entries}
        assert by_tool["mcp__github__create_pull_request"] == "github"
        assert by_tool["mcp__polylogue__search"] == "polylogue"
        assert by_tool["Read"] is None

    def test_filter_by_mcp_server_keeps_only_matching(self) -> None:
        rows = [
            _row(provider="claude-code", tool="mcp__github__create_pull_request", calls=3),
            _row(provider="claude-code", tool="mcp__polylogue__search", calls=2),
        ]
        coverage = [_coverage(provider="claude-code", conversations=1, events=5)]
        insight = build_tool_usage_insight(
            rows=rows,
            coverage_rows=coverage,
            query=ToolUsageInsightQuery(mcp_server="github"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert [entry.normalized_tool_name for entry in insight.entries] == ["mcp__github__create_pull_request"]
        assert insight.entries[0].mcp_server == "github"

    def test_provider_without_conversations_not_counted_as_gap(self) -> None:
        # A provider that has neither conversations nor action_events still
        # appears in coverage with conversation_count=0, but it should not
        # contribute to ``providers_without_data`` because there is nothing
        # to be missing.
        coverage = [
            _coverage(provider="claude-code", conversations=2, events=4),
            _coverage(provider="legacy-empty", conversations=0, events=0),
        ]
        insight = build_tool_usage_insight(
            rows=[_row(provider="claude-code", tool="Read", calls=4)],
            coverage_rows=coverage,
            query=ToolUsageInsightQuery(),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.providers_with_data == 1
        assert insight.providers_without_data == 0
        assert insight.has_coverage_gaps is False

    def test_pagination_respects_offset_and_limit(self) -> None:
        rows = [_row(provider="claude-code", tool=f"tool_{i:02d}", calls=100 - i) for i in range(10)]
        coverage = [_coverage(provider="claude-code", conversations=1, events=10)]
        insight = build_tool_usage_insight(
            rows=rows,
            coverage_rows=coverage,
            query=ToolUsageInsightQuery(offset=2, limit=3),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert [entry.normalized_tool_name for entry in insight.entries] == [
            "tool_02",
            "tool_03",
            "tool_04",
        ]


def _make_action_event(
    event_id: str,
    *,
    conversation_id: str,
    message_id: str,
    provider: str,
    tool: str,
    action_kind: str = "tool_use",
    tool_id: str | None = None,
    affected_paths: tuple[str, ...] = (),
    output_text: str | None = None,
) -> ActionEventRecord:
    return ActionEventRecord(
        event_id=event_id,
        conversation_id=ConversationId(conversation_id),
        message_id=MessageId(message_id),
        materializer_version=1,
        source_block_id=None,
        timestamp="2026-05-17T00:00:00+00:00",
        sort_key=0.0,
        sequence_index=0,
        source_name=provider,
        action_kind=action_kind,
        tool_name=tool,
        normalized_tool_name=tool,
        tool_id=tool_id,
        affected_paths=affected_paths,
        cwd_path=None,
        branch_names=(),
        command=None,
        query_text=None,
        url=None,
        output_text=output_text,
        search_text=f"{tool} {action_kind}",
    )


@pytest.mark.asyncio
class TestListToolUsageInsightsEndToEnd:
    """Drive the real backend with seeded action_events."""

    async def _seed_conversation(
        self,
        repository: ConversationRepository,
        *,
        conv_id: str,
        provider: str,
    ) -> None:
        conv = make_conversation(conv_id, source_name=provider, provider_meta={"source": "inbox"})
        msgs = [make_message(f"{conv_id}-msg", conv_id, text="Hi")]
        await repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

    async def test_aggregates_per_provider_and_tool(
        self,
        workspace_env: dict[str, Path],
        storage_repository: ConversationRepository,
    ) -> None:
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        await self._seed_conversation(storage_repository, conv_id="cc-1", provider="claude-code")
        await self._seed_conversation(storage_repository, conv_id="cx-1", provider="codex")

        backend = storage_repository.backend
        async with backend.connection() as conn:
            await action_events_q.replace_action_events(
                conn,
                "cc-1",
                [
                    _make_action_event(
                        "ev-1",
                        conversation_id="cc-1",
                        message_id="cc-1-msg",
                        provider="claude-code",
                        tool="Read",
                        affected_paths=("a.py",),
                        tool_id="toolu_1",
                    ),
                    _make_action_event(
                        "ev-2",
                        conversation_id="cc-1",
                        message_id="cc-1-msg",
                        provider="claude-code",
                        tool="Read",
                        affected_paths=("b.py",),
                        tool_id="toolu_2",
                    ),
                    _make_action_event(
                        "ev-3",
                        conversation_id="cc-1",
                        message_id="cc-1-msg",
                        provider="claude-code",
                        tool="Bash",
                        output_text="hello",
                    ),
                ],
                transaction_depth=0,
            )
            await action_events_q.replace_action_events(
                conn,
                "cx-1",
                [
                    _make_action_event(
                        "ev-4",
                        conversation_id="cx-1",
                        message_id="cx-1-msg",
                        provider="codex",
                        tool="apply_patch",
                    ),
                ],
                transaction_depth=0,
            )

        result = await list_tool_usage_insights(db_path=db_path)
        assert len(result) == 1
        insight = result[0]
        assert isinstance(insight, ToolUsageInsight)
        # Aggregate counts honor the substrate
        read = next(
            entry
            for entry in insight.entries
            if entry.source_name == "claude-code" and entry.normalized_tool_name == "Read"
        )
        assert read.call_count == 2
        assert read.conversation_count == 1
        assert read.distinct_tool_ids == 2
        assert read.affected_path_calls == 2
        bash = next(
            entry
            for entry in insight.entries
            if entry.source_name == "claude-code" and entry.normalized_tool_name == "Bash"
        )
        assert bash.output_text_calls == 1
        assert bash.affected_path_calls == 0
        # Coverage covers both providers
        providers = {entry.source_name for entry in insight.provider_coverage}
        assert providers == {"claude-code", "codex"}

    async def test_coverage_reports_provider_without_action_events(
        self,
        workspace_env: dict[str, Path],
        storage_repository: ConversationRepository,
    ) -> None:
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        # ChatGPT conversation exists but contributes no action_events —
        # the substrate genuinely has no tool data for this provider.
        await self._seed_conversation(storage_repository, conv_id="gpt-1", provider="chatgpt")
        await self._seed_conversation(storage_repository, conv_id="cc-1", provider="claude-code")
        async with storage_repository.backend.connection() as conn:
            await action_events_q.replace_action_events(
                conn,
                "cc-1",
                [
                    _make_action_event(
                        "ev-only",
                        conversation_id="cc-1",
                        message_id="cc-1-msg",
                        provider="claude-code",
                        tool="Read",
                    ),
                ],
                transaction_depth=0,
            )

        result = await list_tool_usage_insights(db_path=db_path)
        insight = result[0]
        chatgpt = next(entry for entry in insight.provider_coverage if entry.source_name == "chatgpt")
        assert chatgpt.conversation_count == 1
        assert chatgpt.action_event_count == 0
        assert chatgpt.data_available is False
        cc = next(entry for entry in insight.provider_coverage if entry.source_name == "claude-code")
        assert cc.data_available is True
        assert insight.has_coverage_gaps is True
        assert insight.providers_without_data == 1
        assert insight.providers_with_data == 1

    async def test_empty_archive_returns_envelope_with_no_gaps(
        self,
        workspace_env: dict[str, Path],
        storage_repository: ConversationRepository,
    ) -> None:
        del storage_repository  # ensure backend initialized via fixture
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        result = await list_tool_usage_insights(db_path=db_path)
        assert len(result) == 1
        insight = result[0]
        assert insight.entries == ()
        assert insight.provider_coverage == ()
        assert insight.has_coverage_gaps is False
        assert insight.total_call_count == 0


def test_envelope_serializes_to_jsonable_dict() -> None:
    """MCP envelope shape: model dumps to a stable JSON-serializable dict."""

    insight = build_tool_usage_insight(
        rows=[_row(provider="claude-code", tool="Read", calls=3)],
        coverage_rows=[
            _coverage(provider="claude-code", conversations=1, events=3, tools=1, kinds=1),
            _coverage(provider="chatgpt", conversations=2, events=0),
        ],
        query=ToolUsageInsightQuery(),
        materialized_at="2026-05-17T00:00:00+00:00",
    )
    payload = insight.model_dump(mode="json")
    assert payload["insight_kind"] == "tool_usage"
    assert payload["materializer_version"] == TOOL_USAGE_INSIGHT_VERSION
    assert payload["has_coverage_gaps"] is True
    assert {entry["source_name"] for entry in payload["provider_coverage"]} == {
        "claude-code",
        "chatgpt",
    }
    entry = payload["entries"][0]
    assert entry["normalized_tool_name"] == "Read"
    assert entry["mcp_server"] is None
    # `cast` keeps mypy happy with the heterogeneous payload dict.
    provenance = cast(dict[str, object], payload["provenance"])
    assert provenance["materialized_at"] == "2026-05-17T00:00:00+00:00"
    assert provenance["materializer_version"] == TOOL_USAGE_INSIGHT_VERSION
