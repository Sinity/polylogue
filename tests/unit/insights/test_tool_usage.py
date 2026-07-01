"""Tests for the tool usage insight contract.

Covers four shapes:

1. Pure aggregation: ``build_tool_usage_insight`` correctly counts,
   filters, derives MCP servers, and reports coverage gaps.
2. End-to-end through the SQLite backend: per-(provider, tool) rollups
   built from canonical actions.
3. Coverage-gap honesty: providers with sessions but zero
   actions surface as ``data_available=False`` rather than silent
   zeros.
4. MCP envelope shape: ``list_tool_usage_insights`` returns a single
   ``ToolUsageInsight`` whose ``provider_coverage`` is always exhaustive,
   even when ``entries`` are narrowed by a query filter.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from polylogue import Polylogue
from polylogue.insights.tool_usage import (
    TOOL_USAGE_INSIGHT_VERSION,
    ToolUsageInsight,
    ToolUsageInsightQuery,
    build_tool_usage_insight,
    extract_mcp_server,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.queries.tool_usage import (
    ToolUsageProviderCoverageRow,
    ToolUsageRow,
)
from tests.infra.storage_records import SessionBuilder


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
    sessions: int = 1,
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
        "session_count": sessions,
        "message_count": messages,
        "distinct_tool_ids": tool_ids,
        "affected_path_calls": paths,
        "output_text_calls": outputs,
    }


def _coverage(
    *,
    provider: str,
    sessions: int,
    events: int = 0,
    tools: int = 0,
    kinds: int = 0,
    tool_ids: int = 0,
    paths: int = 0,
    outputs: int = 0,
) -> ToolUsageProviderCoverageRow:
    return {
        "source_name": provider,
        "session_count": sessions,
        "action_count": events,
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
            _row(provider="claude-code", tool="Read", calls=10, sessions=3, messages=8),
            _row(provider="claude-code", tool="Bash", calls=4, sessions=2),
        ]
        coverage = [
            _coverage(provider="claude-code", sessions=3, events=14, tools=2, kinds=1, paths=14),
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
            _coverage(provider="claude-code", sessions=2, events=5, tools=1, kinds=1),
            _coverage(provider="codex", sessions=1, events=2, tools=1, kinds=1),
            # ChatGPT has sessions but no tool data — this MUST stay
            # visible even when the user narrows to a different provider.
            _coverage(provider="chatgpt", sessions=4, events=0),
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
        assert chatgpt.action_count == 0

    def test_mcp_server_extracted_for_each_entry(self) -> None:
        rows = [
            _row(provider="claude-code", tool="mcp__github__create_pull_request", calls=3),
            _row(provider="claude-code", tool="mcp__polylogue__search", calls=2),
            _row(provider="claude-code", tool="Read", calls=1),
        ]
        coverage = [_coverage(provider="claude-code", sessions=1, events=6, tools=3, kinds=1)]
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
        coverage = [_coverage(provider="claude-code", sessions=1, events=5)]
        insight = build_tool_usage_insight(
            rows=rows,
            coverage_rows=coverage,
            query=ToolUsageInsightQuery(mcp_server="github"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert [entry.normalized_tool_name for entry in insight.entries] == ["mcp__github__create_pull_request"]
        assert insight.entries[0].mcp_server == "github"

    def test_provider_without_sessions_not_counted_as_gap(self) -> None:
        # A provider that has neither sessions nor actions still
        # appears in coverage with session_count=0, but it should not
        # contribute to ``providers_without_data`` because there is nothing
        # to be missing.
        coverage = [
            _coverage(provider="claude-code", sessions=2, events=4),
            _coverage(provider="legacy-empty", sessions=0, events=0),
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
        coverage = [_coverage(provider="claude-code", sessions=1, events=10)]
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


def _archive(tmp_path: Path) -> Polylogue:
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")


@pytest.mark.asyncio
class TestListToolUsageInsightsEndToEnd:
    """Drive the archive with seeded tool-use content blocks.

    The archive ``actions`` view derives tool calls from ``tool_use`` content
    blocks (paired with ``tool_result`` blocks by ``tool_id``), so the
    per-(provider, tool) rollups are seeded by writing those blocks through the
    archive ``SessionBuilder``. The native ``_tool_usage_rows`` query
    lower-cases ``tool_name``, so the normalized tool names asserted below are
    lower-case (``read`` / ``bash`` / ``apply_patch``).
    """

    async def test_aggregates_per_provider_and_tool(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        # claude-code: two `read` tool_use blocks with distinct tool_ids and
        # distinct affected paths, plus one `bash` tool_use whose paired
        # tool_result carries output text.
        (
            SessionBuilder(db_path, "cc-1")
            .provider("claude-code")
            .title("CC")
            .add_message(
                "cc-1-msg",
                role="assistant",
                text="Working",
                blocks=[
                    {"type": "tool_use", "name": "Read", "id": "toolu_1", "tool_input": {"file_path": "a.py"}},
                    {"type": "tool_use", "name": "Read", "id": "toolu_2", "tool_input": {"file_path": "b.py"}},
                    {"type": "tool_use", "name": "Bash", "id": "toolu_3"},
                    {"type": "tool_result", "tool_id": "toolu_3", "text": "hello"},
                ],
            )
            .save()
        )
        (
            SessionBuilder(db_path, "cx-1")
            .provider("codex")
            .title("CX")
            .add_message(
                "cx-1-msg",
                role="assistant",
                text="Patching",
                blocks=[{"type": "tool_use", "name": "apply_patch", "id": "p1"}],
            )
            .save()
        )

        result = await archive.list_tool_usage_insights(ToolUsageInsightQuery())
        assert len(result) == 1
        insight = result[0]
        assert isinstance(insight, ToolUsageInsight)
        # Aggregate counts honor the substrate (archive lower-cases tool names).
        read = next(
            entry
            for entry in insight.entries
            if entry.source_name == "claude-code" and entry.normalized_tool_name == "read"
        )
        assert read.call_count == 2
        assert read.session_count == 1
        assert read.distinct_tool_ids == 2
        assert read.affected_path_calls == 2
        bash = next(
            entry
            for entry in insight.entries
            if entry.source_name == "claude-code" and entry.normalized_tool_name == "bash"
        )
        assert bash.output_text_calls == 1
        assert bash.affected_path_calls == 0
        # Coverage covers both providers
        providers = {entry.source_name for entry in insight.provider_coverage}
        assert providers == {"claude-code", "codex"}

    async def test_coverage_reports_provider_without_actions(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        # ChatGPT session exists but carries no tool_use blocks — the
        # substrate genuinely has no tool data for this provider.
        (
            SessionBuilder(db_path, "gpt-1")
            .provider("chatgpt")
            .title("GPT")
            .add_message("gpt-1-msg", role="user", text="Hi")
            .save()
        )
        (
            SessionBuilder(db_path, "cc-1")
            .provider("claude-code")
            .title("CC")
            .add_message(
                "cc-1-msg",
                role="assistant",
                text="Reading",
                blocks=[{"type": "tool_use", "name": "Read", "id": "toolu_only"}],
            )
            .save()
        )

        result = await archive.list_tool_usage_insights(ToolUsageInsightQuery())
        insight = result[0]
        chatgpt = next(entry for entry in insight.provider_coverage if entry.source_name == "chatgpt")
        assert chatgpt.session_count == 1
        assert chatgpt.action_count == 0
        assert chatgpt.data_available is False
        cc = next(entry for entry in insight.provider_coverage if entry.source_name == "claude-code")
        assert cc.data_available is True
        assert insight.has_coverage_gaps is True
        assert insight.providers_without_data == 1
        assert insight.providers_with_data == 1

    async def test_query_filters_entries_without_narrowing_coverage(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        (
            SessionBuilder(db_path, "cc-1")
            .provider("claude-code")
            .title("CC")
            .add_message(
                "cc-1-msg",
                role="assistant",
                text="Inspecting",
                blocks=[
                    {"type": "tool_use", "name": "mcp__serena__find_symbol", "id": "s1"},
                    {"type": "tool_use", "name": "mcp__serena__find_symbol", "id": "s2"},
                    {"type": "tool_use", "name": "Read", "id": "r1"},
                ],
            )
            .save()
        )
        (
            SessionBuilder(db_path, "cx-1")
            .provider("codex")
            .title("CX")
            .add_message(
                "cx-1-msg",
                role="assistant",
                text="Searching",
                blocks=[{"type": "tool_use", "name": "mcp__codebase-memory__search_code", "id": "c1"}],
            )
            .save()
        )
        (
            SessionBuilder(db_path, "gpt-1")
            .provider("chatgpt")
            .title("GPT")
            .add_message("gpt-1-msg", role="user", text="Hi")
            .save()
        )

        [insight] = await archive.list_tool_usage_insights(
            ToolUsageInsightQuery(provider="claude-code-session", mcp_server="serena", limit=1)
        )

        assert [entry.normalized_tool_name for entry in insight.entries] == ["mcp__serena__find_symbol"]
        assert insight.entries[0].call_count == 2
        assert insight.entries[0].source_name == "claude-code"
        assert insight.entries[0].mcp_server == "serena"
        assert {entry.source_name for entry in insight.provider_coverage} == {
            "claude-code",
            "codex",
            "chatgpt",
        }
        assert insight.has_coverage_gaps is True

    async def test_query_pagination_is_not_applied_twice(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        (
            SessionBuilder(db_path, "cc-1")
            .provider("claude-code")
            .title("CC")
            .add_message(
                "cc-1-msg",
                role="assistant",
                text="Tools",
                blocks=[
                    {"type": "tool_use", "name": "Alpha", "id": "a1"},
                    {"type": "tool_use", "name": "Alpha", "id": "a2"},
                    {"type": "tool_use", "name": "Beta", "id": "b1"},
                    {"type": "tool_use", "name": "Gamma", "id": "g1"},
                ],
            )
            .save()
        )

        [insight] = await archive.list_tool_usage_insights(ToolUsageInsightQuery(offset=1, limit=1))

        assert [entry.normalized_tool_name for entry in insight.entries] == ["beta"]

    async def test_observed_event_tool_counts_use_materialized_projection(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        (
            SessionBuilder(db_path, "cc-1")
            .provider("claude-code")
            .title("CC")
            .add_message("cc-1-msg", role="assistant", text="Tools")
            .save()
        )
        session_id = "claude-code-session:ext-cc-1"
        conn = sqlite3.connect(db_path)
        try:
            conn.executemany(
                """
                INSERT INTO session_observed_events (
                    event_ref, session_id, run_ref, position, kind, summary,
                    delivery_state, payload_json, search_text
                ) VALUES (?, ?, ?, ?, 'tool_finished', ?, 'observed', ?, ?)
                """,
                [
                    (
                        "event:1",
                        session_id,
                        "run:cc-1",
                        1,
                        "serena ok",
                        json.dumps(
                            {
                                "tool_name": "mcp__serena__find_symbol",
                                "handler_kind": "mcp",
                                "status": "ok",
                            }
                        ),
                        "serena ok",
                    ),
                    (
                        "event:2",
                        session_id,
                        "run:cc-1",
                        2,
                        "serena failed",
                        json.dumps(
                            {
                                "tool_name": "mcp__serena__find_symbol",
                                "handler_kind": "mcp",
                                "status": "failed",
                            }
                        ),
                        "serena failed",
                    ),
                    (
                        "event:3",
                        session_id,
                        "run:cc-1",
                        3,
                        "read ok",
                        json.dumps({"tool_name": "Read", "handler_kind": "file_read", "status": "ok"}),
                        "read ok",
                    ),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        with ArchiveStore.open_existing(archive.archive_root) as store:
            rows = store.list_tool_observed_event_count_rows(
                ToolUsageInsightQuery(provider="claude-code-session", mcp_server="serena", limit=5)
            )

        assert rows == [
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "status": "failed",
                "event_count": 1,
            },
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "status": "ok",
                "event_count": 1,
            },
        ]

    async def test_observed_event_mcp_filter_uses_tool_expression_index(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        (
            SessionBuilder(db_path, "cc-1")
            .provider("claude-code")
            .title("CC")
            .add_message("cc-1-msg", role="assistant", text="Tools")
            .save()
        )
        tool_expr = "COALESCE(NULLIF(json_extract(payload_json, '$.tool_name'), ''), 'unknown')"
        mcp_prefix = "mcp__serena__"

        conn = sqlite3.connect(db_path)
        try:
            plan = [
                str(row[-1])
                for row in conn.execute(
                    f"""
                    EXPLAIN QUERY PLAN
                    SELECT COUNT(*)
                    FROM session_observed_events
                    WHERE kind = 'tool_finished'
                      AND {tool_expr} >= ?
                      AND {tool_expr} < ?
                    """,
                    (mcp_prefix, f"{mcp_prefix}\U0010ffff"),
                )
            ]
        finally:
            conn.close()

        assert any("idx_session_observed_events_kind_tool" in row for row in plan)
        assert any("<expr>>?" in row and "<expr><?" in row for row in plan)

    async def test_action_evidence_counts_normalize_detail_matches(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        (
            SessionBuilder(db_path, "cx-1")
            .provider("codex")
            .title("CX")
            .add_message(
                "cx-1-msg",
                role="assistant",
                text="Searching",
                blocks=[
                    {
                        "type": "tool_use",
                        "name": "functions.exec_command",
                        "id": "t1",
                        "input": {"command": 'codebase-memory-mcp cli search_code \'{"project":"polylogue"}\''},
                    },
                    {
                        "type": "tool_result",
                        "id": "t1",
                        "text": "ok",
                        "tool_result_is_error": 0,
                        "tool_result_exit_code": 0,
                    },
                    {"type": "tool_use", "name": "search_code", "id": "t2", "input": {"query": "ToolUsage"}},
                    {"type": "tool_use", "name": "", "id": "t3", "text": "codebase-memory mentioned in prose"},
                ],
            )
            .save()
        )

        with ArchiveStore.open_existing(archive.archive_root) as store:
            rows = store.list_tool_action_evidence_count_rows(
                ToolUsageInsightQuery(provider="codex-session", limit=5),
                detail_patterns=("codebase-memory",),
            )

        assert rows == [
            {
                "source_name": "codex",
                "origin": "codex-session",
                "normalized_tool_name": "codebase-memory/command-detail",
                "action_kind": "tool_use",
                "evidence_kind": "command_detail",
                "matched_by": "detail",
                "call_count": 1,
                "session_count": 1,
                "error_count": 0,
                "nonzero_exit_count": 0,
            }
        ]

    async def test_action_evidence_since_filter_uses_session_sort_key(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        old_ts = "2026-01-01T00:00:00+00:00"
        recent_ts = "2026-01-03T00:00:00+00:00"
        cutoff_ms = int(datetime(2026, 1, 2, tzinfo=UTC).timestamp() * 1000)
        command_block = {
            "type": "tool_use",
            "name": "functions.exec_command",
            "id": "t1",
            "input": {"command": "codebase-memory-mcp cli search_code '{}'"},
        }
        result_block = {
            "type": "tool_result",
            "id": "t1",
            "text": "ok",
            "tool_result_is_error": 0,
            "tool_result_exit_code": 0,
        }

        (
            SessionBuilder(db_path, "old-cx")
            .provider("codex")
            .updated_at(old_ts)
            .add_message(
                "old-cx-msg", role="assistant", text="Old", timestamp=old_ts, blocks=[command_block, result_block]
            )
            .save()
        )
        (
            SessionBuilder(db_path, "recent-cx")
            .provider("codex")
            .updated_at(recent_ts)
            .add_message(
                "recent-cx-msg",
                role="assistant",
                text="Recent",
                timestamp=recent_ts,
                blocks=[command_block, result_block],
            )
            .save()
        )

        with ArchiveStore.open_existing(archive.archive_root) as store:
            rows = store.list_tool_action_evidence_count_rows(
                ToolUsageInsightQuery(provider="codex-session", limit=5),
                detail_patterns=("codebase-memory",),
                since_ms=cutoff_ms,
            )

        assert rows == [
            {
                "source_name": "codex",
                "origin": "codex-session",
                "normalized_tool_name": "codebase-memory/command-detail",
                "action_kind": "tool_use",
                "evidence_kind": "command_detail",
                "matched_by": "detail",
                "call_count": 1,
                "session_count": 1,
                "error_count": 0,
                "nonzero_exit_count": 0,
            }
        ]

    async def test_call_and_observed_event_counts_since_filter_uses_session_sort_key(self, tmp_path: Path) -> None:
        archive = _archive(tmp_path)
        db_path = archive.archive_root / "index.db"
        old_ts = "2026-01-01T00:00:00+00:00"
        recent_ts = "2026-01-03T00:00:00+00:00"
        cutoff_ms = int(datetime(2026, 1, 2, tzinfo=UTC).timestamp() * 1000)
        tool_block = {"type": "tool_use", "name": "mcp__serena__find_symbol", "id": "t1"}

        (
            SessionBuilder(db_path, "old-cc")
            .provider("claude-code")
            .updated_at(old_ts)
            .add_message("old-cc-msg", role="assistant", text="Old", timestamp=old_ts, blocks=[tool_block])
            .save()
        )
        (
            SessionBuilder(db_path, "recent-cc")
            .provider("claude-code")
            .updated_at(recent_ts)
            .add_message(
                "recent-cc-msg",
                role="assistant",
                text="Recent",
                timestamp=recent_ts,
                blocks=[tool_block],
            )
            .save()
        )

        conn = sqlite3.connect(db_path)
        try:
            conn.executemany(
                """
                INSERT INTO session_observed_events (
                    event_ref, session_id, run_ref, position, kind, summary,
                    delivery_state, payload_json, search_text
                ) VALUES (?, ?, ?, 1, 'tool_finished', ?, 'observed', ?, ?)
                """,
                [
                    (
                        "event:old",
                        "claude-code-session:ext-old-cc",
                        "run:old",
                        "old serena ok",
                        json.dumps(
                            {
                                "tool_name": "mcp__serena__find_symbol",
                                "handler_kind": "mcp",
                                "status": "ok",
                            }
                        ),
                        "old serena ok",
                    ),
                    (
                        "event:recent",
                        "claude-code-session:ext-recent-cc",
                        "run:recent",
                        "recent serena ok",
                        json.dumps(
                            {
                                "tool_name": "mcp__serena__find_symbol",
                                "handler_kind": "mcp",
                                "status": "ok",
                            }
                        ),
                        "recent serena ok",
                    ),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        query = ToolUsageInsightQuery(
            provider="claude-code-session",
            mcp_server="serena",
            since_ms=cutoff_ms,
            limit=5,
        )
        with ArchiveStore.open_existing(archive.archive_root) as store:
            call_rows = store.list_tool_call_count_rows(query)
            event_rows = store.list_tool_observed_event_count_rows(query)

        assert call_rows == [
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "tool_use",
                "call_count": 1,
            }
        ]
        assert event_rows == [
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "status": "ok",
                "event_count": 1,
            }
        ]

    async def test_empty_archive_returns_envelope_with_no_gaps(self, tmp_path: Path) -> None:
        result = await _archive(tmp_path).list_tool_usage_insights(ToolUsageInsightQuery())
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
            _coverage(provider="claude-code", sessions=1, events=3, tools=1, kinds=1),
            _coverage(provider="chatgpt", sessions=2, events=0),
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
