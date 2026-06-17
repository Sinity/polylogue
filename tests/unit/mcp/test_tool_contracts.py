"""Unit contracts for MCP tool surfaces backed by the store.

Read-heavy MCP tools (``search``, ``list_sessions``, ``get_session``,
``get_session_summary``, ``get_messages``, ``stats``) route unconditionally
through ``ArchiveStore`` over the archive. These tests
seed a archive on disk, point the MCP ``_get_config`` seam at it, and
assert against the archive tool output. Monolithic ``_get_archive_ops`` and ``_get_polylogue`` seams are no longer on these read paths.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Session, SessionSummary
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.archive.semantic.pricing import CostEstimatePayload, CostUsagePayload
from polylogue.archive.session.neighbor_candidates import NeighborReason, SessionNeighborCandidate
from polylogue.archive.stats import ArchiveStats
from polylogue.archive.viewport import read_view_profile_payloads
from polylogue.core.enums import Origin
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.insights.archive import (
    ArchiveCoverageInsight,
    ArchiveDebtInsight,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    CostRollupInsight,
    SessionCostInsight,
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    SessionPhaseInsight,
    SessionProfileInsight,
    SessionTagRollupInsight,
    SessionWorkEventInsight,
    ThreadInsight,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.insights.archive_models import (
    ThreadMemberEvidencePayload,
    ThreadPayload,
)
from polylogue.insights.confidence import ConfidenceBand
from polylogue.mcp.archive_support import (
    archive_search_payload,
    archive_session_list_payload,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import TagMutationResult
from polylogue.types import BlockType, Provider, SessionId
from tests.infra.mcp import (
    MCPServerUnderTest,
    invoke_surface,
    invoke_surface_async,
    make_polylogue_mock,
    make_simple_session,
)

STATS_CONFIGS = [
    (
        100,
        5000,
        {"claude-ai-export": 50, "chatgpt-export": 30, "claude-code-session": 20},
        10,
        200,
        90,
        1048576,
        10.0,
        1.0,
    ),
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
        {"query": "hello", "origin": "claude-ai-export", "since": "2024-01-01", "limit": 5},
        {
            "contains": ("hello",),
            "origin": ("claude-ai-export",),
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
        "list_sessions",
        {"limit": 10},
        {
            "limit": (10,),
        },
    ),
    (
        "list_sessions",
        {"retrieval_lane": "hybrid", "limit": 2},
        {
            "retrieval_lane": ("hybrid",),
            "limit": (2,),
        },
    ),
    (
        "list_sessions",
        {"origin": "claude-ai-export", "since": "2024-01-01", "tag": "bug", "title": "incident", "limit": 2},
        {
            "origin": ("claude-ai-export",),
            "since": ("2024-01-01",),
            "tag": ("bug",),
            "title": ("incident",),
            "limit": (2,),
        },
    ),
    (
        "list_sessions",
        {"referenced_path": "/workspace/polylogue/README.md", "limit": 2},
        {
            "referenced_path": ("/workspace/polylogue/README.md",),
            "limit": (2,),
        },
    ),
    (
        "list_sessions",
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
        "list_sessions",
        {"action_sequence": "file_read,file_edit,shell", "limit": 2},
        {
            "action_sequence": (("file_read", "file_edit", "shell"),),
            "limit": (2,),
        },
    ),
    (
        "list_sessions",
        {"action_text": "pytest -q", "limit": 2},
        {
            "action_text": ("pytest -q",),
            "limit": (2,),
        },
    ),
    (
        "list_sessions",
        {"action": "none", "limit": 2},
        {
            "action": ("none",),
            "limit": (2,),
        },
    ),
]


@pytest.fixture
def simple_session() -> Session:
    return make_simple_session()


def _make_neighbor_candidate() -> SessionNeighborCandidate:
    return SessionNeighborCandidate(
        summary=SessionSummary(
            id=SessionId("candidate"),
            origin=Origin.CODEX_SESSION,
            title="Archive Lock Retries",
            message_count=2,
            updated_at=datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc),
        ),
        rank=1,
        score=3.25,
        reasons=(
            NeighborReason(
                kind="nearby_time",
                detail="within 2.0h of source session",
                evidence="source=2026-04-22T12:00:00+00:00 candidate=2026-04-22T14:00:00+00:00",
                weight=1.0,
            ),
        ),
        source_session_id="target",
    )


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(
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


def _seed_archive(
    archive_root: Path,
    *,
    provider: Provider = Provider.CHATGPT,
    native_id: str = "conv-123",
    title: str = "Test Conv",
    text: str = "hello evidence here",
    extra: tuple[tuple[str, str], ...] = (),
) -> str:
    """Write a single native session into ``archive_root`` and return its id."""
    messages = [
        ParsedMessage(
            provider_message_id=f"{native_id}-m1",
            role=Role.USER,
            text=text,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
        )
    ]
    for msg_id, msg_text in extra:
        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=Role.ASSISTANT,
                text=msg_text,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=msg_text)],
            )
        )
    with ArchiveStore(archive_root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=provider,
                provider_session_id=native_id,
                title=title,
                messages=messages,
            )
        )


@contextlib.contextmanager
def _archive_config(archive_root: Path) -> Iterator[None]:
    """Point the MCP ``_get_config`` seam at a archive root."""
    with patch("polylogue.mcp.server._get_config") as mock_get_config:
        mock_get_config.return_value = SimpleNamespace(
            archive_root=archive_root,
            db_path=archive_root / "index.db",
        )
        yield


class TestQueryTools:
    @pytest.mark.parametrize(("tool_name", "args", "expected_calls"), QUERY_TOOL_CASES)
    @pytest.mark.asyncio
    async def test_query_tool_filter_contract(
        self,
        tmp_path: Path,
        tool_name: str,
        args: dict[str, object],
        expected_calls: dict[str, tuple[object, ...]],
        mcp_server: MCPServerUnderTest,
    ) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)

        # The archive read path builds the same SessionQuerySpec and hands it
        # to the archive index payload helpers. Spy on those helpers to capture the
        # spec so the MCP-args -> spec mapping contract stays pinned, while the
        # tool still executes end to end against the seeded archive.
        captured: dict[str, SessionQuerySpec] = {}
        real_search = archive_search_payload
        real_list = archive_session_list_payload

        def spy_search(archive: object, spec: SessionQuerySpec, **kwargs: object) -> object:
            captured["spec"] = spec
            return real_search(archive, spec, **kwargs)  # type: ignore[arg-type]

        def spy_list(archive: object, spec: SessionQuerySpec, **kwargs: object) -> object:
            captured["spec"] = spec
            return real_list(archive, spec, **kwargs)  # type: ignore[arg-type]

        with (
            _archive_config(archive_root),
            patch("polylogue.mcp.server_tools.archive_search_payload", spy_search),
            patch("polylogue.mcp.server_tools.archive_session_list_payload", spy_list),
        ):
            raw = await invoke_surface_async(mcp_server._tool_manager._tools[tool_name].fn, **args)

        payload = json.loads(raw)
        assert isinstance(payload, dict)
        if tool_name == "search":
            assert "hits" in payload and "total" in payload
        else:
            assert "items" in payload and "total" in payload
        spec = captured["spec"]
        assert isinstance(spec, SessionQuerySpec)
        for method_name, method_args in expected_calls.items():
            expected_value = method_args[0] if len(method_args) == 1 else method_args
            if method_name == "contains":
                assert spec.query_terms == (expected_value,)
            elif method_name == "origin":
                assert spec.origins == (expected_value,)
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
    async def test_search_no_match_returns_empty_hits_envelope(
        self, tmp_path: Path, mcp_server: MCPServerUnderTest
    ) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root, text="unrelated content")

        with _archive_config(archive_root):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="zzznomatchxyzzy",
                limit=10,
            )

        parsed = json.loads(result)
        assert parsed["hits"] == []
        assert parsed["total"] == 0
        # Archive search has no miss-diagnostics payload for an empty result;
        # the envelope carries the field, defaulted to null.
        assert parsed["diagnostics"] is None

    @pytest.mark.asyncio
    async def test_search_total_reflects_native_hit_count(self, tmp_path: Path, mcp_server: MCPServerUnderTest) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root, text="semantic planning notes for the refactor")

        with _archive_config(archive_root):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="planning",
                limit=10,
            )

        parsed = json.loads(result)
        assert parsed["total"] == len(parsed["hits"])
        assert parsed["total"] == 1
        assert parsed["hits"][0]["match"]["snippet"]

    @pytest.mark.asyncio
    async def test_query_tools_reject_unknown_message_type(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = MagicMock()
            mock_poly.query_sessions = AsyncMock(return_value=[])
            mock_poly.search_session_hits = AsyncMock(return_value=[])
            mock_get_polylogue.return_value = mock_poly

            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="hello",
                message_type="summmary",
            )

        body = json.loads(result)
        assert body["is_error"] is True
        assert body["tool"] == "search"
        assert body["code"] == "internal_error"
        assert "internal error" in body["message"]
        mock_poly.search_session_hits.assert_not_called()
        mock_poly.query_sessions.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_sessions_returns_native_session_title(
        self, tmp_path: Path, mcp_server: MCPServerUnderTest
    ) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(
            archive_root,
            provider=Provider.GEMINI,
            native_id="gemini-20250422-1234",
            title="Project Plan: Please review the attached project plan.",
            text="Please review the attached project plan.",
        )

        with _archive_config(archive_root):
            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["list_sessions"].fn,
                origin="aistudio-drive",
                limit=10,
            )

        payload = json.loads(raw)
        assert payload["items"][0]["id"] == session_id
        assert payload["items"][0]["origin"] == "aistudio-drive"
        assert payload["items"][0]["title"] == "Project Plan: Please review the attached project plan."

    @pytest.mark.asyncio
    async def test_search_hit_exposes_native_message_match_evidence(
        self, tmp_path: Path, mcp_server: MCPServerUnderTest
    ) -> None:
        # PROD GAP: the archive search-hit payload (archive_search_hit_payload)
        # only emits message-surface matches; it does not expose an attachment
        # match surface / identity-evidence snippet the way the legacy attachment
        # search lane did. This test pins the message-surface contract that the
        # archive path *does* honor (message_id + snippet + dialogue lane).
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(
            archive_root,
            native_id="attachment-evidence",
            text="reference to attachmentidentity inside message body",
        )

        with _archive_config(archive_root):
            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="attachmentidentity",
                limit=10,
            )

        payload = json.loads(raw)
        hits = payload["hits"]
        assert hits[0]["session"]["id"] == session_id
        assert hits[0]["match"]["match_surface"] == "message"
        assert hits[0]["match"]["retrieval_lane"] == "dialogue"
        assert hits[0]["match"]["message_id"]
        assert "attachmentidentity" in hits[0]["match"]["snippet"]

    @pytest.mark.asyncio
    async def test_neighbor_candidates_exposes_candidate_reasons(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.neighbor_candidates = AsyncMock(return_value=[_make_neighbor_candidate()])
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["neighbor_candidates"].fn,
                id="target",
                origin="codex-session",
                limit=3,
            )

        payload = json.loads(raw)
        # 819: bounded envelope rather than bare array.
        assert payload["limit"] == 3
        assert payload["total"] == 1
        items = payload["items"]
        assert items[0]["session"]["id"] == "candidate"
        assert items[0]["reasons"][0]["kind"] == "nearby_time"
        assert items[0]["source_session_id"] == "target"
        mock_poly.neighbor_candidates.assert_awaited_once_with(
            session_id="target",
            query=None,
            provider="codex",
            limit=3,
            window_hours=24,
        )


class TestArchiveTools:
    @pytest.mark.asyncio
    async def test_archive_list_sessions_returns_summary_envelope(self, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.archive_list_sessions = AsyncMock(
                return_value=[
                    ArchiveSessionSummary(
                        session_id="codex-session:native-1",
                        native_id="native-1",
                        origin="codex-session",
                        provider=Provider.CODEX,
                        title="Copied",
                        created_at=None,
                        updated_at=None,
                        message_count=2,
                        word_count=7,
                        tags=("v1",),
                    )
                ]
            )
            mock_poly.archive_count_sessions = AsyncMock(return_value=1)
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["archive_list_sessions"].fn,
                origin="codex-session",
                exclude_origin="chatgpt-export",
                tag="v1,review",
                exclude_tag="archived",
                repo="polylogue",
                has_type="tool_use,thinking",
                has_tool_use=True,
                has_thinking=True,
                tool="read",
                exclude_tool="write",
                action="file_read",
                exclude_action="file_write",
                action_sequence="file_read,shell",
                action_text="README.md",
                referenced_path="README.md,pyproject.toml",
                cwd_prefix="/realm/project/polylogue",
                typed_only=True,
                message_type="tool_use",
                title="Copied",
                max_words=100,
                limit=5,
            )

        payload = json.loads(raw)
        assert payload["total"] == 1
        assert payload["items"][0]["session_id"] == "codex-session:native-1"
        assert payload["items"][0]["source"] == "codex-session"
        assert payload["items"][0]["origin"] == "codex-session"
        mock_poly.archive_list_sessions.assert_awaited_once_with(
            origin="codex-session",
            excluded_origins=("chatgpt-export",),
            tags=("v1", "review"),
            excluded_tags=("archived",),
            repo_names=("polylogue",),
            has_types=("tool_use", "thinking"),
            has_tool_use=True,
            has_thinking=True,
            has_paste=False,
            tool_terms=("read",),
            excluded_tool_terms=("write",),
            action_terms=("file_read",),
            excluded_action_terms=("file_write",),
            action_sequence=("file_read", "shell"),
            action_text_terms=("README.md",),
            referenced_paths=("README.md", "pyproject.toml"),
            cwd_prefix="/realm/project/polylogue",
            typed_only=True,
            message_type="tool_use",
            title="Copied",
            min_messages=None,
            max_messages=None,
            min_words=None,
            max_words=100,
            since=None,
            until=None,
            limit=5,
            offset=0,
            sample=False,
        )
        mock_poly.archive_count_sessions.assert_awaited_once_with(
            origin="codex-session",
            excluded_origins=("chatgpt-export",),
            tags=("v1", "review"),
            excluded_tags=("archived",),
            repo_names=("polylogue",),
            has_types=("tool_use", "thinking"),
            has_tool_use=True,
            has_thinking=True,
            has_paste=False,
            tool_terms=("read",),
            excluded_tool_terms=("write",),
            action_terms=("file_read",),
            excluded_action_terms=("file_write",),
            action_sequence=("file_read", "shell"),
            action_text_terms=("README.md",),
            referenced_paths=("README.md", "pyproject.toml"),
            cwd_prefix="/realm/project/polylogue",
            typed_only=True,
            message_type="tool_use",
            title="Copied",
            min_messages=None,
            max_messages=None,
            min_words=None,
            max_words=100,
            since=None,
            until=None,
        )

    @pytest.mark.asyncio
    async def test_archive_get_session_returns_full_envelope(self, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.storage.sqlite.archive_tiers.write import (
            ArchiveBlockRow,
            ArchiveMessageRow,
            ArchiveSessionEnvelope,
        )

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.archive_get_session = AsyncMock(
                return_value=ArchiveSessionEnvelope(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    title="Copied",
                    active_leaf_message_id="codex-session:native-1:m1",
                    messages=(
                        ArchiveMessageRow(
                            message_id="codex-session:native-1:m1",
                            native_id="m1",
                            role="user",
                            position=0,
                            variant_index=0,
                            is_active_path=True,
                            is_active_leaf=True,
                            blocks=(
                                ArchiveBlockRow(
                                    block_id="codex-session:native-1:m1:0",
                                    message_id="codex-session:native-1:m1",
                                    block_type="text",
                                    text="hello from mcp",
                                ),
                            ),
                        ),
                    ),
                )
            )
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["archive_get_session"].fn,
                session_id="codex-session:native-1",
            )

        payload = json.loads(raw)
        assert payload["session_id"] == "codex-session:native-1"
        assert payload["source"] == "codex-session"
        assert payload["origin"] == "codex-session"
        assert payload["messages"][0]["blocks"][0]["text"] == "hello from mcp"


class TestGetSessionTool:
    def test_get_returns_session(self, tmp_path: Path, mcp_server: MCPServerUnderTest) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(
            archive_root,
            native_id="conv-123",
            title="Test Conv",
            extra=(("conv-123-m2", "reply"),),
        )

        with _archive_config(archive_root):
            result = invoke_surface(mcp_server._tool_manager._tools["get_session"].fn, id=session_id)

        conv = json.loads(result)
        assert conv["id"] == session_id
        assert conv["message_count"] == 2
        assert "messages" not in conv

    def test_get_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_summary = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["get_session"].fn, id="nonexistent")

        parsed = json.loads(result)
        assert "message" in parsed
        assert "not found" in parsed["message"].lower()

    def test_get_messages_returns_full_messages(self, tmp_path: Path, mcp_server: MCPServerUnderTest) -> None:
        long_text = "A" * 2000
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root, native_id="long", text=long_text)

        with _archive_config(archive_root):
            result = invoke_surface(mcp_server._tool_manager._tools["get_messages"].fn, session_id=session_id)

        assert json.loads(result)["messages"][0]["text"] == long_text

    def test_get_messages_rejects_unknown_message_type(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["get_messages"].fn,
                session_id="test:long",
                message_type="summmary",
            )

        body = json.loads(result)
        assert body["is_error"] is True
        assert body["tool"] == "get_messages"
        assert body["code"] == "internal_error"
        mock_poly.get_messages_paginated.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_with_nonexistent_id(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_summary = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["get_session"].fn, id="nonexistent-id-xyz"
            )

        assert isinstance(json.loads(result), dict)


class TestReadViewProfilesTool:
    def test_list_read_view_profiles_uses_facade_contract(self, mcp_server: MCPServerUnderTest) -> None:
        profiles = read_view_profile_payloads()
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.list_read_view_profiles = AsyncMock(return_value=profiles)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["list_read_view_profiles"].fn)

        body = json.loads(result)
        assert body["read_views"] == profiles
        assert body["total"] == len(profiles)
        assert {profile["view_id"] for profile in body["read_views"]} >= {"raw", "summary", "recovery"}


class TestQueryExplanationTool:
    def test_explain_query_expression_uses_facade_contract(self, mcp_server: MCPServerUnderTest) -> None:
        explanation = {
            "source_text": "sessions where repo:polylogue",
            "selected_units": ["sessions"],
            "execution_legs": ["sql"],
        }
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.explain_query_expression = AsyncMock(return_value=explanation)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["explain_query_expression"].fn,
                expression="sessions where repo:polylogue",
            )

        body = json.loads(result)
        assert body["query_explanation"] == explanation
        mock_poly.explain_query_expression.assert_awaited_once_with("sessions where repo:polylogue")


class TestQueryCompletionsTool:
    def test_query_completions_uses_facade_contract(self, mcp_server: MCPServerUnderTest) -> None:
        completions = {
            "kind": "field",
            "incomplete": "d",
            "unit": None,
            "field": None,
            "candidates": [
                {
                    "value": "date",
                    "insert": "date ",
                    "replace_start": None,
                    "replace_end": None,
                    "display": "date ",
                    "kind": "query-date-field",
                    "group": "query readable fields",
                    "description": "Date field.",
                    "score": 1.0,
                    "source": "DATE_QUERY_FIELD_REGISTRY",
                    "stale": False,
                    "danger": False,
                    "unsupported_reason": None,
                    "preview_command": None,
                }
            ],
        }
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.query_completions = AsyncMock(return_value=completions)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["query_completions"].fn,
                kind="field",
                incomplete="d",
            )

        body = json.loads(result)
        assert body["query_completions"] == completions
        mock_poly.query_completions.assert_awaited_once_with("field", incomplete="d", unit=None, field=None)


class TestInsightTools:
    @pytest.mark.asyncio
    async def test_session_profile_tool_uses_archive_insight_contract(self, mcp_server: MCPServerUnderTest) -> None:
        insight = SessionProfileInsight(
            session_id="conv-1",
            logical_session_id="conv-1",
            source_name="claude-code",
            title="Profiled Session",
            semantic_tier="merged",
            provenance=_provenance(),
            evidence=SessionEvidencePayload(canonical_session_date="2026-03-24", message_count=2),
            inference_provenance=_inference_provenance(),
            inference=SessionInferencePayload(engaged_duration_ms=120000),
        )
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_profile_insight = AsyncMock(return_value=insight)
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_profile"].fn,
                session_id="conv-1",
            )

        payload = json.loads(raw)
        assert payload["insight_kind"] == "session_profile"
        assert payload["session_id"] == "conv-1"

    @pytest.mark.asyncio
    async def test_insight_list_tools_use_archive_queries(self, mcp_server: MCPServerUnderTest) -> None:
        profile = SessionProfileInsight(
            session_id="conv-1",
            logical_session_id="conv-1",
            source_name="claude-code",
            title="Profiled Session",
            semantic_tier="merged",
            provenance=_provenance(),
            evidence=SessionEvidencePayload(
                canonical_session_date="2026-03-24",
                message_count=2,
                cwd_paths=("/realm/project/polylogue",),
                terminal_state_evidence={"message_id": "u1"},
            ),
            inference_provenance=_inference_provenance(),
            inference=SessionInferencePayload(
                engaged_duration_ms=120000,
                workflow_shape="agentic_loop",
                workflow_shape_confidence=0.86,
                terminal_state="question_left",
                terminal_state_confidence=0.72,
            ),
            enrichment_provenance=_enrichment_provenance(),
            enrichment=SessionEnrichmentPayload(
                intent_summary="Plan the refactor",
                outcome_summary="Ran tests",
                confidence=0.72,
                support_level=ConfidenceBand.MODERATE,
            ),
        )
        work_event = SessionWorkEventInsight(
            event_id="evt-1",
            session_id="conv-1",
            source_name="claude-code",
            event_index=0,
            provenance=_provenance(),
            inference_provenance=_inference_provenance(),
            evidence=WorkEventEvidencePayload(
                start_index=0, end_index=1, file_paths=("/workspace/polylogue/README.md",)
            ),
            inference=WorkEventInferencePayload(
                heuristic_label="implementation", summary="editing files", confidence=0.8
            ),
        )
        phase = SessionPhaseInsight(
            phase_id="phase-1",
            session_id="conv-1",
            source_name="claude-code",
            phase_index=0,
            provenance=_provenance(),
            inference_provenance=_inference_provenance(),
            evidence=SessionPhaseEvidencePayload(message_range=(0, 2), tool_counts={"edit": 1}),
            inference=SessionPhaseInferencePayload(confidence=0.8),
        )
        thread = ThreadInsight(
            thread_id="conv-1",
            root_id="conv-1",
            dominant_repo="polylogue",
            provenance=_provenance(),
            thread=ThreadPayload(
                session_ids=("conv-1", "conv-2"),
                session_count=2,
                confidence=1.0,
                support_level=ConfidenceBand.STRONG,
                support_signals=("explicit_lineage",),
                member_evidence=(
                    ThreadMemberEvidencePayload(
                        session_id="conv-1",
                        role="root",
                        depth=0,
                        confidence=1.0,
                        support_signals=("root_session",),
                        evidence=("session has no archived parent inside this thread",),
                    ),
                ),
            ),
        )
        tag_rollup = SessionTagRollupInsight(
            tag="provider:claude-code",
            session_count=1,
            explicit_count=0,
            auto_count=1,
            provider_breakdown={"claude-code": 1},
            repo_breakdown={"polylogue": 1},
            provenance=_provenance(),
        )
        coverage = ArchiveCoverageInsight(
            group_by="provider",
            bucket="claude-code",
            source_name="claude-code",
            session_count=1,
            message_count=2,
            user_message_count=1,
            assistant_message_count=1,
            avg_messages_per_session=2.0,
            avg_user_words=3.0,
            avg_assistant_words=4.0,
            tool_use_count=1,
            thinking_count=0,
            total_sessions_with_tools=1,
            total_sessions_with_thinking=0,
            tool_use_percentage=100.0,
            thinking_percentage=0.0,
        )
        session_cost = SessionCostInsight(
            session_id="conv-root",
            source_name="claude-code",
            title="Root Thread",
            estimate=CostEstimatePayload(
                source_name="claude-code",
                session_id="conv-root",
                model_name="claude-sonnet-4-5",
                normalized_model="claude-sonnet-4-5",
                status="exact",
                confidence=1.0,
                total_usd=1.25,
                provenance=("archive_provider_reported_cost",),
            ),
            provenance=_provenance(),
        )
        cost_rollup = CostRollupInsight(
            source_name="claude-code",
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
        debt = ArchiveDebtInsight(
            debt_name="session_insights",
            category="insights",
            maintenance_target="session_insights",
            destructive=False,
            issue_count=1,
            healthy=False,
            detail="1 pending session-insight row",
        )
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.list_session_profile_insights = AsyncMock(return_value=[profile])
            mock_poly.list_session_work_event_insights = AsyncMock(return_value=[work_event])
            mock_poly.list_session_phase_insights = AsyncMock(return_value=[phase])
            mock_poly.list_session_tag_rollup_insights = AsyncMock(return_value=[tag_rollup])
            mock_poly.list_thread_insights = AsyncMock(return_value=[thread])
            mock_poly.list_archive_coverage_insights = AsyncMock(return_value=[coverage])
            mock_poly.list_session_cost_insights = AsyncMock(return_value=[session_cost])
            mock_poly.list_cost_rollup_insights = AsyncMock(return_value=[cost_rollup])
            mock_poly.list_archive_debt_insights = AsyncMock(return_value=[debt])
            mock_get_polylogue.return_value = mock_poly

            profiles_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_profiles"].fn,
                origin="claude-code-session",
                query="profiled",
                first_message_since="2026-03-24T00:00:00+00:00",
                session_date_since="2026-03-24",
                limit=5,
                offset=2,
            )
            events_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_work_events"].fn,
                heuristic_label="implementation",
                limit=5,
            )
            phases_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_phases"].fn,
                kind="implementation",
                limit=5,
            )
            threads_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["threads"].fn,
                query="polylogue",
                limit=5,
            )
            tags_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_tag_rollups"].fn,
                origin="claude-code-session",
                limit=5,
            )
            coverage_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["archive_coverage"].fn,
                group_by="origin",
                origin="claude-code-session",
                limit=5,
            )
            costs_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["session_costs"].fn,
                origin="claude-code-session",
                status="exact",
                limit=5,
            )
            cost_rollups_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["cost_rollups"].fn,
                origin="claude-code-session",
                model="claude-sonnet-4-5",
                limit=5,
            )
            debt_raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["archive_debt"].fn,
                category="insights",
                only_actionable=True,
                limit=5,
            )

        profiles_payload = json.loads(profiles_raw)
        events_payload = json.loads(events_raw)
        phases_payload = json.loads(phases_raw)
        threads_payload = json.loads(threads_raw)
        tags_payload = json.loads(tags_raw)
        coverage_payload = json.loads(coverage_raw)
        costs_payload = json.loads(costs_raw)
        cost_rollups_payload = json.loads(cost_rollups_raw)
        debt_payload = json.loads(debt_raw)

        assert profiles_payload["total"] == 1
        assert profiles_payload["items"][0]["insight_kind"] == "session_profile"
        assert profiles_payload["items"][0]["enrichment"]["intent_summary"] == "Plan the refactor"
        assert events_payload["items"][0]["insight_kind"] == "session_work_event"
        assert phases_payload["items"][0]["insight_kind"] == "session_phase"
        assert tags_payload["items"][0]["insight_kind"] == "session_tag_rollup"
        assert threads_payload["items"][0]["insight_kind"] == "thread"
        assert threads_payload["items"][0]["thread"]["support_level"] == "strong"
        assert threads_payload["items"][0]["thread"]["member_evidence"][0]["role"] == "root"
        assert coverage_payload["items"][0]["insight_kind"] == "archive_coverage"
        assert coverage_payload["items"][0]["group_by"] == "origin"
        assert coverage_payload["items"][0]["origin"] == "claude-code-session"
        assert costs_payload["items"][0]["insight_kind"] == "session_cost"
        assert costs_payload["items"][0]["origin"] == "claude-code-session"
        assert costs_payload["items"][0]["estimate"]["status"] == "exact"
        assert costs_payload["items"][0]["estimate"]["origin"] == "claude-code-session"
        assert cost_rollups_payload["items"][0]["insight_kind"] == "cost_rollup"
        assert cost_rollups_payload["items"][0]["origin"] == "claude-code-session"
        assert cost_rollups_payload["items"][0]["total_usd"] == 1.25
        assert debt_payload["items"][0]["insight_kind"] == "archive_debt"
        assert mock_poly.list_session_profile_insights.await_args.args[0].provider == "claude-code"
        assert mock_poly.list_session_tag_rollup_insights.await_args.args[0].provider == "claude-code"
        assert mock_poly.list_archive_coverage_insights.await_args.args[0].group_by == "provider"
        assert mock_poly.list_archive_coverage_insights.await_args.args[0].provider == "claude-code"
        debt_query = mock_poly.list_archive_debt_insights.await_args.args[0]
        assert debt_query.category == "insights"
        assert debt_query.only_actionable is True


class TestStatsTool:
    @pytest.mark.parametrize(
        (
            "total_sessions",
            "total_messages",
            "origins",
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
        total_sessions: int,
        total_messages: int,
        origins: dict[str, int],
        embedded_convs: int,
        embedded_msgs: int,
        pending_convs: int,
        db_size: int,
        expected_coverage: float,
        expected_mb: float | int,
        tmp_path: Path,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        # The stats tool routes directly through ``ArchiveStore.stats()``. Pin the
        # ``MCPArchiveStatsPayload`` projection (coverage %, db_size_mb) by
        # returning a controlled ``ArchiveStats`` from the archive stats call.
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        configured = ArchiveStats(
            total_sessions=total_sessions,
            total_messages=total_messages,
            origins=origins,
            embedded_sessions=embedded_convs,
            embedded_messages=embedded_msgs,
            pending_embedding_sessions=pending_convs,
            db_size_bytes=db_size,
        )

        with (
            _archive_config(archive_root),
            patch.object(ArchiveStore, "stats", return_value=configured),
        ):
            result = invoke_surface(mcp_server._tool_manager._tools["stats"].fn)

        data = json.loads(result)
        assert data["total_sessions"] == total_sessions
        assert data["total_messages"] == total_messages
        assert data["pending_embedding_sessions"] == pending_convs
        assert data["embedding_coverage_percent"] == expected_coverage
        assert data["db_size_mb"] == expected_mb

    def test_stats_reads_archive_file_set(self, mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="stats-v1",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="stats v1",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="stats v1")],
                        )
                    ],
                )
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("stats must not open archive operations")
            result = invoke_surface(mcp_server._tool_manager._tools["stats"].fn)

        data = json.loads(result)
        assert data["total_sessions"] == 1
        assert data["total_messages"] == 1
        assert data["origins"] == {"codex-session": 1}


class TestMutationTools:
    def test_add_tag_success(self, mcp_server: MCPServerUnderTest) -> None:
        # add_tag goes through _resolve_or_error first, so the facade
        # must report the session as found before the mutation runs.
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
            mock_poly.add_tag = AsyncMock(return_value=TagMutationResult(outcome="added"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["add_tag"].fn, session_id="test:conv-123", tag="important"
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["session_id"] == "test:conv-123"
        assert parsed["tag"] == "important"
        mock_poly.add_tag.assert_awaited_once_with("test:conv-123", "important")

    def test_add_tag_error(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
            mock_poly.add_tag = AsyncMock(side_effect=ValueError("Invalid tag"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["add_tag"].fn, session_id="test:conv-123", tag="invalid"
            )

        body = json.loads(result)
        assert body["is_error"] is True
        assert body["tool"] == "add_tag"
        assert body["code"] == "internal_error"
        mock_poly.add_tag.assert_awaited_once_with("test:conv-123", "invalid")

    def test_remove_tag_success(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
            mock_poly.remove_tag = AsyncMock(return_value=TagMutationResult(outcome="removed"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["remove_tag"].fn, session_id="test:conv-123", tag="important"
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["session_id"] == "test:conv-123"
        assert parsed["tag"] == "important"
        mock_poly.remove_tag.assert_awaited_once_with("test:conv-123", "important")

    def test_remove_tag_error(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock(resolved_id="test:conv-123")
            mock_poly.remove_tag = AsyncMock(side_effect=RuntimeError("Backend error"))
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["remove_tag"].fn,
                session_id="test:conv-123",
                tag="important",
            )

        body = json.loads(result)
        assert body["is_error"] is True
        assert body["tool"] == "remove_tag"
        assert body["code"] == "internal_error"
        mock_poly.remove_tag.assert_awaited_once_with("test:conv-123", "important")

    def test_bulk_tag_sessions_applies_every_tag_to_every_session(
        self,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        from polylogue.surfaces.payloads import BulkTagMutationResult

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.bulk_tag_sessions = AsyncMock(
                return_value=BulkTagMutationResult(
                    session_count=2,
                    tag_count=2,
                    affected_count=4,
                    skipped_count=0,
                )
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["bulk_tag_sessions"].fn,
                session_ids=["conv-1", "conv-2"],
                tags=["review", "important"],
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["session_count"] == 2
        assert parsed["tag_count"] == 2
        assert parsed["affected_count"] == 4
        mock_poly.bulk_tag_sessions.assert_called_once_with(["conv-1", "conv-2"], ["review", "important"])

    def test_bulk_tag_sessions_rejects_empty_inputs(self, mcp_server: MCPServerUnderTest) -> None:
        result = invoke_surface(
            mcp_server._tool_manager._tools["bulk_tag_sessions"].fn,
            session_ids=[],
            tags=["review"],
        )

        assert "requires at least one session_id" in json.loads(result)["message"]

    def test_list_tags_returns_counts(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.list_tags = AsyncMock(return_value={"bug": 3, "feature": 5, "urgent": 1})
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["list_tags"].fn)

        assert json.loads(result) == {"bug": 3, "feature": 5, "urgent": 1}

    def test_list_tags_with_origin(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.list_tags = AsyncMock(return_value={"claude-ai": 2})
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["list_tags"].fn, origin="claude-ai-export")

        assert json.loads(result) == {"claude-ai": 2}
        mock_poly.list_tags.assert_called_once_with(origin="claude-ai-export")

    def test_get_metadata_success(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_metadata = AsyncMock(return_value={"key": "value", "count": 42})
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["get_metadata"].fn, session_id="test:conv-123")

        assert json.loads(result) == {"key": "value", "count": 42}

    def test_set_metadata_string_value(self, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.surfaces.payloads import MetadataMutationResult

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.set_metadata = AsyncMock(
                return_value=MetadataMutationResult(outcome="set", session_id="test:conv-123", key="author")
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["set_metadata"].fn,
                session_id="test:conv-123",
                key="author",
                value="john",
            )

        assert json.loads(result)["status"] == "ok"
        mock_poly.set_metadata.assert_called_once_with("test:conv-123", "author", "john")

    def test_set_metadata_json_value(self, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.surfaces.payloads import MetadataMutationResult

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.set_metadata = AsyncMock(
                return_value=MetadataMutationResult(outcome="set", session_id="test:conv-123", key="config")
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["set_metadata"].fn,
                session_id="test:conv-123",
                key="config",
                value='{"nested": true}',
            )

        assert json.loads(result)["status"] == "ok"
        mock_poly.set_metadata.assert_called_once_with("test:conv-123", "config", "{'nested': True}")

    def test_set_metadata_rejects_empty_key(self, mcp_server: MCPServerUnderTest) -> None:
        """Empty metadata key returns structured error, not exception."""
        result = invoke_surface(
            mcp_server._tool_manager._tools["set_metadata"].fn,
            session_id="test:conv-123",
            key="",
            value="some value",
        )
        parsed = json.loads(result)
        assert parsed["is_error"] is True
        assert "empty" in parsed["message"].lower()

    def test_set_metadata_rejects_whitespace_key(self, mcp_server: MCPServerUnderTest) -> None:
        """Whitespace-only metadata key returns structured error."""
        result = invoke_surface(
            mcp_server._tool_manager._tools["set_metadata"].fn,
            session_id="test:conv-123",
            key="   ",
            value="some value",
        )
        parsed = json.loads(result)
        assert parsed["is_error"] is True
        assert "empty" in parsed["message"].lower()

    def test_set_metadata_rejects_overlong_key(self, mcp_server: MCPServerUnderTest) -> None:
        """Metadata key exceeding 200 characters returns structured error."""
        result = invoke_surface(
            mcp_server._tool_manager._tools["set_metadata"].fn,
            session_id="test:conv-123",
            key="k" * 201,
            value="v",
        )
        parsed = json.loads(result)
        assert parsed["is_error"] is True
        assert "200" in parsed["message"]

    def test_delete_metadata_success(self, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.surfaces.payloads import MetadataMutationResult

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.delete_metadata = AsyncMock(
                return_value=MetadataMutationResult(outcome="deleted", session_id="test:conv-123", key="author")
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_metadata"].fn,
                session_id="test:conv-123",
                key="author",
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["key"] == "author"
        mock_poly.delete_metadata.assert_awaited_once_with("test:conv-123", "author")

    def test_delete_requires_confirm(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_session"].fn,
                session_id="test:conv-123",
                confirm=False,
            )

        parsed = json.loads(result)
        assert "confirm=true" in parsed["message"]
        mock_poly.delete_session.assert_not_called()

    def test_delete_with_confirm(self, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.surfaces.payloads import DeleteSessionResult

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.delete_session_safe = AsyncMock(
                return_value=DeleteSessionResult(outcome="deleted", session_id="test:conv-123")
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_session"].fn,
                session_id="test:conv-123",
                confirm=True,
            )

        assert json.loads(result)["status"] == "deleted"
        mock_poly.delete_session_safe.assert_awaited_once_with("test:conv-123")

    def test_delete_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        # delete_session_safe returns outcome="not_found" idempotently,
        # whether the id never existed or a concurrent delete already removed
        # it. The MCP tool surfaces status="not_found" instead of raising.
        from polylogue.surfaces.payloads import DeleteSessionResult

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.delete_session_safe = AsyncMock(
                return_value=DeleteSessionResult(
                    outcome="not_found",
                    session_id="nonexistent",
                    detail="session_not_found",
                )
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["delete_session"].fn,
                session_id="nonexistent",
                confirm=True,
            )

        assert json.loads(result)["status"] == "not_found"

    def test_summary_returns_metadata(self, tmp_path: Path, mcp_server: MCPServerUnderTest) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(
            archive_root,
            provider=Provider.CHATGPT,
            native_id="conv-123",
            title="Test Conv",
            extra=tuple((f"conv-123-m{i}", f"reply {i}") for i in range(2, 6)),
        )

        with _archive_config(archive_root):
            result = invoke_surface(mcp_server._tool_manager._tools["get_session_summary"].fn, id=session_id)

        parsed = json.loads(result)
        assert parsed["id"] == session_id
        assert parsed["origin"] == "chatgpt-export"
        assert parsed["title"] == "Test Conv"
        assert parsed["message_count"] == 5

    def test_summary_not_found(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_summary = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["get_session_summary"].fn, id="nonexistent")

        assert "not found" in json.loads(result)["message"].lower()

    def test_session_tree_returns_envelope(self, simple_session: Session, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session_tree = AsyncMock(return_value=[simple_session])
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["get_session_tree"].fn, session_id="test:conv-123")

        # 819: bounded envelope rather than bare array.
        parsed = json.loads(result)
        assert parsed["total"] == 1
        items = parsed["items"]
        assert isinstance(items, list)
        assert items[0]["id"] == "test:conv-123"

    @pytest.mark.parametrize(
        ("group_by", "expected"),
        [
            ("provider", {"chatgpt": 10, "claude-ai": 5}),
            ("month", {"2024-01": 15, "2024-02": 20}),
        ],
    )
    def test_stats_by_group(self, group_by: str, expected: dict[str, int], mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = MagicMock()
            mock_poly.get_stats_by = AsyncMock(return_value=expected)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["get_stats_by"].fn, group_by=group_by)

        assert json.loads(result) == expected

    def test_health_check_success(self, mcp_server: MCPServerUnderTest) -> None:
        mock_check = OutcomeCheck("database", OutcomeStatus.OK, summary="All good", count=100)

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
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.rebuild_index = AsyncMock(return_value=True)
            mock_poly.get_index_status = AsyncMock(return_value={"exists": True, "count": 500})
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["rebuild_index"].fn)

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["index_exists"] is True
        assert parsed["indexed_messages"] == 500

    def test_update_index_success(self, mcp_server: MCPServerUnderTest) -> None:
        with patch(
            "polylogue.api.archive.PolylogueArchiveMixin.update_index",
            new=AsyncMock(return_value=True),
        ):
            result = invoke_surface(
                mcp_server._tool_manager._tools["update_index"].fn,
                session_ids=["test:conv-1", "test:conv-2"],
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["session_count"] == 2

    def test_rebuild_session_insights_success(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.rebuild_insights = AsyncMock(
                return_value=MagicMock(
                    to_dict=MagicMock(return_value={"profiles": 2, "work_events": 3, "phases": 1}),
                    total=MagicMock(return_value=6),
                )
            )
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._tool_manager._tools["rebuild_session_insights"].fn,
                session_ids=["conv-1", "conv-2"],
            )

        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["session_count"] == 2
        assert parsed["counts"]["profiles"] == 2
        assert parsed["total"] == 6
        mock_poly.rebuild_insights.assert_awaited_once_with(session_ids=["conv-1", "conv-2"])

    @pytest.mark.asyncio
    async def test_export_query_results_uses_shared_query_contract(
        self,
        simple_session: Session,
        mcp_server: MCPServerUnderTest,
        tmp_path: Path,
    ) -> None:
        from types import SimpleNamespace

        # export_query_results uses the archive path (spec.list), not the facade
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root):
            _seed_archive(
                archive_root,
                provider=Provider.CHATGPT,
                native_id="export-test-1",
                title="Test for export",
                text="hello evidence here",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.rendering.formatting.format_session") as mock_format,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_format.return_value = '{"exported": true}'

            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["export_query_results"].fn,
                query="hello",
                origin="chatgpt-export",
                format="json",
                limit=3,
            )

        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert parsed["format"] == "json"
        assert parsed["exports"][0]["session_id"]
        assert parsed["exports"][0]["content"] == '{"exported": true}'


def test_mcp_search_params_match_query_spec() -> None:
    """search tool parameters must cover the same filter axes as SessionQuerySpec (#819)."""
    from polylogue.archive.query.spec import SessionQuerySpec

    spec_fields = {f.name for f in SessionQuerySpec.__dataclass_fields__.values()}
    mcp_params = {
        "query",
        "origin",
        "origins",
        "since",
        "until",
        "tag",
        "tags",
        "limit",
        "offset",
        "has_tool_use",
        "has_thinking",
        "message_type",
        "repo",
        "cwd_prefix",
        "action_terms",
        "tool_terms",
        "title_contains",
        "referenced_path",
    }
    missing = mcp_params - spec_fields
    contract_projections: set[str] = {
        "limit",
        "offset",
        "tag",  # → tags
        "tags",  # passthrough
        "query",  # → query_terms
        "origin",  # → origins
        "origins",  # passthrough
        "repo",  # → repo_names
        "has_tool_use",  # → filter_has_tool_use
        "has_thinking",  # → filter_has_thinking
        "title_contains",  # → title (substring match)
        "action_terms",  # passthrough
        "tool_terms",  # passthrough
    }
    assert missing.issubset(contract_projections), (
        f"MCP params not in SessionQuerySpec: {missing - contract_projections}"
    )
