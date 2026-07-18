"""Tests for the read-capability gaps closed onto ``query()``/``status()``.

Three capabilities lost no equivalent MCP dispatch during the six-tool
cutover: personal-state listing (marks/annotations/saved-views/recall-packs/
workspaces/corrections/blackboard notes), postmortem/pathology reports, and
``status(scope="sources"/"embeddings")``. The underlying ``Polylogue`` facade
calls were always live and independently tested; only the ``query()``/
``status()`` dispatch was missing. Verified against a real seeded archive via
``RuntimeServices``, matching the pattern in ``test_privileged_tools.py``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.mcp.declarations.models import MCPRole
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async


def _seed_archive(archive_root: Path) -> str:
    """Write one session with searchable text; returns its canonical id."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="query-gap-contract",
                title="Query gap contract probe",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="needle query gap evidence",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle query gap evidence")],
                    )
                ],
            )
        )


@contextmanager
def _installed_runtime_services(archive_root: Path) -> Iterator[None]:
    """Install real RuntimeServices for ``archive_root``, restoring whatever was active before."""
    from polylogue.config import Config
    from polylogue.mcp import server_support
    from polylogue.services import RuntimeServices

    services = RuntimeServices(
        config=Config(archive_root=archive_root, render_root=archive_root.parent / "render", sources=[]),
    )
    try:
        original: RuntimeServices | None = server_support._get_runtime_services()
    except RuntimeError:
        original = None
    server_support._set_runtime_services(services)
    try:
        yield
    finally:
        server_support._set_runtime_services(original)


def _build_tools(role: MCPRole) -> dict[str, Callable[..., str | Awaitable[str]]]:
    from polylogue.mcp.server import build_server

    server = cast(MCPServerUnderTest, build_server(role=role))
    return {name: tool.fn for name, tool in server._tool_manager._tools.items()}


class TestPersonalStateProjections:
    @pytest.mark.asyncio
    async def test_marks_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        tools = _build_tools("write")
        query_fn, write_fn = tools["query"], tools["write"]

        with _installed_runtime_services(archive_root):
            added = json.loads(
                await invoke_surface_async(
                    write_fn, operation="add_mark", session_id=session_id, fields={"mark_type": "star"}
                )
            )
            assert added.get("is_error") is not True, added

            listed = json.loads(await invoke_surface_async(query_fn, projection="marks"))
            assert listed.get("is_error") is not True, listed
            assert listed["total"] >= 1
            assert any(item["mark_type"] == "star" and item["session_id"] == session_id for item in listed["items"])

    @pytest.mark.asyncio
    async def test_annotations_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        tools = _build_tools("write")
        query_fn, write_fn = tools["query"], tools["write"]

        with _installed_runtime_services(archive_root):
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_annotation",
                    session_id=session_id,
                    fields={"annotation_id": "ann-1", "note_text": "a note"},
                )
            )
            assert saved.get("is_error") is not True, saved

            listed = json.loads(await invoke_surface_async(query_fn, projection="annotations"))
            assert listed.get("is_error") is not True, listed
            assert any(item["annotation_id"] == "ann-1" and item["note_text"] == "a note" for item in listed["items"])

    @pytest.mark.asyncio
    async def test_saved_views_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("write")
        query_fn, write_fn = tools["query"], tools["write"]

        with _installed_runtime_services(archive_root):
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_saved_view",
                    fields={"name": "needle sessions", "query_json": json.dumps({"query": "needle"})},
                )
            )
            assert saved.get("is_error") is not True, saved
            view_id = saved["key"]

            listed = json.loads(await invoke_surface_async(query_fn, projection="saved_views"))
            assert listed.get("is_error") is not True, listed
            assert any(item["view_id"] == view_id and item["name"] == "needle sessions" for item in listed["items"])

    @pytest.mark.asyncio
    async def test_recall_packs_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        tools = _build_tools("write")
        query_fn, write_fn = tools["query"], tools["write"]

        with _installed_runtime_services(archive_root):
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_recall_pack",
                    fields={
                        "pack_id": "pack-1",
                        "label": "handoff pack",
                        "payload_json": json.dumps({"items": [{"target_type": "session", "session_id": session_id}]}),
                    },
                )
            )
            assert saved.get("is_error") is not True, saved

            listed = json.loads(await invoke_surface_async(query_fn, projection="recall_packs"))
            assert listed.get("is_error") is not True, listed
            assert any(item["pack_id"] == "pack-1" and item["label"] == "handoff pack" for item in listed["items"])

    @pytest.mark.asyncio
    async def test_workspaces_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("write")
        query_fn, write_fn = tools["query"], tools["write"]

        with _installed_runtime_services(archive_root):
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_workspace",
                    fields={"workspace_id": "ws-1", "name": "my workspace"},
                )
            )
            assert saved.get("is_error") is not True, saved

            listed = json.loads(await invoke_surface_async(query_fn, projection="workspaces"))
            assert listed.get("is_error") is not True, listed
            assert any(item["workspace_id"] == "ws-1" and item["name"] == "my workspace" for item in listed["items"])

    @pytest.mark.asyncio
    async def test_corrections_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        tools = _build_tools("write")
        query_fn, write_fn = tools["query"], tools["write"]

        with _installed_runtime_services(archive_root):
            recorded = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="record_correction",
                    session_id=session_id,
                    fields={"kind": "tag_accept", "payload": {"tag": "reviewed"}},
                )
            )
            assert recorded.get("is_error") is not True, recorded

            listed = json.loads(await invoke_surface_async(query_fn, projection="corrections"))
            assert listed.get("is_error") is not True, listed
            assert any(c["session_id"] == session_id and c["kind"] == "tag_accept" for c in listed["corrections"])

    @pytest.mark.asyncio
    async def test_blackboard_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("write")
        query_fn, write_fn = tools["query"], tools["write"]

        with _installed_runtime_services(archive_root):
            # author_kind="user" is required for the note to land as an
            # active (visible) blackboard note: the promotion gate coerces
            # any other author_kind (the "agent" default) to a candidate
            # status, which list_blackboard_notes() deliberately excludes
            # (candidates surface only through the judgment queue).
            posted = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="blackboard_post",
                    fields={
                        "kind": "finding",
                        "title": "a finding",
                        "content": "some content",
                        "author_kind": "user",
                    },
                )
            )
            assert posted.get("is_error") is not True, posted

            listed = json.loads(await invoke_surface_async(query_fn, projection="blackboard"))
            assert listed.get("is_error") is not True, listed
            assert any(item["title"] == "a finding" for item in listed["items"])

    @pytest.mark.asyncio
    async def test_personal_state_projection_rejects_continuation(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        query_fn = tools["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="marks", continuation="bogus"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_continuation"


class TestInsightProjections:
    @pytest.mark.asyncio
    async def test_postmortem_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        query_fn = tools["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="postmortem"))
            assert result.get("is_error") is not True, result
            assert "scope" in result

    @pytest.mark.asyncio
    async def test_pathologies_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        query_fn = tools["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="pathologies"))
            assert result.get("is_error") is not True, result
            assert "findings" in result

    @pytest.mark.asyncio
    async def test_abandoned_sessions_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        query_fn = tools["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="abandoned_sessions"))
            assert result.get("is_error") is not True, result
            assert "total" in result

    @pytest.mark.asyncio
    async def test_stuck_sessions_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        query_fn = tools["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="stuck_sessions"))
            assert result.get("is_error") is not True, result
            assert "items" in result

    @pytest.mark.asyncio
    async def test_insight_projection_rejects_continuation(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        query_fn = tools["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="postmortem", continuation="bogus"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_continuation"


class TestStatusSourcesAndEmbeddingsScopes:
    @pytest.mark.asyncio
    async def test_status_sources_requires_ref(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        status_fn = tools["status"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(status_fn, scope="sources"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_status_sources_with_ref_returns_freshness_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        status_fn = tools["status"]

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(status_fn, scope="sources", ref=str(tmp_path / "nonexistent-source"))
            )
            assert result.get("is_error") is not True, result
            assert result["scope"] == "sources"
            assert "sources" in result

    @pytest.mark.asyncio
    async def test_status_embeddings_scope_returns_readiness_payload(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = _build_tools("read")
        status_fn = tools["status"]

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(status_fn, scope="embeddings"))
            assert result.get("is_error") is not True, result
            assert result["scope"] == "embeddings"
            assert "embeddings" in result
            assert "component_readiness" in result["embeddings"]
