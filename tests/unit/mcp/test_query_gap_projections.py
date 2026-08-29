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
from pathlib import Path

import pytest

from polylogue.mcp.declarations.models import MCPCapabilities
from tests.infra.mcp import build_tools, installed_runtime_services, invoke_surface_async


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


def _seed_repo_filtered_archive(archive_root: Path) -> str:
    """Write two profiled sessions so repository filtering is observable."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    sessions = (
        ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="query-gap-repo-match",
            title="Query gap repo match",
            working_directories=["/realm/project/polylogue"],
            git_repository_url="https://github.com/Sinity/polylogue.git",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="repo filter match",
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="repo filter match")],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.ASSISTANT,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Bash",
                            tool_id="stuck-tool",
                            tool_input={"command": "sleep 30"},
                        )
                    ],
                ),
            ],
        ),
        ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="query-gap-repo-other",
            title="Query gap repo other",
            working_directories=["/realm/project/other"],
            git_repository_url="https://github.com/example/other.git",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="repo filter other",
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="repo filter other")],
                )
            ],
        ),
    )
    with ArchiveStore(archive_root) as archive:
        session_ids = [archive.write_parsed(session) for session in sessions]
        for session_id in session_ids:
            archive._conn.execute(
                """
                INSERT INTO session_profiles (
                    session_id, workflow_shape, workflow_shape_method, workflow_shape_confidence,
                    terminal_state, terminal_state_method, terminal_state_confidence, search_text
                ) VALUES (?, 'chat', 'fixture', 1.0, 'question_left', 'fixture', 1.0, '')
                """,
                (session_id,),
            )
        archive._conn.execute(
            """
            INSERT INTO session_latency_profiles (session_id, materialized_at, source_name, stuck_tool_count)
            VALUES (?, '2026-01-01T12:00:00+00:00', 'chatgpt', 1)
            """,
            (session_ids[0],),
        )
        archive._conn.commit()
    return session_ids[0]


def _seed_tool_episode_archive(archive_root: Path) -> str:
    """Write a matching and an out-of-scope tagged tool episode."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    sessions = (
        ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="query-gap-tool-episode-match",
            title="Query gap tool episode match",
            working_directories=["/realm/project/polylogue"],
            git_repository_url="https://github.com/Sinity/polylogue.git",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    timestamp="2026-01-01T12:00:00Z",
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Bash",
                            tool_id="tool-1",
                            tool_input={"command": "pwd"},
                        ),
                        ParsedContentBlock(
                            type=BlockType.TOOL_RESULT,
                            tool_id="tool-1",
                            text="/realm/project/polylogue",
                            is_error=True,
                        ),
                    ],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.ASSISTANT,
                    text="The command failed, so I will fix the path.",
                    blocks=[
                        ParsedContentBlock(type=BlockType.TEXT, text="The command failed, so I will fix the path.")
                    ],
                ),
            ],
        ),
        ParsedSession(
            source_name=Provider.CHATGPT,
            provider_session_id="query-gap-tool-episode-other",
            title="Query gap tool episode other",
            working_directories=["/realm/project/other"],
            git_repository_url="https://github.com/example/other.git",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    timestamp="2026-01-05T12:00:00Z",
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Bash",
                            tool_id="tool-2",
                            tool_input={"command": "pwd"},
                        ),
                        ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="tool-2", text="/realm/project/other"),
                    ],
                )
            ],
        ),
    )
    with ArchiveStore(archive_root) as archive:
        session_ids = [archive.write_parsed(session) for session in sessions]
        archive.add_user_tags((session_ids[0],), ("episode-filter",))
        archive.add_user_tags((session_ids[1],), ("other-filter",))
    return session_ids[0]


class TestPersonalStateProjections:
    @pytest.mark.asyncio
    async def test_marks_round_trip(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        tools = build_tools(MCPCapabilities(write=True))
        query_fn, write_fn = tools["query"], tools["write"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools(MCPCapabilities(write=True))
        query_fn, write_fn = tools["query"], tools["write"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools(MCPCapabilities(write=True))
        query_fn, write_fn = tools["query"], tools["write"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools(MCPCapabilities(write=True))
        query_fn, write_fn = tools["query"], tools["write"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools(MCPCapabilities(write=True))
        query_fn, write_fn = tools["query"], tools["write"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools(MCPCapabilities(write=True))
        query_fn, write_fn = tools["query"], tools["write"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools(MCPCapabilities(write=True))
        query_fn, write_fn = tools["query"], tools["write"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools()
        query_fn = tools["query"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="marks", continuation="bogus"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_continuation"


class TestInsightProjections:
    @pytest.mark.asyncio
    async def test_postmortem_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        query_fn = tools["query"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="postmortem"))
            assert result.get("is_error") is not True, result
            assert "scope" in result

    @pytest.mark.asyncio
    async def test_pathologies_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        query_fn = tools["query"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="pathologies"))
            assert result.get("is_error") is not True, result
            assert "findings" in result

    @pytest.mark.asyncio
    async def test_abandoned_sessions_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        query_fn = tools["query"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="abandoned_sessions"))
            assert result.get("is_error") is not True, result
            assert "total" in result

    @pytest.mark.asyncio
    async def test_stuck_sessions_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        query_fn = tools["query"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="stuck_sessions"))
            assert result.get("is_error") is not True, result
            assert "items" in result

    @pytest.mark.asyncio
    async def test_repo_filter_matches_session_scope_for_abandoned_and_stuck(self, tmp_path: Path) -> None:
        """A URL/path predicate would exclude the bare ``polylogue`` match."""
        archive_root = tmp_path / "archive"
        matching_session_id = _seed_repo_filtered_archive(archive_root)
        query_fn = build_tools()["query"]

        with installed_runtime_services(archive_root):
            sessions = json.loads(await invoke_surface_async(query_fn, projection="sessions", repo="polylogue"))
            all_abandoned = json.loads(await invoke_surface_async(query_fn, projection="abandoned_sessions"))
            filtered_abandoned = json.loads(
                await invoke_surface_async(query_fn, projection="abandoned_sessions", repo="polylogue")
            )
            all_stuck = json.loads(await invoke_surface_async(query_fn, projection="stuck_sessions"))
            filtered_stuck = json.loads(
                await invoke_surface_async(query_fn, projection="stuck_sessions", repo="polylogue")
            )
            missing = [
                json.loads(await invoke_surface_async(query_fn, projection=projection, repo="does-not-exist"))
                for projection in ("sessions", "abandoned_sessions", "stuck_sessions")
            ]

        session_ids = {item["id"] for item in sessions["items"]}
        abandoned_ids = {item["session_id"] for item in filtered_abandoned["items"]}
        all_abandoned_ids = {item["session_id"] for item in all_abandoned["items"]}
        stuck_ids = {item["session_id"] for item in filtered_stuck["items"]}
        all_stuck_ids = {item["session_id"] for item in all_stuck["items"]}

        assert session_ids == {matching_session_id}
        assert abandoned_ids == all_abandoned_ids & session_ids
        assert stuck_ids == all_stuck_ids & session_ids
        assert abandoned_ids == {matching_session_id}
        assert all(result.get("total") == 0 for result in missing)

    @pytest.mark.asyncio
    async def test_tool_episode_projection_forwards_all_session_filters(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        session_id = _seed_tool_episode_archive(archive_root)
        query_fn = build_tools()["query"]

        with installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    query_fn,
                    projection="tool-episodes",
                    tag="episode-filter",
                    repo="polylogue",
                    since="2025-12-31",
                    until="2026-01-02",
                )
            )

        assert result.get("is_error") is not True, result
        assert result["total"] == 1
        assert result["tool_episodes"][0]["session_id"] == session_id
        assert result["tool_episodes"][0]["followup_class"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_insight_projection_rejects_continuation(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        query_fn = tools["query"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="postmortem", continuation="bogus"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_continuation"


class TestStatusSourcesAndEmbeddingsScopes:
    @pytest.mark.asyncio
    async def test_status_sources_requires_ref(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        status_fn = tools["status"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(status_fn, scope="sources"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_status_sources_with_ref_returns_freshness_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        status_fn = tools["status"]

        with installed_runtime_services(archive_root):
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
        tools = build_tools()
        status_fn = tools["status"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(status_fn, scope="embeddings"))
            assert result.get("is_error") is not True, result
            assert result["scope"] == "embeddings"
            assert "embeddings" in result
            assert "component_readiness" in result["embeddings"]

    @pytest.mark.asyncio
    async def test_status_sinex_scope_returns_durable_publication_status(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        tools = build_tools()
        status_fn = tools["status"]

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(status_fn, scope="sinex"))

        assert result.get("is_error") is not True, result
        assert result["scope"] == "sinex"
        assert result["sinex"]["mode"] == "off"
        assert result["sinex"]["active_lag"] == 0
