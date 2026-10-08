"""Unit tests for the privileged transaction tools (write/judge/run/maintenance, t46.8.3).

These are thin adapters over the same typed owners the retired per-operation
MCP tools used (write, run, maintenance) or already used (judge) -- see
``register_cutover_privileged_tools`` in ``polylogue/mcp/server_cutover.py``.
Each is verified against a real seeded archive via ``RuntimeServices``, not
mocks, matching the pattern established in ``test_envelope_contracts.py`` and
``test_contract_evidence.py`` (query/context/explain route through the cached
``_get_polylogue()`` facade, so a real runtime service scope is required).

``build_server()`` must be called *before* entering ``installed_runtime_services``
-- it always resolves and installs its own default runtime services when not
given one explicitly, which would otherwise clobber the seeded ones.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from polylogue.mcp.declarations.models import MCPCapabilities
from tests.infra.archive_templates import seeds_off_event_loop
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import ALL_CAPABILITIES, MCPServerUnderTest, installed_runtime_services, invoke_surface_async


@seeds_off_event_loop
def _seed_archive(archive_root: Path) -> str:
    """Write one session with searchable text; returns its canonical id."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="privileged-contract",
                title="Privileged tool contract probe",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="needle privileged contract evidence",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle privileged contract evidence")],
                    )
                ],
            ),
        )


@contextmanager
def _daemon_owned_archive(archive_root: Path) -> Iterator[str]:
    """Seed ``archive_root`` and serve it through a real resident operation stack.

    Public archive mutations are daemon-owned (#5550): the facade submits a
    declared operation to ``polylogued run`` and refuses with
    ``FacadeDaemonRequiredError`` when none answers. The write, run and
    confirm-gate tests here therefore run against the production operation
    stack on the archive's own socket, so the executor they observe is the
    daemon's, reached through the same route an MCP client uses.
    """
    from polylogue.daemon.socket_path import daemon_socket_path
    from tests.infra.daemon_operations import running_daemon_operations

    seeded: list[str] = []
    with (
        patch("polylogue.daemon.api_auth.resolve_api_auth_token", return_value=None),
        running_daemon_operations(
            archive_root,
            seed_archive=lambda root: seeded.append(_seed_archive(root)),
            socket_path=daemon_socket_path(archive_root),
        ),
        installed_runtime_services(archive_root),
    ):
        yield seeded[0]


@seeds_off_event_loop
def _seed_paged_archive(archive_root: Path, *, count: int = 7) -> list[str]:
    """Write enough deterministic rows to exercise three session pages."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        return [
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id=f"session-page-{index}",
                    title=f"Session page {index}",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text=f"pagination session {index}",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"pagination session {index}")],
                        )
                    ],
                ),
            )
            for index in range(count)
        ]


class TestCapabilityGating:
    """polylogue-800m: write/judge/maintenance are independent config opt-ins, not a role ladder.

    Enabling one capability must never leak another -- that would silently
    reintroduce the retired ladder semantics.
    """

    def test_read_only_by_default_has_no_privileged_tools(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server())
        tools = set(server._tool_manager._tools)
        assert tools.isdisjoint({"write", "judge", "run", "maintenance"})

    def test_write_capability_adds_write_and_run_only(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        tools = set(server._tool_manager._tools)
        assert {"write", "run"} <= tools
        assert tools.isdisjoint({"judge", "maintenance"})

    def test_judge_capability_adds_judge_only(self) -> None:
        """Judging assertion candidates does not require write capability."""
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
        tools = set(server._tool_manager._tools)
        assert "judge" in tools
        assert tools.isdisjoint({"write", "run", "maintenance"})

    def test_maintenance_capability_adds_maintenance_only(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        tools = set(server._tool_manager._tools)
        assert "maintenance" in tools
        assert tools.isdisjoint({"write", "run", "judge"})

    def test_all_capabilities_enabled_has_every_privileged_tool(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(capabilities=ALL_CAPABILITIES))
        tools = set(server._tool_manager._tools)
        assert {"write", "run", "judge", "maintenance"} <= tools


class TestWriteTool:
    @pytest.mark.asyncio
    async def test_add_tag_then_remove_tag_round_trips_against_real_archive(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as session_id:
            added = json.loads(
                await invoke_surface_async(write_fn, operation="add_tag", session_id=session_id, tag="reviewed")
            )
            assert added.get("is_error") is not True, added
            assert added["outcome"] == "added"

            removed = json.loads(
                await invoke_surface_async(
                    write_fn, operation="remove_tag", session_id=session_id, tag="reviewed", confirm=True
                )
            )
            assert removed.get("is_error") is not True, removed
            assert removed["outcome"] == "removed"

    @pytest.mark.asyncio
    async def test_missing_required_argument_returns_invalid_argument_envelope(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="add_tag", tag="reviewed"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_operation_specific_field_is_read_from_fields_dict(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as session_id:
            result = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="add_mark",
                    session_id=session_id,
                    fields={"mark_type": "star"},
                )
            )
            assert result.get("is_error") is not True, result
            assert result["outcome"] == "added"

    @pytest.mark.asyncio
    async def test_add_mark_without_mark_type_field_returns_invalid_argument(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="add_mark", session_id=session_id))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_invalid_argument_envelope(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="not_a_real_operation"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_delete_session_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="delete_session", session_id=session_id))
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_clear_corrections_without_confirm_is_refused(self, tmp_path: Path) -> None:
        """Anti-vacuity: the guard must run before durable assertion deletion."""
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(write_fn, operation="clear_corrections", session_id=session_id)
            )
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_save_and_delete_saved_view_round_trips(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as _session_id:
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_saved_view",
                    fields={"name": "needle sessions", "query_json": json.dumps({"query": "needle"})},
                )
            )
            assert saved.get("is_error") is not True, saved
            view_id = saved["key"]

            deleted = json.loads(
                await invoke_surface_async(
                    write_fn, operation="delete_saved_view", fields={"view_id": view_id}, confirm=True
                )
            )
            assert deleted.get("is_error") is not True, deleted
            assert deleted["status"] == "deleted"


class TestWriteToolConfirmGates:
    """polylogue-jn40: every destructive ``write`` operation must fail closed.

    Mirrors ``TestWriteTool.test_delete_session_without_confirm_is_refused``
    for the sibling destructive operations that previously had no gate at
    all: ``remove_tag`` is covered directly in ``TestWriteTool`` (round trip
    now passes ``confirm=True``); the remainder are covered here, each with
    a refusal case (asserting the underlying state is unchanged) and a
    ``confirm=True`` success case.
    """

    @pytest.mark.asyncio
    async def test_remove_tag_without_confirm_is_refused_and_tag_survives(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as session_id:
            added = json.loads(
                await invoke_surface_async(write_fn, operation="add_tag", session_id=session_id, tag="reviewed")
            )
            assert added.get("is_error") is not True, added

            refused = json.loads(
                await invoke_surface_async(write_fn, operation="remove_tag", session_id=session_id, tag="reviewed")
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

            # Prove the tag actually survived the refused call: a confirmed
            # removal afterwards still finds it present ("removed", not
            # "not_found").
            removed = json.loads(
                await invoke_surface_async(
                    write_fn, operation="remove_tag", session_id=session_id, tag="reviewed", confirm=True
                )
            )
            assert removed.get("is_error") is not True, removed
            assert removed["outcome"] == "removed"

    @pytest.mark.asyncio
    async def test_remove_mark_without_confirm_is_refused_and_mark_survives(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as session_id:
            added = json.loads(
                await invoke_surface_async(
                    write_fn, operation="add_mark", session_id=session_id, fields={"mark_type": "star"}
                )
            )
            assert added.get("is_error") is not True, added

            refused = json.loads(
                await invoke_surface_async(
                    write_fn, operation="remove_mark", session_id=session_id, fields={"mark_type": "star"}
                )
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

            removed = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="remove_mark",
                    session_id=session_id,
                    fields={"mark_type": "star"},
                    confirm=True,
                )
            )
            assert removed.get("is_error") is not True, removed
            assert removed["outcome"] == "removed"

    @pytest.mark.asyncio
    async def test_delete_metadata_without_confirm_is_refused_and_key_survives(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as session_id:
            set_result = json.loads(
                await invoke_surface_async(
                    write_fn, operation="set_metadata", session_id=session_id, key="note", value="keep"
                )
            )
            assert set_result.get("is_error") is not True, set_result

            refused = json.loads(
                await invoke_surface_async(write_fn, operation="delete_metadata", session_id=session_id, key="note")
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

            deleted = json.loads(
                await invoke_surface_async(
                    write_fn, operation="delete_metadata", session_id=session_id, key="note", confirm=True
                )
            )
            assert deleted.get("is_error") is not True, deleted
            assert deleted["status"] == "ok"

    @pytest.mark.asyncio
    async def test_delete_annotation_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as session_id:
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_annotation",
                    session_id=session_id,
                    fields={"annotation_id": "note-1", "note_text": "a durable note"},
                )
            )
            assert saved.get("is_error") is not True, saved

            refused = json.loads(
                await invoke_surface_async(write_fn, operation="delete_annotation", fields={"annotation_id": "note-1"})
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

            deleted = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="delete_annotation",
                    fields={"annotation_id": "note-1"},
                    confirm=True,
                )
            )
            assert deleted.get("is_error") is not True, deleted
            assert deleted["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_saved_view_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as _session_id:
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_saved_view",
                    fields={"name": "needle sessions", "query_json": json.dumps({"query": "needle"})},
                )
            )
            assert saved.get("is_error") is not True, saved
            view_id = saved["key"]

            refused = json.loads(
                await invoke_surface_async(write_fn, operation="delete_saved_view", fields={"view_id": view_id})
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_delete_recall_pack_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as _session_id:
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_recall_pack",
                    fields={
                        "pack_id": "pack-1",
                        "label": "Recall pack",
                        "payload_json": json.dumps({"items": []}),
                    },
                )
            )
            assert saved.get("is_error") is not True, saved

            refused = json.loads(
                await invoke_surface_async(write_fn, operation="delete_recall_pack", fields={"pack_id": "pack-1"})
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

            deleted = json.loads(
                await invoke_surface_async(
                    write_fn, operation="delete_recall_pack", fields={"pack_id": "pack-1"}, confirm=True
                )
            )
            assert deleted.get("is_error") is not True, deleted
            assert deleted["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_workspace_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as _session_id:
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_workspace",
                    fields={"workspace_id": "workspace-1", "name": "My workspace"},
                )
            )
            assert saved.get("is_error") is not True, saved

            refused = json.loads(
                await invoke_surface_async(
                    write_fn, operation="delete_workspace", fields={"workspace_id": "workspace-1"}
                )
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

            deleted = json.loads(
                await invoke_surface_async(
                    write_fn, operation="delete_workspace", fields={"workspace_id": "workspace-1"}, confirm=True
                )
            )
            assert deleted.get("is_error") is not True, deleted
            assert deleted["status"] == "deleted"


class TestWriteToolRoutesThroughOperationExecutor:
    """polylogue-t46.8.3: prove ``write()`` cannot bypass ``OperationExecutor``.

    t46.9 phases 1-6 (PRs #3249/#3253/#3258/#3262/#3294/#3376) routed every
    reversible mutation family through ``OperationExecutor`` at the facade
    layer (``polylogue/api/archive.py``); ``write()`` already calls those same
    facade methods for each executor-routed operation). So t46.8.3 required
    no *new* MCP-layer wiring
    -- but nothing previously proved that claim at the MCP adapter boundary
    itself: ``test_mutation_actuators.py`` proves the facade methods use the
    executor, and the round trips above in ``TestWriteTool``/
    ``TestWriteToolConfirmGates`` prove the *tool* succeeds functionally, but
    neither distinguishes "went through OperationExecutor" from "succeeded
    via some other path that happens to produce the same outcome". This class
    closes that gap directly: it patches ``OperationExecutor.execute`` --
    the sole ``apply`` gate the class's own docstring declares ("No adapter
    calls actuator.apply directly") -- to record every actuator it is
    invoked with, then asserts each write() operation drives exactly the
    actuator its census row names.

    Anti-vacuity: ``test_operation_invokes_operation_executor_execute``
    fails immediately if any ``write()`` branch is ever changed to call an
    ``ArchiveStore``/storage primitive directly instead of the facade method
    (the executor spy would simply never fire, or fire for the wrong
    actuator). ``test_executor_failure_propagates_as_error_not_swallowed``
    fails if a future refactor wraps the executor call in a blanket
    try/except that would silently swallow an executor-raised failure.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("operation", "setup", "op_kwargs", "actuator_name"),
        [
            ("add_tag", [], {"tag": "reviewed"}, "TagAddActuator"),
            (
                "remove_tag",
                [{"operation": "add_tag", "tag": "reviewed"}],
                {"tag": "reviewed", "confirm": True},
                "TagRemoveActuator",
            ),
            (
                "bulk_tag_sessions",
                [],
                {"session_ids": None, "tags": ["bulk"]},
                "BulkTagActuator",
            ),
            ("set_metadata", [], {"key": "note", "value": "keep"}, "MetadataSetActuator"),
            (
                "delete_metadata",
                [{"operation": "set_metadata", "key": "note", "value": "keep"}],
                {"key": "note", "confirm": True},
                "MetadataDeleteActuator",
            ),
            ("add_mark", [], {"fields": {"mark_type": "star"}}, "MarkAddActuator"),
            (
                "remove_mark",
                [{"operation": "add_mark", "fields": {"mark_type": "star"}}],
                {"fields": {"mark_type": "star"}, "confirm": True},
                "MarkRemoveActuator",
            ),
            (
                "save_annotation",
                [],
                {"fields": {"annotation_id": "note-1", "note_text": "a durable note"}},
                "AnnotationSaveActuator",
            ),
            (
                "delete_annotation",
                [
                    {
                        "operation": "save_annotation",
                        "fields": {"annotation_id": "note-1", "note_text": "a durable note"},
                    }
                ],
                {"fields": {"annotation_id": "note-1"}, "confirm": True},
                "AnnotationDeleteActuator",
            ),
            (
                "save_saved_view",
                [],
                {"fields": {"name": "needle sessions", "query_json": json.dumps({"query": "needle"})}},
                "SavedViewSaveActuator",
            ),
            (
                "save_recall_pack",
                [],
                {
                    "fields": {
                        "pack_id": "pack-1",
                        "label": "Recall pack",
                        "payload_json": json.dumps({"items": []}),
                    }
                },
                "RecallPackSaveActuator",
            ),
            (
                "delete_recall_pack",
                [
                    {
                        "operation": "save_recall_pack",
                        "fields": {
                            "pack_id": "pack-1",
                            "label": "Recall pack",
                            "payload_json": json.dumps({"items": []}),
                        },
                    }
                ],
                {"fields": {"pack_id": "pack-1"}, "confirm": True},
                "RecallPackDeleteActuator",
            ),
            (
                "save_workspace",
                [],
                {"fields": {"workspace_id": "workspace-1", "name": "My workspace"}},
                "WorkspaceSaveActuator",
            ),
            (
                "delete_workspace",
                [{"operation": "save_workspace", "fields": {"workspace_id": "workspace-1", "name": "My workspace"}}],
                {"fields": {"workspace_id": "workspace-1"}, "confirm": True},
                "WorkspaceDeleteActuator",
            ),
            (
                "record_correction",
                [],
                {"fields": {"kind": "tag_reject", "payload": {"tag": "todo"}}},
                "CorrectionRecordActuator",
            ),
            (
                "clear_corrections",
                [{"operation": "record_correction", "fields": {"kind": "tag_reject", "payload": {"tag": "todo"}}}],
                {"fields": {"kind": "tag_reject"}, "confirm": True},
                "CorrectionDeleteActuator",
            ),
            (
                "clear_corrections",
                [{"operation": "record_correction", "fields": {"kind": "tag_accept", "payload": {"tag": "todo"}}}],
                {"confirm": True},
                "CorrectionsClearActuator",
            ),
            (
                "blackboard_post",
                [],
                {"fields": {"kind": "finding", "title": "t46.8.3 probe", "content": "evidence"}},
                "BlackboardPostActuator",
            ),
            (
                "capture_assertion_candidate",
                [],
                {
                    "fields": {
                        "body_text": "MCP candidate",
                        "author_ref": "agent:mcp-candidate",
                        "kind": "lesson",
                    }
                },
                "CaptureAssertionCandidateActuator",
            ),
            ("delete_session", [], {"confirm": True}, "SessionDeleteActuator"),
        ],
    )
    async def test_operation_invokes_operation_executor_execute(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
        setup: list[dict[str, object]],
        op_kwargs: dict[str, object],
        actuator_name: str,
    ) -> None:
        from polylogue.mcp.server import build_server
        from polylogue.operations.mutation_transaction import OperationExecutor

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _daemon_owned_archive(archive_root) as session_id:
            # bulk_tag_sessions needs a real session_ids list resolved at test
            # time (the fixture only knows its id after seeding).
            if op_kwargs.get("session_ids", "__unset__") is None:
                op_kwargs = {**op_kwargs, "session_ids": [session_id]}

            for setup_call in setup:
                setup_result = json.loads(await invoke_surface_async(write_fn, session_id=session_id, **setup_call))
                assert setup_result.get("is_error") is not True, setup_result
            if operation == "delete_session":
                # A delete applies only the caller's own prepared preview.
                prepared = json.loads(
                    await invoke_surface_async(write_fn, operation="prepare_delete_session", session_id=session_id)
                )
                assert prepared.get("outcome") == "prepared", prepared
                op_kwargs = {**op_kwargs, "fields": {"preview_ref": prepared["preview_ref"]}}

            captured: list[str] = []
            original_execute = OperationExecutor.execute

            def spy(
                self: OperationExecutor,
                actuator: object,
                plan: object,
                authorization: object,
                args: object,
                _original: object = original_execute,
            ) -> object:
                captured.append(type(actuator).__name__)
                return _original(self, actuator, plan, authorization, args)  # type: ignore[operator]

            monkeypatch.setattr(OperationExecutor, "execute", spy)

            call_kwargs: dict[str, object] = dict(op_kwargs)
            session_less_operations = (
                "bulk_tag_sessions",
                "save_saved_view",
                "delete_saved_view",
                "save_recall_pack",
                "delete_recall_pack",
                "save_workspace",
                "delete_workspace",
                "blackboard_post",
                "capture_assertion_candidate",
            )
            if "session_id" not in call_kwargs and operation not in session_less_operations:
                call_kwargs["session_id"] = session_id

            result = json.loads(await invoke_surface_async(write_fn, operation=operation, **call_kwargs))

        assert result.get("is_error") is not True, result
        assert captured == [actuator_name], (
            f"write(operation={operation!r}) invoked executor with actuators {captured}, expected [{actuator_name}]"
        )

    @pytest.mark.asyncio
    async def test_executor_failure_propagates_as_error_not_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anti-vacuity companion: if OperationExecutor.execute raises, write()
        must surface that failure (proving the call sits on the real path)
        instead of returning a fabricated success -- which is what would
        happen if some parallel non-executor code path silently produced the
        response instead.
        """
        from polylogue.mcp.server import build_server
        from polylogue.operations.mutation_transaction import OperationExecutor

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        def boom(
            self: OperationExecutor, actuator: object, plan: object, authorization: object, args: object
        ) -> object:
            raise RuntimeError("t46.8.3-bypass-proof: executor forcibly disabled")

        monkeypatch.setattr(OperationExecutor, "execute", boom)

        with _daemon_owned_archive(archive_root) as session_id:
            result = json.loads(
                await invoke_surface_async(write_fn, operation="add_tag", session_id=session_id, tag="reviewed")
            )

        # The generic MCP exception translator (server_support._exception_to_error_json)
        # deliberately does not echo raw exception text into client-visible
        # payloads, only the exception type name -- so the proof here is that
        # the raise actually reached the tool boundary (an "internal_error"
        # envelope naming RuntimeError) rather than the call quietly reporting
        # success, which is what a bypassing/duplicated non-executor code path
        # would do.
        assert result.get("is_error") is True, result
        assert result.get("code") == "internal_error", result
        assert result.get("detail") == "RuntimeError", result

    @pytest.mark.asyncio
    async def test_capture_candidate_cannot_bypass_executor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A disabled executor must fail the real MCP candidate route.

        Removing the facade's executor dispatch and restoring the direct write
        would make this route return a candidate instead of an internal error.
        """

        from polylogue.mcp.server import build_server
        from polylogue.operations.mutation_transaction import OperationExecutor

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        def boom(
            self: OperationExecutor, actuator: object, plan: object, authorization: object, args: object
        ) -> object:
            raise RuntimeError("t46.9-capture-executor-bypass-proof")

        monkeypatch.setattr(OperationExecutor, "execute", boom)

        with _daemon_owned_archive(archive_root) as _session_id:
            result = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="capture_assertion_candidate",
                    fields={
                        "body_text": "must not write",
                        "author_ref": "agent:mcp-candidate",
                        "kind": "lesson",
                    },
                )
            )

        assert result["code"] == "internal_error"
        assert result["detail"] == "RuntimeError"
        assert result["is_error"] is True


class TestJudgeTool:
    @pytest.mark.asyncio
    async def test_single_candidate_shorthand_builds_a_one_item_bulk_call(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock

        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
        judge_fn = server._tool_manager._tools["judge"].fn

        with installed_runtime_services(archive_root):
            with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
                from polylogue.api import Polylogue
                from polylogue.surfaces.payloads import AssertionBulkJudgmentPayload

                real_poly = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
                real_poly.judge_assertion_candidates = AsyncMock(  # type: ignore[method-assign]
                    return_value=AssertionBulkJudgmentPayload(
                        items=(), applied_count=0, idempotent_count=0, failed_count=0
                    )
                )
                mock_get_polylogue.return_value = real_poly

                single = json.loads(
                    await invoke_surface_async(
                        judge_fn,
                        candidate_ref="assertion:contract-candidate",
                        decision="accept",
                        expected_evidence_digest="a" * 64,
                    )
                )
                assert single.get("is_error") is not True, single
                real_poly.judge_assertion_candidates.assert_awaited_once()
                await_args = real_poly.judge_assertion_candidates.await_args
                assert await_args is not None
                items = await_args.kwargs["items"]
                assert len(items) == 1
                assert items[0].candidate_ref == "assertion:contract-candidate"
                assert items[0].decision == "accept"
                # Removing the singular digest parameter or its forwarding breaks this production adapter call.
                assert items[0].expected_evidence_digest == "a" * 64

    @pytest.mark.asyncio
    async def test_actor_ref_is_not_a_caller_controllable_argument(self, tmp_path: Path) -> None:
        """polylogue-x2y9: the judge tool has no authenticated caller identity
        (37t.11), so it must not accept a caller-supplied ``actor_ref``.

        Before the fix, an MCP caller could pass ``actor_ref="user:local"``
        and have the resulting assertion recorded with
        ``author_kind="user"`` (hardcoded downstream regardless of
        ``actor_ref``) -- exactly the provenance
        ``derive_assertion_context_trust`` uses to grant assertion prose
        "operator" trust. This asserts the parameter is gone from the tool's
        signature (a plain keyword-argument call, not schema validation, so
        a stray ``**kwargs`` catch-all could not hide it) and that every
        judgment is instead pinned to the fixed, non-"user:"-prefixed
        ``_MCP_JUDGE_ACTOR_REF``.
        """
        from unittest.mock import AsyncMock

        from polylogue.mcp.server import build_server
        from polylogue.mcp.server_cutover import _MCP_JUDGE_ACTOR_REF

        assert not _MCP_JUDGE_ACTOR_REF.startswith("user:")

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
        judge_fn = server._tool_manager._tools["judge"].fn

        with pytest.raises(TypeError):
            await invoke_surface_async(
                judge_fn,
                candidate_ref="assertion:contract-candidate",
                decision="accept",
                actor_ref="user:local",
            )

        with installed_runtime_services(archive_root):
            with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
                from polylogue.api import Polylogue
                from polylogue.surfaces.payloads import AssertionBulkJudgmentPayload

                real_poly = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
                real_poly.judge_assertion_candidates = AsyncMock(  # type: ignore[method-assign]
                    return_value=AssertionBulkJudgmentPayload(
                        items=(), applied_count=0, idempotent_count=0, failed_count=0
                    )
                )
                mock_get_polylogue.return_value = real_poly

                result = json.loads(
                    await invoke_surface_async(
                        judge_fn, candidate_ref="assertion:contract-candidate", decision="accept"
                    )
                )
                assert result.get("is_error") is not True, result
                await_args = real_poly.judge_assertion_candidates.await_args
                assert await_args is not None
                items = await_args.kwargs["items"]
                assert len(items) == 1
                assert items[0].actor_ref == _MCP_JUDGE_ACTOR_REF

    @pytest.mark.asyncio
    async def test_neither_items_nor_candidate_ref_returns_invalid_argument(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
        judge_fn = server._tool_manager._tools["judge"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(judge_fn))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"


class TestRunTool:
    @pytest.mark.asyncio
    async def test_run_executes_a_saved_query_ref_and_returns_matching_sessions(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn
        run_fn = server._tool_manager._tools["run"].fn

        with _daemon_owned_archive(archive_root) as _session_id:
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_saved_view",
                    fields={"name": "needle sessions", "query_json": json.dumps({"query": "needle"})},
                )
            )
            assert saved.get("is_error") is not True, saved
            view_id = saved["key"]

            result = json.loads(await invoke_surface_async(run_fn, ref=f"saved-query:{view_id}"))
            assert result.get("is_error") is not True, result
            assert "hits" in result or "items" in result

    @pytest.mark.asyncio
    async def test_unknown_saved_view_ref_returns_not_found(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        run_fn = server._tool_manager._tools["run"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(run_fn, ref="saved-query:does-not-exist"))
            assert result.get("is_error") is True
            assert result.get("code") == "not_found"

    @pytest.mark.asyncio
    async def test_non_saved_query_ref_kind_is_rejected(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        run_fn = server._tool_manager._tools["run"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(run_fn, ref="session:not-a-saved-query"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"


class TestMaintenanceConfirmGates:
    @pytest.mark.asyncio
    async def test_rebuild_insights_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="rebuild_insights"))
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_declared_minimal_call_reaches_the_rebuild_route(self, tmp_path: Path) -> None:
        """The declaration's minimal valid call passes the confirmation gate.

        Anti-vacuity: drop ``("confirm", True)`` from the maintenance row's
        minimal arguments in ``polylogue/mcp/declarations/registry.py`` and the
        call is refused by the gate instead of reaching the daemon route.
        """
        from polylogue.mcp.declarations.registry import MCP_TOOL_DECLARATIONS
        from polylogue.mcp.server import build_server

        declaration = next(item for item in MCP_TOOL_DECLARATIONS if item.name == "maintenance")
        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, **dict(declaration.minimal_arguments)))
        assert "confirm" not in result.get("message", "").lower()
        assert result.get("code") == "daemon_required"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("session_ids", [None, [], ["codex-session:one"]])
    async def test_rebuild_insights_preserves_explicit_session_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_ids: list[str] | None
    ) -> None:
        from polylogue.mcp import server_cutover
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        calls: list[tuple[str, dict[str, object]]] = []

        async def dispatch(hooks: object, name: str, payload: dict[str, object]) -> str:
            calls.append((name, payload))
            return json.dumps({"outcome": {"state": "ok"}})

        monkeypatch.setattr(server_cutover, "_daemon_operation", dispatch)
        with installed_runtime_services(archive_root):
            await invoke_surface_async(
                server._tool_manager._tools["maintenance"].fn,
                operation="rebuild_insights",
                confirm=True,
                session_ids=session_ids,
            )
        assert calls == [("maintenance.insights.rebuild", {"session_ids": session_ids})]

    @pytest.mark.asyncio
    async def test_rebuild_insights_with_confirm_names_its_sealed_owner(self, tmp_path: Path) -> None:
        """A confirmed MCP rebuild is refused with the route the caller can take.

        MCP cannot hold sealed insight-sweep authority, so the honest answer is
        a typed ``daemon_required`` refusal naming ``polylogued run`` and the
        daemon operation. Anti-vacuity: if the facade ever executes the sweep
        in process, ``is_error`` goes false; if it leaks the internal
        transaction guard again, the code reverts to ``internal_error``.
        """

        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="rebuild_insights", confirm=True))
            assert result.get("is_error") is True
            assert result["code"] == "daemon_required"
            message = result.get("message", "")
            assert "polylogued run" in message
            assert "maintenance.insights.rebuild" in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dimension",
        ["origin", "source_family", "source_root", "since", "until", "failure_kind", "parser_version"],
    )
    async def test_maintenance_refuses_every_scope_dimension_it_cannot_apply(
        self, tmp_path: Path, dimension: str
    ) -> None:
        """An unappliable narrowing dimension is refused, never accepted and echoed back.

        polylogue-3ahdg: the maintenance envelope used to accept origin,
        source_family, source_root, time_range, failure_kind and parser_version,
        forward only session_ids, and still echo all of them in the persisted
        snapshot -- so a caller filtering by origin was told a whole-archive
        rebuild was scoped. The envelope now declares only dimensions it
        forwards, so each of these is refused at the call boundary.

        Anti-vacuity: re-add any of these as an accepted parameter that is
        merely echoed and this call stops raising. Asserting only on
        ``session_ids`` leaves that regression green.
        """

        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with installed_runtime_services(archive_root):
            with pytest.raises(TypeError, match=dimension):
                await invoke_surface_async(
                    maintenance_fn,
                    operation="rebuild_insights",
                    confirm=True,
                    **{dimension: "codex-session"},
                )


class TestQuerySessionsProjection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("origin", [None, "chatgpt-export,codex-session"])
    async def test_session_projection_forwards_offset_into_disjoint_pages(
        self, tmp_path: Path, origin: str | None
    ) -> None:
        """Each registered MCP page reaches its requested session window."""
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_paged_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())
        query_fn = server._tool_manager._tools["query"].fn

        with installed_runtime_services(archive_root):
            full = json.loads(await invoke_surface_async(query_fn, projection="sessions", limit=100, origin=origin))
            pages = [
                json.loads(
                    await invoke_surface_async(query_fn, projection="sessions", limit=2, offset=offset, origin=origin)
                )
                for offset in (0, 2, 4, 6)
            ]

        expected_ids = [item["id"] for item in full["items"]]
        page_ids = [[item["id"] for item in page["items"]] for page in pages]
        assert expected_ids == [item_id for page in page_ids for item_id in page]
        assert all(page["total"] == len(expected_ids) for page in pages)
        assert [page["offset"] for page in pages] == [0, 2, 4, 6]
        assert [page["next_offset"] for page in pages] == [2, 4, 6, None]

    @pytest.mark.asyncio
    async def test_ranked_search_finds_the_seeded_session(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())
        query_fn = server._tool_manager._tools["query"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(query_fn, expression="needle", projection="sessions", limit=10)
            )
            assert result.get("is_error") is not True, result
            assert "hits" in result
            assert result["total"] >= 1
            hit_session_ids = {hit["session"]["id"] for hit in result["hits"]}
            assert session_id in hit_session_ids

    @pytest.mark.asyncio
    async def test_exhaustive_listing_without_expression_returns_items(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())
        query_fn = server._tool_manager._tools["query"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="sessions", limit=10))
            assert result.get("is_error") is not True, result
            assert "items" in result
            assert result["total"] >= 1

    @pytest.mark.asyncio
    async def test_sessions_projection_rejects_malformed_continuation(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())
        query_fn = server._tool_manager._tools["query"].fn

        with installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="sessions", continuation="bogus"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_continuation"


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("waits on a real transport thread that outlives the cancelled await")
async def test_cancelled_daemon_submission_cancels_the_same_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled MCP call waits for its transport thread and cancels that request.

    Anti-vacuity: awaiting ``asyncio.to_thread`` directly abandons the thread
    on cancellation, so no ``operation.cancel`` names its request id; giving
    up after one cancel refused as unregistered, or joining the submission
    before a cancel lands, never sets ``cancel_sent`` while it is blocked.
    """
    import asyncio
    import threading
    import time
    from types import SimpleNamespace

    from polylogue.mcp import server_cutover

    submitted = threading.Event()
    registered = threading.Event()
    release = threading.Event()
    cancel_sent = threading.Event()
    calls: list[tuple[str, str | None]] = []

    class SlowClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def operation(
            self, operation: str, payload: object, *, archive_root: str, request_id: str
        ) -> dict[str, object]:
            submitted.set()
            # The daemon registers the request id a moment after the POST
            # starts; a cancel before that finds no such request.
            time.sleep(0.2)
            registered.set()
            release.wait(timeout=10)
            calls.append((operation, request_id))
            return {"outcome": "accepted", "request_id": request_id}

        def cancel(self, request_id: str, *, archive_root: str) -> dict[str, object]:
            if not registered.is_set():
                # The production daemon refuses an unregistered id with a
                # typed envelope, not an exception.
                return {"outcome": "rejected", "error": {"code": "operation_reference_unknown"}}
            calls.append(("operation.cancel", request_id))
            cancel_sent.set()
            return {"outcome": "completed"}

    monkeypatch.setattr("polylogue.daemon_client.DaemonClient", SlowClient)
    monkeypatch.setattr("polylogue.cli.read_dispatch.daemon_route_disabled", lambda **_kwargs: False)
    hooks = SimpleNamespace(
        get_config=lambda: SimpleNamespace(archive_root=tmp_path, api_auth_token=None, api_allow_no_auth=True),
        error_json=lambda message, **extra: json.dumps({"error": message, **extra}),
    )

    task = asyncio.ensure_future(
        server_cutover._daemon_operation(hooks, "maintenance.insights.rebuild", {})  # type: ignore[arg-type]
    )
    await asyncio.to_thread(submitted.wait, 10)
    task.cancel()
    await asyncio.to_thread(cancel_sent.wait, 10)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    # A cancel reaches the daemon while the submission is still blocked --
    # after it registered, before it returned -- and again once it returned;
    # every call names the same request.
    assert [name for name, _ in calls] == ["operation.cancel", "maintenance.insights.rebuild", "operation.cancel"]
    assert len({request_id for _, request_id in calls}) == 1


@pytest.mark.asyncio
async def test_search_continuation_preserves_query_sort_and_omitted_limit(tmp_path: Path) -> None:
    """Without restoring the framed request, page two loses query/sort or widens its limit."""
    from tests.infra.mcp import build_tools

    root = tmp_path / "archive"
    _seed_paged_archive(root)
    query = build_tools()["query"]
    with installed_runtime_services(root):
        first = json.loads(
            await invoke_surface_async(query, expression="pagination", projection="sessions", sort="date", limit=2)
        )
        assert first["continuation"]
        second = json.loads(
            await invoke_surface_async(query, projection="sessions", continuation=first["continuation"])
        )
    assert second.get("is_error") is not True, second
    assert second["query"] == first["query"] == "pagination"
    assert second["sort"] == first["sort"] == "date"
    assert second["limit"] == 2
    assert second["offset"] == 2
    assert {hit["session"]["id"] for hit in first["hits"]}.isdisjoint(hit["session"]["id"] for hit in second["hits"])


@pytest.mark.asyncio
async def test_missing_messages_session_is_not_found(tmp_path: Path) -> None:
    """Without the typed catch, the registered read reports polylogue_error."""
    from tests.infra.mcp import build_tools

    root = tmp_path / "archive"
    _seed_archive(root)
    read = build_tools()["read"]
    with installed_runtime_services(root):
        body = json.loads(
            await invoke_surface_async(read, ref="session:chatgpt-export:does-not-exist", view="messages")
        )
    assert body["is_error"] is True
    assert body["code"] == "not_found"


@pytest.mark.asyncio
async def test_reference_relation_query_advances_requested_offset(tmp_path: Path) -> None:
    """Without offset slicing, every registered query call repeats the first two refs."""
    import sqlite3

    from polylogue.storage.sqlite.query_objects import put_query, put_result_set
    from tests.infra.mcp import build_tools

    root = tmp_path / "archive"
    ids = _seed_paged_archive(root, count=5)
    with sqlite3.connect(root / "user.db") as conn:
        query_object = put_query(
            conn,
            {"field": "origin", "value": "chatgpt-export"},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
        put_result_set(
            conn,
            result_set_id="w13-page",
            query_hash=query_object.query_hash,
            grain="session",
            corpus_epoch="e1",
            member_refs=tuple(f"session:{sid}" for sid in ids),
            exactness="exact",
            persistence_class="pinned",
            created_at_ms=2,
        )
    query = build_tools()["query"]
    with installed_runtime_services(root):
        pages = [
            json.loads(await invoke_surface_async(query, expression="from result-set:w13-page", limit=2, offset=offset))
            for offset in (0, 2, 4)
        ]
    assert [p["offset"] for p in pages] == [0, 2, 4]
    assert [p["next_offset"] for p in pages] == [2, 4, None]
    assert [ref for page in pages for ref in page["members"]] == [f"session:{sid}" for sid in ids]
