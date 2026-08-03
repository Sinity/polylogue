"""Unit tests for the privileged transaction tools (write/judge/run/maintenance, t46.8.3).

These are thin adapters over the same typed owners the retired per-operation
MCP tools used (write, run, maintenance) or already used (judge) -- see
``register_cutover_privileged_tools`` in ``polylogue/mcp/server_cutover.py``.
Each is verified against a real seeded archive via ``RuntimeServices``, not
mocks, matching the pattern established in ``test_envelope_contracts.py`` and
``test_contract_evidence.py`` (query/context/explain route through the cached
``_get_polylogue()`` facade, so a real runtime service scope is required).

``build_server()`` must be called *before* entering ``_installed_runtime_services``
-- it always resolves and installs its own default runtime services when not
given one explicitly, which would otherwise clobber the seeded ones.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.mcp.declarations.models import MCPCapabilities
from tests.infra.mcp import ALL_CAPABILITIES, MCPServerUnderTest, invoke_surface_async


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
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="add_tag", tag="reviewed"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_operation_specific_field_is_read_from_fields_dict(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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

        with _installed_runtime_services(archive_root):
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

        with _installed_runtime_services(archive_root):
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

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="delete_session", session_id=session_id))
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_save_and_delete_saved_view_round_trips(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

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
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

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

            refused = json.loads(
                await invoke_surface_async(write_fn, operation="delete_saved_view", fields={"view_id": view_id})
            )
            assert refused.get("is_error") is True
            assert "confirm" in refused.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_delete_recall_pack_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
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
    facade methods (``docs/plans/mutation-census.yaml``'s ``adapters``
    entries name ``polylogue.mcp.server_cutover._dispatch_write`` for each
    executor-routed operation). So t46.8.3 required no *new* MCP-layer wiring
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
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        # bulk_tag_sessions needs a real session_ids list resolved at test
        # time (the fixture only knows its id after seeding).
        if op_kwargs.get("session_ids", "__unset__") is None:
            op_kwargs = {**op_kwargs, "session_ids": [session_id]}

        with _installed_runtime_services(archive_root):
            for setup_call in setup:
                setup_result = json.loads(await invoke_surface_async(write_fn, session_id=session_id, **setup_call))
                assert setup_result.get("is_error") is not True, setup_result

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
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        def boom(
            self: OperationExecutor, actuator: object, plan: object, authorization: object, args: object
        ) -> object:
            raise RuntimeError("t46.8.3-bypass-proof: executor forcibly disabled")

        monkeypatch.setattr(OperationExecutor, "execute", boom)

        with _installed_runtime_services(archive_root):
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
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        def boom(
            self: OperationExecutor, actuator: object, plan: object, authorization: object, args: object
        ) -> object:
            raise RuntimeError("t46.9-capture-executor-bypass-proof")

        monkeypatch.setattr(OperationExecutor, "execute", boom)

        with _installed_runtime_services(archive_root):
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
        from unittest.mock import AsyncMock, patch

        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
        judge_fn = server._tool_manager._tools["judge"].fn

        with _installed_runtime_services(archive_root):
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
                        judge_fn, candidate_ref="assertion:contract-candidate", decision="accept"
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
        from unittest.mock import AsyncMock, patch

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

        with _installed_runtime_services(archive_root):
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

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(judge_fn))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"


class TestRunTool:
    @pytest.mark.asyncio
    async def test_run_executes_a_saved_query_ref_and_returns_matching_sessions(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn
        run_fn = server._tool_manager._tools["run"].fn

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

        with _installed_runtime_services(archive_root):
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

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(run_fn, ref="session:not-a-saved-query"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"


class TestMaintenanceTool:
    @pytest.mark.asyncio
    async def test_list_returns_empty_envelope_on_a_fresh_archive(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="list"))
            assert result.get("is_error") is not True, result
            assert result["items"] == []
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_status_without_operation_id_returns_invalid_argument(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="status"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_status_for_missing_operation_id_returns_not_found(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(maintenance_fn, operation="status", operation_id="does-not-exist")
            )
            assert result.get("is_error") is True
            assert result.get("code") == "not_found"


class TestMaintenanceConfirmGates:
    """polylogue-jn40: full-effect maintenance operations must fail closed.

    ``rebuild_index`` and ``rebuild_insights`` route through
    ``hooks.get_polylogue()`` (the installed runtime services / seeded
    archive); ``execute`` with ``dry_run=false`` routes through the
    planner's own ``Config`` (a pre-existing, unrelated quirk -- see
    ``_dispatch_maintenance``), so its refusal case is verified independent
    of archive content and its confirmed-success case merely asserts the
    call is not refused.
    """

    @pytest.mark.asyncio
    async def test_execute_with_dry_run_false_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    maintenance_fn, operation="execute", targets=["session_insights"], dry_run=False
                )
            )
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_execute_with_dry_run_true_does_not_require_confirm(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    maintenance_fn, operation="execute", targets=["session_insights"], dry_run=True
                )
            )
            assert result.get("is_error") is not True, result

    @pytest.mark.asyncio
    async def test_execute_with_dry_run_false_and_confirm_true_succeeds(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    maintenance_fn,
                    operation="execute",
                    targets=["session_insights"],
                    dry_run=False,
                    confirm=True,
                )
            )
            assert result.get("is_error") is not True, result

    @pytest.mark.asyncio
    async def test_rebuild_index_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="rebuild_index"))
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_rebuild_index_with_confirm_succeeds(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="rebuild_index", confirm=True))
            assert result.get("is_error") is not True, result
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_rebuild_insights_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="rebuild_insights"))
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_rebuild_insights_with_confirm_succeeds(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(maintenance=True)))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="rebuild_insights", confirm=True))
            assert result.get("is_error") is not True, result
            assert result["status"] == "ok"


class TestQuerySessionsProjection:
    @pytest.mark.asyncio
    async def test_ranked_search_finds_the_seeded_session(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())
        query_fn = server._tool_manager._tools["query"].fn

        with _installed_runtime_services(archive_root):
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

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="sessions", limit=10))
            assert result.get("is_error") is not True, result
            assert "items" in result
            assert result["total"] >= 1

    @pytest.mark.asyncio
    async def test_sessions_projection_rejects_continuation(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())
        query_fn = server._tool_manager._tools["query"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="sessions", continuation="bogus"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_continuation"
