"""MCP surface wiring for the durable context-delivery ledger (polylogue-37t.22).

Exercises the two MCP-reachable ends of the delivery boundary added on top of
the existing storage substrate (PR #2703 `context_delivery_write.py`):

- ``write(operation="deliver_context")`` compiles a bounded context image
  and records its exact receipt.
- ``context(result_ref=..., recipient_ref=...)`` resolves one receipt (full
  disclosure, recipient-scoped); ``context(recipient_ref=...)`` alone lists
  bounded summaries (no full content).

Both run against a real seeded archive via ``RuntimeServices``, matching the
pattern in ``test_privileged_tools.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.mcp.declarations.models import MCPCapabilities
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
                provider_session_id="context-delivery-tool-probe",
                title="Context delivery tool probe",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="needle context delivery evidence",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle context delivery evidence")],
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


class TestDeliverContextOperation:
    @pytest.mark.asyncio
    async def test_deliver_context_records_a_receipt_the_context_tool_can_resolve(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn
        context_fn = server._tool_manager._tools["context"].fn

        with _installed_runtime_services(archive_root):
            delivered = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="deliver_context",
                    fields={
                        "recipient_ref": "agent:codex-main",
                        "delivered_by_ref": "user:local",
                        "boundary": "explicit-recall",
                        "query": "needle context delivery",
                        "max_sessions": 1,
                    },
                )
            )
            assert delivered.get("is_error") is not True, delivered
            assert delivered["recipient_ref"] == "agent:codex-main"
            assert delivered["delivered_by_ref"] == "user:local"
            snapshot_ref = delivered["snapshot_ref"]
            assert snapshot_ref.startswith("context-snapshot:")

            resolved = json.loads(
                await invoke_surface_async(
                    context_fn,
                    intent="lookup",
                    result_ref=snapshot_ref,
                    recipient_ref="agent:codex-main",
                )
            )
            assert resolved.get("is_error") is not True, resolved
            assert resolved["snapshot_ref"] == snapshot_ref
            assert resolved["context_image_sha256"] == delivered["context_image_sha256"]

            wrong_recipient = json.loads(
                await invoke_surface_async(
                    context_fn,
                    intent="lookup",
                    result_ref=snapshot_ref,
                    recipient_ref="agent:someone-else",
                )
            )
            assert wrong_recipient.get("is_error") is True
            assert wrong_recipient.get("code") == "not_found"

    @pytest.mark.asyncio
    async def test_deliver_context_replay_is_idempotent(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        call_fields = {
            "recipient_ref": "agent:codex-main",
            "delivered_by_ref": "user:local",
            "boundary": "explicit-recall",
            "query": "needle context delivery",
            "max_sessions": 1,
        }
        with _installed_runtime_services(archive_root):
            first = json.loads(await invoke_surface_async(write_fn, operation="deliver_context", fields=call_fields))
            replay = json.loads(await invoke_surface_async(write_fn, operation="deliver_context", fields=call_fields))

            assert first["snapshot_ref"] == replay["snapshot_ref"]
            assert first["context_image_sha256"] == replay["context_image_sha256"]

    @pytest.mark.asyncio
    async def test_deliver_context_requires_recipient_and_delivered_by_and_boundary(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="deliver_context",
                    fields={"delivered_by_ref": "user:local", "boundary": "explicit-recall"},
                )
            )
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_deliver_context_is_not_reachable_without_the_write_capability(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())  # read-only, default capabilities

        assert "write" not in server._tool_manager._tools


class TestContextToolListsReceiptSummaries:
    @pytest.mark.asyncio
    async def test_recipient_ref_alone_lists_bounded_summaries_without_full_content(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn
        context_fn = server._tool_manager._tools["context"].fn

        with _installed_runtime_services(archive_root):
            delivered = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="deliver_context",
                    fields={
                        "recipient_ref": "agent:codex-main",
                        "delivered_by_ref": "user:local",
                        "boundary": "explicit-recall",
                        "query": "needle context delivery",
                        "max_sessions": 1,
                    },
                )
            )

            listed = json.loads(
                await invoke_surface_async(context_fn, intent="lookup", recipient_ref="agent:codex-main")
            )
            assert listed.get("is_error") is not True, listed
            assert listed["total"] == 1
            assert len(listed["items"]) == 1
            summary = listed["items"][0]
            assert summary["snapshot_ref"] == delivered["snapshot_ref"]
            assert "context_image" not in summary

            empty = json.loads(await invoke_surface_async(context_fn, intent="lookup", recipient_ref="agent:nobody"))
            assert empty["total"] == 0
            assert empty["items"] == []

    @pytest.mark.asyncio
    async def test_result_ref_without_recipient_ref_is_an_invalid_argument(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server())
        context_fn = server._tool_manager._tools["context"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(context_fn, intent="lookup", result_ref="context-snapshot:deadbeef")
            )
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"
