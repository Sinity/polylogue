"""Maintenance and export MCP tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPMutationStatusPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_maintenance_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def rebuild_index() -> str:
        async def run() -> str:
            from polylogue.pipeline.services.indexing import IndexService

            service = IndexService(config=hooks.get_config(), backend=hooks.get_backend())
            success = await service.rebuild_index()
            status_info = await service.get_index_status()
            index_exists_value = status_info.get("exists", False)
            indexed_messages_value = status_info.get("count", 0)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok" if success else "failed",
                    index_exists=index_exists_value if isinstance(index_exists_value, bool) else False,
                    indexed_messages=indexed_messages_value if isinstance(indexed_messages_value, int) else 0,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("rebuild_index", run)

    @mcp.tool()
    async def update_index(conversation_ids: list[str]) -> str:
        async def run() -> str:
            from polylogue.pipeline.services.indexing import IndexService

            service = IndexService(config=hooks.get_config(), backend=hooks.get_backend())
            success = await service.update_index(conversation_ids)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok" if success else "failed",
                    conversation_count=len(conversation_ids),
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("update_index", run)

    @mcp.tool()
    async def export_conversation(id: str, format: str = "markdown") -> str:
        async def run() -> str:
            from polylogue.rendering.formatting import format_conversation

            conv = await hooks.get_archive_ops().get_conversation(id)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {id}")
            valid_formats = {
                "markdown",
                "json",
                "html",
                "yaml",
                "plaintext",
                "csv",
                "obsidian",
                "org",
            }
            fmt = format if format in valid_formats else "markdown"
            return format_conversation(conv, fmt, None)

        return await hooks.async_safe_call("export_conversation", run)


__all__ = ["register_maintenance_tools"]
