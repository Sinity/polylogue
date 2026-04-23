"""Maintenance and export MCP tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPMutationStatusPayload, MCPRootPayload
from polylogue.mcp.query_contracts import MCPConversationQueryRequest

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
            from polylogue.rendering.formatting import format_conversation, normalize_conversation_output_format

            conv = await hooks.get_archive_ops().get_conversation(id)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {id}")
            fmt = normalize_conversation_output_format(format)
            return format_conversation(conv, fmt, None)

        return await hooks.async_safe_call("export_conversation", run)

    @mcp.tool()
    async def rebuild_session_products(conversation_ids: list[str] | None = None) -> str:
        async def run() -> str:
            counts = await hooks.get_archive_ops().rebuild_session_products(conversation_ids=conversation_ids)
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "status": "ok",
                        "conversation_count": len(conversation_ids) if conversation_ids is not None else None,
                        "counts": counts.to_dict(),
                        "total": counts.total(),
                    }
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("rebuild_session_products", run)

    @mcp.tool()
    async def export_query_results(
        query: str | None = None,
        format: str = "markdown",
        limit: int = 10,
        retrieval_lane: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        tag: str | None = None,
        title: str | None = None,
        path: str | None = None,
        action: str | None = None,
        exclude_action: str | None = None,
        action_sequence: str | None = None,
        action_text: str | None = None,
        tool: str | None = None,
        exclude_tool: str | None = None,
        sort: str | None = None,
        has_tool_use: bool = False,
        has_thinking: bool = False,
        min_messages: int | None = None,
        min_words: int | None = None,
    ) -> str:
        async def run() -> str:
            from polylogue.rendering.formatting import format_conversation, normalize_conversation_output_format

            fmt = normalize_conversation_output_format(format)
            spec = MCPConversationQueryRequest(
                query=query,
                retrieval_lane=retrieval_lane,
                provider=provider,
                since=since,
                tag=tag,
                title=title,
                path=path,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                sort=sort,
                limit=limit,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                min_words=min_words,
            ).build_spec(hooks.clamp_limit)
            conversations = await hooks.get_archive_ops().query_conversations(spec)
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(conversations),
                        "format": fmt,
                        "exports": [
                            {
                                "conversation_id": str(conversation.id),
                                "provider": str(conversation.provider),
                                "title": conversation.display_title,
                                "content": format_conversation(conversation, fmt, None),
                            }
                            for conversation in conversations
                        ],
                    }
                )
            )

        return await hooks.async_safe_call("export_query_results", run)


__all__ = ["register_maintenance_tools"]
