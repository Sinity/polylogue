"""Mutation-oriented MCP tool registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPMetadataPayload, MCPMutationStatusPayload, MCPRootPayload, MCPTagCountsPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_mutation_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def add_tag(conversation_id: str, tag: str) -> str:
        async def run() -> str:
            await hooks.get_tag_store().add_tag(conversation_id, tag)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    tag=tag,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("add_tag", run)

    @mcp.tool()
    async def remove_tag(conversation_id: str, tag: str) -> str:
        async def run() -> str:
            await hooks.get_tag_store().remove_tag(conversation_id, tag)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    tag=tag,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("remove_tag", run)

    @mcp.tool()
    async def bulk_tag_conversations(conversation_ids: list[str], tags: list[str]) -> str:
        async def run() -> str:
            if not conversation_ids:
                return hooks.error_json("bulk_tag_conversations requires at least one conversation_id")
            if not tags:
                return hooks.error_json("bulk_tag_conversations requires at least one tag")
            tag_store = hooks.get_tag_store()
            applied_count = 0
            for conversation_id in conversation_ids:
                for tag in tags:
                    await tag_store.add_tag(conversation_id, tag)
                    applied_count += 1
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "status": "ok",
                        "conversation_count": len(conversation_ids),
                        "tag_count": len(tags),
                        "applied_count": applied_count,
                    }
                )
            )

        return await hooks.async_safe_call("bulk_tag_conversations", run)

    @mcp.tool()
    async def list_tags(provider: str | None = None) -> str:
        async def run() -> str:
            tags = await hooks.get_tag_store().list_tags(provider=provider)
            return hooks.json_payload(MCPTagCountsPayload(root=tags))

        return await hooks.async_safe_call("list_tags", run)

    @mcp.tool()
    async def get_metadata(conversation_id: str) -> str:
        async def run() -> str:
            metadata = await hooks.get_tag_store().get_metadata(conversation_id)
            return hooks.json_payload(MCPMetadataPayload.from_document(metadata))

        return await hooks.async_safe_call("get_metadata", run)

    @mcp.tool()
    async def set_metadata(conversation_id: str, key: str, value: str) -> str:
        async def run() -> str:
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value
            await hooks.get_tag_store().update_metadata(conversation_id, key, parsed_value)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    key=key,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("set_metadata", run)

    @mcp.tool()
    async def delete_metadata(conversation_id: str, key: str) -> str:
        async def run() -> str:
            await hooks.get_tag_store().delete_metadata(conversation_id, key)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=conversation_id,
                    key=key,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_metadata", run)

    @mcp.tool()
    async def delete_conversation(conversation_id: str, confirm: bool = False) -> str:
        async def run() -> str:
            if not confirm:
                return hooks.error_json(
                    "Safety guard: set confirm=true to delete",
                    conversation_id=conversation_id,
                )
            deleted = await hooks.get_query_store().delete_conversation(conversation_id)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="deleted" if deleted else "not_found",
                    conversation_id=conversation_id,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_conversation", run)


__all__ = ["register_mutation_tools"]
