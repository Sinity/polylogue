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
            from polylogue.api.archive import ConversationNotFoundError

            poly = hooks.get_polylogue()
            try:
                await poly.add_tag(conversation_id, tag)
            except ConversationNotFoundError:
                return hooks.error_json("conversation not found", conversation_id=conversation_id)
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
            from polylogue.api.archive import ConversationNotFoundError

            poly = hooks.get_polylogue()
            try:
                await poly.remove_tag(conversation_id, tag)
            except ConversationNotFoundError:
                return hooks.error_json("conversation not found", conversation_id=conversation_id)
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
            if len(conversation_ids) > 100:
                return hooks.error_json("bulk_tag_conversations supports at most 100 conversation_ids")
            if len(tags) > 20:
                return hooks.error_json("bulk_tag_conversations supports at most 20 tags")
            tag_store = hooks.get_tag_store()
            applied_count = await tag_store.bulk_add_tags(conversation_ids, tags)
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
            poly = hooks.get_polylogue()
            tags = await poly.list_tags(provider=provider)
            return hooks.json_payload(MCPTagCountsPayload(root=tags))

        return await hooks.async_safe_call("list_tags", run)

    @mcp.tool()
    async def get_metadata(conversation_id: str) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            metadata = await poly.get_metadata(conversation_id)
            return hooks.json_payload(MCPMetadataPayload.from_document(metadata))

        return await hooks.async_safe_call("get_metadata", run)

    @mcp.tool()
    async def set_metadata(conversation_id: str, key: str, value: str) -> str:
        async def run() -> str:
            from polylogue.api.archive import ConversationNotFoundError

            poly = hooks.get_polylogue()
            try:
                from contextlib import suppress

                parsed_value: object = value
                with suppress(json.JSONDecodeError, TypeError):
                    parsed_value = json.loads(value)
                await poly.update_metadata(conversation_id, key, str(parsed_value))
            except ConversationNotFoundError:
                return hooks.error_json("conversation not found", conversation_id=conversation_id)
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
            resolved = await hooks.get_query_store().resolve_id(conversation_id, strict=True)
            if not resolved:
                return hooks.error_json("conversation not found", conversation_id=conversation_id)
            await hooks.get_tag_store().delete_metadata(resolved, key)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok",
                    conversation_id=resolved,
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
            from polylogue.api.archive import ConversationNotFoundError
            from polylogue.mcp.server_support import _get_polylogue

            poly = _get_polylogue()
            try:
                deleted = await poly.delete_conversation(conversation_id)
            except ConversationNotFoundError:
                return hooks.error_json("conversation not found", conversation_id=conversation_id)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="deleted" if deleted else "not_found",
                    conversation_id=conversation_id,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_conversation", run)


__all__ = ["register_mutation_tools"]
