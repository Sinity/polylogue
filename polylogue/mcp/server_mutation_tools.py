"""Mutation-oriented MCP tool registration."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING

from polylogue.mcp.payloads import (
    MCPMetadataPayload,
    MCPTagCountsPayload,
    MutationResultPayload,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


async def _resolve_or_error(
    hooks: ServerCallbacks, conversation_id: str
) -> tuple[str | None, str | None]:
    """Resolve a conversation ID, returning the canonical ID or an error JSON."""
    resolved = await hooks.get_query_store().resolve_id(conversation_id, strict=True)
    if not resolved:
        return None, hooks.error_json("conversation not found", conversation_id=conversation_id)
    return resolved, None


def register_mutation_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def add_tag(conversation_id: str, tag: str) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract
            poly = hooks.get_polylogue()
            was_added = await poly.add_tag(resolved, tag)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if was_added else "unchanged",
                    conversation_id=resolved,
                    tag=tag,
                    detail=None if was_added else "already_present",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("add_tag", run)

    @mcp.tool()
    async def remove_tag(conversation_id: str, tag: str) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract
            poly = hooks.get_polylogue()
            was_removed = await poly.remove_tag(resolved, tag)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if was_removed else "not_found",
                    conversation_id=resolved,
                    tag=tag,
                    detail=None if was_removed else "tag_not_present",
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
            skipped_count = len(conversation_ids) - applied_count
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok",
                    conversation_count=len(conversation_ids),
                    tag_count=len(tags),
                    affected_count=applied_count,
                    skipped_count=skipped_count,
                ),
                exclude_none=True,
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
            return hooks.json_payload(MCPMetadataPayload.from_document(metadata))  # type: ignore[arg-type]

        return await hooks.async_safe_call("get_metadata", run)

    @mcp.tool()
    async def set_metadata(conversation_id: str, key: str, value: str) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract

            parsed_value: object = value
            with suppress(json.JSONDecodeError, TypeError):
                parsed_value = json.loads(value)
            parsed_str = str(parsed_value)

            poly = hooks.get_polylogue()
            was_changed = await poly.update_metadata(resolved, key, parsed_str)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if was_changed else "unchanged",
                    conversation_id=resolved,
                    key=key,
                    detail=None if was_changed else "value_unchanged",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("set_metadata", run)

    @mcp.tool()
    async def delete_metadata(conversation_id: str, key: str) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract

            from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin

            tag_store: RepositoryWriteMixin = hooks.get_tag_store()
            was_deleted = await tag_store.delete_metadata(resolved, key)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if was_deleted else "not_found",
                    conversation_id=resolved,
                    key=key,
                    detail=None if was_deleted else "key_not_found",
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
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract

            from polylogue.mcp.server_support import _get_polylogue

            poly = _get_polylogue()
            deleted = await poly.delete_conversation(resolved)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    conversation_id=resolved,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_conversation", run)


__all__ = ["register_mutation_tools"]
