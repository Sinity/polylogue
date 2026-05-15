"""Mutation-oriented MCP tool registration."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from polylogue.mcp.payloads import (
    MCPMetadataPayload,
    MCPSavedViewListPayload,
    MCPSavedViewPayload,
    MCPTagCountsPayload,
    MCPUserMarkListPayload,
    MCPUserMarkPayload,
    MutationResultPayload,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


async def _resolve_or_error(hooks: ServerCallbacks, conversation_id: str) -> tuple[str | None, str | None]:
    """Resolve a conversation ID, returning the canonical ID or an error JSON."""
    resolved = await hooks.get_query_store().resolve_id(conversation_id, strict=True)
    if not resolved:
        return None, hooks.error_json("conversation not found", conversation_id=conversation_id)
    return resolved, None


def _saved_view_payload(row: dict[str, str]) -> MCPSavedViewPayload:
    try:
        query = json.loads(row["query_json"])
    except (json.JSONDecodeError, TypeError):
        query = {}
    if not isinstance(query, dict):
        query = {}
    return MCPSavedViewPayload(
        view_id=row["view_id"],
        name=row["name"],
        query=query,
        created_at=row["created_at"],
    )


def register_mutation_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def add_tag(conversation_id: str, tag: str) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract
            poly = hooks.get_polylogue()
            result = await poly.add_tag(resolved, tag)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if result.outcome == "added" else "unchanged",
                    conversation_id=resolved,
                    tag=tag,
                    detail=result.detail,
                    outcome=result.outcome,
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
            result = await poly.remove_tag(resolved, tag)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if result.outcome == "removed" else "not_found",
                    conversation_id=resolved,
                    tag=tag,
                    detail=result.detail,
                    outcome=result.outcome,
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
    async def list_marks(mark_type: str | None = None, conversation_id: str | None = None) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_marks(mark_type=mark_type, conversation_id=conversation_id)
            items = tuple(
                MCPUserMarkPayload(
                    conversation_id=row["conversation_id"],
                    mark_type=row["mark_type"],
                    created_at=row["created_at"],
                )
                for row in rows
            )
            return hooks.json_payload(MCPUserMarkListPayload(items=items, total=len(items)))

        return await hooks.async_safe_call("list_marks", run)

    @mcp.tool()
    async def add_mark(conversation_id: str, mark_type: str) -> str:
        async def run() -> str:
            if mark_type not in {"star", "pin", "archive"}:
                return hooks.error_json("mark_type must be one of: star, pin, archive", detail=mark_type)
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None
            poly = hooks.get_polylogue()
            created = await poly.add_mark(resolved, mark_type)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if created else "unchanged",
                    conversation_id=resolved,
                    detail=None if created else "already_present",
                    key=mark_type,
                    outcome="added" if created else "no_op",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("add_mark", run)

    @mcp.tool()
    async def remove_mark(conversation_id: str, mark_type: str) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None
            poly = hooks.get_polylogue()
            deleted = await poly.remove_mark(resolved, mark_type)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if deleted else "not_found",
                    conversation_id=resolved,
                    detail=None if deleted else "mark_not_present",
                    key=mark_type,
                    outcome="removed" if deleted else "not_present",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("remove_mark", run)

    @mcp.tool()
    async def list_saved_views() -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_views()
            items = tuple(_saved_view_payload(row) for row in rows)
            return hooks.json_payload(MCPSavedViewListPayload(items=items, total=len(items)))

        return await hooks.async_safe_call("list_saved_views", run)

    @mcp.tool()
    async def save_saved_view(name: str, query_json: str, view_id: str | None = None) -> str:
        async def run() -> str:
            if not name.strip():
                return hooks.error_json("saved view name must not be empty")
            try:
                query = json.loads(query_json)
            except json.JSONDecodeError:
                return hooks.error_json("query_json must be valid JSON")
            if not isinstance(query, dict):
                return hooks.error_json("query_json must encode an object")

            from polylogue.archive.query.spec import ConversationQuerySpec

            try:
                ConversationQuerySpec.from_params(query, strict=True)
            except Exception as exc:
                return hooks.error_json("query_json is not a valid ConversationQuerySpec", detail=type(exc).__name__)

            canonical_query_json = json.dumps(query, sort_keys=True, separators=(",", ":"))
            poly = hooks.get_polylogue()
            saved_id = view_id or name.strip().lower().replace(" ", "-")
            created = await poly.save_view(saved_id, name.strip(), canonical_query_json)
            saved = await poly.get_view(saved_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if created else "unchanged",
                    detail=None if saved else "saved_view_not_found",
                    key=saved_id,
                    outcome="added" if created else "updated",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_saved_view", run)

    @mcp.tool()
    async def delete_saved_view(view_id: str) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            deleted = await poly.delete_view(view_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "saved_view_not_found",
                    key=view_id,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_saved_view", run)

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
            # Validate metadata key
            if not key or not key.strip():
                return hooks.error_json(
                    "metadata key must not be empty",
                    conversation_id=conversation_id,
                    code="invalid_key",
                )
            if len(key) > 200:
                return hooks.error_json(
                    "metadata key exceeds 200 characters",
                    conversation_id=conversation_id,
                    code="invalid_key",
                )

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

            tag_store: Any = hooks.get_tag_store()
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
