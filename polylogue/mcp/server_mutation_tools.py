"""Mutation-oriented MCP tool registration."""

from __future__ import annotations

import json
from contextlib import suppress
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from polylogue.mcp.payloads import (
    MCPMetadataPayload,
    MCPReaderWorkspaceListPayload,
    MCPReaderWorkspacePayload,
    MCPRecallPackListPayload,
    MCPRecallPackPayload,
    MCPSavedViewListPayload,
    MCPSavedViewPayload,
    MCPTagCountsPayload,
    MCPUserAnnotationListPayload,
    MCPUserAnnotationPayload,
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
        return None, hooks.error_json("Conversation not found", code="not_found", conversation_id=conversation_id)
    return resolved, None


def _mark_type_error(hooks: ServerCallbacks, mark_type: str) -> str | None:
    if mark_type in {"star", "pin", "archive"}:
        return None
    return hooks.error_json("mark_type must be one of: star, pin, archive", detail=mark_type)


def _default_saved_view_id(name: str, query_json: str) -> str:
    digest = sha256(f"{name}\0{query_json}".encode()).hexdigest()
    return f"saved-view-{digest[:16]}"


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


def _recall_pack_payload(row: dict[str, str]) -> MCPRecallPackPayload:
    try:
        conversation_ids = json.loads(row["conversation_ids_json"])
    except (json.JSONDecodeError, TypeError):
        conversation_ids = []
    if not isinstance(conversation_ids, list):
        conversation_ids = []
    try:
        payload = json.loads(row["payload_json"])
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return MCPRecallPackPayload(
        pack_id=row["pack_id"],
        label=row["label"],
        conversation_ids=tuple(str(item) for item in conversation_ids),
        payload=payload,
        created_at=row["created_at"],
    )


def _workspace_payload(row: dict[str, str]) -> MCPReaderWorkspacePayload:
    try:
        open_targets = json.loads(row["open_targets_json"])
    except (json.JSONDecodeError, TypeError):
        open_targets = []
    if not isinstance(open_targets, list):
        open_targets = []
    try:
        layout = json.loads(row["layout_json"])
    except (json.JSONDecodeError, TypeError):
        layout = {}
    if not isinstance(layout, dict):
        layout = {}
    try:
        active_target = json.loads(row["active_target_json"])
    except (json.JSONDecodeError, TypeError):
        active_target = {}
    if not isinstance(active_target, dict):
        active_target = {}
    return MCPReaderWorkspacePayload(
        workspace_id=row["workspace_id"],
        name=row["name"],
        mode=row["mode"],
        open_targets=tuple(item for item in open_targets if isinstance(item, dict)),
        layout=layout,
        active_target=active_target,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _none_if_empty(value: str | None) -> str | None:
    return value if value else None


def _annotation_payload(row: dict[str, str]) -> MCPUserAnnotationPayload:
    return MCPUserAnnotationPayload(
        annotation_id=row["annotation_id"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        conversation_id=row["conversation_id"],
        message_id=_none_if_empty(row.get("message_id")),
        note_text=row["note_text"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
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
    async def list_marks(
        mark_type: str | None = None,
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_marks(
                mark_type=mark_type,
                conversation_id=conversation_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
            items = tuple(
                MCPUserMarkPayload(
                    target_type=row["target_type"],
                    target_id=row["target_id"],
                    conversation_id=row["conversation_id"],
                    message_id=_none_if_empty(row.get("message_id")),
                    mark_type=row["mark_type"],
                    created_at=row["created_at"],
                )
                for row in rows
            )
            return hooks.json_payload(MCPUserMarkListPayload(items=items, total=len(items)))

        return await hooks.async_safe_call("list_marks", run)

    @mcp.tool()
    async def add_mark(
        conversation_id: str,
        mark_type: str,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            mark_error = _mark_type_error(hooks, mark_type)
            if mark_error:
                return mark_error
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None
            poly = hooks.get_polylogue()
            created = await poly.add_mark(
                resolved,
                mark_type,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
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
    async def remove_mark(
        conversation_id: str,
        mark_type: str,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            mark_error = _mark_type_error(hooks, mark_type)
            if mark_error:
                return mark_error
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None
            poly = hooks.get_polylogue()
            deleted = await poly.remove_mark(
                resolved,
                mark_type,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
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
    async def list_annotations(
        conversation_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_annotations(
                conversation_id=conversation_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
            items = tuple(_annotation_payload(row) for row in rows)
            return hooks.json_payload(MCPUserAnnotationListPayload(items=items, total=len(items)))

        return await hooks.async_safe_call("list_annotations", run)

    @mcp.tool()
    async def save_annotation(
        annotation_id: str,
        conversation_id: str,
        note_text: str,
        target_type: str = "conversation",
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            if not annotation_id.strip():
                return hooks.error_json("annotation_id must not be empty")
            if not note_text.strip():
                return hooks.error_json("note_text must not be empty")
            resolved, err = await _resolve_or_error(hooks, conversation_id)
            if err:
                return err
            assert resolved is not None
            poly = hooks.get_polylogue()
            created = await poly.save_annotation(
                annotation_id,
                resolved,
                note_text,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok",
                    conversation_id=resolved,
                    key=annotation_id,
                    outcome="added" if created else "updated",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_annotation", run)

    @mcp.tool()
    async def delete_annotation(annotation_id: str) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            deleted = await poly.delete_annotation(annotation_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "annotation_not_found",
                    key=annotation_id,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_annotation", run)

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
                detail = f"{type(exc).__name__}: {exc}"
                return hooks.error_json("query_json is not a valid ConversationQuerySpec", detail=detail)

            canonical_query_json = json.dumps(query, sort_keys=True, separators=(",", ":"))
            poly = hooks.get_polylogue()
            saved_id = view_id or _default_saved_view_id(name.strip(), canonical_query_json)
            created = await poly.save_view(saved_id, name.strip(), canonical_query_json)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok",
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
    async def list_recall_packs() -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_recall_packs()
            items = tuple(_recall_pack_payload(row) for row in rows)
            return hooks.json_payload(MCPRecallPackListPayload(items=items, total=len(items)))

        return await hooks.async_safe_call("list_recall_packs", run)

    @mcp.tool()
    async def save_recall_pack(
        pack_id: str,
        label: str,
        payload_json: str = "{}",
    ) -> str:
        async def run() -> str:
            if not pack_id.strip():
                return hooks.error_json("pack_id must not be empty")
            if not label.strip():
                return hooks.error_json("label must not be empty")
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                return hooks.error_json("payload_json must be valid JSON")
            if not isinstance(payload, dict):
                return hooks.error_json("payload_json must encode an object")
            items = payload.get("items")
            if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
                return hooks.error_json("payload_json must include an items list of objects")
            poly = hooks.get_polylogue()
            created = await poly.create_recall_pack(
                pack_id.strip(),
                label.strip(),
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok",
                    key=pack_id.strip(),
                    outcome="added" if created else "updated",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_recall_pack", run)

    @mcp.tool()
    async def delete_recall_pack(pack_id: str) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            deleted = await poly.delete_recall_pack(pack_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "recall_pack_not_found",
                    key=pack_id,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_recall_pack", run)

    @mcp.tool()
    async def list_workspaces() -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_workspaces()
            items = tuple(_workspace_payload(row) for row in rows)
            return hooks.json_payload(MCPReaderWorkspaceListPayload(items=items, total=len(items)))

        return await hooks.async_safe_call("list_workspaces", run)

    @mcp.tool()
    async def save_workspace(
        workspace_id: str,
        name: str,
        mode: str = "tabs",
        open_targets_json: str = "[]",
        layout_json: str = "{}",
        active_target_json: str = "{}",
    ) -> str:
        async def run() -> str:
            if not workspace_id.strip():
                return hooks.error_json("workspace_id must not be empty")
            if not name.strip():
                return hooks.error_json("name must not be empty")
            if mode not in {"tabs", "stack", "compare", "timeline"}:
                return hooks.error_json("mode must be one of: tabs, stack, compare, timeline")
            try:
                open_targets = json.loads(open_targets_json)
            except json.JSONDecodeError:
                return hooks.error_json("open_targets_json must be valid JSON")
            if not isinstance(open_targets, list) or not all(isinstance(item, dict) for item in open_targets):
                return hooks.error_json("open_targets_json must encode a list of objects")
            try:
                layout = json.loads(layout_json)
            except json.JSONDecodeError:
                return hooks.error_json("layout_json must be valid JSON")
            if not isinstance(layout, dict):
                return hooks.error_json("layout_json must encode an object")
            try:
                active_target = json.loads(active_target_json)
            except json.JSONDecodeError:
                return hooks.error_json("active_target_json must be valid JSON")
            if not isinstance(active_target, dict):
                return hooks.error_json("active_target_json must encode an object")
            poly = hooks.get_polylogue()
            created = await poly.save_workspace(
                workspace_id.strip(),
                name.strip(),
                mode,
                json.dumps(open_targets, sort_keys=True, separators=(",", ":")),
                json.dumps(layout, sort_keys=True, separators=(",", ":")),
                json.dumps(active_target, sort_keys=True, separators=(",", ":")),
            )
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok",
                    key=workspace_id.strip(),
                    outcome="added" if created else "updated",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_workspace", run)

    @mcp.tool()
    async def delete_workspace(workspace_id: str) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            deleted = await poly.delete_workspace(workspace_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if deleted else "not_found",
                    detail=None if deleted else "workspace_not_found",
                    key=workspace_id,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_workspace", run)

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
