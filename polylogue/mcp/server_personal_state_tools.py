"""MCP registrations for durable personal-state records.

These tools share one contract boundary: they validate and serialize user-tier
records while leaving the role gate to ``server_tools.register_tools``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from hashlib import sha256
from typing import TYPE_CHECKING

from polylogue.core.user_state_targets import TARGET_SESSION
from polylogue.mcp.mutation_support import page_items, resolve_session_or_error
from polylogue.mcp.payloads import (
    MCPMetadataPayload,
    MCPReaderWorkspaceListPayload,
    MCPReaderWorkspacePayload,
    MCPRecallPackListPayload,
    MCPRecallPackPayload,
    MCPRootPayload,
    MCPSavedViewListPayload,
    MCPSavedViewPayload,
    MCPUserAnnotationListPayload,
    MCPUserAnnotationPayload,
    MutationResultPayload,
)
from polylogue.mcp.query_contracts import MCPCharacterLimit, MCPToolLimit, MCPToolOffset

if TYPE_CHECKING:
    from polylogue.mcp.declarations.adapter import ToolRegistrar
    from polylogue.mcp.server_support import ServerCallbacks


def _default_saved_view_id(name: str, query_json: str) -> str:
    digest = sha256(f"{name}\0{query_json}".encode()).hexdigest()
    return f"saved-view-{digest[:16]}"


def _bounded_text(value: str | None, max_chars: int | None) -> str | None:
    if value is None or max_chars is None or len(value) <= max_chars:
        return value
    return value[:max_chars]


def _bounded_json_value(value: object, max_chars: int | None, *, depth: int = 0) -> object:
    if max_chars is None:
        return value
    if isinstance(value, str):
        return _bounded_text(value, max_chars)
    if depth >= 2:
        return {"truncated": True} if isinstance(value, Mapping | Sequence) else value
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_json_value(item, max_chars, depth=depth + 1) for key, item in list(value.items())[:4]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded_json_value(item, max_chars, depth=depth + 1) for item in value[:4]]
    return value


def _bounded_sequence(value: Sequence[object], max_chars: int | None) -> tuple[object, ...]:
    bounded = _bounded_json_value(value, max_chars)
    if isinstance(bounded, Sequence) and not isinstance(bounded, (str, bytes, bytearray)):
        return tuple(bounded)
    return ()


def _saved_view_payload(row: dict[str, str]) -> MCPSavedViewPayload:
    try:
        query = json.loads(row["query_json"])
    except (json.JSONDecodeError, TypeError):
        query = {}
    if not isinstance(query, dict):
        query = {}
    return MCPSavedViewPayload(view_id=row["view_id"], name=row["name"], query=query, created_at=row["created_at"])


def _recall_pack_payload(row: dict[str, str]) -> MCPRecallPackPayload:
    try:
        session_ids = json.loads(row["session_ids_json"])
    except (json.JSONDecodeError, TypeError):
        session_ids = []
    if not isinstance(session_ids, list):
        session_ids = []
    try:
        payload = json.loads(row["payload_json"])
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return MCPRecallPackPayload(
        pack_id=row["pack_id"],
        label=row["label"],
        session_ids=tuple(str(item) for item in session_ids),
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


def _annotation_payload(row: dict[str, str]) -> MCPUserAnnotationPayload:
    message_id = row.get("message_id") or None
    return MCPUserAnnotationPayload(
        annotation_id=row["annotation_id"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        session_id=row["session_id"],
        message_id=message_id,
        note_text=row["note_text"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def register_personal_state_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    """Register annotations, views, recall packs, workspaces, metadata and corrections."""

    @mcp.tool()
    async def list_annotations(
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
        limit: MCPToolLimit = 50,
        offset: MCPToolOffset = 0,
        max_chars_per_item: MCPCharacterLimit = None,
    ) -> str:
        async def run() -> str:
            rows = await hooks.get_polylogue().list_annotations(
                session_id=session_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
            items = tuple(
                _annotation_payload({**row, "note_text": _bounded_text(row["note_text"], max_chars_per_item) or ""})
                for row in rows
            )
            clamped_limit = hooks.clamp_limit(limit)
            page, total, page_offset, next_offset = page_items(items, limit=clamped_limit, offset=offset)
            with hooks.response_context(
                "list_annotations",
                {
                    "session_id": session_id,
                    "target_type": target_type,
                    "target_id": target_id,
                    "message_id": message_id,
                    "limit": clamped_limit,
                    "offset": page_offset,
                    "max_chars_per_item": max_chars_per_item,
                },
            ):
                return hooks.json_payload(
                    MCPUserAnnotationListPayload(
                        items=page, total=total, limit=clamped_limit, offset=page_offset, next_offset=next_offset
                    )
                )

        return await hooks.async_safe_call("list_annotations", run, session_id=session_id)

    @mcp.tool()
    async def save_annotation(
        annotation_id: str,
        session_id: str,
        note_text: str,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            if not annotation_id.strip():
                return hooks.error_json("annotation_id must not be empty")
            if not note_text.strip():
                return hooks.error_json("note_text must not be empty")
            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            created = await hooks.get_polylogue().save_annotation(
                annotation_id, resolved, note_text, target_type=target_type, target_id=target_id, message_id=message_id
            )
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok", session_id=resolved, key=annotation_id, outcome="added" if created else "updated"
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_annotation", run, session_id=session_id)

    @mcp.tool()
    async def delete_annotation(annotation_id: str) -> str:
        async def run() -> str:
            deleted = await hooks.get_polylogue().delete_annotation(annotation_id)
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
    async def list_saved_views(
        limit: MCPToolLimit = 50, offset: MCPToolOffset = 0, max_chars_per_item: MCPCharacterLimit = None
    ) -> str:
        async def run() -> str:
            rows = await hooks.get_polylogue().list_views()
            items = tuple(
                (payload := _saved_view_payload(row)).model_copy(
                    update={"query": _bounded_json_value(payload.query, max_chars_per_item)}
                )
                for row in rows
            )
            clamped_limit = hooks.clamp_limit(limit)
            page, total, page_offset, next_offset = page_items(items, limit=clamped_limit, offset=offset)
            with hooks.response_context(
                "list_saved_views",
                {"limit": clamped_limit, "offset": page_offset, "max_chars_per_item": max_chars_per_item},
            ):
                return hooks.json_payload(
                    MCPSavedViewListPayload(
                        items=page, total=total, limit=clamped_limit, offset=page_offset, next_offset=next_offset
                    )
                )

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
            from polylogue.archive.query.spec import SessionQuerySpec

            try:
                SessionQuerySpec.from_params(query, strict=True)
            except Exception as exc:
                return hooks.error_json(
                    "query_json is not a valid SessionQuerySpec", detail=f"{type(exc).__name__}: {exc}"
                )
            canonical_query_json = json.dumps(query, sort_keys=True, separators=(",", ":"))
            saved_id = view_id or _default_saved_view_id(name.strip(), canonical_query_json)
            created = await hooks.get_polylogue().save_view(saved_id, name.strip(), canonical_query_json)
            return hooks.json_payload(
                MutationResultPayload(status="ok", key=saved_id, outcome="added" if created else "updated"),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_saved_view", run)

    @mcp.tool()
    async def delete_saved_view(view_id: str) -> str:
        async def run() -> str:
            deleted = await hooks.get_polylogue().delete_view(view_id)
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
    async def list_recall_packs(
        limit: MCPToolLimit = 50, offset: MCPToolOffset = 0, max_chars_per_item: MCPCharacterLimit = None
    ) -> str:
        async def run() -> str:
            rows = await hooks.get_polylogue().list_recall_packs()
            items = tuple(
                (payload := _recall_pack_payload(row)).model_copy(
                    update={
                        "session_ids": tuple(
                            str(item) for item in _bounded_sequence(payload.session_ids, max_chars_per_item)
                        ),
                        "payload": _bounded_json_value(payload.payload, max_chars_per_item),
                    }
                )
                for row in rows
            )
            clamped_limit = hooks.clamp_limit(limit)
            page, total, page_offset, next_offset = page_items(items, limit=clamped_limit, offset=offset)
            with hooks.response_context(
                "list_recall_packs",
                {"limit": clamped_limit, "offset": page_offset, "max_chars_per_item": max_chars_per_item},
            ):
                return hooks.json_payload(
                    MCPRecallPackListPayload(
                        items=page, total=total, limit=clamped_limit, offset=page_offset, next_offset=next_offset
                    )
                )

        return await hooks.async_safe_call("list_recall_packs", run)

    @mcp.tool()
    async def save_recall_pack(pack_id: str, label: str, payload_json: str = "{}") -> str:
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
            created = await hooks.get_polylogue().create_recall_pack(
                pack_id.strip(), label.strip(), json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            return hooks.json_payload(
                MutationResultPayload(status="ok", key=pack_id.strip(), outcome="added" if created else "updated"),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_recall_pack", run)

    @mcp.tool()
    async def delete_recall_pack(pack_id: str) -> str:
        async def run() -> str:
            deleted = await hooks.get_polylogue().delete_recall_pack(pack_id)
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
    async def list_workspaces(
        limit: MCPToolLimit = 50, offset: MCPToolOffset = 0, max_chars_per_item: MCPCharacterLimit = None
    ) -> str:
        async def run() -> str:
            rows = await hooks.get_polylogue().list_workspaces()
            items = tuple(
                (payload := _workspace_payload(row)).model_copy(
                    update={
                        "open_targets": tuple(
                            item
                            for item in _bounded_sequence(payload.open_targets, max_chars_per_item)
                            if isinstance(item, dict)
                        ),
                        "layout": _bounded_json_value(payload.layout, max_chars_per_item),
                        "active_target": _bounded_json_value(payload.active_target, max_chars_per_item),
                    }
                )
                for row in rows
            )
            clamped_limit = hooks.clamp_limit(limit)
            page, total, page_offset, next_offset = page_items(items, limit=clamped_limit, offset=offset)
            with hooks.response_context(
                "list_workspaces",
                {"limit": clamped_limit, "offset": page_offset, "max_chars_per_item": max_chars_per_item},
            ):
                return hooks.json_payload(
                    MCPReaderWorkspaceListPayload(
                        items=page, total=total, limit=clamped_limit, offset=page_offset, next_offset=next_offset
                    )
                )

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
            created = await hooks.get_polylogue().save_workspace(
                workspace_id.strip(),
                name.strip(),
                mode,
                json.dumps(open_targets, sort_keys=True, separators=(",", ":")),
                json.dumps(layout, sort_keys=True, separators=(",", ":")),
                json.dumps(active_target, sort_keys=True, separators=(",", ":")),
            )
            return hooks.json_payload(
                MutationResultPayload(status="ok", key=workspace_id.strip(), outcome="added" if created else "updated"),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_workspace", run)

    @mcp.tool()
    async def delete_workspace(workspace_id: str) -> str:
        async def run() -> str:
            deleted = await hooks.get_polylogue().delete_workspace(workspace_id)
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
    async def get_metadata(session_id: str) -> str:
        async def run() -> str:
            metadata = await hooks.get_polylogue().get_metadata(session_id)
            return hooks.json_payload(MCPMetadataPayload.from_document(metadata))  # type: ignore[arg-type]

        return await hooks.async_safe_call("get_metadata", run, session_id=session_id)

    @mcp.tool()
    async def set_metadata(session_id: str, key: str, value: str) -> str:
        async def run() -> str:
            from polylogue.api.archive import SessionNotFoundError
            from polylogue.surfaces.payloads import MetadataKeyValidationError, validate_metadata_key

            key_error = validate_metadata_key(key)
            if key_error is not None:
                return hooks.error_json(key_error, session_id=session_id, code="invalid_key")
            parsed_value: object = value
            with suppress(json.JSONDecodeError, TypeError):
                parsed_value = json.loads(value)
            try:
                result = await hooks.get_polylogue().set_metadata(session_id, key, str(parsed_value))
            except MetadataKeyValidationError as exc:
                return hooks.error_json(str(exc), session_id=session_id, code="invalid_key")
            except SessionNotFoundError:
                return hooks.error_json("Session not found", code="not_found", session_id=session_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if result.outcome == "set" else "unchanged",
                    session_id=result.session_id,
                    key=result.key,
                    detail=result.detail,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("set_metadata", run, session_id=session_id)

    @mcp.tool()
    async def delete_metadata(session_id: str, key: str) -> str:
        async def run() -> str:
            from polylogue.api.archive import SessionNotFoundError
            from polylogue.surfaces.payloads import MetadataKeyValidationError, validate_metadata_key

            key_error = validate_metadata_key(key)
            if key_error is not None:
                return hooks.error_json(key_error, session_id=session_id, code="invalid_key")
            try:
                result = await hooks.get_polylogue().delete_metadata(session_id, key)
            except MetadataKeyValidationError as exc:
                return hooks.error_json(str(exc), session_id=session_id, code="invalid_key")
            except SessionNotFoundError:
                return hooks.error_json("Session not found", code="not_found", session_id=session_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if result.outcome == "deleted" else "not_found",
                    session_id=result.session_id,
                    key=result.key,
                    detail=result.detail,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_metadata", run, session_id=session_id)

    @mcp.tool()
    async def record_correction(
        session_id: str,
        kind: str,
        payload: dict[str, str],
        note: str | None = None,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> str:
        """Record a user correction targeting a derived insight."""

        async def run() -> str:
            from polylogue.insights.feedback import UnknownCorrectionKindError

            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            try:
                correction = await hooks.get_polylogue().record_correction(
                    resolved, kind, payload, note=note, author_ref=author_ref, author_kind=author_kind
                )
            except UnknownCorrectionKindError as exc:
                return hooks.error_json(str(exc), code="unknown_kind", kind=str(kind or ""))
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok",
                    session_id=correction.session_id,
                    outcome=correction.kind.value,
                    author_ref=author_ref,
                    author_kind=author_kind,
                    detail=correction.note,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("record_correction", run, session_id=session_id)

    @mcp.tool()
    async def list_corrections(
        session_id: str | None = None,
        kind: str | None = None,
        limit: MCPToolLimit = 50,
        offset: MCPToolOffset = 0,
        max_chars_per_correction: MCPCharacterLimit = None,
    ) -> str:
        """List stored learning corrections."""

        async def run() -> str:
            from polylogue.insights.feedback import UnknownCorrectionKindError

            try:
                corrections = await hooks.get_polylogue().list_corrections(session_id=session_id, kind=kind)
            except UnknownCorrectionKindError as exc:
                return hooks.error_json(str(exc), code="unknown_kind", kind=str(kind or ""))
            clamped_limit = hooks.clamp_limit(limit)
            all_items = [
                {
                    "session_id": c.session_id,
                    "kind": c.kind.value,
                    "payload": {
                        key: _bounded_text(value, max_chars_per_correction) for key, value in c.payload.items()
                    },
                    "note": _bounded_text(c.note, max_chars_per_correction),
                    "created_at": c.created_at.isoformat(),
                }
                for c in corrections
            ]
            page_offset = max(0, offset)
            items = all_items[page_offset : page_offset + clamped_limit]
            next_offset = page_offset + len(items) if page_offset + len(items) < len(all_items) else None
            with hooks.response_context(
                "list_corrections",
                {
                    "session_id": session_id,
                    "kind": kind,
                    "limit": clamped_limit,
                    "offset": page_offset,
                    "max_chars_per_correction": max_chars_per_correction,
                },
            ):
                return hooks.json_payload(
                    MCPRootPayload(
                        root={
                            "corrections": items,
                            "total": len(all_items),
                            "limit": clamped_limit,
                            "offset": page_offset,
                            "next_offset": next_offset,
                        }
                    )
                )

        return await hooks.async_safe_call("list_corrections", run, session_id=session_id)

    @mcp.tool()
    async def clear_corrections(session_id: str, kind: str | None = None) -> str:
        """Delete one or all corrections for a session."""

        async def run() -> str:
            from polylogue.insights.feedback import UnknownCorrectionKindError

            resolved, err = await resolve_session_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            try:
                if kind is None:
                    count = await hooks.get_polylogue().clear_corrections(resolved)
                    return hooks.json_payload(
                        MutationResultPayload(
                            status="ok", session_id=resolved, affected_count=count, outcome="cleared"
                        ),
                        exclude_none=True,
                    )
                removed = await hooks.get_polylogue().delete_correction(resolved, kind)
            except UnknownCorrectionKindError as exc:
                return hooks.error_json(str(exc), code="unknown_kind", kind=str(kind or ""))
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if removed else "not_found",
                    session_id=resolved,
                    outcome="deleted" if removed else "not_found",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("clear_corrections", run, session_id=session_id)


__all__ = ["register_personal_state_tools"]
