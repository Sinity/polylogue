"""Mutation-oriented MCP tool registration."""

from __future__ import annotations

import json
from contextlib import suppress
from hashlib import sha256
from typing import TYPE_CHECKING

from polylogue.annotations.importer import (
    AnnotationBatchImportError,
    AnnotationBatchImportRequest,
)
from polylogue.annotations.importer import (
    import_annotation_batch as run_annotation_batch_import,
)
from polylogue.core.user_state_targets import MARK_TYPE_NAMES, TARGET_SESSION, is_mark_type_supported
from polylogue.mcp.archive_support import blackboard_note_payload
from polylogue.mcp.payloads import (
    MCPMetadataPayload,
    MCPReaderWorkspaceListPayload,
    MCPReaderWorkspacePayload,
    MCPRecallPackListPayload,
    MCPRecallPackPayload,
    MCPRootPayload,
    MCPSavedViewListPayload,
    MCPSavedViewPayload,
    MCPTagCountsPayload,
    MCPUserAnnotationListPayload,
    MCPUserAnnotationPayload,
    MCPUserMarkListPayload,
    MCPUserMarkPayload,
    MutationResultPayload,
)
from polylogue.mcp.query_contracts import MCPCharacterLimit, MCPToolLimit, MCPToolOffset

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


async def _resolve_or_error(hooks: ServerCallbacks, session_id: str) -> tuple[str | None, str | None]:
    """Resolve a session ID, returning the canonical ID or an error JSON."""
    summary = await hooks.get_polylogue().get_session_summary(session_id)
    if summary is None:
        return None, hooks.error_json("Session not found", code="not_found", session_id=session_id)
    return str(summary.id), None


def _mark_type_error(hooks: ServerCallbacks, mark_type: str) -> str | None:
    if is_mark_type_supported(mark_type):
        return None
    return hooks.error_json(f"mark_type must be one of: {', '.join(MARK_TYPE_NAMES)}", detail=mark_type)


def _default_saved_view_id(name: str, query_json: str) -> str:
    digest = sha256(f"{name}\0{query_json}".encode()).hexdigest()
    return f"saved-view-{digest[:16]}"


def _bounded_correction_text(value: str | None, max_chars: int | None) -> str | None:
    """Cap user-authored correction fields without changing their structure."""
    if value is None or max_chars is None or len(value) <= max_chars:
        return value
    return value[:max_chars]


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


def _none_if_empty(value: str | None) -> str | None:
    return value if value else None


def _annotation_payload(row: dict[str, str]) -> MCPUserAnnotationPayload:
    return MCPUserAnnotationPayload(
        annotation_id=row["annotation_id"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        session_id=row["session_id"],
        message_id=_none_if_empty(row.get("message_id")),
        note_text=row["note_text"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def register_mutation_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def import_annotation_batch(
        jsonl: str,
        batch_id: str,
        schema_id: str,
        schema_version: int,
        target_ref: str,
        source_result_ref: str,
        actor_ref: str,
        model_ref: str,
        prompt_ref: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Import bounded JSONL labels as provenance-stamped candidates."""

        async def run() -> str:
            try:
                request = AnnotationBatchImportRequest(
                    jsonl=jsonl,
                    batch_id=batch_id,
                    schema_id=schema_id,
                    schema_version=schema_version,
                    target_ref=target_ref,
                    source_result_ref=source_result_ref,
                    actor_ref=actor_ref,
                    model_ref=model_ref,
                    prompt_ref=prompt_ref,
                    metadata={} if metadata is None else metadata,
                )
                result = await run_annotation_batch_import(hooks.get_polylogue(), request)
            except (AnnotationBatchImportError, ValueError) as exc:
                return hooks.error_json(str(exc), code="invalid_annotation_batch")
            return hooks.json_payload(result)

        return await hooks.async_safe_call("import_annotation_batch", run)

    @mcp.tool()
    async def blackboard_post(
        kind: str,
        title: str,
        content: str,
        scope_repo: str | None = None,
        scope_session: str | None = None,
        scope_issue: int | None = None,
        scope_path: str | None = None,
        related_sessions: list[str] | None = None,
        author_ref: str | None = None,
        author_kind: str = "agent",
        evidence_refs: list[str] | None = None,
        staleness: dict[str, object] | None = None,
        context_policy: dict[str, object] | None = None,
    ) -> str:
        """Post a note to the persistent agent blackboard (#1697).

        ``kind`` must be one of finding/blocker/decision/handoff/question/
        observation. Optional scope fields (repo/session/issue/path) and
        related session ids are recorded with the note. Assertion metadata is
        mirrored into the unified assertion row for agent-written notes.
        """

        async def run() -> str:
            poly = hooks.get_polylogue()
            try:
                note = await poly.post_blackboard_note(
                    kind=kind,
                    title=title,
                    content=content,
                    scope_repo=scope_repo,
                    scope_session=scope_session,
                    scope_issue=scope_issue,
                    scope_path=scope_path,
                    related_sessions=tuple(related_sessions or ()),
                    author_ref=author_ref,
                    author_kind=author_kind,
                    evidence_refs=tuple(evidence_refs or ()),
                    staleness=staleness,
                    context_policy=context_policy,
                )
            except ValueError as exc:
                return hooks.error_json(str(exc))
            return hooks.json_payload(blackboard_note_payload(note))

        return await hooks.async_safe_call("blackboard_post", run, session_id=scope_session)

    @mcp.tool()
    async def add_tag(
        session_id: str,
        tag: str,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract
            poly = hooks.get_polylogue()
            result = await poly.add_tag(resolved, tag, author_ref=author_ref, author_kind=author_kind)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if result.outcome == "added" else "unchanged",
                    session_id=resolved,
                    tag=tag,
                    author_ref=author_ref,
                    author_kind=author_kind,
                    detail=result.detail,
                    outcome=result.outcome,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("add_tag", run, session_id=session_id)

    @mcp.tool()
    async def remove_tag(session_id: str, tag: str) -> str:
        async def run() -> str:
            resolved, err = await _resolve_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None  # _resolve_or_error contract
            poly = hooks.get_polylogue()
            result = await poly.remove_tag(resolved, tag)
            return hooks.json_payload(
                MutationResultPayload(
                    status="ok" if result.outcome == "removed" else "not_found",
                    session_id=resolved,
                    tag=tag,
                    detail=result.detail,
                    outcome=result.outcome,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("remove_tag", run, session_id=session_id)

    @mcp.tool()
    async def bulk_tag_sessions(session_ids: list[str], tags: list[str]) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            try:
                result = await poly.bulk_tag_sessions(session_ids, tags)
            except ValueError as exc:
                return hooks.error_json(str(exc))
            return hooks.json_payload(
                MutationResultPayload(
                    status=result.outcome,
                    session_count=result.session_count,
                    tag_count=result.tag_count,
                    affected_count=result.affected_count,
                    skipped_count=result.skipped_count,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("bulk_tag_sessions", run, session_ids=tuple(session_ids))

    @mcp.tool()
    async def list_tags(
        origin: str | None = None,
        limit: MCPToolLimit = 50,
        offset: MCPToolOffset = 0,
    ) -> str:
        """List deterministic tag-count pages without expanding a large tag map."""

        async def run() -> str:
            poly = hooks.get_polylogue()
            tags = await poly.list_tags(origin=origin)
            clamped_limit = hooks.clamp_limit(limit)
            page_offset = max(0, offset)
            page = dict(sorted(tags.items())[page_offset : page_offset + clamped_limit])
            with hooks.response_context(
                "list_tags",
                {
                    "origin": origin,
                    "limit": clamped_limit,
                    "offset": page_offset,
                },
            ):
                return hooks.json_payload(MCPTagCountsPayload(root=page))

        return await hooks.async_safe_call("list_tags", run)

    @mcp.tool()
    async def list_marks(
        mark_type: str | None = None,
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_marks(
                mark_type=mark_type,
                session_id=session_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
            items = tuple(
                MCPUserMarkPayload(
                    target_type=row["target_type"],
                    target_id=row["target_id"],
                    session_id=row["session_id"],
                    message_id=_none_if_empty(row.get("message_id")),
                    mark_type=row["mark_type"],
                    created_at=row["created_at"],
                )
                for row in rows
            )
            return hooks.json_payload(MCPUserMarkListPayload(items=items, total=len(items)))

        return await hooks.async_safe_call("list_marks", run, session_id=session_id)

    @mcp.tool()
    async def add_mark(
        session_id: str,
        mark_type: str,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            mark_error = _mark_type_error(hooks, mark_type)
            if mark_error:
                return mark_error
            resolved, err = await _resolve_or_error(hooks, session_id)
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
                    session_id=resolved,
                    detail=None if created else "already_present",
                    key=mark_type,
                    outcome="added" if created else "no_op",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("add_mark", run, session_id=session_id)

    @mcp.tool()
    async def remove_mark(
        session_id: str,
        mark_type: str,
        target_type: str = TARGET_SESSION,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            mark_error = _mark_type_error(hooks, mark_type)
            if mark_error:
                return mark_error
            resolved, err = await _resolve_or_error(hooks, session_id)
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
                    session_id=resolved,
                    detail=None if deleted else "mark_not_present",
                    key=mark_type,
                    outcome="removed" if deleted else "not_present",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("remove_mark", run, session_id=session_id)

    @mcp.tool()
    async def list_annotations(
        session_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            rows = await poly.list_annotations(
                session_id=session_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            )
            items = tuple(_annotation_payload(row) for row in rows)
            return hooks.json_payload(MCPUserAnnotationListPayload(items=items, total=len(items)))

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
            resolved, err = await _resolve_or_error(hooks, session_id)
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
                    session_id=resolved,
                    key=annotation_id,
                    outcome="added" if created else "updated",
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("save_annotation", run, session_id=session_id)

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

            from polylogue.archive.query.spec import SessionQuerySpec

            try:
                SessionQuerySpec.from_params(query, strict=True)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                return hooks.error_json("query_json is not a valid SessionQuerySpec", detail=detail)

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
    async def get_metadata(session_id: str) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            metadata = await poly.get_metadata(session_id)
            return hooks.json_payload(MCPMetadataPayload.from_document(metadata))  # type: ignore[arg-type]

        return await hooks.async_safe_call("get_metadata", run, session_id=session_id)

    @mcp.tool()
    async def set_metadata(session_id: str, key: str, value: str) -> str:
        async def run() -> str:
            from polylogue.api.archive import SessionNotFoundError
            from polylogue.surfaces.payloads import MetadataKeyValidationError, validate_metadata_key

            # Short-circuit on key validation before opening the facade so
            # the contract is fast and unit tests can exercise the rejection
            # path without standing up an archive.
            key_error = validate_metadata_key(key)
            if key_error is not None:
                return hooks.error_json(
                    key_error,
                    session_id=session_id,
                    code="invalid_key",
                )

            parsed_value: object = value
            with suppress(json.JSONDecodeError, TypeError):
                parsed_value = json.loads(value)
            parsed_str = str(parsed_value)

            poly = hooks.get_polylogue()
            try:
                result = await poly.set_metadata(session_id, key, parsed_str)
            except MetadataKeyValidationError as exc:
                return hooks.error_json(
                    str(exc),
                    session_id=session_id,
                    code="invalid_key",
                )
            except SessionNotFoundError:
                return hooks.error_json(
                    "Session not found",
                    code="not_found",
                    session_id=session_id,
                )
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
                return hooks.error_json(
                    key_error,
                    session_id=session_id,
                    code="invalid_key",
                )

            poly = hooks.get_polylogue()
            try:
                result = await poly.delete_metadata(session_id, key)
            except MetadataKeyValidationError as exc:
                return hooks.error_json(
                    str(exc),
                    session_id=session_id,
                    code="invalid_key",
                )
            except SessionNotFoundError:
                return hooks.error_json(
                    "Session not found",
                    code="not_found",
                    session_id=session_id,
                )
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
    async def delete_session(session_id: str, confirm: bool = False) -> str:
        async def run() -> str:
            if not confirm:
                return hooks.error_json(
                    "Safety guard: set confirm=true to delete",
                    session_id=session_id,
                )

            poly = hooks.get_polylogue()
            result = await poly.delete_session_safe(session_id)
            return hooks.json_payload(
                MutationResultPayload(
                    status="deleted" if result.outcome == "deleted" else "not_found",
                    session_id=result.session_id,
                    detail=result.detail,
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("delete_session", run, session_id=session_id)

    # ------------------------------------------------------------------
    # Learning corrections (#1131)
    # ------------------------------------------------------------------

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

            resolved, err = await _resolve_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            poly = hooks.get_polylogue()
            try:
                correction = await poly.record_correction(
                    resolved,
                    kind,
                    payload,
                    note=note,
                    author_ref=author_ref,
                    author_kind=author_kind,
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

            poly = hooks.get_polylogue()
            try:
                corrections = await poly.list_corrections(session_id=session_id, kind=kind)
            except UnknownCorrectionKindError as exc:
                return hooks.error_json(str(exc), code="unknown_kind", kind=str(kind or ""))
            clamped_limit = hooks.clamp_limit(limit)
            all_items = [
                {
                    "session_id": c.session_id,
                    "kind": c.kind.value,
                    "payload": {
                        key: _bounded_correction_text(value, max_chars_per_correction)
                        for key, value in c.payload.items()
                    },
                    "note": _bounded_correction_text(c.note, max_chars_per_correction),
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

            resolved, err = await _resolve_or_error(hooks, session_id)
            if err:
                return err
            assert resolved is not None
            poly = hooks.get_polylogue()
            try:
                if kind is None:
                    count = await poly.clear_corrections(resolved)
                    return hooks.json_payload(
                        MutationResultPayload(
                            status="ok",
                            session_id=resolved,
                            affected_count=count,
                            outcome="cleared",
                        ),
                        exclude_none=True,
                    )
                removed = await poly.delete_correction(resolved, kind)
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


__all__ = ["register_mutation_tools"]
