"""Mutation-oriented MCP tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polylogue.annotations.importer import (
    AnnotationBatchImportError,
    AnnotationBatchImportRequest,
)
from polylogue.annotations.importer import (
    import_annotation_batch as run_annotation_batch_import,
)
from polylogue.core.user_state_targets import MARK_TYPE_NAMES, TARGET_SESSION, is_mark_type_supported
from polylogue.mcp.archive_support import blackboard_note_payload
from polylogue.mcp.mutation_support import page_items, resolve_session_or_error
from polylogue.mcp.payloads import (
    MCPTagCountsPayload,
    MCPUserMarkListPayload,
    MCPUserMarkPayload,
    MutationResultPayload,
)
from polylogue.mcp.query_contracts import MCPToolLimit, MCPToolOffset
from polylogue.mcp.server_personal_state_tools import register_personal_state_tools

if TYPE_CHECKING:
    from polylogue.mcp.declarations.adapter import ToolRegistrar
    from polylogue.mcp.server_support import ServerCallbacks


def _mark_type_error(hooks: ServerCallbacks, mark_type: str) -> str | None:
    if is_mark_type_supported(mark_type):
        return None
    return hooks.error_json(f"mark_type must be one of: {', '.join(MARK_TYPE_NAMES)}", detail=mark_type)


def _none_if_empty(value: str | None) -> str | None:
    return value if value else None


def register_mutation_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def capture_assertion_candidate(
        body_text: str,
        author_ref: str,
        kind: str = "note",
        refs: list[str] | None = None,
        scope_refs: list[str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """Capture a terminal-equivalent assertion candidate for later judgment."""

        async def run() -> str:
            from pathlib import Path

            from polylogue.api.archive import candidate_capture_kind

            try:
                if not author_ref.startswith("agent:") or author_ref == "agent:":
                    raise ValueError("author_ref must be an agent:<session> ref")
                result = await hooks.get_polylogue().capture_assertion_candidate(
                    body_text=body_text,
                    kind=candidate_capture_kind(kind),
                    refs=tuple(refs or ()),
                    scope_refs=tuple(scope_refs or ()),
                    cwd=Path(cwd) if cwd is not None else None,
                    author_ref=author_ref,
                    author_kind="agent",
                )
            except ValueError as exc:
                return hooks.error_json(str(exc), code="invalid_candidate_capture")
            return hooks.json_payload(result, exclude_none=False)

        return await hooks.async_safe_call("capture_assertion_candidate", run)

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
            resolved, err = await resolve_session_or_error(hooks, session_id)
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
            resolved, err = await resolve_session_or_error(hooks, session_id)
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
        limit: MCPToolLimit = 50,
        offset: MCPToolOffset = 0,
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
            clamped_limit = hooks.clamp_limit(limit)
            page, total, page_offset, next_offset = page_items(items, limit=clamped_limit, offset=offset)
            with hooks.response_context(
                "list_marks",
                {
                    "mark_type": mark_type,
                    "session_id": session_id,
                    "target_type": target_type,
                    "target_id": target_id,
                    "message_id": message_id,
                    "limit": clamped_limit,
                    "offset": page_offset,
                },
            ):
                return hooks.json_payload(
                    MCPUserMarkListPayload(
                        items=page,
                        total=total,
                        limit=clamped_limit,
                        offset=page_offset,
                        next_offset=next_offset,
                    )
                )

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
            resolved, err = await resolve_session_or_error(hooks, session_id)
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
            resolved, err = await resolve_session_or_error(hooks, session_id)
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

    register_personal_state_tools(mcp, hooks)

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


def register_assertion_review_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    """Register promotion tools only for authenticated review capability."""

    @mcp.tool()
    async def judge_assertion_candidate(
        candidate_ref: str,
        decision: Literal["accept", "reject", "defer", "supersede"],
        reason: str | None = None,
        inject: bool = False,
        actor_ref: str = "user:local",
        replacement_kind: str | None = None,
        replacement_body_text: str | None = None,
        replacement_value: object | None = None,
    ) -> str:
        """Judge one candidate; caller text is provenance, not authorization."""

        from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionBulkJudgmentItemEnvelope

        async def run() -> str:
            payload = await hooks.get_polylogue().judge_assertion_candidates(
                items=(
                    ArchiveAssertionBulkJudgmentItemEnvelope(
                        candidate_ref=candidate_ref,
                        decision=decision,
                        reason=reason,
                        inject=inject,
                        actor_ref=actor_ref,
                        replacement_kind=replacement_kind,
                        replacement_body_text=replacement_body_text,
                        replacement_value=replacement_value,
                    ),
                )
            )
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("judge_assertion_candidate", run)

    @mcp.tool()
    async def judge_assertion_candidates(
        items: list[dict[str, object]],
        actor_ref: str = "user:local",
    ) -> str:
        """Bulk judge candidates with independently reported partial success."""

        from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionBulkJudgmentItemEnvelope

        def make_item(item: dict[str, object]) -> ArchiveAssertionBulkJudgmentItemEnvelope:
            candidate_ref = item.get("candidate_ref")
            decision = item.get("decision")
            if not isinstance(candidate_ref, str) or not isinstance(decision, str):
                raise ValueError("each judgment requires string candidate_ref and decision")
            inject = item.get("inject", False)
            if type(inject) is not bool:
                raise ValueError("each judgment requires boolean inject")
            reason = item.get("reason")
            replacement_kind = item.get("replacement_kind")
            replacement_body_text = item.get("replacement_body_text")
            if replacement_kind is not None and not isinstance(replacement_kind, str):
                raise ValueError("replacement_kind must be a string when provided")
            if replacement_body_text is not None and not isinstance(replacement_body_text, str):
                raise ValueError("replacement_body_text must be a string when provided")
            return ArchiveAssertionBulkJudgmentItemEnvelope(
                candidate_ref=candidate_ref,
                decision=decision,
                reason=reason if isinstance(reason, str) else None,
                inject=inject,
                actor_ref=actor_ref,
                replacement_kind=replacement_kind,
                replacement_body_text=replacement_body_text,
                replacement_value=item.get("replacement_value"),
            )

        async def run() -> str:
            try:
                judgments = tuple(make_item(item) for item in items)
            except ValueError as exc:
                return hooks.error_json(str(exc), code="invalid_assertion_judgment")
            payload = await hooks.get_polylogue().judge_assertion_candidates(items=judgments)
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("judge_assertion_candidates", run)


__all__ = ["register_assertion_review_tools", "register_mutation_tools"]
