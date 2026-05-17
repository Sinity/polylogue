"""Maintenance and export MCP tool registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPMutationStatusPayload, MCPRootPayload
from polylogue.mcp.query_contracts import (
    MCPContentProjectionRequest,
    MCPConversationQueryRequest,
    MCPToolLimit,
    MCPToolOffset,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_maintenance_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def maintenance_preview(targets: list[str] | None = None) -> str:
        """Dry-run summary of maintenance backfill targets.

        Returns the shared :class:`MaintenanceOperationEnvelope` so the
        result shape is byte-for-byte identical to the CLI ``polylogue
        maintenance plan --output-format json`` and the daemon HTTP
        ``POST /api/maintenance/plan`` responses.
        """

        async def run() -> str:
            from polylogue.config import Config
            from polylogue.maintenance.envelope import envelope_from_operation
            from polylogue.maintenance.planner import preview_backfill
            from polylogue.paths import archive_root, render_root

            config = Config(
                archive_root=archive_root(),
                render_root=render_root(),
                sources=[],
            )
            resolved = tuple(targets or ())
            result = preview_backfill(config, targets=resolved)
            envelope = envelope_from_operation(result, origin="mcp", mode="preview")
            return hooks.json_payload(envelope)

        return await hooks.async_safe_call("maintenance_preview", run)

    @mcp.tool()
    async def maintenance_execute(targets: list[str] | None = None, dry_run: bool = False) -> str:
        """Run (or dry-run) maintenance backfill targets.

        Returns the shared :class:`MaintenanceOperationEnvelope` with
        ``mode="execute"``. Per-target failures are isolated and
        reported through the bounded ``failure_samples`` envelope.
        """

        async def run() -> str:
            from polylogue.config import Config
            from polylogue.maintenance.envelope import envelope_from_operation
            from polylogue.maintenance.planner import execute_backfill
            from polylogue.paths import archive_root, render_root

            config = Config(
                archive_root=archive_root(),
                render_root=render_root(),
                sources=[],
            )
            resolved = tuple(targets or ())
            result = execute_backfill(config, targets=resolved, dry_run=dry_run)
            envelope = envelope_from_operation(result, origin="mcp", mode="execute")
            return hooks.json_payload(envelope)

        return await hooks.async_safe_call("maintenance_execute", run)

    @mcp.tool()
    async def rebuild_index() -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            success = await ops.rebuild_index()
            status_info = await ops.get_index_status()
            index_exists_value = status_info.get("exists", False)
            indexed_messages_value = status_info.get("count", 0)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok" if success else "failed",
                    index_exists=bool(index_exists_value),
                    indexed_messages=int(indexed_messages_value),
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("rebuild_index", run)

    @mcp.tool()
    async def update_index(conversation_ids: list[str]) -> str:
        async def run() -> str:
            ops = hooks.get_archive_ops()
            success = await ops.update_index(conversation_ids)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok" if success else "failed",
                    conversation_count=len(conversation_ids),
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("update_index", run)

    @mcp.tool()
    async def export_conversation(
        id: str,
        format: str = "markdown",
        no_code_blocks: bool = False,
        no_tool_calls: bool = False,
        no_tool_outputs: bool = False,
        no_file_reads: bool = False,
        prose_only: bool = False,
    ) -> str:
        async def run() -> str:
            from polylogue.rendering.formatting import format_conversation, normalize_conversation_output_format

            projection = MCPContentProjectionRequest(
                no_code_blocks=no_code_blocks,
                no_tool_calls=no_tool_calls,
                no_tool_outputs=no_tool_outputs,
                no_file_reads=no_file_reads,
                prose_only=prose_only,
            ).build_projection()

            poly = hooks.get_polylogue()
            conv = await poly.get_conversation(id, content_projection=projection)
            if conv is None:
                return hooks.error_json(f"Conversation not found: {id}", code="not_found")
            fmt = normalize_conversation_output_format(format)
            return format_conversation(conv, fmt, None, content_projection=projection)

        return await hooks.async_safe_call("export_conversation", run)

    @mcp.tool()
    async def rebuild_session_insights(conversation_ids: list[str] | None = None) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            counts = await poly.rebuild_insights(conversation_ids=conversation_ids)
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

        return await hooks.async_safe_call("rebuild_session_insights", run)

    @mcp.tool()
    async def export_query_results(
        query: str | None = None,
        format: str = "markdown",
        limit: MCPToolLimit = 10,
        offset: MCPToolOffset = 0,
        retrieval_lane: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        tag: str | None = None,
        title: str | None = None,
        referenced_path: str | None = None,
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
        no_code_blocks: bool = False,
        no_tool_calls: bool = False,
        no_tool_outputs: bool = False,
        no_file_reads: bool = False,
        prose_only: bool = False,
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
                referenced_path=referenced_path,
                action=action,
                exclude_action=exclude_action,
                action_sequence=action_sequence,
                action_text=action_text,
                tool=tool,
                exclude_tool=exclude_tool,
                sort=sort,
                limit=limit,
                offset=offset,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                min_messages=min_messages,
                min_words=min_words,
            ).build_spec(hooks.clamp_limit)
            projection = MCPContentProjectionRequest(
                no_code_blocks=no_code_blocks,
                no_tool_calls=no_tool_calls,
                no_tool_outputs=no_tool_outputs,
                no_file_reads=no_file_reads,
                prose_only=prose_only,
            ).build_projection()
            conversations = await hooks.get_archive_ops().query_conversations(spec, content_projection=projection)
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
                                "content": format_conversation(conversation, fmt, None, content_projection=projection),
                            }
                            for conversation in conversations
                        ],
                    }
                )
            )

        return await hooks.async_safe_call("export_query_results", run)


__all__ = ["register_maintenance_tools"]
