"""Maintenance and export MCP tool registration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.mcp.payloads import MCPMutationStatusPayload, MCPRootPayload
from polylogue.mcp.query_contracts import (
    MCPContentProjectionRequest,
    MCPSessionQueryRequest,
    MCPToolLimit,
    MCPToolOffset,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def _build_mcp_scope_filter(
    *,
    session_ids: list[str] | None,
    origin: str | None,
    source_family: str | None,
    source_root: str | None,
    raw_artifact_id: str | None,
    since: str | None,
    until: str | None,
    failure_kind: str | None,
    parser_version: str | None,
) -> MaintenanceScopeFilter:
    """Translate MCP tool args into a :class:`MaintenanceScopeFilter`.

    Shared by ``maintenance_preview`` and ``maintenance_execute`` so the
    two MCP tools never drift on how they parse scope filters.
    """
    time_range: tuple[datetime, datetime] | None
    if since is not None or until is not None:
        if since is None or until is None:
            raise ValueError("since and until must be supplied together")
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00") if since.endswith("Z") else since)
        until_dt = datetime.fromisoformat(until.replace("Z", "+00:00") if until.endswith("Z") else until)
        time_range = (since_dt, until_dt)
    else:
        time_range = None

    return MaintenanceScopeFilter(
        session_ids=tuple(session_ids) if session_ids else None,
        provider=provider_from_origin(Origin(origin)).value if origin is not None else None,
        source_family=source_family,
        source_root=Path(source_root) if source_root else None,
        raw_artifact_id=raw_artifact_id,
        time_range=time_range,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )


def register_maintenance_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def maintenance_preview(
        targets: list[str] | None = None,
        session_ids: list[str] | None = None,
        origin: str | None = None,
        source_family: str | None = None,
        source_root: str | None = None,
        raw_artifact_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        failure_kind: str | None = None,
        parser_version: str | None = None,
    ) -> str:
        """Dry-run summary of maintenance backfill targets.

        Returns the shared :class:`MaintenanceOperationEnvelope` so the
        result shape is byte-for-byte identical to the CLI ``polylogue
        maintenance plan --output-format json`` and the daemon HTTP
        ``POST /api/maintenance/plan`` responses. The scope-filter
        parameters mirror the CLI ``--session-id``, ``--origin``,
        ``--source-family``, ``--source-root``, ``--raw-artifact``,
        ``--since``/``--until``, ``--failure-kind``, and
        ``--parser-version`` flags.
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
            scope_filter = _build_mcp_scope_filter(
                session_ids=session_ids,
                origin=origin,
                source_family=source_family,
                source_root=source_root,
                raw_artifact_id=raw_artifact_id,
                since=since,
                until=until,
                failure_kind=failure_kind,
                parser_version=parser_version,
            )
            result = preview_backfill(config, targets=resolved, scope_filter=scope_filter)
            envelope = envelope_from_operation(result, origin="mcp", mode="preview")
            return hooks.json_payload(envelope)

        return await hooks.async_safe_call("maintenance_preview", run)

    @mcp.tool()
    async def maintenance_execute(
        targets: list[str] | None = None,
        dry_run: bool = False,
        session_ids: list[str] | None = None,
        origin: str | None = None,
        source_family: str | None = None,
        source_root: str | None = None,
        raw_artifact_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        failure_kind: str | None = None,
        parser_version: str | None = None,
    ) -> str:
        """Run (or dry-run) maintenance backfill targets.

        Returns the shared :class:`MaintenanceOperationEnvelope` with
        ``mode="execute"``. Per-target failures are isolated and
        reported through the bounded ``failure_samples`` envelope. The
        scope-filter parameters mirror the CLI flags on
        ``polylogue ops maintenance run``.
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
            scope_filter = _build_mcp_scope_filter(
                session_ids=session_ids,
                origin=origin,
                source_family=source_family,
                source_root=source_root,
                raw_artifact_id=raw_artifact_id,
                since=since,
                until=until,
                failure_kind=failure_kind,
                parser_version=parser_version,
            )
            result = execute_backfill(config, targets=resolved, dry_run=dry_run, scope_filter=scope_filter)
            envelope = envelope_from_operation(result, origin="mcp", mode="execute")
            return hooks.json_payload(envelope)

        return await hooks.async_safe_call("maintenance_execute", run)

    @mcp.tool()
    async def maintenance_status(operation_id: str) -> str:
        """Return one persisted maintenance operation snapshot by id (#1197).

        Wraps the shared
        :class:`~polylogue.maintenance.envelope.MaintenanceOperationEnvelope`
        with the state-file ``updated_at`` and ``state_path`` so the
        result is byte-shape-identical to the daemon HTTP
        ``GET /api/maintenance/status/<id>`` response.
        """

        async def run() -> str:
            from polylogue.config import Config
            from polylogue.maintenance.envelope import envelope_from_operation
            from polylogue.maintenance.registry import MaintenanceOperationRegistry
            from polylogue.paths import archive_root, render_root

            config = Config(
                archive_root=archive_root(),
                render_root=render_root(),
                sources=[],
            )
            registry = MaintenanceOperationRegistry(config=config)
            record = registry.get_operation(operation_id)
            if record is None:
                return hooks.error_json(f"Operation not found: {operation_id}", code="not_found")
            envelope = envelope_from_operation(record.operation, origin="mcp", mode="execute")
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "envelope": envelope.to_dict(),
                        "updated_at": record.updated_at,
                        "state_path": str(record.state_path),
                    }
                )
            )

        return await hooks.async_safe_call("maintenance_status", run)

    @mcp.tool()
    async def maintenance_list() -> str:
        """List every persisted maintenance operation snapshot (#1197).

        Returns a bounded envelope ``{"items": [...], "total": N}`` where
        each item is the same shape as :func:`maintenance_status`. Use
        when the operator (or another agent) needs to see in-flight and
        recently failed operations without knowing their ids.
        """

        async def run() -> str:
            from polylogue.config import Config
            from polylogue.maintenance.envelope import envelope_from_operation
            from polylogue.maintenance.registry import MaintenanceOperationRegistry
            from polylogue.paths import archive_root, render_root

            config = Config(
                archive_root=archive_root(),
                render_root=render_root(),
                sources=[],
            )
            registry = MaintenanceOperationRegistry(config=config)
            records = registry.list_operations()
            items = [
                {
                    "envelope": envelope_from_operation(r.operation, origin="mcp", mode="execute").to_dict(),
                    "updated_at": r.updated_at,
                    "state_path": str(r.state_path),
                }
                for r in records
            ]
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "items": items,
                        "total": len(items),
                    }
                )
            )

        return await hooks.async_safe_call("maintenance_list", run)

    @mcp.tool()
    async def rebuild_index() -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            success = await poly.rebuild_index()
            status_info = await poly.get_index_status()
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
    async def update_index(session_ids: list[str]) -> str:
        async def run() -> str:
            success = await hooks.get_polylogue().update_index(session_ids)
            return hooks.json_payload(
                MCPMutationStatusPayload(
                    status="ok" if success else "failed",
                    session_count=len(session_ids),
                ),
                exclude_none=True,
            )

        return await hooks.async_safe_call("update_index", run)

    @mcp.tool()
    async def export_session(
        id: str,
        format: str = "markdown",
        no_code_blocks: bool = False,
        no_tool_calls: bool = False,
        no_tool_outputs: bool = False,
        no_file_reads: bool = False,
        prose_only: bool = False,
    ) -> str:
        async def run() -> str:
            from polylogue.rendering.formatting import format_session, normalize_session_output_format

            projection = MCPContentProjectionRequest(
                no_code_blocks=no_code_blocks,
                no_tool_calls=no_tool_calls,
                no_tool_outputs=no_tool_outputs,
                no_file_reads=no_file_reads,
                prose_only=prose_only,
            ).build_projection()

            poly = hooks.get_polylogue()
            conv = await poly.get_session(id, content_projection=projection)
            if conv is None:
                return hooks.error_json(f"Session not found: {id}", code="not_found")
            fmt = normalize_session_output_format(format)
            return format_session(conv, fmt, None, content_projection=projection)

        return await hooks.async_safe_call("export_session", run)

    @mcp.tool()
    async def rebuild_session_insights(session_ids: list[str] | None = None) -> str:
        async def run() -> str:
            poly = hooks.get_polylogue()
            counts = await poly.rebuild_insights(session_ids=session_ids)
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "status": "ok",
                        "session_count": len(session_ids) if session_ids is not None else None,
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
        origin: str | None = None,
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
            from polylogue.rendering.formatting import format_session, normalize_session_output_format

            fmt = normalize_session_output_format(format)
            spec = MCPSessionQueryRequest(
                query=query,
                retrieval_lane=retrieval_lane,
                origin=origin,
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
            sessions = await spec.list(hooks.get_config())
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "count": len(sessions),
                        "format": fmt,
                        "exports": [
                            {
                                "session_id": str(session.id),
                                "origin": session.origin.value,
                                "title": session.display_title,
                                "content": format_session(session, fmt, None, content_projection=projection),
                            }
                            for session in sessions
                        ],
                    }
                )
            )

        return await hooks.async_safe_call("export_query_results", run)


__all__ = ["register_maintenance_tools"]
