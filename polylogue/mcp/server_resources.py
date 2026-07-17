"""Resource registration for the MCP server."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from polylogue.mcp.archive_support import (
    archive_messages_payload,
    archive_session_list_payload,
    archive_summary_payload,
    mcp_archive_root,
)
from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPErrorPayload,
    MCPReadinessReportPayload,
    MCPRootPayload,
    MCPTagCountsPayload,
    session_tree_payload,
)
from polylogue.mcp.query_contracts import MCPSessionQueryRequest
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_resources(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    """Register MCP resources on the given server."""

    @mcp.resource("polylogue://stats")
    async def stats_resource() -> str:
        try:
            from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

            config = hooks.get_config()
            transaction = QueryTransaction(
                mcp_archive_root(config),
                QueryTransactionRequest(operation="resource.stats", arguments={}, page_size=1, projection="stats"),
            )
            archive_stats = await transaction.run(lambda archive: archive.stats())
        except Exception as exc:
            return hooks.error_json(
                f"Failed to retrieve archive stats: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        return hooks.json_payload(
            MCPArchiveStatsPayload.from_archive_stats(
                archive_stats,
                include_embedded=False,
                include_db_size=False,
            ),
            exclude_none=True,
        )

    @mcp.resource("polylogue://sessions")
    async def sessions_resource() -> str:
        try:
            from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

            spec = MCPSessionQueryRequest().build_spec(hooks.clamp_limit)
            config = hooks.get_config()
            transaction = QueryTransaction(
                mcp_archive_root(config),
                QueryTransactionRequest(
                    operation="resource.sessions",
                    arguments={"limit": spec.limit, "offset": spec.offset, "sort": spec.sort},
                    page_size=spec.limit or 10,
                    offset=spec.offset,
                    projection="session-summary",
                    stable_order="date",
                ),
            )
            payload = await transaction.run(lambda archive: archive_session_list_payload(archive, spec))
            return hooks.json_payload(payload)
        except Exception as exc:
            return hooks.error_json(
                f"Failed to list sessions: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )

    @mcp.resource("polylogue://session/{conv_id}")
    async def session_resource(conv_id: str) -> str:
        try:
            from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

            config = hooks.get_config()
            transaction = QueryTransaction(
                mcp_archive_root(config),
                QueryTransactionRequest(
                    operation="resource.session",
                    arguments={"conv_id": conv_id},
                    page_size=1,
                    projection="session-summary",
                ),
            )

            def read(archive: ArchiveStore) -> str:
                try:
                    session_id = archive.resolve_session_id(conv_id)
                    archive_summary = archive.read_summary(session_id)
                except (KeyError, ValueError):
                    return hooks.error_json(f"Session not found: {conv_id}", code="not_found")
                return hooks.json_payload(archive_summary_payload(archive_summary))

            return await transaction.run(read)
        except sqlite3.OperationalError:
            return hooks.error_json(f"Session not found: {conv_id}", code="not_found")
        except Exception as exc:
            return hooks.error_json(
                f"Failed to get session {conv_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )

    @mcp.resource("polylogue://tags")
    async def tags_resource() -> str:
        try:
            tags = await hooks.get_polylogue().list_tags()
        except Exception as exc:
            return hooks.error_json(
                f"Failed to list tags: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        with hooks.response_context("list_tags", {"limit": 3, "offset": 0}):
            return hooks.json_payload(MCPTagCountsPayload(root=tags))

    @mcp.resource("polylogue://capabilities/action-affordances")
    def action_affordances_resource() -> str:
        """Return the shared action catalog once, outside ordinary query responses."""
        from polylogue.operations.action_contracts import action_affordance_list_payload

        payload = action_affordance_list_payload()
        return hooks.json_payload(MCPRootPayload(root={"action_affordances": payload.model_dump(mode="json")}))

    @mcp.resource("polylogue://capabilities/query")
    def query_capabilities_resource() -> str:
        """Expose the executable query vocabulary as bounded model-facing data."""
        from polylogue.archive.query.metadata import query_unit_descriptors, terminal_query_source_list

        units = []
        for descriptor in query_unit_descriptors(terminal_supported=True):
            units.append(
                {
                    "unit": descriptor.unit,
                    "source": descriptor.plural_source,
                    "description": descriptor.description,
                    "example": descriptor.terminal_example or descriptor.example,
                    "fields": [
                        {"name": field.name, "description": field.description, "example": field.example}
                        for field in descriptor.fields
                    ],
                    "aggregate_group_fields": list(descriptor.aggregate_group_fields),
                    "stable_order": "time" if descriptor.time_sort_supported else "canonical",
                    "execution": {
                        "result_semantics": "exhaustive_page",
                        "continuation": "query_units response continuation",
                        "cost": "bounded page; aggregate/group stages may queue as scan work",
                    },
                }
            )
        return hooks.json_payload(
            MCPRootPayload(
                root={
                    "version": 1,
                    "kind": "query-capability-catalog",
                    "terminal_sources": terminal_query_source_list(),
                    "grammar": {
                        "terminal_form": "<sources> where <predicate>",
                        "pipeline_form": "<sources> where <predicate> | group by <field> | count",
                        "required_recovery": "Use the returned continuation to advance; do not replay the same page.",
                    },
                    "units": units,
                }
            )
        )

    @mcp.resource("polylogue://messages/{conv_id}")
    async def messages_resource(conv_id: str) -> str:
        try:
            from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

            config = hooks.get_config()
            transaction = QueryTransaction(
                mcp_archive_root(config),
                QueryTransactionRequest(
                    operation="resource.messages",
                    arguments={"conv_id": conv_id, "limit": 20, "offset": 0},
                    page_size=20,
                    projection="message-page",
                    stable_order="session,message,block",
                ),
            )

            def read(archive: ArchiveStore) -> str:
                try:
                    session_id = archive.resolve_session_id(conv_id)
                    session = archive.read_session(session_id)
                except (KeyError, ValueError):
                    return hooks.error_json(f"Session not found: {conv_id}", code="not_found")
                with hooks.response_context(
                    "get_messages",
                    {
                        "session_id": session_id,
                        "limit": 20,
                        "max_chars_per_message": 4096,
                        "excerpt": True,
                    },
                ):
                    return hooks.json_payload(archive_messages_payload(session, limit=20, offset=0))

            return await transaction.run(read)
        except sqlite3.OperationalError:
            return hooks.error_json(f"Session not found: {conv_id}", code="not_found")
        except Exception as exc:
            return hooks.error_json(
                f"Failed to list messages for {conv_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )

    @mcp.resource("polylogue://session-tree/{conv_id}")
    async def session_tree_resource(conv_id: str) -> str:
        try:
            tree = await hooks.get_polylogue().get_session_tree(conv_id)
        except Exception as exc:
            return hooks.error_json(
                f"Failed to get session tree for {conv_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        return hooks.json_payload(session_tree_payload(tree))

    @mcp.resource("polylogue://origin/{name}/recent")
    async def origin_recent_resource(name: str) -> str:
        try:
            from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest

            spec = MCPSessionQueryRequest(origin=name, sort="date", limit=10).build_spec(hooks.clamp_limit)
            config = hooks.get_config()
            transaction = QueryTransaction(
                mcp_archive_root(config),
                QueryTransactionRequest(
                    operation="resource.origin_recent",
                    arguments={"origin": name, "limit": 10, "offset": 0, "sort": "date"},
                    page_size=10,
                    projection="session-summary",
                    stable_order="date",
                ),
            )
            payload = await transaction.run(lambda archive: archive_session_list_payload(archive, spec))
            return hooks.json_payload(payload)
        except Exception as exc:
            return hooks.error_json(
                f"Failed to list recent sessions for origin {name}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )

    @mcp.resource("polylogue://readiness")
    def readiness_resource() -> str:
        try:
            from polylogue.readiness import get_readiness

            report = get_readiness(hooks.get_config())
            return hooks.json_payload(
                MCPReadinessReportPayload.from_report(
                    report,
                    include_counts=False,
                    include_detail=False,
                    include_cached=False,
                ),
                exclude_none=True,
            )
        except Exception as exc:
            return hooks.json_payload(
                MCPErrorPayload(
                    message="internal MCP resource error",
                    code="internal_error",
                    error="internal_error",
                    detail=type(exc).__name__,
                ),
                exclude_none=True,
            )

    @mcp.resource("polylogue://raw-authority-census/{census_id}/{offset}")
    def raw_authority_census_resource(census_id: str, offset: str) -> str:
        """Resolve one bounded page from a durable raw-authority ledger."""
        try:
            from polylogue.storage.raw_authority import read_raw_authority_census

            root = mcp_archive_root(hooks.get_config())
            handle = f"polylogue://raw-authority-census/{census_id}/{offset}"
            return hooks.json_payload(MCPRootPayload(root=read_raw_authority_census(root, handle)))
        except KeyError:
            return hooks.error_json(f"Raw authority census not found: {census_id}", code="not_found")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return hooks.error_json(
                f"Failed to read raw authority census {census_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        except Exception as exc:
            return hooks.error_json(
                f"Failed to read raw authority census {census_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )

    @mcp.resource("polylogue://raw-authority-detail/{census_id}/{record_id}/{revision}/{offset}")
    def raw_authority_detail_resource(census_id: str, record_id: str, revision: str, offset: str) -> str:
        """Resolve one bounded chunk from a complete census or plan document."""
        try:
            from polylogue.storage.raw_authority import read_raw_authority_detail

            root = mcp_archive_root(hooks.get_config())
            handle = f"polylogue://raw-authority-detail/{census_id}/{record_id}/{revision}/{offset}"
            return hooks.json_payload(MCPRootPayload(root=read_raw_authority_detail(root, handle)))
        except KeyError:
            return hooks.error_json(
                f"Raw authority detail not found: {census_id}/{record_id}",
                code="not_found",
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return hooks.error_json(
                f"Failed to read raw authority detail {census_id}/{record_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )
        except Exception as exc:
            return hooks.error_json(
                f"Failed to read raw authority detail {census_id}/{record_id}: {exc}",
                code="internal_error",
                detail=type(exc).__name__,
            )


__all__ = ["register_resources"]
