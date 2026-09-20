"""Resource registration for the MCP server."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from polylogue.mcp.archive_support import (
    archive_session_list_payload,
    archive_summary_payload,
    mcp_archive_root,
)
from polylogue.mcp.payloads import (
    MCPArchiveStatsPayload,
    MCPReadinessReportPayload,
    MCPRootPayload,
    MCPTagCountsPayload,
    session_tree_payload,
)
from polylogue.mcp.query_contracts import MCPSessionQueryRequest
from polylogue.mcp.server_support import _exception_to_error_json
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

    from polylogue.mcp.server_support import ServerCallbacks

_ResourceHandler: TypeAlias = Callable[..., Any]


def register_resources(mcp: MCPServer, hooks: ServerCallbacks) -> None:
    """Register MCP resources on the given server."""

    @mcp.resource("polylogue://agent/manual")
    def agent_manual_resource() -> str:
        """Return the declaration-generated standing manual."""
        from polylogue.agent_integration.assets import read_agent_asset

        return read_agent_asset("standing-manual.md")

    @mcp.resource("polylogue://agent/reference")
    def agent_reference_resource() -> str:
        """Return the declaration-generated deep integration reference."""
        from polylogue.agent_integration.assets import read_agent_asset

        return read_agent_asset("deep-reference.md")

    @mcp.resource("polylogue://agent/manifest")
    def agent_manifest_resource() -> str:
        """Return target/runtime reconciliation for this server's configured capabilities."""
        from polylogue.agent_integration.manifest import build_live_manifest

        return json.dumps(build_live_manifest(hooks.capabilities), ensure_ascii=False, sort_keys=True)

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
            return _exception_to_error_json("resource.stats", exc)
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
            return _exception_to_error_json("resource.sessions", exc)

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
            return _exception_to_error_json("resource.session", exc)

    @mcp.resource("polylogue://tags")
    async def tags_resource() -> str:
        try:
            tags = await hooks.get_polylogue().list_tags()
        except Exception as exc:
            return _exception_to_error_json("resource.tags", exc)
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
        from polylogue.archive.query.discovery import (
            QUERY_DISCOVERY_EXAMPLES,
            QUERY_DISCOVERY_GRAMMAR,
            QUERY_DISCOVERY_NEGATIVE_EXAMPLES,
            RESULT_SEMANTICS_TEACHING,
            query_discovery_examples,
        )
        from polylogue.archive.query.metadata import query_unit_descriptors, terminal_query_source_list
        from polylogue.mcp.declarations import (
            TARGET_DEFAULT_READ_ALGEBRA,
            TARGET_PROMPTS,
            TARGET_RESOURCES,
            MCPResultSemantics,
        )
        from polylogue.mcp.server_support import MCP_RESPONSE_BUDGET_BYTES
        from polylogue.surfaces.payloads import serialize_surface_payload

        mcp_result_semantics = {
            "exhaustive": MCPResultSemantics.EXHAUSTIVE_PAGE,
            "top-k": MCPResultSemantics.TOP_K,
            "sample": MCPResultSemantics.SAMPLE,
            "aggregate": MCPResultSemantics.AGGREGATE,
            "bounded-context": MCPResultSemantics.BOUNDED_CONTEXT,
            "recursive-page": MCPResultSemantics.RECURSIVE_GRAPH,
        }
        descriptors = list(query_unit_descriptors(terminal_supported=True))

        def _units(*, include_field_names: bool) -> list[dict[str, object]]:
            built: list[dict[str, object]] = []
            for descriptor in descriptors:
                declared_examples = query_discovery_examples(unit_source=descriptor.plural_source)
                unit: dict[str, object] = {
                    "unit": descriptor.unit,
                    "source": descriptor.plural_source,
                    "description": descriptor.description,
                    "example": (
                        declared_examples[0].expression
                        if declared_examples
                        else descriptor.terminal_example or descriptor.example
                    ),
                    "aggregate_group_fields": list(descriptor.aggregate_group_fields),
                    "exists_supported": descriptor.exists_supported,
                    "lowerer": descriptor.lowerer_kind,
                    "example_key": declared_examples[0].key if declared_examples else None,
                    "stable_order": "time" if descriptor.time_sort_supported else "canonical",
                    "result_semantics": "exhaustive",
                }
                # Full field descriptions always live behind the declared
                # ``detail`` route; discovery carries the bounded vocabulary,
                # examples, and recovery contract in its first response. The
                # per-unit field NAMES are the largest and fastest-growing part
                # of that catalog, so they are what gives way when the archive's
                # unit set outgrows the response budget -- see below.
                if include_field_names:
                    unit["fields"] = [field.name for field in descriptor.fields]
                else:
                    unit["field_count"] = len(descriptor.fields)
                    unit["fields_via"] = {
                        "tool": "explain",
                        "arguments": {"subject": "capability", "unit": descriptor.unit},
                    }
                built.append(unit)
            return built

        def _catalog(units: list[dict[str, object]]) -> MCPRootPayload[dict[str, object]]:
            return MCPRootPayload(
                root={
                    "version": 2,
                    "kind": "query-capability-catalog",
                    "terminal_sources": terminal_query_source_list(),
                    "grammar": {
                        **QUERY_DISCOVERY_GRAMMAR,
                        "required_recovery": "Use the returned continuation to advance; do not replay the same page.",
                    },
                    "corpus": {
                        "positive_count": len(QUERY_DISCOVERY_EXAMPLES),
                        "negative_count": len(QUERY_DISCOVERY_NEGATIVE_EXAMPLES),
                        "examples_via": {"tool": "query_completions", "arguments": {"kind": "example"}},
                        "errors_via": {"tool": "query_completions", "arguments": {"kind": "error"}},
                        # Completion candidates expose one of these declaration
                        # routes verbatim in their ``route`` field; no
                        # adapter-only aliases obscure route-dependent semantics.
                        "routes": sorted({example.route for example in QUERY_DISCOVERY_EXAMPLES}),
                    },
                    "result_semantics": {
                        contract.coverage: {
                            "total": contract.total,
                            "continuation": contract.continuation,
                            "mcp_declaration": mcp_result_semantics[contract.coverage].value,
                            "teaching": contract.phrase,
                        }
                        for contract in RESULT_SEMANTICS_TEACHING
                    },
                    "row_contract": {
                        "authority": "normalized evidence; derived rows retain source refs",
                        "coverage": "current index generation; check readiness for freshness",
                    },
                    "mcp_algebra": {
                        "read_transactions": [asdict(entry) for entry in TARGET_DEFAULT_READ_ALGEBRA],
                        "resources": [asdict(entry) for entry in TARGET_RESOURCES],
                        "prompts": [asdict(entry) for entry in TARGET_PROMPTS],
                    },
                    "units": units,
                    "detail": {
                        "tool": "explain",
                        "arguments": {"subject": "capability", "limit": 25},
                        "searchable": True,
                        "paged_by": "offset",
                    },
                }
            )

        # This catalog is discovery itself: handing it to the generic
        # over-budget envelope replaces the whole query vocabulary with a
        # retry stub, so a client learns nothing about how to query at all.
        # Every new query unit grows it, so the fit is measured here and the
        # one declared reduction is applied, rather than assumed.
        catalog = _catalog(_units(include_field_names=True))
        if len(serialize_surface_payload(catalog, exclude_none=False).encode("utf-8")) > MCP_RESPONSE_BUDGET_BYTES:
            catalog = _catalog(_units(include_field_names=False))
        return hooks.json_payload(catalog)

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
                    from polylogue.archive.query.spec import DEFAULT_MESSAGE_PAGE_LIMIT
                    from polylogue.mcp.archive_support import archive_message_page_payload

                    return hooks.json_payload(
                        archive_message_page_payload(archive, session_id, limit=DEFAULT_MESSAGE_PAGE_LIMIT, offset=0)
                    )

            return await transaction.run(read)
        except sqlite3.OperationalError:
            return hooks.error_json(f"Session not found: {conv_id}", code="not_found")
        except Exception as exc:
            return _exception_to_error_json("resource.messages", exc)

    @mcp.resource("polylogue://session-tree/{conv_id}")
    async def session_tree_resource(conv_id: str) -> str:
        try:
            tree = await hooks.get_polylogue().get_session_tree(conv_id)
        except Exception as exc:
            return _exception_to_error_json("resource.session-tree", exc)
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
            return _exception_to_error_json("resource.origin-recent", exc)

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
            return _exception_to_error_json("resource.readiness", exc)


class _ResourceRecorder:
    """Stands in for ``MCPServer`` to capture what registration declares.

    ``register_resources`` only *defines and decorates* handlers, so a recorder
    observes the real registration pass without a server, a config or an
    archive.
    """

    def __init__(self) -> None:
        self.uri_templates: list[str] = []

    def resource(self, uri_template: str) -> Callable[[_ResourceHandler], _ResourceHandler]:
        self.uri_templates.append(uri_template)

        def identity(handler: _ResourceHandler) -> _ResourceHandler:
            return handler

        return identity


def registered_resource_uri_templates() -> tuple[str, ...]:
    """Return every resource URI template this module registers, in order.

    This is the live half of the resource reconciliation in
    ``agent_integration/manifest.py`` (polylogue-w17k1). It reads the
    registration pass rather than restating it, so a resource added below with
    no matching ``TARGET_RESOURCES`` entry shows up as an undeclared live
    resource instead of drifting silently.
    """
    recorder = _ResourceRecorder()
    register_resources(cast("MCPServer", recorder), cast("ServerCallbacks", None))
    return tuple(recorder.uri_templates)


__all__ = ["register_resources", "registered_resource_uri_templates"]
