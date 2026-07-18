"""Protocol-native MCP read algebra.

The compatibility registrars remain internal implementation substrate.  This
module is the only public read registration surface: six explicit verbs with
one terminal-query transaction, stable refs, and parser-owned explanations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polylogue.mcp.declarations.adapter import register_declared_handler
from polylogue.mcp.payloads import MCPArchiveStatsPayload, MCPRootPayload, session_topology_payload

if TYPE_CHECKING:
    from polylogue.mcp.declarations.adapter import ToolRegistrar
    from polylogue.mcp.server_support import ServerCallbacks


def _object_ref(ref: str) -> str:
    """Lower a stable Polylogue URI to the public ref accepted by the API."""
    if not ref.startswith("polylogue://"):
        return ref
    prefix, separator, object_id = ref.removeprefix("polylogue://").partition("/")
    if not separator or not object_id:
        raise ValueError("stable Polylogue URIs require an object kind and id")
    return f"{prefix}:{object_id}"


def register_cutover_read_tools(mcp: ToolRegistrar, hooks: ServerCallbacks) -> None:
    """Register exactly the six default, role-read transactions."""

    async def query(
        expression: str | None = None,
        limit: int | None = None,
        projection: str = "default",
        continuation: str | None = None,
    ) -> str:
        """Execute a terminal DSL page, or resume it using only its q2 token."""
        del projection  # Terminal projections remain parser-owned for this first cutover slice.

        async def run() -> str:
            from polylogue.archive.query.transaction import (
                QueryArchiveEpochUnreadableError,
                QueryContinuationInvalidError,
                QueryContinuationStaleError,
            )

            try:
                with hooks.response_context("query", {}):
                    payload = await hooks.get_polylogue().query_units(
                        expression,
                        limit=limit,
                        continuation=continuation,
                    )
                    return hooks.json_payload(payload)
            except QueryContinuationInvalidError as exc:
                return hooks.error_json(str(exc), code=exc.code, tool="query")
            except QueryContinuationStaleError as exc:
                return hooks.error_json(str(exc), code=exc.code, tool="query")
            except QueryArchiveEpochUnreadableError as exc:
                return hooks.error_json(str(exc), code=exc.code, tool="query")

        return await hooks.async_safe_call("query", run)

    async def read(
        ref: str,
        view: str | None = None,
        limit: int | None = None,
        continuation: str | None = None,
    ) -> str:
        """Read a stable URI or public ref through an explicitly named view."""
        if continuation is not None:
            return hooks.error_json(
                "read continuations are not implemented for this view; use query for exhaustive rows",
                code="invalid_continuation",
                tool="read",
            )
        del limit

        normalized = _object_ref(ref)
        session_id = normalized.removeprefix("session:") if normalized.startswith("session:") else None

        async def run() -> str:
            if view == "topology":
                topology_session_id = normalized.removeprefix("session:")
                topology = await hooks.get_polylogue().get_session_topology(topology_session_id)
                if topology is None:
                    return hooks.error_json(f"object not found: {ref}", code="not_found", tool="read")
                return hooks.json_payload(session_topology_payload(topology, session_id=str(topology.target_id)))
            return hooks.json_payload(await hooks.get_polylogue().resolve_ref(normalized))

        return await hooks.async_safe_call("read", run, session_id=session_id)

    async def get(ref: str, projection: str | None = None) -> str:
        """Resolve one exact stable object or evidence identity."""
        del projection
        normalized = _object_ref(ref)
        session_id = normalized.removeprefix("session:") if normalized.startswith("session:") else None

        async def run() -> str:
            return hooks.json_payload(await hooks.get_polylogue().resolve_ref(normalized))

        return await hooks.async_safe_call("get", run, session_id=session_id)

    async def explain(
        subject: Literal["query", "capability", "ref", "result", "recovery"],
        expression: str | None = None,
        ref: str | None = None,
    ) -> str:
        """Explain parser grammar, capabilities, refs, result semantics, or recovery."""

        async def run() -> str:
            if subject == "query":
                if expression is None:
                    return hooks.error_json(
                        "query explanation requires expression", code="invalid_argument", tool="explain"
                    )
                explanation = await hooks.get_polylogue().explain_query_expression(expression)
                return hooks.json_payload(MCPRootPayload(root={"subject": subject, "explanation": explanation}))
            if subject == "ref":
                if ref is None:
                    return hooks.error_json("ref explanation requires ref", code="invalid_argument", tool="explain")
                resolution = await hooks.get_polylogue().resolve_ref(_object_ref(ref))
                return hooks.json_payload(
                    MCPRootPayload(root={"subject": subject, "resolution": resolution.model_dump(mode="json")})
                )
            from polylogue.archive.query.discovery import RESULT_SEMANTICS_TEACHING, query_discovery_examples

            read_views: list[object] = []
            if subject == "capability":
                read_views = list(await hooks.get_polylogue().list_read_view_profiles())
                return hooks.json_payload(
                    MCPRootPayload(root={"subject": subject, "read_views": read_views, "total": len(read_views)})
                )
            return hooks.json_payload(
                MCPRootPayload(
                    root={
                        "subject": subject,
                        "result_semantics": RESULT_SEMANTICS_TEACHING,
                        "examples": query_discovery_examples(),
                        "recovery": "Use a q2 continuation alone to resume an exhaustive query page.",
                        "read_views": read_views,
                        "total": len(read_views),
                    }
                )
            )

        return await hooks.async_safe_call("explain", run)

    async def context(
        intent: str,
        query: str | None = None,
        budget_tokens: int | None = None,
        result_ref: str | None = None,
    ) -> str:
        """Compile a policy-gated bounded context image with receipts."""
        del intent, result_ref

        async def run() -> str:
            payload = await hooks.get_polylogue().context_image_payload(
                query=query,
                max_tokens=budget_tokens,
                max_sessions=5,
                include_messages=True,
                include_assertions=True,
                redact_paths=True,
            )
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("context", run)

    async def status(
        scope: Literal["archive", "sources", "embeddings", "coordination", "operation"],
        include: tuple[str, ...] = (),
        ref: str | None = None,
    ) -> str:
        """Report compact archive authority and readiness status."""

        async def run() -> str:
            root: dict[str, object] = {"scope": scope}
            if scope == "operation":
                from dataclasses import asdict

                from polylogue.mcp.call_log import mcp_call_outbox_status
                from polylogue.mcp.payloads import MCPReadinessReportPayload
                from polylogue.readiness import get_readiness

                report = get_readiness(hooks.get_config())
                root["operation"] = MCPReadinessReportPayload.from_report(
                    report,
                    include_counts=True,
                    include_detail=True,
                    include_cached=True,
                    mcp_call_delivery=asdict(mcp_call_outbox_status()),
                ).model_dump(mode="json", exclude_none=True)
                return hooks.json_payload(MCPRootPayload(root=root), exclude_none=True)

            from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest
            from polylogue.mcp.archive_support import mcp_archive_root

            transaction = QueryTransaction(
                mcp_archive_root(hooks.get_config()),
                QueryTransactionRequest(
                    operation="status", arguments={"scope": scope}, page_size=1, projection="status"
                ),
            )
            stats = await transaction.run(lambda archive: archive.stats())
            if "provider_usage" not in include:
                root["archive"] = MCPArchiveStatsPayload.from_archive_stats(
                    stats, include_embedded=False, include_db_size=False
                ).model_dump(mode="json")
            if "provider_usage" in include:
                report_usage = await hooks.get_polylogue().provider_usage_report(
                    origin=ref,
                    limit=10,
                    detail="headline",
                )
                usage = report_usage.to_dict()
                root["provider_usage"] = {
                    "model_rollup_usage": usage["model_rollup_usage"],
                    "pricing_grain": usage["pricing_grain"],
                    "exact_total_tokens_evidence": usage["exact_total_tokens_evidence"],
                    "detail_level": usage["detail_level"],
                    "caveats": usage["caveats"],
                }
            return hooks.json_payload(MCPRootPayload(root=root), exclude_none=True)

        return await hooks.async_safe_call("status", run)

    for handler in (query, read, get, explain, context, status):
        register_declared_handler(mcp, handler, name=handler.__name__)


__all__ = ["register_cutover_read_tools"]
