"""Session and timeline MCP archive-product tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.mcp.payloads import MCPRootPayload

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks


def register_session_product_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def session_profile(conversation_id: str, tier: str = "merged") -> str:
        async def run() -> str:
            product = await hooks.get_archive_ops().get_session_profile_product(
                conversation_id,
                tier=tier,
            )
            if product is None:
                return hooks.error_json("Conversation not found", conversation_id=conversation_id)
            return hooks.json_payload(product, exclude_none=True)

        return await hooks.async_safe_call("session_profile", run)

    @mcp.tool()
    async def session_profiles(
        since: str | None = None,
        until: str | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        tier: str = "merged",
        provider: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import SessionProfileProductQuery

            products = await hooks.get_archive_ops().list_session_profile_products(
                SessionProfileProductQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    first_message_since=first_message_since,
                    first_message_until=first_message_until,
                    session_date_since=session_date_since,
                    session_date_until=session_date_until,
                    tier=tier,
                    query=query,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("session_profiles", run)

    @mcp.tool()
    async def session_enrichments(
        since: str | None = None,
        until: str | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        refined_work_kind: str | None = None,
        provider: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import SessionEnrichmentProductQuery

            products = await hooks.get_archive_ops().list_session_enrichment_products(
                SessionEnrichmentProductQuery(
                    provider=provider,
                    since=since,
                    until=until,
                    first_message_since=first_message_since,
                    first_message_until=first_message_until,
                    session_date_since=session_date_since,
                    session_date_until=session_date_until,
                    refined_work_kind=refined_work_kind,
                    query=query,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("session_enrichments", run)

    @mcp.tool()
    async def session_work_events(
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import SessionWorkEventProductQuery

            ops = hooks.get_archive_ops()
            if conversation_id:
                products = await ops.get_session_work_event_products(conversation_id)
            else:
                products = await ops.list_session_work_event_products(
                    SessionWorkEventProductQuery(
                        provider=provider,
                        since=since,
                        until=until,
                        kind=kind,
                        query=query,
                        limit=hooks.clamp_limit(limit),
                        offset=max(0, int(offset)),
                    )
                )
            return hooks.json_payload(
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("session_work_events", run)

    @mcp.tool()
    async def session_phases(
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import SessionPhaseProductQuery

            ops = hooks.get_archive_ops()
            if conversation_id:
                products = await ops.get_session_phase_products(conversation_id)
            else:
                products = await ops.list_session_phase_products(
                    SessionPhaseProductQuery(
                        provider=provider,
                        since=since,
                        until=until,
                        kind=kind,
                        limit=hooks.clamp_limit(limit),
                        offset=max(0, int(offset)),
                    )
                )
            return hooks.json_payload(
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("session_phases", run)

    @mcp.tool()
    async def work_threads(
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        async def run() -> str:
            from polylogue.archive_products import WorkThreadProductQuery

            products = await hooks.get_archive_ops().list_work_thread_products(
                WorkThreadProductQuery(
                    since=since,
                    until=until,
                    query=query,
                    limit=hooks.clamp_limit(limit),
                    offset=max(0, int(offset)),
                )
            )
            return hooks.json_payload(
                MCPRootPayload(root={"count": len(products), "items": [product.model_dump(mode="json") for product in products]})
            )

        return await hooks.async_safe_call("work_threads", run)


__all__ = ["register_session_product_tools"]
