"""MCP adapter that implements the shared read-surface protocols.

The MCP server already exposes ``list_conversations``, ``search``,
``stats``, and ``list_tags`` tools that route through
``ArchiveOperations`` with a :class:`ConversationQuerySpec`.  This module
exposes a thin Python-level adapter so cross-surface tests and tools can
invoke the MCP semantics through the shared protocol contracts without
spinning up the FastMCP transport.

The adapter holds a :class:`RuntimeServices` instance and calls the same
``ArchiveOperations`` methods the registered tool handlers use, then
projects the results into the shared
:class:`ConversationListResponse` / :class:`ArchiveStats` envelopes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.api.contracts.assertions import assert_implements
from polylogue.api.contracts.read_surface import (
    ConversationListSurface,
    ConversationSearchSurface,
    ConversationStatsSurface,
    ConversationTagsSurface,
)
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.operations import ArchiveOperations, ArchiveStats
from polylogue.surfaces.payloads import (
    ConversationListResponse,
    ConversationListRowPayload,
    QueryMissDiagnosticsPayload,
)

if TYPE_CHECKING:
    from polylogue.services import RuntimeServices


class MCPReadSurface:
    """Read-surface adapter over :class:`ArchiveOperations`.

    The MCP server tool handlers (``polylogue/mcp/server_tools.py``) call
    the same operations methods with the same :class:`ConversationQuerySpec`
    contract; this adapter is the canonical in-process projection of those
    semantics into the shared envelope.
    """

    def __init__(self, services: RuntimeServices) -> None:
        self._services = services
        self._operations = ArchiveOperations.from_services(services)

    @property
    def operations(self) -> ArchiveOperations:
        return self._operations

    async def list_conversations(self, spec: ConversationQuerySpec) -> ConversationListResponse:
        ops = self._operations
        repo = self._services.get_repository()
        conversations = await ops.query_conversations(spec)
        total = await spec.count(repo)
        diagnostics = None
        if not conversations:
            try:
                miss = await ops.diagnose_query_miss(spec)
            except Exception:
                miss = None
            if miss is not None:
                diagnostics = QueryMissDiagnosticsPayload.from_diagnostics(miss)
        return ConversationListResponse(
            items=tuple(ConversationListRowPayload.from_conversation(conv) for conv in conversations),
            total=total,
            limit=spec.limit if spec.limit is not None else len(conversations),
            offset=spec.offset,
            query_description=list(spec.describe()),
            diagnostics=diagnostics,
        )

    async def search_conversations(self, spec: ConversationQuerySpec) -> ConversationListResponse:
        # MCP routes search through the same spec/operations path; the
        # canonical envelope is identical, so we reuse list_conversations.
        return await self.list_conversations(spec)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        return await self._operations.list_tags(provider=provider)

    async def archive_stats(self) -> ArchiveStats:
        return await self._operations.summary_stats()


assert_implements(MCPReadSurface, ConversationListSurface)
assert_implements(MCPReadSurface, ConversationSearchSurface)
assert_implements(MCPReadSurface, ConversationTagsSurface)
assert_implements(MCPReadSurface, ConversationStatsSurface)


__all__ = ["MCPReadSurface"]
