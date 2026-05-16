"""Python API adapter that implements the shared read-surface protocols.

Wraps :class:`polylogue.api.Polylogue` (the async facade) and projects its
read methods into the canonical :class:`ConversationListResponse`,
:class:`FacetsResponse`, and :class:`ArchiveStats` envelopes defined in
``polylogue.surfaces.payloads``.

The adapter exists so the Python API surface satisfies the same Protocol
contracts as the CLI and MCP adapters.  Callers that want the raw
``Conversation`` domain objects continue to use ``Polylogue`` directly;
the adapter is only needed for cross-surface parity contexts (tests,
mixed-surface tooling).
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
from polylogue.operations import ArchiveStats
from polylogue.surfaces.payloads import (
    ConversationListResponse,
    ConversationListRowPayload,
    QueryMissDiagnosticsPayload,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue


class APIReadSurface:
    """Read-surface adapter over the async :class:`Polylogue` facade."""

    def __init__(self, polylogue: Polylogue) -> None:
        self._polylogue = polylogue

    async def list_conversations(self, spec: ConversationQuerySpec) -> ConversationListResponse:
        ops = self._polylogue.operations
        conversations = await ops.query_conversations(spec)
        repo = self._polylogue.repository
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
        # Search and list share the same envelope; the spec's contains_terms
        # field drives FTS while the list/query paths still apply scalar
        # filters identically.
        return await self.list_conversations(spec)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        return await self._polylogue.list_tags(provider=provider)

    async def archive_stats(self) -> ArchiveStats:
        return await self._polylogue.stats()


# Static conformance pins — mypy fails the build if APIReadSurface drifts.
assert_implements(APIReadSurface, ConversationListSurface)
assert_implements(APIReadSurface, ConversationSearchSurface)
assert_implements(APIReadSurface, ConversationTagsSurface)
assert_implements(APIReadSurface, ConversationStatsSurface)


__all__ = ["APIReadSurface"]
