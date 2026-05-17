"""TUI adapter that implements the shared read-surface protocols.

The Textual TUI screens (``polylogue/ui/tui/screens/``) historically
called :class:`ArchiveOperations` methods directly and rendered widgets
from raw ``Conversation`` rows and ad-hoc dicts.  That bypassed the
typed payload envelopes the web reader, CLI JSON, MCP, and Python API
adapters already share via :mod:`polylogue.surfaces.payloads`.

This adapter conforms to the same :class:`ConversationListSurface`,
:class:`ConversationSearchSurface`, :class:`ConversationStatsSurface`,
and :class:`ConversationTagsSurface` protocols as the CLI, MCP, and
Python API read surfaces, so TUI widgets consume the canonical
:class:`ConversationListResponse` envelope and the typed
:class:`ConversationListRowPayload` rows.

Background — issues #848 (re-scoped: wire the TUI to the same typed
payload envelopes the web reader consumes) and #859 (shared read
surface protocol family).

The TUI's screens already accept an :class:`ArchiveOperations` instance
(ref #860 wired the Textual app through the shared operations contract);
this adapter mirrors that injection model rather than reaching for
:class:`RuntimeServices` so existing TUI construction and tests do not
need to be rebuilt to satisfy the contract.
"""

from __future__ import annotations

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


class TUIReadSurface:
    """Read-surface adapter for the Textual TUI.

    Wraps :class:`ArchiveOperations` (the same dependency the TUI
    ``PolylogueApp`` already accepts) and projects the read methods into
    the canonical :class:`ConversationListResponse` envelope.  TUI
    screens consume the typed payload rows instead of raw repository
    rows so envelope evolution lands once in the substrate and flows to
    the TUI automatically.
    """

    def __init__(self, operations: ArchiveOperations) -> None:
        self._operations = operations

    @property
    def operations(self) -> ArchiveOperations:
        return self._operations

    async def list_conversations(self, spec: ConversationQuerySpec) -> ConversationListResponse:
        ops = self._operations
        conversations = await ops.query_conversations(spec)
        total = await spec.count(ops.repository)
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
        # Search and list share the same envelope; the spec's
        # ``query_terms`` field drives FTS while scalar filters apply
        # identically.  This matches the CLI/MCP/API adapters.
        return await self.list_conversations(spec)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        return await self._operations.list_tags(provider=provider)

    async def archive_stats(self) -> ArchiveStats:
        return await self._operations.summary_stats()


# Static conformance pins — mypy --strict fails the build if TUIReadSurface drifts.
assert_implements(TUIReadSurface, ConversationListSurface)
assert_implements(TUIReadSurface, ConversationSearchSurface)
assert_implements(TUIReadSurface, ConversationTagsSurface)
assert_implements(TUIReadSurface, ConversationStatsSurface)


__all__ = ["TUIReadSurface"]
