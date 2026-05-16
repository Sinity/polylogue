"""CLI adapter that implements the shared read-surface protocols.

The CLI's ``polylogue list``/``polylogue search`` commands already route
input through :class:`ConversationQuerySpec` and emit envelopes
compatible with :class:`ConversationListResponse`.  This adapter is the
in-process projection that lets cross-surface tests assert CLI
conformance to the same Protocol contracts as the MCP and Python API
adapters without invoking Click subprocesses.

The adapter delegates to :class:`ArchiveOperations` — the same layer the
CLI command handlers reach for after parsing flags — so the read
semantics are identical.  CLI-specific concerns (Click flag parsing,
delivery target resolution, terminal formatting) live in
``polylogue/cli/`` and are out of scope here.
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


class CLIReadSurface:
    """Read-surface adapter mirroring the CLI's read semantics.

    Concrete CLI command handlers in ``polylogue/cli/`` construct the
    same :class:`ConversationQuerySpec`, call the same operations
    methods, and project results into the same envelope; this adapter
    captures that projection directly so the Protocol conformance is
    verifiable without an in-process Click runner.
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
        return await self.list_conversations(spec)

    async def list_tags(self, *, provider: str | None = None) -> dict[str, int]:
        return await self._operations.list_tags(provider=provider)

    async def archive_stats(self) -> ArchiveStats:
        return await self._operations.summary_stats()


assert_implements(CLIReadSurface, ConversationListSurface)
assert_implements(CLIReadSurface, ConversationSearchSurface)
assert_implements(CLIReadSurface, ConversationTagsSurface)
assert_implements(CLIReadSurface, ConversationStatsSurface)


__all__ = ["CLIReadSurface"]
