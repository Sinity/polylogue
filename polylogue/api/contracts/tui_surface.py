"""TUI adapter that implements the shared read-surface protocols.

The Textual TUI screens (``polylogue/ui/tui/screens/``) historically
rendered widgets from raw ``Session`` rows and ad-hoc dicts.  That
bypassed the typed payload envelopes the web reader, CLI JSON, MCP, and
Python API adapters already share via :mod:`polylogue.surfaces.payloads`.

This adapter conforms to the same :class:`SessionListSurface`,
:class:`SessionSearchSurface`, :class:`SessionStatsSurface`,
and :class:`SessionTagsSurface` protocols as the CLI, MCP, and
Python API read surfaces, so TUI widgets consume the canonical
:class:`SessionListResponse` envelope and the typed
:class:`SessionListRowPayload` rows.

Background — issues #848 (re-scoped: wire the TUI to the same typed
payload envelopes the web reader consumes) and #859 (shared read
surface protocol family).

The TUI's screens accept an archive :class:`Polylogue` facade instance
(#1743: the TUI/dashboard reads now route through the archive
facade); this adapter mirrors that injection model rather than reaching
for :class:`RuntimeServices` so existing TUI construction does not need
to be rebuilt to satisfy the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.api.contracts.assertions import assert_implements
from polylogue.api.contracts.read_surface import (
    SessionListSurface,
    SessionSearchSurface,
    SessionStatsSurface,
    SessionTagsSurface,
)
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.surfaces.payloads import (
    QueryMissDiagnosticsPayload,
    SessionListResponse,
    SessionListRowPayload,
)

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.operations import ArchiveStats


class TUIReadSurface:
    """Read-surface adapter for the Textual TUI.

    Wraps the archive :class:`Polylogue` facade (the same dependency the
    TUI ``PolylogueApp`` accepts) and projects the read methods into the
    canonical :class:`SessionListResponse` envelope.  TUI screens
    consume the typed payload rows instead of raw repository rows so
    envelope evolution lands once in the substrate and flows to the TUI
    automatically.
    """

    def __init__(self, polylogue: Polylogue) -> None:
        self._polylogue = polylogue

    @property
    def polylogue(self) -> Polylogue:
        return self._polylogue

    async def list_sessions(self, spec: SessionQuerySpec) -> SessionListResponse:
        facade = self._polylogue
        sessions = await facade.list_sessions_for_spec(spec)
        total = await spec.count(facade.config)
        diagnostics = None
        if not sessions:
            try:
                miss = await facade.diagnose_query_miss(spec)
            except Exception:
                miss = None
            if miss is not None:
                diagnostics = QueryMissDiagnosticsPayload.from_diagnostics(miss)
        return SessionListResponse(
            items=tuple(SessionListRowPayload.from_session(conv) for conv in sessions),
            total=total,
            limit=spec.limit if spec.limit is not None else len(sessions),
            offset=spec.offset,
            query_description=list(spec.describe()),
            diagnostics=diagnostics,
        )

    async def search_sessions(self, spec: SessionQuerySpec) -> SessionListResponse:
        # Search and list share the same envelope; the spec's
        # ``query_terms`` field drives FTS while scalar filters apply
        # identically.  This matches the CLI/MCP/API adapters.
        return await self.list_sessions(spec)

    async def list_tags(self, *, origin: str | None = None) -> dict[str, int]:
        return await self._polylogue.list_tags(origin=origin)

    async def archive_stats(self) -> ArchiveStats:
        return await self._polylogue.stats()


# Static conformance pins — mypy --strict fails the build if TUIReadSurface drifts.
assert_implements(TUIReadSurface, SessionListSurface)
assert_implements(TUIReadSurface, SessionSearchSurface)
assert_implements(TUIReadSurface, SessionTagsSurface)
assert_implements(TUIReadSurface, SessionStatsSurface)


__all__ = ["TUIReadSurface"]
