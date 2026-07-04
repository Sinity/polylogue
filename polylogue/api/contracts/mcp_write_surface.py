"""MCP adapter implementing the shared write-surface protocols.

The MCP server's mutation and maintenance tools
(``server_mutation_tools.py``, ``server_maintenance_tools.py``) already
route through the Polylogue facade.
This module exposes a thin Python-level adapter so cross-surface tests
and tools can invoke the MCP semantics through the shared protocol
contracts without spinning up the FastMCP transport.

The adapter holds a :class:`RuntimeServices` instance plus an optional
:class:`Polylogue` facade (for tag/delete mutations that route through
the facade in the actual MCP tools), and calls the same shared
operations the registered tool handlers use.  Returned envelopes are
the canonical :class:`ImportOperation` / :class:`BackfillOperation` /
:class:`TagMutationResult` types.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.api.contracts.assertions import assert_implements
from polylogue.api.contracts.write_surface import (
    IndexMaintenanceSurface,
    IngestSurface,
    MaintenanceSurface,
    SessionDeleteSurface,
    TagMutationSurface,
)
from polylogue.maintenance.planner import BackfillOperation, execute_backfill
from polylogue.operations.import_contracts import ImportOperation
from polylogue.surfaces.payloads import TagMutationResult

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.services import RuntimeServices


class MCPWriteSurface:
    """Write-surface adapter over the Polylogue facade.

    The MCP server tool handlers call the same operations methods with
    the same shared envelopes; this adapter is the canonical in-process
    projection of those semantics into the shared write contracts.

    MCP does *not* currently expose an explicit ingest scheduling tool.
    ``ingest_path`` therefore returns a ``status="failed"`` envelope
    explaining the missing surface, satisfying the protocol shape
    without claiming work was scheduled.
    """

    def __init__(
        self,
        services: RuntimeServices,
        *,
        polylogue: Polylogue | None = None,
    ) -> None:
        from polylogue.api import Polylogue

        self._services = services
        cfg = services.get_config()
        self._facade = Polylogue(
            archive_root=cfg.archive_root,
            db_path=services.db_path or cfg.db_path,
        )
        self._polylogue = polylogue

    async def ingest_path(self, path: Path | str) -> ImportOperation:
        """Report missing MCP ingest tool through the canonical envelope.

        The MCP read role does not expose ingest scheduling; the write
        role would route through the same daemon HTTP boundary the CLI
        uses.  Until that tool is registered (#861 follow-up), the
        adapter returns ``status="failed"`` so callers cannot
        accidentally treat the missing surface as success.
        """
        resolved = Path(path).expanduser().resolve()
        operation_id = str(uuid.uuid4())
        return ImportOperation(
            operation_id=operation_id,
            status="failed",
            path=str(resolved),
            error=(
                "MCPWriteSurface.ingest_path: MCP currently has no ingest "
                "scheduling tool.  Use the daemon HTTP API or the Python "
                "API adapter directly."
            ),
        )

    async def run_maintenance(self, targets: tuple[str, ...], *, dry_run: bool = False) -> BackfillOperation:
        """Delegate to the shared :func:`execute_backfill` planner.

        MCP's ``rebuild_session_insights`` tool covers one specific
        backfill kind; the generic maintenance contract here covers the
        full vocabulary (``MAINTENANCE_TARGET_NAMES``) so the same
        envelope flows through API/CLI/MCP without per-surface drift.
        """
        config = self._services.get_config()
        return execute_backfill(config, targets=targets, dry_run=dry_run)

    async def rebuild_index(self) -> bool:
        return await self._facade.rebuild_index()

    async def update_index(self, session_ids: list[str]) -> bool:
        return await self._facade.update_index(session_ids)

    def _require_polylogue(self) -> Polylogue:
        if self._polylogue is None:
            raise RuntimeError(
                "MCPWriteSurface tag/delete methods require a Polylogue facade. "
                "Construct MCPWriteSurface(services, polylogue=poly) when the "
                "caller owns a facade instance to delegate mutations to."
            )
        return self._polylogue

    async def add_tag(
        self,
        session_id: str,
        tag: str,
        *,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> TagMutationResult:
        return await self._require_polylogue().add_tag(
            session_id,
            tag,
            author_ref=author_ref,
            author_kind=author_kind,
        )

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult:
        return await self._require_polylogue().remove_tag(session_id, tag)

    async def delete_session(self, session_id: str) -> bool:
        return await self._require_polylogue().delete_session(session_id)


assert_implements(MCPWriteSurface, IngestSurface)
assert_implements(MCPWriteSurface, MaintenanceSurface)
assert_implements(MCPWriteSurface, IndexMaintenanceSurface)
assert_implements(MCPWriteSurface, TagMutationSurface)
assert_implements(MCPWriteSurface, SessionDeleteSurface)


__all__ = ["MCPWriteSurface"]
