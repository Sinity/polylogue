"""Python API adapter implementing the shared write-surface protocols.

Wraps :class:`polylogue.api.Polylogue` (the async facade) and projects
its write/import/maintenance methods into the canonical
:class:`ImportOperation`, :class:`BackfillOperation`, and
:class:`TagMutationResult` envelopes from
:mod:`polylogue.api.contracts.write_surface`.

The adapter exists so the Python API surface satisfies the same
Protocol contracts as the CLI and MCP adapters.  Callers that want the
underlying ``Polylogue.parse_file`` / ``parse_sources`` semantics
continue to use ``Polylogue`` directly; those are documented as
low-level pipeline helpers (see ``polylogue/operations/import_contracts.py``)
distinct from daemon scheduling.
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


class APIWriteSurface:
    """Write-surface adapter over the async :class:`Polylogue` facade."""

    def __init__(self, polylogue: Polylogue) -> None:
        self._polylogue = polylogue

    async def ingest_path(self, path: Path | str) -> ImportOperation:
        """Run a low-level parse over ``path`` and project the result.

        The Python API does not stage into the daemon inbox or contact
        a running daemon — it parses directly through archive ingest
        (``Polylogue.parse_file`` →
        ``parse_sources_archive`` → ``ArchiveStore``).  The returned envelope mirrors
        what the daemon HTTP and CLI ingest commands emit so callers
        downstream can consume one shape regardless of surface.
        """
        resolved = Path(path).expanduser().resolve()
        operation_id = str(uuid.uuid4())
        if not resolved.exists():
            return ImportOperation(
                operation_id=operation_id,
                status="failed",
                path=str(resolved),
                error=f"Path does not exist: {resolved}",
            )
        try:
            await self._polylogue.parse_file(resolved)
        except Exception as exc:
            return ImportOperation(
                operation_id=operation_id,
                status="failed",
                path=str(resolved),
                error=f"{type(exc).__name__}: {exc}",
            )
        return ImportOperation(
            operation_id=operation_id,
            status="accepted",
            path=str(resolved),
            message="Parsed via Polylogue.parse_file (library-direct, no daemon).",
        )

    async def run_maintenance(self, targets: tuple[str, ...], *, dry_run: bool = False) -> BackfillOperation:
        """Execute (or dry-run) the named maintenance targets.

        Delegates to :func:`execute_backfill`, the same shared planner
        invoked by ``polylogue ops maintenance run`` so adapters do not
        invent their own maintenance semantics.
        """
        config = self._polylogue.config
        return execute_backfill(config, targets=targets, dry_run=dry_run)

    async def rebuild_index(self) -> bool:
        return await self._polylogue.rebuild_index()

    async def update_index(self, session_ids: list[str]) -> bool:
        return await self._polylogue.update_index(session_ids)

    async def add_tag(
        self,
        session_id: str,
        tag: str,
        *,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> TagMutationResult:
        return await self._polylogue.add_tag(
            session_id,
            tag,
            author_ref=author_ref,
            author_kind=author_kind,
        )

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult:
        return await self._polylogue.remove_tag(session_id, tag)

    async def delete_session(self, session_id: str) -> bool:
        return await self._polylogue.delete_session(session_id)


# Static conformance pins — mypy fails the build if APIWriteSurface drifts.
assert_implements(APIWriteSurface, IngestSurface)
assert_implements(APIWriteSurface, MaintenanceSurface)
assert_implements(APIWriteSurface, IndexMaintenanceSurface)
assert_implements(APIWriteSurface, TagMutationSurface)
assert_implements(APIWriteSurface, SessionDeleteSurface)


__all__ = ["APIWriteSurface"]
