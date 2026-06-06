"""CLI adapter implementing the shared write-surface protocols.

The CLI's ``polylogue ingest`` and ``polylogue maintenance run``
commands already route input through :class:`ImportOperation` and
:class:`BackfillOperation` respectively, and tag/delete subcommands
route through the Polylogue facade write methods.  This adapter is the
in-process projection that lets cross-surface tests assert CLI conformance
to the same Protocol contracts as the MCP and Python API adapters without
invoking Click subprocesses.

The adapter delegates to the Polylogue facade and the shared
:func:`execute_backfill` planner — the same layer the CLI command
handlers reach for after parsing flags — so the write semantics are
identical.  CLI-specific concerns (Click flag parsing, daemon HTTP
staging into ``archive_root/inbox``, terminal formatting) live in
``polylogue/cli/`` and are out of scope here.
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


class CLIWriteSurface:
    """Write-surface adapter mirroring the CLI's write semantics.

    Concrete CLI command handlers in ``polylogue/cli/commands/`` build
    the same operation envelopes and call through the Polylogue facade;
    this adapter captures that projection directly so the Protocol
    conformance is verifiable without an in-process Click runner.
    """

    def __init__(self, services: RuntimeServices, *, polylogue: Polylogue | None = None) -> None:
        from polylogue.api import Polylogue

        self._services = services
        cfg = services.get_config()
        self._facade = Polylogue(
            archive_root=cfg.archive_root,
            db_path=services.db_path or cfg.db_path,
        )
        # Tag/delete mutations live on the Polylogue facade; tests construct
        # both adapter and facade against the same db_path so the adapter does
        # not own a parallel runtime.  Adapter callers without a pre-built
        # facade pass ``polylogue=None`` and the adapter does not expose
        # tag/delete (the static protocols still pass because conformance is
        # method presence, not execution).
        self._polylogue = polylogue

    async def ingest_path(self, path: Path | str) -> ImportOperation:
        """Mirror ``polylogue ingest`` semantics without the daemon hop.

        The CLI command stages the path into ``archive_root/inbox`` and
        POSTs to the daemon HTTP API; the daemon returns the actual
        :class:`ImportOperation`.  For in-process tests and adapters
        that cannot contact the daemon, this method returns the
        equivalent ``status="failed"`` envelope explaining the missing
        daemon scheduling boundary.  Adapters that want the inbox
        staging behavior call ``cli/commands/ingest.py:_stage_for_daemon``
        directly.
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
        return ImportOperation(
            operation_id=operation_id,
            status="failed",
            path=str(resolved),
            error=(
                "CLIWriteSurface.ingest_path is the in-process projection; "
                "the live CLI ingest command requires a running daemon at "
                "/api/ingest to schedule work.  Use the Python API adapter "
                "for direct parse, or invoke the daemon HTTP surface."
            ),
        )

    async def run_maintenance(self, targets: tuple[str, ...], *, dry_run: bool = False) -> BackfillOperation:
        """Delegate to the shared :func:`execute_backfill` planner."""
        config = self._services.get_config()
        return execute_backfill(config, targets=targets, dry_run=dry_run)

    async def rebuild_index(self) -> bool:
        return await self._facade.rebuild_index()

    async def update_index(self, session_ids: list[str]) -> bool:
        return await self._facade.update_index(session_ids)

    def _require_polylogue(self) -> Polylogue:
        if self._polylogue is None:
            raise RuntimeError(
                "CLIWriteSurface tag/delete methods require a Polylogue facade. "
                "Construct CLIWriteSurface(services, polylogue=poly) when the "
                "caller owns a facade instance to delegate mutations to."
            )
        return self._polylogue

    async def add_tag(self, session_id: str, tag: str) -> TagMutationResult:
        return await self._require_polylogue().add_tag(session_id, tag)

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult:
        return await self._require_polylogue().remove_tag(session_id, tag)

    async def delete_session(self, session_id: str) -> bool:
        return await self._require_polylogue().delete_session(session_id)


assert_implements(CLIWriteSurface, IngestSurface)
assert_implements(CLIWriteSurface, MaintenanceSurface)
assert_implements(CLIWriteSurface, IndexMaintenanceSurface)
assert_implements(CLIWriteSurface, TagMutationSurface)
assert_implements(CLIWriteSurface, SessionDeleteSurface)


__all__ = ["CLIWriteSurface"]
