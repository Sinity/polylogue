"""Typed write-surface Protocol family.

Every write/import/maintenance surface (CLI ``ingest`` / ``maintenance``,
MCP mutation/maintenance tools, daemon convergence ingest, Python API
write methods) is expected to expose canonical operations using the
*same* request and result contracts:

* Input: documented primitives (``path``, ``session_id``, ``tag``)
  or typed request models (target tuples, projection specs).
* Output: shared envelopes — ``ImportOperation`` for ingest/scheduling,
  ``BackfillOperation`` for maintenance backfills, ``TagMutationResult``
  for tag mutations.

Background — issue #861 (mirror of #859 for the write side): the read
surface family in :mod:`polylogue.api.contracts.read_surface` defines
which read methods every adapter must implement, with an
``assert_implements`` static check that fails mypy on drift. The same
pattern is needed for the write/import/maintenance side, where adapters
have historically diverged (CLI staging into ``archive_root/inbox``,
daemon HTTP returning ad-hoc dicts, MCP instantiating ``IndexService``
directly while CLI maintenance went through ``execute_backfill``,
Python API ``parse_file`` / ``parse_sources`` bypassing the operation
boundary entirely).

The Protocols here describe the canonical async write surface.  Each
surface provides a thin adapter object in this package that implements
the protocol by delegating to the existing surface implementation; the
adapter modules attach an ``assert_implements`` static check so any
signature drift is caught by ``mypy --strict``.

Protocol composition is preferred over a single fat ``WriteSurface``
protocol because not every adapter implements every capability.  A
read-only MCP role does not implement ``TagMutationSurface``; a CLI
``ingest`` adapter does not implement maintenance/index methods.  The
composite ``WriteSurface`` is supplied for callers that need the full
union.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from polylogue.maintenance.planner import BackfillOperation
from polylogue.operations.import_contracts import ImportOperation
from polylogue.surfaces.payloads import TagMutationResult


@runtime_checkable
class IngestSurface(Protocol):
    """Canonical import/ingest scheduling contract.

    ``ingest_path`` accepts a local path (file or directory) and returns
    a typed :class:`ImportOperation` describing the scheduled work —
    ``operation_id``, ``status`` (``accepted`` / ``pending`` / ``failed``
    / ``error``), and any bounded ``raw_failure_samples``.

    Surfaces are expected to route through the same daemon/operation
    boundary rather than inventing parallel scheduling semantics.  An
    adapter that cannot reach the daemon must surface that as
    ``status="failed"`` with a populated ``error`` field; it must not
    silently downgrade to "scheduled" when nothing was actually queued.
    """

    async def ingest_path(self, path: Path | str) -> ImportOperation: ...


@runtime_checkable
class MaintenanceSurface(Protocol):
    """Canonical maintenance backfill contract.

    ``run_maintenance`` executes (or dry-runs) one or more named
    maintenance targets and returns a typed :class:`BackfillOperation`.
    The target names are the same vocabulary CLI ``maintenance run`` and
    daemon convergence accept (``MAINTENANCE_TARGET_NAMES``).

    Surfaces that only support preview/dry-run set ``dry_run=True``;
    surfaces that cannot execute maintenance (read-only MCP role) simply
    do not implement this protocol.
    """

    async def run_maintenance(self, targets: tuple[str, ...], *, dry_run: bool = False) -> BackfillOperation: ...


@runtime_checkable
class IndexMaintenanceSurface(Protocol):
    """Canonical FTS index maintenance contract.

    ``rebuild_index`` rebuilds the full-text search index from persisted
    message rows.  ``update_index`` repairs FTS rows for a specific set
    of session ids.  Both return ``True`` on success and ``False``
    on a recoverable SQLite error (the caller decides whether to retry
    or surface).

    Every surface routes through the shared indexing-service free
    functions (``polylogue.pipeline.services.indexing``) rather than
    instantiating ``IndexService`` directly.
    """

    async def rebuild_index(self) -> bool: ...

    async def update_index(self, session_ids: list[str]) -> bool: ...


@runtime_checkable
class TagMutationSurface(Protocol):
    """Canonical tag mutation contract.

    Mutation surfaces return the shared :class:`TagMutationResult` so
    the ``added | no_op | removed | not_present`` outcome vocabulary is
    centralized.  Surfaces that do not expose mutations (read-only MCP
    role, daemon read-only role) simply do not implement this protocol.

    This protocol is the write-side counterpart of the read-side
    ``SessionTagsSurface``; it intentionally lives next to the
    other write-surface protocols so the static-conformance machinery
    sees it as part of the write contract family.
    """

    async def add_tag(self, session_id: str, tag: str) -> TagMutationResult: ...

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult: ...


@runtime_checkable
class SessionDeleteSurface(Protocol):
    """Canonical session-delete contract.

    Surfaces returning ``True`` on a successful delete and ``False``
    when the session was not found.  Safety guards (e.g. MCP's
    ``confirm=True`` requirement) live in the adapter layer; the
    protocol expresses only the storage-level outcome.
    """

    async def delete_session(self, session_id: str) -> bool: ...


@runtime_checkable
class WriteSurface(
    IngestSurface,
    MaintenanceSurface,
    IndexMaintenanceSurface,
    TagMutationSurface,
    SessionDeleteSurface,
    Protocol,
):
    """Composite write-surface contract.

    A full write surface implements ingest scheduling, maintenance
    backfills, index maintenance, tag mutations, and session
    delete.  Adapters that only expose a subset (e.g. read-only MCP
    role, CLI ``ingest`` command on its own) declare conformance to the
    narrower protocols above.
    """


__all__ = [
    "SessionDeleteSurface",
    "IndexMaintenanceSurface",
    "IngestSurface",
    "MaintenanceSurface",
    "TagMutationSurface",
    "WriteSurface",
]
