"""Shared read-surface contracts for CLI, MCP, daemon HTTP, and Python API.

This package defines the typed Protocols every read surface implements so
parity between adapters (CLI, MCP, daemon HTTP, Python API) is enforced
statically by mypy rather than only by ad-hoc parity tests.

Background — issue #859: ``SessionQuerySpec`` already unifies the
input contract for query/search, and ``polylogue.surfaces.payloads``
already defines the shared response payloads (``SessionListResponse``,
``FacetsResponse``, ``TagMutationResult``, etc.).  What was missing was a
single ``Protocol`` family declaring *which* read methods every surface is
expected to expose, and a static-assertion mechanism that fails compilation
when a surface drifts out of conformance.

The protocols here describe the canonical async read surface.  Each surface
provides a thin adapter object in this package that implements the
protocol by delegating to the existing surface implementation; the adapter
modules attach a ``assert_implements`` static check so any signature drift
is caught by ``mypy --strict``.

The mirrored write-surface Protocol family (``IngestSurface``,
``MaintenanceSurface``, ``TagMutationSurface``, ``SessionDeleteSurface``,
``WriteSurface``, and their ``{api,cli,mcp}_write_surface.py`` adapters) was
removed (polylogue-a7xr.13): it had zero production consumers outside this
package, and its adapters were hand-written mirrors of the real CLI/MCP/API
handlers that had already diverged from them (``CLIWriteSurface.ingest_path``
unconditionally returned a synthetic failed envelope instead of the real
stage-and-POST flow) — so the "parity guarantee" ``assert_implements``
provided attached to a shadow object free to drift from the surfaces it
claimed to describe, rather than to the real execution paths. If write-surface
parity enforcement is wanted again, re-anchor ``assert_implements`` on the
actual CLI command handlers / MCP tool functions / API facade methods, not a
parallel adapter class.
"""

from __future__ import annotations

from polylogue.api.contracts.assertions import assert_implements
from polylogue.api.contracts.read_surface import (
    DaemonStatusSurface,
    ReadSurface,
    SessionListSurface,
    SessionSearchSurface,
    SessionStatsSurface,
    SessionTagsSurface,
)

__all__ = [
    "SessionListSurface",
    "SessionSearchSurface",
    "SessionStatsSurface",
    "SessionTagsSurface",
    "DaemonStatusSurface",
    "ReadSurface",
    "assert_implements",
]
