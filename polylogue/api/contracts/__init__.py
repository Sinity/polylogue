"""Shared read-surface contracts for CLI, MCP, daemon HTTP, and Python API.

This package defines the typed Protocols every read surface implements so
parity between adapters (CLI, MCP, daemon HTTP, Python API) is enforced
statically by mypy rather than only by ad-hoc parity tests.

Background — issue #859: ``ConversationQuerySpec`` already unifies the
input contract for query/search, and ``polylogue.surfaces.payloads``
already defines the shared response payloads (``ConversationListResponse``,
``FacetsResponse``, ``TagMutationResult``, etc.).  What was missing was a
single ``Protocol`` family declaring *which* read methods every surface is
expected to expose, and a static-assertion mechanism that fails compilation
when a surface drifts out of conformance.

The protocols here describe the canonical async read surface.  Each surface
provides a thin adapter object in this package that implements the
protocol by delegating to the existing surface implementation; the adapter
modules attach a ``assert_implements`` static check so any signature drift
is caught by ``mypy --strict``.
"""

from __future__ import annotations

from polylogue.api.contracts.assertions import assert_implements
from polylogue.api.contracts.read_surface import (
    ConversationListSurface,
    ConversationSearchSurface,
    ConversationStatsSurface,
    ConversationTagsSurface,
    DaemonStatusSurface,
    ReadSurface,
)

__all__ = [
    "ConversationListSurface",
    "ConversationSearchSurface",
    "ConversationStatsSurface",
    "ConversationTagsSurface",
    "DaemonStatusSurface",
    "ReadSurface",
    "assert_implements",
]
