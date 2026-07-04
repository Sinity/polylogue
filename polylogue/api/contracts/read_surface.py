"""Typed read-surface Protocol family.

Every read surface (CLI, MCP, daemon HTTP, Python API) is expected to expose
the canonical archive read operations using the *same* request and response
contracts:

* Input: ``SessionQuerySpec`` for query/search, or a documented
  primitive (``origin``, ``session_id``, etc.) for the simpler reads.
* Output: the shared ``polylogue.surfaces.payloads`` envelopes —
  ``SessionListResponse``, ``FacetsResponse``, ``ArchiveStats``,
  ``DaemonStatus``, ``TagMutationResult``.

The Protocols below are intentionally narrow.  They cover the read surface
contract that #859 requires every adapter to share, not the full surface
API (mutations, ingest, insights, etc.).  Surfaces are free to expose
additional methods; conformance only requires the methods declared here.

Protocol composition is preferred over a single fat ``ReadSurface``
protocol because not every surface implements every capability.  A surface
may declare conformance to ``SessionListSurface`` without also
implementing ``SessionStatsSurface`` (for example, an MCP tool
subset).  The composite ``ReadSurface`` is supplied for callers that need
the full union.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.daemon.status import DaemonStatus
from polylogue.operations import ArchiveStats
from polylogue.surfaces.payloads import (
    FacetsResponse,
    SessionListResponse,
    TagMutationResult,
)


@runtime_checkable
class SessionListSurface(Protocol):
    """Canonical session list/query contract.

    Implementations accept a :class:`SessionQuerySpec` (the shared
    input contract validated by ``QuerySpecError``) and return a typed
    :class:`SessionListResponse` envelope (the shared output contract
    with explicit ``items``, ``total``, ``limit``, ``offset`` fields).
    """

    async def list_sessions(self, spec: SessionQuerySpec) -> SessionListResponse: ...


@runtime_checkable
class SessionSearchSurface(Protocol):
    """Canonical session search/hit contract.

    Search routes through the same :class:`SessionQuerySpec` as
    listing — ``spec.contains_terms`` carries the FTS terms.  The
    envelope is the same :class:`SessionListResponse`; surfaces that
    emit ranked search-hit-shaped rows wrap them into the canonical
    envelope via ``SessionSearchHitPayload.from_hit``.
    """

    async def search_sessions(self, spec: SessionQuerySpec) -> SessionListResponse: ...


@runtime_checkable
class SessionTagsSurface(Protocol):
    """Canonical tag listing contract.

    ``list_tags`` returns ``{tag: session_count}`` for the optionally
    origin-scoped archive.  The mapping shape is shared verbatim across
    CLI, MCP, and Python API.
    """

    async def list_tags(self, *, origin: str | None = None) -> dict[str, int]: ...


@runtime_checkable
class SessionStatsSurface(Protocol):
    """Canonical archive-wide stats contract.

    Returns the typed :class:`ArchiveStats` (counts, per-origin counts,
    last-updated timestamps).  Every surface that exposes archive stats
    consumes the same model.
    """

    async def archive_stats(self) -> ArchiveStats: ...


@runtime_checkable
class FacetsSurface(Protocol):
    """Canonical facets contract.

    Returns the typed :class:`FacetsResponse` envelope.  Facets are an
    aggregation projection over the archive — origins, tags, repos,
    cwd prefixes, message_types, action_types, has_flags, time range.
    """

    async def facets(self, spec: SessionQuerySpec | None = None) -> FacetsResponse: ...


@runtime_checkable
class DaemonStatusSurface(Protocol):
    """Canonical daemon status contract.

    Returns the typed :class:`DaemonStatus` consumed by every status
    surface (CLI ``polylogue ops status``, daemon HTTP ``/api/status``, MCP
    ``readiness_check``, web reader header).  Surfaces that cannot
    materialize a live status (e.g. a stateless CLI invocation when the
    daemon is offline) still produce a ``DaemonStatus`` populated with
    ``daemon_liveness=False`` rather than ``None``.
    """

    async def daemon_status(self) -> DaemonStatus: ...


@runtime_checkable
class TagMutationSurface(Protocol):
    """Canonical tag mutation contract.

    Mutation surfaces return the shared :class:`TagMutationResult` so the
    ``added | no_op | removed | not_present`` outcome vocabulary is
    centralized.  Surfaces that do not expose mutations (read-only MCP
    role, daemon read role) simply do not implement this protocol.
    """

    async def add_tag(
        self,
        session_id: str,
        tag: str,
        *,
        author_ref: str | None = None,
        author_kind: str | None = None,
    ) -> TagMutationResult: ...

    async def remove_tag(self, session_id: str, tag: str) -> TagMutationResult: ...


@runtime_checkable
class ReadSurface(
    SessionListSurface,
    SessionSearchSurface,
    SessionTagsSurface,
    SessionStatsSurface,
    Protocol,
):
    """Composite read-surface contract.

    A full read surface implements list, search, tags, and stats.
    Additional capabilities (facets, status, mutations) are declared via
    separate protocols above so partial conformance is expressible.
    """


__all__ = [
    "SessionListSurface",
    "SessionSearchSurface",
    "SessionStatsSurface",
    "SessionTagsSurface",
    "DaemonStatusSurface",
    "FacetsSurface",
    "ReadSurface",
    "TagMutationSurface",
]
