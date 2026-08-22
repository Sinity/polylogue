"""Typed archive-read capability exposed by the async API facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec
    from polylogue.archive.session.domain_models import Session, SessionSummary


@runtime_checkable
class ArchiveReadCapability(Protocol):
    """Read-only session capability implemented by the async facade."""

    async def list_sessions(
        self,
        origin: str | None = None,
        limit: int | None = None,
        content_projection: ContentProjectionSpec | None = None,
    ) -> list[Session]: ...
    async def list_summaries(
        self, *, limit: int | None = 50, offset: int = 0, origin: str | None = None
    ) -> list[SessionSummary]: ...
    async def list_sessions_for_spec(
        self, spec: SessionQuerySpec, *, content_projection: ContentProjectionSpec | None = None
    ) -> list[Session]: ...
    async def search_session_hits(self, spec: SessionQuerySpec) -> list[SessionSearchHit]: ...


__all__ = ["ArchiveReadCapability"]
