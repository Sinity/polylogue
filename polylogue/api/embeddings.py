"""Embedding readiness and preflight methods for the Python API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class PolylogueEmbeddingsMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def backend(self) -> SQLiteBackend: ...

        @property
        def repository(self) -> Any: ...

    def embedding_status(self, *, detail: bool = False) -> dict[str, object]:
        """Return canonical embedding readiness status for API clients.

        This is the same no-spend payload used by ``polylogue ops embed status``
        and the MCP ``embedding_status`` tool.
        """
        from polylogue.storage.embeddings.status_payload import embedding_status_payload

        return dict(
            embedding_status_payload(
                self,
                include_retrieval_bands=detail,
                include_detail=detail,
            )
        )

    async def search_similar_sessions(
        self,
        session_id: str,
        *,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
        voyage_api_key: str | None = None,
    ) -> dict[str, object]:
        """Return vector-ranked session hits for a stored session."""
        return cast(
            dict[str, object],
            await self.repository.search_similar_sessions(
                session_id,
                limit=limit,
                vector_provider=vector_provider,
                provider_db_path=self.config.archive_root / "embeddings.db",
                voyage_api_key=voyage_api_key,
            ),
        )

    def embedding_preflight(
        self,
        *,
        rebuild: bool = False,
        max_sessions: int | None = None,
        max_messages: int | None = None,
        max_cost_usd: float | None = None,
    ) -> dict[str, object]:
        """Return a no-provider-call embedding catch-up cost window."""
        from polylogue.storage.embeddings.preflight import build_preflight_report, preflight_payload

        return preflight_payload(
            build_preflight_report(
                self.backend.db_path,
                rebuild=rebuild,
                max_sessions=max_sessions,
                max_messages=max_messages,
                max_cost_usd=max_cost_usd,
            )
        )
