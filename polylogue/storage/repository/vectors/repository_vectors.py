"""Vector/search/stats method mixin for the session repository."""

from __future__ import annotations

import builtins
import sqlite3
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive.hydration import archive_envelope_to_session
from polylogue.core.compute import compute_adapter
from polylogue.core.enums import Origin
from polylogue.core.errors import VectorRuntimeUnavailableError
from polylogue.core.protocols import VectorProvider
from polylogue.core.sources import source_name_to_origin
from polylogue.logging import get_logger
from polylogue.storage.embeddings.embedding_stats import read_embedding_stats_async
from polylogue.storage.hydrators import session_event_from_record
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.search_providers.sqlite_vec_runtime import require_vector_seed_session
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
from polylogue.storage.sqlite.queries.session_events import read_session_events

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.archive.session.domain_models import Session
    from polylogue.archive.stats import ArchiveStats
    from polylogue.config import Config
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


def resolve_optional_vector_provider(
    vector_provider: VectorProvider | None,
    *,
    db_path: Path | None = None,
    voyage_api_key: str | None = None,
) -> VectorProvider | None:
    """Resolve the explicitly supplied provider or create the default one."""
    if vector_provider is not None:
        return vector_provider

    from polylogue.storage.search_providers import create_vector_provider

    return create_vector_provider(voyage_api_key=voyage_api_key, db_path=db_path)


logger = get_logger(__name__)


class RepositoryVectorMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol
        queries: SQLiteQueryStore
        _read_blob_store: BlobStore

        async def get_many(self, session_ids: builtins.list[str]) -> builtins.list[Session]: ...

    async def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[Session]:
        if not vector_provider:
            raise ValueError("Semantic search requires a vector provider.")

        def project(connection: sqlite3.Connection, count: int, hits: list[tuple[str, float]]) -> list[Session]:
            del count
            with closing(connection.cursor()) as cursor:
                sessions: list[Session] = []
                for message_id, _distance in hits:
                    row = cursor.execute(
                        "SELECT session_id FROM archive_index.messages WHERE message_id = ?", (message_id,)
                    ).fetchone()
                    if row is None:
                        raise VectorRuntimeUnavailableError("semantic witness has no session in the selected index")
                    session_id = str(row[0])
                    session = archive_envelope_to_session(
                        read_archive_session_envelope(connection, session_id, blob_store=self._read_blob_store)
                    )
                    session.session_events = tuple(
                        session_event_from_record(event) for event in read_session_events(connection, session_id)
                    )
                    sessions.append(session)
                return sessions

        return await vector_provider.read_similarity(
            index_path=self._backend.db_path,
            text=text,
            limit=limit,
            project=project,
        )

    async def search_similar_sessions(
        self,
        session_id: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
        provider_config: Config | None = None,
    ) -> dict[str, object]:
        """Rank sessions by minimum L2 over every distinct retained seed output.

        One best actual message witness represents each returned session;
        ``matched_message_count`` counts that selected witness, not every
        potentially relevant message in the session.
        """
        if vector_provider is None:
            from polylogue.storage.search_providers import create_vector_provider

            vector_provider = create_vector_provider(
                provider_config,
                require_credentials=False,
            )
        if vector_provider is None:
            raise VectorRuntimeUnavailableError("No local vector runtime is available")

        return await vector_provider.read_similarity(
            seed_session_id=session_id,
            index_path=self._backend.db_path,
            limit=limit,
            project=lambda connection, count, hits: self._project_session_similarity(
                connection, session_id, count, hits, limit=limit
            ),
        )

    @staticmethod
    def _project_session_similarity(
        connection: sqlite3.Connection,
        session_id: str,
        source_embedded_messages: int,
        results: list[tuple[str, float]],
        *,
        limit: int,
    ) -> dict[str, object]:
        """Resolve hits and metadata on the same pinned index as vector ranking."""
        with closing(connection.cursor()) as cursor:
            cursor.row_factory = sqlite3.Row
            require_vector_seed_session(connection, session_id)
            if not results:
                return {
                    "source_embedded_messages": source_embedded_messages,
                    "results": [],
                    "unresolved_message_hits": 0,
                }

            message_ids = [message_id for message_id, _ in results]
            placeholders = ",".join("?" * len(message_ids))
            rows = cursor.execute(
                f"SELECT message_id, session_id FROM archive_index.messages WHERE message_id IN ({placeholders})",
                message_ids,
            ).fetchall()
            message_to_session = {str(row["message_id"]): str(row["session_id"]) for row in rows}
            # A hit whose message id resolves to no row is not "no match" -- it means the
            # embeddings tier references messages the index does not have, so the ranking
            # is being computed over a broken join. Count it and report it, rather than
            # letting it disappear into an ordinary empty result set.
            unresolved_hits = sum(1 for message_id, _ in results if message_id not in message_to_session)
            aggregates: dict[str, tuple[float, set[str]]] = {}
            for message_id, distance in results:
                candidate_id = message_to_session.get(message_id)
                if candidate_id is None or candidate_id == session_id:
                    continue
                best_distance, matched_messages = aggregates.setdefault(candidate_id, (float("inf"), set()))
                matched_messages.add(message_id)
                aggregates[candidate_id] = (min(best_distance, distance), matched_messages)

            ranked = sorted(aggregates.items(), key=lambda item: (item[1][0], item[0]))[:limit]
            ranked_ids = [candidate_id for candidate_id, _ in ranked]
            sessions_by_id: dict[str, sqlite3.Row] = {}
            if ranked_ids:
                placeholders = ",".join("?" * len(ranked_ids))
                rows = cursor.execute(
                    f"SELECT session_id, title, origin FROM archive_index.sessions WHERE session_id IN ({placeholders})",
                    ranked_ids,
                ).fetchall()
                sessions_by_id = {str(row["session_id"]): row for row in rows}

            hits: list[dict[str, object]] = []
            for candidate_id, (distance, matched_messages) in ranked:
                candidate = sessions_by_id.get(candidate_id)
                if candidate is None:
                    continue
                score = max(0.0, min(1.0, 1.0 - (distance * distance) / 2.0))
                hits.append(
                    {
                        "session_id": candidate_id,
                        "score": score,
                        "distance": distance,
                        "matched_message_count": len(matched_messages),
                        "title": candidate["title"],
                        "origin": str(Origin.from_string(source_name_to_origin(candidate["origin"]))),
                    }
                )
            return {
                "source_embedded_messages": source_embedded_messages,
                "results": hits,
                "unresolved_message_hits": unresolved_hits,
            }

    async def _get_message_session_mapping(self, message_ids: builtins.list[str]) -> dict[str, str]:
        if not message_ids:
            return {}

        placeholders = ",".join("?" * len(message_ids))
        query = f"SELECT message_id, session_id FROM messages WHERE message_id IN ({placeholders})"

        async with self._backend.read_connection() as conn:
            cursor = await conn.execute(query, message_ids)
            rows = await cursor.fetchall()

        return {row["message_id"]: row["session_id"] for row in rows}

    async def similarity_search(
        self,
        query: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[tuple[str, str, float]]:
        vector_provider = resolve_optional_vector_provider(vector_provider)

        if vector_provider is None:
            raise ValueError("No vector provider configured")

        results = (
            await compute_adapter()
            .submit(
                partial(vector_provider.query, query, limit=limit),
                admission_class="interactive-read",
                estimated_bytes=len(query.encode("utf-8")),
            )
            .wait()
        )
        if not results:
            return []

        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = await self._get_message_session_mapping(message_ids)

        return [(msg_to_conv[msg_id], msg_id, distance) for msg_id, distance in results if msg_id in msg_to_conv]

    async def get_archive_stats(self, *, conn: aiosqlite.Connection | None = None) -> ArchiveStats:
        from polylogue.archive.stats import ArchiveStats

        if conn is None:
            async with self._backend.read_connection() as active_conn:
                return await self.get_archive_stats(conn=active_conn)

        started_snapshot = False
        if not conn.in_transaction:
            await conn.execute("BEGIN")
            started_snapshot = True

        try:
            cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
            conv_row = await cursor.fetchone()
            conv_count = int(conv_row[0]) if conv_row is not None else 0

            cursor = await conn.execute("SELECT COUNT(*) FROM messages")
            msg_row = await cursor.fetchone()
            msg_count = int(msg_row[0]) if msg_row is not None else 0

            cursor = await conn.execute("SELECT COUNT(*) FROM attachments")
            att_row = await cursor.fetchone()
            att_count = int(att_row[0]) if att_row is not None else 0

            cursor = await conn.execute(
                """
                SELECT origin, COUNT(*) as count
                FROM sessions
                GROUP BY origin
                """
            )
            provider_rows = await cursor.fetchall()
            providers = {row["origin"]: row["count"] for row in provider_rows}

            embedding_stats = await read_embedding_stats_async(conn, include_retrieval_bands=False)
        finally:
            if started_snapshot and conn.in_transaction:
                await conn.rollback()

        db_size = 0
        try:
            db_size = self._backend.db_path.stat().st_size
        except Exception as exc:
            logger.warning("DB size check failed: %s", exc)

        return ArchiveStats(
            total_sessions=conv_count,
            total_messages=msg_count,
            total_attachments=att_count,
            origins=providers,
            embedded_sessions=embedding_stats.embedded_sessions,
            embedded_messages=embedding_stats.embedded_messages,
            pending_embedding_sessions=embedding_stats.pending_sessions,
            stale_embedding_messages=embedding_stats.stale_messages,
            messages_missing_embedding_provenance=embedding_stats.messages_missing_provenance,
            embedding_oldest_at=embedding_stats.oldest_embedded_at,
            embedding_newest_at=embedding_stats.newest_embedded_at,
            embedding_models=embedding_stats.model_counts,
            embedding_dimensions=embedding_stats.dimension_counts,
            embedding_coverage_unmeasurable_reason=embedding_stats.coverage_unmeasurable_reason,
            db_size_bytes=db_size,
        )
