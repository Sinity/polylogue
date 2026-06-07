"""Archive session/message/query methods for the async SQLite backend.

This mixin (SQLiteArchiveMixin) serves the SQLiteBackend with read AND
write capability. Most reads delegate to self.queries (a SQLiteQueryStore),
while writes use self._get_connection() directly.

Relationship with SQLiteQueryStoreArchiveMixin (query_store_archive.py):

- SQLiteArchiveMixin is the *backend* layer: write-capable, schema-ensured
  connections. Reads usually delegate to the underlying query store.
- SQLiteQueryStoreArchiveMixin is the *query store* layer: read-only, uses
  plain connection factory, implements the SQLiteQueryStore public API.

Intentional divergences (10 known, all architectural):

1. Naming: _session_id_query (private delegate) vs
   session_id_query (public query-store API). Same function, different
   callers.

2. search_sessions: delegates to queries (backend) vs delegates to
   search_session_hits().session_ids() (query store). Same result.

3. get_messages: content_blocks are pre-attached by the query store layer;
   the backend inherits this. The query store does a two-step load+merge
   because it IS the canonical implementation.

4. Connection management: _get_connection() ensures schema before every use
   (backend); _connection_factory provides pre-configured read-only
   connections (query store). Both correct for their context.

5. Write methods (save_session_record, save_messages, etc.) exist only
   on SQLiteArchiveMixin. The query store is deliberately read-only.

6. Query API methods (list_sessions, count_sessions,
   search_action_*, search_session_evidence_hits) exist only on
   SQLiteQueryStoreArchiveMixin. The backend accesses them through
   self.queries (a SQLiteQueryStore instance).

7. get_session_insight_status exists only on SQLiteArchiveMixin. The query
   store has its own implementation in query_store.py.

8. get_messages_batch: both have equivalent behavior. The backend delegates
   to queries; the query store adds an explicit empty-session_ids early
   exit for clarity.

9. iter_messages: the backend has a chunk_size=100 fast path that delegates
   to the query store; both call messages_q.iter_messages underneath.

10. search_session_hits: backend delegates to self.queries; query store
    opens a direct connection. Same destination through different layers.

These divergences reflect the different roles, not bugs. The backend is a
write-capable full DB owner; the query store is a composable read-only
component usable across multiple backend types."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.core.sources import origin_from_provider
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    MessageRecord,
    SessionEventRecord,
    SessionRecord,
)
from polylogue.storage.search.cache import invalidate_search_cache
from polylogue.storage.search.models import SessionSearchResult
from polylogue.storage.sqlite.archive_tiers.write import _timestamp_ms
from polylogue.storage.sqlite.queries import attachments as attachments_q
from polylogue.storage.sqlite.queries import messages as messages_q
from polylogue.storage.sqlite.queries import sessions as sessions_q
from polylogue.storage.sqlite.queries.stats import (
    AggregateMessageStats,
    ProviderMetricsRow,
    ProviderSessionCountRow,
)
from polylogue.storage.sqlite.queries.tool_usage import (
    ToolUsageProviderCoverageRow,
    ToolUsageRow,
)
from polylogue.types import Provider

if TYPE_CHECKING:
    import aiosqlite

    from polylogue.storage.sqlite.queries.messages import MessageTypeName
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class SQLiteArchiveMixin:
    """Session/message/archive-query methods for ``SQLiteBackend``."""

    if TYPE_CHECKING:
        queries: SQLiteQueryStore
        _transaction_depth: int
        _db_path: Path

        def _get_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]: ...

    @staticmethod
    def _is_current_archive_index_path(path: Path) -> bool:
        if path.name != "index.db":
            return False
        root = path.parent
        return all((root / filename).exists() for filename in ("source.db", "index.db", "user.db", "ops.db"))

    @staticmethod
    def _content_hash_blob(value: str) -> bytes:
        blob = bytes.fromhex(value)
        if len(blob) != 32:
            raise ValueError("content_hash must be a SHA-256 hex digest")
        return blob

    @staticmethod
    def _native_id_from_current_id(current_id: str, origin: str) -> str:
        prefix = f"{origin}:"
        return current_id.removeprefix(prefix)

    async def _resolve_current_session_id(
        self,
        conn: aiosqlite.Connection,
        *,
        session_id: str,
        source_name: str,
    ) -> str:
        cursor = await conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        if row is not None:
            return str(row["session_id"])
        cursor = await conn.execute(
            "SELECT session_id FROM sessions WHERE native_id = ? ORDER BY session_id LIMIT 2",
            (session_id,),
        )
        rows = list(await cursor.fetchall())
        if len(rows) == 1:
            return str(rows[0]["session_id"])
        origin = origin_from_provider(Provider.from_string(source_name)).value
        native_id = self._native_id_from_current_id(session_id, origin)
        cursor = await conn.execute(
            "SELECT session_id FROM sessions WHERE origin = ? AND native_id = ?",
            (origin, native_id),
        )
        row = await cursor.fetchone()
        if row is not None:
            return str(row["session_id"])
        raise ValueError(f"Cannot write child rows for unknown session {session_id!r}")

    async def _replace_working_dirs(
        self,
        conn: aiosqlite.Connection,
        *,
        session_id: str,
        working_directories_json: str | None,
    ) -> None:
        if working_directories_json is None:
            return
        try:
            raw = json.loads(working_directories_json)
        except json.JSONDecodeError as exc:
            raise ValueError("working_directories_json must be a JSON list") from exc
        if not isinstance(raw, list):
            raise ValueError("working_directories_json must be a JSON list")
        paths = [str(item) for item in raw if isinstance(item, str) and item]
        await conn.execute("DELETE FROM session_working_dirs WHERE session_id = ?", (session_id,))
        await conn.executemany(
            """
            INSERT INTO session_working_dirs (session_id, path, position)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id, path) DO UPDATE SET position = excluded.position
            """,
            [(session_id, path, position) for position, path in enumerate(paths)],
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve a session by ID."""
        return await self.queries.get_session(session_id)

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]:
        """Retrieve multiple sessions in a single query.

        Preserves the order of input IDs. Missing IDs are silently skipped.
        """
        return await self.queries.get_sessions_batch(ids)

    async def aggregate_message_stats(
        self,
        session_ids: list[str] | None = None,
    ) -> AggregateMessageStats:
        """Compute aggregate message statistics via SQL."""
        return await self.queries.aggregate_message_stats(session_ids)

    async def session_exists_by_hash(self, content_hash: str) -> bool:
        """Check if session with given content hash exists."""
        return await self.queries.session_exists_by_hash(content_hash)

    async def save_session_record(self, record: SessionRecord) -> None:
        """Persist a session record with upsert semantics."""
        async with self._get_connection() as conn:
            if not self._is_current_archive_index_path(Path(self._db_path)):
                raise RuntimeError("archive writes require the current split archive file set")
            origin = record.origin
            native_id = record.native_id
            await conn.execute(
                """
                INSERT INTO sessions (
                    native_id, origin, parent_session_id, raw_id, branch_type,
                    title, git_branch, git_repository_url, content_hash,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(origin, native_id) DO UPDATE SET
                    parent_session_id = excluded.parent_session_id,
                    raw_id = COALESCE(excluded.raw_id, sessions.raw_id),
                    branch_type = excluded.branch_type,
                    title = COALESCE(excluded.title, sessions.title),
                    git_branch = excluded.git_branch,
                    git_repository_url = excluded.git_repository_url,
                    content_hash = excluded.content_hash,
                    created_at_ms = COALESCE(sessions.created_at_ms, excluded.created_at_ms),
                    updated_at_ms = MAX(COALESCE(sessions.updated_at_ms, 0), COALESCE(excluded.updated_at_ms, 0))
                """,
                (
                    native_id,
                    origin.value,
                    record.parent_session_id,
                    record.raw_id,
                    record.branch_type.value if record.branch_type is not None else None,
                    record.title,
                    record.git_branch,
                    record.git_repository_url,
                    self._content_hash_blob(record.content_hash),
                    _timestamp_ms(record.created_at),
                    _timestamp_ms(record.updated_at),
                ),
            )
            current_session_id = f"{origin.value}:{native_id}"
            await self._replace_working_dirs(
                conn,
                session_id=current_session_id,
                working_directories_json=record.working_directories_json,
            )
            if self._transaction_depth == 0:
                await conn.commit()
            invalidate_search_cache()

    async def get_messages(self, session_id: str) -> list[MessageRecord]:
        """Get all messages for a session, with content_blocks attached."""
        return await self.queries.get_messages(session_id)

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MessageRecord], int]:
        """Get paginated messages for a session with optional filters.

        Returns (messages, total_count) where total_count is the count of
        messages matching the filters before pagination.
        """
        return await self.queries.get_messages_paginated(
            session_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )

    async def get_messages_batch(
        self,
        session_ids: list[str],
        *,
        sort_key_since: float | None = None,
        sort_key_until: float | None = None,
        message_role: MessageRoleFilter = (),
    ) -> dict[str, list[MessageRecord]]:
        """Get messages for multiple sessions in a single query, with content_blocks."""
        return await self.queries.get_messages_batch(
            session_ids,
            sort_key_since=sort_key_since,
            sort_key_until=sort_key_until,
            message_role=message_role,
        )

    @staticmethod
    def _topo_sort_messages(records: list[MessageRecord]) -> list[MessageRecord]:
        """Sort messages so parents come before children (for FK constraint)."""
        return messages_q.topo_sort_messages(records)

    async def save_messages(self, records: list[MessageRecord]) -> None:
        """Persist multiple message records using bulk insert."""
        async with self._get_connection() as conn:
            if not self._is_current_archive_index_path(Path(self._db_path)):
                raise RuntimeError("archive writes require the current split archive file set")
            records = messages_q.topo_sort_messages(records)
            for position, record in enumerate(records):
                session_id = await self._resolve_current_session_id(
                    conn,
                    session_id=str(record.session_id),
                    source_name=record.source_name,
                )
                native_id = record.provider_message_id
                if native_id is None:
                    native_id = str(record.message_id).removeprefix(f"{session_id}:")
                await conn.execute(
                    """
                    INSERT INTO messages (
                        session_id, native_id, parent_message_id, position,
                        role, message_type, model_name, has_tool_use,
                        has_thinking, has_paste, paste_boundary, variant_index,
                        word_count, input_tokens, output_tokens, cache_read_tokens,
                        cache_write_tokens, content_hash, occurred_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, position, variant_index) DO UPDATE SET
                        native_id = excluded.native_id,
                        parent_message_id = excluded.parent_message_id,
                        role = excluded.role,
                        message_type = excluded.message_type,
                        model_name = excluded.model_name,
                        has_tool_use = excluded.has_tool_use,
                        has_thinking = excluded.has_thinking,
                        has_paste = excluded.has_paste,
                        paste_boundary = excluded.paste_boundary,
                        word_count = excluded.word_count,
                        input_tokens = excluded.input_tokens,
                        output_tokens = excluded.output_tokens,
                        cache_read_tokens = excluded.cache_read_tokens,
                        cache_write_tokens = excluded.cache_write_tokens,
                        content_hash = excluded.content_hash,
                        occurred_at_ms = excluded.occurred_at_ms
                    """,
                    (
                        session_id,
                        native_id,
                        record.parent_message_id,
                        position,
                        record.role.value if record.role is not None else "unknown",
                        record.message_type.value,
                        record.model_name,
                        int(record.has_tool_use),
                        int(record.has_thinking),
                        int(record.has_paste),
                        record.paste_boundary_state,
                        record.branch_index,
                        record.word_count,
                        record.input_tokens,
                        record.output_tokens,
                        record.cache_read_tokens,
                        record.cache_write_tokens,
                        self._content_hash_blob(record.content_hash),
                        int(record.sort_key * 1000) if record.sort_key is not None else None,
                    ),
                )
                text = record.text
                if text:
                    await conn.execute(
                        """
                        INSERT INTO blocks (
                            message_id, session_id, position, block_type, text
                        ) VALUES (
                            (SELECT message_id FROM messages WHERE session_id = ? AND position = ? AND variant_index = ?),
                            ?, 0, 'text', ?
                        )
                        ON CONFLICT(message_id, position) DO UPDATE SET
                            block_type = excluded.block_type,
                            text = excluded.text
                        """,
                        (
                            session_id,
                            position,
                            record.branch_index,
                            session_id,
                            text,
                        ),
                    )
            if self._transaction_depth == 0:
                await conn.commit()
            invalidate_search_cache()

    async def save_content_blocks(self, records: list[ContentBlockRecord]) -> None:
        """Persist content block records using bulk insert."""
        async with self._get_connection() as conn:
            if not self._is_current_archive_index_path(Path(self._db_path)):
                raise RuntimeError("archive writes require the current split archive file set")
            for record in records:
                message_id = str(record.message_id)
                session_id = str(record.session_id)
                resolved_session_id = await self._resolve_current_session_id(
                    conn,
                    session_id=session_id,
                    source_name="",
                )
                cursor = await conn.execute(
                    """
                    SELECT message_id, session_id
                    FROM messages
                    WHERE message_id = ?
                       OR (session_id = ? AND native_id = ?)
                    ORDER BY CASE WHEN message_id = ? THEN 0 ELSE 1 END
                    LIMIT 1
                    """,
                    (message_id, resolved_session_id, message_id, message_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise ValueError(f"Cannot write block for unknown message {message_id!r}")
                message_id = str(row["message_id"])
                session_id = str(row["session_id"])
                await conn.execute(
                    """
                    INSERT INTO blocks (
                        message_id, session_id, position, block_type, text,
                        tool_name, tool_id, tool_input, semantic_type, language
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(message_id, position) DO UPDATE SET
                        block_type = excluded.block_type,
                        text = excluded.text,
                        tool_name = excluded.tool_name,
                        tool_id = excluded.tool_id,
                        tool_input = excluded.tool_input,
                        semantic_type = excluded.semantic_type,
                        language = excluded.language
                    """,
                    (
                        message_id,
                        session_id,
                        record.block_index,
                        record.type.value,
                        record.text,
                        record.tool_name,
                        record.tool_id,
                        record.tool_input,
                        record.semantic_type.value if record.semantic_type is not None else None,
                        _metadata_language(record.metadata),
                    ),
                )
            if self._transaction_depth == 0:
                await conn.commit()
            invalidate_search_cache()

    async def get_content_blocks(self, message_ids: list[str]) -> dict[str, list[ContentBlockRecord]]:
        """Get content blocks for a list of message IDs."""
        return await self.queries.get_content_blocks(message_ids)

    async def get_attachments(self, session_id: str) -> list[AttachmentRecord]:
        """Get all attachments for a session."""
        return await self.queries.get_attachments(session_id)

    async def get_attachments_batch(self, session_ids: list[str]) -> dict[str, list[AttachmentRecord]]:
        """Get attachments for multiple sessions in a single query."""
        return await self.queries.get_attachments_batch(session_ids)

    async def get_session_events(self, session_id: str) -> list[SessionEventRecord]:
        """Get timeline events for a session."""
        return await self.queries.get_session_events(session_id)

    async def get_session_events_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[SessionEventRecord]]:
        """Get timeline events for multiple sessions."""
        return await self.queries.get_session_events_batch(session_ids)

    async def save_attachments(self, records: list[AttachmentRecord]) -> None:
        """Persist attachment records with reference counting."""
        async with self._get_connection() as conn:
            await attachments_q.save_attachments(conn, records, self._transaction_depth)

    async def prune_attachments(self, session_id: str, keep_attachment_ids: set[str]) -> None:
        """Remove attachment refs not in keep set and clean up orphaned attachments."""
        async with self._get_connection() as conn:
            await attachments_q.prune_attachments(conn, session_id, keep_attachment_ids, self._transaction_depth)

    async def list_sessions_by_parent(self, parent_id: str) -> list[SessionRecord]:
        """List all sessions that have the given session as parent."""
        async with self._get_connection() as conn:
            return await sessions_q.list_sessions_by_parent(conn, parent_id)

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> str | None:
        """Resolve a partial session ID to a full ID."""
        return await self.queries.resolve_id(id_prefix, strict=strict)

    async def get_last_sync_timestamp(self) -> str | None:
        """Return the timestamp of the most recent ingestion run, or None."""
        return await self.queries.get_last_sync_timestamp()

    def _session_id_query(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Build the canonical scoped session-ID query."""
        return self.queries.session_id_query(source_names=source_names)

    async def count_session_ids(
        self,
        *,
        source_names: list[str] | None = None,
    ) -> int:
        """Count session IDs, optionally scoped to source names."""
        return await self.queries.count_session_ids(source_names=source_names)

    async def iter_session_ids(
        self,
        *,
        source_names: list[str] | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[str]:
        """Iterate session IDs in bounded fetch batches."""
        async for cid in self.queries.iter_session_ids(source_names=source_names, page_size=page_size):
            yield cid

    async def get_session_insight_status(self, *, verify_freshness: bool = True) -> SessionInsightStatusSnapshot:
        """Return materialized session-insight coverage counters."""
        return await self.queries.get_session_insight_status(verify_freshness=verify_freshness)

    async def search_sessions(self, query: str, limit: int = 100, providers: list[str] | None = None) -> list[str]:
        """Search sessions using the canonical ranked FTS session query."""
        return await self.queries.search_sessions(query, limit, providers)

    async def search_session_hits(
        self,
        query: str,
        limit: int = 100,
        providers: list[str] | None = None,
    ) -> SessionSearchResult:
        """Search sessions while preserving ordered session-hit metadata."""
        return await self.queries.search_session_hits(query, limit, providers)

    async def iter_messages(
        self,
        session_id: str,
        *,
        chunk_size: int = 100,
        dialogue_only: bool = False,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[MessageRecord]:
        """Stream messages in chunks instead of loading all at once."""
        if chunk_size != 100:
            async with self._get_connection() as conn:
                async for msg in messages_q.iter_messages(
                    conn,
                    session_id,
                    chunk_size=chunk_size,
                    dialogue_only=dialogue_only,
                    message_roles=message_roles,
                    limit=limit,
                ):
                    yield msg
            return
        async for msg in self.queries.iter_messages(
            session_id,
            dialogue_only=dialogue_only,
            message_roles=message_roles,
            limit=limit,
        ):
            yield msg

    async def get_session_stats(self, session_id: str) -> dict[str, int]:
        """Get message counts without loading messages."""
        return await self.queries.get_session_stats(session_id)

    async def get_message_counts_batch(self, session_ids: list[str]) -> dict[str, int]:
        """Get message counts for multiple sessions in a single query."""
        return await self.queries.get_message_counts_batch(session_ids)

    async def get_stats_by(self, group_by: str = "provider") -> dict[str, int]:
        """Get session counts grouped by provider, month, or year."""
        return await self.queries.get_stats_by(group_by)

    async def get_provider_session_counts(self) -> list[ProviderSessionCountRow]:
        """Return session counts per provider."""
        return await self.queries.get_provider_session_counts()

    async def get_provider_metrics_rows(self) -> list[ProviderMetricsRow]:
        """Return raw provider aggregation rows for analytics reporting."""
        return await self.queries.get_provider_metrics_rows()

    async def get_tool_usage_rows(self) -> list[ToolUsageRow]:
        """Return per-(provider, tool, action_kind) tool usage rows."""
        return await self.queries.get_tool_usage_rows()

    async def get_tool_usage_provider_coverage_rows(
        self,
    ) -> list[ToolUsageProviderCoverageRow]:
        """Return per-provider tool-data coverage signals."""
        return await self.queries.get_tool_usage_provider_coverage_rows()


def _metadata_language(metadata: str | None) -> str | None:
    if not metadata:
        return None
    try:
        import json

        parsed = json.loads(metadata)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("language")
    return str(value) if value is not None else None


__all__ = ["SQLiteArchiveMixin"]
