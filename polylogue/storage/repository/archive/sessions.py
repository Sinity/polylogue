"""Session and message hydration reads for the repository."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.core.sources import provider_from_origin
from polylogue.storage.archive_views import SessionRenderProjection
from polylogue.storage.hydrators import (
    message_from_record,
    session_from_records,
    session_summary_from_record,
)
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import AttachmentRecord, MessageRecord, SessionRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.queries.messages import MessageTypeName
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore
    from polylogue.types import SessionId


class RepositoryArchiveSessionMixin:
    if TYPE_CHECKING:
        _backend: RepositoryBackendProtocol
        queries: SQLiteQueryStore

    async def _fetch_tags_by_session(self, session_ids: list[str]) -> dict[str, tuple[str, ...]]:
        """#1240: batch-fetch M2M tags for hydration of Session/SessionSummary."""
        if not session_ids:
            return {}
        result: dict[str, list[str]] = {cid: [] for cid in session_ids}
        async with self._backend.connection() as conn:
            for table in ("session_tags", "tags"):
                table_cursor = await conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                    (table,),
                )
                if await table_cursor.fetchone() is None:
                    return dict.fromkeys(session_ids, ())
            placeholders = ",".join("?" for _ in session_ids)
            cursor = await conn.execute(
                f"""
                SELECT ct.session_id AS cid, t.name AS name
                FROM session_tags ct
                JOIN tags t ON t.id = ct.tag_id
                WHERE ct.session_id IN ({placeholders})
                ORDER BY t.name
                """,
                session_ids,
            )
            rows = await cursor.fetchall()
            for row in rows:
                cid = row["cid"]
                name = row["name"]
                if cid in result:
                    result[cid].append(name)
        return {cid: tuple(names) for cid, names in result.items()}

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> SessionId | None:
        resolved = await self.queries.resolve_id(id_prefix, strict=strict)
        from polylogue.types import SessionId

        return SessionId(resolved) if resolved else None

    async def get(self, session_id: str) -> Session | None:
        conv_record = await self.queries.get_session(session_id)
        if not conv_record:
            return None
        resolved_session_id = str(conv_record.session_id)

        msg_records, att_records, session_event_records = await asyncio.gather(
            self.queries.get_messages(resolved_session_id),
            self.queries.get_attachments(resolved_session_id),
            self.queries.get_session_events(resolved_session_id),
        )
        tags_by_id = await self._fetch_tags_by_session([resolved_session_id])
        return session_from_records(
            conv_record,
            msg_records,
            att_records,
            session_event_records,
            tags=tags_by_id.get(resolved_session_id, ()),
        )

    async def get_render_projection(self, session_id: str) -> SessionRenderProjection | None:
        conv_record = await self.queries.get_session(session_id)
        if not conv_record:
            return None

        msg_records, att_records = await asyncio.gather(
            self.queries.get_messages(session_id),
            self.queries.get_attachments(session_id),
        )
        return SessionRenderProjection(
            session=conv_record,
            messages=msg_records,
            attachments=att_records,
        )

    async def view(self, session_id: str) -> Session | None:
        full_id = await self.resolve_id(session_id) or session_id
        return await self.get(str(full_id))

    async def get_eager(self, session_id: str) -> Session | None:
        return await self.get(session_id)

    async def get_messages(self, session_id: str) -> list[MessageRecord]:
        return await self.queries.get_messages(session_id)

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Message], int]:
        conv_record = await self.queries.get_session(session_id)
        source_name = provider_from_origin(conv_record.origin).value if conv_record else None
        records, total = await self.queries.get_messages_paginated(
            session_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )
        messages = [message_from_record(r, attachments=[], provider=source_name) for r in records]
        return messages, total

    async def get_sessions_batch(self, ids: list[str]) -> list[SessionRecord]:
        return await self.queries.get_sessions_batch(ids)

    async def get_messages_batch(
        self,
        session_ids: list[str],
        *,
        sort_key_since: float | None = None,
        sort_key_until: float | None = None,
        message_role: MessageRoleFilter = (),
    ) -> dict[str, list[MessageRecord]]:
        return await self.queries.get_messages_batch(
            session_ids,
            sort_key_since=sort_key_since,
            sort_key_until=sort_key_until,
            message_role=message_role,
        )

    async def get_attachments_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[AttachmentRecord]]:
        return await self.queries.get_attachments_batch(session_ids)

    async def _hydrate_sessions(
        self,
        session_records: list[SessionRecord],
        *,
        ordered_ids: list[str] | None = None,
    ) -> list[Session]:
        if not session_records:
            return []

        by_id: dict[str, SessionRecord] = {str(record.session_id): record for record in session_records}
        session_ids = ordered_ids or [record.session_id for record in session_records]
        present_ids = [session_id for session_id in session_ids if session_id in by_id]
        if not present_ids:
            return []

        async with self._backend.read_pool(size=2):
            msgs_by_id, atts_by_id, session_events_by_id = await asyncio.gather(
                self.queries.get_messages_batch(present_ids),
                self.queries.get_attachments_batch(present_ids),
                self.queries.get_session_events_batch(present_ids),
            )
        tags_by_id = await self._fetch_tags_by_session(present_ids)
        return [
            session_from_records(
                by_id[session_id],
                msgs_by_id.get(session_id, []),
                atts_by_id.get(session_id, []),
                session_events_by_id.get(session_id, []),
                tags=tags_by_id.get(session_id, ()),
            )
            for session_id in present_ids
        ]

    async def get_summary(self, session_id: str) -> SessionSummary | None:
        conv_record = await self.queries.get_session(session_id)
        if not conv_record:
            return None
        tags_by_id = await self._fetch_tags_by_session([session_id])
        # Hydrate message_count from the current sessions aggregate.
        counts_by_id = await self.queries.get_message_counts_batch([session_id])
        return session_summary_from_record(
            conv_record,
            tags=tags_by_id.get(session_id, ()),
            message_count=counts_by_id.get(session_id),
        )

    async def list_summaries_by_query(
        self,
        query: SessionRecordQuery,
    ) -> list[SessionSummary]:
        conv_records = await self.queries.list_session_summaries(query)
        ids = [str(record.session_id) for record in conv_records]
        tags_by_id = await self._fetch_tags_by_session(ids)
        # Hydrate message_count from the current sessions aggregate.
        counts_by_id = await self.queries.get_message_counts_batch(ids) if ids else {}
        return [
            session_summary_from_record(
                record,
                tags=tags_by_id.get(str(record.session_id), ()),
                message_count=counts_by_id.get(str(record.session_id)),
            )
            for record in conv_records
        ]

    async def list_by_query(
        self,
        query: SessionRecordQuery,
    ) -> list[Session]:
        conv_records = await self.queries.list_sessions(query)
        return await self._hydrate_sessions(conv_records)

    async def get_many(self, session_ids: list[str]) -> list[Session]:
        if not session_ids:
            return []
        records = await self.queries.get_sessions_batch(session_ids)
        return await self._hydrate_sessions(records, ordered_ids=session_ids)

    async def iter_messages(
        self,
        session_id: str,
        *,
        message_roles: MessageRoleFilter = (),
        limit: int | None = None,
    ) -> AsyncIterator[Message]:
        conv_record = await self.queries.get_session(session_id)
        source_name = provider_from_origin(conv_record.origin).value if conv_record else None
        async for record in self.queries.iter_messages(
            session_id,
            message_roles=message_roles,
            limit=limit,
        ):
            yield message_from_record(record, attachments=[], provider=source_name)

    async def aggregate_facet_families(
        self,
        *,
        session_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Run per-family SQL aggregators for facets that can't be computed
        from session summaries alone.

        Returns a dict keyed by family name (``repos``, ``message_types``,
        ``action_types``, ``has_flags``). When ``session_ids`` is
        provided, results are scoped to those sessions.

        #1672 (phase 2 of #1623).
        """
        result: dict[str, dict[str, int]] = {
            "repos": {},
            "message_types": {},
            "action_types": {},
            "has_flags": {},
        }
        # Empty list produces SQL ``IN ()`` which is a syntax error.
        if session_ids is not None and not session_ids:
            return result
        # SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999. When the
        # scoped path passes >900 IDs we chunk and merge so the query
        # never exceeds the engine limit. The global path (None) has no
        # IN clause and does not need chunking.
        _max_in_vars = 900

        from typing import Any

        async def _scoped_rows(
            conn: Any,
            scoped_sql: str,
            global_sql: str,
            params: list[str] | None,
        ) -> list[Any]:
            """Run a possibly-chunked scoped query or a single global query."""
            if params is None:
                return list(await (await conn.execute(global_sql)).fetchall())
            rows: list[object] = []
            for i in range(0, len(params), _max_in_vars):
                chunk = params[i : i + _max_in_vars]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(await (await conn.execute(scoped_sql.format(placeholders), chunk)).fetchall())
            return rows

        # Accumulate keyed results from possibly-chunked rows.
        def _keyed(rows: list[Any]) -> dict[str, int]:
            return {row[0]: row[1] for row in rows if row[0]}

        async with self._backend.connection() as conn:
            ids = session_ids  # None → global, list → scoped

            result["repos"] = _keyed(
                await _scoped_rows(
                    conn,
                    """SELECT git_repository_url, count(*) AS n
                    FROM sessions
                    WHERE git_repository_url IS NOT NULL
                      AND session_id IN ({})
                    GROUP BY git_repository_url""",
                    """SELECT git_repository_url, count(*) AS n
                    FROM sessions
                    WHERE git_repository_url IS NOT NULL
                    GROUP BY git_repository_url""",
                    ids,
                )
            )

            result["message_types"] = _keyed(
                await _scoped_rows(
                    conn,
                    """SELECT message_type, count(*) AS n
                    FROM messages
                    WHERE session_id IN ({})
                    GROUP BY message_type""",
                    """SELECT message_type, count(*) AS n
                    FROM messages
                    GROUP BY message_type""",
                    ids,
                )
            )

            result["action_types"] = _keyed(
                await _scoped_rows(
                    conn,
                    """SELECT COALESCE(NULLIF(semantic_type, ''), 'tool_use') AS action_kind, count(*) AS n
                    FROM actions
                    WHERE session_id IN ({})
                    GROUP BY COALESCE(NULLIF(semantic_type, ''), 'tool_use')""",
                    """SELECT COALESCE(NULLIF(semantic_type, ''), 'tool_use') AS action_kind, count(*) AS n
                    FROM actions
                    GROUP BY COALESCE(NULLIF(semantic_type, ''), 'tool_use')""",
                    ids,
                )
            )

            flag_rows = await _scoped_rows(
                conn,
                """SELECT coalesce(sum(has_tool_use),0), coalesce(sum(has_thinking),0),
                coalesce(sum(has_paste),0) FROM messages WHERE session_id IN ({})""",
                """SELECT coalesce(sum(has_tool_use),0), coalesce(sum(has_thinking),0),
                coalesce(sum(has_paste),0) FROM messages""",
                ids,
            )
            # Merge chunked flag rows by summing corresponding columns.
            tool_use = sum(r[0] for r in flag_rows)
            thinking = sum(r[1] for r in flag_rows)
            paste = sum(r[2] for r in flag_rows)
            result["has_flags"] = {"has_tool_use": tool_use, "has_thinking": thinking, "has_paste": paste}

        return result


__all__ = ["RepositoryArchiveSessionMixin"]
