"""Session and message hydration reads for the repository."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING

from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import MessageRoleFilter
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.archive.session.events import SessionEvent
from polylogue.storage.derived.session.profiles import hydrate_session_profile
from polylogue.storage.hydrators import (
    message_from_record,
    session_event_from_record,
    session_from_records,
    session_summary_from_record,
)
from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
from polylogue.storage.runtime import (
    SESSION_INSIGHT_MATERIALIZER_VERSION,
    AttachmentRecord,
    FileEditRecord,
    LineageCompleteness,
    MessageRecord,
    SessionCommitRecord,
    SessionRecord,
    SessionRefRecord,
    WebContentConstructRecord,
)
from polylogue.storage.sqlite.archive_tiers.write import ArchiveAgentPolicy

if TYPE_CHECKING:
    from polylogue.archive.session.session_profile import SessionProfile
    from polylogue.core.types import SessionId
    from polylogue.storage.sqlite.queries.messages import MaterialOriginFilter, MessageTypeName
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


def _with_profile(summary: SessionSummary, profile: SessionProfile | None) -> SessionSummary:
    """Overlay materialized profile facts, identically on single and list reads.

    A profile with ``unknown`` cost provenance stores zero as a placeholder;
    the summary keeps ``None`` (cost unavailable) rather than publish that
    zero as a known cost.
    """
    if profile is None:
        return summary
    update: dict[str, object] = {"terminal_state": profile.terminal_state}
    if profile.cost_provenance != "unknown":
        update["total_cost_usd"] = profile.total_cost_usd
        update["cost_provenance"] = profile.cost_provenance
    return summary.model_copy(update=update)


class RepositoryArchiveSessionMixin:
    if TYPE_CHECKING:
        from polylogue.storage.blob_store import BlobStore

        _backend: RepositoryBackendProtocol
        _read_blob_store: BlobStore
        queries: SQLiteQueryStore

    async def _current_profiles(
        self, session_ids: list[str], *, queries: SQLiteQueryStore
    ) -> dict[str, SessionProfile]:
        """Profiles still bound to their session's current input.

        An ingest that changes a session clears the profile's
        ``input_content_hash`` and enqueues profile demand; until convergence
        republishes it, the stale row's facts are not current summary facts.
        """
        if not session_ids:
            return {}
        records = await queries.get_session_profiles_batch(session_ids)
        return {
            session_id: hydrate_session_profile(record)
            for session_id, record in records.items()
            if record.input_content_hash is not None
            and record.materializer_version == SESSION_INSIGHT_MATERIALIZER_VERSION
        }

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> SessionId | None:
        resolved = await self.queries.resolve_id(id_prefix, strict=strict)
        from polylogue.core.types import SessionId

        return SessionId(resolved) if resolved else None

    async def get(self, session_id: str) -> Session | None:
        async with self.queries.read_snapshot() as queries:
            conv_record = await queries.get_session(session_id)
            if not conv_record:
                return None
            resolved_session_id = str(conv_record.session_id)

            msg_records = await queries.get_messages(resolved_session_id)
            session_event_records = await queries.get_session_events(resolved_session_id)
            tags_by_id = await queries.get_session_tags_batch([resolved_session_id])
            return session_from_records(
                conv_record,
                msg_records,
                [attachment for message in msg_records for attachment in message.attachments],
                session_event_records,
                tags=tags_by_id.get(resolved_session_id, ()),
                blob_store=self._read_blob_store,
            )

    async def view(self, session_id: str) -> Session | None:
        full_id = await self.resolve_id(session_id) or session_id
        return await self.get(str(full_id))

    async def get_messages(self, session_id: str) -> list[MessageRecord]:
        return await self.queries.get_messages(session_id)

    async def get_session_event_models(self, session_id: str) -> list[SessionEvent]:
        """Hydrate a session's timeline events without reading its transcript.

        The envelope read behind the API's ``get_session`` carries no
        ``session_events``, so a caller that needs them (the session digest's
        compaction geometry, polylogue-4ts.5) would otherwise re-read the whole
        session through the repository.
        """
        records = await self.queries.get_session_events(session_id)
        return [session_event_from_record(record) for record in records]

    async def get_agent_policies(self, session_id: str) -> list[ArchiveAgentPolicy]:
        """Read agent-policy facts (sandbox/approval/network policy) for a session.

        The writer diverts Codex ``agent_policy`` events out of
        ``session_events`` into the dedicated ``session_agent_policies``
        table (fully re-derivable, zero evidence loss -- see
        ``archive_tiers/write.py:_SESSION_EVENTS_REDUNDANT_TYPES``), but
        until this method existed nothing on the repository/API/MCP surface
        could read it back: the sole prior reader was a sync helper
        (``read_session_agent_policies``) exercised only by tests.
        """
        return await self.queries.get_session_agent_policies(session_id)

    async def get_agent_policies_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[ArchiveAgentPolicy]]:
        return await self.queries.get_session_agent_policies_batch(session_ids)

    async def get_file_edits(self, session_id: str) -> list[FileEditRecord]:
        """Read file-edit tool-call evidence (structuredPatch/originalFile/...) for a session.

        polylogue-2qx.4: the writer materializes ``ParsedFileEdit`` evidence
        into the dedicated ``file_edits`` table, keyed by the tool_use block
        that made the edit -- this is the read surface for it.
        """
        return await self.queries.get_file_edits_for_session(session_id)

    async def get_session_refs(self, session_id: str) -> list[SessionRefRecord]:
        """Read tracker-agnostic external references (pr-link, ...) for a session."""
        return await self.queries.get_session_refs(session_id)

    async def get_session_refs_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[SessionRefRecord]]:
        return await self.queries.get_session_refs_batch(session_ids)

    async def get_session_commits(self, session_id: str) -> list[SessionCommitRecord]:
        """Read checkout-HEAD commit evidence (polylogue-cijx.3) for a session."""
        return await self.queries.get_session_commits(session_id)

    async def get_web_content_constructs(
        self,
        session_id: str,
        *,
        construct_type: str | None = None,
    ) -> list[WebContentConstructRecord]:
        """Read typed web-export constructs (polylogue-kktg) for a session.

        Search queries/results, canvas documents, content references, image
        results, async tasks, selected sources, token budgets, and voice
        notes projected from ChatGPT/Claude web payloads
        (``core.enums.WebConstructType``) -- written every ingest into the
        dedicated ``web_content_constructs`` table but, before this reader,
        reachable only through an orphan-integrity DELETE sweep or a demo
        smoke-probe COUNT(*).
        """
        return await self.queries.get_web_content_constructs(session_id, construct_type=construct_type)

    async def get_web_content_constructs_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, list[WebContentConstructRecord]]:
        return await self.queries.get_web_content_constructs_batch(session_ids)

    async def get_messages_paginated(
        self,
        session_id: str,
        *,
        message_role: MessageRoleFilter = (),
        message_type: MessageTypeName | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Message], int, LineageCompleteness]:
        conv_record = await self.queries.get_session(session_id)
        origin = conv_record.origin if conv_record else None
        records, total, completeness = await self.queries.get_messages_paginated(
            session_id,
            message_role=message_role,
            message_type=message_type,
            limit=limit,
            offset=offset,
        )
        messages = [
            message_from_record(r, r.attachments, origin=origin, blob_store=self._read_blob_store) for r in records
        ]
        return messages, total, completeness

    async def get_effective_context(self, session_id: str, at_position: int | None = None) -> list[Message]:
        """Return the messages the model actually saw at ``at_position``.

        A compaction boundary replaces its recorded range with the materialized
        summary, so this is deliberately narrower than the lineage-composed
        transcript ``get_messages_paginated`` returns.
        """
        conv_record = await self.queries.get_session(session_id)
        origin = conv_record.origin if conv_record else None
        records = await self.queries.get_effective_context(session_id, at_position)
        return [
            message_from_record(record, record.attachments, origin=origin, blob_store=self._read_blob_store)
            for record in records
        ]

    async def get_lineage_completeness(self, session_id: str) -> LineageCompleteness:
        """Report whether ``session_id``'s composed transcript is the full
        logical transcript or was silently truncated (polylogue-ppkj).

        Cheap standalone probe for callers (e.g. a material-origin-filtered
        read) that need the same read-time completeness signal
        ``get_messages_paginated`` reports but build their message list a
        different way and so cannot thread it through that call.
        """
        return await self.queries.get_lineage_completeness(session_id)

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

    async def get_attachment_library_page(
        self, *, limit: int, offset: int, mime_filter: str = "", session_filter: str = "", state_filter: str = ""
    ) -> list[tuple[AttachmentRecord, str, str | None]]:
        return await self.queries.get_attachment_library_page(
            limit=limit,
            offset=offset,
            mime_filter=mime_filter,
            session_filter=session_filter,
            state_filter=state_filter,
            blob_store=self._read_blob_store,
        )

    async def _hydrate_sessions(
        self,
        session_records: list[SessionRecord],
        *,
        ordered_ids: list[str] | None = None,
        queries: SQLiteQueryStore,
    ) -> list[Session]:
        if not session_records:
            return []

        by_id: dict[str, SessionRecord] = {str(record.session_id): record for record in session_records}
        session_ids = ordered_ids or [record.session_id for record in session_records]
        present_ids = [session_id for session_id in session_ids if session_id in by_id]
        if not present_ids:
            return []

        msgs_by_id = await queries.get_messages_batch(present_ids)
        session_events_by_id = await queries.get_session_events_batch(present_ids)
        tags_by_id = await queries.get_session_tags_batch(present_ids)
        return [
            session_from_records(
                by_id[session_id],
                msgs_by_id.get(session_id, []),
                [attachment for message in msgs_by_id.get(session_id, []) for attachment in message.attachments],
                session_events_by_id.get(session_id, []),
                tags=tags_by_id.get(session_id, ()),
                blob_store=self._read_blob_store,
            )
            for session_id in present_ids
        ]

    async def get_summary(self, session_id: str) -> SessionSummary | None:
        async with self.queries.read_snapshot() as queries:
            conv_record = await queries.get_session(session_id)
            if not conv_record:
                return None
            tags_by_id = await queries.get_session_tags_batch([session_id])
            # Hydrate message_count from the current sessions aggregate.
            counts_by_id = await queries.get_message_counts_batch([session_id])
            profiles_by_id = await self._current_profiles([session_id], queries=queries)
            return _with_profile(
                session_summary_from_record(
                    conv_record,
                    tags=tags_by_id.get(session_id, ()),
                    message_count=counts_by_id.get(session_id),
                ),
                profiles_by_id.get(session_id),
            )

    async def list_summaries_by_query(
        self,
        query: SessionRecordQuery,
    ) -> list[SessionSummary]:
        async with self.queries.read_snapshot() as queries:
            conv_records = await queries.list_session_summaries(query)
            ids = [str(record.session_id) for record in conv_records]
            tags_by_id = await queries.get_session_tags_batch(ids)
            # Hydrate message_count from the current sessions aggregate.
            counts_by_id = await queries.get_message_counts_batch(ids) if ids else {}
            profiles_by_id = await self._current_profiles(ids, queries=queries)
            summaries: list[SessionSummary] = []
            for record in conv_records:
                session_id = str(record.session_id)
                summary = session_summary_from_record(
                    record,
                    tags=tags_by_id.get(session_id, ()),
                    message_count=counts_by_id.get(session_id),
                )
                summaries.append(_with_profile(summary, profiles_by_id.get(session_id)))
            return summaries

    async def list_by_query(
        self,
        query: SessionRecordQuery,
    ) -> list[Session]:
        async with self.queries.read_snapshot() as queries:
            conv_records = await queries.list_sessions(query)
            return await self._hydrate_sessions(conv_records, queries=queries)

    async def get_many(self, session_ids: list[str]) -> list[Session]:
        if not session_ids:
            return []
        async with self.queries.read_snapshot() as queries:
            records = await queries.get_sessions_batch(session_ids)
            return await self._hydrate_sessions(records, ordered_ids=session_ids, queries=queries)

    async def iter_messages(
        self,
        session_id: str,
        *,
        message_roles: MessageRoleFilter = (),
        material_origin: MaterialOriginFilter | None = None,
        limit: int | None = None,
    ) -> AsyncGenerator[Message, None]:
        conv_record = await self.queries.get_session(session_id)
        origin = conv_record.origin if conv_record else None
        async with aclosing(
            self.queries.iter_messages(
                session_id,
                message_roles=message_roles,
                material_origin=material_origin,
                limit=limit,
            )
        ) as records:
            async for record in records:
                yield message_from_record(record, record.attachments, origin=origin, blob_store=self._read_blob_store)

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

        async with self._backend.read_connection() as conn:
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
