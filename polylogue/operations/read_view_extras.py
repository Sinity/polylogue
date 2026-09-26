"""Product readers for the effective-context, neighbor, and correlation views."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING

from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.storage.query_models import SessionRecordQuery


def _int_field(payload: Mapping[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_field(payload: Mapping[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _index_connection(archive: ArchiveStore) -> sqlite3.Connection:
    connection = archive.index_connection
    if connection is None:
        raise ValueError("read view requires an index snapshot")
    return connection


def execute_effective_context_read(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Replay the session's own messages across its latest compaction boundary."""

    from polylogue.storage.sqlite.queries.mappers_archive import bind_message_row_mapper
    from polylogue.storage.sqlite.queries.message_query_reads import _MESSAGE_RECORD_SELECT, _TRANSCRIPT_ORDER

    session_id = archive.resolve_session_id(str(payload["session_id"]))
    connection = _index_connection(archive)
    cursor = connection.execute(
        f"SELECT {_MESSAGE_RECORD_SELECT} FROM messages m JOIN sessions s ON s.session_id = m.session_id "
        f"WHERE m.session_id = ? ORDER BY {_TRANSCRIPT_ORDER}",
        (session_id,),
    )
    decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
    messages = [decode(row) for row in cursor.fetchall()]
    requested_position = payload.get("at_position")
    at_position = (
        _int_field(payload, "at_position", -1)
        if requested_position is not None
        else max((message.position for message in messages), default=-1)
    )
    boundary = connection.execute(
        """SELECT boundary_end_position, boundary_message_id FROM session_events
        WHERE session_id = ? AND event_type = 'compaction'
          AND boundary_start_position IS NOT NULL
          AND boundary_end_position IS NOT NULL
          AND boundary_message_id IS NOT NULL
          AND boundary_end_position < ?
        ORDER BY boundary_end_position DESC, position DESC LIMIT 1""",
        (session_id, at_position),
    ).fetchone()
    visible = [message for message in messages if message.position <= at_position]
    if boundary is not None:
        summary = next((message for message in messages if str(message.message_id) == str(boundary[1])), None)
        if summary is not None:
            end_position = int(boundary[0])
            visible = [summary] + [
                message
                for message in messages
                if end_position < message.position <= at_position and message is not summary
            ]
    return {
        "view": "effective_context",
        "payload": {
            "session_id": session_id,
            "at_position": requested_position,
            "messages": [message.model_dump(mode="json", exclude_none=True) for message in visible],
        },
    }


def execute_neighbor_read(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Run the shared neighbor selector on the pinned archive snapshot."""

    from polylogue.archive.session.neighbor_candidates import NeighborDiscoveryRequest, discover_neighbor_candidates
    from polylogue.surfaces.payloads import SessionNeighborCandidatePayload, model_json_document

    request = NeighborDiscoveryRequest(
        session_id=str(payload["session_id"]) if payload.get("session_id") else None,
        query=str(payload["query"]) if payload.get("query") else None,
        origin=str(payload["origin"]) if payload.get("origin") else None,
        limit=_int_field(payload, "limit", 10),
        window_hours=_int_field(payload, "window_hours", 24),
    )
    candidates = asyncio.run(discover_neighbor_candidates(_NeighborSnapshot(archive), request))
    return {
        "view": "neighbors",
        "payload": {
            "neighbors": [
                model_json_document(SessionNeighborCandidatePayload.from_candidate(candidate), exclude_none=True)
                for candidate in candidates
            ]
        },
    }


class _NeighborSnapshot:
    """NeighborStore adapter over the operation's already-pinned archive."""

    def __init__(self, archive: ArchiveStore) -> None:
        self.archive = archive

    async def resolve_id(self, id_prefix: str, *, strict: bool = False) -> SessionId | None:
        del strict
        try:
            return SessionId(self.archive.resolve_session_id(id_prefix))
        except KeyError:
            return None

    async def get(self, session_id: str) -> Session | None:
        try:
            resolved = self.archive.resolve_session_id(session_id)
            summary = self.archive.read_summary(resolved)
            return archive_envelope_to_session(
                self.archive.read_session(resolved),
                display_label=summary.display_label,
                display_label_source=summary.display_label_source,
            )
        except KeyError:
            return None

    async def list_summaries_by_query(self, query: SessionRecordQuery) -> list[SessionSummary]:
        from polylogue.archive.message.types import validate_message_type_filter
        from polylogue.archive.query.spec import parse_query_date

        def date_ms(field: str, value: str | None) -> int | None:
            parsed = parse_query_date(field, value)
            return int(parsed.timestamp() * 1000) if parsed is not None else None

        origin = Origin(query.origin).value if query.origin is not None else None
        origins = tuple(Origin(value).value for value in (query.origins or ()))
        return [
            archive_summary_to_domain(summary)
            for summary in self.archive.list_summaries(
                limit=query.limit or 50,
                offset=query.offset or 0,
                origin=origin,
                origins=origins,
                referenced_paths=tuple(query.referenced_path or ()),
                cwd_prefix=query.cwd_prefix,
                action_terms=tuple(query.action_terms or ()),
                excluded_action_terms=tuple(query.excluded_action_terms or ()),
                tool_terms=tuple(query.tool_terms or ()),
                excluded_tool_terms=tuple(query.excluded_tool_terms or ()),
                has_tool_use=query.has_tool_use or False,
                has_thinking=query.has_thinking or False,
                message_type=validate_message_type_filter(query.message_type).value if query.message_type else None,
                title=query.title_contains,
                min_messages=query.min_messages,
                max_messages=query.max_messages,
                min_words=query.min_words,
                max_words=query.max_words,
                since_ms=date_ms("since", query.since),
                until_ms=date_ms("until", query.until),
            )
        ]

    async def search_summary_hits(
        self, query: str, limit: int = 20, origins: list[str] | None = None, since: str | None = None
    ) -> list[SessionSearchHit]:
        from polylogue.archive.query.search_hits import session_search_hit_from_summary
        from polylogue.archive.query.spec import parse_query_date

        parsed = parse_query_date("since", since)
        hits = self.archive.search_summaries(
            query,
            limit=limit,
            origins=tuple(Origin(value).value for value in (origins or ())),
            since_ms=int(parsed.timestamp() * 1000) if parsed is not None else None,
        )
        result: list[SessionSearchHit] = []
        for hit in hits:
            try:
                summary = archive_summary_to_domain(self.archive.read_summary(hit.session_id))
            except KeyError:
                continue
            result.append(
                session_search_hit_from_summary(
                    summary,
                    rank=hit.rank,
                    retrieval_lane="dialogue",
                    match_surface="message",
                    message_id=hit.message_id,
                    snippet=hit.snippet,
                    score=None,
                )
            )
        return result


def execute_correlation_read(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Build correlation evidence from one pinned session and its typed refs."""

    from polylogue.analysis.session_commit import (
        bridge_session_ids_from_events,
        build_correlation_result,
        correlation_result_to_payload,
        typed_refs_from_session_refs,
    )
    from polylogue.archive.hydration import archive_envelope_to_session
    from polylogue.storage.sqlite.queries.session_events import sync_session_events_batch

    session_id = archive.resolve_session_id(str(payload["session_id"]))
    summary = archive.read_summary(session_id)
    session = archive_envelope_to_session(
        archive.read_session(session_id),
        display_label=summary.display_label,
        display_label_source=summary.display_label_source,
    )
    if session.created_at is None or session.updated_at is None:
        raise ValueError("Session has no timestamp data.")
    repo = (
        payload.get("repo_path")
        or session.git_repository_url
        or (str(session.working_directories[0]) if session.working_directories else ".")
    )
    connection = _index_connection(archive)
    refs = [
        SimpleNamespace(kind=row["kind"], repo=row["repo"], number=row["ref_number"], url=row["url"])
        for row in connection.execute(
            "SELECT kind, repo, ref_number, url FROM session_refs WHERE session_id = ? ORDER BY position",
            (session_id,),
        )
    ]
    typed_pr_refs, typed_issue_refs = typed_refs_from_session_refs(refs)
    events = sync_session_events_batch(connection, [session_id]).get(session_id, [])
    messages = [
        {"id": message.id, "role": message.role.value, "text": message.text, "content_blocks": list(message.blocks)}
        for message in session.messages
    ]
    result = build_correlation_result(
        session_id=session_id,
        messages=messages,
        session_created_at=session.created_at,
        session_updated_at=session.updated_at,
        repo_path=str(repo),
        before_hours=_int_field(payload, "since_hours", 2),
        after_hours=_int_field(payload, "since_hours", 2),
        confidence_threshold=_float_field(payload, "confidence_threshold", 0.3),
        typed_pr_refs=typed_pr_refs,
        typed_issue_refs=typed_issue_refs,
        bridge_session_ids=bridge_session_ids_from_events(events),
    )
    document = correlation_result_to_payload(result)
    document["checkout_commits"] = [
        {
            "commit_sha": row["commit_sha"],
            "short_sha": str(row["commit_sha"])[:8],
            "repo_id": row["repo_id"],
            "detection_type": row["detection_type"],
            "method": row["method"],
            "confidence": row["confidence"],
            "evidence": json.loads(row["evidence_json"]) if row["evidence_json"] else None,
        }
        for row in connection.execute(
            "SELECT commit_sha, repo_id, detection_type, method, confidence, evidence_json "
            "FROM session_commits WHERE session_id = ? ORDER BY created_at_ms",
            (session_id,),
        )
    ]
    return {"view": "correlation", "payload": document}
