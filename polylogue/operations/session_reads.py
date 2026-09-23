"""Bounded session owner operations shared by adapters and direct consumers."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

from pydantic import BaseModel

from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryContinuationInvalidError,
    QueryTransaction,
    QueryTransactionRequest,
    archive_snapshot_epoch,
    validate_continuation_epoch,
)
from polylogue.operations.session_contracts import (
    Coverage,
    RawContent,
    RawList,
    RawMemorySearch,
    RawObservation,
    RawOrigin,
    RawPage,
    RawRead,
    RawSearch,
    RawSourceCoverage,
    RawTimeline,
    ResumeContext,
    SessionList,
    SessionOperation,
    SessionOrchestration,
    SessionPage,
    SessionRead,
    SessionSearch,
    SessionTimeline,
    TimelineEvent,
)

PagedRequest = TypeVar("PagedRequest", SessionList, SessionSearch, SessionRead, SessionTimeline)


def _transaction(request: PagedRequest) -> tuple[PagedRequest, QueryTransactionRequest]:
    continuation = request.continuation
    if continuation:
        decoded = QueryContinuation.decode(continuation)
        tx = decoded.request
        if (
            tx.operation != request.operation
            or tx.projection != "session-owner-v1"
            or decoded.result_ref != tx.result_ref
        ):
            raise QueryContinuationInvalidError("continuation belongs to another session operation")
        arguments = {key: value for key, value in tx.arguments.items() if key != "resolved_dates"}
        original = type(request).model_validate({**arguments, "limit": tx.page_size, "offset": tx.offset})
        supplied = request.model_dump(mode="json", exclude_unset=True, exclude={"continuation", "operation"})
        for name, value in supplied.items():
            if value != original.model_dump(mode="json")[name]:
                raise QueryContinuationInvalidError(f"continuation conflicts with {name}")
        return original, tx
    arguments = request.model_dump(mode="json", exclude={"continuation", "limit", "offset"})
    return request, QueryTransactionRequest(
        operation=request.operation,
        arguments=arguments,
        page_size=request.limit,
        offset=request.offset,
        projection="session-owner-v1",
        stable_order=getattr(request, "sort", None) or "date,identity",
    )


def _frame(archive: Any, tx: QueryTransactionRequest) -> QueryTransactionRequest:
    epoch = validate_continuation_epoch(tx, archive=archive) if tx.archive_epoch else archive_snapshot_epoch(archive)
    return tx.with_archive_epoch(epoch)


def _page(
    items: list[Any],
    total: int,
    tx: QueryTransactionRequest,
    *,
    gaps: list[str] | None = None,
    time_basis: Literal["event-timestamp", "session-file-mtime", "none"] = "none",
) -> SessionPage[Any]:
    next_offset = tx.offset + len(items) if tx.offset + len(items) < total else None
    continuation = (
        QueryContinuation(tx.next(offset=next_offset), tx.result_ref).encode() if next_offset is not None else None
    )
    gaps = gaps or []
    return SessionPage(
        items=items,
        total=total,
        limit=tx.page_size,
        offset=tx.offset,
        next_offset=next_offset,
        continuation=continuation,
        coverage=Coverage(
            authority="indexed-archive", complete=next_offset is None and not gaps, gaps=gaps, time_basis=time_basis
        ),
        outcome="degraded" if gaps else "ok" if items else "empty",
    )


async def session_query(archive_root: Path, request: SessionList | SessionSearch) -> SessionPage[Any]:
    from polylogue.archive.hydration import archive_summary_to_domain
    from polylogue.archive.query.expression import compile_expression_into
    from polylogue.archive.query.filter_kwargs import plan_filter_kwargs
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.surfaces.payloads import session_summary_envelope_from_summary

    request, tx = _transaction(request)
    if isinstance(request, SessionSearch) and not request.expression:
        raise ValueError("sessions.search requires expression or continuation")
    params = request.model_dump(mode="json", exclude={"operation", "expression", "continuation"})
    spec = SessionQuerySpec.from_params(params, strict=True)
    if request.expression:
        spec = compile_expression_into(request.expression, spec)
    plan = spec.to_plan()
    if plan.has_post_filters() or plan.similar_text or plan.similar_session_id or plan.retrieval_lane == "hybrid":
        raise ValueError(
            "session owner pages support lexical search and SQL session filters; use semantic query operations for this expression"
        )
    if "resolved_dates" in tx.arguments:
        dates = tx.arguments["resolved_dates"]
        if not isinstance(dates, dict) or set(dates) != {"since", "until"}:
            raise QueryContinuationInvalidError("continuation has invalid resolved date bounds")
        plan = replace(
            plan,
            since=datetime.fromisoformat(dates["since"]) if dates["since"] else None,
            until=datetime.fromisoformat(dates["until"]) if dates["until"] else None,
        )
    else:
        tx = replace(
            tx,
            arguments={
                **tx.arguments,
                "resolved_dates": {
                    "since": plan.since.isoformat() if plan.since else None,
                    "until": plan.until.isoformat() if plan.until else None,
                },
            },
        )
    filters = plan_filter_kwargs(plan)
    text = " ".join((*spec.query_terms, *spec.contains_terms)).strip()

    if isinstance(request, SessionSearch) and not text:
        raise ValueError("sessions.search requires lexical terms")

    def read(archive: Any) -> SessionPage[Any]:
        framed = _frame(archive, tx)
        if text:
            from polylogue.surfaces.payloads import (
                SessionSearchHitPayload,
                SessionSearchMatchPayload,
                TargetRefPayload,
                reader_anchor,
                reader_message_actions,
            )

            total = archive.count_search_sessions(text, **filters)
            distinct: dict[str, Any] = {}
            raw_offset = 0
            while len(distinct) < min(total, request.offset + request.limit):
                hits = archive.search_summaries(
                    text, limit=250, offset=raw_offset, sort=spec.sort, reverse=spec.reverse, **filters
                )
                if not hits:
                    break
                for hit in hits:
                    distinct.setdefault(hit.session_id, hit)
                raw_offset += len(hits)
            selected = list(distinct.values())[request.offset : request.offset + request.limit]
            items = []
            for hit in selected:
                summary = archive.read_summary(hit.session_id)
                items.append(
                    SessionSearchHitPayload(
                        session=session_summary_envelope_from_summary(
                            archive_summary_to_domain(summary), message_count=summary.message_count
                        ),
                        match=SessionSearchMatchPayload(
                            rank=hit.rank,
                            retrieval_lane="dialogue",
                            match_surface="message",
                            target_ref=TargetRefPayload.message(session_id=hit.session_id, message_id=hit.message_id),
                            anchor=reader_anchor("message", hit.message_id),
                            actions=reader_message_actions(),
                            message_id=hit.message_id,
                            snippet=hit.snippet,
                            score=None,
                            score_kind=None,
                        ),
                    )
                )
            return _page(
                items if isinstance(request, SessionSearch) else [item.session for item in items], total, framed
            )

        else:
            summaries = archive.list_summaries(
                limit=request.limit, offset=request.offset, sort=spec.sort, reverse=spec.reverse, **filters
            )
            total = archive.count_sessions(**filters)
        items = [
            session_summary_envelope_from_summary(
                archive_summary_to_domain(summary), message_count=summary.message_count
            )
            for summary in summaries
        ]
        return _page(items, total, framed)

    result: SessionPage[Any] = await QueryTransaction(archive_root, tx).run(read)
    return result


def _time_ms(value: str | None) -> int | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timeline timestamps must include a timezone")
    return int(parsed.timestamp() * 1000)


async def session_timeline(archive_root: Path, request: SessionTimeline) -> SessionPage[TimelineEvent]:
    request, tx = _transaction(request)
    since, until = _time_ms(request.since), _time_ms(request.until)
    if since is not None and until is not None and since > until:
        raise ValueError("since must not be after until")

    def read(archive: Any) -> SessionPage[TimelineEvent]:
        framed = _frame(archive, tx)
        scope = """WITH events AS (
            SELECT 'message:' || m.message_id AS reference, m.session_id, s.origin,
                   'message' AS kind, m.role AS event_type, m.occurred_at_ms AS timestamp_ms,
                   NULL AS event_id, m.message_id,
                   COALESCE((SELECT group_concat(b.text, char(10)) FROM blocks b WHERE b.message_id=m.message_id), '') AS text
            FROM messages m JOIN sessions s ON s.session_id=m.session_id
            UNION ALL
            SELECT 'session:' || e.session_id, e.session_id, s.origin, 'session-event',
                   e.event_type, e.occurred_at_ms, e.event_id, e.source_message_id,
                   -- polylogue-kc8eq retired the stored ``summary`` column as a
                   -- write-time render of the payload. This is that render,
                   -- moved to the read path: same COALESCE order, same '' floor,
                   -- so the timeline text for a session-event is unchanged.
                   COALESCE(json_extract(e.payload_json, '$.summary'),
                            json_extract(e.payload_json, '$.text'), '')
            FROM session_events e JOIN sessions s ON s.session_id=e.session_id
        ) """
        where = "WHERE (? IS NULL OR origin=?) AND (? IS NULL OR instr(lower(text), lower(?))>0)"
        origin = request.origin.value if request.origin is not None else None
        params: tuple[object, ...] = (origin, origin, request.expression, request.expression)
        unknown = archive._conn.execute(
            scope + "SELECT count(*) FROM events " + where + " AND timestamp_ms IS NULL", params
        ).fetchone()[0]
        where += " AND timestamp_ms IS NOT NULL AND (? IS NULL OR timestamp_ms>=?) AND (? IS NULL OR timestamp_ms<=?)"
        params += (since, since, until, until)
        total = archive._conn.execute(scope + "SELECT count(*) FROM events " + where, params).fetchone()[0]
        rows = archive._conn.execute(
            scope
            + "SELECT reference, session_id, origin, kind, event_type, timestamp_ms, event_id, message_id, substr(text, 1, 2000) AS text FROM events "
            + where
            + " ORDER BY timestamp_ms DESC, reference ASC, event_id ASC LIMIT ? OFFSET ?",
            (*params, request.limit, request.offset),
        ).fetchall()
        return _page(
            [TimelineEvent(**dict(row)) for row in rows],
            total,
            framed,
            gaps=[f"{unknown} matching events have no event timestamp and cannot be placed in time"] if unknown else [],
            time_basis="event-timestamp",
        )

    result: SessionPage[Any] = await QueryTransaction(archive_root, tx).run(read)
    return result


def _raw_origin(provider: str) -> RawOrigin:
    origins: dict[str, RawOrigin] = {"codex": "codex-session", "claude-code": "claude-code-session"}
    return origins[provider]


def _provider(origin: str) -> str:
    return {"codex-session": "codex", "claude-code-session": "claude-code"}[origin]


@overload
def raw_operation(request: RawRead, *, sources: Any = None, max_result_bytes: int = 256_000) -> RawContent: ...


@overload
def raw_operation(
    request: RawList | RawSearch | RawTimeline | RawMemorySearch,
    *,
    sources: Any = None,
    max_result_bytes: int = 256_000,
) -> RawPage: ...


def raw_operation(
    request: RawList | RawSearch | RawRead | RawTimeline | RawMemorySearch,
    *,
    sources: Any = None,
    max_result_bytes: int = 256_000,
) -> RawPage | RawContent:
    from polylogue.operations.raw_sessions.memory import MemoryService
    from polylogue.operations.raw_sessions.sessions import SessionLogService
    from polylogue.operations.raw_sessions.timeline import TimelineService

    service = SessionLogService(sources=sources, max_result_bytes=max_result_bytes)
    # Tokens validate an observation and scope, never grant filesystem authority.
    # Roots are exclusively owner configuration, never supplied in a request.
    key = hashlib.sha256(repr(service.sources).encode()).digest()
    if isinstance(request, RawRead):
        _, path = service._path_from_reference(request.reference)
        before = path.stat()
        result = service.read(request.reference, request.offset, request.max_bytes)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            from polylogue.operations.raw_sessions.sessions import SessionError

            raise SessionError("session source changed during read")
        return RawContent(
            reference=result["reference"],
            origin=_raw_origin(result["provider"]),
            mtime_ns=before.st_mtime_ns,
            offset=result["offset"],
            bytes=result["bytes"],
            next_offset=result["next_offset"],
            content=result["content"],
            coverage=Coverage(
                authority="original-local-session-jsonl",
                complete=not result["truncated"],
                time_basis="none",
                scanned_bytes=result["bytes"],
            ),
            outcome="ok" if result["bytes"] else "empty",
        )
    if isinstance(request, RawMemorySearch):
        cursors = {_provider(k): v for k, v in request.source_cursors.items()} if request.source_cursors else None
        result = MemoryService(service).search(
            request.query,
            [_provider(o) for o in request.origins],
            request.limit,
            source_cursors=cursors,
            cursor_key=key,
            scan_bytes=request.scan_bytes,
        )
        rows = result["matches"]
    elif isinstance(request, RawTimeline):
        result = TimelineService(service).query(
            request.since,
            request.until,
            request.query,
            [_provider(o) for o in request.origins],
            request.limit,
            cursor=request.continuation,
            cursor_key=key,
            scan_bytes=request.scan_bytes,
        )
        rows = result.get("entries", [])
    else:
        provider = _provider(request.origin)
        if isinstance(request, RawSearch):
            result = service.search(
                provider,
                request.query,
                request.limit,
                reference=request.reference,
                cursor=request.continuation,
                cursor_key=key,
                scan_bytes=request.scan_bytes,
            )
            rows = result["matches"]
        else:
            result = service.timeline(
                provider, None, None, None, request.limit, cursor=request.continuation, cursor_key=key
            )
            rows = result["entries"]
        result["sources"] = [
            {
                "source": provider,
                "availability": "available",
                "coverage": {"scanned_bytes": result["scanned_bytes"], "truncated": result["truncated"]},
            }
        ]
    observations = []
    for row in rows:
        reference = row.get("reference") or row["object_reference"]
        source, path = service._path_from_reference(reference)
        info = path.stat()
        expected = row["source_observation"]
        if tuple(expected) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
            from polylogue.operations.raw_sessions.sessions import SessionError

            raise SessionError("session source changed between scan and emission")
        observations.append(
            RawObservation(
                reference=reference,
                origin=_raw_origin(source.provider),
                mtime_ns=info.st_mtime_ns,
                bytes=info.st_size,
                line=row.get("line"),
                offset=row.get("offset"),
                text=row.get("text", row.get("snippet")),
            )
        )
    sources_out = [
        RawSourceCoverage(
            origin=_raw_origin(row["source"]),
            availability=row["availability"],
            reason=row.get("reason"),
            scanned_bytes=row.get("coverage", {}).get("scanned_bytes", 0),
            truncated=row.get("coverage", {}).get("truncated", False),
        )
        for row in result.get("sources", [])
    ]
    gaps = [source.reason or "source unavailable" for source in sources_out if source.availability == "unavailable"]
    if result.get("available") is False:
        gaps.append(result["reason"])
    return RawPage(
        items=observations,
        sources=sources_out,
        coverage=Coverage(
            authority="original-local-session-jsonl",
            complete=not result.get("truncated", False) and not gaps,
            time_basis="session-file-mtime" if isinstance(request, (RawList, RawTimeline)) else "none",
            scanned_bytes=sum(source.scanned_bytes for source in sources_out),
            gaps=gaps,
        ),
        continuation=result.get("next_cursor"),
        source_cursors={_raw_origin(k): v for k, v in result["next_cursors"].items()}
        if result.get("next_cursors")
        else None,
        outcome="degraded"
        if gaps or (not observations and result.get("truncated", False))
        else "ok"
        if observations
        else "empty",
    )


@overload
async def execute_session_operation(
    api: Any, request: SessionList | SessionRead | SessionTimeline, *, raw_sources: Any = None
) -> SessionPage[Any]: ...


@overload
async def execute_session_operation(api: Any, request: SessionOperation, *, raw_sources: Any = None) -> BaseModel: ...


async def execute_session_operation(api: Any, request: SessionOperation, *, raw_sources: Any = None) -> BaseModel:
    if isinstance(request, (RawList, RawSearch, RawRead, RawTimeline, RawMemorySearch)):
        import asyncio

        if isinstance(request, RawRead):
            return await asyncio.to_thread(raw_operation, request, sources=raw_sources)
        return await asyncio.to_thread(raw_operation, request, sources=raw_sources)
    if isinstance(request, (SessionList, SessionSearch)):
        return await session_query(api.archive_root, request)
    if isinstance(request, SessionTimeline):
        return await session_timeline(api.archive_root, request)
    if isinstance(request, SessionOrchestration):
        from polylogue.analysis.orchestration_evidence import SessionOrchestrationEvidence

        evidence: SessionOrchestrationEvidence | None = await api.get_session_orchestration(
            request.ref.removeprefix("session:")
        )
        if evidence is None:
            raise ValueError("session not found")
        return evidence
    if isinstance(request, SessionRead):
        from polylogue.operations.transcript_window import message_transcript_window
        from polylogue.surfaces.payloads import message_row_envelope_from_domain

        # The transcript window has exactly one execution route
        # (polylogue-ijbwq): this operation owns the *projection* onto the
        # session-owner page, not the window arithmetic, snapshot binding or
        # continuation token, which every public surface now shares.
        session_id = request.ref.removeprefix("session:")
        window = await message_transcript_window(api, request)
        return _page(
            [message_row_envelope_from_domain(message, session_id=session_id) for message in window.rows],
            window.total,
            window.transaction,
            gaps=window.gaps,
        )
    assert isinstance(request, ResumeContext)
    from polylogue.context.preamble import build_context_preamble_payload
    from polylogue.surfaces.payloads import ContextPreamble

    result = await build_context_preamble_payload(
        api,
        session_id=request.session_id or "",
        repo_path=request.repo_path,
        cwd=request.cwd,
        recent_files=tuple(request.recent_files),
        related_limit=request.related_limit,
        require_session=False,
        source_tool_calls={"context": "polylogue-session-owner"},
    )
    return result or ContextPreamble(preamble_version="1.0", source_tool_calls={"context": "polylogue-session-owner"})


async def session_operation_response(api: Any, request: SessionOperation, *, raw_sources: Any = None) -> BaseModel:
    """Serialize operation failures without changing the declared result shape."""
    from polylogue.operations.session_contracts import SessionOperationError

    try:
        return await execute_session_operation(api, request, raw_sources=raw_sources)
    except Exception as exc:
        return SessionOperationError(code=str(getattr(exc, "code", None) or "session_read_failed"), message=str(exc))
