"""Multi-session context-pack read-view implementation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.context.assertion_claims import context_claim_text
from polylogue.mcp.context_pack import (
    ContextPackDateRange,
    ContextPackDecisions,
    ContextPackIntent,
    ContextPackMessage,
    ContextPackOmission,
    ContextPackPayload,
    ContextPackProvenance,
    ContextPackQueryContext,
    ContextPackScope,
    ContextPackSession,
    ContextPackSizeEstimate,
    _build_project_context,
    _summarize_actions,
    archive_context_pack_active,
    query_archive_context_pack,
    select_context_pack_sessions,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import serialize_surface_payload

if TYPE_CHECKING:
    from polylogue.cli.shared.types import AppEnv

_DEFAULT_MAX_SESSIONS = 5
_DEFAULT_MAX_MESSAGES = 20


def _clamp_context_pack_limit(value: int | object) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, min(value, 20))
    if isinstance(value, str | bytes | bytearray):
        return max(1, min(int(value), 20))
    return 1


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate context-pack message text for compact cross-surface payloads."""
    if max_length > 0 and len(text) > max_length:
        return text[:max_length] + "..."
    return text


def _date_text(value: object) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _context_pack_redaction_policy(redact_paths: bool) -> str:
    if redact_paths:
        return "public_refs_and_redacted_paths"
    return "raw_paths_explicit_opt_in"


def _context_pack_path_value(value: str | None, *, redact_paths: bool) -> str | None:
    if value is None or not redact_paths or not value.startswith("/"):
        return value
    name = Path(value).name or "path"
    return f"<redacted-path>/{name}"


def _context_pack_provenance(
    *,
    archive_root: Path,
    active_db_path: Path,
    redact_paths: bool,
) -> ContextPackProvenance:
    return ContextPackProvenance(
        generated_at=datetime.now(UTC).isoformat(),
        redacted=redact_paths,
        archive_runtime="archive_file_set",
        archive_root=None if redact_paths else str(archive_root),
        active_db_path=None if redact_paths else str(active_db_path),
        redaction_policy=_context_pack_redaction_policy(redact_paths),
    )


def _context_pack_scope(
    *,
    seed_session_id: str | None,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    origin: str | None,
    query: str | None,
    conv_limit: int,
    msg_limit: int,
    max_text: int,
    include_messages: bool,
    redact_paths: bool,
) -> ContextPackScope:
    return ContextPackScope(
        seed_refs=([f"session:{seed_session_id}"] if seed_session_id else []),
        read_views=["context-pack"],
        project_path=_context_pack_path_value(project_path, redact_paths=redact_paths),
        project_repo=project_repo,
        origin=origin,
        since=since,
        until=until,
        query=query,
        include_messages=include_messages,
        limits={"sessions": conv_limit, "messages_per_session": msg_limit, "message_text_chars": max_text},
    )


def _context_pack_evidence_refs(pack_sessions: list[ContextPackSession]) -> list[str]:
    return [session.session_id for session in pack_sessions]


def _context_pack_message_text_bytes(pack_sessions: list[ContextPackSession]) -> int:
    return sum(len(message.text.encode("utf-8")) for session in pack_sessions for message in session.messages)


def _context_pack_token_estimate(
    *,
    pack_sessions: list[ContextPackSession],
    decisions: list[str],
    omissions: list[ContextPackOmission],
) -> int:
    texts: list[str] = [*decisions]
    texts.extend(session.title or "" for session in pack_sessions)
    texts.extend(message.text for session in pack_sessions for message in session.messages)
    texts.extend(omission.detail for omission in omissions)
    word_count = sum(len(text.split()) for text in texts if text)
    return max(1, word_count) if texts or pack_sessions or omissions else 0


def _context_pack_omissions(
    *,
    pack_sessions: list[ContextPackSession],
    seed_session_id: str | None,
    query: str | None,
    include_messages: bool,
    msg_limit: int,
    max_text: int,
    redact_paths: bool,
) -> list[ContextPackOmission]:
    omissions: list[ContextPackOmission] = []
    if seed_session_id is not None and not pack_sessions:
        omissions.append(
            ContextPackOmission(
                ref=f"session:{seed_session_id}",
                view="context-pack",
                reason="not_found",
                detail="Requested seed session was not found in the archive index.",
            )
        )
    elif query and not pack_sessions:
        omissions.append(
            ContextPackOmission(
                query=query,
                view="context-pack",
                reason="not_found",
                detail="No sessions matched the strict or recall context-pack selection attempts.",
            )
        )
    if not include_messages:
        omissions.append(
            ContextPackOmission(
                view="messages",
                reason="policy",
                detail="Message bodies were excluded by the caller; session-level metadata remains selected.",
                evidence_refs=[session.session_id for session in pack_sessions],
            )
        )
    for session in pack_sessions:
        if len(session.messages) < session.message_count and include_messages:
            omissions.append(
                ContextPackOmission(
                    ref=f"session:{session.session_id}",
                    view="messages",
                    reason="budget",
                    detail=(
                        f"Included {len(session.messages)} message snippets out of {session.message_count}; "
                        f"limit={msg_limit}, max_text={max_text}."
                    ),
                    evidence_refs=[session.session_id],
                )
            )
        if include_messages and session.message_count and not session.messages:
            omissions.append(
                ContextPackOmission(
                    ref=f"session:{session.session_id}",
                    view="messages",
                    reason="not_found",
                    detail="Session was selected, but message text was unavailable to this read view.",
                    evidence_refs=[session.session_id],
                )
            )
    if redact_paths:
        omissions.append(
            ContextPackOmission(
                view="provenance.paths",
                reason="redacted",
                detail="Absolute archive and active database paths are omitted unless the caller opts into raw paths.",
                evidence_refs=[session.session_id for session in pack_sessions],
            )
        )
    return omissions


def _context_pack_caveats(
    *,
    pack_sessions: list[ContextPackSession],
    relaxed_filters: list[str],
    omissions: list[ContextPackOmission],
) -> list[str]:
    caveats: list[str] = []
    if not pack_sessions:
        caveats.append("no_sessions_selected")
    if relaxed_filters:
        caveats.append("selection_relaxed:" + ",".join(relaxed_filters))
    for omission in omissions:
        caveat = f"omitted:{omission.view or omission.reason}:{omission.reason}"
        if caveat not in caveats:
            caveats.append(caveat)
    return caveats


def _finalize_context_pack_payload(payload: ContextPackPayload) -> ContextPackPayload:
    json_bytes = len(payload.model_dump_json(exclude_none=True).encode("utf-8"))
    return payload.model_copy(
        update={"size_estimate": payload.size_estimate.model_copy(update={"json_bytes": json_bytes})}
    )


async def build_archive_context_pack_payload(
    *,
    archive_root: Path,
    polylogue: Any,
    clamp_limit: Callable[[int | object], int] = _clamp_context_pack_limit,
    project_path: str | None = None,
    project_repo: str | None = None,
    since: str | None = None,
    until: str | None = None,
    origin: str | None = None,
    query: str | None = None,
    conv_limit: int = _DEFAULT_MAX_SESSIONS,
    msg_limit: int = _DEFAULT_MAX_MESSAGES,
    max_text: int = 200,
    include_messages: bool = True,
    redact_paths: bool = True,
    seed_session_id: str | None = None,
) -> ContextPackPayload:
    """Build the shared archive-backed context-pack payload.

    MCP, daemon HTTP, and future API consumers use this helper instead of
    maintaining separate context-pack serializers.
    """

    with ArchiveStore.open_existing(archive_root) as archive:

        async def query_sessions(spec: SessionQuerySpec) -> list[Any]:
            return query_archive_context_pack(archive, spec, default_limit=_DEFAULT_MAX_SESSIONS)

        envelope_by_id: dict[str, Any] = {}
        if seed_session_id is not None:
            try:
                seed_envelope = archive.read_session(seed_session_id)
            except Exception:
                seed_envelope = None
            if seed_envelope is None:
                sessions = []
            else:
                envelope_by_id[seed_session_id] = seed_envelope
                sessions = [
                    SimpleNamespace(
                        id=seed_envelope.session_id,
                        origin=seed_envelope.origin,
                        title=seed_envelope.title,
                        created_at=seed_envelope.created_at,
                        updated_at=seed_envelope.updated_at,
                        message_count=len(seed_envelope.messages),
                    )
                ]
            query_total: int | None = len(sessions)
            match_strategy = "session-id"
            relaxed_filters: list[str] = []
        else:
            selection = await select_context_pack_sessions(
                query_sessions,
                clamp_limit,
                project_path=project_path,
                project_repo=project_repo,
                since=since,
                until=until,
                origin=origin,
                query=query,
                limit=conv_limit,
            )
            sessions = selection.sessions
            query_total = selection.query_total
            match_strategy = selection.match_strategy
            relaxed_filters = list(selection.relaxed_filters)
        dates = [
            d for conv in sessions for d in (_date_text(conv.created_at), _date_text(conv.updated_at)) if d is not None
        ]
        pack_sessions: list[ContextPackSession] = []
        assertion_decisions: list[str] = []

        for conv in sessions[:conv_limit]:
            conv_id = str(conv.id)
            try:
                claims = await polylogue.list_assertion_claim_payloads(
                    target_ref=f"session:{conv_id}",
                    statuses=("active",),
                    context_inject=True,
                    limit=20,
                )
            except Exception:
                claims = []
            assertion_decisions.extend(
                context_claim_text(kind=claim.kind, body_text=claim.body_text, target_ref=claim.target_ref)
                for claim in claims
            )

            messages: list[ContextPackMessage] = []
            if include_messages:
                envelope = envelope_by_id.get(conv_id)
                if envelope is None:
                    try:
                        envelope = archive.read_session(conv_id)
                    except Exception:
                        envelope = None
                if envelope is not None:
                    for message in envelope.messages[:msg_limit]:
                        text = "\n".join(block.text or "" for block in message.blocks if block.text)
                        messages.append(
                            ContextPackMessage(
                                role=message.role,
                                text=_truncate_text(text, max_text),
                                has_tool_use=getattr(message, "has_tool_use", False),
                                has_thinking=getattr(message, "has_thinking", False),
                            )
                        )

            session_origin = str(conv.origin) if getattr(conv, "origin", None) is not None else "unknown"
            pack_sessions.append(
                ContextPackSession(
                    session_id=conv_id,
                    origin=session_origin,
                    title=conv.title,
                    created_at=_date_text(conv.created_at),
                    updated_at=_date_text(conv.updated_at),
                    message_count=int(conv.message_count),
                    tool_use_count=None,
                    messages=messages,
                )
            )

    total_matching = len(sessions)
    omissions = _context_pack_omissions(
        pack_sessions=pack_sessions,
        seed_session_id=seed_session_id,
        query=query,
        include_messages=include_messages,
        msg_limit=msg_limit,
        max_text=max_text,
        redact_paths=redact_paths,
    )
    token_estimate = _context_pack_token_estimate(
        pack_sessions=pack_sessions,
        decisions=assertion_decisions,
        omissions=omissions,
    )
    payload = ContextPackPayload(
        selection_strategy=match_strategy,
        scope=_context_pack_scope(
            seed_session_id=seed_session_id,
            project_path=project_path,
            project_repo=project_repo,
            since=since,
            until=until,
            origin=origin,
            query=query,
            conv_limit=conv_limit,
            msg_limit=msg_limit,
            max_text=max_text,
            include_messages=include_messages,
            redact_paths=redact_paths,
        ),
        omissions=omissions,
        evidence_refs=_context_pack_evidence_refs(pack_sessions),
        caveats=_context_pack_caveats(
            pack_sessions=pack_sessions,
            relaxed_filters=relaxed_filters,
            omissions=omissions,
        ),
        redaction_policy=_context_pack_redaction_policy(redact_paths),
        token_estimate=token_estimate,
        size_estimate=ContextPackSizeEstimate(
            message_text_bytes=_context_pack_message_text_bytes(pack_sessions),
            session_count=len(pack_sessions),
            message_count=sum(len(session.messages) for session in pack_sessions),
            token_estimate=token_estimate,
        ),
        decisions=ContextPackDecisions(items=assertion_decisions),
        project=_build_project_context((), redact=redact_paths),
        date_range=ContextPackDateRange(
            since=since,
            until=until,
            earliest=min(dates) if dates else None,
            latest=max(dates) if dates else None,
            session_count_in_range=total_matching,
        ),
        query_context=ContextPackQueryContext(
            total_matching_sessions=total_matching,
            sessions_included=min(total_matching, conv_limit),
            project_path=project_path,
            project_repo=project_repo,
            origin=origin,
            query=query,
            query_matched=total_matching,
            query_total=query_total if query_total is not None else total_matching,
            match_strategy=match_strategy,
            relaxed_filters=relaxed_filters,
        ),
        sessions=pack_sessions,
        action_summaries=[],
        provenance=_context_pack_provenance(
            archive_root=archive_root,
            active_db_path=archive_root / "index.db",
            redact_paths=redact_paths,
        ),
        total_sessions=total_matching,
        total_messages=sum(int(conv.message_count) for conv in sessions[:conv_limit]),
        total_tool_calls=0,
    )
    return _finalize_context_pack_payload(payload)


def run_context_pack_view(
    env: AppEnv,
    *,
    project_path: str | None = None,
    project_repo: str | None = None,
    since: str | None = None,
    until: str | None = None,
    origin: str | None = None,
    query: str | None = None,
    max_sessions: int = _DEFAULT_MAX_SESSIONS,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
    no_redact: bool = False,
) -> None:
    """Build a provenance-rich multi-session context pack for agent analysis."""
    conv_limit = max(1, min(max_sessions, 20))
    msg_limit = max(1, min(max_messages, 100))

    if _archive_context_pack_active(env):
        payload = _build_archive_context_pack(
            env,
            project_path=project_path,
            project_repo=project_repo,
            since=since,
            until=until,
            origin=origin,
            query=query,
            conv_limit=conv_limit,
            msg_limit=msg_limit,
            redact=not no_redact,
        )
        click.echo(serialize_surface_payload(payload))
        return

    poly = env.polylogue

    selection = run_coroutine_sync(
        select_context_pack_sessions(
            poly.list_sessions_for_spec,
            _clamp_context_pack_limit,
            project_path=project_path,
            project_repo=project_repo,
            since=since,
            until=until,
            origin=origin,
            query=query,
            limit=conv_limit,
        )
    )
    sessions = selection.sessions
    total_matching = len(sessions)
    conv_ids = [str(conv.id) for conv in sessions]

    all_actions: dict[str, tuple[Any, ...]] = {}
    if conv_ids:
        try:
            all_actions = run_coroutine_sync(env.polylogue.get_actions_batch(conv_ids))
        except Exception:
            all_actions = {}

    aggregated_events: list[Any] = []
    for events in all_actions.values():
        aggregated_events.extend(events)

    action_summaries = _summarize_actions(aggregated_events, redact=not no_redact)
    assertion_decisions: list[str] = []
    for conv_id in conv_ids:
        try:
            claims = run_coroutine_sync(
                poly.list_assertion_claim_payloads(
                    target_ref=f"session:{conv_id}",
                    statuses=("active",),
                    context_inject=True,
                    limit=20,
                )
            )
        except Exception:
            claims = []
        assertion_decisions.extend(
            context_claim_text(kind=claim.kind, body_text=claim.body_text, target_ref=claim.target_ref)
            for claim in claims
        )

    dates: list[str] = []
    for conv in sessions:
        if conv.created_at is not None:
            dates.append(str(conv.created_at))
        if conv.updated_at is not None:
            dates.append(str(conv.updated_at))
    earliest = min(dates) if dates else None
    latest = max(dates) if dates else None

    pack_sessions: list[ContextPackSession] = []
    total_msg = 0
    total_tools = 0
    for conv in sessions[:conv_limit]:
        conv_id = str(conv.id)
        total_msg += conv.message_count
        tool_count = getattr(conv, "tool_use_count", 0) or 0
        total_tools += tool_count

        messages: list[ContextPackMessage] = []
        try:
            msg_list, _ = run_coroutine_sync(poly.get_messages_paginated(conv_id, limit=msg_limit, offset=0))
        except Exception:
            msg_list = []
        for m in msg_list:
            messages.append(
                ContextPackMessage(
                    role=m.role.value if m.role else "unknown",
                    text=m.text or "",
                )
            )

        pack_sessions.append(
            ContextPackSession(
                session_id=conv_id,
                title=conv.title,
                origin=conv.origin or "unknown",
                created_at=str(conv.created_at) if conv.created_at is not None else None,
                updated_at=str(conv.updated_at) if conv.updated_at is not None else None,
                message_count=conv.message_count,
                tool_use_count=tool_count if tool_count else None,
                messages=messages,
            )
        )

    redact_paths = not no_redact
    omissions = _context_pack_omissions(
        pack_sessions=pack_sessions,
        seed_session_id=None,
        query=query,
        include_messages=True,
        msg_limit=msg_limit,
        max_text=0,
        redact_paths=redact_paths,
    )
    token_estimate = _context_pack_token_estimate(
        pack_sessions=pack_sessions,
        decisions=assertion_decisions,
        omissions=omissions,
    )
    payload = ContextPackPayload(
        selection_strategy=selection.match_strategy,
        scope=_context_pack_scope(
            seed_session_id=None,
            project_path=project_path,
            project_repo=project_repo,
            since=since,
            until=until,
            origin=origin,
            query=query,
            conv_limit=conv_limit,
            msg_limit=msg_limit,
            max_text=0,
            include_messages=True,
            redact_paths=redact_paths,
        ),
        omissions=omissions,
        evidence_refs=_context_pack_evidence_refs(pack_sessions),
        caveats=_context_pack_caveats(
            pack_sessions=pack_sessions,
            relaxed_filters=list(selection.relaxed_filters),
            omissions=omissions,
        ),
        redaction_policy=_context_pack_redaction_policy(redact_paths),
        token_estimate=token_estimate,
        size_estimate=ContextPackSizeEstimate(
            message_text_bytes=_context_pack_message_text_bytes(pack_sessions),
            session_count=len(pack_sessions),
            message_count=sum(len(session.messages) for session in pack_sessions),
            token_estimate=token_estimate,
        ),
        intent=ContextPackIntent(),
        decisions=ContextPackDecisions(items=assertion_decisions),
        project=_build_project_context(aggregated_events, redact=redact_paths),
        date_range=ContextPackDateRange(
            since=since,
            until=until,
            earliest=earliest,
            latest=latest,
            session_count_in_range=total_matching,
        ),
        query_context=ContextPackQueryContext(
            total_matching_sessions=total_matching,
            sessions_included=min(total_matching, conv_limit),
            project_path=project_path,
            project_repo=project_repo,
            origin=origin,
            query=query,
            query_matched=total_matching,
            query_total=selection.query_total,
            match_strategy=selection.match_strategy,
            relaxed_filters=list(selection.relaxed_filters),
        ),
        sessions=pack_sessions,
        action_summaries=action_summaries,
        provenance=_context_pack_provenance(
            archive_root=env.config.archive_root,
            active_db_path=env.config.db_path,
            redact_paths=redact_paths,
        ),
        total_sessions=total_matching,
        total_messages=total_msg,
        total_tool_calls=total_tools,
    )

    click.echo(serialize_surface_payload(_finalize_context_pack_payload(payload)))


def _archive_context_pack_active(env: AppEnv) -> bool:
    return archive_context_pack_active(
        archive_root=env.config.archive_root,
        db_anchor_path=env.config.db_path,
    )


def _build_archive_context_pack(
    env: AppEnv,
    *,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    origin: str | None,
    query: str | None,
    conv_limit: int,
    msg_limit: int,
    redact: bool,
) -> ContextPackPayload:
    archive_root = env.config.archive_root
    with ArchiveStore.open_existing(archive_root) as archive:

        async def query_sessions(spec: SessionQuerySpec) -> list[SimpleNamespace]:
            return query_archive_context_pack(archive, spec, default_limit=_DEFAULT_MAX_SESSIONS)

        selection = run_coroutine_sync(
            select_context_pack_sessions(
                query_sessions,
                _clamp_context_pack_limit,
                project_path=project_path,
                project_repo=project_repo,
                since=since,
                until=until,
                origin=origin,
                query=query,
                limit=conv_limit,
            )
        )
        sessions = selection.sessions
        total_matching = len(sessions)
        assertion_decisions: list[str] = []
        pack_sessions: list[ContextPackSession] = []
        total_msg = 0
        total_tools = 0
        dates: list[str] = []

        for conv in sessions[:conv_limit]:
            conv_id = str(conv.id)
            total_msg += int(conv.message_count)
            tool_count = int(getattr(conv, "tool_use_count", 0) or 0)
            total_tools += tool_count
            if conv.created_at is not None:
                dates.append(str(conv.created_at))
            if conv.updated_at is not None:
                dates.append(str(conv.updated_at))
            try:
                claims = run_coroutine_sync(
                    env.polylogue.list_assertion_claim_payloads(
                        target_ref=f"session:{conv_id}",
                        statuses=("active",),
                        context_inject=True,
                        limit=20,
                    )
                )
            except Exception:
                claims = []
            assertion_decisions.extend(
                context_claim_text(kind=claim.kind, body_text=claim.body_text, target_ref=claim.target_ref)
                for claim in claims
            )

            messages: list[ContextPackMessage] = []
            try:
                envelope = archive.read_session(conv_id)
            except Exception:
                envelope = None
            if envelope is not None:
                for message in envelope.messages[:msg_limit]:
                    text = "\n".join(block.text or "" for block in message.blocks if block.text)
                    messages.append(ContextPackMessage(role=message.role, text=text))

            pack_sessions.append(
                ContextPackSession(
                    session_id=conv_id,
                    title=conv.title,
                    origin=conv.origin,
                    created_at=str(conv.created_at) if conv.created_at is not None else None,
                    updated_at=str(conv.updated_at) if conv.updated_at is not None else None,
                    message_count=int(conv.message_count),
                    tool_use_count=tool_count if tool_count else None,
                    messages=messages,
                )
            )

    omissions = _context_pack_omissions(
        pack_sessions=pack_sessions,
        seed_session_id=None,
        query=query,
        include_messages=True,
        msg_limit=msg_limit,
        max_text=0,
        redact_paths=redact,
    )
    token_estimate = _context_pack_token_estimate(
        pack_sessions=pack_sessions,
        decisions=assertion_decisions,
        omissions=omissions,
    )
    payload = ContextPackPayload(
        selection_strategy=selection.match_strategy,
        scope=_context_pack_scope(
            seed_session_id=None,
            project_path=project_path,
            project_repo=project_repo,
            since=since,
            until=until,
            origin=origin,
            query=query,
            conv_limit=conv_limit,
            msg_limit=msg_limit,
            max_text=0,
            include_messages=True,
            redact_paths=redact,
        ),
        omissions=omissions,
        evidence_refs=_context_pack_evidence_refs(pack_sessions),
        caveats=_context_pack_caveats(
            pack_sessions=pack_sessions,
            relaxed_filters=list(selection.relaxed_filters),
            omissions=omissions,
        ),
        redaction_policy=_context_pack_redaction_policy(redact),
        token_estimate=token_estimate,
        size_estimate=ContextPackSizeEstimate(
            message_text_bytes=_context_pack_message_text_bytes(pack_sessions),
            session_count=len(pack_sessions),
            message_count=sum(len(session.messages) for session in pack_sessions),
            token_estimate=token_estimate,
        ),
        intent=ContextPackIntent(),
        decisions=ContextPackDecisions(items=assertion_decisions),
        project=_build_project_context((), redact=redact),
        date_range=ContextPackDateRange(
            since=since,
            until=until,
            earliest=min(dates) if dates else None,
            latest=max(dates) if dates else None,
            session_count_in_range=total_matching,
        ),
        query_context=ContextPackQueryContext(
            total_matching_sessions=total_matching,
            sessions_included=min(total_matching, conv_limit),
            project_path=project_path,
            project_repo=project_repo,
            origin=origin,
            query=query,
            query_matched=total_matching,
            query_total=selection.query_total,
            match_strategy=selection.match_strategy,
            relaxed_filters=list(selection.relaxed_filters),
        ),
        sessions=pack_sessions,
        action_summaries=[],
        provenance=_context_pack_provenance(
            archive_root=archive_root,
            active_db_path=archive_root / "index.db",
            redact_paths=redact,
        ),
        total_sessions=total_matching,
        total_messages=total_msg,
        total_tool_calls=total_tools,
    )
    return _finalize_context_pack_payload(payload)
