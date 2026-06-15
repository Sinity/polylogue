"""Multi-session context-pack view — assemble provenance-rich context bundles.

The capability behind ``read --view context-pack`` (#1842): it absorbed the
former standalone ``context-pack`` command. The MCP ``build_context_pack`` tool
exposes the same capability programmatically.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.shared.types import AppEnv
from polylogue.mcp.context_pack import (
    ContextPackDateRange,
    ContextPackDecisions,
    ContextPackIntent,
    ContextPackMessage,
    ContextPackPayload,
    ContextPackProvenance,
    ContextPackQueryContext,
    ContextPackSession,
    _build_project_context,
    _summarize_actions,
    archive_context_pack_active,
    query_archive_context_pack,
    select_context_pack_sessions,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import serialize_surface_payload

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
    """Build a provenance-rich multi-session context pack for agent analysis.

    The capability behind ``read --view context-pack`` (#1842): selects matching
    sessions by project/origin/date/query filters and emits a single provenance
    JSON document to stdout.
    """
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

    payload = ContextPackPayload(
        intent=ContextPackIntent(),
        decisions=ContextPackDecisions(),
        project=_build_project_context(aggregated_events, redact=not no_redact),
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
        provenance=ContextPackProvenance(
            redacted=not no_redact,
            archive_runtime="archive_file_set",
            archive_root=str(env.config.archive_root),
            active_db_path=str(env.config.db_path),
        ),
        total_sessions=total_matching,
        total_messages=total_msg,
        total_tool_calls=total_tools,
    )

    click.echo(serialize_surface_payload(payload))


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

    return ContextPackPayload(
        intent=ContextPackIntent(),
        decisions=ContextPackDecisions(),
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
        provenance=ContextPackProvenance(
            redacted=redact,
            archive_runtime="archive_file_set",
            archive_root=str(archive_root),
            active_db_path=str(archive_root / "index.db"),
        ),
        total_sessions=total_matching,
        total_messages=total_msg,
        total_tool_calls=total_tools,
    )
