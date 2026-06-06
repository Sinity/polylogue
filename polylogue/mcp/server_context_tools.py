"""Context pack MCP tool registration.

Registers ``build_context_pack`` — the agent-facing context assembly tool
that produces provenance-rich project/date/query context packs from canonical
archive tables.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.mcp.context_pack import (
    ContextPackDateRange,
    ContextPackMessage,
    ContextPackPayload,
    ContextPackProvenance,
    ContextPackQueryContext,
    ContextPackSession,
    _build_project_context,
    query_archive_context_pack,
    select_context_pack_sessions,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks

_DEFAULT_MAX_SESSIONS = 5
_DEFAULT_MAX_MESSAGES = 20
_DETAIL_DEFAULTS: dict[str, tuple[int, bool]] = {
    "summary": (0, False),  # metadata only, no messages
    "compact": (200, True),  # truncated messages
    "full": (0, True),  # full messages, untruncated
}


def _get_detail_settings(detail_level: str) -> tuple[int, bool]:
    """Return (max_text_length, include_messages) for a detail level."""
    return _DETAIL_DEFAULTS.get(detail_level, _DETAIL_DEFAULTS["compact"])


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, appending ellipsis if truncated."""
    if max_length > 0 and len(text) > max_length:
        return text[:max_length] + "..."
    return text


async def _build_archive_context_pack_payload(
    *,
    archive_root: Path,
    clamp_limit: Callable[[int | object], int],
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
) -> ContextPackPayload:
    with ArchiveStore.open_existing(archive_root) as archive:

        async def query_sessions(spec: SessionQuerySpec) -> list[Any]:
            return query_archive_context_pack(archive, spec, default_limit=_DEFAULT_MAX_SESSIONS)

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
        dates = [d for conv in sessions for d in (conv.created_at, conv.updated_at) if d is not None]
        pack_sessions: list[ContextPackSession] = []

        for conv in sessions[:conv_limit]:
            conv_id = str(conv.id)
            messages: list[ContextPackMessage] = []
            if include_messages:
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
                                has_tool_use=message.has_tool_use,
                                has_thinking=message.has_thinking,
                            )
                        )

            session_origin = str(conv.origin) if getattr(conv, "origin", None) is not None else "unknown"
            pack_sessions.append(
                ContextPackSession(
                    session_id=conv_id,
                    origin=session_origin,
                    title=conv.title,
                    created_at=conv.created_at.isoformat() if conv.created_at else None,
                    updated_at=conv.updated_at.isoformat() if conv.updated_at else None,
                    message_count=int(conv.message_count),
                    tool_use_count=None,
                    messages=messages,
                )
            )

    total_matching = len(sessions)
    return ContextPackPayload(
        project=_build_project_context((), redact=redact_paths),
        date_range=ContextPackDateRange(
            since=since,
            until=until,
            earliest=min(dates).isoformat() if dates else None,
            latest=max(dates).isoformat() if dates else None,
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
            generated_at=datetime.now(UTC).isoformat(),
            redacted=redact_paths,
            archive_runtime="archive_file_set",
            archive_root=str(archive_root),
            active_db_path=str(archive_root / "index.db"),
        ),
        total_sessions=total_matching,
        total_messages=sum(int(conv.message_count) for conv in sessions[:conv_limit]),
        total_tool_calls=0,
    )


def register_context_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def build_context_pack(
        project_path: str | None = None,
        project_repo: str | None = None,
        since: str | None = None,
        until: str | None = None,
        origin: str | None = None,
        query: str | None = None,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
        max_messages_per_session: int = _DEFAULT_MAX_MESSAGES,
        detail_level: str = "compact",
        redact_paths: bool = True,
    ) -> str:
        """Build a provenance-rich context pack for agent analysis.

        Assembles project context, date range, filtered sessions,
        action summaries, and unresolved work from the archive.

        Parameters:
            project_path: Filter sessions by cwd prefix pattern.
            project_repo: Filter sessions by git repo URL or name.
            since: Start date for session filter (ISO format).
            until: End date for session filter (ISO format).
            origin: Source-origin filter.
            query: Free-text query for semantic narrowing.
            max_sessions: Maximum sessions to include (1-20).
            max_messages_per_session: Max messages per session (1-100).
            detail_level: 'summary' (metadata only), 'compact' (truncated), 'full'.
            redact_paths: Redact filesystem paths for privacy (default True).
        """

        async def run() -> str:
            conv_limit = max(1, min(max_sessions, 20))
            msg_limit = max(1, min(max_messages_per_session, 100))
            valid_detail = detail_level if detail_level in _DETAIL_DEFAULTS else "compact"
            max_text, include_messages = _get_detail_settings(valid_detail)

            config = hooks.get_config()
            payload = await _build_archive_context_pack_payload(
                archive_root=config.archive_root,
                clamp_limit=hooks.clamp_limit,
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
            )
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("build_context_pack", run)

    @mcp.tool()
    async def compose_context_preamble(
        repo_path: str | None = None,
        cwd: str | None = None,
        recent_files: tuple[str, ...] = (),
        limit: int = 5,
    ) -> str:
        """Compose a ContextPreamble for SessionStart context injection (#1494).

        Builds a typed preamble with session lineage, recent related sessions,
        and project state for Claude Code and Codex sessions. Designed for
        SessionStart hook scripts that inject context into new agent sessions.
        """

        async def run() -> str:
            from datetime import datetime, timezone

            from polylogue.surfaces.payloads import (
                ContextPreamble,
                ContextPreambleProjectState,
                ContextPreambleSession,
            )

            poly = hooks.get_polylogue()

            # Recent related sessions from resume candidates.
            recent: list[ContextPreambleSession] = []
            try:
                candidates = await poly.find_resume_candidates(
                    repo_path=repo_path or ".",
                    cwd=cwd,
                    recent_files=recent_files,
                    limit=hooks.clamp_limit(limit),
                )
                for c in candidates:
                    cid: str = getattr(c, "logical_session_id", None) or getattr(c, "session_id", "") or "?"
                    recent.append(
                        ContextPreambleSession(
                            session_id=cid,
                            title=getattr(c, "title", None),
                            date=getattr(c, "date", None),
                            terminal_state=getattr(c, "terminal_state", None),
                            summary=getattr(c, "summary", None),
                            origin=getattr(c, "origin", None),
                        )
                    )
            except Exception:
                pass

            # Project state from git.
            project: ContextPreambleProjectState | None = None
            try:
                import subprocess

                branch: str | None = None
                commits: list[str] = []
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=cwd or ".",
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()
                result2 = subprocess.run(
                    ["git", "log", "--oneline", "-5"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=cwd or ".",
                )
                if result2.returncode == 0:
                    commits = [line.strip() for line in result2.stdout.strip().split("\n") if line]
                if branch or commits:
                    project = ContextPreambleProjectState(branch=branch, recent_commits=commits)
            except Exception:
                pass

            preamble = ContextPreamble(
                preamble_version="1.0",
                injected_at=datetime.now(timezone.utc).isoformat(),
                source_tool_calls={"compose_context_preamble": "polylogue-mcp"},
                recent_related_sessions=recent,
                project_state=project,
            )
            return hooks.json_payload(preamble, exclude_none=True)

        return await hooks.async_safe_call("compose_context_preamble", run)


__all__ = ["register_context_tools"]
