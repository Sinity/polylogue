"""Context pack MCP tool registration.

Registers ``build_context_pack`` — the agent-facing context assembly tool
that produces provenance-rich project/date/query context packs from canonical
archive tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.context.pack import build_archive_context_pack_payload

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
            payload = await build_archive_context_pack_payload(
                archive_root=config.archive_root,
                polylogue=hooks.get_polylogue(),
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
            from polylogue.context.preamble import build_context_preamble_payload
            from polylogue.surfaces.payloads import ContextPreamble, ContextPreambleProjectState

            poly = hooks.get_polylogue()
            preamble = await build_context_preamble_payload(
                poly,
                session_id="",
                related_limit=hooks.clamp_limit(limit),
                repo_path=repo_path,
                cwd=cwd,
                recent_files=recent_files,
                source_tool_calls={"compose_context_preamble": "polylogue-mcp"},
                require_session=False,
            )

            # ``compose_context_preamble`` for SessionStart may run before a
            # seed session exists. Keep the project/recent-session behavior
            # useful by returning an empty payload enriched with git state.
            if preamble is None:
                preamble = ContextPreamble(
                    preamble_version="1.0",
                    source_tool_calls={"compose_context_preamble": "polylogue-mcp"},
                )

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

            if project is not None:
                payload = preamble.model_dump(mode="json", exclude_none=True)
                payload["project_state"] = project.model_dump(mode="json", exclude_none=True)
                preamble = ContextPreamble.model_validate(payload)
            return hooks.json_payload(preamble, exclude_none=True)

        return await hooks.async_safe_call("compose_context_preamble", run)


__all__ = ["register_context_tools"]
