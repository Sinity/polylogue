"""Context pack MCP tool registration.

Registers ``build_context_image`` — the agent-facing context assembly tool. It is
a thin lens over the shared ``compile_context`` engine: session selection runs
through the query algebra, then the bounded ``ContextImage`` payload (segments,
token-budgeted accumulation, omission accounting, assertions) is compiled by the
same code path the CLI ``read`` modifiers and the ``compile_context`` tool use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from polylogue.mcp.server_support import ServerCallbacks

_DEFAULT_MAX_SESSIONS = 5
_DEFAULT_MAX_MESSAGES = 20
_DETAIL_INCLUDES_MESSAGES: dict[str, bool] = {
    "summary": False,  # metadata only, no message bodies
    "compact": True,
    "full": True,
}


def _detail_includes_messages(detail_level: str) -> bool:
    """Return whether a detail level includes message bodies."""
    return _DETAIL_INCLUDES_MESSAGES.get(detail_level, True)


def register_context_tools(mcp: FastMCP, hooks: ServerCallbacks) -> None:
    @mcp.tool()
    async def build_context_image(
        project_path: str | None = None,
        project_repo: str | None = None,
        since: str | None = None,
        until: str | None = None,
        origin: str | None = None,
        query: str | None = None,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
        max_tokens: int | None = None,
        detail_level: str = "compact",
        redact_paths: bool = True,
        include_assertions: bool = True,
    ) -> str:
        """Build a bounded context image for agent analysis.

        Selects sessions through the query algebra (filters / free-text query)
        and compiles the requested views into the shared ``ContextImage`` payload
        with token-budgeted accumulation and explicit omission accounting.

        Parameters:
            project_path: Filter sessions by cwd prefix pattern.
            project_repo: Filter sessions by git repo URL or name.
            since: Start date for session filter (ISO format).
            until: End date for session filter (ISO format).
            origin: Source-origin filter.
            query: Free-text query for semantic narrowing.
            max_sessions: Maximum sessions to include (1-20).
            max_tokens: Optional token budget; segments over budget are omitted
                with reason 'budget' instead of being silently truncated.
            detail_level: 'summary' (metadata only), 'compact'/'full' (messages).
            redact_paths: Redact filesystem paths for privacy (default True).
            include_assertions: Include context-inject assertion claims.
        """

        async def run() -> str:
            include_messages = _detail_includes_messages(detail_level)
            payload = await hooks.get_polylogue().context_image_payload(
                project_path=project_path,
                project_repo=project_repo,
                since=since,
                until=until,
                origin=origin,
                query=query,
                max_sessions=hooks.clamp_limit(max_sessions),
                max_tokens=max_tokens,
                include_messages=include_messages,
                include_assertions=include_assertions,
                redact_paths=redact_paths,
            )
            return hooks.json_payload(payload, exclude_none=True)

        return await hooks.async_safe_call("build_context_image", run)

    @mcp.tool()
    async def compose_context_preamble(
        repo_path: str | None = None,
        cwd: str | None = None,
        recent_files: tuple[str, ...] = (),
        limit: int = 5,
        successor_session_id: str | None = None,
    ) -> str:
        """Compose a ContextPreamble for SessionStart context injection (#1494).

        Builds a typed preamble with session lineage, recent related sessions,
        and project state for Claude Code and Codex sessions. Designed for
        SessionStart hook scripts that inject context into new agent sessions.
        When the harness has allocated the new session identity, pass it as
        ``successor_session_id`` so the call-log row and resulting preamble are
        queryable against the session that received the context.
        """

        async def run() -> str:
            from polylogue.context.preamble import build_context_preamble_payload
            from polylogue.surfaces.payloads import ContextPreamble, ContextPreambleProjectState

            poly = hooks.get_polylogue()
            preamble = await build_context_preamble_payload(
                poly,
                session_id=successor_session_id or "",
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

        return await hooks.async_safe_call(
            "compose_context_preamble",
            run,
            session_id=successor_session_id,
        )


__all__ = ["register_context_tools"]
