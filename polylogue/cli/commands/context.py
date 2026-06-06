"""Context Composer CLI — assemble context packs from archive objects (#1494)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import click

from polylogue.cli.shared.types import AppEnv


@click.group("context")
def context_command() -> None:
    """Context Composer — assemble context packs from archive objects."""


@context_command.command("compose")
@click.argument("session_id")
@click.option("--related-limit", "-n", type=int, default=5, help="Number of related sessions to include.")
@click.pass_obj
def compose_command(env: AppEnv, session_id: str, related_limit: int) -> None:
    """Compose a context preamble for a session (#1494)."""
    from polylogue.api.sync.bridge import run_coroutine_sync

    conv = run_coroutine_sync(env.polylogue.get_session(session_id))
    if conv is None:
        env.ui.error(f"Session not found: {session_id}")
        raise SystemExit(1)

    # Session lineage from topology.
    lineage: dict[str, object] = {}
    try:
        topology = run_coroutine_sync(env.polylogue.get_session_topology(session_id))
        if topology:
            lineage = {
                "logical_session_root": getattr(topology, "logical_session_id", None),
                "parent_session_id": getattr(topology, "parent_session_id", None),
            }
    except Exception:
        pass

    # Recent related sessions.
    related: list[dict[str, object]] = []
    try:
        meta = conv.provider_meta or {}
        repo: object = meta.get("git_repository_url")
        candidates = run_coroutine_sync(
            env.polylogue.find_resume_candidates(
                repo_path=str(repo) if repo else ".",
                limit=related_limit,
            )
        )
        for c in candidates:
            related.append(
                {
                    "session_id": getattr(c, "logical_session_id", None) or getattr(c, "session_id", "?"),
                    "title": getattr(c, "title", None),
                    "terminal_state": getattr(c, "terminal_state", None),
                }
            )
    except Exception:
        pass

    # Project state.
    project: dict[str, object] = {}
    meta = conv.provider_meta or {}
    if isinstance(meta, dict):
        git_repo = meta.get("git_repository_url")
        git_branch = meta.get("git_branch")
        if git_repo or git_branch:
            project = {
                "repo": str(git_repo) if git_repo else None,
                "branch": str(git_branch) if git_branch else None,
            }

    preamble: dict[str, object] = {
        "preamble_version": "1.0",
        "injected_at": datetime.now(timezone.utc).isoformat(),
        "session_lineage": lineage or None,
        "recent_related_sessions": related,
        "open_issues": [],
        "project_state": project or None,
        "guidance": None,
    }
    env.ui.console.print(json.dumps(preamble, indent=2, default=str))
