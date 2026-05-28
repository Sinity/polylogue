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
@click.argument("conversation_id")
@click.option("--related-limit", "-n", type=int, default=5, help="Number of related sessions to include.")
@click.pass_obj
def compose_command(env: AppEnv, conversation_id: str, related_limit: int) -> None:
    """Compose a context preamble for a session (#1494)."""
    from polylogue.api.sync.bridge import run_coroutine_sync

    conv = run_coroutine_sync(env.polylogue.get_conversation(conversation_id))
    if conv is None:
        env.ui.error(f"Conversation not found: {conversation_id}")
        raise SystemExit(1)

    # Session lineage from topology.
    lineage: dict[str, object] = {}
    try:
        topology = run_coroutine_sync(env.polylogue.get_session_topology(conversation_id))
        if topology:
            lineage = {
                "logical_session_root": topology.logical_conversation_id,
                "parent_session_id": topology.parent_conversation_id,
                "sibling_count": len(topology.siblings or ()),
                "chain_depth": len(topology.ancestors or ()),
            }
    except Exception:
        pass

    # Recent related sessions.
    related: list[dict[str, object]] = []
    try:
        repo = (conv.provider_meta or {}).get("git_repository_url") if isinstance(conv.provider_meta, dict) else None
        candidates = run_coroutine_sync(
            env.polylogue.find_resume_candidates(
                repo_path=str(repo) if repo else ".",
                limit=related_limit,
            )
        )
        for c in candidates:
            related.append(
                {
                    "session_id": c.conversation_id,
                    "title": c.title,
                    "date": c.date,
                    "terminal_state": c.terminal_state,
                    "summary": c.summary,
                }
            )
    except Exception:
        pass

    # Project state — git branch and recent commits.
    project: dict[str, object] = {}
    git_repo = (conv.provider_meta or {}).get("git_repository_url") if isinstance(conv.provider_meta, dict) else None
    git_branch = (conv.provider_meta or {}).get("git_branch") if isinstance(conv.provider_meta, dict) else None
    if git_repo or git_branch:
        project = {
            "repo": str(git_repo) if git_repo else None,
            "branch": str(git_branch) if git_branch else None,
        }

    preamble = {
        "preamble_version": "1.0",
        "injected_at": datetime.now(timezone.utc).isoformat(),
        "session_lineage": lineage or None,
        "recent_related_sessions": related,
        "open_issues": [],
        "project_state": project or None,
        "guidance": None,
    }
    env.ui.console.print(json.dumps(preamble, indent=2, default=str))
