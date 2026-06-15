"""Context preamble composition for the ``read --view context`` surface.

The capability behind ``read --view context`` (#1842): it absorbed the former
standalone ``context compose`` command (#1494). The MCP
``compose_context_preamble`` tool exposes the same capability programmatically.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from polylogue.cli.shared.types import AppEnv


def run_context_compose(env: AppEnv, *, session_id: str, related_limit: int = 5) -> str:
    """Compose a context preamble JSON document for a seed session (#1494).

    Resolves the session's lineage (from topology), recent related sessions,
    and project state into a single preamble and returns it as a JSON string.
    """
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
        repo: object = getattr(conv, "git_repository_url", None)
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
    git_repo = getattr(conv, "git_repository_url", None)
    git_branch = getattr(conv, "git_branch", None)
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
    return json.dumps(preamble, indent=2, default=str)
