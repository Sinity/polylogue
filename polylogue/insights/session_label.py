"""Session structural label projection (polylogue-cijx.4 decision 3).

A readable session label is a **read-time projection**, never a stored
column. It is computed from three inputs that must themselves be resolved
correctly first (decisions 1 and 2 of the same bead):

- the session's normalized repository name (or a directory fallback when no
  repository could be resolved -- see ``archive/session/repo_identity.py``
  and the write-path fix in ``storage/sqlite/archive_tiers/write.py``),
- the session's dominant repo-relative file path (``repo_relative_path``),
- the session's message count.

When the provider supplied a real title, that title is used in place of the
path clause -- the structural form only fires as a fallback for the large
share of sessions that never get a provider title (auto-generated agent
sessions, subagent dispatch, etc). The label is never written to
``sessions.title``: doing so would collide with genuine provider titles and
freeze mid-session ("340 msgs" becomes wrong the moment message 341 lands).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass

from polylogue.archive.session.repo_identity import repo_relative_path

__all__ = [
    "SessionLabelInputs",
    "SessionRepoRootPath",
    "compute_session_structural_label",
    "dominant_repo_relative_path_for_session",
    "session_structural_label_for_session",
]


@dataclass(frozen=True, slots=True)
class SessionLabelInputs:
    """Inputs to the structural label projection for one session."""

    provider_title: str | None
    repo_name: str | None
    """``None`` when no repository could be resolved for this session --
    the session resolves to a directory, not a repository (decision 1)."""
    is_directory: bool
    """``True`` when ``repo_name`` names a bare directory rather than a
    resolved repository (no git remote, no discoverable git root)."""
    dominant_path: str | None
    """The most-touched repo-relative file path, already stripped of the
    checkout root (decision 2), or ``None`` when no file evidence exists."""
    additional_file_count: int
    """Count of OTHER distinct files touched besides ``dominant_path``."""
    message_count: int


def compute_session_structural_label(inputs: SessionLabelInputs) -> str:
    """Compose the structural label for one session.

    Provider title wins when present. Otherwise:
    ``<repo-or-directory> · <dominant path>[ +N] · <messages> msgs``,
    degrading gracefully as pieces of evidence are missing.
    """
    if inputs.provider_title and inputs.provider_title.strip():
        return inputs.provider_title.strip()

    parts: list[str] = []
    # polylogue-cijx.2 AC4: a bare directory must not be displayed as if it
    # were a repository, even defensively -- if ``is_directory`` is ever True
    # while ``repo_name`` is populated (e.g. stale pre-fix data, or a future
    # regression in the write/read-path invariant this guards), the read
    # surface still refuses to print it as a repo name.
    if inputs.repo_name and not inputs.is_directory:
        parts.append(inputs.repo_name)

    if inputs.dominant_path:
        path_clause = inputs.dominant_path
        if inputs.additional_file_count > 0:
            path_clause = f"{path_clause} +{inputs.additional_file_count}"
        parts.append(path_clause)

    parts.append(f"{inputs.message_count} msgs")
    return " · ".join(parts)


@dataclass(frozen=True, slots=True)
class SessionRepoRootPath:
    repo_name: str | None
    root_path: str
    is_directory: bool


def _session_repo_root(conn: sqlite3.Connection, session_id: str) -> SessionRepoRootPath | None:
    """Resolve the checkout root/name a session is associated with.

    Picks the repo with the most recent ``observed_at_ms`` when a session
    touched more than one (rare, but possible for a long-running session
    whose cwd changed). Returns ``None`` when the session has no repo
    observation at all.
    """
    row = conn.execute(
        """
        SELECT r.root_path, r.repo_name
        FROM session_repos sr
        JOIN repos r ON r.repo_id = sr.repo_id
        WHERE sr.session_id = ?
        ORDER BY sr.observed_at_ms DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    root_path, repo_name = row[0], row[1]
    # A ``session_repos`` row exists at all only because ``_write_repo_edges``
    # (storage/sqlite/archive_tiers/write.py, polylogue-cijx.2 AC4) refuses to
    # write one unless real git evidence was found -- either a discovered
    # ``.git`` root on disk (``root_path`` non-empty) or an explicit remote
    # (``repos.origin_url`` non-empty). Testing ``not origin_url`` alone was
    # wrong: it mislabeled every locally-checked-out repo with no configured
    # remote (root_path set, origin_url empty) as ``is_directory=True``,
    # exactly the "sinity repo from a bare directory" mislabeling this AC
    # exists to kill -- just moved one layer down from the write path to the
    # read path. A row's mere existence is proof of a real repository; the
    # only genuine "directory" case is no row at all (handled by the
    # caller's fallback), so this no longer needs ``origin_url`` at all.
    return SessionRepoRootPath(
        repo_name=repo_name or None,
        root_path=root_path or "",
        is_directory=False,
    )


def dominant_repo_relative_path_for_session(
    conn: sqlite3.Connection,
    session_id: str,
) -> tuple[str | None, int]:
    """Return ``(dominant_repo_relative_path, additional_file_count)``.

    Reads ``action_pairs.tool_path`` for the session, strips the resolved
    checkout root (decision 2), and picks the most frequently touched
    distinct path as dominant. Ties break on lexical order for determinism.
    """
    repo_root = _session_repo_root(conn, session_id)
    root_path = repo_root.root_path if repo_root else ""

    rows = conn.execute(
        """
        SELECT tool_path
        FROM action_pairs
        WHERE session_id = ? AND tool_path IS NOT NULL AND tool_path != ''
        """,
        (session_id,),
    ).fetchall()
    if not rows:
        return None, 0

    counts: Counter[str] = Counter()
    for (raw_path,) in rows:
        relative = repo_relative_path(str(raw_path), root_path) if root_path else str(raw_path)
        if relative:
            counts[relative] += 1

    if not counts:
        return None, 0

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    dominant_path = ranked[0][0]
    additional_file_count = len(ranked) - 1
    return dominant_path, additional_file_count


def session_structural_label_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    message_count: int,
    provider_title: str | None,
) -> str:
    """Compute the structural label for ``session_id`` against a live index.db.

    Pure read: issues SELECTs only (``repos``/``session_repos``/
    ``action_pairs``), never writes.
    """
    repo_root = _session_repo_root(conn, session_id)
    dominant_path, additional_file_count = dominant_repo_relative_path_for_session(conn, session_id)

    inputs = SessionLabelInputs(
        provider_title=provider_title,
        repo_name=repo_root.repo_name if repo_root else None,
        is_directory=repo_root.is_directory if repo_root else True,
        dominant_path=dominant_path,
        additional_file_count=additional_file_count,
        message_count=message_count,
    )
    return compute_session_structural_label(inputs)
