"""Session display-label projection (polylogue-6e7m).

A readable session label is a **read-time projection**, never a stored
column. It is computed from structural inputs already present in the index:

- the session's normalized repository name (or a directory fallback when no
  repository could be resolved -- see ``archive/session/repo_identity.py``
  and the write-path fix in ``storage/sqlite/archive_tiers/write.py``),
- the number of distinct touched repo-relative file paths,
- the session's message count,
- the session date.

The label is never written to
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
    "distinct_repo_relative_file_count_for_session",
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
    """Retained for compatibility with the earlier path-based projection."""
    additional_file_count: int
    """Retained for compatibility with the earlier path-based projection."""
    message_count: int
    distinct_file_count: int | None = None
    session_date: str | None = None


def compute_session_structural_label(inputs: SessionLabelInputs) -> str:
    """Compose the structural label for one session.

    Provider title wins when present. Otherwise:
    ``<repo> · <files> files · <messages> msgs · <date>``, degrading
    gracefully as pieces of evidence are missing.
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

    distinct_file_count = inputs.distinct_file_count
    if distinct_file_count is None:
        distinct_file_count = (1 + inputs.additional_file_count) if inputs.dominant_path else 0
    if distinct_file_count:
        noun = "file" if distinct_file_count == 1 else "files"
        parts.append(f"{distinct_file_count} {noun}")

    parts.append(f"{inputs.message_count} msgs")
    if inputs.session_date:
        parts.append(inputs.session_date[:10])
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


def distinct_repo_relative_file_count_for_session(
    conn: sqlite3.Connection,
    session_id: str,
) -> int:
    """Return the number of distinct repo-relative files touched by a session."""
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
    paths = {repo_relative_path(str(raw_path), root_path) if root_path else str(raw_path) for (raw_path,) in rows}
    return len({path for path in paths if path})


def session_structural_label_for_session(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    message_count: int,
    provider_title: str | None,
    session_date: str | None = None,
) -> str:
    """Compute the structural label for ``session_id`` against a live index.db.

    Pure read: issues SELECTs only (``repos``/``session_repos``/
    ``action_pairs``), never writes.
    """
    repo_root = _session_repo_root(conn, session_id)
    distinct_file_count = distinct_repo_relative_file_count_for_session(conn, session_id)

    inputs = SessionLabelInputs(
        provider_title=provider_title,
        repo_name=repo_root.repo_name if repo_root else None,
        is_directory=repo_root.is_directory if repo_root else True,
        dominant_path=None,
        additional_file_count=0,
        message_count=message_count,
        distinct_file_count=distinct_file_count,
        session_date=session_date,
    )
    return compute_session_structural_label(inputs)
