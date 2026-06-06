"""Repo-identity observation writers and readers (#1253).

This module owns the per-session maintenance of the
``repo_identities`` and ``session_repo_observations`` tables. It
deduplicates repo identity by ``(origin_url, root_path)`` and records a
many-to-many observation row per session.

Population is invoked from the session-insight refresh path so the
canonical projection stays in lockstep with attribution rebuilds. The
data lives outside the session content-hash boundary by
construction — re-deriving from action events is deterministic.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import aiosqlite

from polylogue.archive.session.attribution import SessionAttribution
from polylogue.archive.session.repo_identity import normalize_repo_name

__all__ = [
    "RepoObservation",
    "attribution_to_observations",
    "list_sessions_for_repo",
    "list_repo_identities",
    "refresh_session_repo_observations",
    "refresh_session_repo_observations_sync",
]


@dataclass(frozen=True, slots=True)
class RepoObservation:
    """One (session, repo) observation prepared for upsert."""

    origin_url: str
    root_path: str
    repo_name: str
    branch_name: str


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def attribution_to_observations(
    attribution: SessionAttribution,
    *,
    git_repository_url: str | None = None,
) -> tuple[RepoObservation, ...]:
    """Translate an attribution bundle into deduplicated repo observations.

    A single session that touched multiple repo roots yields one
    observation per root. ``git_repository_url`` (typed sessions
    column) is associated with every root because the parsers cannot
    currently tell which root the origin URL points at — the join
    surface treats this as best-effort attribution.
    """
    origin_url = (git_repository_url or "").strip()
    branch_name = attribution.branch_names[0] if attribution.branch_names else ""
    observations: list[RepoObservation] = []
    seen: set[tuple[str, str]] = set()
    for root_path in attribution.repo_paths:
        root = root_path.strip()
        if not root:
            continue
        key = (origin_url, root)
        if key in seen:
            continue
        seen.add(key)
        observations.append(
            RepoObservation(
                origin_url=origin_url,
                root_path=root,
                repo_name=normalize_repo_name(root) or "",
                branch_name=branch_name,
            )
        )
    # Some sources (e.g. ChatGPT mentions of GitHub URLs) carry an
    # origin URL but no resolvable local root. Surface the origin
    # anyway so the cross-source "all sessions touching repo X"
    # query can find them, keyed by an empty root_path.
    if origin_url and not observations:
        observations.append(
            RepoObservation(
                origin_url=origin_url,
                root_path="",
                repo_name=normalize_repo_name(origin_url) or "",
                branch_name=branch_name,
            )
        )
    return tuple(observations)


async def _upsert_repo_identity_async(
    conn: aiosqlite.Connection,
    observation: RepoObservation,
    *,
    now_iso: str,
) -> int:
    cursor = await conn.execute(
        "SELECT id FROM repo_identities WHERE origin_url = ? AND root_path = ?",
        (observation.origin_url, observation.root_path),
    )
    row = await cursor.fetchone()
    if row is not None:
        identity_id = int(row[0])
        await conn.execute(
            "UPDATE repo_identities SET last_seen_at = ?, repo_name = ? WHERE id = ?",
            (now_iso, observation.repo_name, identity_id),
        )
        return identity_id
    cursor = await conn.execute(
        "INSERT INTO repo_identities (origin_url, root_path, repo_name, first_seen_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            observation.origin_url,
            observation.root_path,
            observation.repo_name,
            now_iso,
            now_iso,
        ),
    )
    return int(cursor.lastrowid or 0)


async def refresh_session_repo_observations(
    conn: aiosqlite.Connection,
    session_id: str,
    observations: Sequence[RepoObservation],
    *,
    now_iso: str | None = None,
) -> int:
    """Replace the repo observations for ``session_id``.

    Returns the number of observation rows written. Existing rows for
    this session are deleted before insertion so the projection
    cannot drift on rebuild.
    """
    timestamp = now_iso or _utc_now_iso()
    await conn.execute(
        "DELETE FROM session_repo_observations WHERE session_id = ?",
        (session_id,),
    )
    if not observations:
        return 0
    written = 0
    for observation in observations:
        identity_id = await _upsert_repo_identity_async(conn, observation, now_iso=timestamp)
        if identity_id <= 0:
            continue
        await conn.execute(
            "INSERT OR REPLACE INTO session_repo_observations "
            "(session_id, repo_identity_id, branch_name, observed_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, identity_id, observation.branch_name, timestamp),
        )
        written += 1
    return written


def _upsert_repo_identity_sync(
    conn: sqlite3.Connection,
    observation: RepoObservation,
    *,
    now_iso: str,
) -> int:
    row = conn.execute(
        "SELECT id FROM repo_identities WHERE origin_url = ? AND root_path = ?",
        (observation.origin_url, observation.root_path),
    ).fetchone()
    if row is not None:
        identity_id = int(row[0])
        conn.execute(
            "UPDATE repo_identities SET last_seen_at = ?, repo_name = ? WHERE id = ?",
            (now_iso, observation.repo_name, identity_id),
        )
        return identity_id
    cursor = conn.execute(
        "INSERT INTO repo_identities (origin_url, root_path, repo_name, first_seen_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            observation.origin_url,
            observation.root_path,
            observation.repo_name,
            now_iso,
            now_iso,
        ),
    )
    return int(cursor.lastrowid or 0)


def refresh_session_repo_observations_sync(
    conn: sqlite3.Connection,
    session_id: str,
    observations: Sequence[RepoObservation],
    *,
    now_iso: str | None = None,
) -> int:
    """Sync sibling of :func:`refresh_session_repo_observations`."""
    timestamp = now_iso or _utc_now_iso()
    conn.execute(
        "DELETE FROM session_repo_observations WHERE session_id = ?",
        (session_id,),
    )
    if not observations:
        return 0
    written = 0
    for observation in observations:
        identity_id = _upsert_repo_identity_sync(conn, observation, now_iso=timestamp)
        if identity_id <= 0:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO session_repo_observations "
            "(session_id, repo_identity_id, branch_name, observed_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, identity_id, observation.branch_name, timestamp),
        )
        written += 1
    return written


async def list_repo_identities(
    conn: aiosqlite.Connection,
    *,
    repo_name: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Enumerate repo identities, optionally filtered by ``repo_name``."""
    if repo_name:
        cursor = await conn.execute(
            "SELECT id, origin_url, root_path, repo_name, first_seen_at, last_seen_at "
            "FROM repo_identities WHERE repo_name = ? ORDER BY id",
            (repo_name,),
        )
    else:
        cursor = await conn.execute(
            "SELECT id, origin_url, root_path, repo_name, first_seen_at, last_seen_at FROM repo_identities ORDER BY id",
        )
    rows = await cursor.fetchall()
    # sqlite3.Row iteration yields values, not keys — surface columns
    # by name through ``.keys()``.
    return tuple({key: row[key] for key in list(row.keys())} for row in rows)


async def list_sessions_for_repo(
    conn: aiosqlite.Connection,
    *,
    origin_url: str | None = None,
    root_path: str | None = None,
    repo_name: str | None = None,
) -> tuple[str, ...]:
    """Return session IDs touching the matching repo identity.

    At least one of ``origin_url``, ``root_path``, or ``repo_name`` must
    be supplied; results are the union across all matching identities.
    """
    conditions: list[str] = []
    params: list[object] = []
    if origin_url is not None:
        conditions.append("ri.origin_url = ?")
        params.append(origin_url)
    if root_path is not None:
        conditions.append("ri.root_path = ?")
        params.append(root_path)
    if repo_name is not None:
        conditions.append("ri.repo_name = ?")
        params.append(repo_name)
    if not conditions:
        raise ValueError("list_sessions_for_repo requires at least one filter")
    where = " AND ".join(conditions)
    cursor = await conn.execute(
        "SELECT DISTINCT cro.session_id "
        "FROM session_repo_observations cro "
        "JOIN repo_identities ri ON ri.id = cro.repo_identity_id "
        f"WHERE {where} "
        "ORDER BY cro.session_id",
        params,
    )
    rows = await cursor.fetchall()
    return tuple(str(row["session_id"]) for row in rows)
