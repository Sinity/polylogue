"""Tests for the repo observation projection (#1253)."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from polylogue.archive.session.attribution import SessionAttribution, extract_attribution_from_actions
from polylogue.storage.derived.session.repo_observations import (
    RepoObservation,
    attribution_to_observations,
    list_repos,
    list_sessions_for_repo,
    refresh_session_repos,
    refresh_session_repos_sync,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _session_id(origin: str, native_id: str) -> str:
    return f"{origin}:{native_id}"


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _seed_session(conn: sqlite3.Connection, native_id: str, *, origin: str = "claude-code-session") -> str:
    conn.execute(
        """
        INSERT INTO sessions (
            native_id, origin, title, content_hash, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, zeroblob(32), 0, 0)
        """,
        (native_id, origin, native_id),
    )
    return _session_id(origin, native_id)


def _attr(
    *,
    repo_paths: tuple[str, ...] = (),
    repo_names: tuple[str, ...] = (),
    branch_names: tuple[str, ...] = (),
) -> SessionAttribution:
    return SessionAttribution(
        repo_paths=repo_paths,
        repo_names=repo_names,
        cwd_paths=repo_paths,
        branch_names=branch_names,
        file_paths_touched=(),
        languages_detected=(),
    )


def test_attribution_to_observations_dedupes_paths() -> None:
    attr = _attr(
        repo_paths=("/realm/project/polylogue", "/realm/project/polylogue"),
        branch_names=("master",),
    )
    observations = attribution_to_observations(attr, git_repository_url="https://github.com/Sinity/polylogue.git")
    assert len(observations) == 1
    assert observations[0].origin_url == "https://github.com/Sinity/polylogue.git"
    assert observations[0].root_path == "/realm/project/polylogue"
    assert observations[0].repo_name == "polylogue"
    assert observations[0].branch_name == "master"


def test_attribution_to_observations_origin_only_when_no_paths() -> None:
    attr = _attr(repo_paths=(), branch_names=())
    observations = attribution_to_observations(attr, git_repository_url="https://github.com/Sinity/sinex.git")
    assert len(observations) == 1
    assert observations[0].origin_url == "https://github.com/Sinity/sinex.git"
    assert observations[0].root_path == ""
    assert observations[0].repo_name == "sinex"


def test_attribution_to_observations_returns_empty_without_signal() -> None:
    assert attribution_to_observations(_attr()) == ()


def test_attribution_to_observations_local_repo_without_origin() -> None:
    attr = _attr(repo_paths=("/realm/project/polylogue",))
    observations = attribution_to_observations(attr, git_repository_url=None)
    assert len(observations) == 1
    assert observations[0].origin_url == ""
    assert observations[0].root_path == "/realm/project/polylogue"
    assert observations[0].repo_name == "polylogue"


def test_refresh_sync_upserts_identity_and_observation() -> None:
    conn = _make_db()
    session_id = _seed_session(conn, "conv-a")
    obs = (
        RepoObservation(
            origin_url="https://github.com/Sinity/polylogue.git",
            root_path="/realm/project/polylogue",
            repo_name="polylogue",
            branch_name="master",
        ),
    )

    written = refresh_session_repos_sync(conn, session_id, obs)
    assert written == 1
    repos = conn.execute("SELECT repo_id, origin_url, root_path FROM repos").fetchall()
    assert [(r["origin_url"], r["root_path"]) for r in repos] == [
        ("https://github.com/Sinity/polylogue.git", "/realm/project/polylogue"),
    ]
    observations = conn.execute("SELECT session_id, repo_id, branch_name FROM session_repos").fetchall()
    assert [(r["session_id"], r["branch_name"]) for r in observations] == [(session_id, "master")]

    # Re-running deduplicates the repo but updates last_seen_at and branch.
    obs_v2 = (
        RepoObservation(
            origin_url="https://github.com/Sinity/polylogue.git",
            root_path="/realm/project/polylogue",
            repo_name="polylogue",
            branch_name="feature/foo",
        ),
    )
    refresh_session_repos_sync(conn, session_id, obs_v2)
    repos_after = conn.execute("SELECT repo_id FROM repos").fetchall()
    assert len(repos_after) == 1
    branch_after = conn.execute(
        "SELECT branch_name FROM session_repos WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert branch_after["branch_name"] == "feature/foo"


def test_refresh_stamps_the_session_time_not_the_wall_clock() -> None:
    """A rebuild reproduces repo rows from the archive alone.

    Anti-vacuity: stamp the rows with the current time instead, as the
    materializer once did, and every rebuild moves ``last_seen_at_ms`` and
    ``observed_at_ms`` (the convergence idempotence properties went red on it).
    """
    conn = _make_db()
    session_id = _seed_session(conn, "conv-time")
    conn.execute("UPDATE sessions SET created_at_ms = 1000, updated_at_ms = 5000 WHERE session_id = ?", (session_id,))
    obs = (
        RepoObservation(
            origin_url="https://example.invalid/repo.git",
            root_path="/work/repo",
            repo_name="repo",
            branch_name="main",
        ),
    )

    refresh_session_repos_sync(conn, session_id, obs)

    assert conn.execute("SELECT observed_at_ms FROM session_repos").fetchall()[0][0] == 5000
    assert tuple(conn.execute("SELECT first_seen_at_ms, last_seen_at_ms FROM repos").fetchone()) == (5000, 5000)
    assert tuple(conn.execute("SELECT first_seen_at_ms, last_seen_at_ms FROM repo_checkouts").fetchone()) == (
        5000,
        5000,
    )


def test_refresh_sync_replaces_existing_observations() -> None:
    conn = _make_db()
    session_id = _seed_session(conn, "conv-b")
    refresh_session_repos_sync(
        conn,
        session_id,
        (
            RepoObservation(
                origin_url="",
                root_path="/realm/project/A",
                repo_name="a",
                branch_name="",
            ),
            RepoObservation(
                origin_url="",
                root_path="/realm/project/B",
                repo_name="b",
                branch_name="",
            ),
        ),
    )
    count_a = conn.execute("SELECT COUNT(*) FROM session_repos").fetchone()[0]
    assert count_a == 2

    refresh_session_repos_sync(
        conn,
        session_id,
        (
            RepoObservation(
                origin_url="",
                root_path="/realm/project/A",
                repo_name="a",
                branch_name="",
            ),
        ),
    )
    count_b = conn.execute("SELECT COUNT(*) FROM session_repos").fetchone()[0]
    assert count_b == 1


def test_session_cascade_drops_observations() -> None:
    conn = _make_db()
    conn.execute("PRAGMA foreign_keys = ON")
    session_id = _seed_session(conn, "conv-c")
    refresh_session_repos_sync(
        conn,
        session_id,
        (
            RepoObservation(
                origin_url="",
                root_path="/realm/project/polylogue",
                repo_name="polylogue",
                branch_name="",
            ),
        ),
    )
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    remaining = conn.execute("SELECT COUNT(*) FROM session_repos").fetchone()[0]
    assert remaining == 0
    # Repo row survives the cascade (it can still be observed by another session).
    assert conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0] == 1


@pytest.mark.parametrize(
    "lookup, expected",
    [
        (
            {"origin_url": "https://github.com/Sinity/polylogue.git"},
            (_session_id("claude-code-session", "conv-x"), _session_id("codex-session", "conv-y")),
        ),
        (
            {"repo_name": "polylogue"},
            (_session_id("claude-code-session", "conv-x"), _session_id("codex-session", "conv-y")),
        ),
        (
            {"root_path": "/realm/project/polylogue"},
            (_session_id("claude-code-session", "conv-x"), _session_id("codex-session", "conv-y")),
        ),
        ({"origin_url": "https://github.com/other/repo.git"}, ()),
    ],
)
def test_list_sessions_for_repo_async(lookup: dict[str, Any], expected: tuple[str, ...]) -> None:
    async def _run() -> tuple[str, ...]:
        async with aiosqlite.connect(":memory:") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(archive_tier_spec(ArchiveTier.INDEX).ddl)
            await conn.execute(
                """
                INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
                VALUES (?, 'claude-code-session', ?, zeroblob(32), 0, 0)
                """,
                ("conv-x", "conv-x"),
            )
            await conn.execute(
                """
                INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
                VALUES (?, 'codex-session', ?, zeroblob(32), 0, 0)
                """,
                ("conv-y", "conv-y"),
            )
            await conn.execute(
                """
                INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
                VALUES (?, 'chatgpt-export', ?, zeroblob(32), 0, 0)
                """,
                ("conv-z", "conv-z"),
            )
            obs = (
                RepoObservation(
                    origin_url="https://github.com/Sinity/polylogue.git",
                    root_path="/realm/project/polylogue",
                    repo_name="polylogue",
                    branch_name="master",
                ),
            )
            await refresh_session_repos(conn, _session_id("claude-code-session", "conv-x"), obs)
            await refresh_session_repos(conn, _session_id("codex-session", "conv-y"), obs)
            await refresh_session_repos(
                conn,
                _session_id("chatgpt-export", "conv-z"),
                (
                    RepoObservation(
                        origin_url="",
                        root_path="/realm/project/other",
                        repo_name="other",
                        branch_name="",
                    ),
                ),
            )
            return await list_sessions_for_repo(conn, **lookup)

    assert asyncio.run(_run()) == expected


def test_list_sessions_for_repo_requires_filter() -> None:
    async def _run() -> None:
        async with aiosqlite.connect(":memory:") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(archive_tier_spec(ArchiveTier.INDEX).ddl)
            with pytest.raises(ValueError):
                await list_sessions_for_repo(conn)

    asyncio.run(_run())


def test_list_repos_filter_by_name() -> None:
    async def _run() -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
        async with aiosqlite.connect(":memory:") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(archive_tier_spec(ArchiveTier.INDEX).ddl)
            await conn.execute(
                """
                INSERT INTO sessions (native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
                VALUES (?, 'claude-code-session', ?, zeroblob(32), 0, 0)
                """,
                ("conv-list", "conv-list"),
            )
            await refresh_session_repos(
                conn,
                _session_id("claude-code-session", "conv-list"),
                (
                    RepoObservation(
                        origin_url="",
                        root_path="/realm/project/polylogue",
                        repo_name="polylogue",
                        branch_name="",
                    ),
                    RepoObservation(
                        origin_url="",
                        root_path="/realm/project/sinex",
                        repo_name="sinex",
                        branch_name="",
                    ),
                ),
            )
            return (
                await list_repos(conn),
                await list_repos(conn, repo_name="polylogue"),
            )

    all_repos, filtered = asyncio.run(_run())
    assert {row["repo_name"] for row in all_repos} == {"polylogue", "sinex"}
    assert {row["repo_name"] for row in filtered} == {"polylogue"}


@pytest.mark.parametrize("component", ["project copy other", "project copy ", "project "])
def test_materializer_preserves_literal_repo_root_next_to_trimmed_neighbor(tmp_path: Path, component: str) -> None:
    neighbor = tmp_path / (component.rstrip() if component.endswith(" ") else "project copy")
    root = tmp_path / component
    for repo in (root, neighbor):
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    attr = extract_attribution_from_actions([], working_directories=[str(root)])
    conn = _make_db()
    try:
        session_id = _seed_session(conn, "literal-root")
        observations = attribution_to_observations(attr)
        refresh_session_repos_sync(conn, session_id, observations)
        assert [(r["root_path"], r["repo_id"]) for r in conn.execute("SELECT root_path,repo_id FROM repos")] == [
            (str(root), f"dir:{root}")
        ]
        assert attr.repo_paths == (str(root),)
        assert attr.repo_names == (component,)
        assert conn.execute("SELECT repo_name FROM repos").fetchone()[0] == component
        conn.commit()
        database = tmp_path / "derived.db"
        with sqlite3.connect(database) as persisted:
            conn.backup(persisted)

        async def read_filters() -> None:
            async with aiosqlite.connect(database) as reader:
                reader.row_factory = aiosqlite.Row
                assert await list_sessions_for_repo(reader, root_path=str(root)) == (session_id,)
                assert await list_sessions_for_repo(reader, repo_name=component) == (session_id,)
                assert await list_sessions_for_repo(reader, root_path=str(neighbor)) == ()
                assert await list_sessions_for_repo(reader, repo_name=neighbor.name) == ()

        asyncio.run(read_filters())
    finally:
        conn.close()
