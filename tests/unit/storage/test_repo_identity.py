"""Tests for the repo-identity projection (#1253)."""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import aiosqlite
import pytest

from polylogue.archive.session.attribution import SessionAttribution
from polylogue.storage.insights.session.repo_identity import (
    RepoObservation,
    attribution_to_observations,
    list_repo_identities,
    list_sessions_for_repo,
    refresh_session_repo_observations,
    refresh_session_repo_observations_sync,
)
from polylogue.storage.sqlite.schema_ddl import SCHEMA_DDL


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_DDL)
    return conn


def _seed_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "INSERT INTO sessions (session_id, source_name, provider_session_id, version) VALUES (?, ?, ?, ?)",
        (session_id, "claude-code", f"pcid-{session_id}", 1),
    )


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
    _seed_session(conn, "conv-a")
    obs = (
        RepoObservation(
            origin_url="https://github.com/Sinity/polylogue.git",
            root_path="/realm/project/polylogue",
            repo_name="polylogue",
            branch_name="master",
        ),
    )

    written = refresh_session_repo_observations_sync(conn, "conv-a", obs)
    assert written == 1
    identities = conn.execute("SELECT id, origin_url, root_path FROM repo_identities").fetchall()
    assert [(r["origin_url"], r["root_path"]) for r in identities] == [
        ("https://github.com/Sinity/polylogue.git", "/realm/project/polylogue"),
    ]
    observations = conn.execute(
        "SELECT session_id, repo_identity_id, branch_name FROM session_repo_observations"
    ).fetchall()
    assert [(r["session_id"], r["branch_name"]) for r in observations] == [("conv-a", "master")]

    # Re-running deduplicates the identity but updates last_seen_at and branch.
    obs_v2 = (
        RepoObservation(
            origin_url="https://github.com/Sinity/polylogue.git",
            root_path="/realm/project/polylogue",
            repo_name="polylogue",
            branch_name="feature/foo",
        ),
    )
    refresh_session_repo_observations_sync(conn, "conv-a", obs_v2)
    identities_after = conn.execute("SELECT id FROM repo_identities").fetchall()
    assert len(identities_after) == 1
    branch_after = conn.execute(
        "SELECT branch_name FROM session_repo_observations WHERE session_id = ?",
        ("conv-a",),
    ).fetchone()
    assert branch_after["branch_name"] == "feature/foo"


def test_refresh_sync_replaces_existing_observations() -> None:
    conn = _make_db()
    _seed_session(conn, "conv-b")
    refresh_session_repo_observations_sync(
        conn,
        "conv-b",
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
    count_a = conn.execute("SELECT COUNT(*) FROM session_repo_observations").fetchone()[0]
    assert count_a == 2

    refresh_session_repo_observations_sync(
        conn,
        "conv-b",
        (
            RepoObservation(
                origin_url="",
                root_path="/realm/project/A",
                repo_name="a",
                branch_name="",
            ),
        ),
    )
    count_b = conn.execute("SELECT COUNT(*) FROM session_repo_observations").fetchone()[0]
    assert count_b == 1


def test_session_cascade_drops_observations() -> None:
    conn = _make_db()
    conn.execute("PRAGMA foreign_keys = ON")
    _seed_session(conn, "conv-c")
    refresh_session_repo_observations_sync(
        conn,
        "conv-c",
        (
            RepoObservation(
                origin_url="",
                root_path="/realm/project/polylogue",
                repo_name="polylogue",
                branch_name="",
            ),
        ),
    )
    conn.execute("DELETE FROM sessions WHERE session_id = ?", ("conv-c",))
    remaining = conn.execute("SELECT COUNT(*) FROM session_repo_observations").fetchone()[0]
    assert remaining == 0
    # Identity row survives the cascade (it can still be observed by another session).
    assert conn.execute("SELECT COUNT(*) FROM repo_identities").fetchone()[0] == 1


@pytest.mark.parametrize(
    "lookup, expected",
    [
        ({"origin_url": "https://github.com/Sinity/polylogue.git"}, ("conv-x", "conv-y")),
        ({"repo_name": "polylogue"}, ("conv-x", "conv-y")),
        ({"root_path": "/realm/project/polylogue"}, ("conv-x", "conv-y")),
        ({"origin_url": "https://github.com/other/repo.git"}, ()),
    ],
)
def test_list_sessions_for_repo_async(lookup: dict[str, Any], expected: tuple[str, ...]) -> None:
    async def _run() -> tuple[str, ...]:
        async with aiosqlite.connect(":memory:") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(SCHEMA_DDL)
            await conn.execute(
                "INSERT INTO sessions (session_id, source_name, provider_session_id, version) VALUES (?, ?, ?, ?)",
                ("conv-x", "claude-code", "px", 1),
            )
            await conn.execute(
                "INSERT INTO sessions (session_id, source_name, provider_session_id, version) VALUES (?, ?, ?, ?)",
                ("conv-y", "codex", "py", 1),
            )
            await conn.execute(
                "INSERT INTO sessions (session_id, source_name, provider_session_id, version) VALUES (?, ?, ?, ?)",
                ("conv-z", "chatgpt", "pz", 1),
            )
            obs = (
                RepoObservation(
                    origin_url="https://github.com/Sinity/polylogue.git",
                    root_path="/realm/project/polylogue",
                    repo_name="polylogue",
                    branch_name="master",
                ),
            )
            await refresh_session_repo_observations(conn, "conv-x", obs)
            await refresh_session_repo_observations(conn, "conv-y", obs)
            await refresh_session_repo_observations(
                conn,
                "conv-z",
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
            await conn.executescript(SCHEMA_DDL)
            with pytest.raises(ValueError):
                await list_sessions_for_repo(conn)

    asyncio.run(_run())


def test_list_repo_identities_filter_by_name() -> None:
    async def _run() -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
        async with aiosqlite.connect(":memory:") as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(SCHEMA_DDL)
            await conn.execute(
                "INSERT INTO sessions (session_id, source_name, provider_session_id, version) VALUES (?, ?, ?, ?)",
                ("conv-list", "claude-code", "pl", 1),
            )
            await refresh_session_repo_observations(
                conn,
                "conv-list",
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
                await list_repo_identities(conn),
                await list_repo_identities(conn, repo_name="polylogue"),
            )

    all_identities, filtered = asyncio.run(_run())
    assert {row["repo_name"] for row in all_identities} == {"polylogue", "sinex"}
    assert {row["repo_name"] for row in filtered} == {"polylogue"}
