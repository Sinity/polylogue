"""Repo facet canonicalization contracts."""

from __future__ import annotations

import sqlite3

from polylogue.api.archive import _archive_aggregate_facet_families, _canonical_repo_facet_label


def test_canonical_repo_facet_label_prefers_product_repo_identity() -> None:
    assert (
        _canonical_repo_facet_label(
            repo_name="polylogue",
            root_path="/realm/project/polylogue",
            origin_url="https://github.com/Sinity/polylogue.git",
        )
        == "polylogue"
    )
    assert (
        _canonical_repo_facet_label(
            repo_name="",
            root_path="/realm/data/exports/chatlog",
            origin_url="https://github.com/Sinity/polylogue.git",
        )
        == "polylogue"
    )


def test_canonical_repo_facet_label_omits_archive_path_noise() -> None:
    assert _canonical_repo_facet_label(repo_name="", root_path="/realm/data/exports/2025", origin_url="") is None
    assert (
        _canonical_repo_facet_label(repo_name=".agent", root_path="/realm/project/polylogue/.agent", origin_url="")
        is None
    )
    assert _canonical_repo_facet_label(repo_name="", root_path="/realm/data/exports/Takeout", origin_url="") is None


def test_archive_facet_aggregation_counts_omitted_repo_noise() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE repos (
            repo_id TEXT PRIMARY KEY,
            origin_url TEXT,
            root_path TEXT,
            repo_name TEXT
        );
        CREATE TABLE session_repos (
            session_id TEXT,
            repo_id TEXT
        );
        CREATE TABLE messages (
            session_id TEXT,
            role TEXT,
            material_origin TEXT,
            message_type TEXT,
            has_tool_use INTEGER,
            has_thinking INTEGER,
            has_paste INTEGER
        );
        CREATE TABLE actions (
            session_id TEXT,
            semantic_type TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO repos(repo_id, origin_url, root_path, repo_name) VALUES (?, ?, ?, ?)",
        [
            ("repo-good", "https://github.com/Sinity/polylogue.git", "/realm/project/polylogue", ""),
            ("repo-noise", "", "/realm/data/exports/2025", ""),
        ],
    )
    conn.executemany(
        "INSERT INTO session_repos(session_id, repo_id) VALUES (?, ?)",
        [
            ("s1", "repo-good"),
            ("s2", "repo-noise"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO messages(session_id, role, material_origin, message_type, has_tool_use, has_thinking, has_paste)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("s1", "user", "human_authored", "message", 0, 0, 0),
            ("s1", "assistant", "assistant_authored", "message", 1, 0, 0),
            ("s2", "user", "runtime_protocol", "tool_result", 0, 0, 0),
        ],
    )

    buckets = _archive_aggregate_facet_families(conn, session_ids=None)

    assert buckets["repos"] == {"polylogue": 1}
    assert buckets["omitted"] == {"repos": 1}
    assert buckets["role_counts"] == {"user": 2, "assistant": 1}
    assert buckets["material_origins"] == {
        "human_authored": 1,
        "assistant_authored": 1,
        "runtime_protocol": 1,
    }


def test_archive_facet_aggregation_deduplicates_after_canonicalization() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE repos (
            repo_id TEXT PRIMARY KEY,
            origin_url TEXT,
            root_path TEXT,
            repo_name TEXT
        );
        CREATE TABLE session_repos (
            session_id TEXT,
            repo_id TEXT
        );
        CREATE TABLE messages (
            session_id TEXT,
            role TEXT,
            material_origin TEXT,
            message_type TEXT,
            has_tool_use INTEGER,
            has_thinking INTEGER,
            has_paste INTEGER
        );
        CREATE TABLE actions (
            session_id TEXT,
            semantic_type TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO repos(repo_id, origin_url, root_path, repo_name) VALUES (?, ?, ?, ?)",
        [
            ("repo-name", "", "/realm/tmp/worktrees/polylogue-a", "polylogue"),
            ("repo-url", "https://github.com/Sinity/polylogue.git", "/realm/project/polylogue", ""),
            ("repo-other", "https://github.com/Sinity/sinex.git", "/realm/project/sinex", ""),
        ],
    )
    conn.executemany(
        "INSERT INTO session_repos(session_id, repo_id) VALUES (?, ?)",
        [
            ("s1", "repo-name"),
            ("s1", "repo-url"),
            ("s2", "repo-url"),
            ("s3", "repo-other"),
        ],
    )

    buckets = _archive_aggregate_facet_families(conn, session_ids=None)

    assert buckets["repos"] == {"polylogue": 2, "sinex": 1}
