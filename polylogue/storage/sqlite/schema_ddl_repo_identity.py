"""Repo / cwd identity DDL fragments (#1253, slice C of #864).

Graduates ``cwd`` / ``repo`` / ``git origin`` context out of provider-meta
JSON into typed storage. Provides a canonical join surface for repo-scoped
queries such as "show me every session that touched repo X" across
providers (Claude Code, Codex, ChatGPT, ...).

Three tables:

* ``repo_identities`` — canonical (origin_url, root_path) tuple with a
  display ``repo_name``. ``origin_url`` may be empty when only a local
  root path was observed; the unique key includes both columns so the
  same root path can carry distinct origins over its lifetime.
* ``session_repo_observations`` — many-to-many association from a
  session to the repos it touched, with the most recent branch
  name and observation timestamp.
* ``session_commit_edges`` — links archived sessions to git commits they
  likely produced (#1690 phase 2). Each edge records the detection method,
  confidence score, and file-overlap count. Rows live outside the content-
  hash boundary.

Population: see ``polylogue/storage/insights/session/repo_identity.py``
and ``polylogue/insights/session_commit.py``.
These rows live OUTSIDE the session content-hash boundary — they
are derived per ingest from action events plus typed session
columns, and rebuilt deterministically.
"""

from __future__ import annotations

REPO_IDENTITY_DDL = """
        CREATE TABLE IF NOT EXISTS repo_identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_url TEXT NOT NULL DEFAULT '',
            root_path TEXT NOT NULL,
            repo_name TEXT NOT NULL DEFAULT '',
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            UNIQUE (origin_url, root_path)
        );

        CREATE INDEX IF NOT EXISTS idx_repo_identities_repo_name
        ON repo_identities(repo_name)
        WHERE repo_name != '';

        CREATE INDEX IF NOT EXISTS idx_repo_identities_root_path
        ON repo_identities(root_path);

        CREATE TABLE IF NOT EXISTS session_repo_observations (
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            repo_identity_id INTEGER NOT NULL REFERENCES repo_identities(id) ON DELETE CASCADE,
            branch_name TEXT NOT NULL DEFAULT '',
            observed_at TEXT NOT NULL,
            PRIMARY KEY (session_id, repo_identity_id)
        );

        CREATE INDEX IF NOT EXISTS idx_conv_repo_obs_repo
        ON session_repo_observations(repo_identity_id, session_id);

        CREATE TABLE IF NOT EXISTS session_commit_edges (
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            commit_sha TEXT NOT NULL,
            repo_id INTEGER REFERENCES repo_identities(id) ON DELETE CASCADE,
            detection_method TEXT NOT NULL CHECK (
                detection_method IN ('time_window', 'file_overlap', 'explicit_ref')
            ),
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            file_overlap_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, commit_sha)
        );

        CREATE INDEX IF NOT EXISTS idx_session_commit_edges_repo
        ON session_commit_edges(repo_id, session_id);

        CREATE INDEX IF NOT EXISTS idx_session_commit_edges_confidence
        ON session_commit_edges(confidence DESC);
"""

__all__ = ["REPO_IDENTITY_DDL"]
