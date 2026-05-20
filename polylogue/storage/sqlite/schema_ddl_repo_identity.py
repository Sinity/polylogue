"""Repo / cwd identity DDL fragments (#1253, slice C of #864).

Graduates ``cwd`` / ``repo`` / ``git origin`` context out of provider-meta
JSON into typed storage. Provides a canonical join surface for repo-scoped
queries such as "show me every conversation that touched repo X" across
providers (Claude Code, Codex, ChatGPT, ...).

Two tables:

* ``repo_identities`` — canonical (origin_url, root_path) tuple with a
  display ``repo_name``. ``origin_url`` may be empty when only a local
  root path was observed; the unique key includes both columns so the
  same root path can carry distinct origins over its lifetime.
* ``conversation_repo_observations`` — many-to-many association from a
  conversation to the repos it touched, with the most recent branch
  name and observation timestamp.

Population: see ``polylogue/storage/insights/session/repo_identity.py``.
These rows live OUTSIDE the conversation content-hash boundary — they
are derived per ingest from action events plus typed conversation
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

        CREATE TABLE IF NOT EXISTS conversation_repo_observations (
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            repo_identity_id INTEGER NOT NULL REFERENCES repo_identities(id) ON DELETE CASCADE,
            branch_name TEXT NOT NULL DEFAULT '',
            observed_at TEXT NOT NULL,
            PRIMARY KEY (conversation_id, repo_identity_id)
        );

        CREATE INDEX IF NOT EXISTS idx_conv_repo_obs_repo
        ON conversation_repo_observations(repo_identity_id, conversation_id);
"""

__all__ = ["REPO_IDENTITY_DDL"]
