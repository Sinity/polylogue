"""Persisted inferred-origin receipts feed the real multi-origin corpus."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    rich_convergence_pathology,
)


def test_convergence_property_persisted_origins_and_unsupported_receipts(tmp_path: Path) -> None:
    corpus = rich_convergence_pathology()
    assert {member.provider for member in corpus.members} == {"codex", "chatgpt", "claude-ai"}
    assert all(member.receipt is not None for member in corpus.members[1:])
    archive = build_converged_archive(tmp_path / "archive", corpus)
    with sqlite3.connect(archive.root / "index.db") as conn:
        origins = {str(row[0]) for row in conn.execute("SELECT DISTINCT origin FROM sessions")}
        assert origins == {"codex-session", "chatgpt-export", "claude-ai-export"}
        assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] > 0

    replay = build_converged_archive(tmp_path / "replay", corpus)
    assert_archives_equivalent(archive, replay)
