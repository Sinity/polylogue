"""Gemini CLI writes one session as several complete checkpoint files.

Two files under one ``~/.gemini/tmp/<project>/chats`` directory can carry the
same ``sessionId`` with different bytes -- neither is a prefix of the other,
so a byte-revision cohort over that logical key can only fail to order them.
This drives the real ``LiveBatchProcessor`` over that exact shape and proves
admission settles it: exactly one accepted head, and no raw left quarantined
without a receipt for the raw-frontier gate to trip over.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore

_SESSION_ID = "56cb9ec1-ece5-4705-ac40-a3e1b1569c25"


def _checkpoint(path: Path, *, kind: str, turns: int, started: str) -> None:
    """One synthetic Gemini CLI checkpoint document, no real transcript bytes."""
    path.write_text(
        json.dumps(
            {
                "sessionId": _SESSION_ID,
                "projectHash": "0f0f0f0f",
                "kind": kind,
                "startTime": started,
                "lastUpdated": started,
                "messages": [
                    {
                        "id": f"{kind}-{index}",
                        "type": "user" if index % 2 == 0 else "gemini",
                        "content": f"synthetic {kind} turn {index}",
                    }
                    for index in range(turns)
                ],
            }
        )
    )


@pytest.mark.asyncio
async def test_sibling_checkpoints_sharing_a_session_id_produce_one_accepted_head(
    workspace_env: dict[str, Path],
) -> None:
    """polylogue-6vacn: sibling checkpoints settle into one head, not an untyped quarantine.

    Anti-vacuity: reverting ``Origin.GEMINI_CLI_SESSION``'s declared
    ``frontier_kind`` to ``exact-prefix`` routes both files back through
    byte-revision cohort classification, which accepts neither -- the head
    count assertion reads 0 and the second raw is a ``quarantined`` row with
    no ``raw_artifacts`` and no ``raw_membership_census`` receipt, which is
    the rehearsal-4 state this test exists to forbid.
    """
    archive_root = workspace_env["archive_root"]
    chats_root = workspace_env["data_root"] / "gemini" / "tmp" / "project" / "chats"
    chats_root.mkdir(parents=True)
    archive = Polylogue(archive_root=archive_root, db_path=workspace_env["data_root"] / "gemini-live.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="gemini-cli", root=chats_root, suffixes=(".json", ".jsonl")),),
        cursor=CursorStore(archive_root / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        first = chats_root / "session-2026-04-02T10-09-56cb9ec1.json"
        second = chats_root / "session-2026-04-02T10-12-56cb9ec1.json"
        _checkpoint(first, kind="main", turns=5, started="2026-04-02T10:09:30.776Z")
        _checkpoint(second, kind="subagent", turns=13, started="2026-04-02T10:12:24.099Z")

        assert (await processor.ingest_files([first], emit_event=False)).failed_file_count == 0
        assert (await processor.ingest_files([second], emit_event=False)).failed_file_count == 0

        with sqlite3.connect(archive_root / "source.db") as conn:
            untyped = conn.execute(
                """
                SELECT r.raw_id
                FROM raw_sessions AS r
                LEFT JOIN raw_artifacts AS a ON a.raw_id = r.raw_id
                LEFT JOIN raw_membership_census AS c ON c.raw_id = r.raw_id
                WHERE r.source_path IN (?, ?)
                  AND r.revision_authority = 'quarantined'
                  AND a.raw_id IS NULL
                  AND c.raw_id IS NULL
                """,
                (str(first), str(second)),
            ).fetchall()
        # Every retained raw carries a receipt: a census, an artifact, or both.
        assert untyped == []

        with sqlite3.connect(archive_root / "index.db") as conn:
            heads = conn.execute(
                "SELECT COUNT(*) FROM raw_revision_heads WHERE logical_source_key = ?",
                (f"gemini-cli-session:{_SESSION_ID}",),
            ).fetchone()[0]
        assert heads == 1
    finally:
        await archive.close()
