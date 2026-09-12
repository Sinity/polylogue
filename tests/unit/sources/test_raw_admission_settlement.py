"""Gemini CLI may retain competing snapshots of one checkpoint identity.

Two files under one ``~/.gemini/tmp/<project>/chats`` directory can carry the
same wire identity with different bytes -- neither is a prefix of the other.
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
        second = chats_root / "session-2026-04-02T10-09-56cb9ec1-copy.json"
        _checkpoint(first, kind="main", turns=5, started="2026-04-02T10:09:30.776Z")
        _checkpoint(second, kind="main", turns=13, started="2026-04-02T10:09:30.776Z")

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
                (f"gemini-cli-session:{_SESSION_ID}:main:2026-04-02T10:09:30.776Z",),
            ).fetchone()[0]
        assert heads == 1
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_every_planned_path_is_accounted_for(
    workspace_env: dict[str, Path],
) -> None:
    """A planned path lands in exactly one of succeeded, failed, or excluded.

    A path that reaches none of them is invisible in the batch counters, which
    reads exactly like an idle source: a fresh-root run logged a thousand
    chunks of "ingesting 4 file(s)" followed by "succeeded=0 failed=0" with no
    reason recorded anywhere.

    Anti-vacuity: drop the ``excluded=excluded_paths`` argument from the
    ``_mark_excluded_cursor`` calls in ``_ingest_full_records`` and
    ``excluded_file_count`` reads 0 while ``needed_file_count`` stays 2 --
    the totals stop adding up, which is the final assertion below.
    """
    archive_root = workspace_env["archive_root"]
    chats_root = workspace_env["data_root"] / "gemini" / "tmp" / "project" / "chats"
    chats_root.mkdir(parents=True)
    archive = Polylogue(archive_root=archive_root, db_path=workspace_env["data_root"] / "gemini-account.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="gemini-cli", root=chats_root, suffixes=(".json", ".jsonl")),),
        cursor=CursorStore(archive_root / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        admissible = chats_root / "session-2026-04-02T12-00-bbbbbbbb.json"
        _checkpoint(admissible, kind="main", turns=2, started="2026-04-02T12:00:00.000Z")
        # Structurally valid JSON under a watched root that is no origin's
        # session: the acquisition loop quarantines it and moves on.
        inadmissible = chats_root / "notes.json"
        inadmissible.write_text(json.dumps({"unrelated": "content", "n": 1}))

        metrics = await processor.ingest_files([admissible, inadmissible], emit_event=False)

        assert metrics.needed_file_count == 2
        assert metrics.succeeded_file_count == 1
        assert metrics.excluded_file_count == 1
        assert metrics.excluded_reasons
        assert (
            metrics.succeeded_file_count + metrics.failed_file_count + metrics.excluded_file_count
            == metrics.needed_file_count
        )
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_offered_bytes_split_puts_a_refused_file_in_the_refused_bucket(
    workspace_env: dict[str, Path],
) -> None:
    """Offered bytes split into ingested, failed and refused with no remainder.

    ``input_bytes`` is what the batch was OFFERED. A file it planned and then
    declined counts there too, so a throughput figure computed from it credits
    the run with bytes it never ingested -- 6.83 GB offered against 2.09 GB
    ingested in one rehearsal, a 3.3x overstatement.

    Anti-vacuity: collapse the buckets -- credit the refused file's bytes to
    ``ingested_bytes``, or derive ``refused_bytes`` as a residual that a
    double-counted path can drive to zero -- and the per-reason assertion and
    the reconciliation below both go red.
    """
    archive_root = workspace_env["archive_root"]
    chats_root = workspace_env["data_root"] / "gemini" / "tmp" / "project" / "chats"
    chats_root.mkdir(parents=True)
    archive = Polylogue(archive_root=archive_root, db_path=workspace_env["data_root"] / "gemini-account.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="gemini-cli", root=chats_root, suffixes=(".json", ".jsonl")),),
        cursor=CursorStore(archive_root / "ops.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        admissible = chats_root / "session-2026-04-02T12-00-cccccccc.json"
        _checkpoint(admissible, kind="main", turns=2, started="2026-04-02T12:00:00.000Z")
        refused = chats_root / "notes.json"
        # Padded so the refused file is unmistakably the larger of the two:
        # a residual-derived refused bucket cannot be confused with rounding.
        refused.write_text(json.dumps({"unrelated": "content", "padding": "x" * 4096}))
        admissible_bytes = admissible.stat().st_size
        refused_bytes = refused.stat().st_size

        metrics = await processor.ingest_files([admissible, refused], emit_event=False)

        assert metrics.input_bytes == admissible_bytes + refused_bytes
        assert metrics.ingested_bytes == admissible_bytes
        assert metrics.refused_bytes == refused_bytes
        assert sum(metrics.refused_bytes_by_reason.values()) == refused_bytes
        # The refusal carries its typed reason, not just a byte count.
        assert set(metrics.refused_bytes_by_reason) == set(metrics.excluded_reasons)
        assert metrics.unaccounted_bytes == 0

        payload = metrics.to_payload()
        assert payload["ingested_bytes"] == admissible_bytes
        assert payload["refused_bytes"] == refused_bytes
        assert payload["unaccounted_bytes"] == 0
        # Throughput is computed from what was ingested, never from what was
        # offered: the two differ here by the whole refused file.
        assert payload["ingested_mb_per_second"] != pytest.approx(
            (metrics.input_bytes / 1_000_000) / metrics.total_time_s
        )
    finally:
        await archive.close()
