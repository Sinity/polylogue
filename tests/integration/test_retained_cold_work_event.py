"""Original retained transcript and work-event cold rebuild state parity."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus


@pytest.mark.asyncio
async def test_cold_build_rebuilds_a_session_that_retains_an_agent_work_event(tmp_path: Path) -> None:
    """Cold replay preserves the transcript, event, and accepted head together.

    The complete state comparison detects a lost event or changed transcript.
    The head agreement detects event publication replacing transcript authority.
    """
    root = tmp_path / "archive"

    def acquire() -> tuple[str, ...]:
        build_independent_raw_corpus(root, raw_count=1, avg_payload_bytes=1_000, authoritative_source=True)
        with closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as source:
            return tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))

    transcript_raw_ids = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        (await owner.replay_retained_raw_ids(transcript_raw_ids)).require_complete()
    session_id = "codex-session:amg1-session-000000"

    def append_event() -> dict[str, object]:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            appended = archive.admit_work_event(
                session_id=session_id,
                event_type="decision",
                payload={"decision": "continue"},
                event_id="evt-cold-build",
                summary="kept going",
            )
        return appended

    appended = await run_archive_fixture_write(root, append_event)
    async with prepared_live_convergence_owner(root) as owner:
        appended_receipts = (await owner.replay_retained_raw_ids((str(appended["raw_id"]),))).require_complete()
    assert any(receipt.changed_session_ids for receipt in appended_receipts), appended_receipts

    def session_state(
        index_path: Path,
    ) -> tuple[list[Any], list[Any], list[tuple[str, str]], list[Any]]:
        with sqlite3.connect(index_path) as conn:
            header = conn.execute(
                "SELECT session_id, title, created_at_ms, updated_at_ms FROM sessions ORDER BY session_id"
            ).fetchall()
            messages = conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position", (session_id,)
            ).fetchall()
            events = conn.execute(
                "SELECT event_type, payload_json FROM session_events WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
            head_agreement = conn.execute(
                """
                SELECT h.logical_source_key, s.raw_id = h.accepted_raw_id AND s.content_hash = h.accepted_content_hash
                FROM raw_revision_heads AS h JOIN sessions AS s ON s.session_id = h.session_id
                ORDER BY h.logical_source_key
                """
            ).fetchall()
        return (
            header,
            messages,
            [(event_type, json.loads(payload)["event_id"]) for event_type, payload in events],
            head_agreement,
        )

    active = session_state(root / "index.db")
    assert active[0][0][0] == session_id
    assert len(active[1]) == 1
    assert active[2] == [("decision", "evt-cold-build")]
    assert active[3] == [(session_id, 1)]

    def begin() -> ColdBuildGeneration:
        return ColdBuildGeneration.begin(
            root,
            reason="work-event-cold-build",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent"),)),
        )

    generation = await run_archive_fixture_write(root, begin)
    with closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as source:
        raw_ids = tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))
    register_cold_build_generation(generation)
    try:
        async with prepared_live_convergence_owner(root) as owner:
            results = (await owner.replay_retained_raw_ids(raw_ids)).require_complete()
        await run_archive_fixture_write(root, generation.prepare_promotion_candidate)
        assert sum(result.replayed_logical_sources for result in results) == 2
        assert session_state(Path(generation.generation.index_path)) == active
    finally:
        clear_cold_build_generation()
        await run_archive_fixture_write(root, generation.discard)
