"""Real end-to-end proof of Hermes source coverage through the existing
named-source freshness surface (polylogue-1xc.13), plus a confirmed live-
ingestion bug discovered while proving it.

fs1.15 ("Expose Hermes-to-Polylogue integration liveness and coverage") asks
for coverage/freshness reporting to "flow through the EXISTING named-source
freshness surface... extend that projection with the Hermes source classes
rather than inventing a reporting command." ``project_named_source_freshness``
is entirely origin-agnostic (keyed by exact ``source_path`` string, queries
cursor/raw-revision/index/fts/insight evidence generically) -- the tests
below prove empirically, by driving the real ``LiveBatchProcessor`` and then
the real freshness projection, that Hermes's state.db and ATOF sources are
already covered for free once ingestion succeeds, closing that part of
fs1.15's ask without new code.

Driving state.db through the real live watcher (not the marker-payload/CLI
route the rest of this repo's Hermes tests use) surfaced a second confirmed
bug, distinct from polylogue-flxh: ``revision_backfill.py``'s ``_parse_one``
(used by the live watcher's single-session raw-revision-chain replay branch,
and shared with historical repair) has zero SQLite awareness -- unlike the
two OTHER call sites in ``live/batch.py`` that correctly special-case Hermes
SQLite sources before falling back to JSON parsing. A state.db (or
verification_evidence.db) with exactly one session at ingest time hits this
branch and crashes; two or more sessions route through the working
membership-census branch instead.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.archive.query.source_freshness import NamedSourceStage, project_named_source_freshness
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore

_STATE_DB_SCHEMA = """
CREATE TABLE schema_version(version INTEGER NOT NULL);
INSERT INTO schema_version(version) VALUES (19);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    model_config TEXT,
    parent_session_id TEXT,
    started_at REAL,
    ended_at REAL,
    end_reason TEXT,
    title TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL,
    tool_calls TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1,
    compacted INTEGER DEFAULT 0
);
"""


def _make_processor(
    workspace_env: dict[str, Path], root_name: str, db_name: str
) -> tuple[Polylogue, LiveBatchProcessor, Path]:
    root = workspace_env["data_root"] / root_name
    root.mkdir(parents=True)
    db_path = workspace_env["data_root"] / db_name
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="hermes", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    return archive, processor, root


@pytest.mark.xfail(
    reason=(
        "Confirmed live-ingestion bug polylogue-1zex (distinct from polylogue-flxh): "
        "revision_backfill.py's _parse_one (used by live/batch.py's single-"
        "session raw-revision-chain replay branch at _parse_raw_revision_chain, "
        "shared with historical repair) has zero SQLite awareness -- unlike "
        "the two other Hermes call sites in live/batch.py that correctly check "
        "hermes_state.looks_like_state_db_path / "
        "hermes_verification.looks_like_verification_evidence_db_path before "
        "falling back to JSON parsing. A state.db (or verification_evidence.db) "
        "with exactly one session at ingest time -- a brand new Hermes install, "
        "or any minimal/test fixture -- takes this branch and crashes with "
        "UnicodeDecodeError trying to json-parse raw SQLite bytes. Two or more "
        "sessions route through the working membership-census branch instead "
        "(see test_hermes_state_db_multi_session_source_reaches_indexed_"
        "through_named_freshness below), which is presumably why this was not "
        "caught by prior review -- real installs almost always have many "
        "sessions. Remove this xfail once fixed."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_hermes_state_db_single_session_full_ingest_crashes(
    workspace_env: dict[str, Path],
) -> None:
    archive, processor, root = _make_processor(workspace_env, "hermes-home-single", "hermes-state-single.db")
    source_path = root / "state.db"
    try:
        with sqlite3.connect(source_path) as conn:
            conn.executescript(_STATE_DB_SCHEMA)
            conn.execute(
                "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
                "VALUES ('root', 'cli', '{}', 1.0, 8.0, 'completed', 'root')"
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, 'root', 'user', 'hi', 2.0)"
            )

        metrics = await processor.ingest_files([source_path], emit_event=False)
        assert metrics.failed_file_count == 0
        assert metrics.ingested_session_count == 1
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_hermes_state_db_multi_session_source_reaches_indexed_through_named_freshness(
    workspace_env: dict[str, Path],
) -> None:
    """Two or more sessions dodge the single-session bug above (membership-
    census branch, not the broken raw-revision-chain replay branch) -- this
    proves the EXISTING named-source freshness surface already covers Hermes
    state.db correctly once ingestion itself succeeds."""

    archive, processor, root = _make_processor(workspace_env, "hermes-home-multi", "hermes-state-multi.db")
    source_path = root / "state.db"
    try:
        with sqlite3.connect(source_path) as conn:
            conn.executescript(_STATE_DB_SCHEMA)
            conn.execute(
                "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
                "VALUES ('root', 'cli', '{}', 1.0, 8.0, 'completed', 'root')"
            )
            conn.execute(
                "INSERT INTO sessions (id, source, model_config, parent_session_id, started_at, ended_at, "
                "end_reason, title) VALUES ('child', 'cli', '{}', 'root', 2.0, 7.0, 'completed', 'child')"
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, 'root', 'user', 'hi', 2.0)"
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) "
                "VALUES (2, 'child', 'user', 'hi2', 3.0)"
            )

        metrics = await processor.ingest_files([source_path], emit_event=False)
        assert metrics.failed_file_count == 0
        assert metrics.ingested_session_count == 2

        freshness = project_named_source_freshness(workspace_env["archive_root"], source_path)
        assert freshness.stage in {NamedSourceStage.INDEXED_UNCONVERGED, NamedSourceStage.SEARCHABLE}
        assert freshness.accepted_raw_revision is not None
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_hermes_atof_source_reaches_indexed_through_named_freshness(
    workspace_env: dict[str, Path],
) -> None:
    _archive, processor, root = _make_processor(workspace_env, "hermes-observability", "hermes-atof-freshness.db")
    source_path = root / "events.jsonl"
    archive = _archive
    try:
        record = {
            "atof_version": "0.1",
            "kind": "mark",
            "uuid": "turn-1",
            "timestamp": "2026-07-18T09:00:00Z",
            "name": "hermes.turn.start",
            "metadata": {"session_id": "atof-freshness-session"},
        }
        source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        metrics = await processor.ingest_files([source_path], emit_event=False)
        assert metrics.failed_file_count == 0
        assert metrics.ingested_session_count == 1

        freshness = project_named_source_freshness(workspace_env["archive_root"], source_path)
        assert freshness.stage in {NamedSourceStage.INDEXED_UNCONVERGED, NamedSourceStage.SEARCHABLE}
        assert freshness.accepted_raw_revision is not None
    finally:
        await archive.close()
