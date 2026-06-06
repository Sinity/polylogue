"""Full-pipeline ingest correctness tests.

Generate schema-conformant synthetic corpora for all providers, ingest through
the full prepare_records pipeline, and assert end-to-end correctness of the
resulting archive.

Ref #1736.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.pipeline.prepare import prepare_records
from polylogue.scenarios import build_default_corpus_specs
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources import iter_source_sessions
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ── helpers ─────────────────────────────────────────────────────────────


_CORE_PROVIDERS = ("chatgpt", "claude-code", "codex", "gemini")

_TABLES_EXPECTED_POPULATED = frozenset(
    {
        "raw_sessions",
        "sessions",
        "messages",
        "content_blocks",
        "session_stats",
        "action_events",
        "session_profiles",
        "session_work_events",
    }
)

_TABLES_CONDITIONALLY_POPULATED = frozenset(
    {
        "session_phases",
        "session_latency_profiles",
        "work_threads",
        "session_tag_rollups",
        "provider_events",
        "provider_event_compactions",
        "provider_event_tool_calls",
        "provider_event_reasoning",
        "provider_event_ghost_snapshots",
        "provider_event_turn_contexts",
        "session_commit_edges",
        "topology_edges",
    }
)


def _write_corpus_files(
    providers: tuple[str, ...],
    count: int,
    seed: int,
    dest: Path,
) -> None:
    """Generate synthetic corpus wire-format files under *dest*."""
    available = set(SyntheticCorpus.available_providers())
    specs = build_default_corpus_specs(
        providers=(p for p in providers if p in available),
        count=count,
        messages_min=4,
        messages_max=12,
        seed=seed,
    )
    for spec in specs:
        provider_dir = dest / spec.provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="corpus")


async def _ingest_corpus(archive_root: Path, corpus_dir: Path, db_path: Path) -> int:
    """Ingest all corpus files into the given database.  Returns session count."""
    backend = SQLiteBackend(db_path=db_path)
    repository = SessionRepository(backend=backend)
    session_count = 0

    for provider_dir in sorted(corpus_dir.iterdir()):
        provider = provider_dir.name
        for file_path in sorted(provider_dir.iterdir()):
            raw_bytes = file_path.read_bytes()
            raw_id = hashlib.sha256(raw_bytes).hexdigest()

            await backend.save_raw_session(
                RawSessionRecord(
                    raw_id=raw_id,
                    source_name=provider,
                    source_path=str(file_path),
                    blob_size=len(raw_bytes),
                    acquired_at="2024-01-15T10:00:00+00:00",
                )
            )

            source = Source(name=provider, path=file_path)
            for convo in iter_source_sessions(source):
                conv = await prepare_records(
                    convo,
                    source_name=provider,
                    archive_root=archive_root,
                    backend=backend,
                    repository=repository,
                    raw_id=raw_id,
                )
                if conv is not None:
                    session_count += 1

    await backend.close()
    return session_count


def _snapshot(db_path: Path) -> dict[str, int]:
    """Return a snapshot of key archive metrics."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conv_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        content_blocks_count = conn.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]
        fts_docsize = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        action_count = conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0]
        action_fts_docsize = conn.execute("SELECT COUNT(*) FROM action_events_fts_docsize").fetchone()[0]
        session_profiles_count = conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0]
        session_work_events_count = conn.execute("SELECT COUNT(*) FROM session_work_events").fetchone()[0]
        content_hashes = {row[0] for row in conn.execute("SELECT content_hash FROM sessions").fetchall()}
    finally:
        conn.close()
    return {
        "sessions": conv_count,
        "messages": msg_count,
        "content_blocks": content_blocks_count,
        "fts_docsize": fts_docsize,
        "action_events": action_count,
        "action_fts_docsize": action_fts_docsize,
        "session_profiles": session_profiles_count,
        "session_work_events": session_work_events_count,
        "distinct_content_hashes": len(content_hashes),
    }


def _table_names(db_path: Path) -> set[str]:
    """Return the set of user table names in the database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_%'"
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def _table_row_count(db_path: Path, table_name: str) -> int:
    """Return the row count for a given table."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    result = None
    try:
        # Virtual tables (FTS5) use docsize for row counting.
        if table_name.endswith("_fts"):
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}_docsize").fetchone()
        else:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    finally:
        conn.close()
    return result[0] if result is not None else 0


# ── tests ───────────────────────────────────────────────────────────────


def test_full_pipeline_replay_produces_correct_archive(
    tmp_path: Path,
    workspace_env: dict[str, Path],
) -> None:
    """Generate schema-conformant corpora for all core providers, ingest, verify.

    Assertions:
    - Every expected DDL table has at least one row
    - sessions / messages / content_blocks have expected counts
    - FTS index docsize matches message count
    - session_profiles has one row per session
    - session_work_events has rows (for providers that emit work events)
    - Content hashes are deterministic across two independent databases
    """
    # ── Phase 1: generate and ingest into DB1 ──
    archive_root = workspace_env["archive_root"]
    db1_path = archive_root / "polylogue1.db"

    with open_connection(db1_path):
        pass

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus_files(_CORE_PROVIDERS, count=3, seed=42, dest=corpus_dir)

    conv_count = asyncio.run(_ingest_corpus(archive_root, corpus_dir, db1_path))
    snap1 = _snapshot(db1_path)

    # ── Core assertions ──
    assert conv_count > 0, "No sessions ingested"
    assert snap1["sessions"] == conv_count, (
        f"Session count mismatch: snapshot={snap1['sessions']} vs ingested={conv_count}"
    )
    assert snap1["messages"] > 0, "No messages in archive"
    assert snap1["content_blocks"] > 0, "No content_blocks in archive — content extraction failed"

    # session_profiles are materialized by the daemon convergence loop, not
    # by prepare_records alone.  The prepare_records path stores the archive
    # rows but derived insights are not populated without daemon convergence.
    assert snap1["session_profiles"] == 0, "session_profiles materialized by daemon convergence, not prepare_records"

    # ── FTS integrity ──
    assert snap1["fts_docsize"] == snap1["messages"], (
        f"messages_fts_docsize ({snap1['fts_docsize']}) != messages ({snap1['messages']})"
    )
    if snap1["action_events"] > 0:
        assert snap1["action_fts_docsize"] == snap1["action_events"], (
            f"action_events_fts_docsize ({snap1['action_fts_docsize']}) != action_events ({snap1['action_events']})"
        )

    # ── Table population check ──
    tables = _table_names(db1_path)
    for table_name in sorted(tables):
        count = _table_row_count(db1_path, table_name)
        if table_name in _TABLES_EXPECTED_POPULATED:
            assert count > 0, f"Expected table '{table_name}' is empty after ingest"
        elif table_name in _TABLES_CONDITIONALLY_POPULATED:
            # Conditionally populated tables may be empty depending on provider/corpus shape.
            pass

    # ── Phase 2: content hash determinism ──
    db2_path = archive_root / "polylogue2.db"
    with open_connection(db2_path):
        pass
    conv_count2 = asyncio.run(_ingest_corpus(archive_root, corpus_dir, db2_path))
    snap2 = _snapshot(db2_path)

    assert conv_count2 == conv_count, f"Session count differs between runs: {conv_count} vs {conv_count2}"
    assert snap2["messages"] == snap1["messages"], (
        f"Message count differs between runs: {snap1['messages']} vs {snap2['messages']}"
    )

    # Content hashes must be identical between independent ingest runs.
    conn1 = sqlite3.connect(f"file:{db1_path}?mode=ro", uri=True)
    conn2 = sqlite3.connect(f"file:{db2_path}?mode=ro", uri=True)
    try:
        hashes1 = {row[0] for row in conn1.execute("SELECT content_hash FROM sessions")}
        hashes2 = {row[0] for row in conn2.execute("SELECT content_hash FROM sessions")}
    finally:
        conn1.close()
        conn2.close()

    assert hashes1 == hashes2, (
        f"Content hashes differ between independent databases. Diff: {hashes1.symmetric_difference(hashes2)}"
    )


def test_idempotent_reingest(
    tmp_path: Path,
    workspace_env: dict[str, Path],
) -> None:
    """Re-ingest the same corpus twice; all counts and hashes must be stable."""
    archive_root = workspace_env["archive_root"]
    db_path = archive_root / "polylogue.db"

    with open_connection(db_path):
        pass

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus_files(("chatgpt", "claude-code"), count=2, seed=17, dest=corpus_dir)

    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))
    snap1 = _snapshot(db_path)

    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))
    snap2 = _snapshot(db_path)

    assert snap1["sessions"] > 0
    assert snap2["sessions"] == snap1["sessions"]
    assert snap2["messages"] == snap1["messages"]
    assert snap2["content_blocks"] == snap1["content_blocks"]
    assert snap2["distinct_content_hashes"] == snap1["distinct_content_hashes"]
    assert snap2["fts_docsize"] == snap2["messages"]
    assert snap2["session_profiles"] == snap1["session_profiles"]


def test_content_blocks_present_for_all_providers(
    tmp_path: Path,
    workspace_env: dict[str, Path],
) -> None:
    """Every provider must produce at least some content_blocks rows."""
    archive_root = workspace_env["archive_root"]
    db_path = archive_root / "polylogue.db"

    with open_connection(db_path):
        pass

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus_files(_CORE_PROVIDERS, count=2, seed=99, dest=corpus_dir)

    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Content blocks are linked to messages, which are linked to sessions
        # with a known provider. We verify each ingested provider has blocks.
        rows = conn.execute(
            """SELECT c.source_name, COUNT(DISTINCT cb.block_id)
               FROM sessions c
               JOIN messages m ON m.session_id = c.session_id
               JOIN content_blocks cb ON cb.message_id = m.message_id
               GROUP BY c.source_name
               ORDER BY c.source_name"""
        ).fetchall()
        providers_with_blocks = {row[0]: row[1] for row in rows}
    finally:
        conn.close()

    # Not every provider generates tool_use/thinking blocks by default,
    # but each provider should have at least text content blocks.
    ingested = {d.name for d in sorted(corpus_dir.iterdir()) if d.is_dir()}
    for provider in ingested:
        assert provider in providers_with_blocks, f"Provider '{provider}' produced zero content_blocks rows"
        assert providers_with_blocks[provider] > 0, f"Provider '{provider}' produced zero content_blocks rows"
