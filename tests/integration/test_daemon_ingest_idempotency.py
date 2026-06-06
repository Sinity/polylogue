"""Daemon ingest idempotency integration test.

Integration test that ingests the same source files twice through the
pipeline and asserts:

- Session count unchanged
- Message count unchanged
- Content hashes identical across both passes
- No duplicate FTS rows (messages_fts_docsize matches messages count)

Ref #1722 item 10.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.scenarios import build_default_corpus_specs
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources import iter_source_sessions
from polylogue.storage.repository import SessionRepository
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ── helpers ─────────────────────────────────────────────────────────


def _write_corpus_files(providers: tuple[str, ...], count: int, seed: int, dest: Path) -> list[Path]:
    """Generate synthetic corpus wire-format files and return their paths."""

    available = set(SyntheticCorpus.available_providers())
    specs = build_default_corpus_specs(
        providers=(p for p in providers if p in available),
        count=count,
        messages_min=4,
        messages_max=8,
        seed=seed,
    )
    written_files: list[Path] = []
    for spec in specs:
        provider = spec.provider
        provider_dir = dest / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        result = SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="corpus")
        written_files.extend((provider_dir / f) for f in result.files)
    return written_files


async def _ingest_corpus(archive_root: Path, corpus_dir: Path, db_path: Path) -> None:
    """Ingest all corpus files into the given database."""
    from polylogue.config import Source
    from polylogue.pipeline.prepare import prepare_records

    backend = SQLiteBackend(db_path=db_path)
    repository = SessionRepository(backend=backend)

    for provider_dir in sorted(corpus_dir.iterdir()):
        provider = provider_dir.name
        for file_path in sorted(provider_dir.iterdir()):
            raw_bytes = file_path.read_bytes()
            raw_id = hashlib.sha256(raw_bytes).hexdigest()

            # Write raw record if not present.
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
                await prepare_records(
                    convo,
                    source_name=provider,
                    archive_root=archive_root,
                    backend=backend,
                    repository=repository,
                    raw_id=raw_id,
                )

    await backend.close()


def _snapshot(db_path: Path) -> dict[str, int]:
    """Return a snapshot of key archive metrics."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conv_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        fts_docsize = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        content_hashes = {row[0] for row in conn.execute("SELECT content_hash FROM sessions").fetchall()}
        action_count = conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0]
        action_fts_docsize = conn.execute("SELECT COUNT(*) FROM action_events_fts_docsize").fetchone()[0]
    finally:
        conn.close()
    return {
        "sessions": conv_count,
        "messages": msg_count,
        "fts_docsize": fts_docsize,
        "distinct_content_hashes": len(content_hashes),
        "action_events": action_count,
        "action_fts_docsize": action_fts_docsize,
    }


# ── test ────────────────────────────────────────────────────────────


def test_double_ingest_is_idempotent(
    tmp_path: Path,
    workspace_env: dict[str, Path],
) -> None:
    """Ingest same corpus twice; verify all counts and hashes are stable."""
    archive_root = workspace_env["archive_root"]
    db_path = archive_root / "polylogue.db"

    # Initialize schema.
    with open_connection(db_path):
        pass

    # Generate corpus files.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus_files(("chatgpt",), count=2, seed=42, dest=corpus_dir)

    # First ingest.
    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))
    snap1 = _snapshot(db_path)

    # Second ingest — same files, same DB.
    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))
    snap2 = _snapshot(db_path)

    # Assertions.
    assert snap1["sessions"] > 0, "No sessions after first ingest"
    assert snap1["messages"] > 0, "No messages after first ingest"

    assert snap2["sessions"] == snap1["sessions"], f"Session count changed: {snap1['sessions']} → {snap2['sessions']}"
    assert snap2["messages"] == snap1["messages"], f"Message count changed: {snap1['messages']} → {snap2['messages']}"
    assert snap2["distinct_content_hashes"] == snap1["distinct_content_hashes"], (
        "Content hashes changed between ingest passes"
    )

    # FTS integrity: docsize must match message count (no duplicates).
    assert snap1["fts_docsize"] == snap1["messages"], (
        f"First pass: FTS docsize {snap1['fts_docsize']} != messages {snap1['messages']}"
    )
    assert snap2["fts_docsize"] == snap2["messages"], (
        f"Second pass: FTS docsize {snap2['fts_docsize']} != messages {snap2['messages']}"
    )
    assert snap2["fts_docsize"] == snap1["fts_docsize"], "FTS docsize changed between passes"

    # Action events integrity (if any).
    if snap1["action_events"] > 0:
        assert snap2["action_events"] == snap1["action_events"], (
            f"Action event count changed: {snap1['action_events']} → {snap2['action_events']}"
        )
        assert snap2["action_fts_docsize"] == snap2["action_events"], "Action FTS docsize mismatch after second pass"


def test_triple_ingest_is_idempotent(
    tmp_path: Path,
    workspace_env: dict[str, Path],
) -> None:
    """Three ingest passes — stronger check against non-deterministic drift."""
    archive_root = workspace_env["archive_root"]
    db_path = archive_root / "polylogue.db"

    with open_connection(db_path):
        pass

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus_files(("chatgpt",), count=2, seed=99, dest=corpus_dir)

    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))
    snap1 = _snapshot(db_path)

    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))
    snap2 = _snapshot(db_path)

    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db_path))
    snap3 = _snapshot(db_path)

    assert snap2["sessions"] == snap1["sessions"]
    assert snap3["sessions"] == snap1["sessions"]
    assert snap3["messages"] == snap1["messages"]
    assert snap3["fts_docsize"] == snap3["messages"]


def test_content_hashes_are_deterministic(
    tmp_path: Path,
    workspace_env: dict[str, Path],
) -> None:
    """Content hashes must be identical between two independent ingest runs."""
    # Two independent DBs, same corpus → same content hashes.
    archive_root = workspace_env["archive_root"]
    db1_path = archive_root / "polylogue1.db"
    db2_path = archive_root / "polylogue2.db"

    with open_connection(db1_path):
        pass
    with open_connection(db2_path):
        pass

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus_files(("chatgpt", "claude-ai"), count=2, seed=17, dest=corpus_dir)

    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db1_path))
    asyncio.run(_ingest_corpus(archive_root, corpus_dir, db2_path))

    # Read content hashes from both.
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
