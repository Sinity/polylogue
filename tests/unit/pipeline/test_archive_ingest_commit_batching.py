"""Work-based commit batching for the re-ingest path (parse_sources_archive).

Covers:
- (a) batched mode: a tiny message threshold commits in several batches yet the
  final archive contains every session and message.
- (b) per-session escape hatch (threshold <= 0) still ingests everything.
- (c) atomicity: a write that raises mid-batch rolls back the uncommitted batch
  so no partial rows from that batch survive, and rollback() is invoked.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.pipeline.services import archive_ingest
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import build_default_corpus_specs
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _build_sources(tmp_path: Path, *, count: int, seed: int = 7) -> list[Source]:
    available = set(SyntheticCorpus.available_providers())
    providers = [p for p in ("chatgpt", "claude-ai") if p in available]
    specs = build_default_corpus_specs(
        providers=providers,
        count=count,
        messages_min=4,
        messages_max=11,
        seed=seed,
    )
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    sources: list[Source] = []
    for spec in specs:
        provider_dir = corpus_dir / spec.provider
        written = SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="corpus")
        sources.extend(Source(name=spec.provider, path=file_path) for file_path in written.files)
    return sources


def _counts(index_db: Path) -> tuple[int, int]:
    conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    try:
        sessions = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    finally:
        conn.close()
    return sessions, messages


def _expected_session_count(sources: Sequence[Source]) -> int:
    # Each synthetic artifact in this corpus produces exactly one session.
    return len(sources)


def test_batched_commit_persists_all_sessions(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny threshold forces multiple batch commits; all data still lands."""
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "5")
    archive_root = workspace_env["archive_root"]
    sources = _build_sources(tmp_path, count=4)
    assert len(sources) >= 4  # multiple batches given >=4 msgs/session and threshold 5

    result = asyncio.run(parse_sources_archive(archive_root, sources))

    index_db = archive_root / "index.db"
    sessions, messages = _counts(index_db)
    assert sessions == _expected_session_count(sources)
    assert sessions == result.counts["sessions"]
    assert messages == result.counts["messages"]
    assert messages > 0


def test_per_session_escape_hatch(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """threshold <= 0 preserves per-session commit behavior and ingests all."""
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "0")
    archive_root = workspace_env["archive_root"]
    sources = _build_sources(tmp_path, count=3, seed=11)

    result = asyncio.run(parse_sources_archive(archive_root, sources))

    sessions, messages = _counts(archive_root / "index.db")
    assert sessions == _expected_session_count(sources)
    assert sessions == result.counts["sessions"]
    assert messages == result.counts["messages"]


def test_failed_write_rolls_back_uncommitted_batch(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-batch failure rolls back so no prior session in the batch survives."""
    # Large threshold so nothing commits until the (never reached) tail flush:
    # the whole run is one uncommitted batch.
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "1000000")
    archive_root = workspace_env["archive_root"]
    sources = _build_sources(tmp_path, count=4, seed=23)
    assert len(sources) >= 2

    original_write = ArchiveStore.write_raw_and_parsed
    calls = {"n": 0}

    def failing_write(self: ArchiveStore, *args: object, **kwargs: object) -> tuple[str, str]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated mid-batch write failure")
        return original_write(self, *args, **kwargs)  # type: ignore[arg-type]

    rollbacks = {"n": 0}
    original_rollback = ArchiveStore.rollback

    def spy_rollback(self: ArchiveStore) -> None:
        rollbacks["n"] += 1
        original_rollback(self)

    monkeypatch.setattr(ArchiveStore, "write_raw_and_parsed", failing_write)
    monkeypatch.setattr(ArchiveStore, "rollback", spy_rollback)

    with pytest.raises(RuntimeError, match="simulated mid-batch write failure"):
        asyncio.run(parse_sources_archive(archive_root, sources))

    assert calls["n"] == 2
    assert rollbacks["n"] == 1
    # The first session's index rows were written but never committed; rollback
    # must have discarded them so the archive carries no partial batch rows.
    sessions, messages = _counts(archive_root / "index.db")
    assert sessions == 0
    assert messages == 0


def test_invalid_env_falls_back_to_default_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "not-an-int")
    assert archive_ingest._commit_batch_message_threshold() == archive_ingest.COMMIT_BATCH_MESSAGE_THRESHOLD
    monkeypatch.delenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", raising=False)
    assert archive_ingest._commit_batch_message_threshold() == archive_ingest.COMMIT_BATCH_MESSAGE_THRESHOLD
