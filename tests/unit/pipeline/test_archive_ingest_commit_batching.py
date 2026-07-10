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
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.config import Source
from polylogue.pipeline.services import archive_ingest
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import build_default_corpus_specs
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.blob_gc import BlobGCResult, run_blob_gc_report
from polylogue.storage.blob_publication import ArchiveBlobPublisher, BlobPublicationReceipt
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveRawParsedWriteResult, ArchiveStore
from polylogue.storage.sqlite.maintenance import SqliteOptimizeObservation
from polylogue.storage.sqlite.wal_checkpoint import WalCheckpointObservation


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


def test_direct_grouped_reingest_reserves_raw_blob_until_source_commit(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYLOGUE_INGEST_PARSE_WORKERS", "1")
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "0")
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "session.jsonl"
    source_path.write_text(
        '{"type":"user","uuid":"u1","sessionId":"s1","message":{"content":"hello"}}\n'
        '{"type":"assistant","uuid":"a1","sessionId":"s1","message":{"content":"hi"}}\n',
        encoding="utf-8",
    )
    gc_reports: list[BlobGCResult] = []
    original_flush = ArchiveBlobPublisher.flush

    def flush_then_gc(publisher: ArchiveBlobPublisher) -> tuple[BlobPublicationReceipt, ...]:
        receipts = original_flush(publisher)
        if receipts and not gc_reports:
            os.utime(publisher.blob_path(receipts[0].blob_hash), (1_700_000_000, 1_700_000_000))
            gc_reports.append(run_blob_gc_report(archive_root / "source.db", archive_root / "blob"))
        return receipts

    monkeypatch.setattr(ArchiveBlobPublisher, "flush", flush_then_gc)
    result = asyncio.run(parse_sources_archive(archive_root, [Source(name="claude-code", path=source_path)]))

    assert result.counts["sessions"] == 1
    assert len(gc_reports) == 1
    assert gc_reports[0].deleted_count == 0
    assert gc_reports[0].skipped_reserved == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
    final_gc = run_blob_gc_report(archive_root / "source.db", archive_root / "blob")
    assert final_gc.deleted_count == 0
    assert final_gc.skipped_reserved == 0
    assert final_gc.skipped_referenced >= 1


def test_archive_ingest_raw_payload_uses_explicit_archive_blob_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "explicit-archive"
    ambient_blob_root = tmp_path / "ambient-xdg" / "blob"
    blob_hash, _blob_size = BlobStore(archive_root / "blob").write_from_bytes(b'{"mapping":{"from":"archive"}}')
    raw_data = SimpleNamespace(raw_bytes=b"", blob_hash=blob_hash)
    session = SimpleNamespace(model_dump_json=lambda: '{"fallback": true}')

    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: ambient_blob_root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: ambient_blob_root, raising=False)

    assert archive_ingest._archive_raw_payload(raw_data, session, blob_root=archive_root / "blob") == (
        b'{"mapping":{"from":"archive"}}'
    )


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

    original_write = ArchiveStore.write_raw_and_parsed_result
    calls = {"n": 0}

    def failing_write(self: ArchiveStore, *args: object, **kwargs: object) -> ArchiveRawParsedWriteResult:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated mid-batch write failure")
        return original_write(self, *args, **kwargs)  # type: ignore[arg-type]

    rollbacks = {"n": 0}
    original_rollback = ArchiveStore.rollback

    def spy_rollback(self: ArchiveStore) -> None:
        rollbacks["n"] += 1
        original_rollback(self)

    monkeypatch.setattr(ArchiveStore, "write_raw_and_parsed_result", failing_write)
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


def test_batched_archive_ingest_runs_post_commit_upkeep(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "5")
    archive_root = workspace_env["archive_root"]
    sources = _build_sources(tmp_path, count=4, seed=31)
    wal_calls: list[Path] = []
    optimize_calls: list[Path] = []

    def fake_checkpoint_archive_wals(
        root: Path,
        *,
        reason: str,
        allow_truncate: bool = True,
        **_: object,
    ) -> tuple[WalCheckpointObservation, ...]:
        assert root == archive_root
        assert reason == archive_ingest.POST_COMMIT_UPKEEP_REASON
        assert allow_truncate is False
        wal_calls.append(root)
        return (
            WalCheckpointObservation(
                reason=reason,
                mode="passive",
                wal_bytes_before=12,
                wal_bytes_after=4,
                checkpointed_pages=2,
            ),
        )

    def fake_optimize_archive_tiers(root: Path, *, reason: str, **_: object) -> tuple[SqliteOptimizeObservation, ...]:
        assert root == archive_root
        assert reason == archive_ingest.POST_COMMIT_UPKEEP_REASON
        optimize_calls.append(root)
        return (SqliteOptimizeObservation(reason=reason, ran=True, analysis_limit=1000),)

    monkeypatch.setattr(archive_ingest, "maybe_checkpoint_archive_wals", fake_checkpoint_archive_wals)
    monkeypatch.setattr(archive_ingest, "maybe_optimize_archive_tiers", fake_optimize_archive_tiers)

    result = asyncio.run(parse_sources_archive(archive_root, sources))

    upkeep_observations = [item for item in result.batch_observations if item.get("archive_post_commit_upkeep")]
    assert len(wal_calls) >= 2
    assert len(optimize_calls) == len(wal_calls)
    assert len(upkeep_observations) == len(wal_calls)
    assert upkeep_observations[0]["wal_checkpoint_modes"] == ["passive"]
    assert upkeep_observations[0]["sqlite_optimize_ran"] == 1
    assert result.batch_observations[-1]["primary_ingest_store"] == "archive_file_set"


def test_per_session_archive_ingest_runs_post_commit_upkeep(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES", "0")
    archive_root = workspace_env["archive_root"]
    sources = _build_sources(tmp_path, count=2, seed=41)
    wal_calls = 0

    def fake_checkpoint_archive_wals(
        root: Path,
        *,
        reason: str,
        allow_truncate: bool = True,
        **_: object,
    ) -> tuple[WalCheckpointObservation, ...]:
        nonlocal wal_calls
        assert root == archive_root
        assert reason == archive_ingest.POST_COMMIT_UPKEEP_REASON
        assert allow_truncate is False
        wal_calls += 1
        return ()

    def fake_optimize_archive_tiers(root: Path, *, reason: str, **_: object) -> tuple[SqliteOptimizeObservation, ...]:
        assert root == archive_root
        assert reason == archive_ingest.POST_COMMIT_UPKEEP_REASON
        return ()

    monkeypatch.setattr(archive_ingest, "maybe_checkpoint_archive_wals", fake_checkpoint_archive_wals)
    monkeypatch.setattr(archive_ingest, "maybe_optimize_archive_tiers", fake_optimize_archive_tiers)

    result = asyncio.run(parse_sources_archive(archive_root, sources))

    assert wal_calls == len(sources)
    assert sum(1 for item in result.batch_observations if item.get("archive_post_commit_upkeep")) == len(sources)
    assert result.batch_observations[-1]["archive_write_targets"] == ["source.db", "index.db"]
