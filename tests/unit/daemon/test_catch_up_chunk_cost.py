"""A catch-up chunk's convergence cost is bounded by the chunk, not the archive.

The live batch processor runs the daemon's real convergence stages after each
catch-up chunk. Archive-wide work (the exact FTS readiness audit, the raw
authority verdict warmer, graph rebuilds from every raw artifact, a scan of
every hook sidecar journal) must not be paid per chunk: it runs once, on the
catch-up's final chunk.

Anti-vacuity: the statement count is measured inside ``_converge_paths`` only.
Reverting any ``whole_archive`` deferral (raw authority: two statements per
cohort; Claude workflow: one per artifact row) makes the large archive's chunk
issue more statements than the small archive's, reverting the FTS readiness
deferral makes the snapshot spy fire, and reverting sidecar scoping makes the
scan touch every session's journal.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon import convergence_stages
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_default_convergence_stages
from polylogue.sources.live import hook_paste_enrichment
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import WatchSource

_MESSAGES_PER_SESSION = 8


def _session_records(uuid: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(_MESSAGES_PER_SESSION):
        is_user = index % 2 == 0
        records.append(
            {
                "parentUuid": None if index == 0 else f"{uuid}-msg-{index - 1:04d}",
                "sessionId": uuid,
                "type": "user" if is_user else "assistant",
                "message": {
                    "role": "user" if is_user else "assistant",
                    "content": f"Synthetic message {index} of {uuid}. Searchable prose about convergence cost.",
                },
                "uuid": f"{uuid}-msg-{index:04d}",
                "timestamp": f"2026-05-05T00:{index // 60:02d}:{index % 60:02d}.000Z",
                "cwd": "/realm/project/polylogue",
                "version": "1.0.6",
                "isSidechain": False,
                "userType": "external",
            }
        )
    return records


def _write_session(corpus_root: Path, hooks_dir: Path, ordinal: int) -> Path:
    uuid = f"deadbeef-0000-0000-0000-{ordinal:012x}"
    path = corpus_root / "test-project" / f"{uuid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in _session_records(uuid)), encoding="utf-8")
    (hooks_dir / f"claude-code-{uuid}.jsonl").write_text(
        json.dumps(
            {
                "event_type": "UserPromptSubmit",
                "timestamp": "2026-05-05T00:00:00Z",
                "payload": {"session_id": uuid, "prompt": "Inspect [Pasted text #1]"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class _Polylogue:
    def __init__(self, archive_root: Path, db_path: Path) -> None:
        self.archive_root = archive_root
        self.backend = SimpleNamespace(db_path=db_path)


class _ChunkProbe:
    """Statements issued and sidecars read while one chunk converges."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.statements = 0
        self.sidecars_read = 0
        self.snapshot_calls = 0
        self.active = False
        real_connect = sqlite3.connect
        real_sidecar_paths = hook_paste_enrichment._sidecar_paths
        real_snapshot = convergence_stages._record_fts_freshness_after_insights

        def counting_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            conn = cast(sqlite3.Connection, real_connect(*args, **kwargs))
            conn.set_trace_callback(self._count_statement)
            return conn

        def counting_sidecar_paths(hooks_dir: Path, session_ids: Any) -> list[Path]:
            paths = real_sidecar_paths(hooks_dir, session_ids)
            if self.active:
                self.sidecars_read += len(paths)
            return paths

        def counting_snapshot(conn: sqlite3.Connection) -> bool:
            if self.active:
                self.snapshot_calls += 1
            return real_snapshot(conn)

        monkeypatch.setattr(sqlite3, "connect", counting_connect)
        monkeypatch.setattr(hook_paste_enrichment, "_sidecar_paths", counting_sidecar_paths)
        monkeypatch.setattr(convergence_stages, "_record_fts_freshness_after_insights", counting_snapshot)

    def _count_statement(self, sql: str) -> None:
        # SQLite reports its own virtual-table maintenance (FTS5 segment
        # merges) as ``--``-prefixed statements; only product statements
        # measure the convergence pass.
        if self.active and not sql.lstrip().startswith("--"):
            self.statements += 1


def _build(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    seeded_sessions: int,
) -> tuple[LiveBatchProcessor, Path, Path]:
    archive_root = tmp_path / f"archive-{seeded_sessions}"
    archive_root.mkdir()
    hooks_dir = archive_root / "hooks"
    hooks_dir.mkdir()
    corpus_root = tmp_path / f"corpus-{seeded_sessions}"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(archive_root / "polylogue.toml"))
    db_path = archive_root / "index.db"
    converger = DaemonConverger(stages=make_default_convergence_stages(db_path))
    processor = LiveBatchProcessor(
        cast(Any, _Polylogue(archive_root, db_path)),
        (WatchSource(name="claude-code", root=corpus_root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="chunk-cost-v1",
        converger=converger,
    )
    seeded = [_write_session(corpus_root, hooks_dir, ordinal) for ordinal in range(seeded_sessions)]
    metrics = asyncio.run(processor.ingest_files(seeded, emit_event=False))
    assert metrics.succeeded_file_count == seeded_sessions
    return processor, corpus_root, hooks_dir


def _converge_chunk(
    processor: LiveBatchProcessor,
    probe: _ChunkProbe,
    paths: list[Path],
    *,
    whole_archive: bool,
) -> Any:
    real_converge = processor._converge_paths

    def probed_converge(*args: Any, **kwargs: Any) -> Any:
        probe.active = True
        try:
            return real_converge(*args, **kwargs)
        finally:
            probe.active = False

    processor._converge_paths = probed_converge  # type: ignore[method-assign]
    try:
        return asyncio.run(processor.ingest_files(paths, emit_event=False, whole_archive_convergence=whole_archive))
    finally:
        processor._converge_paths = real_converge  # type: ignore[method-assign]


@pytest.mark.parametrize("chunk_files", [2])
def test_chunk_convergence_cost_does_not_grow_with_archive_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_files: int
) -> None:
    small_sessions, large_sessions = 2, 14
    probe = _ChunkProbe(monkeypatch)
    results: dict[int, tuple[int, int, int, dict[str, float]]] = {}
    for seeded_sessions in (small_sessions, large_sessions):
        processor, corpus_root, hooks_dir = _build(tmp_path, monkeypatch=monkeypatch, seeded_sessions=seeded_sessions)
        chunk = [_write_session(corpus_root, hooks_dir, seeded_sessions + offset) for offset in range(chunk_files)]
        probe.statements = probe.sidecars_read = probe.snapshot_calls = 0
        metrics = _converge_chunk(processor, probe, chunk, whole_archive=False)
        assert metrics.succeeded_file_count == chunk_files
        results[seeded_sessions] = (
            probe.statements,
            probe.sidecars_read,
            probe.snapshot_calls,
            dict(metrics.stage_timings_s),
        )

    small_statements, small_sidecars, small_snapshots, small_stages = results[small_sessions]
    large_statements, large_sidecars, large_snapshots, large_stages = results[large_sessions]
    deferred = {"raw_authority_verdict_cache", "claude_workflow", "delegation_work_evidence", "fts_readiness"}

    assert small_sidecars == large_sidecars == chunk_files
    assert small_snapshots == large_snapshots == 0
    assert not deferred & set(small_stages) and not deferred & set(large_stages)
    assert "hook_paste_enrichment" in large_stages
    # A chunk's statements are a function of its own files; the slack covers
    # per-chunk variance and stays far below one deferred stage's growth
    # (two statements per cohort, one per artifact row).
    assert large_statements <= small_statements + 4, (small_statements, large_statements)


def test_final_catch_up_chunk_runs_the_whole_archive_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _ChunkProbe(monkeypatch)
    processor, corpus_root, hooks_dir = _build(tmp_path, monkeypatch=monkeypatch, seeded_sessions=2)
    chunk = [_write_session(corpus_root, hooks_dir, 2)]

    probe.statements = probe.sidecars_read = probe.snapshot_calls = 0
    metrics = _converge_chunk(processor, probe, chunk, whole_archive=True)

    assert metrics.succeeded_file_count == 1
    assert probe.snapshot_calls == 1
    assert {"fts_readiness", "raw_authority_verdict_cache"} <= set(metrics.stage_timings_s)
