"""Tests for the polylogue-omsw tool-result/file-history reclassification sweep.

Covers the read-only sweep (``scan_tool_result_and_file_history_artifacts``)
and the ``--apply``-gated actuator
(``devtools/tool_result_history_reclassify_apply.py``), mirroring the
existing ``binary_artifact_sweep``/``binary_artifact_reclassify_apply``
pattern for an earlier miscapture class (polylogue-hbtj2).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from devtools.tool_result_history_reclassify_apply import main as reclassify_apply_main
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.tool_result_history_sweep import scan_tool_result_and_file_history_artifacts

_ORIGIN = "claude-code-session"


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    blob_root = tmp_path / "blob"
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: blob_root)
    monkeypatch.setattr("polylogue.storage.blob_store.blob_store_root", lambda: blob_root, raising=False)
    reset_blob_store()
    yield tmp_path
    reset_blob_store()


def _insert_raw_session(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    native_id: str,
    source_path: str,
    blob_size: int,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
        ) VALUES (?, ?, ?, ?, 0, ?, ?, 1)
        """,
        (raw_id, _ORIGIN, native_id, source_path, bytes.fromhex(raw_id), blob_size),
    )


def _seed(archive: Path) -> None:
    store = BlobStore(archive / "blob")

    tool_result_raw_id, tool_result_size = store.write_from_bytes(
        b'{"messages":[{"role":"user","content":"hi"}],"chat_messages":[{"role":"user","text":"hi"}],'
        b'"mapping":{"a":{"message":{"role":"user","content":[{"type":"text","text":"hi"}]}}}}'
    )
    history_only_raw_id, history_only_size = store.write_from_bytes(
        b'{"type":"file-history-snapshot","messageId":"m1","sessionId":"history-only-session",'
        b'"snapshot":{"messageId":"m1","trackedFileBackups":{}}}\n'
        b'{"type":"file-history-snapshot","messageId":"m2","sessionId":"history-only-session",'
        b'"snapshot":{"messageId":"m2","trackedFileBackups":{}}}\n'
    )
    genuine_raw_id, genuine_size = store.write_from_bytes(
        b'{"type":"user","sessionId":"s1","uuid":"u1","message":{"role":"user","content":"hi"}}\n'
        b'{"type":"assistant","sessionId":"s1","uuid":"u2","parentUuid":"u1",'
        b'"message":{"role":"assistant","content":"hey"}}\n'
    )

    with sqlite3.connect(archive / "source.db") as conn:
        _insert_raw_session(
            conn,
            raw_id=tool_result_raw_id,
            native_id="toolu_01scratchprobe",
            source_path="/home/user/.claude/projects/proj/tool-results/toolu_01scratchprobe.json",
            blob_size=tool_result_size,
        )
        _insert_raw_session(
            conn,
            raw_id=history_only_raw_id,
            native_id="history-only-session",
            source_path="/home/user/.claude/projects/proj/history-only-session.jsonl",
            blob_size=history_only_size,
        )
        _insert_raw_session(
            conn,
            raw_id=genuine_raw_id,
            native_id="genuine-session",
            source_path="/home/user/.claude/projects/proj/genuine-session.jsonl",
            blob_size=genuine_size,
        )
        conn.commit()


def test_sweep_finds_tool_result_and_file_history_rows_but_not_genuine_sessions(archive: Path) -> None:
    _seed(archive)

    with sqlite3.connect(f"file:{archive / 'source.db'}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        plan = scan_tool_result_and_file_history_artifacts(conn)

    assert plan.scanned_count == 3
    found_native_ids = {
        # source_path stem doubles as native_id in this fixture
        Path(candidate.source_path).stem
        for candidate in plan.candidates
    }
    assert found_native_ids == {"toolu_01scratchprobe", "history-only-session"}
    assert plan.by_kind() == {"tool_result_sidecar": 1, "file_history_snapshot": 1}


def test_reclassify_apply_dry_run_writes_nothing(archive: Path) -> None:
    _seed(archive)

    exit_code = reclassify_apply_main(["--archive-root", str(archive), "--json"])
    assert exit_code == 0

    with sqlite3.connect(archive / "source.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone()[0]
    assert count == 0


def test_reclassify_apply_persists_raw_artifacts_without_touching_raw_sessions(archive: Path) -> None:
    _seed(archive)

    with sqlite3.connect(archive / "source.db") as conn:
        raw_sessions_before = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]

    exit_code = reclassify_apply_main(["--archive-root", str(archive), "--apply", "--json"])
    assert exit_code == 0

    with sqlite3.connect(archive / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        raw_sessions_after = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
        artifact_rows = conn.execute("SELECT artifact_kind, source_path FROM raw_artifacts").fetchall()

    # Never deletes raw_sessions rows -- durable-tier evidence retention
    # precedent (the 2026-07-22 hook-inflation postmortem).
    assert raw_sessions_after == raw_sessions_before == 3

    kinds_by_path = {row["source_path"]: row["artifact_kind"] for row in artifact_rows}
    assert kinds_by_path["/home/user/.claude/projects/proj/tool-results/toolu_01scratchprobe.json"] == (
        "tool_result_sidecar"
    )
    assert kinds_by_path["/home/user/.claude/projects/proj/history-only-session.jsonl"] == "file_history_snapshot"
    assert kinds_by_path["/home/user/.claude/projects/proj/genuine-session.jsonl"] != "file_history_snapshot"
