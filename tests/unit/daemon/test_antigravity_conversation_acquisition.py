"""Periodic Antigravity ``.pb`` conversation acquisition (polylogue-3m3de).

Exercises ``acquire_antigravity_conversations_once`` -- the bounded sync body
the periodic daemon loop (``periodic_antigravity_conversation_acquisition_check``)
schedules on a quiet cadence -- against a real archive fixture with a fake
language-server client, proving:

1. Real ``conversations/*.pb`` cascades get acquired into ``raw_sessions``
   with their own ``.pb`` source_path (not the metadata-sidecar fragments
   the live watcher was stuck admitting forever).
2. Re-running is a no-op once every cascade is acquired (content-hash /
   already-acquired dedup, not a re-conversion on every tick).
3. The per-tick batch is bounded so one reconciliation pass never converts
   an unbounded number of cascades.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from polylogue.daemon.antigravity_conversation_acquisition import (
    ANTIGRAVITY_ACQUISITION_MAX_PER_TICK,
    acquire_antigravity_conversations_once,
)
from polylogue.sources import source_parsing
from polylogue.sources.parsers.antigravity import AntigravitySessionSummary
from polylogue.storage.blob_store import BlobStore


class _FakeLanguageServerClient:
    """Drop-in fake of AntigravityLanguageServerClient for driver tests."""

    def __init__(self, markdown_by_cascade: dict[str, str]) -> None:
        self._markdown = markdown_by_cascade
        self.started = False
        self.closed = False
        self.exported_cascade_ids: list[str] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]:
        return []

    def export_markdown(self, cascade_id: str) -> str:
        self.exported_cascade_ids.append(cascade_id)
        return self._markdown[cascade_id]


def _touch_conversation_pb(antigravity_root: Path, *cascade_ids: str) -> None:
    conversations = antigravity_root / "conversations"
    conversations.mkdir(parents=True, exist_ok=True)
    for cascade_id in cascade_ids:
        (conversations / f"{cascade_id}.pb").write_bytes(b"fake-protobuf-bytes-" + cascade_id.encode())


def _raw_pb_blobs_by_source_path(source_db: Path) -> dict[str, str]:
    import sqlite3

    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT source_path, blob_hash FROM raw_sessions WHERE origin = 'antigravity-session'"
        ).fetchall()
    finally:
        conn.close()
    return {str(source_path): bytes(blob_hash).hex() for source_path, blob_hash in rows}


@pytest.fixture
def _fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeLanguageServerClient:
    fake = _FakeLanguageServerClient(
        {f"cascade-{i}": f"### User Input\n\nhello {i}\n" for i in range(ANTIGRAVITY_ACQUISITION_MAX_PER_TICK + 5)}
    )
    monkeypatch.setattr(
        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
        lambda root: fake,
    )
    return fake


def test_no_conversations_dir_reports_gap_without_promoting_brain_sidecar(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    antigravity_root = tmp_path / "antigravity-empty"
    sidecar = antigravity_root / "brain" / "work-session" / "plan.md.metadata.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.with_name("plan.md").write_text("# Plan\n\nInspect the archive.\n", encoding="utf-8")
    sidecar.write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Plan","updatedAt":"2026-08-04T08:00:00Z"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("polylogue.paths.antigravity_path", lambda: antigravity_root)
    gap_logger = Mock()
    monkeypatch.setattr(source_parsing, "logger", gap_logger)

    archive_root = workspace_env["archive_root"]
    acquired = acquire_antigravity_conversations_once(archive_root)

    assert acquired == 0
    assert _raw_pb_blobs_by_source_path(archive_root / "source.db") == {}
    gap_logger.warning.assert_called_once()
    warning = gap_logger.warning.call_args
    assert warning.args == ("antigravity_coverage_gap",)
    assert warning.kwargs["source_name"] == "antigravity"
    assert warning.kwargs["source_path"] == str(antigravity_root)
    assert warning.kwargs["reason"] == "conversations_directory_missing"
    assert "restore" in warning.kwargs["action"]


def test_acquires_real_pb_conversations_not_yet_in_raw_sessions(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    _fake_client: _FakeLanguageServerClient,
) -> None:
    antigravity_root = tmp_path / "antigravity"
    _touch_conversation_pb(antigravity_root, "cascade-0", "cascade-1")
    sidecar = antigravity_root / "brain" / "work-session" / "plan.md.metadata.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Plan","updatedAt":"2026-08-04T08:00:00Z"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("polylogue.paths.antigravity_path", lambda: antigravity_root)

    archive_root = workspace_env["archive_root"]
    acquired = acquire_antigravity_conversations_once(archive_root)

    assert acquired == 2
    expected_payloads = {
        str(path): path.read_bytes() for path in sorted((antigravity_root / "conversations").glob("*.pb"))
    }
    raw_blobs = _raw_pb_blobs_by_source_path(archive_root / "source.db")

    assert sorted(raw_blobs) == sorted(expected_payloads)
    blob_store = BlobStore(archive_root / "blob")
    assert {
        source_path: blob_store.read_all(blob_hash) for source_path, blob_hash in raw_blobs.items()
    } == expected_payloads
    assert str(sidecar) not in raw_blobs


def test_rerun_after_full_acquisition_is_a_noop(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    _fake_client: _FakeLanguageServerClient,
) -> None:
    antigravity_root = tmp_path / "antigravity"
    _touch_conversation_pb(antigravity_root, "cascade-0", "cascade-1")
    monkeypatch.setattr("polylogue.paths.antigravity_path", lambda: antigravity_root)

    archive_root = workspace_env["archive_root"]
    first = acquire_antigravity_conversations_once(archive_root)
    assert first == 2

    _fake_client.exported_cascade_ids.clear()
    second = acquire_antigravity_conversations_once(archive_root)

    assert second == 0
    # No language-server RPC at all for the already-acquired cascades.
    assert _fake_client.exported_cascade_ids == []


def test_batch_is_bounded_per_tick(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    _fake_client: _FakeLanguageServerClient,
) -> None:
    antigravity_root = tmp_path / "antigravity"
    cascade_ids = [f"cascade-{i}" for i in range(ANTIGRAVITY_ACQUISITION_MAX_PER_TICK + 5)]
    _touch_conversation_pb(antigravity_root, *cascade_ids)
    monkeypatch.setattr("polylogue.paths.antigravity_path", lambda: antigravity_root)

    archive_root = workspace_env["archive_root"]
    acquired = acquire_antigravity_conversations_once(archive_root)

    assert acquired == ANTIGRAVITY_ACQUISITION_MAX_PER_TICK
    # A second tick picks up the remainder.
    remaining = acquire_antigravity_conversations_once(archive_root)
    assert remaining == 5
