"""Tests for the one shared real-pipeline seeded archive adapter."""

from __future__ import annotations

import gc
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from tests.infra.workload_artifacts import (
    _journal_mode_delete_with_retry,
    build_seeded_archive,
    clone_seeded_archive,
)


def test_seeded_archive_publishes_valid_immutable_real_pipeline_artifact(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"

    first = build_seeded_archive(cache_root=cache_root)
    second = build_seeded_archive(cache_root=cache_root)

    assert first.root == second.root
    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert first.manifest.receipt["status"] == "succeeded"
    assert len(first.facts) == 64
    assert first.facts[0].expected_session_id == "codex-session:c03-target"
    assert first.root.joinpath("index.db").exists()
    assert raw_materialization_ready(raw_materialization_readiness_snapshot(first.root))
    phases = first.manifest.receipt["phases"]
    assert isinstance(phases, list)
    assert any(isinstance(phase, dict) and phase.get("name") == "raw_authority_frontier" for phase in phases)
    assert not (first.root.stat().st_mode & os.W_OK)


def test_seeded_archive_clone_is_private_full_root_and_preserves_base(tmp_path: Path) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    base_manifest = artifact.root.joinpath("manifest.json").read_bytes()

    clone = clone_seeded_archive(artifact, tmp_path / "clone")
    clone.root.joinpath("private-mutation.txt").write_text("private")

    assert clone.clone_method in {"reflink", "copy"}
    assert clone.source_manifest_id == artifact.manifest.manifest_id
    assert clone.root.joinpath("source.db").exists()
    assert clone.root.joinpath("index.db").exists()
    assert artifact.root.joinpath("manifest.json").read_bytes() == base_manifest
    assert not artifact.root.joinpath("private-mutation.txt").exists()


def test_seeded_archive_rejects_corrupt_published_cache_and_rebuilds(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    original = build_seeded_archive(cache_root=cache_root)
    index_path = original.root / "index.db"
    index_path.chmod(index_path.stat().st_mode | os.W_OK)
    index_path.unlink()

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.root == original.root
    assert rebuilt.root.joinpath("index.db").is_file()
    assert rebuilt.manifest.key == original.manifest.key
    assert rebuilt.manifest.profile_id == original.manifest.profile_id
    assert rebuilt.manifest.recipe_id == original.manifest.recipe_id
    assert rebuilt.facts == original.facts
    assert not (rebuilt.root.stat().st_mode & os.W_OK)


def test_seeded_archive_failure_never_publishes_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    async def fail_parse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected ingest failure")

    monkeypatch.setattr(artifacts, "parse_sources_archive", fail_parse)

    with pytest.raises(RuntimeError, match="injected ingest failure"):
        build_seeded_archive(cache_root=tmp_path / "cache")

    cache_root = tmp_path / "cache"
    assert not list((cache_root / "artifacts").iterdir())
    assert not list((cache_root / ".staging").iterdir())


class _FlakyLockConnection:
    """Fakes ``PRAGMA journal_mode=DELETE`` raising a transient same-process lock.

    Mirrors CPython's ``sqlite3_close_v2`` zombie-connection footgun
    (polylogue-lbgc): a not-yet-finalized cursor/connection from earlier in
    the same worker process keeps SQLite's per-process shared pager-cache
    entry alive, so ``PRAGMA journal_mode=DELETE`` on a legitimate connection
    raises ``sqlite3.OperationalError: database is locked`` until that zombie
    is garbage-collected.
    """

    def __init__(self, *, locked_attempts: int) -> None:
        self.locked_attempts = locked_attempts
        self.attempts = 0

    def execute(self, sql: str) -> _FlakyLockConnection:
        assert sql == "PRAGMA journal_mode=DELETE"
        self.attempts += 1
        if self.attempts <= self.locked_attempts:
            raise sqlite3.OperationalError("database is locked")
        return self

    def fetchone(self) -> tuple[str]:
        return ("delete",)


def test_journal_mode_delete_retry_survives_a_transient_same_process_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: reverting to a bare ``conn.execute(...)`` call (no retry/gc.collect)
    makes this fail on the very first simulated lock, since there would be no
    mechanism left to absorb it."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    gc_collect_calls = 0

    def fake_collect() -> int:
        nonlocal gc_collect_calls
        gc_collect_calls += 1
        return 0

    monkeypatch.setattr(gc, "collect", fake_collect)

    conn = _FlakyLockConnection(locked_attempts=2)
    _journal_mode_delete_with_retry(conn, name="index.db")  # type: ignore[arg-type]

    assert conn.attempts == 3
    assert gc_collect_calls == 2


def test_journal_mode_delete_does_not_retry_non_lock_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-lock OperationalError (e.g. real corruption/IO error) must propagate
    immediately -- retrying it would hide a genuine failure, not a transient
    same-process race."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", sleep_calls.append)

    class _BrokenConnection:
        def execute(self, sql: str) -> Any:
            raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        _journal_mode_delete_with_retry(_BrokenConnection(), name="index.db")  # type: ignore[arg-type]

    assert sleep_calls == []


def test_journal_mode_delete_reraises_lock_once_deadline_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent (non-transient) same-process lock must still surface as a
    real failure rather than retry forever; the fake clock jumps straight past
    the deadline so the test doesn't need to sleep for real."""
    clock = iter([0.0, 10.0, 10.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    class _AlwaysLockedConnection:
        def execute(self, sql: str) -> Any:
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _journal_mode_delete_with_retry(_AlwaysLockedConnection(), name="index.db")  # type: ignore[arg-type]
