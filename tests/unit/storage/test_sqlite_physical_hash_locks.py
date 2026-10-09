"""Physical SQLite hashes must preserve the caller's kernel locks."""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import compute_cancel
from polylogue.storage.sqlite.archive_tiers.schema_inventory import capture_schema_census
from polylogue.storage.sqlite.migration_runner import _validate_live_source_fingerprint
from polylogue.storage.sqlite.physical_file import physical_file_sha256
from tests.infra.sqlite_lock_probe import sqlite_lock_state

pytestmark = pytest.mark.uses_real_clock


def _database(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA user_version=1")
        connection.execute("CREATE TABLE raw_sessions(raw_id TEXT)")
        connection.execute("INSERT INTO raw_sessions VALUES ('neutral')")
        connection.commit()


@contextmanager
def _live_reader(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        assert connection.execute("SELECT raw_id FROM raw_sessions").fetchone() == ("neutral",)
        yield connection


def _assert_protected(path: Path) -> None:
    assert sqlite_lock_state(path) == {"main": "protected", "shm": "protected"}


def test_schema_census_hash_preserves_an_existing_tier_readers_locks(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    _database(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with _live_reader(path):
        census = capture_schema_census(tmp_path, observed_at_ns=0, count_rows=False)
        _assert_protected(path)
    assert next(tier for tier in census.tiers if tier.tier.value == "source").file_sha256 == expected


def test_migration_physical_fingerprint_preserves_its_transaction_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.db"
    _database(path)
    fingerprint = {
        "path": str(path),
        "live_cut_stable": True,
        "wal": None,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "user_version": 1,
    }
    monkeypatch.setattr(
        "polylogue.storage.sqlite.migration_runner._sqlite_user_version",
        lambda *_args, **_kwargs: pytest.fail("live user_version must come from the held transaction"),
    )
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _assert_protected(path)
        _validate_live_source_fingerprint(connection, {"source_fingerprint": fingerprint})
        _assert_protected(path)


def test_physical_hash_rejects_replacement_after_parent_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.db"
    replacement = tmp_path / "replacement.db"
    _database(path)
    _database(replacement)
    selected = path.stat()
    original_popen = subprocess.Popen

    def replace_then_start(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        replacement.replace(path)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr("polylogue.storage.sqlite.physical_file.subprocess.Popen", replace_then_start)
    with pytest.raises(OSError):
        physical_file_sha256(path, expected_device=selected.st_dev, expected_inode=selected.st_ino)


def test_physical_hash_cancellation_kills_and_reaps_exact_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.db"
    _database(path)
    selected = path.stat()
    cancellation = threading.Event()
    context_token = compute_cancel.set(cancellation)
    original_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def start_then_cancel(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original_popen(*args, **kwargs)
        children.append(child)
        cancellation.set()
        return child

    monkeypatch.setattr("polylogue.storage.sqlite.physical_file.subprocess.Popen", start_then_cancel)
    try:
        with pytest.raises(DaemonOperationCancelled):
            physical_file_sha256(path, expected_device=selected.st_dev, expected_inode=selected.st_ino)
    finally:
        compute_cancel.reset(context_token)
    assert len(children) == 1
    assert children[0].returncode is not None


def test_physical_hash_reaps_successful_child_and_returns_exact_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.db"
    _database(path)
    selected = path.stat()
    expected = path.read_bytes()
    original_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def capture_child(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("polylogue.storage.sqlite.physical_file.subprocess.Popen", capture_child)
    observed = physical_file_sha256(path, expected_device=selected.st_dev, expected_inode=selected.st_ino)
    assert observed.sha256 == hashlib.sha256(expected).hexdigest()
    assert observed.size_bytes == len(expected)
    assert len(children) == 1
    assert children[0].returncode == 0
