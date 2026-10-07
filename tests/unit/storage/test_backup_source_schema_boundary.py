"""Opaque backup evidence does not authorize current-coordinate interpretation."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.errors import SchemaSkew
from polylogue.core.write_lease import write_lease
from polylogue.storage import backup_package as backup
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import _MeasuredConnection, _MeasuredCursor
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import seed_durable_tier


def _seed(root: Path, source: Path, *, retain: bool) -> tuple[bytes, str]:
    payload = b'{"messages":[]}\n'
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with seed_durable_tier(root / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_sessions(raw_id,origin,source_path,source_index,blob_hash,blob_size,acquired_at_ms) "
            "VALUES('neutral-raw','chatgpt-export',?,0,?,?,1)",
            (str(source), bytes.fromhex(digest), len(payload)),
        ).close()
        conn.execute(
            "INSERT INTO blob_refs VALUES(?,'neutral-raw','raw_payload',?,?,1)",
            (bytes.fromhex(digest), str(source), len(payload)),
        ).close()
    if retain:
        assert BlobStore(root / "blob").write_from_bytes(payload)[0] == digest
    return payload, digest


def _package(root: Path, destination: Path) -> backup.BackupResult:
    with write_lease("synthetic-backup-boundary", archive_root=root):
        with OwnedArchiveLocation.acquire(ArchiveLocation.resolve(root)) as owner:
            return backup.create_backup_package(
                output_dir=destination, archive_root_path=root, profile="rebuildable_cache_exclude", archive_owner=owner
            )


@pytest.mark.parametrize("source_state", ["exact", "missing", "changed"])
def test_current_backup_acquires_only_exact_bytes_into_closed_package(tmp_path: Path, source_state: str) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    source = tmp_path / "source.json"
    payload, digest = _seed(root, source, retain=False)
    if source_state == "missing":
        source.unlink()
    elif source_state == "changed":
        source.write_bytes(b'{"messages":[1]}\n')
    result = _package(root, tmp_path / "package")
    assert not BlobStore(root / "blob").exists(digest)
    with closing(sqlite3.connect(root / "source.db")) as conn:
        assert conn.execute("SELECT hex(blob_hash),blob_size FROM raw_sessions").fetchall() == [
            (digest.upper(), len(payload))
        ]
    if source_state == "exact":
        assert result.ok and result.verified, result.error
        assert result.verification["recovered_source_blob_count"] == 1
        assert result.output_path is not None
        package = Path(result.output_path)
        assert (package / "blob" / digest[:2] / digest[2:]).read_bytes() == payload
        source.unlink()
        assert backup._verify_archive_file_set_backup(package)["ok"]
    else:
        assert not result.ok and not result.verified
        assert result.verification["canonical_blobs_resolved"] is False
        assert result.verification["missing_canonical_blob_count"] == 1
        assert result.verification["unproven_source_blob_count"] == 1


@pytest.mark.parametrize("failure", [sqlite3.OperationalError, DaemonOperationCancelled])
def test_current_coordinate_schema_read_failure_settles_original_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _payload, digest = _seed(root, tmp_path / "source.json", retain=False)
    original_open = backup._open_backup_readonly_connection
    original_execute = _MeasuredCursor.execute
    handles: list[_MeasuredConnection] = []
    primary = failure("synthetic schema read failure")
    observations = 0

    def opened(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_open(*args, **kwargs)
        assert isinstance(conn, _MeasuredConnection)
        handles.append(conn)
        return conn

    def execute(self: _MeasuredCursor, sql: str, parameters: Any = ()) -> _MeasuredCursor:
        nonlocal observations
        result = original_execute(self, sql, parameters)
        if sql == "PRAGMA user_version":
            observations += 1
            if observations == 2:
                raise primary
        return result

    monkeypatch.setattr(backup, "_open_backup_readonly_connection", opened)
    monkeypatch.setattr(_MeasuredCursor, "execute", execute)
    with pytest.raises(failure) as caught:
        backup._source_recoverability_proofs(root / "source.db", root=root, missing_hashes={digest})
    assert caught.value is primary and observations == 2
    assert len(handles) == 1 and not handles[0].live_cursors()
    with pytest.raises(sqlite3.ProgrammingError):
        handles[0].cursor()


@pytest.mark.parametrize("source_version", [0, 1, 2])
def test_recovery_reader_interprets_only_the_declared_coordinate_schema(tmp_path: Path, source_version: int) -> None:
    """Only a Source tier at a version this runtime declares has coordinates to replay.

    A version outside the baseline-to-runtime range is opaque evidence: the
    reader refuses it with a typed skew instead of guessing current coordinates.
    """
    from polylogue.core.enums import Origin, Provider
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    source_db = tmp_path / "source.db"
    original = tmp_path / "captured.json"
    payload = b'{"messages":[]}\n'
    original.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with closing(sqlite3.connect(source_db)) as conn:
        conn.executescript(ARCHIVE_BASELINE_DDL_BY_TIER[ArchiveTier.SOURCE])
        conn.execute(f"PRAGMA user_version={source_version}")
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CHATGPT_EXPORT,
            capture_mode=Provider.CHATGPT,
            source_path=str(original),
            canonical_source_path=str(original),
            source_index=0,
            payload=payload,
            acquired_at_ms=1,
        )
        conn.commit()
    if source_version != ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]:
        with pytest.raises(SchemaSkew) as refused:
            backup._source_recoverability_proofs(source_db, root=tmp_path, missing_hashes={digest})
        assert refused.value.found == source_version
        return
    proofs = backup._source_recoverability_proofs(source_db, root=tmp_path, missing_hashes={digest})
    assert len(proofs) == 1
    assert proofs[0]["raw_id"] == raw_id
    assert proofs[0]["blob_hash"] == digest
    assert proofs[0]["blob_size"] == str(len(payload))
    assert proofs[0]["source_path"] == str(original)
