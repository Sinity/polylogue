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
from polylogue.storage.blob_integrity import classify_blob_reference_debt
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import _MeasuredConnection, _MeasuredCursor
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import bootstrap_baseline_archive, seed_durable_tier


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


def test_pre003_complete_backup_is_verified_without_current_coordinate_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    bootstrap_baseline_archive(root, monkeypatch)
    payload, digest = _seed(root, tmp_path / "source.json", retain=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("opaque complete backup interpreted current coordinates")

    monkeypatch.setattr(backup, "_raw_session_reference_rows", forbidden)
    result = _package(root, tmp_path / "package")
    assert result.ok and result.verified, result.error
    assert result.output_path is not None
    package = Path(result.output_path)
    assert (package / "blob" / digest[:2] / digest[2:]).read_bytes() == payload
    with closing(sqlite3.connect(package / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert result.verification["missing_canonical_blob_count"] == 0


def test_pre003_missing_bytes_refuse_before_current_coordinate_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    bootstrap_baseline_archive(root, monkeypatch)
    _payload, digest = _seed(root, tmp_path / "source.json", retain=False)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("incompatible snapshot reached current coordinate SQL")

    monkeypatch.setattr(backup, "_raw_session_reference_rows", forbidden)
    with pytest.raises(SchemaSkew) as refused:
        _package(root, tmp_path / "package")
    assert refused.value.tier == "source" and refused.value.found == 1
    assert refused.value.expected == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    partial_packages = list((tmp_path / "package").iterdir())
    assert len(partial_packages) == 1
    assert partial_packages[0].name.startswith("polylogue-archive-")
    assert not (partial_packages[0] / "verification-receipt.json").exists()
    assert not BlobStore(root / "blob").exists(digest)
    with closing(sqlite3.connect(root / "source.db")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] == 1
    with pytest.raises(SchemaSkew):
        classify_blob_reference_debt(root / "source.db")
    renamed = tmp_path / "source-evidence.sqlite"
    renamed.write_bytes((root / "source.db").read_bytes())
    with pytest.raises(SchemaSkew) as renamed_refusal:
        classify_blob_reference_debt(renamed)
    assert renamed_refusal.value.tier == "source" and renamed_refusal.value.found == 1


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


@pytest.mark.parametrize("source_version", [3, 4, 5, 7])
def test_recovery_reader_uses_declared_coordinate_schema_across_additive_trains(
    tmp_path: Path, source_version: int
) -> None:
    from importlib import resources

    from polylogue.core.enums import Origin, Provider
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_DDL_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    source_db = tmp_path / "source.db"
    original = tmp_path / "captured.json"
    payload = b'{"messages":[]}\n'
    original.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    migrations = resources.files("polylogue.storage.sqlite.migrations.source")
    with closing(sqlite3.connect(source_db)) as conn:
        conn.executescript(ARCHIVE_BASELINE_DDL_BY_TIER[ArchiveTier.SOURCE])
        for name in (
            "002_raw_artifact_failure_identity.sql",
            "003_attachment_coordinate_identity.sql",
            "004_captured_profile_identity.sql",
            "005_raw_byte_revision_dependents.sql",
            "006_raw_frontier_dependency_journal.sql",
        ):
            if int(name[:3]) <= source_version:
                conn.executescript(migrations.joinpath(name).read_text(encoding="utf-8"))
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
    if source_version in (3, 7):
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
