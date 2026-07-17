"""Measured regression scenario for backup verification read amplification."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.daemon import backup as backup_mod
from polylogue.daemon.backup import backup_archive
from polylogue.storage.blob_store import BlobStore
from tests.infra.backup_read_counter import backup_verification_read_counter
from tests.infra.storage_records import db_setup


def _seed_referenced_blob(archive_root: Path, *, raw_id: str, payload: bytes) -> None:
    blob_hash, _ = BlobStore(archive_root / "blob").write_from_bytes(payload)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "chatgpt-export",
                raw_id,
                f"/tmp/{raw_id}.json",
                0,
                bytes.fromhex(blob_hash),
                len(payload),
                1,
                "passed",
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (bytes.fromhex(blob_hash), raw_id, "raw_payload", f"/tmp/{raw_id}.json", len(payload), 1),
        )


def test_verified_backup_avoids_rehashing_scratch_blobs_after_validation(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    """Pin the two required proof reads: scratch validation and original stability.

    Three deliberately unequal blobs ensure the measurement is byte-accurate,
    not merely a call-count artefact.  This scenario is intentionally a
    characterization.  Receipt construction must reuse the hash just obtained
    from the scratch payload rather than rereading every scratch blob.
    """
    db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    payloads = (b"a" * 17, b"b" * 1031, b"c" * 16_385)
    for index, payload in enumerate(payloads):
        _seed_referenced_blob(archive_root, raw_id=f"raw-{index}", payload=payload)
    total_blob_bytes = sum(map(len, payloads))

    result = backup_archive(output_dir=tmp_path / "backups", verify=False)
    assert result.ok

    with backup_verification_read_counter() as counter:
        backup_mod._verify_backup_result(result)

    assert result.ok
    assert result.verified
    assert counter.calls_by_site == {
        "read_bytes:scratch": len(payloads),
        "sha256_file:backup": len(payloads),
    }
    assert counter.bytes_by_site == {
        "read_bytes:scratch": total_blob_bytes,
        "sha256_file:backup": total_blob_bytes,
    }
