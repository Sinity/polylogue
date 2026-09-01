"""Backup verification tests."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zipfile
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes
from polylogue.core.raw_coordinates import zip_member_raw_id
from polylogue.daemon import backup as backup_mod
from polylogue.daemon.backup import backup_archive
from polylogue.operations.zip_acquisition_replay import zip_reacquisition_payload
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.backup_attestation import attestation_key_path
from polylogue.storage.blob_integrity import BlobLivenessProjection
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest
from tests.infra.storage_records import SessionBuilder, db_setup


def _tier_files(*tiers: ArchiveTier) -> list[str]:
    return [ARCHIVE_TIER_SPECS[tier].filename for tier in tiers]


def _tier_integrity(*tiers: ArchiveTier) -> dict[str, bool]:
    return {tier.value: True for tier in tiers}


@pytest.mark.contract
def test_backup_archive_copy_can_be_opened_and_queried(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "backup-conv")
        .provider("claude-code")
        .add_message(role="user", text="backup restore smoke")
    )
    builder.save()
    session_id = builder.native_session_id()

    result = backup_archive(output_dir=tmp_path / "backups")

    # The archive backup is an archive directory: it copies the precious
    # tiers (source/user/embeddings/audit) and omits the rebuildable index/ops
    # tiers. Each copied tier must open cleanly and pass integrity_check.
    assert result.ok
    assert result.backup_mode == "archive_file_set"
    assert result.output_path is not None
    backup_path = Path(result.output_path)
    assert backup_path.is_dir()
    for tier in _tier_files(ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.AUDIT):
        tier_path = backup_path / tier
        assert tier_path.exists(), f"backup missing precious tier {tier}"
        with sqlite3.connect(f"file:{tier_path}?mode=ro", uri=True) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not (backup_path / "index.db").exists()
    assert not (backup_path / "ops.db").exists()

    # The pre-backup index.db still carries the seeded session/messages;
    # the backup intentionally omits this rebuildable tier.
    index_db = workspace_env["archive_root"] / "index.db"
    with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
        session_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        message_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
    assert session_count == 1
    assert message_count == 1


def test_full_evidence_backup_does_not_carry_gc_marker_without_bound_namespace(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    """A restored pending intent blocks rather than claiming a new namespace.

    Anti-vacuity: copying the marker as an ordinary blob artifact lets this
    source-tier intent resume against a separately recreated blob root.
    """
    db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    from polylogue.storage import blob_gc

    blob_root = archive_root / "blob"
    blob_root.mkdir(exist_ok=True)
    marker = blob_gc._blob_namespace_identity(blob_root, create_marker=True).marker
    pending_hash = "a" * 64
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            "INSERT INTO gc_generations "
            "(generation_id, started_at_ms, completed_at_ms, reclaimed_count, reclaimed_bytes, blob_namespace_marker) "
            "VALUES ('backup-pending', 1, NULL, 0, 0, ?)",
            (marker,),
        )
        conn.execute(
            "INSERT INTO gc_generation_members "
            "(generation_id, blob_hash, candidate_size_bytes, intent_committed_at_ms, outcome) "
            "VALUES ('backup-pending', ?, 1, 1, 'pending')",
            (bytes.fromhex(pending_hash),),
        )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert not (backup_root / "blob" / ".polylogue-blob-namespace").exists()
    report = blob_gc.run_blob_gc_report(backup_root / "source.db", backup_root / "blob")
    assert report.blocked_reason is not None
    with sqlite3.connect(backup_root / "source.db") as conn:
        assert conn.execute(
            "SELECT outcome FROM gc_generation_members WHERE generation_id = 'backup-pending'"
        ).fetchone() == ("pending",)


@pytest.mark.contract
def test_backup_archive_includes_archive_files(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    data_home = workspace_env["data_root"] / "polylogue"
    archive_root = workspace_env["archive_root"]
    data_home.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    db_anchor = data_home / "index.db"
    user_db = archive_root / "user.db"
    embeddings_db = archive_root / "embeddings.db"
    index_db = archive_root / "index.db"

    with sqlite3.connect(db_anchor) as conn:
        conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO marker VALUES ('legacy')")
    with sqlite3.connect(user_db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO marker VALUES ('native-user')")
    with sqlite3.connect(embeddings_db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO marker VALUES ('native-embeddings')")
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO marker VALUES ('native')")

    result = backup_archive(output_dir=tmp_path / "backups")

    assert result.ok
    assert result.output_path is not None
    backup_path = Path(result.output_path)
    assert backup_path.name.startswith("polylogue-archive-")
    assert backup_path.is_dir()
    assert not (backup_path / "index.db").exists()
    assert not (backup_path / "ops.db").exists()
    assert (backup_path / "source.db").exists()
    assert (backup_path / "user.db").exists()
    assert (backup_path / "embeddings.db").exists()
    with sqlite3.connect(backup_path / "user.db") as conn:
        marker = conn.execute("SELECT value FROM marker").fetchone()[0]
    assert marker == "native-user"


def test_backup_uses_a_valid_external_active_index_target(workspace_env: dict[str, Path], tmp_path: Path) -> None:
    """The active pointer wins over a stale conventional index file.

    Anti-vacuity: the real tier selector receives an absolute, readable index
    outside the archive root. Root-only fallback would silently back up the
    stale conventional index instead.
    """
    root = workspace_env["archive_root"]
    conventional = root / "index.db"
    with sqlite3.connect(conventional) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('stale')")
    external = tmp_path / "external" / "index.db"
    external.parent.mkdir()
    with sqlite3.connect(external) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('active')")
    pointer = root / ".index-active-pointer"
    pointer.unlink(missing_ok=True)
    pointer.write_text(str(external) + "\n", encoding="utf-8")

    assert backup_mod._all_archive_tiers(root)["index"] == external


def test_backup_ignores_an_invalid_external_active_index_target(workspace_env: dict[str, Path], tmp_path: Path) -> None:
    """A malformed external pointer cannot poison full-evidence backup input."""
    root = workspace_env["archive_root"]
    conventional = root / "index.db"
    with sqlite3.connect(conventional) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO marker VALUES ('conventional')")
    external = tmp_path / "external" / "index.db"
    external.parent.mkdir()
    external.write_bytes(b"not a sqlite database")
    pointer = root / ".index-active-pointer"
    pointer.unlink(missing_ok=True)
    pointer.write_text(str(external) + "\n", encoding="utf-8")

    assert backup_mod._all_archive_tiers(root)["index"] == conventional


def test_backup_maps_a_retired_nested_active_index_without_recursive_search(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moved-root active pointer uses only its bounded path suffixes.

    Anti-vacuity: the real selector must map the retired nested conventional
    link to its new generation file while recursive traversal is unavailable.
    """
    root = workspace_env["archive_root"]
    nested = root / "nested"
    generation = nested / ".index-generations" / "gen-retained" / "index.db"
    generation.parent.mkdir(parents=True)
    with sqlite3.connect(generation) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
    retired_root = root.parent / "retired-archive"
    retired_index = retired_root / "nested" / "index.db"
    nested_index = nested / "index.db"
    nested_index.parent.mkdir(exist_ok=True)
    nested_index.symlink_to(retired_index.parent / ".index-generations" / "gen-retained" / "index.db")
    pointer = root / ".index-active-pointer"
    pointer.unlink(missing_ok=True)
    pointer.write_text(str(retired_index) + "\n", encoding="utf-8")
    monkeypatch.setattr(Path, "rglob", lambda *_args, **_kwargs: pytest.fail("fallback must remain bounded"))

    assert backup_mod._all_archive_tiers(root)["index"] == generation


def test_backup_retries_a_writer_commit_between_checkpoint_and_lock(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = workspace_env["archive_root"]
    archive_root.mkdir(parents=True, exist_ok=True)
    user_db = archive_root / "user.db"
    with sqlite3.connect(user_db) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE events (value TEXT NOT NULL)")
        conn.execute("INSERT INTO events VALUES ('before')")

    original_checkpoint = backup_mod._checkpoint_sqlite_for_snapshot
    injected = False

    def checkpoint_then_commit(conn: sqlite3.Connection, path: Path) -> None:
        nonlocal injected
        original_checkpoint(conn, path)
        if not injected:
            injected = True
            with sqlite3.connect(user_db) as writer:
                writer.execute("INSERT INTO events VALUES ('during-gap')")

    monkeypatch.setattr(backup_mod, "_checkpoint_sqlite_for_snapshot", checkpoint_then_commit)

    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays")

    assert result.ok
    assert injected
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    with sqlite3.connect(backup_root / "user.db") as conn:
        assert conn.execute("SELECT value FROM events ORDER BY rowid").fetchall() == [
            ("before",),
            ("during-gap",),
        ]
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = manifest["tier_source_fingerprints"]["user.db"]
    assert fingerprint["sha256"] == hashlib.sha256((backup_root / "user.db").read_bytes()).hexdigest()


def test_backup_verifier_refuses_artifact_source_fingerprint_mismatch(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_setup(workspace_env)
    other_db = tmp_path / "other-user.db"
    with sqlite3.connect(other_db) as conn:
        conn.execute("CREATE TABLE other_state (value TEXT)")
        conn.execute("PRAGMA user_version = 3")

    original_backup = backup_mod._backup_sqlite

    def copy_with_wrong_fingerprint(src: Path, dst: Path) -> tuple[int, dict[str, object]]:
        size, fingerprint = original_backup(src, dst)
        if src.name == "user.db":
            fingerprint = backup_mod._sqlite_source_fingerprint(other_db)
        return size, fingerprint

    monkeypatch.setattr(backup_mod, "_backup_sqlite", copy_with_wrong_fingerprint)

    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=True)

    assert result.ok is False
    assert result.verified is False
    assert "backup artifact does not match its live source fingerprint" in str(result.error)
    assert result.output_path is not None
    assert not (Path(result.output_path) / "verification-receipt.json").exists()


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_backup_verification_refuses_tier_artifact_aliases(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    alias_kind: str,
) -> None:
    db_setup(workspace_env)
    live_user_db = workspace_env["archive_root"] / "user.db"
    with sqlite3.connect(live_user_db) as conn:
        live_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=False)
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    copied_user_db = backup_root / "user.db"
    copied_user_db.unlink()
    if alias_kind == "symlink":
        copied_user_db.symlink_to(live_user_db)
    else:
        os.link(live_user_db, copied_user_db)

    backup_mod._verify_backup_result(result)

    assert result.ok is False
    assert result.verified is False
    assert "real regular file" in str(result.error) or "multiple hard links" in str(result.error)
    assert not (backup_root / "verification-receipt.json").exists()
    with sqlite3.connect(live_user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == live_version


def test_backup_verification_refuses_linked_sqlite_sidecar(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_setup(workspace_env)
    live_user_db = workspace_env["archive_root"] / "user.db"
    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=False)
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    (backup_root / "user.db-wal").symlink_to(live_user_db)

    backup_mod._verify_backup_result(result)

    assert result.ok is False
    assert result.verified is False
    assert "unbound SQLite sidecar" in str(result.error)
    assert not (backup_root / "verification-receipt.json").exists()


@pytest.mark.contract
def test_backup_archive_copies_precious_tiers_and_referenced_blobs(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    archive_root = workspace_env["archive_root"]
    archive_root.mkdir(parents=True, exist_ok=True)
    source_db = archive_root / "source.db"
    user_db = archive_root / "user.db"
    embeddings_db = archive_root / "embeddings.db"

    payload = b"precious raw payload"
    blob_hash, _ = BlobStore(archive_root / "blob").write_from_bytes(payload)
    blob_hash_bytes = bytes.fromhex(blob_hash)

    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-one", "codex-session", "one", "/tmp/raw.jsonl", 0, blob_hash_bytes, len(payload), 1, "passed"),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash_bytes, "raw-one", "raw_payload", "/tmp/raw.jsonl", len(payload), 1),
        )
    with sqlite3.connect(user_db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS backup_test_marks (mark_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO backup_test_marks VALUES ('mark-one')")
    with sqlite3.connect(embeddings_db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS backup_test_embedding_status (session_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO backup_test_embedding_status VALUES ('codex-session:one')")

    result = backup_archive(output_dir=tmp_path / "backups", verify=True)

    assert result.ok
    assert result.backup_mode == "archive_file_set"
    assert result.backup_profile == "rebuildable_cache_exclude"
    assert result.verified is True
    assert result.verification["ok"] is True
    assert result.verification["tier_integrity"] == _tier_integrity(
        ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.AUDIT
    )
    assert result.verification["omitted_tiers_absent"] is True
    assert result.verification["restored_blob_count"] == 1
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert backup_root.is_dir()
    assert result.omitted_tiers == _tier_files(ArchiveTier.INDEX, ArchiveTier.OPS)
    assert (backup_root / "source.db").exists()
    assert (backup_root / "user.db").exists()
    assert (backup_root / "embeddings.db").exists()
    assert (backup_root / "audit.db").exists()
    assert not (backup_root / "index.db").exists()
    assert not (backup_root / "ops.db").exists()
    assert not list(backup_root.glob("*.db-wal"))
    assert not list(backup_root.glob("*.db-shm"))
    assert (backup_root / "blob" / blob_hash[:2] / blob_hash[2:]).read_bytes() == payload
    receipt_path = backup_root / "verification-receipt.json"
    assert receipt_path.exists()
    assert result.verification["receipt_path"] == str(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["format"] == "polylogue-backup-verification-receipt-v2"
    attestations = {item["tier"]: item for item in receipt["attestations"]}
    assert set(attestations) == {"audit", "source", "user"}
    assert attestations["user"]["algorithm"] == "hmac-sha256"
    assert len(attestations["user"]["mac"]) == 64
    key_path = attestation_key_path(workspace_env["archive_root"] / "user.db")
    assert key_path.exists()
    assert len(key_path.read_bytes()) == 32
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert not (backup_root / key_path.name).exists()
    assert receipt["verdict"] == "success"
    assert receipt["manifest_sha256"] == hashlib.sha256((backup_root / "manifest.json").read_bytes()).hexdigest()
    artifact_inventory = {item["path"]: item for item in receipt["artifact_inventory"]}
    assert set(artifact_inventory) == {
        "blob",
        f"blob/{blob_hash[:2]}",
        f"blob/{blob_hash[:2]}/{blob_hash[2:]}",
        "blob-inventory.json",
        "blob-reference-evidence.json",
        "embeddings.db",
        "audit.db",
        "manifest.json",
        "source.db",
        "user.db",
    }
    assert artifact_inventory["user.db"]["sha256"] == hashlib.sha256((backup_root / "user.db").read_bytes()).hexdigest()
    assert "verification-receipt.json" not in artifact_inventory
    assert {artifact["path"] for artifact in receipt["tier_artifacts"]} == {
        "source.db",
        "user.db",
        "embeddings.db",
        "audit.db",
    }
    for artifact in receipt["tier_artifacts"]:
        fingerprint = artifact["source_fingerprint"]
        copied_tier = backup_root / artifact["path"]
        assert fingerprint["sha256"] == hashlib.sha256(copied_tier.read_bytes()).hexdigest()
        assert fingerprint["size_bytes"] == copied_tier.stat().st_size
    assert receipt["blobs"] == [
        {
            "blob_hash": blob_hash,
            "path": f"blob/{blob_hash[:2]}/{blob_hash[2:]}",
            "protection": ["committed"],
            "sha256": blob_hash,
            "size_bytes": len(payload),
        }
    ]

    with sqlite3.connect(backup_root / "source.db") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    with sqlite3.connect(backup_root / "user.db") as conn:
        assert conn.execute("SELECT mark_id FROM backup_test_marks").fetchone()[0] == "mark-one"
    with sqlite3.connect(backup_root / "embeddings.db") as conn:
        assert conn.execute("SELECT session_id FROM backup_test_embedding_status").fetchone()[0] == "codex-session:one"

    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "archive_file_set"
    assert manifest["profile"] == "rebuildable_cache_exclude"
    assert manifest["included_tiers"] == _tier_files(
        ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.AUDIT
    )
    assert manifest["omitted_tiers"] == _tier_files(ArchiveTier.INDEX, ArchiveTier.OPS)
    assert manifest["blob_count"] == 1


@pytest.mark.contract
@pytest.mark.parametrize("source_kind", ["direct", "zip"])
def test_full_evidence_backup_accepts_proven_recoverable_missing_raw_blob(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    source_kind: str,
) -> None:
    """A pruned raw blob is valid backup evidence only when reacquisition matches it."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / ("source.json" if source_kind == "direct" else "source.zip")
    member_payload = b""
    if source_kind == "direct":
        source_path.write_bytes(b'{"messages":[]}')
        recorded_path = str(source_path)
        source_index = 0
        payload = source_path.read_bytes()
    else:
        records = [
            {"metadata": "bundle sibling"},
            {"id": "recoverable", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
            {"id": "second", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
        ]
        member_payload = json.dumps(records, separators=(",", ":")).encode()
        with zipfile.ZipFile(source_path, "w") as archive:
            archive.writestr("conversations.json", member_payload)
        recorded_path = f"{source_path}:conversations.json"
        source_index = 0
        payload = dumps_bytes(records[1])
    blob_hash = hashlib.sha256(payload).digest()
    raw_id = (
        zip_member_raw_id(
            source_path=recorded_path,
            entry_ordinal=0,
            split_index=0,
            blob_hash=blob_hash.hex(),
        )
        if source_kind == "zip"
        else "recoverable-raw"
    )
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
                "claude-ai-export" if source_kind == "zip" else "chatgpt-export",
                "recoverable",
                recorded_path,
                source_index,
                blob_hash,
                len(payload),
                1,
                "passed",
            ),
        )
        if source_kind == "zip":
            conn.execute("UPDATE raw_sessions SET capture_mode = 'unknown' WHERE raw_id = ?", (raw_id,))
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash, raw_id, "raw_payload", recorded_path, len(payload), 1),
        )
        if source_kind == "zip":
            conn.execute(
                "INSERT INTO raw_container_coordinates VALUES (?, 'zip-v2', ?, ?)",
                (raw_id, 0, 0),
            )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verified
    assert result.verification["missing_canonical_blob_count"] == 0
    assert result.verification["recoverable_source_blob_count"] == 1

    if source_kind == "direct":
        source_path.write_bytes(b"changed source bytes")
    else:
        drifted_records = [
            {"id": "recoverable", "mapping": {"node": {"message": {"author": {"role": "assistant"}}}}},
            {"id": "second", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
        ]
        with zipfile.ZipFile(source_path, "w") as archive:
            archive.writestr("conversations.json", json.dumps(drifted_records, separators=(",", ":")))
    drifted_source = backup_mod._verify_archive_file_set_backup(Path(result.output_path or ""))
    assert drifted_source["ok"] is False
    assert drifted_source["missing_canonical_blob_count"] == 1

    if source_kind == "direct":
        source_path.write_bytes(payload)
    else:
        with zipfile.ZipFile(source_path, "w") as archive:
            archive.writestr("conversations.json", member_payload)
    source_path.unlink()
    missing_source = backup_mod._verify_archive_file_set_backup(Path(result.output_path or ""))
    assert missing_source["ok"] is False
    assert missing_source["missing_canonical_blob_count"] == 1


@pytest.mark.contract
def test_full_evidence_backup_proves_retired_root_recorded_path(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A recorded path under a retired archive root re-anchors onto the root in force.

    Anti-vacuity: proofs resolve against the archive root, not the backup
    staging directory — staging holds no inbox/, so resolving there reports
    source_missing for material the archive still has.
    """
    archive_root = workspace_env["archive_root"]
    inbox = archive_root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    source_path = inbox / "export.zip"
    records = [
        {"metadata": "bundle sibling"},
        {"id": "recoverable", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
        {"id": "second", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
    ]
    member_payload = json.dumps(records, separators=(",", ":")).encode()
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("conversations.json", member_payload)
    retired_recorded_path = f"{tmp_path / 'retired-root' / 'inbox' / 'export.zip'}:conversations.json"
    payload = dumps_bytes(records[1])
    blob_hash = hashlib.sha256(payload).digest()
    raw_id = zip_member_raw_id(
        source_path=retired_recorded_path,
        entry_ordinal=0,
        split_index=0,
        blob_hash=blob_hash.hex(),
    )
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
                "claude-ai-export",
                "recoverable",
                retired_recorded_path,
                0,
                blob_hash,
                len(payload),
                1,
                "passed",
            ),
        )
        conn.execute("UPDATE raw_sessions SET capture_mode = 'unknown' WHERE raw_id = ?", (raw_id,))
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash, raw_id, "raw_payload", retired_recorded_path, len(payload), 1),
        )
        conn.execute(
            "INSERT INTO raw_container_coordinates VALUES (?, 'zip-v2', ?, ?)",
            (raw_id, 0, 0),
        )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verified
    assert result.verification["missing_canonical_blob_count"] == 0
    assert result.verification["recoverable_source_blob_count"] == 1


def test_full_evidence_backup_reacquires_legacy_zip_row_without_coordinates(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A legacy ZIP row still replays through acquisition without coordinate metadata."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "legacy.zip"
    records = [
        {"metadata": "bundle sibling"},
        {"id": "first", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
        {"id": "recoverable", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
    ]
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("conversations.json", json.dumps(records, separators=(",", ":")))
    payload = dumps_bytes(records[2])
    blob_hash = hashlib.sha256(payload).digest()
    recorded_path = f"{source_path}:conversations.json"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("DROP TABLE raw_container_coordinates")
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hashlib.sha256(payload).hexdigest(),
                "chatgpt-export",
                "recoverable",
                recorded_path,
                1,
                blob_hash,
                len(payload),
                1,
                "passed",
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash, hashlib.sha256(payload).hexdigest(), "raw_payload", recorded_path, len(payload), 1),
        )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verified
    assert result.verification["recoverable_source_blob_count"] == 1


def test_zip_replay_derives_member_and_split_from_empty_legacy_coordinates(tmp_path: Path) -> None:
    """Legacy rows use the recorded member suffix and source index."""
    source_path = tmp_path / "legacy-empty-coordinate.zip"
    records = [
        {"metadata": "bundle sibling"},
        {"id": "first", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
        {"id": "recoverable", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
    ]
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("conversations.json", json.dumps(records, separators=(",", ":")))
    recorded_path = f"{source_path}:conversations.json"
    expected = dumps_bytes(records[2])

    payload, error = zip_reacquisition_payload(
        {
            "coordinate_format": "",
            "entry_ordinal": None,
            "split_index": None,
            "raw_id": "legacy-raw-id",
            "source_path": recorded_path,
            "source_index": 1,
            "blob_hash": hashlib.sha256(expected).hexdigest(),
            "capture_mode": "chatgpt",
        },
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert payload == expected


@pytest.mark.parametrize(
    ("origin", "revision_kind"),
    [
        ("codex-session", "full"),
        ("claude-code-session", "full"),
        ("claude-code-session", "unknown"),
        ("hermes-session", "unknown"),
    ],
)
def test_backup_replays_historical_full_snapshot_prefix(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    origin: str,
    revision_kind: str,
) -> None:
    """A later append leaves the writer's full-file observation at the prefix."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / f"{origin}.jsonl"
    historical = b'{"type":"session_meta","payload":{"id":"snapshot"}}\n'
    source_path.write_bytes(historical + b'{"type":"message","payload":{"text":"later"}}\n')
    blob_hash = hashlib.sha256(historical).digest()
    raw_id = f"historical-{origin}-{revision_kind}"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status, revision_kind
            ) VALUES (?, ?, ?, 0, ?, ?, 1, 'passed', ?)""",
            (raw_id, origin, str(source_path), blob_hash, len(historical), revision_kind),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert len(proofs) == 1
    assert proofs[0]["kind"] == "historical_snapshot_prefix_sha256"
    assert unproven == []


def test_backup_falls_back_to_current_decoder_after_prefix_mismatch(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed historical prefix can still be recovered by normal decoding."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "grown.jsonl"
    historical = b'{"id":"stale"}\n'
    decoded = b'{"id":"recoverable"}\n'
    source_path.write_bytes(historical + decoded)
    blob_hash = hashlib.sha256(decoded).digest()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status, revision_kind
            ) VALUES ('decoder-fallback', 'hermes-session', ?, 0, ?, ?, 1, 'passed', 'full')""",
            (str(source_path), blob_hash, len(decoded)),
        )

    def decode_current_payload(*args: object, **kwargs: object) -> tuple[bytes, None]:
        return decoded, None

    monkeypatch.setattr(backup_mod, "_current_raw_payload_bytes", decode_current_payload)
    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert len(proofs) == 1
    assert proofs[0]["kind"] == "direct_file_sha256"
    assert unproven == []


def test_backup_types_legacy_codex_append_without_window(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A legacy Codex append without offsets remains explicitly unproven."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "legacy-codex-append.jsonl"
    source_path.write_bytes(b'{"type":"session_meta"}\n{"type":"event_msg"}\n')
    payload = b'{"type":"event_msg"}\n'
    blob_hash = hashlib.sha256(payload).digest()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status, revision_kind
            ) VALUES ('legacy-codex-append', 'codex-session', ?, -1, ?, ?, 1, 'passed', 'unknown')""",
            (str(source_path), blob_hash, len(payload)),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert proofs == []
    assert unproven[0]["kind"] == "legacy_append_window_missing"
    assert unproven[0]["reason"] == "legacy_append_window_missing"


@pytest.mark.parametrize("origin", ["codex-session", "claude-code-session"])
def test_backup_replays_legacy_append_from_preceding_full_snapshot(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    origin: str,
) -> None:
    """Pre-envelope append rows use the immediately preceding full snapshot as their window."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / f"legacy-{origin}.jsonl"
    identity = "019f4d42-1794-7280-b329-ed31152df30e"
    prefix = dumps_bytes({"type": "session_meta", "payload": {"id": identity}}) + b"\n"
    append = b'{"type":"event_msg","payload":{"message":"legacy"}}\n'
    source_path.write_bytes(prefix + append)
    if origin == "codex-session":
        from polylogue.sources.live.batch_support import codex_append_payload

        expected = codex_append_payload(append, identity=identity, legacy_header=True)
        capture_mode = "codex"
    else:
        expected = append
        capture_mode = None
    prior_hash = hashlib.sha256(prefix).digest()
    append_hash = hashlib.sha256(expected).digest()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status, revision_kind
            ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, 1, 'passed', 'full')""",
            (f"prior-{origin}", origin, capture_mode, identity, str(source_path), prior_hash, len(prefix)),
        )
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status, revision_kind
            ) VALUES (?, ?, ?, ?, ?, -1, ?, ?, 2, 'passed', 'unknown')""",
            (f"append-{origin}", origin, capture_mode, identity, str(source_path), append_hash, len(expected)),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={append_hash.hex()},
        unproven=unproven,
    )

    assert len(proofs) == 1
    assert proofs[0]["kind"] == "historical_append_segment_sha256"
    assert proofs[0]["append_start_offset"] == str(len(prefix))
    assert proofs[0]["append_end_offset"] == str(len(prefix) + len(append))
    assert unproven == []


def test_full_evidence_backup_reacquires_live_append_segment_after_file_grows(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An append raw is proved from its recorded byte window, not file length."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "live.jsonl"
    prefix = b'{"id":"prior"}\n'
    append = b'{"id":"new"}\n'
    source_path.write_bytes(prefix + append)
    blob_hash = hashlib.sha256(append).digest()
    raw_id = "append-recoverable"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status, revision_kind,
                append_start_offset, append_end_offset
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "codex-session",
                "codex",
                "session",
                str(source_path),
                0,
                blob_hash,
                len(append),
                1,
                "passed",
                "append",
                len(prefix),
                len(prefix) + len(append),
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash, raw_id, "raw_payload", str(source_path), len(append), 1),
        )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verification["recoverable_source_blob_count"] == 1
    source_path.write_bytes(prefix + append + b'{"id":"later"}\n')
    verified_after_growth = backup_mod._verify_archive_file_set_backup(Path(result.output_path or ""))
    assert verified_after_growth["ok"] is True
    assert verified_after_growth["missing_canonical_blob_count"] == 0


def test_backup_replays_historical_codex_append_header(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Codex append proof reproduces the historical writer payload exactly."""
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "codex.jsonl"
    identity = "019f4d42-1794-7280-b329-ed31152df30e"
    prefix = dumps_bytes({"type": "session_meta", "payload": {"id": identity}})
    append = b'{"type":"event_msg","payload":{"type":"agent_message","message":"hello"}}\n'
    source_payload = prefix + b"\n" + append
    source_path.write_bytes(source_payload)
    from polylogue.sources.live.batch_support import codex_append_payload

    expected_payload = codex_append_payload(append, identity=identity, legacy_header=True)
    blob_hash = hashlib.sha256(expected_payload).digest()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status, revision_kind,
                append_start_offset, append_end_offset
            ) VALUES (?, 'codex-session', 'codex', ?, 0, ?, ?, 1, 'passed', 'append', ?, ?)""",
            (
                "codex-historical-header",
                str(source_path),
                blob_hash,
                len(expected_payload),
                len(prefix) + 1,
                len(source_payload),
            ),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert len(proofs) == 1
    assert proofs[0]["kind"] == "live_append_segment_sha256"
    assert unproven == []


def test_backup_reanchors_dead_root_before_zip_member_replay(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A stale absolute ZIP root is replaced by the active archive root."""
    archive_root = workspace_env["archive_root"]
    inbox = archive_root / "inbox"
    inbox.mkdir()
    zip_path = inbox / "bundle.zip"
    member_payload = dumps_bytes({"id": "recoverable"})
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("conversation.json", member_payload)
    blob_hash = hashlib.sha256(member_payload).digest()
    stale_source_path = f"{tmp_path / 'old-clone' / 'inbox' / 'bundle.zip'}:conversation.json"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status
            ) VALUES (?, 'chatgpt-export', ?, 0, ?, ?, 1, 'passed')""",
            ("zip-dead-root", stale_source_path, blob_hash, len(member_payload)),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert len(proofs) == 1
    assert proofs[0]["kind"] == "zip_reacquired_payload"
    assert proofs[0]["source_path"] == f"{zip_path}:conversation.json"
    assert unproven == []


def test_backup_archive_full_evidence_profile_includes_all_tiers(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    for name in ("index.db", "ops.db"):
        with sqlite3.connect(archive_root / name) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS marker (value TEXT NOT NULL)")
            conn.execute("INSERT INTO marker VALUES (?)", (name,))

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok
    assert result.backup_profile == "full_evidence"
    assert result.omitted_tiers == []
    assert result.verified is True
    assert result.verification["tier_integrity"] == _tier_integrity(*ARCHIVE_TIER_SPECS)
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert {path.name for path in backup_root.glob("*.db")} == {spec.filename for spec in ARCHIVE_TIER_SPECS.values()}
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "full_evidence"
    assert manifest["included_tiers"] == _tier_files(*ARCHIVE_TIER_SPECS)
    assert manifest["omitted_tiers"] == []


def test_full_evidence_backup_restores_index_only_attachment_blob(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    archive_root = workspace_env["archive_root"]
    payload = b"index-only attachment evidence"
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="backup-index-attachment",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="attachment", position=0)],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="a1",
                message_provider_id="m1",
                inline_bytes=payload,
            )
        ],
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        archive.write_parsed(session)

    blob_hash = hashlib.sha256(payload).hexdigest()
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM blob_refs WHERE blob_hash = ?", (bytes.fromhex(blob_hash),)).fetchone()[
                0
            ]
            == 0
        )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok
    assert result.verified
    assert result.verification["canonical_blobs_resolved"] is True
    backup_root = Path(result.output_path or "")
    assert (backup_root / "blob" / blob_hash[:2] / blob_hash[2:]).read_bytes() == payload
    inventory = json.loads((backup_root / "blob-inventory.json").read_text(encoding="utf-8"))
    item = next(row for row in inventory if row["blob_hash"] == blob_hash)
    assert item["protection"] == ["committed"]


def test_full_evidence_backup_preserves_index_attachment_for_historical_source_schema(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A v21 source fallback must retain the readable index attachment owner.

    Anti-vacuity: removing ``raw_hook_events.blob_hash`` makes the current
    source capability projection non-authoritative.  The attachment exists
    only in index.db, so a fallback that substitutes source carriers for the
    complete projection produces a verified-looking backup with missing bytes.
    """
    archive_root = workspace_env["archive_root"]
    payload = b"historical-source index-only attachment evidence"
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="backup-historical-index-attachment",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="attachment", position=0)],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="a1",
                message_provider_id="m1",
                inline_bytes=payload,
            )
        ],
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        archive.write_parsed(session)

    blob_hash = hashlib.sha256(payload).hexdigest()
    with sqlite3.connect(archive_root / "source.db") as source:
        source.execute("DROP INDEX idx_raw_hook_events_source_hash")
        source.execute("ALTER TABLE raw_hook_events DROP COLUMN blob_hash")
        source.execute("PRAGMA user_version = 21")
        assert source.execute(
            "SELECT COUNT(*) FROM blob_refs WHERE blob_hash = ?", (bytes.fromhex(blob_hash),)
        ).fetchone() == (0,)

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verified
    assert result.verification["reference_evidence_resolved"] is True
    backup_root = Path(result.output_path or "")
    assert (backup_root / "blob" / blob_hash[:2] / blob_hash[2:]).read_bytes() == payload
    evidence = json.loads((backup_root / "blob-reference-evidence.json").read_text(encoding="utf-8"))
    assert evidence["index_attachment_hashes"] == [blob_hash]


def test_backup_attachment_oracle_rejects_a_projection_that_omits_readable_index_owner(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent attachment evidence must be able to contradict copying.

    Anti-vacuity: a copy/verification pair derived solely from the same
    liveness projection would accept this deliberately incomplete copy plan.
    """
    archive_root = workspace_env["archive_root"]
    payload = b"independent backup attachment oracle"
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="backup-attachment-oracle",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="attachment", position=0)],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="a1",
                message_provider_id="m1",
                inline_bytes=payload,
            )
        ],
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        archive.write_parsed(session)

    original_inventory = backup_mod._inventory_from_liveness

    def omit_attachment(
        projection: BlobLivenessProjection,
        reservations: set[str],
    ) -> dict[str, set[str]]:
        inventory = original_inventory(projection, reservations)
        inventory.pop(hashlib.sha256(payload).hexdigest(), None)
        return inventory

    monkeypatch.setattr(backup_mod, "_inventory_from_liveness", omit_attachment)

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok is False
    assert result.verified is False
    assert result.verification["reference_evidence_resolved"] is True
    assert result.verification["canonical_blobs_resolved"] is False
    assert result.verification["missing_canonical_blob_count"] == 1


def test_backup_creation_oracle_rejects_projection_omitting_independent_attachment(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copying side reads index attachments independently before any blob copy."""
    archive_root = workspace_env["archive_root"]
    payload = b"creation-side independent attachment oracle"
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="backup-creation-attachment-oracle",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="attachment", position=0)],
        attachments=[ParsedAttachment(provider_attachment_id="a1", message_provider_id="m1", inline_bytes=payload)],
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        archive.write_parsed(session)
    omitted_hash = hashlib.sha256(payload).hexdigest()
    original_projection = backup_mod._source_blob_liveness_projection

    def omit_from_projection(source_db: Path, *, index_db: Path | None) -> tuple[BlobLivenessProjection, set[str]]:
        projection, reservations = original_projection(source_db, index_db=index_db)
        return (
            BlobLivenessProjection(
                frozenset(blob_hash for blob_hash in projection.live_hashes if blob_hash != omitted_hash),
                owner_hashes=projection.owner_hashes,
            ),
            reservations,
        )

    monkeypatch.setattr(backup_mod, "_source_blob_liveness_projection", omit_from_projection)
    with pytest.raises(RuntimeError, match="omitted independent attachment evidence"):
        backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)


@pytest.mark.parametrize(
    "evidence, message",
    [
        (None, "missing or has an unknown format"),
        ({"format": "polylogue-blob-reference-evidence-v1"}, "invalid owner payloads"),
        (
            {
                "format": "polylogue-blob-reference-evidence-v1",
                "source_owner_hashes": {},
                "index_attachment_evidence": "not_consulted",
                "index_attachment_hashes": ["0" * 64],
            },
            "unconsulted index attachments",
        ),
        (
            {
                "format": "polylogue-blob-reference-evidence-v1",
                "source_owner_hashes": {"source.db.raw_sessions": "not-a-list"},
                "index_attachment_evidence": "consulted",
                "index_attachment_hashes": [],
            },
            "invalid source owner payloads",
        ),
    ],
)
def test_backup_reference_evidence_refuses_malformed_payloads(evidence: object, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        backup_mod._expected_blob_hashes_from_evidence(evidence)


def test_backup_attachment_oracle_refuses_missing_invalid_and_unreadable_evidence(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    with sqlite3.connect(missing) as conn:
        conn.execute("CREATE TABLE attachments (attachment_id TEXT PRIMARY KEY) STRICT")
    with pytest.raises(RuntimeError, match="missing columns"):
        backup_mod._index_attachment_hashes(missing)

    invalid = tmp_path / "invalid.db"
    with sqlite3.connect(invalid) as conn:
        conn.execute("CREATE TABLE attachments (blob_hash BLOB) STRICT")
        conn.execute("INSERT INTO attachments VALUES (X'00')")
    with pytest.raises(RuntimeError, match="invalid blob_hash"):
        backup_mod._index_attachment_hashes(invalid)

    unreadable = tmp_path / "unreadable.db"
    unreadable.mkdir()
    with pytest.raises(RuntimeError, match="unreadable"):
        backup_mod._index_attachment_hashes(unreadable)


def test_backup_verification_refuses_missing_reference_evidence_artifact(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=False)
    assert result.ok and result.output_path is not None
    (Path(result.output_path) / "blob-reference-evidence.json").unlink()
    backup_mod._verify_backup_result(result)
    assert result.verified is False
    assert result.ok is False
    assert "reference evidence is missing" in str(result.error)


def test_backup_archive_full_evidence_profile_treats_ops_as_optional(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    (archive_root / "ops.db").unlink()

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok
    assert result.backup_profile == "full_evidence"
    assert result.omitted_tiers == ["ops.db"]
    assert result.verified is True
    assert result.verification["tier_integrity"] == _tier_integrity(
        ArchiveTier.SOURCE, ArchiveTier.INDEX, ArchiveTier.EMBEDDINGS, ArchiveTier.USER, ArchiveTier.AUDIT
    )
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert not (backup_root / "ops.db").exists()
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_tiers"] == _tier_files(
        ArchiveTier.SOURCE, ArchiveTier.INDEX, ArchiveTier.EMBEDDINGS, ArchiveTier.USER, ArchiveTier.AUDIT
    )
    assert manifest["omitted_tiers"] == ["ops.db"]


def test_backup_archive_user_overlays_profile_copies_only_user_tier(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_setup(workspace_env)
    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=True)

    assert result.ok
    assert result.backup_profile == "user_overlays"
    assert result.verified is True
    assert result.verification["tier_integrity"] == _tier_integrity(ArchiveTier.USER, ArchiveTier.AUDIT)
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert (backup_root / "user.db").exists()
    assert not (backup_root / "source.db").exists()
    assert not (backup_root / "index.db").exists()
    assert not (backup_root / "embeddings.db").exists()
    assert not (backup_root / "ops.db").exists()
    assert not (backup_root / "blob").exists()
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "user_overlays"
    assert manifest["included_tiers"] == _tier_files(ArchiveTier.USER, ArchiveTier.AUDIT)
    assert manifest["omitted_tiers"] == _tier_files(
        ArchiveTier.SOURCE, ArchiveTier.INDEX, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS
    )


def test_backup_archive_verify_false_does_not_write_success_receipt(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_setup(workspace_env)

    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=False)

    assert result.ok
    assert result.verified is False
    assert result.output_path is not None
    assert not (Path(result.output_path) / "verification-receipt.json").exists()


def test_backup_archive_diagnostics_profile_copies_only_ops_tier(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_setup(workspace_env)
    result = backup_archive(output_dir=tmp_path / "backups", profile="diagnostics_bundle", verify=True)

    assert result.ok
    assert result.backup_profile == "diagnostics_bundle"
    assert result.verified is True
    assert result.verification["tier_integrity"] == {"ops": True}
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert (backup_root / "ops.db").exists()
    assert not (backup_root / "source.db").exists()
    assert not (backup_root / "index.db").exists()
    assert not (backup_root / "embeddings.db").exists()
    assert not (backup_root / "user.db").exists()
    assert not (backup_root / "blob").exists()
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "diagnostics_bundle"
    assert manifest["included_tiers"] == ["ops.db"]
    assert manifest["omitted_tiers"] == _tier_files(
        ArchiveTier.SOURCE, ArchiveTier.INDEX, ArchiveTier.EMBEDDINGS, ArchiveTier.USER, ArchiveTier.AUDIT
    )


def test_backup_verification_scratch_stays_near_backup_output(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_setup(workspace_env)
    backup_parent = tmp_path / "backups"

    result = backup_archive(output_dir=backup_parent, profile="user_overlays", verify=True)

    assert result.ok
    assert result.verified is True
    scratch_parent = Path(str(result.verification["scratch_parent"]))
    assert scratch_parent == backup_parent
    assert not str(scratch_parent).startswith("/tmp/")


def test_backup_result_formats_non_default_omissions_neutrally() -> None:
    from polylogue.daemon.backup import BackupResult, format_backup_result

    lines = format_backup_result(
        BackupResult(ok=True, output_path="/tmp/backup", backup_profile="user_overlays", omitted_tiers=["source.db"])
    )

    assert "  Omitted by profile: source.db" in lines
    assert all("rebuildable/disposable" not in line for line in lines)


def test_backup_missing_blob_warnings_are_bounded(tmp_path: Path) -> None:
    hashes = tuple(f"{idx:064x}" for idx in range(25))
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    with sqlite3.connect(source_db) as conn:
        for idx, blob_hash in enumerate(hashes):
            conn.execute(
                """INSERT INTO raw_sessions
                (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
                VALUES (?, 'codex-session', ?, 0, ?, 1, 1)""",
                (f"raw-{idx}", f"/raw/{idx}.jsonl", bytes.fromhex(blob_hash)),
            )
            conn.execute(
                """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
                VALUES (?, ?, 'attachment', ?, 1, 1)""",
                (bytes.fromhex(blob_hash), f"raw-{idx}", f"/raw/{idx}.jsonl"),
            )
    warnings: list[str] = []
    backup_root = tmp_path / "backup"
    backup_root.mkdir()
    count, size, debt = backup_mod._copy_referenced_blobs(
        source_db=source_db,
        source_blob_root=archive_root / "blob",
        index_db=None,
        backup_root=tmp_path / "backup",
        warnings=warnings,
    )

    assert count == 0
    assert size == 0
    assert debt.missing_referenced_blobs == 25
    assert len(warnings) == 1
    assert "25 total" in warnings[0]
    assert "blob-reference-debt.json" in warnings[0]
    assert hashes[0] in warnings[0]
    assert hashes[9] in warnings[0]
    assert hashes[10] not in warnings[0]
    debt_payload = json.loads((backup_root / "blob-reference-debt.json").read_text())
    assert debt_payload["missing_referenced_blobs"] == 25
    assert debt_payload["sample"] == list(hashes[:10])


def test_full_evidence_backup_scopes_blob_attestation_to_latest_sealed_generation(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Frozen-generation references are declared expected, not backup debt.

    Anti-vacuity: removing the generation predicate makes the old missing blob
    participate in the copied inventory and causes verification to fail.
    """
    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    clean_payload = b"reacquired source generation"
    clean_hash = hashlib.sha256(clean_payload).digest()
    frozen_hash = hashlib.sha256(b"deliberately pruned frozen source").digest()
    store = BlobStore(archive_root / "blob")
    store.write_from_bytes(clean_payload)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "INSERT INTO source_generations VALUES ('frozen', ?, 'path', 1, 10, 1)",
            ("a" * 64,),
        )
        conn.execute(
            "INSERT INTO source_generations VALUES ('reacquired', ?, 'path', 1, 20, 2)",
            ("b" * 64,),
        )
        for generation, raw_id, blob_hash in (
            ("frozen", "old-raw", frozen_hash),
            ("reacquired", "clean-raw", clean_hash),
        ):
            conn.execute(
                """INSERT INTO raw_sessions
                   (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
                   VALUES (?, 'codex-session', ?, 0, ?, ?, 1)""",
                (raw_id, f"/{raw_id}.jsonl", blob_hash, len(clean_payload)),
            )
            conn.execute(
                """INSERT INTO source_items
                   (source_generation_id, source_item_id, logical_coordinate, addressing_mode,
                    disposition, outcome_code, stage, raw_id, observed_at_ms, updated_at_ms)
                   VALUES (?, ?, ?, 'path', 'admitted', 'success', 'done', ?, 1, 1)""",
                (generation, f"item-{generation}", f"{generation}.jsonl", raw_id),
            )
            conn.execute(
                """INSERT INTO blob_refs
                   (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
                   VALUES (?, ?, 'raw_payload', ?, ?, 1)""",
                (blob_hash, raw_id, f"/{raw_id}.jsonl", len(clean_payload)),
            )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok
    assert result.verified
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    manifest = json.loads((backup_root / "manifest.json").read_text())
    assert manifest["source_generation_id"] == "reacquired"
    assert manifest["blob_reference_debt"]["missing_referenced_blobs"] == 0
    assert (backup_root / "blob" / clean_hash.hex()[:2] / clean_hash.hex()[2:]).exists()
    assert not (backup_root / "blob" / frozen_hash.hex()[:2] / frozen_hash.hex()[2:]).exists()


def test_backup_includes_reserved_blob_and_verifies_exact_hash_inventory(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    archive_root = workspace_env["archive_root"]
    blob_root = archive_root / "blob"
    publisher = ArchiveBlobPublisher(archive_root / "source.db", blob_root)
    payload = b"reservation-only backup evidence"
    blob_hash, _ = publisher.write_from_bytes(payload)
    publisher.flush()

    result = backup_archive(output_dir=tmp_path / "backups", verify=True)

    assert result.ok
    assert result.verified
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    copied = backup_root / "blob" / blob_hash[:2] / blob_hash[2:]
    assert copied.read_bytes() == payload
    inventory = json.loads((backup_root / "blob-inventory.json").read_text(encoding="utf-8"))
    assert inventory == [
        {
            "blob_hash": blob_hash,
            "protection": ["reserved"],
            "size_bytes": len(payload),
        }
    ]

    copied.write_bytes(b"x" * len(payload))
    verification = backup_mod._verify_archive_file_set_backup(backup_root)
    assert verification["ok"] is False
    assert verification["blob_inventory_exact"] is False


def test_backup_verification_rejects_missing_source_references_and_reservations(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    archive_root = workspace_env["archive_root"]
    source_db = archive_root / "source.db"
    missing_reference_hash = hashlib.sha256(b"missing referenced blob").digest()
    missing_reservation_hash = hashlib.sha256(b"missing reserved blob").digest()
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "missing-raw",
                "codex-session",
                "missing",
                "/tmp/missing.jsonl",
                0,
                missing_reference_hash,
                23,
                1,
                "passed",
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (missing_reference_hash, "missing-raw", "raw_payload", "/tmp/missing.jsonl", 23, 1),
        )
        conn.execute(
            """
            INSERT INTO blob_publication_reservations (
                publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("missing-publication", missing_reservation_hash, 21, "publisher", 1),
        )

    result = backup_archive(output_dir=tmp_path / "backups", verify=True)

    assert result.ok is False
    assert result.verified is False
    assert result.verification["canonical_blobs_resolved"] is False
    assert result.verification["missing_canonical_blob_count"] == 2
    assert result.verification["unproven_source_blob_count"] == 1
    assert result.verification["blob_inventory_exact"] is True
    evidence = json.loads((Path(result.output_path or "") / "blob-reference-evidence.json").read_text(encoding="utf-8"))
    assert evidence["recoverability_unproven"] == [
        {
            "blob_hash": missing_reference_hash.hex(),
            "kind": "source_missing",
            "raw_id": "missing-raw",
            "reason": "source_missing",
            "source_path": "/tmp/missing.jsonl",
        }
    ]


def test_pre_generation_source_uses_operator_declared_absence_without_vacuity(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A v30 source may excuse only an operator-declared missing blob.

    Anti-vacuity: before writing the declaration, verification must reject the
    missing reference; after writing it, deleting the retained blob must still
    reject the backup. A reservation remains outside the effective source
    scope, so restoring the verifier's reference union must make the empty
    effective-scope assertion green and turn this test red.
    """
    db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]
    source_db = archive_root / "source.db"
    retained_payload = b"pre-generation retained source"
    absent_payload = b"pre-generation deliberately absent source"
    reserved_payload = b"pre-generation publication reservation"
    retained_hash = hashlib.sha256(retained_payload).digest()
    absent_hash = hashlib.sha256(absent_payload).digest()
    reserved_hash = hashlib.sha256(reserved_payload).digest()
    BlobStore(archive_root / "blob").write_from_bytes(retained_payload)
    BlobStore(archive_root / "blob").write_from_bytes(reserved_payload)
    with sqlite3.connect(source_db) as conn:
        conn.execute("DROP VIEW IF EXISTS source_item_reconciliation")
        conn.execute("DROP TABLE IF EXISTS source_items")
        conn.execute("DROP TABLE IF EXISTS source_generations")
        conn.execute("PRAGMA user_version = 30")
        for raw_id, blob_hash, payload in (
            ("retained-raw", retained_hash, retained_payload),
            ("absent-raw", absent_hash, absent_payload),
        ):
            conn.execute(
                """INSERT INTO raw_sessions
                   (raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, validation_status)
                   VALUES (?, 'codex-session', ?, ?, 0, ?, ?, 1, 'passed')""",
                (raw_id, raw_id, f"/{raw_id}.jsonl", blob_hash, len(payload)),
            )
            conn.execute(
                """INSERT INTO blob_refs
                   (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
                   VALUES (?, ?, 'raw_payload', ?, ?, 1)""",
                (blob_hash, raw_id, f"/{raw_id}.jsonl", len(payload)),
            )
        conn.execute(
            """INSERT INTO blob_publication_reservations
               (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms)
               VALUES ('pre-generation-reservation', ?, ?, 'test-publisher', 1)""",
            (reserved_hash, len(reserved_payload)),
        )

    without_assertion = backup_archive(output_dir=tmp_path / "without-assertion", verify=True)
    assert without_assertion.ok is False
    assert without_assertion.verification["missing_canonical_blob_count"] == 1

    assertion_path = archive_root / backup_mod._SOURCE_DECLARED_ABSENT_FILE
    assertion_path.write_text(
        json.dumps(
            {
                "format": backup_mod._SOURCE_DECLARED_ABSENT_FORMAT,
                "freeze_authority": backup_mod._SOURCE_DECLARED_ABSENT_AUTHORITY,
                "source_db_sha256": hashlib.sha256(source_db.read_bytes()).hexdigest(),
                "declared_absent_blob_hashes": [absent_hash.hex()],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with_assertion = backup_archive(output_dir=tmp_path / "with-assertion", verify=True)
    assert with_assertion.ok
    assert with_assertion.verified
    assert with_assertion.verification["missing_canonical_blob_count"] == 0
    assert with_assertion.verification["source_effective_scope_nonempty"] is True
    assert Path(with_assertion.output_path or "", backup_mod._SOURCE_DECLARED_ABSENT_FILE).is_file()
    with sqlite3.connect(source_db) as conn:
        assert (
            validate_migration_backup_manifest(
                Path(with_assertion.output_path or "") / "manifest.json",
                ArchiveTier.SOURCE,
                connection=conn,
            ).name
            == "verification-receipt.json"
        )

    generation_backup_root = Path(with_assertion.output_path or "")
    with sqlite3.connect(generation_backup_root / "source.db") as conn:
        conn.execute("CREATE TABLE source_generations (marker INTEGER)")
        conn.execute("CREATE TABLE source_items (marker INTEGER)")
    with pytest.raises(RuntimeError, match="only valid before source generations exist"):
        backup_mod._verify_archive_file_set_backup(generation_backup_root)

    assertion = json.loads(assertion_path.read_text(encoding="utf-8"))
    assertion["declared_absent_blob_hashes"] = []
    assertion_path.write_text(json.dumps(assertion, indent=2, sort_keys=True), encoding="utf-8")
    empty_declared_set = backup_archive(output_dir=tmp_path / "empty-declared-set", verify=True)
    assert empty_declared_set.ok is False
    assert "empty declared set" in str(empty_declared_set.error)

    assertion["declared_absent_blob_hashes"] = [absent_hash.hex(), "0" * 64]
    assertion_path.write_text(json.dumps(assertion, indent=2, sort_keys=True), encoding="utf-8")
    assert json.loads(assertion_path.read_text(encoding="utf-8"))["declared_absent_blob_hashes"] == [
        absent_hash.hex(),
        "0" * 64,
    ]
    unreferenced_declaration = backup_archive(output_dir=tmp_path / "unreferenced-declaration", verify=True)
    assert unreferenced_declaration.ok is False, unreferenced_declaration.verification
    assert unreferenced_declaration.verification["reference_evidence_resolved"] is False

    assertion["declared_absent_blob_hashes"] = [absent_hash.hex(), retained_hash.hex()]
    assertion_path.write_text(json.dumps(assertion, indent=2, sort_keys=True), encoding="utf-8")
    empty_effective_scope = backup_archive(output_dir=tmp_path / "empty-effective-scope", verify=True)
    assert empty_effective_scope.ok is False
    assert empty_effective_scope.verification["missing_canonical_blob_count"] == 0
    assert empty_effective_scope.verification["source_effective_scope_nonempty"] is False

    assertion["declared_absent_blob_hashes"] = [absent_hash.hex()]
    assertion_path.write_text(json.dumps(assertion, indent=2, sort_keys=True), encoding="utf-8")
    (archive_root / "blob" / retained_hash.hex()[:2] / retained_hash.hex()[2:]).unlink()
    with_unasserted_loss = backup_archive(output_dir=tmp_path / "with-unasserted-loss", verify=True)
    assert with_unasserted_loss.ok is False
    assert with_unasserted_loss.verification["missing_canonical_blob_count"] == 1


@pytest.mark.parametrize(
    "mutation, message",
    [
        pytest.param({"format": "wrong-format"}, "unknown format", id="format"),
        pytest.param({"freeze_authority": "untrusted"}, "freeze authority", id="freeze-authority"),
        pytest.param({"source_db_sha256": "0" * 64}, "different source.db bytes", id="source-db-sha256"),
        pytest.param(
            {"declared_absent_blob_hashes": ["a" * 64, "a" * 64]},
            "duplicate blob hashes",
            id="duplicate-hash",
        ),
        pytest.param(
            {"declared_absent_blob_hashes": ["not-a-blob-hash"]},
            "invalid blob hashes",
            id="invalid-hash",
        ),
    ],
)
def test_source_declared_absent_authentication_rejects_each_mutation(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    """Each declared-absence authentication mutation must refuse verification."""
    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO marker VALUES ('source')")
    assertion_path = tmp_path / backup_mod._SOURCE_DECLARED_ABSENT_FILE
    assertion: dict[str, object] = {
        "format": backup_mod._SOURCE_DECLARED_ABSENT_FORMAT,
        "freeze_authority": backup_mod._SOURCE_DECLARED_ABSENT_AUTHORITY,
        "source_db_sha256": hashlib.sha256(source_db.read_bytes()).hexdigest(),
        "declared_absent_blob_hashes": ["a" * 64],
    }
    assertion.update(mutation)
    assertion_path.write_text(json.dumps(assertion), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        backup_mod._load_source_declared_absent(source_db, assertion_path)


def test_backup_reservation_only_bytes_are_not_committed_reference_debt(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    reserved_hash = hashlib.sha256(b"receipt only").digest()
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO blob_publication_reservations
            (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms)
            VALUES ('receipt', ?, 1, 'publisher', 1)""",
            (reserved_hash,),
        )
    backup_root = tmp_path / "backup"
    backup_root.mkdir()
    warnings: list[str] = []

    count, size, debt = backup_mod._copy_referenced_blobs(
        source_db=archive_root / "source.db",
        source_blob_root=archive_root / "blob",
        index_db=None,
        backup_root=backup_root,
        warnings=warnings,
    )

    assert (count, size) == (0, 0)
    assert debt.total_references_seen == 0
    assert debt.missing_referenced_blobs == 0
    assert debt.reference_sources == {}
    assert (
        json.loads((backup_root / "blob-reference-evidence.json").read_text(encoding="utf-8"))[
            "index_attachment_evidence"
        ]
        == "not_consulted"
    )
    assert len(warnings) == 1
    assert "reservations missing blob bytes" in warnings[0]
    assert "referenced blobs missing" not in warnings[0]


def test_backup_refuses_source_schema_without_hook_evidence(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("ALTER TABLE raw_hook_events DROP COLUMN native_id")
    backup_root = tmp_path / "backup"
    backup_root.mkdir()

    with pytest.raises(RuntimeError, match="raw_hook_events is missing columns: native_id"):
        backup_mod._copy_referenced_blobs(
            source_db=archive_root / "source.db",
            source_blob_root=archive_root / "blob",
            index_db=None,
            backup_root=backup_root,
            warnings=[],
        )


def test_backup_archive_requires_precious_tiers(workspace_env: dict[str, Path], tmp_path: Path) -> None:
    archive_root = workspace_env["archive_root"]
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "source.db").unlink()

    result = backup_archive(output_dir=tmp_path / "backups")

    assert not result.ok
    assert result.backup_mode == "archive_file_set"
    assert result.output_path is None
    assert "source.db not found" in str(result.error)


def test_backup_archive_verify_marks_failed_artifact_unhealthy(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = workspace_env["archive_root"]
    archive_root.mkdir(parents=True, exist_ok=True)
    for name in ("source.db", "user.db", "embeddings.db"):
        with sqlite3.connect(archive_root / name) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS marker (value TEXT NOT NULL)")

    monkeypatch.setattr(
        "polylogue.daemon.backup._verify_archive_file_set_backup", lambda _path: {"ok": False, "error": "bad"}
    )

    result = backup_archive(output_dir=tmp_path / "backups", verify=True)

    assert not result.ok
    assert result.backup_mode == "archive_file_set"
    assert result.verified is False
    assert result.error == "bad"
    assert result.output_path is not None
    assert not (Path(result.output_path) / "verification-receipt.json").exists()


def test_backup_verification_removes_stale_receipt_after_failure(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_setup(workspace_env)
    result = backup_archive(output_dir=tmp_path / "backups", verify=False)
    assert result.output_path is not None
    receipt = Path(result.output_path) / "verification-receipt.json"
    receipt.write_text('{"verdict":"success"}', encoding="utf-8")
    monkeypatch.setattr(
        backup_mod,
        "_verify_archive_file_set_backup",
        lambda _path: {"ok": False, "error": "forced verification failure"},
    )

    backup_mod._verify_backup_result(result)

    assert result.ok is False
    assert result.verified is False
    assert not receipt.exists()


def test_backup_verification_refuses_receipt_when_backup_changes_after_scratch_restore(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_setup(workspace_env)
    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=False)
    assert result.output_path is not None
    original_verify = backup_mod._verify_archive_file_set_backup

    def verify_then_mutate(path: Path) -> dict[str, object]:
        verification = original_verify(path)
        copied_tier = path / "user.db"
        copied_tier.write_bytes(copied_tier.read_bytes() + b"x")
        return verification

    monkeypatch.setattr(backup_mod, "_verify_archive_file_set_backup", verify_then_mutate)

    backup_mod._verify_backup_result(result)

    assert result.ok is False
    assert result.verified is False
    assert "backup changed after scratch verification" in str(result.error)
    assert not (Path(result.output_path) / "verification-receipt.json").exists()
