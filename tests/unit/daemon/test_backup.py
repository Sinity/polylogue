"""Backup verification tests."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import zipfile
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.content_identity import structural_content_identity
from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode, zip_member_raw_id
from polylogue.operations import archive_backup as backup_operations
from polylogue.operations.archive_backup import backup_archive
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.sources.source_acquisition_components import zip_acquisition_fingerprint
from polylogue.storage import backup_package as backup_mod
from polylogue.storage.backup_attestation import attestation_key_path
from polylogue.storage.backup_blob_closure import (
    SOURCE_DECLARED_ABSENT_AUTHORITY,
    SOURCE_DECLARED_ABSENT_FILE,
    SOURCE_DECLARED_ABSENT_FORMAT,
    load_source_declared_absent,
)
from polylogue.storage.blob_integrity import BlobLivenessProjection
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    initialize_active_archive_root,
    initialize_archive_database,
)
from polylogue.storage.sqlite.archive_tiers.source_write import record_raw_container_coordinate
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.durable_tier_fixtures import (
    checkpoint_durable_tier,
    initialize_runtime_source_fixture,
    rebind_archive_format_fingerprints,
    refresh_archive_format_marker,
    seed_durable_tier,
)
from tests.infra.live_ingest import write_index_session
from tests.infra.storage_records import SessionBuilder, db_setup


def _record_zip_fixture_coordinate(
    conn: sqlite3.Connection,
    raw_id: str,
    container: Path,
    *,
    mode: MemberAddressingMode,
    canonical_container: Path | None = None,
    content_identity: str | None = None,
    member_name: str = "conversations.json",
) -> None:
    declared = canonical_container or container
    record_raw_container_coordinate(
        conn,
        raw_id,
        coordinate_format="zip-v2",
        entry_ordinal=0,
        split_index=0,
        addressing_mode=mode,
        content_identity=content_identity,
        captured_coordinate=CapturedZipMemberCoordinate(
            str(declared.resolve()),
            str(declared.resolve()),
            member_name,
            0,
            0,
            mode,
            hashlib.sha256(container.read_bytes()).hexdigest(),
            zip_acquisition_fingerprint(Provider.CHATGPT),
        ),
        manage_transaction=False,
    )


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
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(external, ArchiveTier.INDEX)
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


@pytest.mark.parametrize("fault", [sqlite3.OperationalError, PermissionError])
def test_backup_does_not_replace_selected_sqlite_evidence_after_a_read_fault(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: type[Exception]
) -> None:
    root = workspace_env["archive_root"]
    external = tmp_path / "selected" / "index.db"
    external.parent.mkdir()
    initialize_archive_database(external, ArchiveTier.INDEX)
    pointer = root / ".index-active-pointer"
    pointer.unlink(missing_ok=True)
    pointer.write_text(str(external) + "\n", encoding="utf-8")

    def refuse_read(path: Path) -> int:
        assert path == external
        raise fault("synthetic selected evidence read fault")

    monkeypatch.setattr(backup_mod, "_sqlite_user_version", refuse_read)
    with pytest.raises(fault):
        backup_mod._all_archive_tiers(root)


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
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(generation, ArchiveTier.INDEX)
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

    def copy_with_wrong_fingerprint(src: Path, dst: Path, *, archive_root_path: Path) -> tuple[int, dict[str, object]]:
        size, fingerprint = original_backup(src, dst, archive_root_path=archive_root_path)
        if src.name == "user.db":
            fingerprint = backup_mod._sqlite_source_fingerprint(other_db, snapshot_path=other_db, user_version=3)
        return size, fingerprint

    monkeypatch.setattr(backup_mod, "_backup_sqlite", copy_with_wrong_fingerprint)

    result = backup_archive(output_dir=tmp_path / "backups", profile="user_overlays", verify=True)

    assert result.ok is False
    assert result.verified is False
    assert "backup artifact does not match its pinned snapshot fingerprint" in str(result.error)
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


def test_full_evidence_backup_verifies_wal_mode_durable_tiers(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Verification must not manufacture the sidecars it then refuses.

    ``_backup_sqlite`` closes the SQLite backup image before publication,
    so a copied WAL-mode tier declares WAL journalling with no ``-wal``
    beside it. The archive-format lineage gate reads that copy through a
    ``mode=ro`` connection, which makes SQLite materialize an empty
    ``-shm``/``-wal`` pair inside the scratch restore. Without the cleanup the
    scratch inventory then diverges from the published backup and every
    verified full-evidence backup of a complete archive is refused.
    """
    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    for tier in ("source", "user", "audit"):
        with sqlite3.connect(archive_root / f"{tier}.db") as conn:
            conn.execute("PRAGMA journal_mode=WAL")
    for tier in ("source", "user", "audit"):
        with sqlite3.connect(f"file:{archive_root / f'{tier}.db'}?mode=ro", uri=True) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok is True, result.error
    assert result.verified is True
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert (backup_root / "verification-receipt.json").exists()
    assert not [path.name for path in backup_root.rglob("*") if path.name.endswith(("-wal", "-shm", "-journal"))]


def test_backup_verification_refuses_a_copied_omitted_tier_sidecar(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Scratch cleanup stays scoped to sidecars verification itself created.

    ``index.db`` is omitted from this profile, so a planted ``index.db-wal``
    is reached by no per-tier refusal: only the scratch artifact inventory
    sees it. Widening the cleanup to "drop every sidecar from the scratch
    copy" leaves the published root as the only witness, and the refusal
    arrives as a receipt-write failure instead -- so the error identity, not
    merely ``ok is False``, is what this pins.
    """
    db_setup(workspace_env)
    result = backup_archive(output_dir=tmp_path / "backups", verify=False)
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    (backup_root / "index.db-wal").write_bytes(b"published-sidecar")

    backup_mod._verify_backup_result(result)

    assert result.ok is False
    assert result.verified is False
    assert str(result.error).startswith("backup contains an unbound SQLite sidecar")
    assert str(result.error).endswith("index.db-wal")
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
    # ``user.db`` is one of the three tiers the archive format marker
    # fingerprints, so adding a table to it invalidates the marker exactly the
    # way a transplanted tier would. The fixture authors the tier on purpose,
    # so it restates the evidence; ``embeddings.db`` is derived and carries no
    # fingerprint.
    with seed_durable_tier(user_db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS backup_test_marks (mark_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO backup_test_marks VALUES ('mark-one')")
    refresh_archive_format_marker(archive_root)
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
    # Format birth authority and every released Source train's history are
    # retained together; neither is rebuildable cache.
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    released_trains = {
        f".maintenance-state/durable-change-trains/source-{step:03d}.json"
        for step in range(2, ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] + 1)
    }
    # The history directory exists only to carry released trains.
    history_directories = (
        {".maintenance-state", ".maintenance-state/durable-change-trains"} if released_trains else set()
    )
    expected_inventory = (
        released_trains
        | history_directories
        | {
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
            ".polylogue-format.json",
        }
    )
    assert set(artifact_inventory) == expected_inventory, sorted(set(artifact_inventory) ^ expected_inventory)
    for released in sorted(released_trains):
        train_path = Path(released)
        assert (backup_root / train_path).read_bytes() == (archive_root / train_path).read_bytes()
        assert not (backup_root / train_path.with_suffix(".json.lock")).exists()
    assert artifact_inventory["user.db"]["sha256"] == hashlib.sha256((backup_root / "user.db").read_bytes()).hexdigest()
    assert "verification-receipt.json" not in artifact_inventory
    assert {artifact["path"] for artifact in receipt["tier_artifacts"]} == {
        "source.db",
        "user.db",
        "embeddings.db",
        "audit.db",
    }
    for artifact in receipt["tier_artifacts"]:
        fingerprint = artifact["source_fingerprint"]["snapshot"]
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
def test_full_evidence_backup_carries_recovered_missing_raw_blob(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    source_kind: str,
) -> None:
    """A pruned raw blob enters the package as its reacquired exact bytes.

    Anti-vacuity: subtracting source-recoverable hashes from the required set
    instead of copying them lets verification pass with the blob absent from
    the package, and then verification of the closed package depends on the
    source file, so drifting or deleting it turns the verdict red.
    """
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
            _record_zip_fixture_coordinate(conn, raw_id, source_path, mode=MemberAddressingMode.ELEMENT_OF_CONTAINER)

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verified
    assert result.verification["missing_canonical_blob_count"] == 0
    assert result.verification["recovered_source_blob_count"] == 1
    backup_root = Path(result.output_path or "")
    blob_hex = blob_hash.hex()
    assert (backup_root / "blob" / blob_hex[:2] / blob_hex[2:]).read_bytes() == payload
    assert not (backup_root / "blob" / ".staging").exists()

    if source_kind == "direct":
        source_path.write_bytes(b"changed source bytes")
    else:
        drifted_records = [
            {"id": "recoverable", "mapping": {"node": {"message": {"author": {"role": "assistant"}}}}},
            {"id": "second", "mapping": {"node": {"message": {"author": {"role": "user"}}}}},
        ]
        with zipfile.ZipFile(source_path, "w") as archive:
            archive.writestr("conversations.json", json.dumps(drifted_records, separators=(",", ":")))
    drifted_source = backup_mod._verify_archive_file_set_backup(backup_root)
    assert drifted_source["ok"] is True
    assert drifted_source["missing_canonical_blob_count"] == 0

    source_path.unlink()
    missing_source = backup_mod._verify_archive_file_set_backup(backup_root)
    assert missing_source["ok"] is True
    assert missing_source["missing_canonical_blob_count"] == 0

    # Historical live-store debt remains evidence after exact package recovery.
    manifest_bytes = (backup_root / "manifest.json").read_bytes()
    assert json.loads(manifest_bytes)["blob_reference_debt"]["missing_referenced_blobs"] == 1
    destination = tmp_path / "restored-recovered"
    detail = backup_operations.restore_verified_backup(backup_dir=backup_root, destination=destination)
    assert detail["unrestored_referenced_blobs"] == 0
    assert detail["operational_admission"] == "ready"
    assert (destination / "blob" / blob_hex[:2] / blob_hex[2:]).read_bytes() == payload
    assert (backup_root / "manifest.json").read_bytes() == manifest_bytes


def test_migration_gate_refuses_package_missing_a_source_recoverable_blob(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A successful receipt over a package lacking a required blob is refused.

    This is the package an earlier verifier signed when a live acquisition
    file could reproduce the blob: the receipt re-hashes cleanly, yet
    restoring the package leaves a raw row whose bytes are absent once the
    file is gone. Anti-vacuity: drop the package blob-closure check from
    ``_validate_closed_backup_package`` and the gate accepts it.
    """
    from polylogue.storage.sqlite.migration_runner import MigrationError

    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "direct-source.json"
    payload = b'{"messages":["kept only by its source"]}'
    source_path.write_bytes(payload)
    blob_hash = hashlib.sha256(payload).digest()
    source_db = archive_root / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status
            ) VALUES ('source-recoverable', 'chatgpt-export', 'recoverable', ?, 0, ?, ?, 1, 'passed')""",
            (str(source_path), blob_hash, len(payload)),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash, "source-recoverable", "raw_payload", str(source_path), len(payload), 1),
        )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert result.ok, result.error
    backup_root = Path(result.output_path or "")
    source_path.unlink()
    with sqlite3.connect(source_db) as conn:
        receipt = validate_migration_backup_manifest(backup_root / "manifest.json", ArchiveTier.SOURCE, connection=conn)
    assert receipt.name == "verification-receipt.json"

    # Re-create the incomplete package an external-recoverability verifier
    # signed: the blob is gone from the package and the receipt binds that.
    blob_hex = blob_hash.hex()
    (backup_root / "blob" / blob_hex[:2] / blob_hex[2:]).unlink()
    (backup_root / "blob" / blob_hex[:2]).rmdir()
    backup_mod._write_successful_verification_receipt(
        backup_root,
        {"ok": True, "receipt_evidence": backup_mod._receipt_evidence(backup_root)},
    )
    with sqlite3.connect(source_db) as conn, pytest.raises(MigrationError, match="omits 1 blob"):
        validate_migration_backup_manifest(backup_root / "manifest.json", ArchiveTier.SOURCE, connection=conn)


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
        _record_zip_fixture_coordinate(
            conn,
            raw_id,
            source_path,
            mode=MemberAddressingMode.ELEMENT_OF_CONTAINER,
            canonical_container=tmp_path / "retired-root" / "inbox" / "export.zip",
        )

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verified
    assert result.verification["missing_canonical_blob_count"] == 0
    assert result.verification["recovered_source_blob_count"] == 1


def test_resolved_direct_path_keeps_colon_as_filename_data(tmp_path: Path) -> None:
    """Anti-vacuity: treating every colon as a ZIP separator mangles this path."""
    source = str(tmp_path / "session:export.json")
    from polylogue.storage.source_blob_restoration import retained_source_location

    assert retained_source_location({"source_path": source}, tmp_path) == (source, False)


def test_full_evidence_backup_refuses_coordinate_less_zip_row(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An operational suffix cannot establish a missing member coordinate."""
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
    # The intentionally absent coordinate receipt must remain unproven.
    with seed_durable_tier(archive_root / "source.db") as conn:
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

    assert not result.ok
    assert result.verification["recovered_source_blob_count"] == 0


def test_full_evidence_backup_streams_a_recovered_whole_zip_member(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10.F039: a missing whole-member blob is proven and recovered as a stream.

    Anti-vacuity: replay or recover the member with ``handle.read()`` and the
    guarded unbounded read below fails the backup.
    """
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "whole.zip"
    member = json.dumps(
        {"id": "whole", "mapping": {"node": {"message": {"author": {"role": "user"}}}}}, separators=(",", ":")
    ).encode()
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("conversations.json", member)
    blob_hash = hashlib.sha256(member).digest()
    recorded_path = f"{source_path}:conversations.json"
    with seed_durable_tier(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hashlib.sha256(member).hexdigest(),
                "chatgpt-export",
                "whole",
                recorded_path,
                0,
                blob_hash,
                len(member),
                1,
                "passed",
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash, hashlib.sha256(member).hexdigest(), "raw_payload", recorded_path, len(member), 1),
        )

        _record_zip_fixture_coordinate(
            conn,
            hashlib.sha256(member).hexdigest(),
            source_path,
            mode=MemberAddressingMode.WHOLE_MEMBER,
        )

    original_read = zipfile.ZipExtFile.read

    def bounded_read(self: zipfile.ZipExtFile, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            raise AssertionError("a preserved ZIP member must stream, never be read whole")
        return original_read(self, n)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", bounded_read)
    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verified
    assert result.verification["recovered_source_blob_count"] == 1


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
    with seed_durable_tier(archive_root / "source.db") as conn:
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


def test_backup_retains_prefix_mismatch_when_grown_source_fallback_fails(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A grown file cannot replace a mismatching historical prefix proof.

    The only candidate window of a full observation is its recorded-size
    prefix (``retained_blob_source_candidates``); a prefix that hashes
    differently is a typed ``hash_mismatch``, and no whole-file read proves
    the blob instead.
    """
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "grown.jsonl"
    historical = b'{"id":"stale"}\n'
    expected = b'{"id":"expected"}\n'
    source_path.write_bytes(historical + b'{"id":"later"}\n')
    blob_hash = hashlib.sha256(expected).digest()
    with seed_durable_tier(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status, revision_kind
            ) VALUES ('decoder-fallback', 'hermes-session', ?, 0, ?, ?, 1, 'passed', 'full')""",
            (str(source_path), blob_hash, len(expected)),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert proofs == []
    assert unproven == [
        {
            "blob_hash": blob_hash.hex(),
            "kind": "hash_mismatch",
            "reason": "hash_mismatch",
            "raw_id": "decoder-fallback",
            "source_path": str(source_path),
        }
    ]


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
    with seed_durable_tier(archive_root / "source.db") as conn:
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
    expected = append
    capture_mode = "codex" if origin == "codex-session" else None
    prior_hash = hashlib.sha256(prefix).digest()
    append_hash = hashlib.sha256(expected).digest()
    with seed_durable_tier(archive_root / "source.db") as conn:
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
        # Currency is durable receipt order: the full observation, then the append.
        for raw_id, blob_hash, size in (
            (f"prior-{origin}", prior_hash, len(prefix)),
            (f"append-{origin}", append_hash, len(expected)),
        ):
            conn.execute(
                "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
                (blob_hash, raw_id, "raw_payload", str(source_path), size, 1),
            )
    # Backup reads these proofs from its closed snapshot of the tier; fold
    # the seeded WAL into the main file the same way.
    checkpoint_durable_tier(archive_root / "source.db")

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={append_hash.hex()},
        unproven=unproven,
    )

    assert len(proofs) == 1, (proofs, unproven)
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
    assert result.verification["recovered_source_blob_count"] == 1
    source_path.write_bytes(prefix + append + b'{"id":"later"}\n')
    verified_after_growth = backup_mod._verify_archive_file_set_backup(Path(result.output_path or ""))
    assert verified_after_growth["ok"] is True
    assert verified_after_growth["missing_canonical_blob_count"] == 0


def test_full_evidence_backup_verifies_a_full_prefix_append_proof(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An append row whose retained blob is the whole prefix ``[0, end)`` is
    proved from that prefix, and the proof names the window it proved.

    Anti-vacuity: emit the proof with the row's own ``append_start_offset`` (3)
    and verification rebuilds ``[3, 6)``, whose hash cannot match the six-byte
    blob, so the backup's reference evidence is rejected.
    """
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "prefix.jsonl"
    snapshot = b"abcdef"
    source_path.write_bytes(snapshot)
    blob_hash = hashlib.sha256(snapshot).digest()
    raw_id = "append-full-prefix"
    # Closed, not only committed: the proof reader opens the tier immutable,
    # which refuses a live WAL.
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
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
                len(snapshot),
                1,
                "passed",
                "append",
                3,
                6,
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs VALUES (?, ?, ?, ?, ?, ?)",
            (blob_hash, raw_id, "raw_payload", str(source_path), len(snapshot), 1),
        )

    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=[],
    )
    assert [(proof["append_start_offset"], proof["append_end_offset"]) for proof in proofs] == [("0", "6")]

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok, result.error
    assert result.verification["recovered_source_blob_count"] == 1


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
    with seed_durable_tier(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status
            ) VALUES (?, 'chatgpt-export', ?, 0, ?, ?, 1, 'passed')""",
            ("zip-dead-root", stale_source_path, blob_hash, len(member_payload)),
        )
        _record_zip_fixture_coordinate(
            conn,
            "zip-dead-root",
            zip_path,
            mode=MemberAddressingMode.WHOLE_MEMBER,
            canonical_container=tmp_path / "old-clone" / "inbox" / "bundle.zip",
            member_name="conversation.json",
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


def test_backup_proves_zip_member_by_structural_identity_after_reserialization(
    workspace_env: dict[str, Path],
) -> None:
    """A harmless provider rewrite remains recoverable without byte equality."""
    archive_root = workspace_env["archive_root"]
    zip_path = archive_root / "inbox" / "structural.zip"
    zip_path.parent.mkdir()
    expected = {"id": "kept", "ordinal": 1, "mapping": {"node": {"message": {"author": {"role": "user"}}}}}
    current = {"mapping": expected["mapping"], "ordinal": 1.0, "id": expected["id"]}
    other = {"id": "other", "mapping": {"node": {"message": {"author": {"role": "user"}}}}}
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("conversations.json", json.dumps([current, other], separators=(",", ":")))
    blob_hash = hashlib.sha256(dumps_bytes(expected)).digest()
    content_identity = structural_content_identity(expected)
    source_path = f"{zip_path}:conversations.json"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, source_path, source_index, blob_hash,
                blob_size, acquired_at_ms, validation_status
            ) VALUES (?, 'chatgpt-export', 'chatgpt', ?, 0, ?, ?, 1, 'passed')""",
            ("structural-zip", source_path, blob_hash, len(dumps_bytes(expected))),
        )
        _record_zip_fixture_coordinate(
            conn,
            "structural-zip",
            zip_path,
            mode=MemberAddressingMode.ELEMENT_OF_CONTAINER,
            content_identity=content_identity,
        )
        assert (
            conn.execute("SELECT lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id='structural-zip'").fetchone()[0]
            == blob_hash.hex()
        )
    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
        immutable=False,
    )

    assert len(proofs) == 1, (proofs, unproven)
    assert proofs[0]["kind"] == "zip_reacquired_payload"
    assert proofs[0]["content_identity"] == content_identity
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
    with (
        write_lease("test.backup-fixture", archive_root=archive_root),
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
    ):
        write_index_session(archive, session)

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


def test_full_evidence_backup_keeps_index_only_attachment(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A complete source projection must retain an attachment owned by index.db.

    Anti-vacuity: the payload has no source owner or blob_refs row. Omitting
    index attachment owners from the full-evidence projection leaves its bytes
    out of the backup.
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
    with (
        write_lease("test.backup-fixture", archive_root=archive_root),
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
    ):
        write_index_session(archive, session)

    blob_hash = hashlib.sha256(payload).hexdigest()
    with seed_durable_tier(archive_root / "source.db") as source:
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
    with (
        write_lease("test.backup-fixture", archive_root=archive_root),
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
    ):
        write_index_session(archive, session)

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
    with (
        write_lease("test.backup-fixture", archive_root=archive_root),
        ArchiveStore.open_existing(archive_root, read_only=False) as archive,
    ):
        write_index_session(archive, session)
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
    assert not (backup_root / ".maintenance-state/durable-change-trains/source-002.json").exists()
    assert all(
        not name.startswith(".maintenance-state/durable-change-trains/source-")
        for name in manifest["archive_authority_files"]
    )
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
    from polylogue.operations.archive_backup import BackupResult, format_backup_result

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
    with seed_durable_tier(source_db) as conn:
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


def _seed_two_source_generations(archive_root: Path, *, earlier_payload: bytes | None) -> tuple[bytes, bytes]:
    """Seed an earlier and a newer sealed generation, each owning one raw blob.

    ``earlier_payload=None`` records the earlier generation's reference without
    its bytes, i.e. a row whose blob is missing from the store.
    """
    source_db = archive_root / "source.db"
    newer_payload = b"newer source generation"
    newer_hash = hashlib.sha256(newer_payload).digest()
    earlier_hash = hashlib.sha256(earlier_payload or b"earlier source generation, bytes missing").digest()
    store = BlobStore(archive_root / "blob")
    store.write_from_bytes(newer_payload)
    if earlier_payload is not None:
        store.write_from_bytes(earlier_payload)
    with sqlite3.connect(source_db) as conn:
        conn.execute("INSERT INTO source_generations VALUES ('earlier', ?, 'path', 1, 10, 1)", ("a" * 64,))
        conn.execute("INSERT INTO source_generations VALUES ('newer', ?, 'path', 1, 20, 2)", ("b" * 64,))
        for generation, raw_id, blob_hash in (
            ("earlier", "earlier-raw", earlier_hash),
            ("newer", "newer-raw", newer_hash),
        ):
            conn.execute(
                """INSERT INTO raw_sessions
                   (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
                   VALUES (?, 'codex-session', ?, 0, ?, ?, 1)""",
                (raw_id, f"/{raw_id}.jsonl", blob_hash, len(newer_payload)),
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
                (blob_hash, raw_id, f"/{raw_id}.jsonl", len(newer_payload)),
            )
    return earlier_hash, newer_hash


def test_full_evidence_backup_copies_every_source_generation_blob(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The backup's blob set matches the rows it copies: every generation's.

    Anti-vacuity (polylogue-mdtrf): scoping the copied set to the newest sealed
    generation leaves the earlier generation's blob out of the backup, although
    its raw row is restored with the whole ``source.db``.
    """
    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    earlier_hash, newer_hash = _seed_two_source_generations(archive_root, earlier_payload=b"earlier generation bytes")

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.ok
    assert result.verified
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    manifest = json.loads((backup_root / "manifest.json").read_text())
    assert manifest["blob_reference_debt"]["missing_referenced_blobs"] == 0
    for blob_hash in (earlier_hash, newer_hash):
        assert (backup_root / "blob" / blob_hash.hex()[:2] / blob_hash.hex()[2:]).exists()


def test_backup_counts_an_earlier_generation_blob_missing_from_the_store(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A referenced blob missing from any generation is counted debt and fails verification.

    Anti-vacuity: computing the debt report from a newest-generation projection
    reports zero missing blobs and verifies an archive whose restored rows
    reference bytes the backup does not hold.
    """
    archive_root = workspace_env["archive_root"]
    initialize_active_archive_root(archive_root)
    earlier_hash, _newer_hash = _seed_two_source_generations(archive_root, earlier_payload=None)

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)

    assert result.output_path is not None
    backup_root = Path(result.output_path)
    manifest = json.loads((backup_root / "manifest.json").read_text())
    assert manifest["blob_reference_debt"]["missing_referenced_blobs"] == 1
    debt = json.loads((backup_root / "blob-reference-debt.json").read_text())
    assert debt["sample"] == [earlier_hash.hex()]
    assert not result.verified


def test_backup_includes_reserved_blob_and_verifies_exact_hash_inventory(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    archive_root = workspace_env["archive_root"]
    blob_root = archive_root / "blob"
    publisher = ArchiveBlobPublisher(archive_root / "source.db", blob_root)
    payload = b"reservation-only backup evidence"
    blob_hash, _ = publisher.write_from_bytes(payload)
    with write_lease("test.backup-fixture", archive_root=archive_root):
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
    package = Path(result.output_path or "")
    assert json.loads((package / "manifest.json").read_text())["blob_reference_debt"]["missing_referenced_blobs"] == 1
    assert not (package / "verification-receipt.json").exists()
    destination = tmp_path / "unresolved-restore"
    with pytest.raises(FileNotFoundError):
        backup_operations.restore_verified_backup(backup_dir=package, destination=destination)
    assert not destination.exists()


def test_pre_generation_source_uses_declared_absence(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A source without generation tables may excuse only a declared missing blob.

    The pre-reset fixture stamped ``30`` to describe "before source
    generations"; this lineage has no such version and the backup evidence
    reader refuses the stamp outright. The shape that still matters is the
    catalog one -- no ``source_generations``/``source_items`` -- so the stamp
    is left at the floor and the format marker restated after the deliberate
    schema edit.

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
    with seed_durable_tier(source_db) as conn:
        conn.execute("DROP VIEW IF EXISTS source_item_reconciliation")
        conn.execute("DROP TABLE IF EXISTS source_items")
        conn.execute("DROP TABLE IF EXISTS source_generations")
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

    refresh_archive_format_marker(archive_root)

    without_assertion = backup_archive(output_dir=tmp_path / "without-assertion", verify=True)
    assert without_assertion.ok is False
    assert without_assertion.verification["missing_canonical_blob_count"] == 1

    assertion_path = archive_root / SOURCE_DECLARED_ABSENT_FILE
    assertion_path.write_text(
        json.dumps(
            {
                "format": SOURCE_DECLARED_ABSENT_FORMAT,
                "freeze_authority": SOURCE_DECLARED_ABSENT_AUTHORITY,
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
    assert Path(with_assertion.output_path or "", SOURCE_DECLARED_ABSENT_FILE).is_file()
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
    with seed_durable_tier(generation_backup_root / "source.db") as conn:
        conn.execute("CREATE TABLE source_generations (marker INTEGER)")
        conn.execute("CREATE TABLE source_items (marker INTEGER)")
    # The backup now carries this lineage's format marker, which fingerprints
    # the source tier just edited. Restate it, or the restore refuses on
    # lineage and never reaches the declared-absent scoping rule.
    rebind_archive_format_fingerprints(generation_backup_root)
    # Anything that opened the tier while restating it may have left a WAL
    # beside it, and backup publication refuses an unbound SQLite sidecar
    # before it reads any schema -- which would mask the refusal under test.
    # Every tier bootstraps in WAL, so fold each tier the restatement opened.
    for tier_path in sorted(generation_backup_root.glob("*.db")):
        checkpoint_durable_tier(tier_path)
    assert not list(generation_backup_root.glob("*.db-wal"))
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
    assertion_path = tmp_path / SOURCE_DECLARED_ABSENT_FILE
    assertion: dict[str, object] = {
        "format": SOURCE_DECLARED_ABSENT_FORMAT,
        "freeze_authority": SOURCE_DECLARED_ABSENT_AUTHORITY,
        "source_db_sha256": hashlib.sha256(source_db.read_bytes()).hexdigest(),
        "declared_absent_blob_hashes": ["a" * 64],
    }
    assertion.update(mutation)
    assertion_path.write_text(json.dumps(assertion), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_source_declared_absent(source_db, assertion_path)


def test_backup_reservation_only_bytes_are_not_committed_reference_debt(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    reserved_hash = hashlib.sha256(b"receipt only").digest()
    with seed_durable_tier(archive_root / "source.db") as conn:
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
    """A canonical blob-owner query refuses a missing carrier column.

    Removing raw_hook_events from BLOB_OWNERS would make this damaged source
    look complete. The actual owner query must fail before copying any bytes.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with seed_durable_tier(archive_root / "source.db") as conn:
        conn.execute("DROP INDEX idx_raw_hook_events_source_hash")
        conn.execute("ALTER TABLE raw_hook_events DROP COLUMN blob_hash")
    backup_root = tmp_path / "backup"
    backup_root.mkdir()

    with pytest.raises(RuntimeError):
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
        "polylogue.storage.backup_package._verify_archive_file_set_backup",
        lambda _path: {"ok": False, "error": "bad"},
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


def test_backup_proves_a_full_snapshot_append_row(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An append row whose retained blob is the whole payload is still provable.

    ``admit_raw_observation``'s append arm stores ``resolved_payload`` -- the
    complete observation -- while recording the appended tail's offsets. The
    proof path reads ``[start, end)`` and hashes that window, which can never
    equal a blob holding ``[0, end)``, so such a row is reported unproven even
    though its bytes are sitting in the source file.

    Anti-vacuity: removing the full-prefix fallback from
    ``_source_recoverability_proofs`` leaves the window hash mismatching and
    turns the ``len(proofs) == 1`` assertion red (``unproven`` gains a
    ``hash_mismatch`` entry). The companion test below pins that a genuine
    window-shaped append still proves through the window, so the fix cannot be
    "always read the full prefix".
    """
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "full-snapshot-append.jsonl"
    baseline = b'{"id":"baseline"}\n'
    tail = b'{"id":"tail"}\n'
    source_path.write_bytes(baseline + tail)
    snapshot = baseline + tail
    blob_hash = hashlib.sha256(snapshot).digest()
    with seed_durable_tier(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status, revision_kind,
                append_start_offset, append_end_offset
            ) VALUES ('full-snapshot-append', 'hermes-session', ?, 0, ?, ?, 1, 'passed', 'append', ?, ?)""",
            (str(source_path), blob_hash, len(snapshot), len(baseline), len(snapshot)),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert unproven == []
    assert len(proofs) == 1
    assert proofs[0]["blob_hash"] == blob_hash.hex()


def test_backup_still_proves_a_window_shaped_append_row(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A genuine tail-window append row proves through its window, unchanged.

    Anti-vacuity: replacing the window read with an unconditional full-prefix
    read makes the hash mismatch the retained tail-only blob and turns the
    ``len(proofs) == 1`` assertion red.
    """
    archive_root = workspace_env["archive_root"]
    source_path = tmp_path / "window-append.jsonl"
    baseline = b'{"id":"baseline"}\n'
    tail = b'{"id":"tail"}\n'
    source_path.write_bytes(baseline + tail)
    blob_hash = hashlib.sha256(tail).digest()
    with seed_durable_tier(archive_root / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, validation_status, revision_kind,
                append_start_offset, append_end_offset
            ) VALUES ('window-append', 'hermes-session', ?, 0, ?, ?, 1, 'passed', 'append', ?, ?)""",
            (str(source_path), blob_hash, len(tail), len(baseline), len(baseline) + len(tail)),
        )

    unproven: list[dict[str, str]] = []
    proofs = backup_mod._source_recoverability_proofs(
        archive_root / "source.db",
        root=archive_root,
        missing_hashes={blob_hash.hex()},
        unproven=unproven,
    )

    assert unproven == []
    assert len(proofs) == 1
    assert proofs[0]["kind"] == "live_append_segment_sha256"


@pytest.mark.parametrize("tier", [ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT])
def test_backup_evidence_opens_stamped_durable_tier_versions(
    tmp_path: Path,
    tier: ArchiveTier,
) -> None:
    """The current v2 tier is readable; unstamped and future versions are not.

    Fresh v2 archives begin at version 1 across all six tiers, so the former
    below-current backup window is empty for each durable tier.

    Anti-vacuity: keying the allowance on ``source.db`` again makes the user
    case impossible to distinguish here while the six tiers share the same
    current version; the newer-version and unstamped cases still prove the
    admission guard on each durable tier.
    """
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_FORMAT_FLOOR_VERSION, ARCHIVE_VERSION_BY_TIER

    expected = ARCHIVE_VERSION_BY_TIER[tier]
    assert expected >= ARCHIVE_FORMAT_FLOOR_VERSION == 1
    path = tmp_path / f"{tier.value}.db"
    if tier is ArchiveTier.SOURCE:
        initialize_runtime_source_fixture(path)
    else:
        initialize_archive_database(path, tier)

    with backup_mod._open_backup_readonly_connection(path, immutable=True, timeout_class="offline-bulk") as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == expected

    for refused in (expected + 1, 0):
        with sqlite3.connect(path) as stamp:
            stamp.execute(f"PRAGMA user_version = {refused}")
        checkpoint_durable_tier(path)
        with pytest.raises(SchemaSkew):
            backup_mod._open_backup_readonly_connection(path, immutable=True, timeout_class="offline-bulk")


def test_backup_verifies_a_multi_chunk_blob_without_materializing_it(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A blob larger than one read chunk verifies by streaming size and digest.

    Anti-vacuity: the payload must exceed the 1 MiB chunk, and the assertions
    are on the reported size and digest rather than merely on ``ok``. A fix
    that hashes only the first chunk, or that records ``len(chunk)`` instead
    of the running total, yields a wrong digest/size and turns
    ``blob_inventory_exact`` and the size assertion red. The corruption case
    below proves the digest is still compared rather than trivially accepted.
    """
    archive_root = workspace_env["archive_root"]
    blob_root = archive_root / "blob"
    publisher = ArchiveBlobPublisher(archive_root / "source.db", blob_root)
    # Deterministic, larger than the 1 MiB streaming chunk so the read loop
    # must run more than once.
    payload = (b"multi-chunk backup evidence " * 64) * 1024
    assert len(payload) > 1024 * 1024
    blob_hash, _ = publisher.write_from_bytes(payload)
    with write_lease("test.backup-fixture", archive_root=archive_root):
        publisher.flush()

    result = backup_archive(output_dir=tmp_path / "backups", verify=True)

    assert result.ok
    assert result.verified
    assert result.verification["blob_inventory_exact"] is True
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    inventory = json.loads((backup_root / "blob-inventory.json").read_text(encoding="utf-8"))
    assert inventory == [
        {
            "blob_hash": blob_hash,
            "protection": ["reserved"],
            "size_bytes": len(payload),
        }
    ]

    # The streamed digest is still an equality check, not a formality: flip one
    # byte beyond the first chunk and verification must refuse.
    copied = backup_root / "blob" / blob_hash[:2] / blob_hash[2:]
    corrupted = bytearray(payload)
    corrupted[1024 * 1024 + 7] ^= 0xFF
    copied.write_bytes(bytes(corrupted))
    verification = backup_mod._verify_archive_file_set_backup(backup_root)
    assert verification["ok"] is False
    assert verification["blob_inventory_exact"] is False


def _hold_daemon_pidfile(pidfile: Path, tmp_path: Path) -> tuple[int, subprocess.Popen[str]]:
    """Hold the daemon's own exclusive pidfile lock from a live process.

    The same token ``polylogued run`` takes in
    ``polylogue.daemon.cli._acquire_pidfile``, reproduced rather than patched so
    the residency probe in ``polylogue.maintenance.offline_guard`` stays under
    test rather than being stubbed out of it.
    """
    import sys

    script = tmp_path / "hold_pidfile.py"
    script.write_text(
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "os.write(fd, str(os.getpid()).encode())\n"
        "os.fsync(fd)\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(300)\n"
    )
    process = subprocess.Popen([sys.executable, str(script), str(pidfile)], stdout=subprocess.PIPE, text=True)
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    return process.pid, process


def test_embedded_backup_refused_beside_resident_daemon(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An embedded backup must hold the same archive authority as the daemon.

    The per-tier read snapshots do not independently bind sibling tiers or
    retain blobs. Removing the public ownership admission permits an embedded
    backup beside the live owner and violates that cross-tier custody.
    """
    from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError

    db_setup(workspace_env)
    root = Path(workspace_env["archive_root"])
    pid, process = _hold_daemon_pidfile(root / "daemon.pid", tmp_path)
    try:
        before = sorted((path.name, path.read_bytes()) for path in root.glob("*.db*"))
        with pytest.raises(ArchiveWriterOwnershipError) as caught:
            backup_archive(output_dir=tmp_path / "backups")
        message = str(caught.value)
        assert f"PID {pid}" in message, message
        assert caught.value.archive_root == str(root), caught.value.archive_root
        assert sorted((path.name, path.read_bytes()) for path in root.glob("*.db*")) == before
        # ``--check`` stays usable: it opens nothing writable, and it is what an
        # operator runs before stopping the daemon.
        assert backup_archive(output_dir=tmp_path / "backups", check_only=True).check_only
    finally:
        process.kill()
        process.wait(timeout=30)
        if process.stdout is not None:
            process.stdout.close()


@pytest.mark.parametrize(
    ("profile", "include_embeddings"),
    (("full_evidence", True), ("rebuildable_cache_exclude", True), ("rebuildable_cache_exclude", False)),
)
def test_verified_backup_restore_owns_destination_train_and_preserves_original_evidence(
    workspace_env: dict[str, Path], tmp_path: Path, profile: backup_mod.BackupProfile, include_embeddings: bool
) -> None:
    from polylogue.core.enums import Origin
    from polylogue.storage.sqlite import migration_runner
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session

    original_root = workspace_env["archive_root"]
    if not include_embeddings:
        (original_root / "embeddings.db").unlink()
    payload = b'{"synthetic_record":"restore-custody"}\n'
    BlobStore(original_root / "blob").write_from_bytes(payload)
    with closing(sqlite3.connect(original_root / "source.db")) as conn, conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            capture_mode=Provider.CODEX,
            source_path="/synthetic/restore-custody",
            canonical_source_path="/synthetic/restore-custody",
            source_index=0,
            native_id=None,
            payload=payload,
            acquired_at_ms=2,
        )
    result = backup_archive(output_dir=tmp_path / "backups", profile=profile, verify=True)
    assert result.ok and result.verified and result.output_path is not None
    package = Path(result.output_path)
    # The fresh v1 archive released no train, so the package carries no
    # train history and restore has none to detach.
    assert not (package / ".maintenance-state/durable-change-trains").exists()
    receipt_bytes = (package / "verification-receipt.json").read_bytes()
    destination = tmp_path / "restored"
    detail = backup_operations.restore_verified_backup(backup_dir=package, destination=destination)
    assert detail["operational_admission"] == ("ready" if profile == "full_evidence" else "degraded")
    assert detail["unrestored_purchased_tiers"] == ([] if include_embeddings else ["embeddings.db"])
    assert detail["restored_tiers"] == sorted(json.loads((package / "manifest.json").read_text())["included_tiers"])
    assert not list((destination / ".maintenance-state/durable-change-trains").glob("source-*.json"))
    assert (package / "verification-receipt.json").read_bytes() == receipt_bytes
    namespace = hashlib.sha256(str(detail["source_manifest_id"]).encode()).hexdigest()
    provenance = destination / ".archive-population-provenance" / namespace
    assert not (provenance / "original-history").exists()
    receipts = json.loads((provenance / "source.json").read_text())["original_receipts"]
    assert [item[0] for item in receipts] == [".polylogue-format.json"]
    assert (provenance / "original-backup/verification-receipt.json").read_bytes() == receipt_bytes
    for tier in ("source", "user", "audit"):
        with (
            closing(sqlite3.connect((package / f"{tier}.db").as_uri() + "?mode=ro&immutable=1", uri=True)) as source,
            closing(sqlite3.connect(destination / f"{tier}.db")) as restored,
        ):
            assert migration_runner._durable_literal_rows_digest(
                source
            ) == migration_runner._durable_literal_rows_digest(restored)
    with ArchiveStore(destination) as store:
        assert [tuple(row) for row in store.source_connection.execute("SELECT raw_id,native_id FROM raw_sessions")] == [
            (raw_id, None)
        ]
    assert (
        destination / "blob" / hashlib.sha256(payload).hexdigest()[:2] / hashlib.sha256(payload).hexdigest()[2:]
    ).read_bytes() == payload


@pytest.mark.parametrize("profile", ("user_overlays", "diagnostics_bundle"))
def test_verified_backup_restore_does_not_claim_partial_profiles_are_operational(
    workspace_env: dict[str, Path], tmp_path: Path, profile: backup_mod.BackupProfile
) -> None:
    result = backup_archive(output_dir=tmp_path / "backups", profile=profile, verify=True)
    assert result.ok and result.verified and result.output_path is not None
    destination = tmp_path / "restored"
    with pytest.raises(backup_operations.ArchiveRestoreRefusalError) as refusal:
        backup_operations.restore_verified_backup(backup_dir=Path(result.output_path), destination=destination)
    assert refusal.value.code == "restore_partial_durable_core"
    assert not destination.exists()


def test_verified_backup_restore_refuses_existing_destination_without_changing_evidence(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    result = backup_archive(
        output_dir=tmp_path / "packages",
        verify=True,
        profile="full_evidence",
        archive_root_path=workspace_env["archive_root"],
    )
    assert result.ok and result.output_path
    backup = Path(result.output_path)
    destination = tmp_path / "occupied"
    destination.mkdir()
    evidence = destination / "retained.txt"
    evidence.write_bytes(b"retained")
    receipt = (backup / "verification-receipt.json").read_bytes()
    with pytest.raises(backup_operations.ArchiveRestoreRefusalError, match="restore_destination_exists"):
        backup_operations.restore_verified_backup(backup_dir=backup, destination=destination)
    assert evidence.read_bytes() == b"retained"
    assert (backup / "verification-receipt.json").read_bytes() == receipt


def test_verified_backup_restore_refuses_modified_signed_evidence_before_destination_creation(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    result = backup_archive(
        output_dir=tmp_path / "packages",
        verify=True,
        profile="full_evidence",
        archive_root_path=workspace_env["archive_root"],
    )
    assert result.ok and result.output_path
    backup = Path(result.output_path)
    receipt_path = backup / "verification-receipt.json"
    payload = json.loads(receipt_path.read_text())
    payload["manifest_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(payload))
    changed_receipt = receipt_path.read_bytes()
    destination = tmp_path / "refused"
    from polylogue.storage.sqlite.migration_runner import MigrationError

    with pytest.raises(MigrationError):
        backup_operations.restore_verified_backup(backup_dir=backup, destination=destination)
    assert not destination.exists()
    assert receipt_path.read_bytes() == changed_receipt


def test_restore_pending_population_excludes_actual_readers_and_second_creator(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from typing import Any

    from polylogue.storage.sqlite import archive_population
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.audit_leaf import open_verified_sqlite_read_connection
    from polylogue.storage.sqlite.connection_profile import attach_database, open_readonly_connection
    from polylogue.storage.sqlite.population_admission import ArchivePopulationPendingError

    package_result = backup_archive(
        output_dir=tmp_path / "packages",
        profile="full_evidence",
        verify=True,
        archive_root_path=workspace_env["archive_root"],
    )
    assert package_result.ok and package_result.output_path
    package = Path(package_result.output_path)
    destination = tmp_path / "new-root"
    actual = archive_population.populate_authenticated_archive
    observed: list[str] = []
    failures: list[BaseException] = []

    def populate(source: Path, target: Path, **kwargs: Any) -> Any:
        proof = actual(source, target, **kwargs)

        def reader() -> None:
            try:
                with pytest.raises(ArchivePopulationPendingError):
                    with ArchiveStore.open_existing(target, read_only=True):
                        pass
                observed.append("archive")
                with pytest.raises(ArchivePopulationPendingError):
                    open_readonly_connection(target / "source.db")
                observed.append("profile")
                with pytest.raises(ArchivePopulationPendingError):
                    with open_verified_sqlite_read_connection(target / "user.db"):
                        pass
                observed.append("verified-leaf")
                with closing(sqlite3.connect(":memory:")) as query:
                    with pytest.raises(ArchivePopulationPendingError):
                        attach_database(query, target / "source.db", alias="source_tier")
                observed.append("attached-query")
                with pytest.raises(ArchivePopulationPendingError):
                    initialize_active_archive_root(target)
                observed.append("bootstrap")
                with pytest.raises(backup_operations.ArchiveRestoreRefusalError) as refusal:
                    backup_operations.restore_verified_backup(backup_dir=package, destination=target)
                assert refusal.value.code == "restore_destination_exists"
                observed.append("second-creator")
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=reader)
        thread.start()
        thread.join()
        if failures:
            raise failures[0]
        return proof

    monkeypatch.setattr(archive_population, "populate_authenticated_archive", populate)
    result = backup_operations.restore_verified_backup(backup_dir=package, destination=destination)
    assert observed == ["archive", "profile", "verified-leaf", "attached-query", "bootstrap", "second-creator"]
    assert result["operational_admission"] == "ready"
    with ArchiveStore.open_existing(destination, read_only=True):
        pass


def test_failed_restore_retains_pending_evidence_and_refuses_restart(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    from polylogue.storage.sqlite import archive_population
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    package_result = backup_archive(
        output_dir=tmp_path / "packages", verify=True, archive_root_path=workspace_env["archive_root"]
    )
    assert package_result.ok and package_result.output_path
    package = Path(package_result.output_path)
    original_receipt = (package / "verification-receipt.json").read_bytes()
    destination = tmp_path / "interrupted"
    actual = archive_population.populate_authenticated_archive

    def interrupt(source: Path, target: Path, **kwargs: Any) -> Any:
        actual(source, target, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(archive_population, "populate_authenticated_archive", interrupt)
    with pytest.raises(KeyboardInterrupt):
        backup_operations.restore_verified_backup(backup_dir=package, destination=destination)
    assert (destination / POPULATION_PENDING).is_file()
    assert (destination / "source.db").is_file()
    assert (package / "verification-receipt.json").read_bytes() == original_receipt
    with pytest.raises(ArchivePopulationPendingError):
        initialize_active_archive_root(destination)
    with pytest.raises(ArchivePopulationPendingError):
        with ArchiveStore.open_existing(destination, read_only=True):
            pass


def test_verified_baseline_backup_restores_under_destination_owned_authority(
    workspace_paths: dict[str, Path], tmp_path: Path
) -> None:
    import struct

    from polylogue.core.enums import Origin
    from polylogue.maintenance.offline_guard import scoped_offline_archive_writer
    from polylogue.storage.sqlite import migration_runner
    from polylogue.storage.sqlite.archive_tiers.bootstrap import _initialize_population_archive_stage
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.population_admission import (
        POPULATION_PENDING,
        _bound_population_stage,
        owned_population_admission,
    )
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
    from polylogue.storage.sqlite.write_lease import write_lease

    original = workspace_paths["archive_root"]
    original.mkdir(parents=True)
    pending = original / POPULATION_PENDING
    pending.write_text('{"fixture":"authenticated-current-format-baseline"}')
    with scoped_offline_archive_writer(original, owner_id="fixture.source1") as owner:
        with write_lease("fixture.source1", archive_root=original), owned_population_admission(original, owner):
            with _bound_population_stage(original, {"source": 1, "user": 1, "audit": 1}):
                _initialize_population_archive_stage(original)
            with closing(sqlite3.connect(original / "source.db")) as conn:
                assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
                payload = b'{"synthetic_record":"baseline-restore"}\n'
                BlobStore(original / "blob").write_from_bytes(payload)
                raw_id = write_source_raw_session(
                    conn,
                    origin=Origin.CODEX_SESSION,
                    capture_mode=Provider.CODEX,
                    source_path="/synthetic/baseline-restore",
                    canonical_source_path="/synthetic/baseline-restore",
                    source_index=0,
                    native_id=None,
                    payload=payload,
                    acquired_at_ms=2,
                )
                conn.commit()
                before = migration_runner._durable_literal_rows_digest(conn)
            for tier, statement in (
                (
                    "index",
                    "INSERT INTO sessions(native_id,origin,title,content_hash) VALUES ('old-derived','codex-session','old title',zeroblob(32))",
                ),
                (
                    "ops",
                    "INSERT INTO ingest_cursor(source_path,record_count,updated_at_ms) VALUES ('/synthetic/old',1,2)",
                ),
            ):
                with closing(sqlite3.connect(original / f"{tier}.db")) as conn:
                    conn.execute(statement)
                    assert (
                        conn.execute("UPDATE schema_identity SET identity=? WHERE tier=?", ("0" * 64, tier)).rowcount
                        == 1
                    )
                    assert (
                        conn.execute("SELECT identity FROM schema_identity WHERE tier=?", (tier,)).fetchone()[0]
                        == "0" * 64
                    )
                    conn.commit()
            vector = struct.pack("<1024f", *([0.25] * 1024))
            with closing(sqlite3.connect(original / "embeddings.db")) as conn:
                loaded, error = try_load_sqlite_vec(conn)
                assert loaded, error
                conn.execute(
                    "INSERT INTO message_embeddings(vector_derivation_hash,embedding,model) VALUES (?,?,?)",
                    ("synthetic-purchased-vector", vector, "synthetic-model"),
                )
                conn.commit()
            pending.unlink()
    original_marker = (original / ".maintenance-state/durable-change-trains/.bootstrap").read_bytes()
    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert result.ok and result.verified and result.output_path is not None
    package = Path(result.output_path)
    with closing(sqlite3.connect(package.joinpath("source.db").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert migration_runner._durable_literal_rows_digest(conn) == before
    destination = tmp_path / "restored-baseline"
    from tests.infra.workload_artifacts import _archive_files

    package_before = (_archive_files(package), (package / "manifest.json").read_bytes())
    detail = backup_operations.restore_verified_backup(backup_dir=package, destination=destination)
    assert (_archive_files(package), (package / "manifest.json").read_bytes()) == package_before
    assert not any(package.glob("*.db-wal"))
    assert not any(package.glob("*.db-shm"))
    assert detail["operational_admission"] == "degraded"
    assert detail["requires_convergence"] == ["index.db", "ops.db"]
    restored_tiers = detail["restored_tiers"]
    assert isinstance(restored_tiers, list)
    assert "index.db" not in restored_tiers
    assert "ops.db" not in restored_tiers
    assert "embeddings.db" in restored_tiers
    assert (package / ".maintenance-state/durable-change-trains/.bootstrap").read_bytes() == original_marker
    # The baseline is the current Source schema: the destination owns no train.
    assert not list((destination / ".maintenance-state/durable-change-trains").glob("source-*.json"))
    namespace = hashlib.sha256(str(detail["source_manifest_id"]).encode()).hexdigest()
    assert (
        destination / ".archive-population-provenance" / namespace / "original-history/.bootstrap"
    ).read_bytes() == original_marker
    for tier in ("index", "ops"):
        assert (
            destination / ".archive-population-provenance" / namespace / "original-derived" / f"{tier}.db"
        ).read_bytes() == (package / f"{tier}.db").read_bytes()
    with closing(sqlite3.connect(destination / "embeddings.db")) as conn:
        loaded, error = try_load_sqlite_vec(conn)
        assert loaded, error
        row = conn.execute(
            "SELECT embedding,model FROM message_embeddings WHERE vector_derivation_hash=?",
            ("synthetic-purchased-vector",),
        ).fetchone()
        assert tuple(row) == (vector, "synthetic-model")
    original_stat, destination_stat = (original / "source.db").stat(), (destination / "source.db").stat()
    assert (original_stat.st_dev, original_stat.st_ino) != (destination_stat.st_dev, destination_stat.st_ino)
    with closing(sqlite3.connect(destination / "source.db")) as conn:
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

        # The destination reaches this runtime's Source version through any
        # numbered steps it owns. The retained version-1 rows are compared
        # literally on the version-1 relations and columns, and every relation
        # a later step adds starts empty.
        restored_version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert restored_version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE], (restored_version, detail)
        with closing(
            sqlite3.connect(package.joinpath("source.db").as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as baseline:
            baseline_tables = [
                str(row[0])
                for row in baseline.execute(
                    "SELECT name FROM pragma_table_list WHERE schema='main' AND type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            for table in baseline_tables:
                columns = ",".join(
                    f'"{row[1]}"' for row in baseline.execute("SELECT * FROM pragma_table_info(?)", (table,))
                )
                retained = sorted(map(repr, baseline.execute(f'SELECT {columns} FROM "{table}"')))
                restored = sorted(map(repr, conn.execute(f'SELECT {columns} FROM "{table}"')))
                assert restored == retained, table
        added = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM pragma_table_list WHERE schema='main' AND type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            if str(row[0]) not in baseline_tables
        ]
        for name in added:
            assert conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone() == (0,), name
    # The ordinary startup owner validates the new physical receipt; no copied
    # baseline history is admitted as destination authority.
    initialize_active_archive_root(destination)
    with closing(sqlite3.connect(destination / "index.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    with closing(sqlite3.connect(destination / "ops.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0] == 0
    with ArchiveStore.open_existing(destination) as store:
        assert tuple(store.source_connection.execute("SELECT raw_id,native_id FROM raw_sessions").fetchone()) == (
            raw_id,
            None,
        )


def test_pending_destination_refuses_archive_admission_before_any_tier_exists(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import _initialize_population_archive_stage
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    root = tmp_path / "reserved"
    root.mkdir()
    (root / POPULATION_PENDING).write_text('{"fixture":"unfinished"}')
    for route in (
        lambda: ArchiveStore.open_existing(root),
        lambda: ArchiveStore(root, source_tier_acquisition=True),
        lambda: open_readonly_connection(root / "source.db"),
        lambda: initialize_active_archive_root(root),
        lambda: _initialize_population_archive_stage(root),
    ):
        with pytest.raises(ArchivePopulationPendingError):
            route()
    assert {path.name for path in root.iterdir()} == {POPULATION_PENDING}


@pytest.mark.parametrize("leaf_kind", ["symlink", "hardlink", "directory", "fifo", "ahead", "pre_reset"])
def test_verified_restore_refuses_unsafe_derived_leaf_and_retains_pending_custody(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leaf_kind: str
) -> None:
    import os
    from typing import Any

    from polylogue.storage.sqlite import archive_population
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    result = backup_archive(output_dir=tmp_path / "backups", profile="full_evidence", verify=True)
    assert result.ok and result.verified and result.output_path is not None
    package = Path(result.output_path)
    original_files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "unsafe-restoration"
    outsider = tmp_path / "unrelated-evidence"
    outsider.write_bytes(b"unrelated immutable evidence")
    populate = archive_population._populate_authenticated_archive

    def replace_derived_leaf(source: Path, target: Path, **kwargs: Any) -> object:
        leaf = target / "index.db"
        if leaf_kind in {"ahead", "pre_reset"}:
            with closing(sqlite3.connect(leaf)) as conn:
                conn.execute(f"PRAGMA user_version={2 if leaf_kind == 'ahead' else 0}")
            return populate(source, target, **kwargs)
        leaf.unlink()
        if leaf_kind == "symlink":
            leaf.symlink_to(outsider)
        elif leaf_kind == "hardlink":
            os.link(outsider, leaf)
        elif leaf_kind == "directory":
            leaf.mkdir()
        else:
            os.mkfifo(leaf)
        return populate(source, target, **kwargs)

    monkeypatch.setattr(archive_population, "_populate_authenticated_archive", replace_derived_leaf)
    with pytest.raises(archive_population.ArchivePopulationError) as exc:
        backup_operations.restore_verified_backup(backup_dir=package, destination=destination)
    assert exc.value.code == (
        "unsupported_derived_version" if leaf_kind in {"ahead", "pre_reset"} else "invalid_derived_leaf"
    )
    assert outsider.read_bytes() == b"unrelated immutable evidence"
    assert (destination / POPULATION_PENDING).is_file()
    with pytest.raises(ArchivePopulationPendingError):
        ArchiveStore.open_existing(destination)
    assert {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*")
        if path.is_file()
    } == original_files
