"""Backup verification tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon.backup import backup_archive
from polylogue.storage.blob_store import BlobStore
from tests.infra.storage_records import SessionBuilder, db_setup


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
    # tiers (source/user/embeddings) and omits the rebuildable index/ops
    # tiers. Each copied tier must open cleanly and pass integrity_check.
    assert result.ok
    assert result.backup_mode == "archive_file_set"
    assert result.output_path is not None
    backup_path = Path(result.output_path)
    assert backup_path.is_dir()
    for tier in ("source.db", "user.db", "embeddings.db"):
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
    blob_hash, _ = BlobStore(workspace_env["data_root"] / "polylogue" / "blob").write_from_bytes(payload)
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
    assert result.verified is True
    assert result.verification["ok"] is True
    assert result.verification["tier_integrity"] == {
        "source": True,
        "user": True,
        "embeddings": True,
    }
    assert result.verification["omitted_tiers_absent"] is True
    assert result.verification["restored_blob_count"] == 1
    assert result.output_path is not None
    backup_root = Path(result.output_path)
    assert backup_root.is_dir()
    assert result.omitted_tiers == ["index.db", "ops.db"]
    assert (backup_root / "source.db").exists()
    assert (backup_root / "user.db").exists()
    assert (backup_root / "embeddings.db").exists()
    assert not (backup_root / "index.db").exists()
    assert not (backup_root / "ops.db").exists()
    assert (backup_root / "blob" / blob_hash[:2] / blob_hash[2:]).read_bytes() == payload

    with sqlite3.connect(backup_root / "source.db") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
    with sqlite3.connect(backup_root / "user.db") as conn:
        assert conn.execute("SELECT mark_id FROM backup_test_marks").fetchone()[0] == "mark-one"
    with sqlite3.connect(backup_root / "embeddings.db") as conn:
        assert conn.execute("SELECT session_id FROM backup_test_embedding_status").fetchone()[0] == "codex-session:one"

    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "archive_file_set"
    assert manifest["omitted_tiers"] == ["index.db", "ops.db"]
    assert manifest["blob_count"] == 1


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
