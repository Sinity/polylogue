from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import devtools.index_v37_fast_forward as forward
from devtools.index_v37_fast_forward import IndexV37FastForwardError, activate_forward, prepare_forward
from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync


def _archive(tmp_path: Path) -> tuple[Path, int]:
    """Create a test archive with a v36 index database for devtools testing.

    The devtools index_v37_fast_forward module is specifically for v36→v37 migration.
    These tests verify the fast-forward executor works correctly on this legacy path.
    V36 had three cache tables that v37 drops (session_runs, session_observed_events,
    session_context_snapshots), so we create them here to simulate v36.

    Returns: (archive_root, created_index_version)
    """
    ffw_version = 36  # devtools index_v37_fast_forward tests v36→v37 migration

    root = tmp_path / "archive"
    root.mkdir()
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS):
        initialize_archive_database(root / f"{tier.value}.db", tier)
    storage = tmp_path / "storage"
    active_root = storage / ".index-generations" / f"v{ffw_version}"
    active_root.mkdir(parents=True)
    active = active_root / "index.db"
    with sqlite3.connect(active) as conn:
        # Use current schema but set version to v36 for devtools testing.
        # This tests the fast-forward executor without relying on historical DDL.
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        ensure_runtime_indexes_sync(conn)

        # Add v36 cache tables that v37 removes (CACHE_REMOVAL delta).
        # These are simple placeholder tables needed for devtools v36→v37 testing.
        conn.execute("CREATE TABLE IF NOT EXISTS session_runs(id TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS session_observed_events(id TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS session_context_snapshots(id TEXT)")

        conn.execute(f"PRAGMA user_version = {ffw_version}")
        # Add test data that will be preserved by fast-forward
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES ('session', 'chatgpt-export', ?)",
            (b"s" * 32,),
        )
        conn.commit()
    (storage / "index.db").symlink_to(active)
    (root / "index.db").symlink_to(storage / "index.db")
    return root, ffw_version


@pytest.fixture
def _patch_lifecycle_for_v36_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch lifecycle declarations to allow v36→v37 fast-forward only.

    The devtools v36→v37 forward tool is specifically designed to upgrade from v36
    to v37 using fast-forward (dropping cache tables). We patch the declarations
    to stop at v37 instead of extending to v45, so the tool can complete its
    specific migration without hitting SEMANTIC_REPARSE blocks from later versions.
    """
    import polylogue.storage.sqlite.lifecycle as lifecycle
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    # Keep only v36→v37 declarations.
    # The real lifecycle has v37 as CACHE_REMOVAL (fast-forwardable).
    # We exclude v38+ to avoid SEMANTIC_REPARSE blocks at v39+.
    synthetic_decls = tuple(d for d in lifecycle.INDEX_DELTA_DECLARATIONS if 36 <= d.version <= 37)

    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        synthetic_decls,
    )
    # Patch the expected version to v37 (devtools v36→v37 destination).
    monkeypatch.setitem(ARCHIVE_VERSION_BY_TIER, ArchiveTier.INDEX, 37)


def test_prepare_and_activate_preserve_surviving_rows_without_raw_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, ffw_version = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    prepared = prepare_forward(archive_root=root, receipt_path=receipt)

    assert prepared["status"] == "prepared"
    assert prepared["raw_reparse"] is False
    generation = prepared["generation"]
    assert isinstance(generation, dict)
    clone = Path(str(generation["index_path"]))
    assert (root / "index.db").resolve().parent.name == f"v{ffw_version}"
    with sqlite3.connect(clone) as conn:
        # With synthetic declarations, the target is v37 (devtools v36→v37).
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 37
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        for table in forward.RETIRED_TABLES:
            assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is None

    activated = activate_forward(receipt_path=receipt)

    assert activated["status"] == "activated"
    assert (root / "index.db").resolve() == clone.resolve()
    retired = list(IndexGenerationStore.for_archive_root(root).generations_root.glob("retired-*/index.db"))
    assert retired
    with sqlite3.connect(retired[0]) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == ffw_version


def test_ddl_normalization_preserves_token_and_literal_boundaries() -> None:
    assert forward._normalize_ddl("CREATE TABLE t(a TEXT)") == forward._normalize_ddl(
        'create table IF NOT EXISTS "t" ( "a" text )'
    )
    assert forward._normalize_ddl("CREATE TABLE t(a b)") != forward._normalize_ddl("CREATE TABLE t(ab)")
    assert forward._normalize_ddl("CREATE TABLE t(a DEFAULT 'a b')") != forward._normalize_ddl(
        "CREATE TABLE t(a DEFAULT 'ab')"
    )
    assert forward._normalize_ddl("CREATE TABLE t(a CHECK(a = 1-2))") != forward._normalize_ddl(
        "CREATE TABLE t(a CHECK(a = 1e-2))"
    )


def test_activate_keeps_completed_receipt_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, _ = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    prepare_forward(archive_root=root, receipt_path=receipt)
    activated = activate_forward(receipt_path=receipt)
    before = receipt.read_bytes()
    active = root / "index.db"
    with sqlite3.connect(active) as conn:
        conn.execute("UPDATE sessions SET title = 'later write'")
        conn.commit()

    repeated = activate_forward(receipt_path=receipt)

    assert repeated == activated
    assert receipt.read_bytes() == before


def test_prepare_refuses_unexpected_schema_surplus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, ffw_version = _archive(tmp_path)
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("CREATE TABLE unexpected_cache(value TEXT)")
        conn.commit()
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    with pytest.raises(IndexV37FastForwardError, match="unexpected_surplus"):
        prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    generations = IndexGenerationStore.for_archive_root(root).generations_root
    assert not [path for path in generations.iterdir() if path.name != f"v{ffw_version}"]


def test_prepare_repairs_preexisting_orphan_attachment_native_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, _ = _archive(tmp_path)
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(
            "INSERT INTO attachment_native_ids(ref_id, id_kind, native_id) VALUES ('missing', 'file', 'stale')"
        )
        conn.commit()
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    prepared = prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    postflight = prepared["postflight"]
    assert isinstance(postflight, dict)
    assert postflight["repaired_orphan_attachment_native_ids"] == 1
    generation = prepared["generation"]
    assert isinstance(generation, dict)
    with sqlite3.connect(str(generation["index_path"])) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_prepare_refuses_running_daemon_before_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, ffw_version = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: 1234)

    with pytest.raises(IndexV37FastForwardError, match="1234"):
        prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    generations = IndexGenerationStore.for_archive_root(root).generations_root
    assert list(generations.iterdir()) == [generations / f"v{ffw_version}"]


def test_prepare_refuses_unwritable_receipt_before_checkpoint_or_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, ffw_version = _archive(tmp_path)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block receipt parent creation", encoding="utf-8")
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    def unexpected_checkpoint(_path: Path, *, label: str = "active index") -> None:
        pytest.fail(f"receipt preflight ran after {label} checkpoint")

    monkeypatch.setattr(forward, "_checkpoint_stopped_database", unexpected_checkpoint)

    with pytest.raises(IndexV37FastForwardError, match="receipt destination is not writable"):
        prepare_forward(archive_root=root, receipt_path=blocker / "receipt.json")

    generations = IndexGenerationStore.for_archive_root(root).generations_root
    assert list(generations.iterdir()) == [generations / f"v{ffw_version}"]


def test_prepare_checkpoints_stopped_active_index_before_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, _ = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    active = IndexGenerationStore.for_archive_root(root).active_pointer.resolve()
    Path(f"{active}-shm").write_bytes(b"stopped-writer-residue")
    observed: list[tuple[Path, str]] = []
    original = forward._checkpoint_stopped_database

    def checkpoint(path: Path, *, label: str = "active index") -> None:
        observed.append((path, label))
        original(path, label=label)

    monkeypatch.setattr(forward, "_checkpoint_stopped_database", checkpoint)

    prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    assert observed[0] == (IndexGenerationStore.for_archive_root(root).active_pointer, "active index")
    assert observed[1][1] == "prepared clone"
    assert not Path(f"{active}-shm").exists()


def test_activate_refuses_changed_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, ffw_version = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    prepare_forward(archive_root=root, receipt_path=receipt)
    monkeypatch.setattr(forward, "source_revision_snapshot", lambda _root: "changed")

    with pytest.raises(IndexV37FastForwardError, match="source evidence changed"):
        activate_forward(receipt_path=receipt)

    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve().parent.name == f"v{ffw_version}"


def test_activate_refuses_in_place_clone_mutation_with_preserved_stat_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, ffw_version = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    prepared = prepare_forward(archive_root=root, receipt_path=receipt)
    generation = prepared["generation"]
    assert isinstance(generation, dict)
    clone = Path(str(generation["index_path"]))
    before = clone.stat()
    with sqlite3.connect(clone) as conn:
        conn.execute("UPDATE sessions SET content_hash = ?", (b"x" * 32,))
        conn.commit()
    forward._checkpoint_stopped_database(clone, label="mutated test clone")
    os.utime(clone, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(IndexV37FastForwardError, match="clone bytes changed"):
        activate_forward(receipt_path=receipt)

    assert clone.stat().st_size == before.st_size
    assert clone.stat().st_mtime_ns == before.st_mtime_ns
    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve().parent.name == f"v{ffw_version}"


def test_activate_recovers_after_pointer_swap_before_final_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_lifecycle_for_v36_upgrade: None
) -> None:
    root, _ = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    prepared = prepare_forward(archive_root=root, receipt_path=receipt)
    generation_payload = prepared["generation"]
    assert isinstance(generation_payload, dict)
    clone = Path(str(generation_payload["index_path"]))
    original_promote = IndexGenerationStore.promote

    def promote_then_crash(store: IndexGenerationStore, generation: IndexGeneration) -> IndexGeneration:
        original_promote(store, generation)
        raise RuntimeError("simulated crash after pointer swap")

    monkeypatch.setattr(IndexGenerationStore, "promote", promote_then_crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        activate_forward(receipt_path=receipt)
    assert forward._load_receipt(receipt)["status"] == "activating"
    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve() == clone.resolve()

    monkeypatch.setattr(IndexGenerationStore, "promote", original_promote)
    activated = activate_forward(receipt_path=receipt)

    assert activated["status"] == "activated"
    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve() == clone.resolve()
