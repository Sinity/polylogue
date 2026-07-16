from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import devtools.index_v37_fast_forward as forward
from devtools.index_v37_fast_forward import IndexV37FastForwardError, activate_forward, prepare_forward
from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers import index as index_tier
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync


def _v36_ddl() -> str:
    commit = "5d99611f4^"
    import subprocess

    source = subprocess.run(
        ["git", "show", f"{commit}:polylogue/storage/sqlite/archive_tiers/index.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    namespace: dict[str, object] = {}
    exec(compile(source, "index-v36.py", "exec"), namespace)
    return str(namespace["INDEX_DDL"])


def _archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS):
        initialize_archive_database(root / f"{tier.value}.db", tier)
    storage = tmp_path / "storage"
    active_root = storage / ".index-generations" / "v36"
    active_root.mkdir(parents=True)
    active = active_root / "index.db"
    with sqlite3.connect(active) as conn:
        conn.executescript(_v36_ddl())
        ensure_runtime_indexes_sync(conn)
        conn.execute("PRAGMA user_version = 36")
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES ('session', 'chatgpt-export', ?)",
            (b"s" * 32,),
        )
        conn.execute(
            "INSERT INTO session_runs(run_ref, session_id, position, harness, role, status, confidence) "
            "VALUES ('run', 'chatgpt-export:session', 0, 'chatgpt', 'main', 'completed', 'raw')"
        )
        conn.commit()
    (storage / "index.db").symlink_to(active)
    (root / "index.db").symlink_to(storage / "index.db")
    return root


def test_prepare_and_activate_preserve_surviving_rows_without_raw_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    prepared = prepare_forward(archive_root=root, receipt_path=receipt)

    assert prepared["status"] == "prepared"
    assert prepared["raw_reparse"] is False
    generation = prepared["generation"]
    assert isinstance(generation, dict)
    clone = Path(str(generation["index_path"]))
    assert (root / "index.db").resolve().parent.name == "v36"
    with sqlite3.connect(clone) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == index_tier.INDEX_SCHEMA_VERSION
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        for table in forward.RETIRED_TABLES:
            assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is None

    activated = activate_forward(receipt_path=receipt)

    assert activated["status"] == "activated"
    assert (root / "index.db").resolve() == clone.resolve()
    retired = list(IndexGenerationStore(root).generations_root.glob("retired-*/index.db"))
    assert retired
    with sqlite3.connect(retired[0]) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 36


def test_prepare_refuses_unexpected_v36_schema_surplus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _archive(tmp_path)
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("CREATE TABLE unexpected_cache(value TEXT)")
        conn.commit()
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    with pytest.raises(IndexV37FastForwardError, match="unexpected_surplus"):
        prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    generations = IndexGenerationStore(root).generations_root
    assert not [path for path in generations.iterdir() if path.name != "v36"]


def test_prepare_repairs_preexisting_orphan_attachment_native_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _archive(tmp_path)
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


def test_prepare_refuses_running_daemon_before_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: 1234)

    with pytest.raises(IndexV37FastForwardError, match="1234"):
        prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    generations = IndexGenerationStore(root).generations_root
    assert list(generations.iterdir()) == [generations / "v36"]


def test_prepare_checkpoints_stopped_active_index_before_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    active = IndexGenerationStore(root).active_pointer.resolve()
    Path(f"{active}-shm").write_bytes(b"stopped-writer-residue")
    observed: list[tuple[Path, str]] = []
    original = forward._checkpoint_stopped_database

    def checkpoint(path: Path, *, label: str = "active index") -> None:
        observed.append((path, label))
        original(path, label=label)

    monkeypatch.setattr(forward, "_checkpoint_stopped_database", checkpoint)

    prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    assert observed[0] == (IndexGenerationStore(root).active_pointer, "active index")
    assert observed[1][1] == "prepared clone"
    assert not Path(f"{active}-shm").exists()


def test_activate_refuses_changed_source_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    prepare_forward(archive_root=root, receipt_path=receipt)
    monkeypatch.setattr(forward, "source_revision_snapshot", lambda _root: "changed")

    with pytest.raises(IndexV37FastForwardError, match="source evidence changed"):
        activate_forward(receipt_path=receipt)

    assert IndexGenerationStore(root).active_pointer.resolve().parent.name == "v36"


def test_activate_refuses_in_place_clone_mutation_with_preserved_stat_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _archive(tmp_path)
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
    assert IndexGenerationStore(root).active_pointer.resolve().parent.name == "v36"


def test_activate_recovers_after_pointer_swap_before_final_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _archive(tmp_path)
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
    assert IndexGenerationStore(root).active_pointer.resolve() == clone.resolve()

    monkeypatch.setattr(IndexGenerationStore, "promote", original_promote)
    activated = activate_forward(receipt_path=receipt)

    assert activated["status"] == "activated"
    assert IndexGenerationStore(root).active_pointer.resolve() == clone.resolve()
