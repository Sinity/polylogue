from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import devtools.index_v37_fast_forward as forward
from devtools.index_v37_fast_forward import IndexV37FastForwardError, activate_forward, prepare_forward
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers import index as index_tier
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


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


def test_prepare_refuses_running_daemon_before_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: 1234)

    with pytest.raises(IndexV37FastForwardError, match="1234"):
        prepare_forward(archive_root=root, receipt_path=tmp_path / "receipt.json")

    generations = IndexGenerationStore(root).generations_root
    assert list(generations.iterdir()) == [generations / "v36"]


def test_activate_refuses_changed_source_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _archive(tmp_path)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    prepare_forward(archive_root=root, receipt_path=receipt)
    monkeypatch.setattr(forward, "source_revision_snapshot", lambda _root: "changed")

    with pytest.raises(IndexV37FastForwardError, match="source evidence changed"):
        activate_forward(receipt_path=receipt)

    assert IndexGenerationStore(root).active_pointer.resolve().parent.name == "v36"
