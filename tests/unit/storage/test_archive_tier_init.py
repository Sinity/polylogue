from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import archive_init
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitBlockedError,
    initialize_archive_tier_files,
)


def _fake_initialize_archive_database(path: Path, tier: object) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE initialized (tier TEXT PRIMARY KEY) STRICT")
        conn.execute("INSERT INTO initialized VALUES (?)", (str(tier),))
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()


def test_initialize_archive_tier_files_creates_all_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "stray.sqlite").write_text("unrelated file", encoding="utf-8")
    monkeypatch.setattr(archive_init, "initialize_archive_database", _fake_initialize_archive_database)

    result = initialize_archive_tier_files(archive_root=tmp_path)

    assert not (tmp_path / "stray.sqlite.retired.bak").exists()
    assert {tier.path.name for tier in result.tier_results} == {
        "source.db",
        "index.db",
        "embeddings.db",
        "user.db",
        "ops.db",
    }
    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        conn = sqlite3.connect(tmp_path / name)
        try:
            assert conn.execute("SELECT COUNT(*) FROM initialized").fetchone()[0] == 1
        finally:
            conn.close()


def test_initialize_archive_tier_files_backs_up_replaceable_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "source.db").write_text("existing source target", encoding="utf-8")
    (tmp_path / "index.db").write_text("existing index target", encoding="utf-8")
    (tmp_path / "embeddings.db").write_text("existing embeddings target", encoding="utf-8")
    (tmp_path / "user.db").write_text("existing user target", encoding="utf-8")
    (tmp_path / "ops.db").write_text("existing ops target", encoding="utf-8")
    monkeypatch.setattr(archive_init, "initialize_archive_database", _fake_initialize_archive_database)

    result = initialize_archive_tier_files(
        archive_root=tmp_path,
        replace_existing=True,
    )

    assert (tmp_path / "source.db.pre-archive-init.bak").read_text(encoding="utf-8") == "existing source target"
    assert (tmp_path / "embeddings.db.pre-archive-init.bak").read_text(encoding="utf-8") == "existing embeddings target"
    assert (tmp_path / "user.db.pre-archive-init.bak").read_text(encoding="utf-8") == "existing user target"
    assert not (tmp_path / "index.db.pre-archive-init.bak").exists()
    assert not (tmp_path / "ops.db.pre-archive-init.bak").exists()
    assert {tier.initialized for tier in result.tier_results} == {True}


def test_initialize_archive_tier_files_refuses_blocked_plan(tmp_path: Path) -> None:
    (tmp_path / "source.db").write_text("existing source target", encoding="utf-8")

    with pytest.raises(ArchiveInitBlockedError, match="source target already exists"):
        initialize_archive_tier_files(archive_root=tmp_path)
