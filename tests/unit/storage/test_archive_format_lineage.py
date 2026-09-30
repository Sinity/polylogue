from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_FORMAT_FLOOR_VERSION, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive_plan import ARCHIVE_FORMAT_LINEAGE
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_marker_refuses_a_foreign_lineage_tier(tmp_path: Path) -> None:
    """A stamped version needs format evidence, not a coincidentally equal integer.

    The transplant carries the *birth* version this archive's marker recorded,
    because that is the only integer a foreign file can borrow to look current.
    A tier standing at any other version is already refused by the ordinary
    version comparison, so pinning this to the literal ``1`` would stop
    exercising the marker the moment a numbered migration raised the durable
    target above the floor.

    Anti-vacuity: binding the fingerprint check to ``ARCHIVE_FORMAT_FLOOR_VERSION``
    instead of the recorded birth version admits this file, and ``user.db``
    is then rewritten by a bootstrap that should never have started.
    """
    initialize_active_archive_root(tmp_path)

    marker = json.loads((tmp_path / ".polylogue-format.json").read_text(encoding="utf-8"))
    assert marker["format"] == ARCHIVE_FORMAT_LINEAGE
    assert marker["floor_version"] == ARCHIVE_FORMAT_FLOOR_VERSION
    assert marker["tier_versions"] == {tier.value: ARCHIVE_VERSION_BY_TIER[tier] for tier in ArchiveTier}

    birth_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
    user_path = tmp_path / "user.db"
    user_before = user_path.read_bytes()
    source_path = tmp_path / "source.db"
    source_path.unlink()
    with sqlite3.connect(source_path) as historical:
        historical.execute("CREATE TABLE historical_lineage (id INTEGER PRIMARY KEY) STRICT")
        historical.execute(f"PRAGMA user_version = {birth_version}")

    with pytest.raises(RuntimeError, match=f"historical version-{birth_version} schema"):
        initialize_active_archive_root(tmp_path)

    assert user_path.read_bytes() == user_before


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "directory"])
def test_lineage_refuses_unsafe_tier_before_sqlite_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_kind: str
) -> None:
    """Moving safe-file validation after open reaches an unrelated SQLite file."""
    import os

    from polylogue.storage.sqlite.archive_tiers import archive_plan

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    tier = root / "user.db"
    unrelated = tmp_path / "unrelated.db"
    with closing(sqlite3.connect(unrelated)) as connection, connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    before = unrelated.read_bytes()
    tier.unlink()
    if unsafe_kind == "symlink":
        tier.symlink_to(unrelated)
    elif unsafe_kind == "hardlink":
        os.link(unrelated, tier)
    else:
        tier.mkdir()

    def forbidden_open(*args: object, **kwargs: object) -> None:
        pytest.fail("unsafe tier reached SQLite open")

    monkeypatch.setattr(archive_plan, "open_readonly_connection", forbidden_open)
    with pytest.raises(RuntimeError, match="unsafe durable tier file"):
        archive_plan.assert_archive_format_lineage(root, tiers=frozenset({ArchiveTier.USER}))
    assert unrelated.read_bytes() == before
    assert not Path(str(unrelated) + "-wal").exists()
    assert not Path(str(unrelated) + "-shm").exists()
