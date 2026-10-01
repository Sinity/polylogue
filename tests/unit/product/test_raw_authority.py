from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Config, Source, resolve_runtime_config
from polylogue.maintenance import raw_authority
from polylogue.storage.index_generation import RebuildLease, RebuildLeaseUnavailableError


def test_materialization_generation_lease_pins_active_index_and_excludes_promotion(tmp_path: Path) -> None:
    active_index = tmp_path / "generations" / "active" / "index.db"
    active_index.parent.mkdir(parents=True)
    active_index.touch()
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
    config = Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[])

    with raw_authority.materialization_generation_lease(config) as index_db:
        assert index_db == active_index
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass


def test_materialization_generation_lease_uses_explicit_split_root(tmp_path: Path) -> None:
    configured_root = tmp_path / "configured"
    active_root = tmp_path / "active"
    configured_root.mkdir()
    active_root.mkdir()
    active_index = active_root / "index.db"
    active_index.touch()
    config = Config(
        archive_root=configured_root,
        render_root=tmp_path / "render",
        sources=[],
        db_path=active_index,
    )

    with raw_authority.materialization_generation_lease(config) as index_db:
        assert index_db == active_index
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(active_root):
                pass
        with RebuildLease(configured_root):
            pass


@pytest.mark.parametrize("explicit", [False, True])
def test_runtime_projection_generation_lease_reads_promoted_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """Passing a resolved default as db_path makes the implicit case read A."""
    root = tmp_path / "selected"
    root.mkdir()
    first = root / "a" / "index.db"
    second = root / "b" / "index.db"
    for database, content in ((first, "A"), (second, "B")):
        database.parent.mkdir()
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE witness(value TEXT)")
            connection.execute("INSERT INTO witness VALUES (?)", (content,))
    pointer = root / ".index-active-pointer"
    pointer.write_text(str(first), encoding="utf-8")
    runtime = resolve_runtime_config(
        environment={
            "HOME": str(tmp_path),
            "POLYLOGUE_ARCHIVE_ROOT": str(root),
            "POLYLOGUE_SITE_CONFIG": "",
        },
    )
    config = (
        Config(archive_root=root, render_root=runtime.paths.render_root, sources=[], db_path=first)
        if explicit
        else runtime.as_config()
    )
    clone = config.with_sources([Source(name="fixture", path=tmp_path / "source")])
    assert config.db_path == first
    with raw_authority.materialization_generation_lease(config) as database:
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT value FROM witness").fetchone() == ("A",)

    pointer.write_text(str(second), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "ambient"))
    for projected in (config, clone):
        with raw_authority.materialization_generation_lease(projected) as database:
            assert database == (first if explicit else second)
            with sqlite3.connect(database) as connection:
                assert connection.execute("SELECT value FROM witness").fetchone() == ("A" if explicit else "B",)
            with pytest.raises(RebuildLeaseUnavailableError):
                with RebuildLease(root):
                    pass
