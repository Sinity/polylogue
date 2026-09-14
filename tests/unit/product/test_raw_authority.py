from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config
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
