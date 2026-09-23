"""polylogue-623q: the bulk-build pragma profile is scoped to owned inactive
index generations only -- it must never leak onto the live active-archive
writer connection.

Production dependencies exercised here: ``ArchiveStore.__init__`` (the real
constructor, not a reimplementation), ``ArchiveStore.open_owned_inactive_generation``,
``ArchiveStore.open_existing``, and ``IndexGenerationStore.create`` (the real
generation bootstrap the offline rebuild uses). Reverting the
``bulk_build_profile`` wiring in ``ArchiveStore._initialize_store`` (or
reverting ``BULK_BUILD_WRITE_CONNECTION_PROFILE`` back to the live profile)
makes ``test_owned_inactive_generation_uses_bulk_build_pragma_profile`` fail,
not merely produce a wrong number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore, _assert_active_cold_build_index_only
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import (
    BULK_BUILD_CACHE_SIZE_KIB,
    WRITE_CACHE_SIZE_KIB,
    WRITE_CONNECTION_PROFILE,
    write_connection_pragma_statements,
)


def test_owned_inactive_generation_uses_bulk_build_pragma_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    store = IndexGenerationStore.for_archive_root(root)
    generation = store.create(source_snapshot="snapshot-a")
    generation_root = Path(generation.index_path).parent

    with ArchiveStore.open_owned_inactive_generation(
        generation_root, generation_id=generation.generation_id, owner_id=generation.owner_id
    ) as archive:
        journal_mode = archive._conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = archive._conn.execute("PRAGMA synchronous").fetchone()[0]
        cache_size = archive._conn.execute("PRAGMA cache_size").fetchone()[0]

    assert journal_mode.lower() == "memory"
    # PRAGMA synchronous readback is an integer: 0=OFF, 1=NORMAL, 2=FULL.
    assert synchronous == 0
    assert abs(cache_size) == BULK_BUILD_CACHE_SIZE_KIB


def test_live_active_archive_writer_keeps_wal_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact same constructor, without ``owned_inactive_generation``, must
    keep the live writer's crash-safe WAL/NORMAL profile unchanged."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        journal_mode = archive._conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = archive._conn.execute("PRAGMA synchronous").fetchone()[0]
        cache_size = archive._conn.execute("PRAGMA cache_size").fetchone()[0]

    assert journal_mode.lower() == "wal"
    # The live writer applies the write profile as built at import: NORMAL,
    # or OFF when the harness's scratch override dropped fsync for the run.
    assert synchronous == _declared_synchronous(write_connection_pragma_statements(WRITE_CONNECTION_PROFILE))
    assert abs(cache_size) == WRITE_CACHE_SIZE_KIB


def test_active_cold_build_profile_refuses_a_durable_tier_path(tmp_path: Path) -> None:
    """The unsafe profile is structurally limited to the rebuildable index.

    Anti-vacuity: removing the path guard (or changing it to inspect only a
    caller flag) lets this durable source path through and makes the assertion
    green, so this test exercises the actual safety boundary.
    """
    source = tmp_path / "source.db"
    source.touch()
    with pytest.raises(RuntimeError, match="restricted to the rebuildable index"):
        _assert_active_cold_build_index_only(source, durable_paths=(source,))


def test_active_cold_build_routes_guard_through_index_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production cold-build branch invokes the durable-tier guard."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    calls: list[tuple[Path, tuple[Path, ...]]] = []

    import polylogue.storage.sqlite.archive_tiers.archive as archive_module

    real_guard = archive_module._assert_active_cold_build_index_only

    def record(index_path: Path, *, durable_paths: tuple[Path, ...]) -> None:
        calls.append((index_path, durable_paths))
        real_guard(index_path, durable_paths=durable_paths)

    monkeypatch.setattr(archive_module, "_assert_active_cold_build_index_only", record)
    with ArchiveStore.open_active_cold_build(root):
        pass

    assert calls, "active cold-build profile bypassed its index-only guard"
    index_path, durable_paths = calls[0]
    assert index_path.resolve() not in {path.resolve() for path in durable_paths}


def _declared_synchronous(statements: tuple[str, ...]) -> int:
    levels = {"OFF": 0, "NORMAL": 1, "FULL": 2}
    declared = [stmt.rsplit("=", 1)[1].strip().upper() for stmt in statements if stmt.startswith("PRAGMA synchronous")]
    assert len(declared) == 1, statements
    return levels[declared[0]]
