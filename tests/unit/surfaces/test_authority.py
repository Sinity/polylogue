"""Tests for the authority snapshot shared by read surfaces."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from time import monotonic

import pytest

from polylogue.config import Config
from polylogue.operations.authority import authority_for_config, authority_for_reader, authority_for_root
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER


def _archive_tiers(root: Path) -> Path:
    root.mkdir()
    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db", "audit.db"):
        (root / name).write_bytes(name.encode())
    return root


def _archive_with_active_generation(root: Path) -> Path:
    _archive_tiers(root)
    active_index = root / ".index-generations" / "generation-1" / "index.db"
    active_index.parent.mkdir(parents=True)
    active_index.touch()
    (root / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
    return active_index


def test_authority_snapshot_uses_active_generation_and_marks_direct_reads_degraded(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    active_index = _archive_with_active_generation(root)

    authority = authority_for_root(
        root,
        server_identity="direct",
        run_id="run-1",
        degraded=("reader_lag", "reader_lag"),
    )
    expected = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(root))

    assert authority.run_id == "run-1"
    assert authority.archive_epoch == expected.authority_identity_digest
    assert active_index.exists()
    assert authority.generation_id == expected.active_generation
    assert authority.tier_schema_versions == {
        tier.value: int(version) for tier, version in ARCHIVE_VERSION_BY_TIER.items()
    }
    assert authority.server_identity == "direct"
    assert authority.degraded == ("reader_lag", "daemon_unavailable")


def test_authority_elapsed_ms_measures_the_operation_boundary(tmp_path: Path) -> None:
    """Anti-vacuity: drop ``started_at`` from the call and elapsed_ms pins to 0.

    Every production call site previously omitted ``started_at``, so the field
    was structurally always zero. This asserts a non-zero elapsed interval is
    actually carried through.
    """

    root = tmp_path / "archive"
    _archive_with_active_generation(root)

    started_at = monotonic() - 1.5
    authority = authority_for_root(root, server_identity="direct", started_at=started_at)

    assert authority.elapsed_ms >= 1000
    assert authority_for_root(root, server_identity="direct").elapsed_ms == 0


def test_authority_for_reader_binds_the_generation_the_reader_opened(tmp_path: Path) -> None:
    """Anti-vacuity: resolve from the root instead of the pinned index and the
    assertion below flips to the newly published generation.

    A reader holds one index open for the life of a read. Publishing a new
    generation mid-read must not re-attribute the rows already served.
    """

    root = tmp_path / "archive"
    opened_index = _archive_with_active_generation(root)

    class _Reader:
        archive_root = root
        index_db_path = opened_index

    superseded = root / ".index-generations" / "generation-2" / "index.db"
    superseded.parent.mkdir(parents=True)
    superseded.write_bytes(b"newer")
    (root / ".index-active-pointer").write_text(str(superseded), encoding="utf-8")

    pinned = authority_for_reader(_Reader(), server_identity="direct")
    active = authority_for_root(root, server_identity="direct")

    assert pinned.generation_id != active.generation_id
    assert pinned.generation_id == ArchiveIdentity.resolve_pinned_index(root, opened_index).active_generation


def test_authority_for_config_follows_an_explicit_db_path(tmp_path: Path) -> None:
    """Anti-vacuity: read ``config.archive_root`` instead of the active archive
    root and this returns the decoy archive's generation.

    ``archive_root`` and ``db_path`` are independently settable; the read
    follows ``db_path``, so attribution must too.
    """

    # No pointer at either root: `archive_file_set_root` then follows db_path,
    # which is exactly the split-root override this attributes.
    decoy = _archive_tiers(tmp_path / "decoy")
    real = _archive_tiers(tmp_path / "real")

    config = Config(
        archive_root=decoy,
        render_root=tmp_path / "render",
        sources=[],
        db_path=real / "index.db",
    )
    authority = authority_for_config(config, server_identity="direct")

    assert authority.generation_id == authority_for_root(real, server_identity="direct").generation_id
    assert authority.generation_id != authority_for_root(decoy, server_identity="direct").generation_id


def test_surface_authority_module_resolves_no_archive_state() -> None:
    """Anti-vacuity: re-import archive_identity into surfaces/authority.py and
    this fails.

    The surface layer serializes attribution; the operation boundary resolves
    it. A surface that reached into storage would rebuild archive facts after
    the read that produced them.
    """

    module = importlib.import_module("polylogue.surfaces.authority")
    source = inspect.getsource(module)

    assert "polylogue.storage" not in source
    signature = inspect.signature(module.build_authority_envelope)
    assert list(signature.parameters) == [
        "archive_epoch",
        "generation_id",
        "tier_schema_versions",
        "server_identity",
        "started_at",
        "run_id",
        "degraded",
    ]
    with pytest.raises(TypeError):
        module.build_authority_envelope(Path("/nonexistent"), server_identity="direct")
