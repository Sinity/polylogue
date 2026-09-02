"""Tests for the authority snapshot shared by read surfaces."""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.surfaces.authority import build_authority_envelope


def _archive_with_active_generation(root: Path) -> Path:
    root.mkdir()
    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db", "audit.db"):
        (root / name).touch()
    active_index = root / ".index-generations" / "generation-1" / "index.db"
    active_index.parent.mkdir(parents=True)
    active_index.touch()
    (root / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
    return active_index


def test_authority_snapshot_uses_active_generation_and_marks_direct_reads_degraded(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    active_index = _archive_with_active_generation(root)

    authority = build_authority_envelope(
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
