"""Contract tests for whole-archive inactive tuple allocation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.archive_tuple_location import (
    ArchiveTupleAllocator,
    ArchiveTupleCollisionError,
    ArchiveTupleError,
    ArchiveTupleForeignError,
    ArchiveTuplePathError,
    ArchiveTupleStaleError,
    validate_inactive_destination,
)
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _archive(root: Path) -> None:
    initialize_active_archive_root(root)


def test_allocation_is_collision_safe_complete_and_does_not_copy_stable_authority(tmp_path: Path) -> None:
    _archive(tmp_path)
    location = ArchiveLocation.resolve(tmp_path)
    allocator = ArchiveTupleAllocator(location)
    ids = iter(("tuple-1-aaaaaaaaaaaaaaaa", "tuple-1-aaaaaaaaaaaaaaaa", "tuple-1-bbbbbbbbbbbbbbbb"))

    candidate = allocator.allocate(owner_id="test-owner", allocation_id_factory=lambda: next(ids))

    assert candidate.candidate_root.is_dir()
    assert candidate.manifest_path.is_file()
    assert {destination.tier for destination in candidate.destinations} == {
        ArchiveTier.SOURCE,
        ArchiveTier.INDEX,
        ArchiveTier.EMBEDDINGS,
    }
    assert all(not destination.path.exists() for destination in candidate.destinations)
    assert not any(path.is_symlink() for path in candidate.candidate_root.iterdir())
    assert not (candidate.candidate_root / "user.db").exists()
    assert not (candidate.candidate_root / "audit.db").exists()
    assert ArchiveLocation.resolve(tmp_path).active_index_path == location.active_index_path


def test_allocation_exhausts_repeated_collisions_without_opening_sqlite(tmp_path: Path) -> None:
    _archive(tmp_path)
    allocator = ArchiveTupleAllocator(ArchiveLocation.resolve(tmp_path))
    allocator.allocate(owner_id="first", allocation_id_factory=lambda: "tuple-1-aaaaaaaaaaaaaaaa")

    with pytest.raises(ArchiveTupleCollisionError):
        allocator.allocate(
            owner_id="second",
            allocation_id_factory=lambda: "tuple-1-aaaaaaaaaaaaaaaa",
            max_attempts=2,
        )


def test_manifest_round_trip_binds_generations_schemas_stable_ids_and_seals(tmp_path: Path) -> None:
    _archive(tmp_path)
    allocator = ArchiveTupleAllocator(ArchiveLocation.resolve(tmp_path))
    candidate = allocator.allocate(
        owner_id="manifest-owner",
        source_generation="source-v1",
        index_generation="index-v1",
        embeddings_generation="embeddings-v1",
    )
    loaded = allocator.load(candidate.tuple_id)

    assert loaded.manifest == candidate.manifest
    assert loaded.manifest.manifest_version == 1
    assert loaded.manifest.source_generation == "source-v1"
    assert loaded.manifest.index_generation == "index-v1"
    assert loaded.manifest.embeddings_generation == "embeddings-v1"
    assert loaded.manifest.user_identity
    assert loaded.manifest.audit_identity
    assert loaded.manifest.schema_fingerprints
    assert loaded.manifest.semantic_fingerprints
    assert loaded.manifest.expected_seals
    assert loaded.manifest.manifest_digest == loaded.manifest.seal


def test_wrong_active_foreign_and_escaped_destinations_fail_before_schema_open(tmp_path: Path) -> None:
    _archive(tmp_path)
    location = ArchiveLocation.resolve(tmp_path)
    candidate = ArchiveTupleAllocator(location).allocate(owner_id="owner")
    active = candidate.index

    escaped = replace(active, path=tmp_path / "outside.db")
    with pytest.raises(ArchiveTuplePathError):
        validate_inactive_destination(escaped, location, path=escaped.path, expected_tier=ArchiveTier.INDEX)

    wrong_active = replace(active, path=location.active_index_path)
    with pytest.raises(ArchiveTuplePathError):
        validate_inactive_destination(wrong_active, location, path=wrong_active.path, expected_tier=ArchiveTier.INDEX)

    foreign_root = tmp_path / "foreign"
    _archive(foreign_root)
    with pytest.raises(ArchiveTupleForeignError):
        validate_inactive_destination(active, ArchiveLocation.resolve(foreign_root))


def test_wrong_generation_and_stale_active_identity_are_rejected(tmp_path: Path) -> None:
    _archive(tmp_path)
    location = ArchiveLocation.resolve(tmp_path)
    candidate = ArchiveTupleAllocator(location).allocate(owner_id="owner", index_generation="index-v1")

    wrong_generation = replace(candidate.index, generation_id="index-v2")
    with pytest.raises(ArchiveTupleStaleError):
        validate_inactive_destination(wrong_generation, location, expected_generation="index-v1")

    # A promotion changes the active ArchiveIdentity digest.  The candidate is
    # still physically present, but it is no longer a safe base for a writer.
    generation_store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = generation_store.create(owner_id="promoter", source_snapshot="snapshot")
    generation_store.promote(generation)
    with pytest.raises(ArchiveTupleStaleError):
        ArchiveTupleAllocator(ArchiveLocation.resolve(tmp_path)).load(candidate.tuple_id)


def test_inactive_tier_writer_requires_typed_destination_before_sqlite_opens(tmp_path: Path) -> None:
    _archive(tmp_path)
    candidate = ArchiveTupleAllocator(ArchiveLocation.resolve(tmp_path)).allocate(owner_id="writer")

    with pytest.raises(ArchiveTupleError, match="typed inactive_destination"):
        initialize_archive_database(candidate.index.path, ArchiveTier.INDEX)

    initialize_archive_database(candidate.index.path, ArchiveTier.INDEX, inactive_destination=candidate.index)
    assert candidate.index.path.is_file()


def test_candidate_path_cannot_be_reinterpreted_as_an_active_archive_root(tmp_path: Path) -> None:
    _archive(tmp_path)
    candidate = ArchiveTupleAllocator(ArchiveLocation.resolve(tmp_path)).allocate(owner_id="writer")

    with pytest.raises(ArchiveTupleError):
        initialize_active_archive_root(candidate.candidate_root)
