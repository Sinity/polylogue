"""Archive location, readiness, and inactive-tuple admission at their production entry points."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.storage.archive_identity import ArchiveLocation, archive_root_for_index_path
from polylogue.storage.archive_readiness import (
    RawMaterializationAssessmentState,
    assess_raw_materialization,
)
from polylogue.storage.archive_tuple_location import (
    ArchiveTupleAllocator,
    ArchiveTupleError,
    ArchiveTupleLocation,
    ArchiveTuplePathError,
    ArchiveTupleStaleError,
    validate_inactive_destination,
)
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.tuple_generation import prepare_inactive_embedding_generation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _candidate(root: Path) -> tuple[ArchiveLocation, ArchiveTupleAllocator, ArchiveTupleLocation]:
    initialize_active_archive_root(root)
    location = ArchiveLocation.resolve(root)
    allocator = ArchiveTupleAllocator(location)
    return location, allocator, allocator.allocate(owner_id="codex-w5")


def test_generation_root_uses_the_final_index_shape(tmp_path: Path) -> None:
    """The first-marker implementation returns tmp_path, not root."""
    root = tmp_path / ".index-generations"
    physical = root / ".index-generations" / "gen-owned" / "index.db"
    assert archive_root_for_index_path(physical) == root
    assert archive_root_for_index_path(root / "index.db") == root


def test_durable_symlink_farm_allows_local_disposable_tiers(tmp_path: Path) -> None:
    """Local ops/embeddings used to make valid durable links fail admission."""
    backing = tmp_path / "backing"
    farm = tmp_path / "farm"
    backing.mkdir()
    farm.mkdir()
    for tier in ("source", "user", "audit", "index"):
        (backing / f"{tier}.db").touch()
        (farm / f"{tier}.db").symlink_to(backing / f"{tier}.db")
    for tier in ("ops", "embeddings"):
        (farm / f"{tier}.db").touch()
    (farm / ".index-active-pointer").write_text(str(farm / "index.db"), encoding="utf-8")
    location = ArchiveLocation.resolve(farm)
    assert location.active_index_path.resolve() == backing / "index.db"
    assert location.active_tier("ops").resolved_path == farm / "ops.db"


def test_zero_denominator_does_not_erase_measured_evidence_loss() -> None:
    """The previous early return reported UNMEASURED despite positive loss."""
    payload = {"available": True, "raw_artifact_count": 0, "lost_source_evidence_count": 1}
    assessment = assess_raw_materialization(payload)
    assert assessment.state is RawMaterializationAssessmentState.POPULATED_UNCONVERGED
    assert assessment.reason == "blocking_materialization_debt"
    assert dict(assessment.blocking_counts)["lost_source_evidence_count"] == 1
    assert assess_raw_materialization({**payload, "lost_source_evidence_count": 0}).reason == "zero_denominator"


@pytest.mark.parametrize("mutation", ["unknown", "nested_unknown", "missing_state", "numeric_string", "boolean"])
def test_tuple_loader_rejects_unsealed_manifest_shape_changes(tmp_path: Path, mutation: str) -> None:
    """Normalization previously erased these changes before checking the seal."""
    _, allocator, candidate = _candidate(tmp_path / "archive")
    payload = candidate.manifest.as_dict()
    if mutation == "unknown":
        payload["extra_authority"] = "not sealed"
    elif mutation == "nested_unknown":
        cast(dict[str, str], payload["generations"])["foreign"] = "not sealed"
    elif mutation == "missing_state":
        del payload["state"]
    elif mutation == "numeric_string":
        payload["manifest_version"] = str(payload["manifest_version"])
    else:
        payload["manifest_version"] = True
    candidate.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArchiveTupleError, match="invalid archive tuple manifest"):
        allocator.load(candidate.tuple_id)


def test_tuple_loader_rejects_a_sealed_obsolete_schema(tmp_path: Path) -> None:
    """Matching tier names and a valid seal cannot admit old DDL fingerprints."""
    _, allocator, candidate = _candidate(tmp_path / "archive")
    obsolete = replace(
        candidate.manifest,
        schema_fingerprints=tuple((tier, "0" * 64) for tier, _ in candidate.manifest.schema_fingerprints),
    )
    candidate.manifest_path.write_text(json.dumps(obsolete.as_dict()), encoding="utf-8")
    with pytest.raises(ArchiveTupleStaleError, match="schema fingerprints"):
        allocator.load(candidate.tuple_id)


@pytest.mark.parametrize("tier", [ArchiveTier.SOURCE, ArchiveTier.INDEX, ArchiveTier.EMBEDDINGS])
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_inactive_writer_refuses_foreign_final_file_links(tmp_path: Path, tier: ArchiveTier, link_kind: str) -> None:
    """The production bootstrap previously followed/admitted the final linked file."""
    _, _, candidate = _candidate(tmp_path / "archive")
    foreign_root = tmp_path / "foreign"
    initialize_active_archive_root(foreign_root)
    foreign = foreign_root / f"{tier.value}.db"
    initialize_archive_database(foreign, tier)
    before = foreign.read_bytes()
    destination = candidate.destination(tier)
    if link_kind == "symlink":
        destination.path.symlink_to(foreign)
    else:
        os.link(foreign, destination.path)
    with pytest.raises(ArchiveTuplePathError, match="unshared regular file"):
        initialize_archive_database(destination.path, tier, inactive_destination=destination)
    assert foreign.read_bytes() == before


def test_destination_generation_must_match_its_manifest_without_a_caller_hint(tmp_path: Path) -> None:
    """Omitting expected_generation previously admitted this forged label."""
    location, _, candidate = _candidate(tmp_path / "archive")
    destination = replace(candidate.index, generation_id="foreign-generation")
    with pytest.raises(ArchiveTupleStaleError, match="generation does not match its manifest"):
        validate_inactive_destination(destination, location)


def test_tuple_writer_rechecks_audit_identity_after_allocation(tmp_path: Path) -> None:
    """Audit is outside the authority digest, and location is a frozen snapshot."""
    root = tmp_path / "archive"
    location, _, candidate = _candidate(root)
    replacement = root / "replacement-audit.db"
    shutil.copyfile(root / "audit.db", replacement)
    os.replace(replacement, root / "audit.db")
    with pytest.raises(ArchiveTupleStaleError, match="audit identity"):
        validate_inactive_destination(candidate.index, location)


def test_allocation_fsyncs_both_new_directory_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fsyncing tuple.json's directory alone never persists its parent name."""
    import polylogue.storage.archive_tuple_location as tuples

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    synced: list[Path] = []
    original = tuples._fsync_directory

    def record(path: Path) -> None:
        original(path)
        synced.append(path)

    monkeypatch.setattr(tuples, "_fsync_directory", record)
    candidate = ArchiveTupleAllocator(ArchiveLocation.resolve(root)).allocate(owner_id="fsync-owner")
    assert root in synced
    assert root / ".archive-tuples" in synced
    assert candidate.candidate_root in synced
    assert synced.index(root) < synced.index(root / ".archive-tuples") < synced.index(candidate.candidate_root)


def test_reserved_name_alone_is_not_an_inactive_tuple(tmp_path: Path) -> None:
    """The former component-membership test refused this legitimate root."""
    root = tmp_path / ".archive-tuples"
    initialize_active_archive_root(root)
    location = ArchiveLocation.resolve(root)
    candidate = ArchiveTupleAllocator(location).allocate(owner_id="reserved-name")
    with pytest.raises(ArchiveTupleError):
        initialize_active_archive_root(candidate.candidate_root)


def test_embedding_preparation_rejects_a_foreign_final_symlink(tmp_path: Path) -> None:
    """Exercise embedding preparation, not merely the common destination helper."""
    _, _, candidate = _candidate(tmp_path / "archive")
    foreign = tmp_path / "foreign-embeddings.db"
    initialize_archive_database(foreign, ArchiveTier.EMBEDDINGS)
    before = foreign.read_bytes()
    candidate.embeddings.path.symlink_to(foreign)
    with pytest.raises(ArchiveTuplePathError, match="unshared regular file"):
        prepare_inactive_embedding_generation(
            candidate.embeddings,
            recipe=EmbeddingRecipe.current(model="voyage-3", dimensions=1024),
            source_generation=candidate.manifest.source_generation,
            index_generation=candidate.manifest.index_generation,
        )
    assert foreign.read_bytes() == before
