"""Whole-tree capacity accounting for a candidate index build.

Anti-vacuity: every refusal test here fails if a population is dropped from
:data:`POPULATION_NAMES` or measured by apparent size instead of allocated
blocks — both mutations turn a boundary refusal into acceptance.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.maintenance.candidate_capacity import (
    ALLOCATION_UNIT_BYTES,
    CAPACITY_RECEIPT_SCHEMA,
    DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE,
    POPULATION_NAMES,
    RECEIPT_GROWTH_FLOOR_BYTES,
    RESERVE_FLOOR_BYTES,
    ArchiveCapacityError,
    CandidateCapacityProjection,
    InsufficientCapacityError,
    calibrated_index_ratio,
    measure_archive_capacity,
    project_candidate_capacity,
    read_capacity_receipts,
    record_capacity_observation,
    record_capacity_prediction,
    require_candidate_capacity,
)
from polylogue.storage.archive_identity import (
    ACTIVE_POINTER_FILENAME,
    GENERATIONS_DIRNAME,
    MAINTENANCE_STATE_DIRNAME,
)

_POINTER_STUB_BYTES = 77


def _dense(path: Path, size: int) -> None:
    """Write ``size`` bytes that the filesystem must actually allocate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"\xa5" * size)
        stream.flush()
        os.fsync(stream.fileno())


def _sparse(path: Path, size: int) -> None:
    """Create a file whose apparent size is ``size`` but which allocates nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)


def _archive_with_generation(root: Path, *, generation_bytes: int, sparse: bool = False) -> Path:
    """Build the canonical layout: a pointer stub at the root, the index hidden."""
    root.mkdir(parents=True, exist_ok=True)
    generation = root / GENERATIONS_DIRNAME / "gen-1-aaaaaaaa"
    generation.mkdir(parents=True)
    index = generation / "index.db"
    if sparse:
        _sparse(index, generation_bytes)
    else:
        _dense(index, generation_bytes)
    # The archive root's own index.db is the stub the pointer supersedes.
    _dense(root / "index.db", _POINTER_STUB_BYTES)
    (root / ACTIVE_POINTER_FILENAME).write_text(str(index.absolute()), encoding="utf-8")
    return generation


def _allocated(path: Path) -> int:
    return path.stat().st_blocks * ALLOCATION_UNIT_BYTES


def test_pointer_stub_cannot_hide_a_generation_from_capacity_accounting(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    generation = _archive_with_generation(root, generation_bytes=4 * 1024 * 1024)

    inventory = measure_archive_capacity(root)

    stub = inventory.population("index_root")
    hidden = inventory.population("index_generations")
    assert stub.logical_bytes == _POINTER_STUB_BYTES
    assert stub.allocated_bytes <= 64 * 1024
    assert hidden.allocated_bytes >= _allocated(generation / "index.db")
    assert hidden.allocated_bytes > stub.allocated_bytes * 16
    assert inventory.total_allocated_bytes >= _allocated(generation / "index.db")
    assert inventory.active_generation_id == generation.name
    assert [measurement.generation_id for measurement in inventory.generations] == [generation.name]
    assert inventory.generations[0].active is True


def test_inventory_totals_cover_every_declared_population(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _archive_with_generation(root, generation_bytes=1024 * 1024)
    _dense(root / "source.db", 512 * 1024)
    _dense(root / "source.db-wal", 128 * 1024)
    _dense(root / "embeddings.db", 64 * 1024)
    _dense(root / "blob" / "ab" / ("c" * 62), 256 * 1024)
    _dense(root / "hooks" / "spool.jsonl", 32 * 1024)
    _dense(root / MAINTENANCE_STATE_DIRNAME / "candidate-capacity" / "old.json", 4096)
    _dense(root / "something-nobody-declared.bin", 128 * 1024)

    inventory = measure_archive_capacity(root)

    assert tuple(population.name for population in inventory.populations) == POPULATION_NAMES
    assert inventory.population("durable_tiers").allocated_bytes >= (512 + 128) * 1024
    assert inventory.population("derived_tiers").allocated_bytes >= 64 * 1024
    assert inventory.population("blob").allocated_bytes >= 256 * 1024
    assert inventory.population("spools").allocated_bytes >= 32 * 1024
    assert inventory.population("maintenance_receipts").allocated_bytes >= 4096
    assert inventory.population("unclassified").allocated_bytes >= 128 * 1024


def test_hardlinked_blob_is_charged_once(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _archive_with_generation(root, generation_bytes=64 * 1024)
    blob = root / "blob" / "ab" / ("c" * 62)
    _dense(blob, 1024 * 1024)
    link = root / "blob" / "ab" / ("d" * 62)
    os.link(blob, link)

    inventory = measure_archive_capacity(root)

    assert inventory.population("blob").allocated_bytes < 2 * _allocated(blob)


def test_sparse_generation_is_accounted_by_allocation_not_apparent_size(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    generation = _archive_with_generation(root, generation_bytes=8 * 1024 * 1024 * 1024, sparse=True)

    inventory = measure_archive_capacity(root)
    population = inventory.population("index_generations")

    assert population.logical_bytes >= 8 * 1024 * 1024 * 1024
    assert population.allocated_bytes < 1024 * 1024
    assert population.sparse_entries >= 1
    assert inventory.generations[0].allocated_bytes < inventory.generations[0].logical_bytes
    assert _allocated(generation / "index.db") < 1024 * 1024


def test_generations_root_replaced_by_a_symlink_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _archive_with_generation(root, generation_bytes=1024)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = root / GENERATIONS_DIRNAME
    for child in sorted(real.iterdir()):
        child.rename(elsewhere / child.name)
    real.rmdir()
    real.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ArchiveCapacityError, match="index generations root is a symlink"):
        measure_archive_capacity(root)


def test_links_inside_the_tree_are_not_followed_out_of_the_archive(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    generation = _archive_with_generation(root, generation_bytes=64 * 1024)
    outside = tmp_path / "outside"
    _dense(outside / "huge.bin", 16 * 1024 * 1024)
    (generation / "blob").symlink_to(outside, target_is_directory=True)

    inventory = measure_archive_capacity(root)

    assert inventory.total_allocated_bytes < 8 * 1024 * 1024


def test_pointer_target_outside_the_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _archive_with_generation(root, generation_bytes=1024)
    outside = tmp_path / "outside" / "index.db"
    _dense(outside, 1024)
    (root / ACTIVE_POINTER_FILENAME).write_text(str(outside.absolute()), encoding="utf-8")

    with pytest.raises(ArchiveCapacityError, match="cannot resolve archive identity"):
        measure_archive_capacity(root)


def test_missing_archive_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArchiveCapacityError, match="archive root is not a directory"):
        measure_archive_capacity(tmp_path / "absent")


def _projection_with_free_space(
    monkeypatch: pytest.MonkeyPatch, root: Path, available_bytes: int
) -> CandidateCapacityProjection:
    monkeypatch.setattr(
        "polylogue.maintenance.candidate_capacity._available_bytes",
        lambda _path: available_bytes,
    )
    return project_candidate_capacity(root)


def test_projection_refuses_when_the_hidden_generation_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    generation_bytes = 4 * 1024 * 1024
    _archive_with_generation(root, generation_bytes=generation_bytes)

    projection = _projection_with_free_space(monkeypatch, root, RESERVE_FLOOR_BYTES + generation_bytes)

    assert projection.observed_index_bytes >= generation_bytes
    assert projection.projected_index_bytes >= generation_bytes
    assert projection.sufficient is False
    assert projection.shortfall_bytes > 0
    with pytest.raises(InsufficientCapacityError, match="insufficient free space"):
        require_candidate_capacity(root, operation_id="refused-op")
    assert read_capacity_receipts(root) == ()


def test_projection_accepts_when_free_space_covers_the_projected_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _archive_with_generation(root, generation_bytes=4 * 1024 * 1024)

    generous = _projection_with_free_space(monkeypatch, root, 64 * 1024 * 1024 * 1024)

    assert generous.sufficient is True
    assert generous.shortfall_bytes == 0
    assert generous.predicted_peak_allocated_bytes > generous.retained_allocated_bytes


def test_ignoring_the_generation_population_would_accept_an_impossible_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generation term alone decides this boundary case.

    The durable evidence is negligible here, so a projection that read only
    the visible root files would size the build from a 77-byte stub and
    accept. The refusal is evidence the hidden generation is counted.
    """
    root = tmp_path / "archive"
    generation_bytes = 8 * 1024 * 1024
    _archive_with_generation(root, generation_bytes=generation_bytes)
    _dense(root / "source.db", 4096)

    # Free space covers the reserve, the receipt allowance, and a megabyte of
    # index: everything the projection needs except the hidden generation.
    available = RESERVE_FLOOR_BYTES + RECEIPT_GROWTH_FLOOR_BYTES + 1024 * 1024
    projection = _projection_with_free_space(monkeypatch, root, available)

    evidence_only_estimate = projection.evidence_bytes * DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE
    assert evidence_only_estimate < 1024 * 1024
    assert projection.observed_index_bytes >= generation_bytes
    assert projection.projected_index_bytes > available - RESERVE_FLOOR_BYTES - RECEIPT_GROWTH_FLOOR_BYTES
    assert projection.sufficient is False


def test_projection_includes_wal_temporary_receipt_and_reserve_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _archive_with_generation(root, generation_bytes=4 * 1024 * 1024)

    projection = _projection_with_free_space(monkeypatch, root, 64 * 1024 * 1024 * 1024)

    assert projection.wal_amplification_bytes > 0
    assert projection.temporary_amplification_bytes > 0
    assert projection.receipt_growth_bytes > 0
    assert projection.reserve_bytes >= RESERVE_FLOOR_BYTES
    assert projection.required_free_bytes == (
        projection.projected_index_bytes
        + projection.wal_amplification_bytes
        + projection.temporary_amplification_bytes
        + projection.receipt_growth_bytes
        + projection.reserve_bytes
    )
    assert projection.retained_allocated_bytes == projection.inventory.total_allocated_bytes


def test_prediction_and_observed_peak_are_recorded_and_calibrate_the_next_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    generation = _archive_with_generation(root, generation_bytes=1024 * 1024)
    _dense(root / "source.db", 1024 * 1024)

    assert calibrated_index_ratio(root) == (DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE, "default")

    projection = _projection_with_free_space(monkeypatch, root, 64 * 1024 * 1024 * 1024)
    record_capacity_prediction(root, operation_id="op-1", projection=projection)

    candidate = root / GENERATIONS_DIRNAME / "gen-2-bbbbbbbb"
    _dense(candidate / "index.db", 32 * 1024 * 1024)
    observed = record_capacity_observation(root, operation_id="op-1", candidate_root=candidate)

    assert observed is not None
    assert observed.as_dict()["schema"] == CAPACITY_RECEIPT_SCHEMA
    assert observed.predicted_peak_allocated_bytes == projection.predicted_peak_allocated_bytes
    assert observed.actual_peak_index_bytes >= 32 * 1024 * 1024
    assert observed.actual_peak_allocated_bytes == observed.baseline_allocated_bytes + observed.actual_peak_index_bytes

    ratio, source = calibrated_index_ratio(root)
    assert source == "recorded"
    assert ratio > DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE
    assert generation.exists()


def test_observed_peak_never_regresses_across_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "archive"
    _archive_with_generation(root, generation_bytes=1024 * 1024)
    projection = _projection_with_free_space(monkeypatch, root, 64 * 1024 * 1024 * 1024)
    record_capacity_prediction(root, operation_id="op-2", projection=projection)

    candidate = root / GENERATIONS_DIRNAME / "gen-3-cccccccc"
    _dense(candidate / "index.db", 8 * 1024 * 1024)
    high = record_capacity_observation(root, operation_id="op-2", candidate_root=candidate)
    assert high is not None
    peak = high.actual_peak_index_bytes

    (candidate / "index.db").unlink()
    _dense(candidate / "index.db", 1024)
    later = record_capacity_observation(root, operation_id="op-2", candidate_root=candidate)

    assert later is not None
    assert later.actual_peak_index_bytes == peak
    assert later.observations == 2


def test_observation_without_a_prediction_records_nothing(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    candidate = _archive_with_generation(root, generation_bytes=1024)

    assert record_capacity_observation(root, operation_id="never-predicted", candidate_root=candidate) is None
    assert read_capacity_receipts(root) == ()


def test_operation_id_may_not_escape_the_receipt_directory(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    candidate = _archive_with_generation(root, generation_bytes=1024)

    with pytest.raises(ArchiveCapacityError, match="single path component"):
        record_capacity_observation(root, operation_id="../escape", candidate_root=candidate)
