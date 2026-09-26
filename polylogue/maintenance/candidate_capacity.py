"""Whole-tree capacity accounting and the candidate-build headroom refusal.

An archive's ``index.db`` at the root is a pointer stub, not the index: the
promoted generation lives under :data:`GENERATIONS_DIRNAME`. Any measurement
that reads only the visible root files under-reads by the entire index, so a
build sized from it allocates against free space that was never there.

Accounting is by *allocated* blocks (``st_blocks``), never by apparent size:
a sparse file's logical length is not space anyone has to find.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from polylogue.maintenance.receipt_fs import (
    atomic_replace_receipt,
    existing_maintenance_receipt_directory,
    iter_pinned_receipts,
    maintenance_receipt_directory,
    read_optional_receipt,
)
from polylogue.storage.archive_identity import (
    GENERATIONS_DIRNAME,
    MAINTENANCE_STATE_DIRNAME,
    REBUILD_TRANSACTIONS_DIRNAME,
    ArchiveLocation,
    ArchiveLocationError,
)

#: ``st_blocks`` is defined in 512-byte units regardless of the filesystem's
#: own block size (POSIX ``stat``), so allocated bytes are always this scale.
ALLOCATION_UNIT_BYTES: Final = 512

#: Index bytes produced per byte of durable evidence (``source.db`` + blob
#: store) when no build has yet been measured on this archive. Superseded for
#: an archive as soon as one recorded receipt carries a final candidate sample.
DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE: Final = 4.0

#: WAL written against the candidate index during construction, and SQLite
#: temporary/journal space for the index builds that follow the row load.
WAL_AMPLIFICATION_FRACTION: Final = 0.25
TEMPORARY_AMPLIFICATION_FRACTION: Final = 0.10

#: Space the build refuses to consume so the filesystem is never driven to
#: zero by a rebuild that would then take the daemon down with it.
RESERVE_FRACTION: Final = 0.05
RESERVE_FLOOR_BYTES: Final = 256 * 1024 * 1024

#: A build writes one transaction record plus a numbered receipt per pass.
RECEIPT_GROWTH_FLOOR_BYTES: Final = 16 * 1024 * 1024

CAPACITY_RECEIPT_DIRNAME: Final = "candidate-capacity"
CAPACITY_RECEIPT_SCHEMA: Final = "polylogue.candidate-capacity.v2"

_BLOB_DIRNAME: Final = "blob"
_DURABLE_TIER_FILENAMES: Final = ("source.db", "user.db", "audit.db")
_DERIVED_TIER_FILENAMES: Final = ("embeddings.db", "ops.db")
_SPOOL_DIRNAMES: Final = ("hooks", "browser-capture", "render")
_SQLITE_SIDECAR_SUFFIXES: Final = ("-wal", "-shm", "-journal")

POPULATION_NAMES: Final = (
    "index_generations",
    "index_root",
    "rebuild_transactions",
    "durable_tiers",
    "derived_tiers",
    "blob",
    "spools",
    "maintenance_receipts",
    "unclassified",
)


class ArchiveCapacityError(RuntimeError):
    """The archive tree could not be measured safely and completely."""


class InsufficientCapacityError(RuntimeError):
    """Verified free space cannot hold the candidate this build would allocate."""

    def __init__(self, projection: CandidateCapacityProjection) -> None:
        deficient = ", ".join(
            f"{'/'.join(row.destinations)}: required {row.required_bytes}, available {row.available_bytes}"
            for row in projection.filesystem_requirements
            if row.shortfall_bytes > 0
        )
        super().__init__(
            "insufficient free space for a candidate index build: "
            f"{deficient}; short by {projection.shortfall_bytes} bytes "
            f"(projected index {projection.projected_index_bytes} bytes from "
            f"{projection.evidence_bytes} existing evidence bytes and "
            f"{projection.prospective_retained_allocation_bytes} prospective retained bytes, "
            f"calibration {projection.calibration_source})"
        )
        self.projection = projection


@dataclass(frozen=True, slots=True)
class PopulationMeasurement:
    """One named population of the archive tree, measured by allocated blocks."""

    name: str
    allocated_bytes: int
    logical_bytes: int
    entries: int
    sparse_entries: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "allocated_bytes": self.allocated_bytes,
            "logical_bytes": self.logical_bytes,
            "entries": self.entries,
            "sparse_entries": self.sparse_entries,
        }


@dataclass(frozen=True, slots=True)
class GenerationMeasurement:
    """One index generation directory, measured independently of its siblings."""

    generation_id: str
    allocated_bytes: int
    logical_bytes: int
    active: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "generation_id": self.generation_id,
            "allocated_bytes": self.allocated_bytes,
            "logical_bytes": self.logical_bytes,
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class ArchiveCapacityInventory:
    """Complete allocated-byte accounting for one archive tree."""

    archive_root: Path
    active_index_path: Path
    active_generation_id: str | None
    populations: tuple[PopulationMeasurement, ...]
    generations: tuple[GenerationMeasurement, ...]
    available_bytes: int
    destinations: tuple[CapacityDestination, ...] = ()

    @property
    def total_allocated_bytes(self) -> int:
        return sum(population.allocated_bytes for population in self.populations)

    @property
    def total_logical_bytes(self) -> int:
        return sum(population.logical_bytes for population in self.populations)

    def population(self, name: str) -> PopulationMeasurement:
        for population in self.populations:
            if population.name == name:
                return population
        raise KeyError(name)

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "active_index_path": str(self.active_index_path),
            "active_generation_id": self.active_generation_id,
            "populations": [population.as_dict() for population in self.populations],
            "generations": [generation.as_dict() for generation in self.generations],
            "available_bytes": self.available_bytes,
            "destinations": [destination.as_dict() for destination in self.destinations],
            "total_allocated_bytes": self.total_allocated_bytes,
            "total_logical_bytes": self.total_logical_bytes,
        }


@dataclass(frozen=True, slots=True)
class CapacityDestination:
    name: str
    probe: Path
    device: int
    available_bytes: int
    block_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "probe": str(self.probe),
            "device": self.device,
            "available_bytes": self.available_bytes,
            "block_bytes": self.block_bytes,
        }


@dataclass(frozen=True, slots=True)
class FilesystemRequirement:
    device: int
    destinations: tuple[str, ...]
    required_bytes: int
    available_bytes: int

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_bytes - self.available_bytes)

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "destinations": list(self.destinations),
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
            "shortfall_bytes": self.shortfall_bytes,
        }


@dataclass(frozen=True, slots=True)
class CapacityReceipt:
    """One build's prediction and sampled candidate allocation."""

    operation_id: str
    evidence_bytes: int
    calibration_ratio: float
    calibration_source: str
    predicted_index_bytes: int
    predicted_peak_allocated_bytes: int
    required_free_bytes: int
    available_bytes_at_prediction: int
    shortfall_bytes: int
    baseline_allocated_bytes: int
    prospective_material_bytes: int = 0
    prospective_retained_allocation_bytes: int = 0
    prospective_source_db_allocation_bytes: int = 0
    baseline_digest: str | None = None
    material_byte_definition: str | None = None
    status: str = "admitted"
    actual_evidence_bytes: int = 0
    final_candidate_allocated_bytes: int = 0
    candidate_generation_id: str | None = None
    filesystem_requirements: tuple[FilesystemRequirement, ...] = ()
    observations: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": CAPACITY_RECEIPT_SCHEMA,
            "operation_id": self.operation_id,
            "evidence_bytes": self.evidence_bytes,
            "calibration_ratio": self.calibration_ratio,
            "calibration_source": self.calibration_source,
            "predicted_index_bytes": self.predicted_index_bytes,
            "predicted_peak_allocated_bytes": self.predicted_peak_allocated_bytes,
            "required_free_bytes": self.required_free_bytes,
            "available_bytes_at_prediction": self.available_bytes_at_prediction,
            "shortfall_bytes": self.shortfall_bytes,
            "baseline_allocated_bytes": self.baseline_allocated_bytes,
            "prospective_material_bytes": self.prospective_material_bytes,
            "prospective_retained_allocation_bytes": self.prospective_retained_allocation_bytes,
            "prospective_source_db_allocation_bytes": self.prospective_source_db_allocation_bytes,
            "baseline_digest": self.baseline_digest,
            "material_byte_definition": self.material_byte_definition,
            "status": self.status,
            "actual_evidence_bytes": self.actual_evidence_bytes,
            "final_candidate_allocated_bytes": self.final_candidate_allocated_bytes,
            "candidate_generation_id": self.candidate_generation_id,
            "filesystem_requirements": [requirement.as_dict() for requirement in self.filesystem_requirements],
            "observations": self.observations,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object] | None) -> CapacityReceipt | None:
        if payload is None or payload.get("schema") != CAPACITY_RECEIPT_SCHEMA:
            return None
        operation_id = _str_field(payload, "operation_id")
        if not operation_id:
            return None
        return cls(
            operation_id=operation_id,
            evidence_bytes=_int_field(payload, "evidence_bytes"),
            calibration_ratio=_float_field(payload, "calibration_ratio"),
            calibration_source=_str_field(payload, "calibration_source"),
            predicted_index_bytes=_int_field(payload, "predicted_index_bytes"),
            predicted_peak_allocated_bytes=_int_field(payload, "predicted_peak_allocated_bytes"),
            required_free_bytes=_int_field(payload, "required_free_bytes"),
            available_bytes_at_prediction=_int_field(payload, "available_bytes_at_prediction"),
            shortfall_bytes=_int_field(payload, "shortfall_bytes"),
            baseline_allocated_bytes=_int_field(payload, "baseline_allocated_bytes"),
            prospective_material_bytes=_int_field(payload, "prospective_material_bytes"),
            prospective_retained_allocation_bytes=_int_field(payload, "prospective_retained_allocation_bytes"),
            prospective_source_db_allocation_bytes=_int_field(payload, "prospective_source_db_allocation_bytes"),
            baseline_digest=_str_field(payload, "baseline_digest") or None,
            material_byte_definition=_str_field(payload, "material_byte_definition") or None,
            status=_str_field(payload, "status"),
            actual_evidence_bytes=_int_field(payload, "actual_evidence_bytes"),
            final_candidate_allocated_bytes=_int_field(payload, "final_candidate_allocated_bytes"),
            candidate_generation_id=_str_field(payload, "candidate_generation_id") or None,
            filesystem_requirements=_requirements_from_payload(payload.get("filesystem_requirements")),
            observations=_int_field(payload, "observations"),
        )


@dataclass(frozen=True, slots=True)
class CandidateCapacityProjection:
    """What one candidate index build would allocate, against verified free space."""

    inventory: ArchiveCapacityInventory
    evidence_bytes: int
    observed_index_bytes: int
    calibration_ratio: float
    calibration_source: str
    projected_index_bytes: int
    wal_amplification_bytes: int
    temporary_amplification_bytes: int
    receipt_growth_bytes: int
    reserve_bytes: int
    retained_allocated_bytes: int
    prospective_material_bytes: int = 0
    prospective_retained_allocation_bytes: int = 0
    prospective_source_db_allocation_bytes: int = 0
    baseline_digest: str | None = None
    material_byte_definition: str | None = None
    filesystem_requirements: tuple[FilesystemRequirement, ...] = ()

    @property
    def required_free_bytes(self) -> int:
        return sum(requirement.required_bytes for requirement in self.filesystem_requirements)

    @property
    def predicted_peak_allocated_bytes(self) -> int:
        return (
            self.retained_allocated_bytes
            + self.prospective_retained_allocation_bytes
            + self.prospective_source_db_allocation_bytes
            + self.projected_index_bytes
            + self.wal_amplification_bytes
            + self.temporary_amplification_bytes
            + self.receipt_growth_bytes
        )

    @property
    def available_bytes(self) -> int:
        return sum(requirement.available_bytes for requirement in self.filesystem_requirements)

    @property
    def sufficient(self) -> bool:
        return all(requirement.shortfall_bytes == 0 for requirement in self.filesystem_requirements)

    @property
    def shortfall_bytes(self) -> int:
        return sum(requirement.shortfall_bytes for requirement in self.filesystem_requirements)

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_bytes": self.evidence_bytes,
            "observed_index_bytes": self.observed_index_bytes,
            "calibration_ratio": self.calibration_ratio,
            "calibration_source": self.calibration_source,
            "projected_index_bytes": self.projected_index_bytes,
            "wal_amplification_bytes": self.wal_amplification_bytes,
            "temporary_amplification_bytes": self.temporary_amplification_bytes,
            "receipt_growth_bytes": self.receipt_growth_bytes,
            "reserve_bytes": self.reserve_bytes,
            "retained_allocated_bytes": self.retained_allocated_bytes,
            "prospective_material_bytes": self.prospective_material_bytes,
            "prospective_retained_allocation_bytes": self.prospective_retained_allocation_bytes,
            "prospective_source_db_allocation_bytes": self.prospective_source_db_allocation_bytes,
            "baseline_digest": self.baseline_digest,
            "material_byte_definition": self.material_byte_definition,
            "filesystem_requirements": [requirement.as_dict() for requirement in self.filesystem_requirements],
            "required_free_bytes": self.required_free_bytes,
            "predicted_peak_allocated_bytes": self.predicted_peak_allocated_bytes,
            "available_bytes": self.available_bytes,
            "sufficient": self.sufficient,
            "shortfall_bytes": self.shortfall_bytes,
            "inventory": self.inventory.as_dict(),
        }


def allocated_bytes(metadata: os.stat_result) -> int:
    """Blocks actually charged to the filesystem for one inode."""
    return int(getattr(metadata, "st_blocks", 0)) * ALLOCATION_UNIT_BYTES


class _Accumulator:
    """Allocated/logical totals for one population, deduplicated by inode.

    The dedup set is shared across every population in one inventory, so a
    hardlink, a generation's back-link to a durable tier, and a pointer target
    that resolves inside an already-counted generation are each charged once.
    """

    def __init__(self, name: str, seen: set[tuple[int, int]]) -> None:
        self.name = name
        self._seen = seen
        self.allocated = 0
        self.logical = 0
        self.entries = 0
        self.sparse = 0

    def add(self, metadata: os.stat_result) -> bool:
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in self._seen:
            return False
        self._seen.add(identity)
        allocated = allocated_bytes(metadata)
        logical = int(metadata.st_size)
        self.allocated += allocated
        self.logical += logical
        self.entries += 1
        if logical > allocated:
            self.sparse += 1
        return True

    def counted(self, metadata: os.stat_result) -> bool:
        return (metadata.st_dev, metadata.st_ino) in self._seen

    def measurement(self) -> PopulationMeasurement:
        return PopulationMeasurement(
            name=self.name,
            allocated_bytes=self.allocated,
            logical_bytes=self.logical,
            entries=self.entries,
            sparse_entries=self.sparse,
        )


def _lstat(path: Path, *, label: str) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ArchiveCapacityError(f"cannot inspect {label}: {path}") from exc


def _iter_tree(root: Path, *, label: str) -> Iterator[os.stat_result]:
    """Yield every inode under ``root`` without following any link out of it.

    A symlink is charged as its own (tiny) inode and never descended: below the
    archive root every link either points back at a tier this inventory already
    counts, or out of the archive entirely.
    """
    # Keep the inode identity observed when a directory is queued.  A
    # rebuild can be writing alongside this read, and a path that was a real
    # directory at enumeration time must not become an external symlink
    # before we descend into it.  Re-checking here makes the no-follow
    # guarantee hold across that small race as well as for a static tree.
    metadata = _lstat(root, label=label)
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        return
    pending: list[tuple[Path, tuple[int, int]]] = [(root, (metadata.st_dev, metadata.st_ino))]
    while pending:
        directory, expected_identity = pending.pop()
        current = _lstat(directory, label=label)
        if (
            current is None
            or not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
        ):
            continue
        try:
            with os.scandir(directory) as scan:
                entries = list(scan)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ArchiveCapacityError(f"cannot enumerate {label}: {directory}") from exc
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ArchiveCapacityError(f"cannot inspect {label} entry: {entry.path}") from exc
            yield metadata
            if stat.S_ISDIR(metadata.st_mode):
                pending.append((Path(entry.path), (metadata.st_dev, metadata.st_ino)))


def _measure_path(
    accumulator: _Accumulator,
    path: Path,
    *,
    label: str,
    follow_root_link: bool = False,
    allowed_link_root: Path | None = None,
) -> None:
    """Charge ``path`` (and its tree, when a directory) to ``accumulator``.

    ``follow_root_link`` admits the symlink-farm archive root, whose tier
    entries legitimately point at the real location. It applies to the named
    entry only; nothing discovered underneath it is ever followed.
    """
    metadata = _lstat(path, label=label)
    if metadata is None:
        return
    if stat.S_ISLNK(metadata.st_mode):
        accumulator.add(metadata)
        if not follow_root_link:
            return
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArchiveCapacityError(f"cannot resolve {label} symlink: {path}") from exc
        if allowed_link_root is not None:
            try:
                target.relative_to(allowed_link_root)
            except ValueError as exc:
                raise ArchiveCapacityError(f"{label} symlink escapes the physical archive: {path} -> {target}") from exc
        _measure_path(accumulator, target, label=label)
        return
    accumulator.add(metadata)
    if stat.S_ISDIR(metadata.st_mode):
        for child in _iter_tree(path, label=label):
            accumulator.add(child)


def _measure_database(
    accumulator: _Accumulator,
    path: Path,
    *,
    label: str,
    allowed_link_root: Path | None = None,
) -> None:
    """Charge one SQLite tier and its WAL/shm/journal sidecars."""
    _measure_path(
        accumulator,
        path,
        label=label,
        follow_root_link=True,
        allowed_link_root=allowed_link_root,
    )
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        _measure_path(
            accumulator,
            path.with_name(path.name + suffix),
            label=f"{label} sidecar",
            follow_root_link=True,
            allowed_link_root=allowed_link_root,
        )


def _checked_generations_root(root: Path, *, label: str) -> Path:
    """Refuse a generation-root link instead of measuring an arbitrary target."""
    metadata = _lstat(root, label="index generations root")
    if metadata is not None and stat.S_ISLNK(metadata.st_mode):
        raise ArchiveCapacityError(f"{label} is a symlink: {root}")
    return root


def _generations_root(archive_root: Path) -> Path:
    """The configured archive's generations root."""
    return _checked_generations_root(archive_root / GENERATIONS_DIRNAME, label="index generations root")


def _generation_roots(configured: Path, location: ArchiveLocation) -> tuple[Path, ...]:
    """Every generation root reachable through the configured archive topology.

    A symlink-farm archive keeps its active index and future candidates beside
    the canonical pointer target, not beside the configured root. That target
    is an authenticated archive member through ``ArchiveLocation``. Its
    generation directory is therefore retained state to inventory, while any
    links encountered *inside* either generation root remain untrusted and are
    never followed by ``_iter_tree``.
    """
    from polylogue.storage.index_generation import canonical_active_index_path

    target = canonical_active_index_path(location).parent / GENERATIONS_DIRNAME
    target = _checked_generations_root(target, label="active pointer generation root")
    if target.absolute() == configured.absolute():
        return (configured,)
    return (configured, target)


def _physical_archive_root(location: ArchiveLocation) -> Path:
    """Return the authenticated physical root for root-level symlink checks.

    A promoted index normally resolves below ``.index-generations`` while a
    symlink-farm archive resolves the same way from a different configured
    root.  In both cases the parent of that hidden directory is the physical
    archive root.  Before generations exist, the active index's parent is the
    only honest root available.
    """
    active = location.active_index_path.resolve(strict=False)
    for ancestor in (active, *active.parents):
        if ancestor.name == GENERATIONS_DIRNAME:
            return ancestor.parent
    return active.parent


def _capacity_destination(name: str, path: Path) -> CapacityDestination:
    """Probe the filesystem of an existing destination or its nearest ancestor."""
    probe = path.resolve(strict=False)
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise ArchiveCapacityError(f"cannot locate filesystem for {name}: {path}")
        probe = parent
    try:
        metadata = probe.stat()
        statistics = os.statvfs(probe)
    except OSError as exc:
        raise ArchiveCapacityError(f"cannot measure filesystem for {name}: {probe}") from exc
    block_bytes = int(statistics.f_frsize)
    if block_bytes <= 0:
        raise ArchiveCapacityError(f"invalid allocation block size for {name}: {probe}")
    return CapacityDestination(name, probe, int(metadata.st_dev), _available_bytes(probe), block_bytes)


def evidence_allocation_block_bytes(archive_root: Path) -> tuple[int, int]:
    """Return allocation units at the blob and source database destinations."""
    root = Path(archive_root)
    return (
        _capacity_destination("blob", root / _BLOB_DIRNAME).block_bytes,
        _capacity_destination("source_db", root / "source.db").block_bytes,
    )


def _filesystem_requirements(
    destinations: tuple[CapacityDestination, ...], growth: dict[str, int]
) -> tuple[tuple[FilesystemRequirement, ...], int]:
    grouped: dict[int, tuple[list[str], int, int]] = {}
    for destination in destinations:
        amount = growth[destination.name]
        if amount <= 0:
            continue
        names, total, available = grouped.get(destination.device, ([], 0, destination.available_bytes))
        names.append(destination.name)
        grouped[destination.device] = (names, total + amount, min(available, destination.available_bytes))
    requirements: list[FilesystemRequirement] = []
    total_reserve = 0
    for device, (names, amount, available) in grouped.items():
        reserve = max(RESERVE_FLOOR_BYTES, math.ceil(amount * RESERVE_FRACTION))
        total_reserve += reserve
        requirements.append(FilesystemRequirement(device, tuple(names), amount + reserve, available))
    return tuple(requirements), total_reserve


def _available_bytes(path: Path) -> int:
    try:
        statistics = os.statvfs(path)
    except OSError as exc:
        raise ArchiveCapacityError(f"cannot read free space for: {path}") from exc
    # ``f_bavail`` is what an unprivileged writer may actually use; ``f_bfree``
    # includes the filesystem's own root reserve and would over-promise.
    return int(statistics.f_frsize) * int(statistics.f_bavail)


def measure_archive_capacity(archive_root: Path) -> ArchiveCapacityInventory:
    """Measure every population of ``archive_root`` by allocated bytes.

    Resolves the active-index pointer to its target so a stub can never stand
    in for the generation it names, and buckets whatever is left at the root
    into ``unclassified`` so the total is the whole tree.
    """
    root = Path(archive_root).absolute()
    if not root.is_dir():
        raise ArchiveCapacityError(f"archive root is not a directory: {root}")
    # Path safety before identity: a generations root replaced by a link is
    # refused on its own terms, whatever the pointer then claims.
    configured_generations_root = _generations_root(root)
    try:
        location = ArchiveLocation.resolve(root)
    except ArchiveLocationError as exc:
        raise ArchiveCapacityError(f"cannot resolve archive identity: {root}") from exc
    # The store writes generations under ``canonical_active_index_path(...)``'s
    # directory (IndexGenerationStore.generations_root), which in a
    # symlink-farm layout is not the archive root and can sit on another
    # filesystem.
    generation_roots = _generation_roots(configured_generations_root, location)
    store_generations_root = generation_roots[-1]
    physical_root = _physical_archive_root(location)

    # A pointer names a required active index.  ArchiveLocation validates its
    # topology and containment, but intentionally permits a dangling target so
    # callers can inspect identity. Capacity preflight must fail closed before
    # a build allocates against an archive whose active evidence is absent.
    if location.active_pointer is not None:
        try:
            active_metadata = location.active_index_path.stat()
        except OSError as exc:
            raise ArchiveCapacityError(f"cannot inspect active pointer target: {location.active_index_path}") from exc
        if not stat.S_ISREG(active_metadata.st_mode):
            raise ArchiveCapacityError(f"active pointer target is not a regular file: {location.active_index_path}")

    seen: set[tuple[int, int]] = set()
    accumulators = {name: _Accumulator(name, seen) for name in POPULATION_NAMES}

    generations = _measure_generations(accumulators["index_generations"], generation_roots, location)

    # The root's own ``index.db`` is whatever is literally there -- a stub, a
    # promotion symlink, or a pre-generation index. The pointer target is the
    # index, and is charged whether or not it sits under the generations root,
    # so a stub can never stand in for it.
    # The conventional root index may be an explicit split-root symlink even
    # before a pointer exists; keep charging that configured path while the
    # authenticated pointer target is checked above.
    _measure_database(accumulators["index_root"], root / "index.db", label="root index")
    _measure_database(
        accumulators["index_generations"],
        location.active_index_path,
        label="active index",
        allowed_link_root=physical_root,
    )
    _measure_path(
        accumulators["rebuild_transactions"], root / REBUILD_TRANSACTIONS_DIRNAME, label="rebuild transactions"
    )
    for filename in _DURABLE_TIER_FILENAMES:
        _measure_database(
            accumulators["durable_tiers"],
            root / filename,
            label=f"durable tier {filename}",
            allowed_link_root=physical_root,
        )
    for filename in _DERIVED_TIER_FILENAMES:
        _measure_database(
            accumulators["derived_tiers"],
            root / filename,
            label=f"derived tier {filename}",
            allowed_link_root=physical_root,
        )
    _measure_path(
        accumulators["blob"],
        root / _BLOB_DIRNAME,
        label="blob store",
        follow_root_link=True,
        allowed_link_root=physical_root,
    )
    for dirname in _SPOOL_DIRNAMES:
        _measure_path(
            accumulators["spools"],
            root / dirname,
            label=f"spool {dirname}",
            follow_root_link=True,
            allowed_link_root=physical_root,
        )
    _measure_path(
        accumulators["maintenance_receipts"],
        root / MAINTENANCE_STATE_DIRNAME,
        label="maintenance receipts",
    )

    unclassified = accumulators["unclassified"]
    # The root directory is itself allocated filesystem space.  Charge only
    # its inode here; recursively walking it would repeat the whole archive
    # after the named populations have already been measured.
    root_metadata = _lstat(root, label="archive root")
    if root_metadata is not None:
        unclassified.add(root_metadata)
    try:
        with os.scandir(root) as scan:
            remainder = list(scan)
    except OSError as exc:
        raise ArchiveCapacityError(f"cannot enumerate archive root: {root}") from exc
    for entry in remainder:
        metadata = _lstat(Path(entry.path), label="archive root entry")
        # An already-charged population is not re-walked: the dedup set would
        # drop every inode anyway, at the cost of a second pass over the blob
        # store's millions of files.
        if metadata is None or unclassified.counted(metadata):
            continue
        _measure_path(unclassified, Path(entry.path), label="archive root entry")

    destinations = (
        _capacity_destination("candidate", store_generations_root),
        _capacity_destination("blob", root / _BLOB_DIRNAME),
        _capacity_destination("source_db", root / "source.db"),
        _capacity_destination("receipt", root / MAINTENANCE_STATE_DIRNAME),
    )
    return ArchiveCapacityInventory(
        archive_root=root,
        active_index_path=location.active_index_path,
        active_generation_id=_active_generation_id(location),
        populations=tuple(accumulators[name].measurement() for name in POPULATION_NAMES),
        generations=generations,
        available_bytes=destinations[0].available_bytes,
        destinations=destinations,
    )


def _active_generation_id(location: ArchiveLocation) -> str | None:
    parts = location.active_index_path.resolve(strict=False).parts
    try:
        depth = parts.index(GENERATIONS_DIRNAME)
    except ValueError:
        return None
    return parts[depth + 1] if len(parts) > depth + 1 else None


def _measure_generations(
    accumulator: _Accumulator, generations_roots: tuple[Path, ...], location: ArchiveLocation
) -> tuple[GenerationMeasurement, ...]:
    """Measure every retained generation directory, then charge each root."""
    active_id = _active_generation_id(location)
    measurements: list[GenerationMeasurement] = []
    for generations_root in generations_roots:
        try:
            with os.scandir(generations_root) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except FileNotFoundError:
            entries = []
        except OSError as exc:
            raise ArchiveCapacityError(f"cannot enumerate index generations: {generations_root}") from exc
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False) or not entry.name.startswith("gen-"):
                continue
            # Per-generation figures are measured on their own dedup set: a
            # generation's size is a property of that generation, not of the
            # order the inventory happened to walk its siblings in.
            isolated = _Accumulator(entry.name, set())
            _measure_path(isolated, Path(entry.path), label=f"index generation {entry.name}")
            measurements.append(
                GenerationMeasurement(
                    generation_id=entry.name,
                    allocated_bytes=isolated.allocated,
                    logical_bytes=isolated.logical,
                    active=entry.name == active_id,
                )
            )
        _measure_path(accumulator, generations_root, label="index generations")
    return tuple(measurements)


def project_candidate_capacity(
    archive_root: Path,
    *,
    prospective_material_bytes: int = 0,
    prospective_retained_allocation_bytes: int | None = None,
    prospective_source_db_allocation_bytes: int = 0,
    baseline_digest: str | None = None,
    material_byte_definition: str | None = None,
) -> CandidateCapacityProjection:
    """Project one candidate index build against the measured archive tree."""
    if prospective_material_bytes < 0:
        raise ArchiveCapacityError("prospective material bytes cannot be negative")
    if prospective_retained_allocation_bytes is None:
        prospective_retained_allocation_bytes = prospective_material_bytes
    if prospective_retained_allocation_bytes < prospective_material_bytes:
        raise ArchiveCapacityError("prospective retained allocation cannot be smaller than material bytes")
    if prospective_source_db_allocation_bytes < 0:
        raise ArchiveCapacityError("prospective source database allocation cannot be negative")
    inventory = measure_archive_capacity(archive_root)
    evidence_bytes = (
        inventory.population("durable_tiers").allocated_bytes + inventory.population("blob").allocated_bytes
    )
    observed_index_bytes = max(
        (generation.allocated_bytes for generation in inventory.generations),
        default=inventory.population("index_generations").allocated_bytes,
    )
    ratio, source = calibrated_index_ratio(Path(archive_root))
    # The candidate reproduces the current index from the same evidence: it is
    # no smaller than the largest generation on disk, and no smaller than what
    # the evidence has been observed to expand into.
    projected_index_bytes = max(
        observed_index_bytes, math.ceil((evidence_bytes + prospective_retained_allocation_bytes) * ratio)
    )
    wal_amplification_bytes = math.ceil(projected_index_bytes * WAL_AMPLIFICATION_FRACTION)
    temporary_amplification_bytes = math.ceil(projected_index_bytes * TEMPORARY_AMPLIFICATION_FRACTION)
    receipt_growth_bytes = max(
        RECEIPT_GROWTH_FLOOR_BYTES,
        inventory.population("maintenance_receipts").allocated_bytes
        + inventory.population("rebuild_transactions").allocated_bytes,
    )
    requirements, reserve_bytes = _filesystem_requirements(
        inventory.destinations,
        {
            "candidate": projected_index_bytes + wal_amplification_bytes + temporary_amplification_bytes,
            "blob": prospective_retained_allocation_bytes,
            "source_db": prospective_source_db_allocation_bytes,
            "receipt": receipt_growth_bytes,
        },
    )
    return CandidateCapacityProjection(
        inventory=inventory,
        evidence_bytes=evidence_bytes,
        observed_index_bytes=observed_index_bytes,
        calibration_ratio=ratio,
        calibration_source=source,
        projected_index_bytes=projected_index_bytes,
        wal_amplification_bytes=wal_amplification_bytes,
        temporary_amplification_bytes=temporary_amplification_bytes,
        receipt_growth_bytes=receipt_growth_bytes,
        reserve_bytes=reserve_bytes,
        retained_allocated_bytes=inventory.total_allocated_bytes,
        prospective_material_bytes=prospective_material_bytes,
        prospective_retained_allocation_bytes=prospective_retained_allocation_bytes,
        prospective_source_db_allocation_bytes=prospective_source_db_allocation_bytes,
        baseline_digest=baseline_digest,
        material_byte_definition=material_byte_definition,
        filesystem_requirements=requirements,
    )


def require_candidate_capacity(
    archive_root: Path,
    *,
    operation_id: str,
    prospective_material_bytes: int = 0,
    prospective_retained_allocation_bytes: int | None = None,
    prospective_source_db_allocation_bytes: int = 0,
    baseline_digest: str | None = None,
    material_byte_definition: str | None = None,
) -> CandidateCapacityProjection:
    """Refuse a candidate build whose projected peak does not fit, before allocating."""
    projection = project_candidate_capacity(
        archive_root,
        prospective_material_bytes=prospective_material_bytes,
        prospective_retained_allocation_bytes=prospective_retained_allocation_bytes,
        prospective_source_db_allocation_bytes=prospective_source_db_allocation_bytes,
        baseline_digest=baseline_digest,
        material_byte_definition=material_byte_definition,
    )
    if not projection.sufficient:
        record_capacity_prediction(
            Path(archive_root), operation_id=operation_id, projection=projection, status="refused"
        )
        raise InsufficientCapacityError(projection)
    record_capacity_prediction(Path(archive_root), operation_id=operation_id, projection=projection)
    return projection


def _ensure_maintenance_state(archive_root: Path) -> None:
    """Create ``.maintenance-state`` under a root that is not a link."""
    state = Path(archive_root) / MAINTENANCE_STATE_DIRNAME
    metadata = _lstat(state, label="maintenance state")
    if metadata is not None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArchiveCapacityError(f"maintenance state is not a real directory: {state}")
        return
    try:
        state.mkdir(mode=0o700, parents=False, exist_ok=True)
    except OSError as exc:
        raise ArchiveCapacityError(f"cannot create maintenance state: {state}") from exc


def _receipt_filename(operation_id: str) -> str:
    if not operation_id or Path(operation_id).name != operation_id or operation_id in {".", ".."}:
        raise ArchiveCapacityError(f"operation id is not a single path component: {operation_id!r}")
    return f"{operation_id}.json"


def record_capacity_prediction(
    archive_root: Path, *, operation_id: str, projection: CandidateCapacityProjection, status: str = "admitted"
) -> CapacityReceipt:
    """Persist what this build predicted, so its outcome can calibrate the next one."""
    filename = _receipt_filename(operation_id)
    receipt = CapacityReceipt(
        operation_id=operation_id,
        evidence_bytes=projection.evidence_bytes,
        calibration_ratio=projection.calibration_ratio,
        calibration_source=projection.calibration_source,
        predicted_index_bytes=projection.projected_index_bytes,
        predicted_peak_allocated_bytes=projection.predicted_peak_allocated_bytes,
        required_free_bytes=projection.required_free_bytes,
        available_bytes_at_prediction=projection.available_bytes,
        shortfall_bytes=projection.shortfall_bytes,
        baseline_allocated_bytes=projection.retained_allocated_bytes,
        prospective_material_bytes=projection.prospective_material_bytes,
        prospective_retained_allocation_bytes=projection.prospective_retained_allocation_bytes,
        prospective_source_db_allocation_bytes=projection.prospective_source_db_allocation_bytes,
        baseline_digest=projection.baseline_digest,
        material_byte_definition=projection.material_byte_definition,
        status=status,
        filesystem_requirements=projection.filesystem_requirements,
    )
    _ensure_maintenance_state(Path(archive_root))
    with maintenance_receipt_directory(Path(archive_root), CAPACITY_RECEIPT_DIRNAME) as directory_fd:
        atomic_replace_receipt(directory_fd, filename, _encode(receipt.as_dict()))
    return receipt


def record_capacity_observation(
    archive_root: Path, *, operation_id: str, candidate_root: Path
) -> CapacityReceipt | None:
    """Record the candidate's allocation at this observation boundary.

    Returns ``None`` when no prediction was recorded for ``operation_id``: an
    observation without a prediction calibrates nothing.
    """
    filename = _receipt_filename(operation_id)
    isolated = _Accumulator("candidate", set())
    _measure_path(isolated, Path(candidate_root), label="candidate generation")
    with existing_maintenance_receipt_directory(Path(archive_root), CAPACITY_RECEIPT_DIRNAME) as directory_fd:
        if directory_fd is None:
            return None
        raw = read_optional_receipt(directory_fd, filename)
        if raw is None:
            return None
        recorded = CapacityReceipt.from_payload(_decode(raw))
        if recorded is None or recorded.status != "admitted":
            return None
        inventory = measure_archive_capacity(Path(archive_root))
        actual_evidence_bytes = (
            inventory.population("durable_tiers").allocated_bytes + inventory.population("blob").allocated_bytes
        )
        observed = replace(
            recorded,
            actual_evidence_bytes=actual_evidence_bytes,
            final_candidate_allocated_bytes=isolated.allocated,
            candidate_generation_id=Path(candidate_root).name,
            observations=recorded.observations + 1,
        )
        atomic_replace_receipt(directory_fd, filename, _encode(observed.as_dict()))
        return observed


def read_capacity_receipts(archive_root: Path) -> tuple[CapacityReceipt, ...]:
    """Every recorded capacity receipt for ``archive_root``, by receipt name."""
    receipts: list[CapacityReceipt] = []
    with existing_maintenance_receipt_directory(Path(archive_root), CAPACITY_RECEIPT_DIRNAME) as directory_fd:
        if directory_fd is None:
            return ()
        for _name, raw in iter_pinned_receipts(directory_fd):
            recorded = CapacityReceipt.from_payload(_decode(raw))
            if recorded is not None:
                receipts.append(recorded)
    return tuple(receipts)


def calibrated_index_ratio(archive_root: Path) -> tuple[float, str]:
    """Index-bytes-per-evidence-byte from recorded outcomes, else the default.

    The largest observed ratio wins: a projection that under-reads is the
    failure this accounting exists to prevent, and one build that expanded
    further than its siblings is evidence the next one can too.
    """
    try:
        receipts = read_capacity_receipts(Path(archive_root))
    except (OSError, RuntimeError):
        return DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE, "default"
    ratios = [
        receipt.final_candidate_allocated_bytes / receipt.actual_evidence_bytes
        for receipt in receipts
        if receipt.status == "admitted"
        and receipt.final_candidate_allocated_bytes > 0
        and receipt.actual_evidence_bytes > 0
    ]
    if not ratios:
        return DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE, "default"
    return max(ratios), "recorded"


def _encode(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _decode(raw: bytes) -> dict[str, object] | None:
    try:
        payload: Any = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _requirements_from_payload(value: object) -> tuple[FilesystemRequirement, ...]:
    if not isinstance(value, list):
        return ()
    requirements: list[FilesystemRequirement] = []
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("destinations"), list):
            return ()
        names = row["destinations"]
        if not all(isinstance(name, str) for name in names):
            return ()
        requirements.append(
            FilesystemRequirement(
                device=_int_field(row, "device"),
                destinations=tuple(names),
                required_bytes=_int_field(row, "required_bytes"),
                available_bytes=_int_field(row, "available_bytes"),
            )
        )
    return tuple(requirements)


def _int_field(payload: dict[str, object], name: str) -> int:
    value = payload.get(name, 0)
    return int(value) if isinstance(value, int | float) else 0


def _float_field(payload: dict[str, object], name: str) -> float:
    value = payload.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _str_field(payload: dict[str, object], name: str) -> str:
    value = payload.get(name, "")
    return value if isinstance(value, str) else ""


__all__ = [
    "ALLOCATION_UNIT_BYTES",
    "CAPACITY_RECEIPT_DIRNAME",
    "CAPACITY_RECEIPT_SCHEMA",
    "DEFAULT_INDEX_BYTES_PER_EVIDENCE_BYTE",
    "POPULATION_NAMES",
    "RECEIPT_GROWTH_FLOOR_BYTES",
    "RESERVE_FLOOR_BYTES",
    "RESERVE_FRACTION",
    "TEMPORARY_AMPLIFICATION_FRACTION",
    "WAL_AMPLIFICATION_FRACTION",
    "ArchiveCapacityError",
    "ArchiveCapacityInventory",
    "CandidateCapacityProjection",
    "CapacityReceipt",
    "GenerationMeasurement",
    "InsufficientCapacityError",
    "PopulationMeasurement",
    "allocated_bytes",
    "calibrated_index_ratio",
    "measure_archive_capacity",
    "project_candidate_capacity",
    "read_capacity_receipts",
    "record_capacity_observation",
    "record_capacity_prediction",
    "require_candidate_capacity",
]
