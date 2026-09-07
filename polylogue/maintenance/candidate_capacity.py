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
#: an archive as soon as one recorded receipt carries an observed peak.
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
CAPACITY_RECEIPT_SCHEMA: Final = "polylogue.candidate-capacity.v1"

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
        super().__init__(
            "insufficient free space for a candidate index build: "
            f"required {projection.required_free_bytes} bytes, "
            f"available {projection.available_bytes} bytes, "
            f"short by {projection.shortfall_bytes} bytes "
            f"(projected index {projection.projected_index_bytes} bytes from "
            f"{projection.evidence_bytes} bytes of durable evidence, "
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
            "total_allocated_bytes": self.total_allocated_bytes,
            "total_logical_bytes": self.total_logical_bytes,
        }


@dataclass(frozen=True, slots=True)
class CapacityReceipt:
    """One build's predicted and observed peak, the input to later calibration."""

    operation_id: str
    evidence_bytes: int
    calibration_ratio: float
    calibration_source: str
    predicted_index_bytes: int
    predicted_peak_allocated_bytes: int
    required_free_bytes: int
    available_bytes_at_prediction: int
    baseline_allocated_bytes: int
    actual_peak_index_bytes: int = 0
    observations: int = 0

    @property
    def actual_peak_allocated_bytes(self) -> int:
        if self.actual_peak_index_bytes <= 0:
            return 0
        return self.baseline_allocated_bytes + self.actual_peak_index_bytes

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
            "baseline_allocated_bytes": self.baseline_allocated_bytes,
            "actual_peak_index_bytes": self.actual_peak_index_bytes,
            "actual_peak_allocated_bytes": self.actual_peak_allocated_bytes,
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
            baseline_allocated_bytes=_int_field(payload, "baseline_allocated_bytes"),
            actual_peak_index_bytes=_int_field(payload, "actual_peak_index_bytes"),
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

    @property
    def required_free_bytes(self) -> int:
        return (
            self.projected_index_bytes
            + self.wal_amplification_bytes
            + self.temporary_amplification_bytes
            + self.receipt_growth_bytes
            + self.reserve_bytes
        )

    @property
    def predicted_peak_allocated_bytes(self) -> int:
        return (
            self.retained_allocated_bytes
            + self.projected_index_bytes
            + self.wal_amplification_bytes
            + self.temporary_amplification_bytes
            + self.receipt_growth_bytes
        )

    @property
    def available_bytes(self) -> int:
        return self.inventory.available_bytes

    @property
    def sufficient(self) -> bool:
        return self.available_bytes >= self.required_free_bytes

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_free_bytes - self.available_bytes)

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
    pending = [root]
    while pending:
        directory = pending.pop()
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
                pending.append(Path(entry.path))


def _measure_path(accumulator: _Accumulator, path: Path, *, label: str, follow_root_link: bool = False) -> None:
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
        except (OSError, RuntimeError):
            return
        _measure_path(accumulator, target, label=label)
        return
    accumulator.add(metadata)
    if stat.S_ISDIR(metadata.st_mode):
        for child in _iter_tree(path, label=label):
            accumulator.add(child)


def _measure_database(accumulator: _Accumulator, path: Path, *, label: str) -> None:
    """Charge one SQLite tier and its WAL/shm/journal sidecars."""
    _measure_path(accumulator, path, label=label, follow_root_link=True)
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        _measure_path(accumulator, path.with_name(path.name + suffix), label=f"{label} sidecar")


def _generations_root(archive_root: Path) -> Path:
    """The generations root, refusing the one link that would hide the index."""
    root = archive_root / GENERATIONS_DIRNAME
    metadata = _lstat(root, label="index generations root")
    if metadata is not None and stat.S_ISLNK(metadata.st_mode):
        raise ArchiveCapacityError(f"index generations root is a symlink: {root}")
    return root


def _available_bytes(path: Path) -> int:
    try:
        statistics = os.statvfs(path)
    except OSError as exc:
        raise ArchiveCapacityError(f"cannot read free space for archive root: {path}") from exc
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
    generations_root = _generations_root(root)
    try:
        location = ArchiveLocation.resolve(root)
    except ArchiveLocationError as exc:
        raise ArchiveCapacityError(f"cannot resolve archive identity: {root}") from exc

    seen: set[tuple[int, int]] = set()
    accumulators = {name: _Accumulator(name, seen) for name in POPULATION_NAMES}

    generations = _measure_generations(accumulators["index_generations"], generations_root, location)

    # The root's own ``index.db`` is whatever is literally there -- a stub, a
    # promotion symlink, or a pre-generation index. The pointer target is the
    # index, and is charged whether or not it sits under the generations root,
    # so a stub can never stand in for it.
    _measure_database(accumulators["index_root"], root / "index.db", label="root index")
    _measure_database(accumulators["index_generations"], location.active_index_path, label="active index")
    _measure_path(
        accumulators["rebuild_transactions"], root / REBUILD_TRANSACTIONS_DIRNAME, label="rebuild transactions"
    )
    for filename in _DURABLE_TIER_FILENAMES:
        _measure_database(accumulators["durable_tiers"], root / filename, label=f"durable tier {filename}")
    for filename in _DERIVED_TIER_FILENAMES:
        _measure_database(accumulators["derived_tiers"], root / filename, label=f"derived tier {filename}")
    _measure_path(accumulators["blob"], root / _BLOB_DIRNAME, label="blob store", follow_root_link=True)
    for dirname in _SPOOL_DIRNAMES:
        _measure_path(accumulators["spools"], root / dirname, label=f"spool {dirname}", follow_root_link=True)
    _measure_path(
        accumulators["maintenance_receipts"],
        root / MAINTENANCE_STATE_DIRNAME,
        label="maintenance receipts",
    )

    unclassified = accumulators["unclassified"]
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

    return ArchiveCapacityInventory(
        archive_root=root,
        active_index_path=location.active_index_path,
        active_generation_id=_active_generation_id(location),
        populations=tuple(accumulators[name].measurement() for name in POPULATION_NAMES),
        generations=generations,
        available_bytes=_available_bytes(root),
    )


def _active_generation_id(location: ArchiveLocation) -> str | None:
    parts = location.active_index_path.resolve(strict=False).parts
    try:
        depth = parts.index(GENERATIONS_DIRNAME)
    except ValueError:
        return None
    return parts[depth + 1] if len(parts) > depth + 1 else None


def _measure_generations(
    accumulator: _Accumulator, generations_root: Path, location: ArchiveLocation
) -> tuple[GenerationMeasurement, ...]:
    """Measure each generation directory separately, then charge the whole root."""
    active_id = _active_generation_id(location)
    measurements: list[GenerationMeasurement] = []
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
        # generation's size is a property of that generation, not of the order
        # the inventory happened to walk its siblings in.
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


def project_candidate_capacity(archive_root: Path) -> CandidateCapacityProjection:
    """Project one candidate index build against the measured archive tree."""
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
    projected_index_bytes = max(observed_index_bytes, math.ceil(evidence_bytes * ratio))
    wal_amplification_bytes = math.ceil(projected_index_bytes * WAL_AMPLIFICATION_FRACTION)
    temporary_amplification_bytes = math.ceil(projected_index_bytes * TEMPORARY_AMPLIFICATION_FRACTION)
    receipt_growth_bytes = max(
        RECEIPT_GROWTH_FLOOR_BYTES,
        inventory.population("maintenance_receipts").allocated_bytes
        + inventory.population("rebuild_transactions").allocated_bytes,
    )
    construction_bytes = projected_index_bytes + wal_amplification_bytes + temporary_amplification_bytes
    reserve_bytes = max(RESERVE_FLOOR_BYTES, math.ceil(construction_bytes * RESERVE_FRACTION))
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
    )


def require_candidate_capacity(archive_root: Path, *, operation_id: str) -> CandidateCapacityProjection:
    """Refuse a candidate build whose projected peak does not fit, before allocating."""
    projection = project_candidate_capacity(archive_root)
    if not projection.sufficient:
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
    archive_root: Path, *, operation_id: str, projection: CandidateCapacityProjection
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
        baseline_allocated_bytes=projection.retained_allocated_bytes,
    )
    _ensure_maintenance_state(Path(archive_root))
    with maintenance_receipt_directory(Path(archive_root), CAPACITY_RECEIPT_DIRNAME) as directory_fd:
        atomic_replace_receipt(directory_fd, filename, _encode(receipt.as_dict()))
    return receipt


def record_capacity_observation(
    archive_root: Path, *, operation_id: str, candidate_root: Path
) -> CapacityReceipt | None:
    """Raise the recorded peak to what the candidate has actually allocated.

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
        if recorded is None:
            return None
        observed = replace(
            recorded,
            actual_peak_index_bytes=max(recorded.actual_peak_index_bytes, isolated.allocated),
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
        receipt.actual_peak_index_bytes / receipt.evidence_bytes
        for receipt in receipts
        if receipt.actual_peak_index_bytes > 0 and receipt.evidence_bytes > 0
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
