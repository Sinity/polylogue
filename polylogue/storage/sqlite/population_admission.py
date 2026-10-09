"""Admission fence for an explicitly owned, unfinished archive population."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, unquote_to_bytes, urlsplit

if TYPE_CHECKING:
    from polylogue.storage.archive_identity import OwnedArchiveLocation

POPULATION_PENDING = ".archive-population.pending"


class ArchivePopulationPendingError(RuntimeError):
    """An explicit population has not published its complete proof."""

    code = "archive_population_pending"


@dataclass(frozen=True)
class _PopulationAdmission:
    root: Path
    root_identity: tuple[int, int]
    marker_identity: tuple[int, int]
    owner: OwnedArchiveLocation
    durable_versions: tuple[tuple[str, int], ...] | None = None


_CURRENT_POPULATION: ContextVar[_PopulationAdmission | None] = ContextVar("archive_population", default=None)


def assert_population_admitted(path: str | bytes | Path) -> None:
    """Refuse every ordinary connection beneath a pending destination."""
    text = os.fsdecode(path)
    if text == ":memory:":
        return
    if text.startswith("file:"):
        parsed = urlsplit(text)
        if parsed.path == ":memory:" or parse_qs(parsed.query).get("mode") == ["memory"]:
            return
        text = os.fsdecode(unquote_to_bytes(parsed.path))
    candidate = Path(text).resolve(strict=False)
    for root in (candidate, *candidate.parents):
        marker = root / POPULATION_PENDING
        if not marker.exists() and not marker.is_symlink():
            continue
        admission = _CURRENT_POPULATION.get()
        if admission is not None and root == admission.root and admission.owner.holds_ownership:
            from polylogue.storage.sqlite.write_lease import UnleasedWriteError, require_write_lease

            root_stat, marker_stat = root.stat(), marker.lstat()
            if (
                (root_stat.st_dev, root_stat.st_ino) == admission.root_identity
                and (marker_stat.st_dev, marker_stat.st_ino) == admission.marker_identity
                and not marker.is_symlink()
            ):
                held = os.fstat(admission.owner.directory_fd)
                if (held.st_dev, held.st_ino) != admission.root_identity:
                    raise ArchivePopulationPendingError(str(root))
                try:
                    lease = require_write_lease("archive population", archive_root=root)
                except UnleasedWriteError as exc:
                    raise ArchivePopulationPendingError(str(root)) from exc
                if lease is not None:
                    return
        raise ArchivePopulationPendingError(str(root))


def require_population_admission(root: Path) -> _PopulationAdmission:
    """Return only the live, exact population capability in this owner context."""
    root = root.resolve(strict=True)
    admission = _CURRENT_POPULATION.get()
    if admission is None or admission.root != root or not (root / POPULATION_PENDING).is_file():
        raise ArchivePopulationPendingError(str(root))
    assert_population_admitted(root)
    return admission


@contextmanager
def _bound_population_stage(root: Path, versions: Mapping[str, int]) -> Iterator[None]:
    """Bind authenticated installed durable targets to the exact pending owner."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, DURABLE_MIGRATION_TIERS
    from polylogue.storage.sqlite.migration_runner import durable_migration_claims

    admission = require_population_admission(root)
    if set(versions) != {tier.value for tier in DURABLE_MIGRATION_TIERS}:
        raise ArchivePopulationPendingError("unsupported population durable target map")
    for tier in DURABLE_MIGRATION_TIERS:
        target = versions[tier.value]
        spec = ARCHIVE_TIER_SPECS[tier]
        if type(target) is not int or not spec.baseline_version <= target <= spec.version:
            raise ArchivePopulationPendingError("unsupported population durable target")
        declared = {claim.target_version for claim in durable_migration_claims(tier)}
        if any(step not in declared for step in range(spec.baseline_version + 1, target + 1)):
            raise ArchivePopulationPendingError("undeclared population durable target")
    token = _CURRENT_POPULATION.set(replace(admission, durable_versions=tuple(sorted(versions.items()))))
    try:
        yield
    finally:
        _CURRENT_POPULATION.reset(token)


@contextmanager
def owned_population_admission(root: Path, owner: OwnedArchiveLocation) -> Iterator[None]:
    """Scope internal admission to the held owner and exact reserved inode."""
    from polylogue.storage.archive_identity import ArchiveLocation, assert_owns_archive_location
    from polylogue.storage.sqlite.write_lease import require_write_lease

    root = root.resolve(strict=True)
    assert_owns_archive_location(owner, ArchiveLocation.resolve(root))
    if require_write_lease("archive population", archive_root=root) is None:
        raise ArchivePopulationPendingError(str(root))
    marker = root / POPULATION_PENDING
    root_stat, marker_stat = root.stat(), marker.lstat()
    if root.is_symlink() or marker.is_symlink():
        raise ArchivePopulationPendingError(str(root))
    token = _CURRENT_POPULATION.set(
        _PopulationAdmission(
            root,
            (root_stat.st_dev, root_stat.st_ino),
            (marker_stat.st_dev, marker_stat.st_ino),
            owner,
        )
    )
    try:
        yield
    finally:
        _CURRENT_POPULATION.reset(token)


@contextmanager
def reserve_population_destination(root: Path, *, source_manifest_id: str) -> Iterator[_PopulationAdmission]:
    """Exclusively reserve a new product destination through complete publication."""
    root = root.absolute()
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ArchivePopulationDestinationExistsError(str(root)) from exc
    reserved = root.stat()
    with _admit_population_destination(
        root, source_manifest_id=source_manifest_id, root_identity=(reserved.st_dev, reserved.st_ino)
    ):
        yield require_population_admission(root)


class ArchivePopulationDestinationExistsError(ValueError):
    """An explicit restore cannot adopt an existing destination."""


@contextmanager
def _admit_population_destination(
    root: Path, *, source_manifest_id: str, root_identity: tuple[int, int]
) -> Iterator[None]:
    from polylogue.maintenance.offline_guard import scoped_offline_archive_writer
    from polylogue.storage.sqlite.archive_tiers.archive_plan import _fsync_directory
    from polylogue.storage.sqlite.write_lease import write_lease

    root = root.resolve(strict=True)
    identity = root.stat()
    if (identity.st_dev, identity.st_ino) != root_identity:
        raise ArchivePopulationPendingError(str(root))
    marker = root / POPULATION_PENDING
    with marker.open("x", encoding="utf-8") as output:
        json.dump(
            {
                "format": "polylogue.archive-population.pending/v1",
                "source_manifest_id": source_manifest_id,
                "destination_identity": [identity.st_dev, identity.st_ino],
            },
            output,
            sort_keys=True,
        )
        output.flush()
        os.fsync(output.fileno())
    _fsync_directory(root)
    _fsync_directory(root.parent)
    # Failure deliberately retains the exact pending root and all partial
    # evidence. No generic cleanup can prove replacement/unrelated files safe.
    with scoped_offline_archive_writer(root, owner_id="archive.population") as owner:
        held = os.fstat(owner.directory_fd)
        if (held.st_dev, held.st_ino) != (identity.st_dev, identity.st_ino):
            raise ArchivePopulationPendingError(str(root))
        with write_lease("archive.population", archive_root=root), owned_population_admission(root, owner):
            yield
            admission = require_population_admission(root)
            if admission.root_identity != (identity.st_dev, identity.st_ino):
                raise ArchivePopulationPendingError(str(root))
            pending_evidence = marker.read_bytes()
            try:
                marker.unlink()
                _fsync_directory(root)
            except BaseException:
                from polylogue.core.durable_fs import atomic_replace

                # A failed retirement cannot leave an admitted-looking root.
                # Reinstall the same fence and preserve the completed evidence.
                atomic_replace(marker, pending_evidence)
                raise
