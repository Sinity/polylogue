"""Read-only planning for archive initialization."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_FORMAT_FLOOR_VERSION, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

ARCHIVE_FORMAT_MARKER_NAME = ".polylogue-format.json"
ARCHIVE_FORMAT_LINEAGE = "polylogue.archive-format.v1"
_DURABLE_FORMAT_TIERS = frozenset({ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT})


def _fsync_directory(path: Path) -> None:
    """Durably publish a replacement marker directory entry."""
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class ArchiveInitAction(StrEnum):
    """Operator action needed for one archive tier file."""

    CREATE = "create"
    REPLACE_WITH_BACKUP = "replace_with_backup"
    RECREATE_DISPOSABLE = "recreate_disposable"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ArchiveTierPlan:
    """Planned handling for one durability-tier database file."""

    tier: ArchiveTier
    path: Path
    durability: str
    exists: bool
    user_version: int | None
    expected_user_version: int
    backup_required: bool
    action: ArchiveInitAction
    backup_path: Path | None
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchiveInitPlan:
    """Complete read-only initialization plan for an archive root."""

    archive_root: Path
    tiers: tuple[ArchiveTierPlan, ...]
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers


def archive_format_marker_path(archive_root: Path) -> Path:
    """Return the marker that distinguishes this reset v1 from historical v1 files."""
    return archive_root / ARCHIVE_FORMAT_MARKER_NAME


def record_fresh_archive_format(archive_root: Path) -> Path:
    """Publish the format identity for a completely bootstrapped fresh archive.

    The marker records the schema shape of the durable floor rather than a
    path or inode.  A copied historical v1 source/user/audit file consequently
    cannot become a member of this lineage merely by sharing ``user_version``.
    """
    marker_path = archive_format_marker_path(archive_root)
    if marker_path.exists():
        raise RuntimeError(f"archive format marker already exists: {marker_path}")
    missing = [
        archive_root / ARCHIVE_TIER_SPECS[tier].filename
        for tier in ArchiveTier
        if not (archive_root / ARCHIVE_TIER_SPECS[tier].filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"cannot record a six-tier archive format floor; missing: {', '.join(map(str, missing))}")
    versions = _archive_tier_versions()
    durable_fingerprints = {
        tier.value: _tier_schema_fingerprint(archive_root / ARCHIVE_TIER_SPECS[tier].filename)
        for tier in _DURABLE_FORMAT_TIERS
    }
    payload: dict[str, object] = {
        "format": ARCHIVE_FORMAT_LINEAGE,
        "floor_version": ARCHIVE_FORMAT_FLOOR_VERSION,
        "tier_versions": versions,
        "durable_schema_fingerprints": durable_fingerprints,
    }
    payload["digest"] = _format_digest(payload)
    # Publish the marker as one complete file.  A torn JSON marker would make
    # the next startup refuse the archive, so never write directly to the
    # authority path.
    archive_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{marker_path.name}.", dir=archive_root)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, marker_path)
        _fsync_directory(marker_path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return marker_path


def assert_archive_format_lineage(
    archive_root: Path,
    *,
    tiers: frozenset[ArchiveTier] = _DURABLE_FORMAT_TIERS,
) -> None:
    """Refuse an unmarked or historical v1 archive before bootstrap can write.

    This uses read-only SQLite catalog access only.  In particular it neither
    opens nor initializes optional derived services while reporting a format
    mismatch.

    ``tiers`` narrows only the per-tier *file* proof; every marker payload
    check stays full strength, including the requirement that the marker
    record all six tier versions and all three durable fingerprints.  A caller
    narrows it when one durable tier is known to be absent and the archive has
    its own typed recovery route for that absence: proving the surviving tiers
    still belong to this lineage is what makes that route safe to name, and
    reporting ``names a missing durable tier`` instead would strand the
    operator on a message that describes no action.
    """
    if not tiers <= _DURABLE_FORMAT_TIERS:
        raise RuntimeError(f"archive format lineage has no proof for tiers: {sorted(tiers - _DURABLE_FORMAT_TIERS)}")
    marker_path = archive_format_marker_path(archive_root)
    try:
        raw = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"archive format marker is missing: {marker_path}; historical archive lineages require an explicit upgrade"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"archive format marker is unreadable: {marker_path}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"archive format marker is invalid: {marker_path}")
    payload = dict(raw)
    digest = payload.pop("digest", None)
    if (
        payload.get("format") != ARCHIVE_FORMAT_LINEAGE
        or payload.get("floor_version") != ARCHIVE_FORMAT_FLOOR_VERSION
        or not isinstance(digest, str)
        or digest != _format_digest(payload)
    ):
        raise RuntimeError(f"archive format marker does not identify {ARCHIVE_FORMAT_LINEAGE}: {marker_path}")
    versions = payload.get("tier_versions")
    fingerprints = payload.get("durable_schema_fingerprints")
    # ``tier_versions`` is this archive's birth record: the durable version
    # each tier was bootstrapped at. The lineage floor is the lower bound of
    # that record, not a permanent pin -- a numbered migration raises the
    # runtime's durable target, and every archive bootstrapped afterwards is
    # born above the floor. Requiring exact equality here would make a fresh
    # archive refuse its own freshly written marker the moment the first
    # durable migration lands, which is the evolution regime this floor was
    # introduced to restart, not to forbid.
    if (
        not isinstance(versions, dict)
        or set(versions) != {tier.value for tier in ArchiveTier}
        or any(
            not isinstance(versions.get(tier.value), int) or versions[tier.value] < ARCHIVE_FORMAT_FLOOR_VERSION
            for tier in _DURABLE_FORMAT_TIERS
        )
    ):
        raise RuntimeError(f"archive format marker has an incomplete six-tier floor: {marker_path}")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {tier.value for tier in _DURABLE_FORMAT_TIERS}:
        raise RuntimeError(f"archive format marker has incomplete durable schema evidence: {marker_path}")
    for tier in tiers:
        path = archive_root / ARCHIVE_TIER_SPECS[tier].filename
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"archive format marker names a missing durable tier: {path}") from exc
        except OSError as exc:
            raise RuntimeError(f"cannot inspect archive format tier: {path}") from exc
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
            raise RuntimeError(f"archive format marker names an unsafe durable tier file: {path}")
        version = _read_user_version(path)
        if version is None:
            raise RuntimeError(f"archive format marker names a missing durable tier: {path}")
        if version < ARCHIVE_FORMAT_FLOOR_VERSION:
            raise RuntimeError(f"{path.name} predates the {ARCHIVE_FORMAT_LINEAGE} floor")
        # The recorded fingerprint describes this tier at its birth version, so
        # it is evidence exactly while the file still sits there. Binding the
        # check to the birth version rather than to the floor keeps that
        # discrimination alive once archives are born above the floor: without
        # it, a lineage whose durable target has moved would admit any file
        # carrying the right integer, which is the historical-version-1
        # confusion this marker exists to prevent.
        # At or *below* the recorded birth version, not only at it. A durable
        # tier only ever moves up, so a file below its birth version is either
        # this lineage's own one numbered slot behind -- which still carries
        # the recorded schema and is owed the migration route -- or a
        # transplanted older-lineage file. Testing equality alone skipped the
        # fingerprint on the version mismatch, so once a tier is born above
        # the floor (the source tier is, at slot 002) a historical version-1
        # file cleared the floor and was accepted as lineage evidence.
        if version <= versions[tier.value] and fingerprints[tier.value] != _tier_schema_fingerprint(path):
            raise RuntimeError(
                f"{path.name} has a historical version-{version} schema and is not part of {ARCHIVE_FORMAT_LINEAGE}"
            )


def _format_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()


def _archive_tier_versions() -> dict[str, int]:
    return {tier.value: ARCHIVE_VERSION_BY_TIER[tier] for tier in ArchiveTier}


def _tier_schema_fingerprint(path: Path) -> str:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot inspect archive format tier: {path}") from exc
    try:
        rows = connection.execute(
            """
            SELECT type, name, tbl_name, COALESCE(sql, '')
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name, tbl_name
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot read archive format tier schema: {path}") from exc
    finally:
        connection.close()
    return hashlib.sha256(
        (json.dumps(rows, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()


def build_archive_init_plan(
    *,
    archive_root: Path,
    replace_existing: bool = False,
) -> ArchiveInitPlan:
    """Inspect the current archive root and return the initialization plan.

    The planner is deliberately read-only. It does not create backups,
    initialize archive files, or delete disposable state; it only determines
    whether those operations are safe to start and records the exact file
    targets the mutating initialization command must use.
    """

    resolved_archive_root = archive_root.expanduser()

    blockers: list[str] = []

    tier_plans: list[ArchiveTierPlan] = []
    for tier in ARCHIVE_TIER_SPECS:
        tier_plan = _plan_tier(
            archive_root=resolved_archive_root,
            tier=tier,
            replace_existing=replace_existing,
        )
        tier_plans.append(tier_plan)
        blockers.extend(tier_plan.blockers)

    return ArchiveInitPlan(
        archive_root=resolved_archive_root,
        tiers=tuple(tier_plans),
        blockers=tuple(blockers),
    )


def _plan_tier(
    *,
    archive_root: Path,
    tier: ArchiveTier,
    replace_existing: bool,
) -> ArchiveTierPlan:
    spec = ARCHIVE_TIER_SPECS[tier]
    path = archive_root / spec.filename
    exists = path.exists()
    user_version = _read_user_version(path) if exists and path.is_file() else None
    backup_path = path.with_name(f"{path.name}.pre-archive-init.bak") if exists and spec.backup_required else None
    blockers: list[str] = []

    if exists and tier in _DURABLE_FORMAT_TIERS:
        blockers.append(
            f"{tier.value} target already exists; a fresh format floor cannot replace durable evidence: {path}"
        )
        action = ArchiveInitAction.BLOCKED
    elif exists and not replace_existing:
        blockers.append(f"{tier.value} target already exists; rerun with replace_existing after backing it up: {path}")
        action = ArchiveInitAction.BLOCKED
    elif exists and spec.backup_required:
        action = ArchiveInitAction.REPLACE_WITH_BACKUP
    elif exists:
        action = ArchiveInitAction.RECREATE_DISPOSABLE
    else:
        action = ArchiveInitAction.CREATE

    return ArchiveTierPlan(
        tier=tier,
        path=path,
        durability=spec.durability,
        exists=exists,
        user_version=user_version,
        expected_user_version=spec.version,
        backup_required=spec.backup_required,
        action=action,
        backup_path=backup_path,
        blockers=tuple(blockers),
    )


def _read_user_version(path: Path) -> int | None:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()


__all__ = [
    "ArchiveInitAction",
    "ArchiveInitPlan",
    "ArchiveTierPlan",
    "ARCHIVE_FORMAT_LINEAGE",
    "ARCHIVE_FORMAT_MARKER_NAME",
    "archive_format_marker_path",
    "assert_archive_format_lineage",
    "build_archive_init_plan",
    "record_fresh_archive_format",
]
