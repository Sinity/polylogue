"""Archive format lineage: the marker a freshly bootstrapped archive publishes and every open checks."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_VERSION_BY_TIER, ARCHIVE_FORMAT_FLOOR_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

ARCHIVE_FORMAT_MARKER_NAME = ".polylogue-format.json"
ARCHIVE_FORMAT_LINEAGE = "polylogue.archive-format.v6"
_DURABLE_FORMAT_TIERS = frozenset({ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT})


def _fsync_directory(path: Path) -> None:
    """Durably publish a replacement marker directory entry."""
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def archive_format_marker_path(archive_root: Path) -> Path:
    """Return the marker that distinguishes the current format lineage."""
    return archive_root / ARCHIVE_FORMAT_MARKER_NAME


def record_fresh_archive_format(archive_root: Path) -> Path:
    """Publish the format identity for a completely bootstrapped fresh archive.

    The marker records the schema shape of the durable tiers rather than a
    path or inode. A copied historical file consequently cannot become a
    member of this lineage merely by sharing ``user_version``.
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
    """Refuse an unmarked or historical archive before bootstrap can write.

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
    birth = _read_archive_format_birth(archive_root)
    for tier in tiers:
        path = archive_root / ARCHIVE_TIER_SPECS[tier].filename
        _assert_safe_format_tier_file(path)
        try:
            with closing(open_readonly_connection(path, validate_schema=False)) as connection:
                _assert_archive_format_tier_lineage(archive_root, tier, connection, birth)
        except sqlite3.Error as exc:
            raise RuntimeError(f"cannot inspect archive format tier: {path}") from exc


@dataclass(frozen=True)
class _ArchiveFormatBirth:
    versions: tuple[tuple[str, int], ...]
    fingerprints: tuple[tuple[str, str], ...]


def _read_archive_format_birth(archive_root: Path) -> _ArchiveFormatBirth:
    """Decode the complete immutable birth marker without opening SQLite."""
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(archive_root)
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
    # each tier was bootstrapped at. The lineage floor is a lower bound, not a
    # permanent pin. A numbered migration can raise the runtime target while
    # the birth record and its schema fingerprints continue to identify the
    # archive's original tier shapes.
    if (
        not isinstance(versions, dict)
        or set(versions) != {tier.value for tier in ArchiveTier}
        or any(
            type(versions.get(tier.value)) is not int or versions[tier.value] < ARCHIVE_FORMAT_FLOOR_VERSION
            for tier in ArchiveTier
        )
    ):
        raise RuntimeError(f"archive format marker has an incomplete six-tier floor: {marker_path}")
    if (
        not isinstance(fingerprints, dict)
        or set(fingerprints) != {tier.value for tier in _DURABLE_FORMAT_TIERS}
        or any(not isinstance(value, str) for value in fingerprints.values())
    ):
        raise RuntimeError(f"archive format marker has incomplete durable schema evidence: {marker_path}")
    return _ArchiveFormatBirth(tuple(sorted(versions.items())), tuple(sorted(fingerprints.items())))


def _assert_safe_format_tier_file(path: Path) -> None:
    """Refuse unsafe leaves before opening and recheck supplied handles."""
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"archive format marker names a missing durable tier: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot inspect archive format tier: {path}") from exc
    if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
        raise RuntimeError(f"archive format marker names an unsafe durable tier file: {path}")


def _assert_archive_format_tier_lineage(
    archive_root: Path,
    tier: ArchiveTier,
    connection: sqlite3.Connection,
    birth: _ArchiveFormatBirth,
) -> None:
    """Check one actual owned tier handle against the same birth authority."""
    path = archive_root / ARCHIVE_TIER_SPECS[tier].filename
    _assert_safe_format_tier_file(path)
    with closing(connection.execute("PRAGMA database_list")) as cursor:
        main_path = next(
            (
                row[2].decode("utf-8") if isinstance(row[2], bytes) else str(row[2])
                for row in cursor
                if (row[1].decode("utf-8") if isinstance(row[1], bytes) else row[1]) == "main"
            ),
            None,
        )
    if main_path is None or Path(main_path).resolve() != path.resolve():
        raise RuntimeError(f"archive format tier connection does not own {path}")
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version < ARCHIVE_FORMAT_FLOOR_VERSION:
        raise RuntimeError(f"{path.name} predates the {ARCHIVE_FORMAT_LINEAGE} floor")
    # The fingerprint identifies the tier shape recorded at birth. Check
    # it whenever a file's version is at or below that birth version: a
    # transplanted older-lineage file can share the same version integer
    # while carrying a different schema. Files advanced by a numbered
    # migration are above their birth version and follow the migration
    # lineage's normal admission rules.
    if version <= dict(birth.versions)[tier.value] and dict(birth.fingerprints)[
        tier.value
    ] != _connection_schema_fingerprint(connection):
        raise RuntimeError(
            f"{path.name} has a historical version-{version} schema and is not part of {ARCHIVE_FORMAT_LINEAGE}"
        )


def _format_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()


def _archive_tier_versions() -> dict[str, int]:
    return {tier.value: ARCHIVE_BASELINE_VERSION_BY_TIER[tier] for tier in ArchiveTier}


def _tier_schema_fingerprint(path: Path) -> str:
    try:
        connection = open_readonly_connection(path, validate_schema=False)
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot inspect archive format tier: {path}") from exc
    try:
        return _connection_schema_fingerprint(connection)
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot read archive format tier schema: {path}") from exc
    finally:
        connection.close()


def _connection_schema_fingerprint(connection: sqlite3.Connection) -> str:
    with closing(
        connection.execute(
            "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name, tbl_name"
        )
    ) as cursor:
        rows = [tuple(value.decode("utf-8") if isinstance(value, bytes) else value for value in row) for row in cursor]
    return hashlib.sha256(
        (json.dumps(rows, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()


__all__ = [
    "ARCHIVE_FORMAT_LINEAGE",
    "ARCHIVE_FORMAT_MARKER_NAME",
    "archive_format_marker_path",
    "assert_archive_format_lineage",
    "record_fresh_archive_format",
]
