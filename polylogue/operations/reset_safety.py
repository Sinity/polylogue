"""Safety predicates shared by reset previews and daemon mutation handlers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from polylogue.core.raw_coordinates import zip_member_container

_SQLITE_FILE_SUFFIXES = ("", "-wal", "-shm", "-journal")


class LiveArchiveTierResetError(ValueError):
    """A daemon reset named archive tier files that the daemon holds open.

    The resident daemon keeps every tier database open for its lifetime: the
    watcher cursor store, the status registry, readers and the write
    coordinator hold their own connections. Unlinking a tier and its WAL/SHM
    sidecars under them leaves those handles on deleted inodes while new
    connections create fresh, empty files, so writes land in a file nobody
    can reach and the reset still reports success. The daemon owns no route
    that closes and reopens every handle, so the refusal is raised before any
    audit row or deletion.
    """

    code = "reset_live_archive_tier"

    def __init__(self, targets: tuple[str, ...]) -> None:
        self.targets = targets
        super().__init__(
            "refusing to delete archive tier files the serving daemon holds open: "
            + ", ".join(targets)
            + "; lifecycle-coordinated database reset is not implemented; no files were deleted"
        )


def live_archive_tier_targets(
    archive_root: Path, targets: Iterable[tuple[str, Path]], *, served_index_path: Path
) -> tuple[str, ...]:
    """Name each target that is, or is a directory holding, a tier database or sidecar.

    ``served_index_path`` is the index generation the daemon's pinned read
    opened, which a pointer-managed archive keeps outside ``index.db``.
    """
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    databases = [archive_root / f"{tier.value}.db" for tier in ArchiveTier]
    databases.append(served_index_path)
    tier_files = {
        database.with_name(f"{database.name}{suffix}").resolve(strict=False)
        for database in databases
        for suffix in _SQLITE_FILE_SUFFIXES
    }
    return tuple(
        name
        for name, path in targets
        if any(tier_file.is_relative_to(path.resolve(strict=False)) for tier_file in tier_files)
    )


def unresolvable_raw_source_count(archive_root: Path) -> int:
    """Count retained raw rows whose original source path cannot be reacquired.

    A ZIP member is reacquirable only when replaying it from the retained
    container still yields the recorded payload: a container that was replaced
    by another valid ZIP without the member, or with different member bytes,
    no longer holds the evidence ``source.db`` is the only copy of.
    """
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return 0
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(source_db, validate_schema=False)
    try:
        rows = conn.execute(
            "SELECT source_path, lower(hex(blob_hash)), COUNT(*) FROM raw_sessions WHERE source_path IS NOT NULL "
            "AND source_path != '' GROUP BY source_path, blob_hash"
        ).fetchall()
    finally:
        conn.close()

    at_risk = 0
    member_rows: dict[str, int] = {}
    for source_path, blob_hash, count in rows:
        text = str(source_path)
        if Path(text).exists():
            continue
        # ZIP rows address members as ``<container>:<member>``. The member
        # suffix is not a filesystem path; the container must hold it.
        if zip_member_container(text) is None:
            at_risk += int(count)
            continue
        member_rows[str(blob_hash)] = member_rows.get(str(blob_hash), 0) + int(count)
    if member_rows:
        from polylogue.operations.archive_backup import _source_recoverability_proofs

        proven = {
            proof["blob_hash"]
            for proof in _source_recoverability_proofs(
                source_db, root=archive_root, missing_hashes=set(member_rows), immutable=False
            )
        }
        at_risk += sum(count for blob_hash, count in member_rows.items() if blob_hash not in proven)
    return at_risk


__all__ = ["LiveArchiveTierResetError", "live_archive_tier_targets", "unresolvable_raw_source_count"]
