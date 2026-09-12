"""Product seam for raw recovery through the derivation kernel."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from polylogue.daemon.derivation import (
    Budget,
    DerivationFrame,
    DerivationRegistry,
    DerivationReport,
    PassCursor,
    converge,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN as _RAW_OBSERVATION_DOMAIN
from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationScope

RAW_OBSERVATION_DOMAIN = _RAW_OBSERVATION_DOMAIN


def make_raw_observation_derivation(archive_root: Path, *, max_payload_bytes: int) -> RawObservationDerivation:
    """Construct the storage-owned raw adapter from the operations boundary."""
    return RawObservationDerivation(archive_root, max_payload_bytes=max_payload_bytes)


def raw_observation_output_session_ids(archive_root: Path, raw_id: str) -> tuple[str, ...]:
    """Read every active session output in the seed raw's replay component."""
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    with ArchiveStore.open_existing(archive_root, read_only=True) as source:
        component_raw_ids, _logical_keys = source.expand_raw_membership_selection([raw_id])
    if not component_raw_ids:
        return ()
    index_db = resolve_active_index_path(archive_root)
    with closing(open_readonly_connection(index_db, timeout=5.0)) as conn:
        rows = conn.execute(
            f"SELECT session_id FROM sessions WHERE raw_id IN ({','.join('?' for _ in component_raw_ids)}) "
            "ORDER BY session_id",
            component_raw_ids,
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def raw_observation_frame(
    archive_root: Path,
    *,
    source_roots: Sequence[Path] = (),
    raw_ids: Sequence[str] = (),
) -> DerivationFrame:
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=str(ArchiveLocation.resolve(archive_root).active_index_path.resolve()),
        recipe_versions={RAW_OBSERVATION_DOMAIN: RawObservationDerivation.recipe_version},
        scope=RawObservationScope(source_roots=tuple(source_roots), raw_ids=tuple(raw_ids)),
    )


def raw_observation_pending_roots(archive_root: Path, paths: Sequence[Path]) -> set[Path]:
    adapter = make_raw_observation_derivation(archive_root, max_payload_bytes=64 * 1024 * 1024)
    pending: set[Path] = set()
    ordered = tuple(dict.fromkeys(paths))
    for offset in range(0, len(ordered), 128):
        chunk = ordered[offset : offset + 128]
        frame = raw_observation_frame(archive_root, source_roots=chunk)
        cursor = None
        while True:
            keys, cursor = adapter.required_page(frame, cursor=cursor, limit=128)
            if keys:
                stale = tuple(key for key, status in adapter.inspect(frame, keys).items() if status != "valid")
                for source_path in adapter.source_paths(stale).values():
                    pending.update(
                        path
                        for path in chunk
                        if source_path == str(path).rstrip("/") or source_path.startswith(str(path).rstrip("/") + "/")
                    )
            if cursor is None or all(path in pending for path in chunk):
                break
    return pending


def converge_raw_observations(
    archive_root: Path,
    *,
    source_roots: Sequence[Path],
    limit: int,
    max_payload_bytes: int,
    cursor: PassCursor | None = None,
) -> DerivationReport:
    adapter = make_raw_observation_derivation(archive_root, max_payload_bytes=max_payload_bytes)
    return converge(
        DerivationRegistry((adapter,)),
        raw_observation_frame(archive_root, source_roots=source_roots),
        budget=Budget(page=min(128, limit), discovery=limit, inspection=limit, compute=limit, publication=limit),
        cursor=cursor,
    )
