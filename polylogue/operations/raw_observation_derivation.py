"""Product seam for raw recovery through the derivation kernel."""

from __future__ import annotations

from collections.abc import Sequence
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
from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN, RawObservationDerivation, RawObservationScope


def raw_observation_frame(archive_root: Path, *, source_roots: Sequence[Path] = ()) -> DerivationFrame:
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=str(ArchiveLocation.resolve(archive_root).active_index_path.resolve()),
        recipe_versions={RAW_OBSERVATION_DOMAIN: RawObservationDerivation.recipe_version},
        scope=RawObservationScope(source_roots=tuple(source_roots)),
    )


def raw_observation_pending_roots(archive_root: Path, paths: Sequence[Path]) -> set[Path]:
    adapter = RawObservationDerivation(archive_root)
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
    adapter = RawObservationDerivation(archive_root, max_payload_bytes=max_payload_bytes)
    return converge(
        DerivationRegistry((adapter,)),
        raw_observation_frame(archive_root, source_roots=source_roots),
        budget=Budget(page=min(128, limit), discovery=limit, inspection=limit, compute=limit, publication=limit),
        cursor=cursor,
    )
