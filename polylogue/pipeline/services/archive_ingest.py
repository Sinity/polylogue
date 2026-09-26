"""Compatibility entry point for one-shot source ingestion.

Acquisition, cursor commits, replay, and convergence are owned by the live
batch processor. This module keeps the established Python API name while
callers migrate; it contains no independent archive publisher.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.config import Source
from polylogue.maintenance import offline_guard
from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError
from polylogue.operations.canonical_archive_ingest import (
    ingest_sources_archive,
    scoped_one_shot_archive_owner,
)
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.storage.archive_identity import resolve_active_index_path

_ONE_SHOT_MARKER = ".one-shot-ingest-owner"


def _admit_one_shot_root(root: Path) -> None:
    """Claim an empty root once; subsequent calls may only reuse that claim."""
    pid = offline_guard.resident_daemon_pid(root)
    if pid is not None:
        raise ArchiveWriterOwnershipError(
            f"polylogued PID {pid} owns {root}; submit ingestion to that daemon",
            archive_root=root,
            resident_writer=f"polylogued PID {pid}",
        )
    marker = root / _ONE_SHOT_MARKER
    root_stat = root.stat()
    claim = f"canonical-one-shot-v1 {root_stat.st_dev}:{root_stat.st_ino}\n"
    if marker.exists():
        try:
            if marker.read_text(encoding="utf-8") != claim:
                raise ArchiveWriterOwnershipError(
                    f"{root} has an invalid one-shot owner claim; refusing an offline write",
                    archive_root=root,
                )
        except OSError as exc:
            raise ArchiveWriterOwnershipError(
                f"cannot verify the one-shot owner claim for {root}", archive_root=root
            ) from exc
        return
    prior_tiers = tuple(
        name
        for name in (
            "source.db",
            "index.db",
            "user.db",
            "embeddings.db",
            "audit.db",
            "ops.db",
            ".index-active-pointer",
        )
        if (root / name).exists()
    )
    for db_path, table in (
        (root / "source.db", "raw_sessions"),
        (root / "user.db", "assertions"),
        (resolve_active_index_path(root), "sessions"),
    ):
        if not db_path.exists():
            continue
        try:
            with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
                present = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if present and conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                    raise ArchiveWriterOwnershipError(
                        f"{root} already contains archive content; submit ingestion to the resident daemon",
                        archive_root=root,
                    )
        except sqlite3.Error as exc:
            raise ArchiveWriterOwnershipError(
                f"cannot prove {root} is an empty one-shot archive",
                archive_root=root,
            ) from exc
    if prior_tiers:
        raise ArchiveWriterOwnershipError(
            f"{root} is an existing archive ({', '.join(prior_tiers)}); "
            "one-shot ingestion requires a newly isolated root",
            archive_root=root,
        )
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(claim)
        handle.flush()
        os.fsync(handle.fileno())


async def parse_sources_archive(
    archive_root: Path,
    sources: list[Source],
    *,
    parse_workers: int | None = None,
) -> ParseResult:
    """Offer local sources to the canonical acquisition and convergence owner."""
    if not any(source.path is not None for source in sources):
        return ParseResult()
    root = archive_root.expanduser().resolve()
    with scoped_one_shot_archive_owner(root):
        _admit_one_shot_root(root)
        return await ingest_sources_archive(root, sources, parse_workers=parse_workers)


__all__ = ["parse_sources_archive"]
