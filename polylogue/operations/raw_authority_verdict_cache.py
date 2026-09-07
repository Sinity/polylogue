"""Daemon-facing adapter for raw-authority verdict cache convergence."""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.raw_authority_verdict_cache import (
    RawAuthorityVerdictCacheWarmup,
    RawAuthorityVerdictCacheWork,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def find_raw_authority_verdict_cache_work(archive_root: Path) -> RawAuthorityVerdictCacheWork | None:
    """Inspect source-tier cache readiness without opening writable archive state."""
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return None
    conn = open_readonly_connection(source_db, timeout_class="background-read")
    try:
        from polylogue.storage.introspection import table_exists
        from polylogue.storage.raw_authority_verdict_cache import find_raw_authority_verdict_cache_work as find_work

        if not table_exists(conn, "raw_sessions") or not table_exists(conn, "raw_authority_verdicts"):
            return None
        return find_work(conn, max_cohorts=1)
    finally:
        conn.close()


def warm_raw_authority_verdict_cache(
    archive_root: Path, *, max_cohorts: int, now_ms: int
) -> RawAuthorityVerdictCacheWarmup:
    """Warm source-tier verdicts through the source-only archive writer."""
    from polylogue.storage.raw_authority_verdict_cache import warm_raw_authority_verdict_cache as warm_cache
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore.open_source_tier_acquisition(archive_root) as archive:
        return warm_cache(archive, max_cohorts=max_cohorts, now_ms=now_ms)


__all__ = [
    "RawAuthorityVerdictCacheWarmup",
    "RawAuthorityVerdictCacheWork",
    "find_raw_authority_verdict_cache_work",
    "warm_raw_authority_verdict_cache",
]
