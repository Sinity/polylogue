"""Archive-root-addressed daemon stage-event recording.

Daemon maintenance loops report their per-tick outcome into the disposable
``ops.db`` tier. Routing that through operations keeps the loops off a direct
storage import while the write itself stays in one place.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["record_daemon_stage_event_for_archive"]


def record_daemon_stage_event_for_archive(
    archive_root: Path,
    *,
    stage: str,
    status: str,
    observed_at_ms: int,
    payload: dict[str, object] | None = None,
) -> str:
    """Record one daemon stage event into the archive's ``ops.db``.

    Creates the archive root and the OPS tier when absent, so a probe that
    runs before the archive is fully materialized still leaves evidence.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import open_daemon_connection

    archive_root.mkdir(parents=True, exist_ok=True)
    with open_daemon_connection(archive_root / "ops.db", timeout=30.0) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        return record_daemon_stage_event(
            conn,
            stage=stage,
            status=status,
            observed_at_ms=observed_at_ms,
            payload=payload,
        )
