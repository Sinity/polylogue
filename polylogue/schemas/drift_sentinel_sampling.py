"""Best-effort ops.db bridge for the format-drift sentinel (polylogue-da1).

Ingest classifies per-record schema drift in-process
(``polylogue.pipeline.services.ingest_worker``) and hands the resulting
:class:`~polylogue.schemas.drift_sentinel.SchemaDriftObservation` list back
to the caller on the same ``_IngestBatchSummary`` used for every other
ingest counter. This module persists that list into ``ops.db`` -- a
sibling database file next to the index.db the caller already has open --
using the same best-effort contract as
:func:`polylogue.storage.fts.drift_sampling.sample_fts_drift_to_ops_sync`:
a missing or unreachable ``ops.db`` (synthetic fixture, archive
mid-bootstrap) is swallowed, never raised. Recording drift telemetry must
never turn an ingest commit into an ops.db write dependency -- the whole
point of the sentinel is that it augments, never gates, ingest.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger

if TYPE_CHECKING:
    from polylogue.schemas.drift_sentinel import SchemaDriftObservation

logger = get_logger(__name__)


def record_schema_drift_observations_to_ops_sync(
    index_db_path: Path,
    observations: list[SchemaDriftObservation],
    *,
    archive_root: Path | None = None,
) -> int:
    """Append one ops.db sample per observation. Returns the count written.

    ``index_db_path`` is the just-committed index.db connection's file
    path; ``ops.db`` is resolved as its sibling by default. ``index_db_path``
    can be ``Config.db_path``, which per polylogue-yla8.1 names the resolved
    active generation rather than the plain archive root once an
    ``.index-active-pointer`` is active -- callers that already have the
    plain archive root in scope (e.g. the ingest writer, which threads both
    ``db_path`` and ``archive_root_str`` through the same commit path)
    should pass it explicitly via ``archive_root`` instead of relying on
    the sibling derivation.
    """
    if not observations:
        return 0

    ops_db_path = (archive_root / "ops.db") if archive_root is not None else index_db_path.with_name("ops.db")
    if not ops_db_path.exists():
        return 0

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_schema_drift_sample
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import open_connection

    ops_conn: sqlite3.Connection | None = None
    try:
        ops_conn = open_connection(ops_db_path, timeout=5.0)
        initialize_archive_tier(ops_conn, ArchiveTier.OPS)
    except sqlite3.Error:
        logger.debug("schema drift sampling: ops.db open/init failed", exc_info=True)
        if ops_conn is not None:
            with contextlib.suppress(sqlite3.Error):
                ops_conn.close()
        return 0

    observed_at_ms = int(time.time() * 1000)
    written = 0
    try:
        for observation in observations:
            try:
                record_schema_drift_sample(
                    ops_conn,
                    origin=observation.origin,
                    element_kind=observation.element_kind,
                    classification=observation.classification,
                    unseen_key_signature=observation.unseen_key_signature,
                    native_id_example=observation.native_id_example,
                    raw_id=observation.raw_id,
                    observed_at_ms=observed_at_ms,
                )
            except sqlite3.IntegrityError:
                # polylogue-sd9s: a CHECK/constraint rejection on ONE
                # observation (e.g. a stale live ops.db CHECK predating a new
                # DriftClassification member) must not discard the rest of
                # this batch -- each call is already its own commit
                # (record_schema_drift_sample's `with conn:`), so prior rows
                # in this loop are safely persisted; only the offending
                # observation is lost. Logged above debug because a CHECK
                # violation here means the schema and the writer have
                # drifted apart, which is itself the class of defect this
                # sentinel exists to surface -- swallowing it silently would
                # hide the sentinel's own malfunction.
                logger.warning(
                    "schema drift sampling: rejected one observation (origin=%s, classification=%s) "
                    "by ops.db CHECK constraint -- schema_drift_samples DDL may be stale",
                    observation.origin,
                    observation.classification,
                    exc_info=True,
                )
                continue
            written += 1
        return written
    except sqlite3.Error:
        # A non-constraint sqlite error (locked db, disk full, ...) is a
        # genuine best-effort-tier failure, not a per-row data problem --
        # stop attempting further writes for this batch as before.
        logger.debug("schema drift sampling: ops.db write failed", exc_info=True)
        return written
    finally:
        with contextlib.suppress(sqlite3.Error):
            ops_conn.close()


__all__ = ["record_schema_drift_observations_to_ops_sync"]
