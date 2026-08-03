"""Persisted read-through cache for :class:`RawAuthorityVerdict` (polylogue-tw4ar).

Follow-up from polylogue-w6hql (PR #3593): :func:`polylogue.storage.
raw_authority_verdict_projection.project_raw_authority_verdicts` recomputes a
cohort's verdicts on demand by re-running
``classify_historical_full_revision_streams`` against live blob storage every
call -- correct, but not cheap at scale. This module adds a persisted cache
table (``raw_authority_verdicts``, migration 022) in front of that projection.

The cache is **not a new source of truth**. Every row is invalidated by
*content*, not by elapsed time: :func:`_cohort_fingerprint` hashes the
cohort's own ``(raw_id, revision_kind, blob_hash)`` rows, so any membership or
content change to a ``logical_source_key`` cohort (a new revision acquired, a
row's ``revision_kind`` resolved from ``unknown`` to ``full``) changes the
fingerprint and the previously-cached rows read back as stale on the very
next lookup, rather than being trusted for some duration. This makes the
cache impossible to observe returning a verdict that disagrees with what
``project_raw_authority_verdicts`` would compute right now for the same
cohort shape.

This module only fills the cache lazily on read
(:func:`get_or_compute_raw_authority_verdicts`). Wiring a ``DaemonConverger``
stage to keep it warm proactively ahead of reads is deliberately deferred --
see polylogue-tw4ar's own notes for why that needs its own careful staging
rather than landing bundled with the cache table itself.
"""

from __future__ import annotations

import hashlib

from polylogue.core.enums import RawAuthorityVerdict
from polylogue.storage.raw_authority_verdict_projection import (
    RawAuthorityVerdictProjectionHost,
    project_raw_authority_verdicts,
)


def _current_cohort_rows(
    store: RawAuthorityVerdictProjectionHost, logical_source_key: str
) -> list[tuple[str, str, str]]:
    """Return every ``(raw_id, revision_kind, blob_hash_hex)`` row for a cohort."""
    conn = store._ensure_source_conn()
    rows = conn.execute(
        """
        SELECT raw_id, revision_kind, lower(hex(blob_hash))
        FROM raw_sessions
        WHERE logical_source_key = ?
        """,
        (logical_source_key,),
    ).fetchall()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def _cohort_fingerprint(rows: list[tuple[str, str, str]]) -> bytes:
    """SHA-256 over the sorted ``(raw_id, revision_kind, blob_hash)`` rows defining a cohort.

    Sorting first makes the fingerprint independent of SQL row order. Each
    field is length-delimited by a trailing NUL/SOH-style separator byte
    (rather than plain concatenation) so no ambiguous split exists between
    e.g. ``raw_id="ab"`` + ``revision_kind="c"`` and ``raw_id="a"`` +
    ``revision_kind="bc"``.
    """
    hasher = hashlib.sha256()
    for raw_id, revision_kind, blob_hash in sorted(rows):
        hasher.update(raw_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(revision_kind.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(blob_hash.encode("utf-8"))
        hasher.update(b"\x01")
    return hasher.digest()


def read_cached_raw_authority_verdicts(
    store: RawAuthorityVerdictProjectionHost, logical_source_key: str
) -> dict[str, RawAuthorityVerdict] | None:
    """Return cached verdicts for a cohort if the cache is fresh, else ``None``.

    ``None`` covers both "never cached" and "cached but stale" -- callers
    that get ``None`` back should recompute via
    :func:`~polylogue.storage.raw_authority_verdict_projection.project_raw_authority_verdicts`
    (or just call :func:`get_or_compute_raw_authority_verdicts`, which does
    exactly that and persists the result).
    """
    current_rows = _current_cohort_rows(store, logical_source_key)
    if not current_rows:
        return None
    fingerprint = _cohort_fingerprint(current_rows)

    conn = store._ensure_source_conn()
    cached_rows = conn.execute(
        "SELECT raw_id, verdict, cohort_fingerprint FROM raw_authority_verdicts WHERE logical_source_key = ?",
        (logical_source_key,),
    ).fetchall()
    if not cached_rows or len(cached_rows) != len(current_rows):
        return None

    verdicts: dict[str, RawAuthorityVerdict] = {}
    for raw_id, verdict, cached_fingerprint in cached_rows:
        if bytes(cached_fingerprint) != fingerprint:
            return None
        verdicts[str(raw_id)] = RawAuthorityVerdict(str(verdict))
    return verdicts


def write_raw_authority_verdict_cache(
    store: RawAuthorityVerdictProjectionHost,
    logical_source_key: str,
    verdicts: dict[str, RawAuthorityVerdict],
    *,
    now_ms: int,
) -> None:
    """Persist ``verdicts`` -- the *complete* result for one cohort -- to the cache.

    Callers must pass the full verdict dict
    :func:`~polylogue.storage.raw_authority_verdict_projection.project_raw_authority_verdicts`
    returned for this ``logical_source_key``, never a partial subset: a
    partial write would leave a stale row behind that silently fails the next
    read's ``len(cached_rows) != len(current_rows)`` check for the wrong
    reason (row-count mismatch masking a real content change). Replaces the
    cohort's entire prior cache row set atomically in one transaction.
    """
    fingerprint = _cohort_fingerprint(_current_cohort_rows(store, logical_source_key))
    member_count = len(verdicts)

    conn = store._ensure_source_conn()
    with conn:
        conn.execute(
            "DELETE FROM raw_authority_verdicts WHERE logical_source_key = ?",
            (logical_source_key,),
        )
        conn.executemany(
            """
            INSERT INTO raw_authority_verdicts
                (raw_id, logical_source_key, verdict, cohort_member_count, cohort_fingerprint, computed_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (raw_id, logical_source_key, str(verdict), member_count, fingerprint, now_ms)
                for raw_id, verdict in verdicts.items()
            ],
        )


def get_or_compute_raw_authority_verdicts(
    store: RawAuthorityVerdictProjectionHost,
    logical_source_key: str,
    *,
    now_ms: int,
) -> dict[str, RawAuthorityVerdict]:
    """Read-through cache: return a fresh cached result, else recompute and persist.

    This is the entry point real consumers (blob-GC invariant checks,
    operator surfaces) should call instead of
    ``project_raw_authority_verdicts`` directly, so a warm cache is used when
    available and a cold/stale one is filled as a side effect of the read.
    """
    cached = read_cached_raw_authority_verdicts(store, logical_source_key)
    if cached is not None:
        return cached
    verdicts = project_raw_authority_verdicts(store, logical_source_key)
    if verdicts:
        write_raw_authority_verdict_cache(store, logical_source_key, verdicts, now_ms=now_ms)
    return verdicts


__all__ = [
    "get_or_compute_raw_authority_verdicts",
    "read_cached_raw_authority_verdicts",
    "write_raw_authority_verdict_cache",
]
