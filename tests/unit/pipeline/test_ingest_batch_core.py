"""Production-route laws for ``ingest_batch._core``'s Drive revision cohort.

polylogue-ojjet: the Drive revision-cohort classifier is called
unconditionally for every Gemini raw that reaches ``_write_session``, and
each call re-derives every cohort member's content hash from its bytes and
then streams pairwise prefix comparisons between the survivors. Both phases
reach the blob store through ``ArchiveBlobPublisher.open``, so the cohort's
bytes were re-loaded from disk once per hash and once per candidate pair,
per raw. These tests pin the bound that replaced that growth and prove the
bound changed no lineage.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.sources.dispatch import parse_payload
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection import open_connection

_COHORT_SIZE = 12
_LINEAGE_COLUMNS = (
    "blob_size",
    "logical_source_key",
    "revision_kind",
    "revision_authority",
    "predecessor_raw_id IS NOT NULL",
    "baseline_raw_id IS NOT NULL",
    "acquisition_generation",
)


def _drive_revision_payload(revision: int, *, filler: str = "") -> dict[str, JSONValue]:
    """One Gemini AI Studio document, grown by one turn per revision.

    A later revision is a strict structural superset of its predecessor, the
    shape the byte-prefix classifier exists to chain, so every raw below
    lands in one real cohort rather than being trivially deduplicated.
    """
    chunks: list[JSONValue] = [{"role": "user", "text": f"turn {turn}{filler}"} for turn in range(revision + 1)]
    return {"chunkedPrompt": {"chunks": chunks}}


def _ingest_drive_cohort(
    archive_root: Path,
    *,
    cohort_cache: ingest_batch_core.DriveRevisionCohortCache | None,
    revisions: int = _COHORT_SIZE,
    filler: str = "",
) -> tuple[int, list[tuple[object, ...]]]:
    """Write ``revisions`` Drive raws of one logical session through ``_write_session``.

    Returns the number of real ``ArchiveBlobPublisher.open`` calls made
    during the write pass and the resulting durable lineage columns.
    """
    initialize_active_archive_root(archive_root)
    blob_publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")

    raw_ids: list[str] = []
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for revision in range(revisions):
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.GEMINI,
                    payload=json.dumps(_drive_revision_payload(revision, filler=filler)).encode("utf-8"),
                    source_path="Google AI Studio/chat.json",
                    acquired_at_ms=1_767_000_000_000 + revision,
                )
            )

    opens: collections.Counter[str] = collections.Counter()
    unpatched_open = ArchiveBlobPublisher.open

    def counting_open(self: ArchiveBlobPublisher, hash_hex: str):  # type: ignore[no-untyped-def]
        opens[hash_hex] += 1
        return unpatched_open(self, hash_hex)

    ArchiveBlobPublisher.open = counting_open  # type: ignore[method-assign]
    try:
        with (
            open_connection(archive_root / "index.db") as conn,
            sqlite3.connect(str(archive_root / "source.db")) as source_conn,
        ):
            for revision, raw_id in enumerate(raw_ids):
                session = parse_payload("gemini", _drive_revision_payload(revision, filler=filler), "fallback-id")[0]
                ingest_batch_core._write_session(
                    conn,
                    SessionWritePayload(
                        session_id=str(make_session_id(session.source_name, session.provider_session_id)),
                        content_hash=session_content_hash(session),
                        parsed_session=session,
                        message_count=len(session.messages),
                        attachment_count=len(session.attachments),
                        raw_id=raw_id,
                    ),
                    blob_publisher=blob_publisher,
                    source_conn=source_conn,
                    drive_cohort_cache=cohort_cache,
                )
                conn.commit()
    finally:
        ArchiveBlobPublisher.open = unpatched_open  # type: ignore[method-assign]

    with sqlite3.connect(str(archive_root / "source.db")) as verify_conn:
        lineage = verify_conn.execute(
            f"SELECT {', '.join(_LINEAGE_COLUMNS)} FROM raw_sessions ORDER BY blob_size, raw_id"
        ).fetchall()
    assert len(lineage) == revisions
    return sum(opens.values()), [tuple(row) for row in lineage]


def test_drive_cohort_blob_loads_do_not_grow_with_the_cohort(tmp_path: Path) -> None:
    """polylogue-ojjet AC1: one disk load per distinct cohort blob, not per comparison.

    Concrete input: twelve Gemini raws of one logical session, each a
    structural growth of the last, driven through the production
    ``_write_session`` route.

    Wrong observable outcome prevented: the cohort's bytes being re-read from
    disk once per content hash and once per prefix comparison, per raw --
    measured at 646 reads of 12 distinct blobs before this bound (and 4,896
    reads of 24 blobs at twice the cohort size).

    Anti-vacuity: dropping ``drive_cohort_cache`` from ``_write_session``'s
    call to ``_bind_drive_revision_lineage``, or making
    ``DriveRevisionCohortCache.read`` always return ``None``, restores the
    growth and makes the ``disk_loads`` assertion red -- the uncached arm
    measured in the same test is the live witness that the growth is real
    and is not an assertion about a hypothetical.
    """
    cache = ingest_batch_core.DriveRevisionCohortCache()
    cached_opens, _ = _ingest_drive_cohort(tmp_path / "cached", cohort_cache=cache)
    uncached_opens, _ = _ingest_drive_cohort(tmp_path / "uncached", cohort_cache=None)

    # The bound: one read per distinct blob in the cohort, whatever the
    # number of hash derivations and prefix comparisons above it.
    assert cache.disk_loads == _COHORT_SIZE
    assert cached_opens == 0
    # The growth being bounded, measured on this exact corpus rather than
    # asserted from the design.
    assert uncached_opens > _COHORT_SIZE * _COHORT_SIZE
    assert cache.served_from_cache == uncached_opens - _COHORT_SIZE


def test_drive_cohort_blob_cache_leaves_lineage_bit_identical(tmp_path: Path) -> None:
    """polylogue-ojjet AC3/AC4: the bound is a memo, not a semantic change.

    Concrete input: the same twelve-raw cohort written twice, once with the
    cache and once without.

    Wrong observable outcome prevented: a performance bound that changes
    which raw binds to which lineage. Every durable revision column --
    ``logical_source_key``, ``revision_kind``, ``revision_authority``,
    whether a predecessor and a baseline were bound, and
    ``acquisition_generation`` -- must match row for row. The classifier is
    still reached by every raw: nothing here caps a cohort or moves the call
    behind ``_write_session``'s skip check.

    Anti-vacuity: serving any blob's bytes from the wrong cache entry, or
    caching bytes that are not the blob's own, makes the chain decisions
    diverge and this comparison red.
    """
    cached_opens, cached_lineage = _ingest_drive_cohort(
        tmp_path / "cached", cohort_cache=ingest_batch_core.DriveRevisionCohortCache()
    )
    uncached_opens, uncached_lineage = _ingest_drive_cohort(tmp_path / "uncached", cohort_cache=None)

    assert cached_opens != uncached_opens, "the two arms must differ in work, or the comparison is vacuous"
    assert cached_lineage == uncached_lineage
    # The cohort really was governed, not skipped: every raw carries a real
    # logical_source_key and a 'full' revision kind.
    logical_keys = {row[1] for row in cached_lineage}
    assert logical_keys == {"aistudio-drive:fallback-id"}
    assert {row[2] for row in cached_lineage} == {"full"}


def test_drive_cohort_blob_cache_refuses_a_blob_over_its_budget(tmp_path: Path) -> None:
    """polylogue-ojjet: the memo's byte budget is real, so RSS stays bounded.

    Concrete input: a two-raw cohort whose blobs are larger than the cache's
    configured budget.

    Wrong observable outcome prevented: an unbounded pass-scoped cache that
    pins a whole cohort of large Drive documents in the writer's resident
    set. An over-budget blob is read straight from disk and never retained.

    Anti-vacuity: removing the ``size > self._remaining`` refusal in
    ``DriveRevisionCohortCache.read`` makes ``disk_loads`` non-zero and the
    ``cached_opens > 0`` assertion red, because every read would then be
    served from memory.
    """
    cache = ingest_batch_core.DriveRevisionCohortCache(max_bytes=16)
    cached_opens, lineage = _ingest_drive_cohort(tmp_path / "over-budget", cohort_cache=cache, revisions=2)

    assert cache.disk_loads == 0
    assert cache.served_from_cache == 0
    assert cached_opens > 0
    _, uncached_lineage = _ingest_drive_cohort(tmp_path / "uncached", cohort_cache=None, revisions=2)
    assert lineage == uncached_lineage


@pytest.mark.parametrize("hash_hex", ["0" * 64])
def test_drive_cohort_blob_cache_falls_through_for_an_absent_blob(tmp_path: Path, hash_hex: str) -> None:
    """A hash with no file on disk is never cached and never raises here.

    Wrong observable outcome prevented: the cache turning a missing blob into
    an empty-bytes cache entry, which would silently make an absent payload
    look like a zero-length one to the byte-prefix classifier.

    Anti-vacuity: replacing the ``OSError`` guard with a ``b""`` default
    makes the ``is None`` assertion red.
    """
    initialize_active_archive_root(tmp_path / "archive")
    publisher = ArchiveBlobPublisher(tmp_path / "archive" / "source.db", tmp_path / "archive" / "blob")
    cache = ingest_batch_core.DriveRevisionCohortCache()
    assert cache.read(publisher, hash_hex) is None
    assert cache.disk_loads == 0
