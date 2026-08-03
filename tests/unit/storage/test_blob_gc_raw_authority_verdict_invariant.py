"""Blob-GC invariant verification against Phase 2's verdict vocabulary (polylogue-ds4b4 item 4).

``run_blob_gc`` protects a blob purely by *row reference*: any ``raw_sessions``
row whose ``blob_hash`` matches a candidate keeps that blob alive, regardless
of what that row's revision authority says about it (#storage/blob_gc.py
Safety invariant #1). ``RawAuthorityVerdict`` (polylogue-w6hql) is a much
richer classification -- VERIFIED / SOLE_COPY / SUPERSEDED / DIVERGED /
UNCHECKED -- describing whether a raw's *content* is authoritative, current,
or provably a duplicate/fork of another raw in its cohort.

These two concepts must never contradict each other: a raw carrying a
"non-authoritative" verdict (SUPERSEDED, DIVERGED, UNCHECKED) still has a live
``raw_sessions`` row (lineage/audit requires the historical bytes to survive),
so blob GC's row-reference invariant must continue to protect its blob. Blob
GC deliberately does not consult ``RawAuthorityVerdict`` at all -- this suite
proves that omission is safe: for every verdict value the vocabulary can
produce, the raw's blob survives a live GC pass exactly because a
``raw_sessions`` row for it still exists, independent of what verdict that
row would project to.

Anti-vacuity: every scenario binds real cohorts through the production writer
(``ArchiveStore.write_raw_payload`` / ``bind_raw_revision``), projects real
verdicts via ``project_raw_authority_verdicts`` (the same Phase 2 machinery
``tests/unit/storage/test_raw_authority_verdict_projection.py`` exercises),
and then runs the real ``run_blob_gc_report`` against the resulting
``source.db`` + on-disk blob store -- not a mock reference check.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import (
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.core.enums import Provider, RawAuthorityVerdict
from polylogue.storage.blob_gc import run_blob_gc_report
from polylogue.storage.raw_authority_verdict_projection import project_raw_authority_verdicts
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _bind_full(archive: ArchiveStore, *, raw_id: str, payload: bytes, logical_source_key: str) -> str:
    written_id = archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=payload,
        source_path="session.jsonl",
        acquired_at_ms=1,
        raw_id=raw_id,
    )
    archive.bind_raw_revision(
        written_id,
        RawRevisionEnvelope(logical_source_key, RawRevisionKind.FULL, f"revision-{raw_id}", 0),
    )
    return written_id


def _backdate_all_blobs(blob_root: Path, *, seconds: float = 3600) -> None:
    """Backdate every blob file's mtime past ``MIN_AGE_S`` so GC considers it."""
    past = time.time() - seconds
    if not blob_root.is_dir():
        return
    for prefix_dir in blob_root.iterdir():
        if not prefix_dir.is_dir():
            continue
        for entry in prefix_dir.iterdir():
            if entry.is_file():
                os.utime(entry, (past, past))


@pytest.mark.uses_real_clock(
    "backdates real blob mtimes via os.utime; blob_gc.py's age gate compares them against a "
    "real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_blob_gc_protects_every_verdict_value_while_the_raw_row_survives(tmp_path: Path) -> None:
    """Seed one raw per ``RawAuthorityVerdict`` value, then prove GC deletes none of them."""
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        # VERIFIED head + SUPERSEDED ancestor: a proven byte-contiguous chain.
        _bind_full(archive, raw_id="chain-old", payload=b"one\n", logical_source_key="codex:chain")
        _bind_full(archive, raw_id="chain-new", payload=b"one\ntwo\n", logical_source_key="codex:chain")

        # DIVERGED: an unprovable same-cohort fork (no byte-prefix relation).
        _bind_full(archive, raw_id="fork-left", payload=b"left", logical_source_key="codex:fork")
        _bind_full(archive, raw_id="fork-right", payload=b"right", logical_source_key="codex:fork")

        # SOLE_COPY: the only raw in its cohort.
        _bind_full(archive, raw_id="alone", payload=b"solo payload", logical_source_key="codex:solo")

        # UNCHECKED: revision_kind not yet resolved.
        unresolved_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"not yet classified",
            source_path="session.jsonl",
            acquired_at_ms=1,
            raw_id="pending",
        )
        archive.bind_raw_revision(
            unresolved_id,
            RawRevisionEnvelope("codex:pending", RawRevisionKind.UNKNOWN, "revision-pending", 0),
        )

        verdicts = {
            **project_raw_authority_verdicts(archive, "codex:chain"),
            **project_raw_authority_verdicts(archive, "codex:fork"),
            **project_raw_authority_verdicts(archive, "codex:solo"),
            **project_raw_authority_verdicts(archive, "codex:pending"),
        }

        source_db_path = archive.source_db_path
        blob_root = archive.archive_root / "blob"

    # Anti-vacuity: the seeded cohorts really do cover every verdict value --
    # if a future change to the classifier collapses one of these shapes, this
    # assertion (not a silent no-op) is what catches it.
    assert set(verdicts.values()) == set(RawAuthorityVerdict)
    assert verdicts == {
        "chain-old": RawAuthorityVerdict.SUPERSEDED,
        "chain-new": RawAuthorityVerdict.VERIFIED,
        "fork-left": RawAuthorityVerdict.DIVERGED,
        "fork-right": RawAuthorityVerdict.DIVERGED,
        "alone": RawAuthorityVerdict.SOLE_COPY,
        "pending": RawAuthorityVerdict.UNCHECKED,
    }

    _backdate_all_blobs(blob_root)
    report = run_blob_gc_report(source_db_path, blob_root, max_batch=100)

    assert report.candidate_count == 6, "every raw's blob must be a GC candidate (old enough)"
    assert report.deleted_count == 0, (
        "blob GC must not delete a blob for any verdict value while its raw_sessions row "
        f"survives; verdict mix under test = {sorted(v.value for v in verdicts.values())}"
    )
    assert report.skipped_referenced == 6


@pytest.mark.uses_real_clock(
    "backdates real blob mtimes via os.utime; blob_gc.py's age gate compares them against a "
    "real time.time() call in production code, so frozen_clock cannot intercept either side"
)
def test_blob_gc_reclaims_only_after_the_verdict_owning_row_is_actually_gone(tmp_path: Path) -> None:
    """A SUPERSEDED raw's blob is reclaimable once (and only once) its own row is deleted.

    Verdict projection never deletes rows -- it is read-only by construction.
    This proves the *other* half of the invariant: the vocabulary saying
    "not authoritative" is not itself a deletion trigger, and GC's row-based
    check is the only thing standing between a raw and its blob being
    reclaimed once something else (governance retirement, not this suite's
    concern) removes the row.
    """
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="chain-old", payload=b"one\n", logical_source_key="codex:chain")
        _bind_full(archive, raw_id="chain-new", payload=b"one\ntwo\n", logical_source_key="codex:chain")

        verdicts = project_raw_authority_verdicts(archive, "codex:chain")
        assert verdicts["chain-old"] is RawAuthorityVerdict.SUPERSEDED

        source_db_path = archive.source_db_path
        blob_root = archive.archive_root / "blob"

        # Simulate a hypothetical future retirement path removing every
        # GC-visible reference to the superseded raw (no such production path
        # exists today -- this is exercising GC's reaction, not asserting
        # retirement exists). ``write_raw_payload`` records the blob in BOTH
        # ``raw_sessions`` and ``blob_refs`` (ref_type='raw_payload'); GC's own
        # ``_archive_reference_surfaces`` checks both tables, so a real
        # retirement path would need to clear both -- deleting only
        # ``raw_sessions`` leaves the blob protected by its still-live
        # ``blob_refs`` row, which is itself part of the invariant this test
        # is pinning down.
        conn = archive._ensure_source_conn()
        conn.execute("DELETE FROM raw_sessions WHERE raw_id = 'chain-old'")
        conn.execute("DELETE FROM blob_refs WHERE ref_id = 'chain-old'")
        conn.commit()

    _backdate_all_blobs(blob_root)
    report = run_blob_gc_report(source_db_path, blob_root, max_batch=100)

    assert report.deleted_count == 1, "with its row gone, the SUPERSEDED raw's blob is reclaimable"
    assert report.skipped_referenced == 1, "the VERIFIED head's row still protects its own blob"
