"""Purge ``agent-*.meta.json`` subagent-sidecar phantom sessions from ``index.db``.

polylogue-ioz7 (direct fix for the polylogue-b508 audit finding): before a
producer fix landed 2026-07-28, a per-subagent metadata sidecar file
(``subagents/agent-<id>.meta.json``) was materialized into the archive as its
own, empty ``claude-code-session`` row -- a duplicate phantom standing beside
the real, correctly-ingested subagent transcript. 4,945 such rows survive in
the live archive as of 2026-08-02 (verified via
``polylogue.storage.agent_meta_sidecar_sweep``). This module is the "act"
half, following the identical safety pattern as
``raw_membership_writeback_apply`` (polylogue-lb39z) and
``binary_artifact_reclassify_apply`` (polylogue-hbtj2):

* Dry-run by default. ``dry_run=False`` requires a verified backup manifest
  that includes ``index.db`` -- the default ``rebuildable_cache_exclude``
  backup profile omits it, so ``polylogue backup --profile full_evidence
  --verify`` (or an equivalent profile that includes ``index``) is required.
* Classification is re-run *live* against the current archive state
  immediately before deletion, never trusting a previously computed plan.
* Deletion goes through ``ArchiveStore.delete_sessions`` -- the same tested,
  incident-hardened bulk-delete primitive (polylogue-meoz) that CLI ``delete``
  and MCP ``write(operation='delete_session')`` already use, instead of a
  plain per-row ``DELETE FROM sessions`` (which detonates per-row derived-
  refresh triggers; see that method's docstring for the 2026-07-21 incident:
  3h runtime, 375GB reads, zero commit, for 91 sessions).
* Every purged row gets an immutable receipt in
  ``agent_meta_sidecar_purge_receipts`` (index.db, v57).
* ``raw_sessions`` rows and blobs in ``source.db`` are never touched, and
  re-ingest cannot resurrect a purged row: the producer bug that created this
  shape is already fixed, so a re-acquire of the same ``.meta.json`` sidecar
  produces no session at all.
* Refuses to apply (even under ``--force-shape-mismatch``'s absence) if any
  matched candidate's ``native_id`` does not also match the expected
  ``agent-<hash>.meta`` shape -- belt-and-suspenders beyond the bead's own
  ``source_path``-only repro predicate, since live verification found the two
  predicates agree on all 4,945 rows and a disagreement would mean the
  candidate set has drifted from what was audited.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.agent_meta_sidecar_sweep import (
    AgentMetaSidecarCandidate,
    AgentMetaSidecarSweepPlan,
    scan_agent_meta_sidecar_sessions,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_backup_manifest_covers_derived_tier

TOOL_VERSION = "agent-meta-sidecar-purge-apply-v1"


class AgentMetaSidecarPurgeApplyError(RuntimeError):
    """Raised when applying the agent-meta-sidecar purge is refused."""


@dataclass(frozen=True, slots=True)
class AgentMetaSidecarPurgeApplyReport:
    scanned_count: int
    purged_count: int
    purged_bytes: int
    purged_session_ids: tuple[str, ...]
    shape_mismatch_count: int
    applied: bool
    backup_manifest: Path | None = None

    @classmethod
    def from_plan(
        cls,
        plan: AgentMetaSidecarSweepPlan,
        *,
        applied: bool,
        purged_session_ids: tuple[str, ...] = (),
        backup_manifest: Path | None = None,
    ) -> AgentMetaSidecarPurgeApplyReport:
        by_id = {candidate.session_id: candidate for candidate in plan.candidates}
        purged_bytes = (
            sum(by_id[sid].blob_size for sid in purged_session_ids if sid in by_id) if applied else plan.candidate_bytes
        )
        return cls(
            scanned_count=plan.scanned_count,
            purged_count=len(purged_session_ids) if applied else plan.candidate_count,
            purged_bytes=purged_bytes,
            purged_session_ids=purged_session_ids,
            shape_mismatch_count=plan.shape_mismatch_count,
            applied=applied,
            backup_manifest=backup_manifest,
        )


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _checkpoint_live_tier(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise AgentMetaSidecarPurgeApplyError("could not checkpoint index.db before backup validation") from exc
    if row is None:
        raise AgentMetaSidecarPurgeApplyError("could not checkpoint index.db before backup validation")


def _write_receipts(
    index_db: Path,
    candidates: tuple[AgentMetaSidecarCandidate, ...],
    *,
    purged_at_ms: int,
    backup_manifest: Path,
) -> None:
    if not candidates:
        return
    conn = sqlite3.connect(index_db)
    try:
        with conn:
            conn.executemany(
                """
                INSERT INTO agent_meta_sidecar_purge_receipts (
                    session_id, origin, native_id, raw_id, source_path,
                    purged_at_ms, tool_version, backup_manifest_path, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '')
                """,
                [
                    (
                        candidate.session_id,
                        candidate.origin,
                        candidate.native_id,
                        candidate.raw_id,
                        candidate.source_path,
                        purged_at_ms,
                        TOOL_VERSION,
                        str(backup_manifest),
                    )
                    for candidate in candidates
                ],
            )
    finally:
        conn.close()


def apply_agent_meta_sidecar_purge(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    limit: int | None = None,
    dry_run: bool = True,
) -> AgentMetaSidecarPurgeApplyReport:
    """Classify agent-meta-sidecar phantom sessions, then purge them.

    ``dry_run=True`` (the default) never opens a write transaction. It runs
    the same classifier a real apply would and reports what it would do.

    ``dry_run=False`` requires ``backup_manifest`` (covering ``index.db`` --
    the default backup profile omits the rebuildable index tier, so a
    ``full_evidence``-profile backup or equivalent is required), refuses if
    any candidate fails the ``native_id`` shape cross-check, re-runs
    classification live, deletes the matched sessions via
    ``ArchiveStore.delete_sessions``, and writes one immutable receipt per
    purged row.
    """
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    if not index_db.exists():
        raise FileNotFoundError(f"no index.db at {index_db}")
    if not source_db.exists():
        raise FileNotFoundError(f"no source.db at {source_db}")

    if dry_run:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            plan = scan_agent_meta_sidecar_sessions(conn, source_db, limit=limit)
        finally:
            conn.close()
        return AgentMetaSidecarPurgeApplyReport.from_plan(plan, applied=False)

    if backup_manifest is None:
        raise AgentMetaSidecarPurgeApplyError(
            "applying the agent-meta-sidecar purge requires a verified backup manifest "
            "(--backup-manifest) that includes index.db, e.g. "
            "'polylogue backup --output-dir <dir> --profile full_evidence --verify'"
        )
    if reason := offline_maintenance_block_reason(_offline_config(archive_root), active=True, dry_run=False):
        raise AgentMetaSidecarPurgeApplyError(reason)

    # uri=True enables URI-filename recognition for the whole connection,
    # including the subsequent ATTACH DATABASE '?mode=ro' the classifier
    # issues -- without it ATTACH would try (and fail) to open the literal
    # string "file:...?mode=ro" as a plain filesystem path.
    validate_conn = sqlite3.connect(str(index_db), uri=True)
    try:
        _checkpoint_live_tier(validate_conn)
        validate_backup_manifest_covers_derived_tier(backup_manifest, ArchiveTier.INDEX, connection=validate_conn)
        plan = scan_agent_meta_sidecar_sessions(validate_conn, source_db, limit=limit)
    finally:
        validate_conn.close()

    if plan.shape_mismatch_count:
        raise AgentMetaSidecarPurgeApplyError(
            f"refusing to apply: {plan.shape_mismatch_count} candidate(s) match the source_path "
            "predicate but not the expected 'agent-<hash>.meta' native_id shape -- inspect the "
            "read-only sweep report before applying"
        )

    session_ids = tuple(candidate.session_id for candidate in plan.candidates)
    purged_at_ms = int(time.time() * 1000)
    purged: tuple[str, ...] = ()
    if session_ids:
        store = ArchiveStore(archive_root, read_only=False)
        try:
            # Authoritative re-validation now that the writer lease is held
            # (ArchiveStore.__init__ acquires it synchronously above) --
            # matches attachment_reacquisition.py's / migrate_archive_tier's
            # pattern: a concurrent write between the first, lock-free
            # precheck and this lease acquisition would make the backup
            # stale, and the lease guarantees nothing else can write between
            # this check and delete_sessions below.
            revalidate_conn = sqlite3.connect(str(index_db), uri=True)
            try:
                _checkpoint_live_tier(revalidate_conn)
                validate_backup_manifest_covers_derived_tier(
                    backup_manifest, ArchiveTier.INDEX, connection=revalidate_conn
                )
            finally:
                revalidate_conn.close()
            store.delete_sessions(session_ids)
        finally:
            store.close()
        # Re-check which session_ids are actually gone rather than trusting
        # delete_sessions's aggregate count -- it returns a total, not a
        # per-id verdict, so this is the only precise way to know exactly
        # which rows a receipt may honestly be written for.
        confirm_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            placeholders = ", ".join("?" for _ in session_ids)
            still_present = {
                str(row[0])
                for row in confirm_conn.execute(
                    f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders})", session_ids
                ).fetchall()
            }
        finally:
            confirm_conn.close()
        purged = tuple(sid for sid in session_ids if sid not in still_present)
        if purged:
            _write_receipts(
                index_db,
                tuple(c for c in plan.candidates if c.session_id in purged),
                purged_at_ms=purged_at_ms,
                backup_manifest=backup_manifest,
            )

    return AgentMetaSidecarPurgeApplyReport.from_plan(
        plan,
        applied=True,
        purged_session_ids=purged,
        backup_manifest=backup_manifest,
    )


__all__ = [
    "TOOL_VERSION",
    "AgentMetaSidecarPurgeApplyError",
    "AgentMetaSidecarPurgeApplyReport",
    "apply_agent_meta_sidecar_purge",
]
