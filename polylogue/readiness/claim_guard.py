"""Claim-guard vocabulary: what may honestly be claimed about this archive.

Ports the claim discipline that used to live only in the archived devloop
status script (``.agent/archive/devloop-2026-07/scripts/devloop-status`` —
frozen evidence, never resurrected or executed live; see repo ``CLAUDE.md``)
into the product's own ``polylogue ops status`` surface (polylogue-avg). That
script gated four distinct claims behind four distinct signals:

* per-tier schema-version match => the archive is **openable**, but that is
  *not* the same claim as being converged.
* zero (or fully classified) raw-materialization debt => **converged**.
* FTS freshness => **search-ready**.
* absence of concurrent heavy archive activity => **perf-measurable**. The
  script's ``live_performance_proof_blocked`` flag grepped the host process
  table for unrelated tools (``borg create``, ``lynchpin.analysis
  materialize``). The product surface generalizes this to polylogue's own
  concurrent-write signal — a live ingest attempt or an index-rebuild attempt
  in flight — since hardcoding unrelated host-tool process names into the
  public product would be a layering violation and wouldn't generalize past
  one operator's machine.

``derive_claim_guard`` is a pure function over already-computed readiness
primitives so both the daemon-serving path (``daemon/status.py``) and the
no-daemon direct SQLite fallback path (``cli/commands/status.py``) share one
derivation and cannot silently drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClaimGuardEntry:
    """One claim state: may you honestly say X about this archive right now?"""

    claim: str
    value: bool
    reason: str
    signal: str

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "value": self.value,
            "reason": self.reason,
            "signal": self.signal,
        }


@dataclass(frozen=True, slots=True)
class ClaimGuard:
    """The four claim-guard states surfaced by ``polylogue ops status``."""

    openable: ClaimGuardEntry
    converged: ClaimGuardEntry
    search_ready: ClaimGuardEntry
    perf_measurable: ClaimGuardEntry

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            "openable": self.openable.to_dict(),
            "converged": self.converged.to_dict(),
            "search_ready": self.search_ready.to_dict(),
            "perf_measurable": self.perf_measurable.to_dict(),
        }


def derive_claim_guard(
    *,
    archive_schema_ready: bool,
    schema_mismatches: Sequence[str] = (),
    missing_tiers: Sequence[str] = (),
    raw_materialization_ready: bool,
    raw_materialization_summary: str,
    raw_frontier_integrity_ready: bool,
    raw_frontier_integrity_summary: str,
    search_ready: bool,
    search_summary: str,
    active_writer: bool,
    active_writer_summary: str = "",
) -> ClaimGuard:
    """Derive the claim-guard block from already-computed readiness signals.

    Every argument here is a primitive the caller already derived from a
    canonical readiness surface (``ArchiveStorageStatus.archive_schema_ready``,
    :func:`polylogue.storage.archive_readiness.raw_materialization_ready`, the
    raw-frontier-integrity projection
    (:func:`polylogue.storage.raw_retention.raw_frontier_integrity_snapshot`,
    polylogue-yla8.7 — accepted append head chains and ingest cursors proven
    consistent), the ``search``/FTS component, and the
    live-ingest/rebuild-attempt signal) — this function only classifies, it
    never queries storage itself, so it is cheap to call from both the
    daemon and direct-fallback status paths.
    """
    if archive_schema_ready:
        openable_reason = "all archive tiers present with matching schema version"
    elif missing_tiers:
        openable_reason = f"missing archive tier(s): {', '.join(sorted(missing_tiers))}"
    elif schema_mismatches:
        openable_reason = f"schema version mismatch on tier(s): {', '.join(sorted(schema_mismatches))}"
    else:
        openable_reason = "archive tiers not verified"

    openable = ClaimGuardEntry(
        claim="openable",
        value=archive_schema_ready,
        reason=openable_reason,
        signal="archive_storage.archive_schema_ready (per-tier PRAGMA user_version match)",
    )

    if not archive_schema_ready:
        converged = ClaimGuardEntry(
            claim="converged",
            value=False,
            reason=f"not openable: {openable_reason}",
            signal="archive_storage.archive_schema_ready and raw_materialization_readiness",
        )
    elif not raw_materialization_ready:
        converged = ClaimGuardEntry(
            claim="converged",
            value=False,
            reason=raw_materialization_summary,
            signal="raw_materialization_readiness (debt zero or fully classified)",
        )
    elif not raw_frontier_integrity_ready:
        # polylogue-yla8.7: raw materialization can look fully converged while
        # an accepted append head references a deleted predecessor or an
        # ingest cursor sits ahead of accepted material — yla8.6 found this
        # only through manual SQL. Converged must not be claimable while that
        # authority gap is open or unproven.
        converged = ClaimGuardEntry(
            claim="converged",
            value=False,
            reason=raw_frontier_integrity_summary,
            signal="raw_frontier_integrity (accepted append head chains and ingest cursors proven consistent)",
        )
    else:
        converged = ClaimGuardEntry(
            claim="converged",
            value=True,
            reason=raw_materialization_summary,
            signal="raw_materialization_readiness (debt zero or fully classified)",
        )

    search = ClaimGuardEntry(
        claim="search_ready",
        value=search_ready,
        reason=search_summary,
        signal="component_readiness.search (FTS freshness)",
    )

    if active_writer:
        perf_reason = active_writer_summary or "an archive write/rebuild is in flight"
    else:
        perf_reason = "no concurrent archive write/rebuild detected"

    perf = ClaimGuardEntry(
        claim="perf_measurable",
        value=not active_writer,
        reason=perf_reason,
        signal="live_ingest_attempts.running_count / active_rebuild_index_attempts",
    )

    return ClaimGuard(openable=openable, converged=converged, search_ready=search, perf_measurable=perf)


__all__ = ["ClaimGuard", "ClaimGuardEntry", "derive_claim_guard"]
