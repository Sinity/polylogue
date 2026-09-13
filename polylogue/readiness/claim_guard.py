"""Claim-guard vocabulary: what may honestly be claimed about this archive.

Ports the claim discipline that used to live only in the archived devloop
status script (``.agent/archive/devloop-2026-07/scripts/devloop-status`` —
frozen evidence, never resurrected or executed live; see repo ``CLAUDE.md``)
into the product's own ``polylogue ops status`` surface (polylogue-avg). That
script gated four distinct claims behind four distinct signals:

* per-tier schema-version match => the archive is **openable**, but that is
  *not* the same claim as being converged.
* authoritative domain inspections => **converged**.
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
    value: bool | None
    reason: str
    signal: str

    @property
    def determinate(self) -> bool:
        """Whether inspection actually reached a verdict for this claim.

        ``value is None`` is the third state: not "false", but "not proven
        either way". It exists because collapsing an unfinished inspection
        into ``False`` publishes a claim about the archive that was never
        measured. It mirrors the terminal-outcome rule that a named gap is
        never reported as an ordinary negative result.
        """
        return self.value is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "value": self.value,
            "determinate": self.determinate,
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


@dataclass(frozen=True, slots=True)
class DerivedDomainReadiness:
    """One authoritative derived-domain verdict used to certify convergence.

    This is deliberately a small status projection, rather than a new
    freshness/debt store.  Its producer has already inspected the domain's
    own output relation and inputs; the claim guard only combines those
    verdicts.  Operation attempts and convergence-debt rows remain useful
    health evidence, but cannot certify or withhold this public claim.
    """

    domain: str
    ready: bool
    summary: str
    determinate: bool = True
    """False when the domain's inspection did not finish inside its budget.

    An indeterminate domain cannot certify convergence and cannot refute it
    either, so it withholds the claim instead of negating it.
    """


def derive_claim_guard(
    *,
    archive_schema_ready: bool,
    schema_mismatches: Sequence[str] = (),
    missing_tiers: Sequence[str] = (),
    derived_domains: Sequence[DerivedDomainReadiness],
    search_ready: bool,
    search_summary: str,
    active_writer: bool,
    active_writer_summary: str = "",
) -> ClaimGuard:
    """Derive the claim-guard block from already-computed readiness signals.

    Every argument here is a primitive the caller already derived from a
    canonical readiness surface (``ArchiveStorageStatus.archive_schema_ready``
    and domain-owned raw, frontier, profile, FTS, and enabled-embedding
    inspections) plus the live-ingest/rebuild-attempt signal.  This function
    only classifies, so both daemon and direct status can reuse the same
    already-inspected domain facts without opening another connection.
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
    elif refuted := next((domain for domain in derived_domains if domain.determinate and not domain.ready), None):
        # Proven not converged: a domain inspected its own output relation and
        # found it wanting. This is the only case that may say False.
        converged = ClaimGuardEntry(
            claim="converged",
            value=False,
            reason=refuted.summary,
            signal=f"derived_domain_readiness.{refuted.domain}",
        )
    elif unmeasured := next((domain for domain in derived_domains if not domain.determinate), None):
        # Inspection incomplete: withhold the claim and name the gap rather
        # than publishing a negative verdict nothing measured.
        converged = ClaimGuardEntry(
            claim="converged",
            value=None,
            reason=f"inspection incomplete for {unmeasured.domain}: {unmeasured.summary}",
            signal=f"derived_domain_readiness.{unmeasured.domain}",
        )
    else:
        converged = ClaimGuardEntry(
            claim="converged",
            value=True,
            reason="ready",
            signal="derived_domain_readiness (all required domains inspected ready)",
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
        signal="live_ingest_attempts.running_count",
    )

    return ClaimGuard(openable=openable, converged=converged, search_ready=search, perf_measurable=perf)


__all__ = ["ClaimGuard", "ClaimGuardEntry", "DerivedDomainReadiness", "derive_claim_guard"]
