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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


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


def raw_materialization_unmeasured_reason(readiness: Mapping[str, Any] | object | None) -> str | None:
    """Name the precondition a raw-materialization snapshot never measured.

    ``storage.archive_readiness.raw_materialization_ready`` returns False both
    when an inspection ran and found debt and when the inspection never ran at
    all. Only the first is a refutation; the second is an unmeasured state, and
    collapsing it into ``converged: false`` publishes a claim about the archive
    that nothing checked (polylogue-kjy0a). Status producers call this to mark
    the domain indeterminate instead of negating it.

    This lives beside the claim guard rather than in ``archive_readiness``
    because that module is inside the derived-schema identity closure: the
    predicate it mirrors is unchanged, and classifying its output is a
    claim-publication concern, not a storage one.

    It no longer asks whether a frontier census row exists. That row came from
    the retired per-pass census ledger (polylogue-6kur ruling 2026-09-15), and
    nothing on the daemon route ever wrote one, so the condition marked every
    production archive permanently indeterminate. An unresolved frontier
    obligation is a durable blocker row and refutes
    ``raw_materialization_ready`` directly instead.
    """
    payload: Mapping[str, Any] | None
    if readiness is None:
        payload = None
    elif isinstance(readiness, Mapping):
        payload = readiness
    else:
        model_dump = getattr(readiness, "model_dump", None)
        dumped = model_dump() if callable(model_dump) else None
        payload = dumped if isinstance(dumped, Mapping) else None
    if payload is None or not bool(payload.get("available", False)):
        return "raw-materialization readiness was not inspected"
    if payload.get("debt_classifier_error"):
        return "raw debt classifier did not run"
    parser_census = payload.get("raw_authority_parser_census")
    if not isinstance(parser_census, Mapping) or parser_census.get("available") is not True:
        return "source parser census not measured"
    return None


def search_unmeasured_reason(indexable_count: int | None) -> str | None:
    """Name the gap when FTS coverage has a zero denominator.

    ``messages_ready`` is the invariant "every indexable row is indexed". At
    zero indexable rows that is vacuously true, so an archive with nothing in
    it published ``search_ready: true`` beside ``message_indexable_count: 0``
    (polylogue-o6oct). ``daemon/fts_status`` already treats the same zero
    denominator as unmeasured (``coverage_pct = None``); this is the claim-level
    counterpart, shared by both status producers so they cannot disagree.

    Only an explicit zero is classified here: a producer that did not report a
    count keeps its own verdict rather than having a real refutation (a missing
    FTS surface beside populated blocks) masked into "unknown".
    """
    if indexable_count == 0:
        return "no indexable rows: fts coverage is undefined at a zero denominator, not complete"
    return None


def derive_claim_guard(
    *,
    archive_schema_ready: bool,
    schema_mismatches: Sequence[str] = (),
    missing_tiers: Sequence[str] = (),
    derived_domains: Sequence[DerivedDomainReadiness],
    search_ready: bool,
    search_summary: str,
    search_unmeasured: str | None = None,
    active_writer: bool,
    active_writer_summary: str = "",
    active_writer_determinate: bool = True,
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
        value=None if search_unmeasured is not None else search_ready,
        reason=search_unmeasured if search_unmeasured is not None else search_summary,
        signal="component_readiness.search (FTS freshness)",
    )

    if not active_writer_determinate:
        # The writer evidence itself could not be read. "I could not look" is
        # not "nothing is writing", and it is not a refutation either
        # (polylogue-g88v4): withhold the claim and name the gap.
        perf_reason = active_writer_summary or "concurrent-writer evidence could not be read"
    elif active_writer:
        perf_reason = active_writer_summary or "an archive write/rebuild is in flight"
    else:
        perf_reason = "no concurrent archive write/rebuild detected"

    perf = ClaimGuardEntry(
        claim="perf_measurable",
        value=None if not active_writer_determinate else not active_writer,
        reason=perf_reason,
        signal="live_ingest_attempts.running_count",
    )

    return ClaimGuard(openable=openable, converged=converged, search_ready=search, perf_measurable=perf)


__all__ = [
    "ClaimGuard",
    "ClaimGuardEntry",
    "DerivedDomainReadiness",
    "derive_claim_guard",
    "search_unmeasured_reason",
    "raw_materialization_unmeasured_reason",
]
