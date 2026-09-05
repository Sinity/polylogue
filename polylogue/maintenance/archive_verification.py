"""Read-only archive-wide coherence verification.

Backs ``polylogue ops maintenance verify-archive``: a repeatable substitute
for the manual restore-verification checklist an operator runs by hand after
a blue-green index rebuild or a full archive restore ("does the archive
prove its own restore?"). Every check here is read-only and independent --
one check's exception, or one tier being temporarily busy/locked (a
concurrent rebuild is a first-class scenario, not an edge case), must never
prevent the remaining checks from reporting. Evidence is numbers, not just
booleans, so an operator can decide "is this drift acceptable right now?"
rather than only getting a red light.

Each check is declared by the domain adapter that owns its predicate and
population. Routes compose those declarations and the result runner remains
deliberately unaware of domain-specific status, waivers, or incident ledgers.

**Registry contract rule (polylogue-r4jiu):** a check's universe must be a
ground-truth table, never a derived ledger of the mechanism under audit. The
canonical violation this rule exists to prevent: an earlier version of
``source-index-coverage`` computed its universe from ``raw_membership_census``
(the *census machinery's own bookkeeping* of what it considered complete)
instead of ``raw_sessions`` (the durable, ground-truth table every acquired
raw lands in regardless of what any downstream census/reconciliation stage
later decides about it). Because the census never blesses a quarantined or
unreconciled raw, that raw was invisible to the check by construction -- the
check reported OK while a real, large materialization gap sat underneath it.
When adding a new check, ask: "if the mechanism I'm checking silently drops a
row from its own ledger, does my universe query still see that row?" If the
answer is no, the check is auditing the mechanism against itself, not against
reality.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from polylogue.archive.revision_authority import logical_head_cohort_sql
from polylogue.archive.topology.edge import HOOK_AUTHORITATIVE_LINK_METHOD, HOOK_CONTRADICTED_LINK_METHOD
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.outcomes import (
    BoundOutcomeOwner,
    OutcomeCheck,
    OutcomeCheckFn,
    OutcomeReport,
    OutcomeStatus,
    compose_outcome_checks,
)
from polylogue.logging import get_logger
from polylogue.maintenance.corpus_fidelity import (
    audit_absences,
    audit_attachment_fidelity,
    audit_chatgpt_content_conservation,
    audit_revision_fidelity,
)
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.storage.blob_integrity import scan_attachment_coverage, scan_blob_integrity
from polylogue.storage.blob_liveness import validated_blob_ref_liveness_joins
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.introspection import table_exists
from polylogue.storage.raw_failure_lifecycle import read_raw_failure_lifecycle
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

#: Default cap on per-check sample evidence (worst sessions, offending ids, ...).
DEFAULT_SAMPLE_LIMIT = 10


#: index-tier tables the planner-stats check expects ``ANALYZE`` coverage for
#: (polylogue-l3tk: fresh generations without stats pick pathological plans).
_PLANNER_STATS_COVERED_TABLES: tuple[str, ...] = ("blocks", "messages", "action_pairs")


@dataclass
class ArchiveVerificationCheck(OutcomeCheck):
    """One archive-coherence check outcome with structured evidence.

    Extends the shared :class:`~polylogue.core.outcomes.OutcomeCheck` grammar
    (``ok``/``warning``/``error``/``skip``) with a free-form ``evidence``
    payload for numbers that don't fit the base ``breakdown: dict[str, int]``
    shape (sample ids, per-tier dicts, worst-offender rows, ...).
    """

    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> JSONDocument:
        return archive_verification_check_json(self)


def archive_verification_check_json(check: OutcomeCheck) -> JSONDocument:
    """Return the JSON payload for one check, base attrs plus its evidence.

    Mirrors :func:`polylogue.schemas.audit.models.audit_check_json`: reads
    the shared :class:`OutcomeCheck` attrs directly and reaches for the
    subclass-only ``evidence`` field via
    ``getattr`` so callers holding a plain ``OutcomeCheck``-typed reference
    (e.g. ``ArchiveVerificationReport.checks``, typed at its base-class
    element type) can still serialize a concrete :class:`ArchiveVerificationCheck`
    without a narrowing cast.
    """
    payload = dict(check.to_dict())
    payload["evidence"] = json_document(getattr(check, "evidence", {}))
    return payload


def _error_check(
    name: str,
    summary: str,
    *,
    exc: Exception | None = None,
    evidence: dict[str, Any] | None = None,
) -> ArchiveVerificationCheck:
    payload = dict(evidence or {})
    if exc is not None:
        payload["error"] = str(exc)
    return ArchiveVerificationCheck(name=name, status=OutcomeStatus.ERROR, summary=summary, count=1, evidence=payload)


def _skip_check(name: str, summary: str) -> ArchiveVerificationCheck:
    return ArchiveVerificationCheck(name=name, status=OutcomeStatus.SKIP, summary=summary)


@dataclass
class ArchiveVerificationReport(OutcomeReport):
    """Full archive-verification report across every selected check."""

    archive_root: str = ""
    generated_at: str = ""
    coverage: ArchiveVerificationCoverage | None = None

    @property
    def blocking(self) -> bool:
        """True when any selected owner reports an error."""
        return any(check.status is OutcomeStatus.ERROR for check in self.checks)

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "archive_root": self.archive_root,
                "generated_at": self.generated_at,
                "summary": self.summary_counts(include_skip=True),
                "blocking": self.blocking,
                "coverage": self.coverage.to_json() if self.coverage is not None else None,
                "checks": [archive_verification_check_json(check) for check in self.checks],
            }
        )


def strict_acceptance_failures(
    report: ArchiveVerificationReport,
    *,
    required_checks: Sequence[str] | None = None,
    allow_not_applicable: bool = False,
) -> tuple[str, ...]:
    """Return every result that prevents a strict acceptance decision.

    Strict acceptance is deliberately separate from :attr:`blocking`.
    Acceptance proves a candidate is complete enough to activate: every
    required check must be present and ``ok``. Warnings, errors, and ordinary
    skips all fail this predicate. A caller may explicitly allow a typed
    not-applicable result, such as an origin-scoped check with no population.
    """
    required = (
        tuple(dict.fromkeys(required_checks))
        if required_checks is not None
        else tuple(check.name for check in report.checks)
    )
    by_name = {check.name: check for check in report.checks}
    failures: list[str] = []
    for name in required:
        check = by_name.get(name)
        if check is None:
            failures.append(f"{name}: required check result is missing")
            continue
        if check.status is OutcomeStatus.OK:
            continue
        if (
            allow_not_applicable
            and check.status is OutcomeStatus.SKIP
            and getattr(check, "evidence", {}).get("outcome_reason") == "no_chatgpt_population"
        ):
            continue
        failures.append(f"{check.name} [{check.status.value}]: {check.summary}")
    return tuple(failures)


def passes_strict_acceptance(
    report: ArchiveVerificationReport,
    *,
    required_checks: Sequence[str] | None = None,
    allow_not_applicable: bool = False,
) -> bool:
    """Return whether a report is complete and wholly ``ok`` for activation."""
    return not strict_acceptance_failures(
        report,
        required_checks=required_checks,
        allow_not_applicable=allow_not_applicable,
    )


def _tier_path(archive_root: Path, tier: ArchiveTier) -> Path:
    return archive_root / ARCHIVE_TIER_SPECS[tier].filename


def _resolve_index_path(archive_root: Path) -> Path:
    from polylogue.storage.archive_identity import resolve_active_index_path

    return resolve_active_index_path(archive_root)


def _open_ro(path: Path) -> sqlite3.Connection:
    return open_readonly_connection(path)


# ---------------------------------------------------------------------------
# Check 1: tier presence + schema versions
# ---------------------------------------------------------------------------


def _check_tier_schema(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    tiers_evidence: dict[str, JSONDocument] = {}
    missing: list[str] = []
    mismatched: list[str] = []

    for tier, spec in ARCHIVE_TIER_SPECS.items():
        path = _resolve_index_path(archive_root) if tier is ArchiveTier.INDEX else _tier_path(archive_root, tier)
        entry: dict[str, Any] = {
            "path": str(path),
            "expected_version": spec.version,
            "durability": spec.durability,
        }
        if not path.exists():
            missing.append(tier.value)
            entry["exists"] = False
            entry["actual_version"] = None
        else:
            entry["exists"] = True
            try:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
                try:
                    row = conn.execute("PRAGMA user_version").fetchone()
                finally:
                    conn.close()
                actual = int(row[0]) if row is not None else 0
                entry["actual_version"] = actual
                if actual != spec.version:
                    mismatched.append(tier.value)
            except sqlite3.Error as exc:
                entry["actual_version"] = None
                entry["error"] = str(exc)
                mismatched.append(tier.value)
        tiers_evidence[tier.value] = entry

    if missing or mismatched:
        status = OutcomeStatus.ERROR
        parts = []
        if missing:
            parts.append(f"missing: {', '.join(sorted(missing))}")
        if mismatched:
            parts.append(f"schema mismatch: {', '.join(sorted(mismatched))}")
        summary = "; ".join(parts)
    else:
        status = OutcomeStatus.OK
        summary = f"all {len(ARCHIVE_TIER_SPECS)} tiers present at their current schema version"

    return ArchiveVerificationCheck(
        name="tier-schema",
        status=status,
        summary=summary,
        count=len(missing) + len(mismatched),
        details=[*(f"missing:{t}" for t in sorted(missing)), *(f"schema-mismatch:{t}" for t in sorted(mismatched))],
        evidence={"tiers": tiers_evidence},
    )


# ---------------------------------------------------------------------------
# Check 2: pointer coherence (polylogue-k8kj class)
# ---------------------------------------------------------------------------


def _check_pointer_coherence(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    from polylogue.storage.archive_identity import ArchiveLocation, ArchiveLocationError

    try:
        location = ArchiveLocation.resolve(archive_root)
    except ArchiveLocationError as exc:
        return _error_check("pointer-coherence", f"invalid active index pointer: {exc}", exc=exc)

    configured_index = location.configured_tier("index")
    evidence: dict[str, Any] = {
        "configured_index_path": str(configured_index.configured_path),
        "active_pointer": str(location.active_pointer) if location.active_pointer is not None else None,
        "active_index_path": str(location.active_index.configured_path),
        "active_index_resolved_path": str(location.active_index.resolved_path),
    }

    if location.shadow_index is not None:
        evidence["shadow_index_resolved_path"] = str(location.shadow_index.resolved_path)
        return ArchiveVerificationCheck(
            name="pointer-coherence",
            status=OutcomeStatus.ERROR,
            summary=(
                "conventional index.db path diverges from the active .index-active-pointer target "
                "(interrupted rebuild promotion, polylogue-k8kj class)"
            ),
            count=1,
            details=[
                f"conventional={location.shadow_index.resolved_path}",
                f"active={location.active_index.resolved_path}",
            ],
            evidence=evidence,
        )

    return ArchiveVerificationCheck(
        name="pointer-coherence",
        status=OutcomeStatus.OK,
        summary="conventional index.db path and the active pointer target agree",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Check 3: source-vs-index coverage
# ---------------------------------------------------------------------------


def _valid_byte_duplicate_supersession_expr(conn: sqlite3.Connection, *, raw_alias: str) -> str:
    """Return the receipt predicate shared by source/index coverage checks.

    A supersession receipt is authority only when it still names the same
    bytes and an index materialization of the recorded duplicate twin. Keep
    the predicate in one place so backlog freshness cannot classify a receipt
    differently from source-index coverage.
    """
    if not table_exists(conn, "raw_byte_duplicate_supersession_receipts"):
        return "0"
    return f"""
        EXISTS(
            SELECT 1
            FROM raw_byte_duplicate_supersession_receipts receipt
            JOIN raw_sessions twin ON twin.raw_id = receipt.duplicate_of_raw_id
            JOIN idx_tier.sessions twin_session
              ON twin_session.raw_id = twin.raw_id
             AND twin_session.session_id = receipt.duplicate_of_session_id
            WHERE receipt.raw_id = {raw_alias}.raw_id
              AND receipt.blob_hash = {raw_alias}.blob_hash
              AND receipt.blob_size = {raw_alias}.blob_size
              AND twin.blob_hash = {raw_alias}.blob_hash
              AND twin.blob_size = {raw_alias}.blob_size
              AND twin.origin IS {raw_alias}.origin
              AND twin.source_path IS {raw_alias}.source_path
              AND twin.source_index IS {raw_alias}.source_index
        )
    """


def _logical_head_cohort_expr(conn: sqlite3.Connection, *, raw_alias: str) -> str:
    """Return the durable identity used to group raw revisions into one head.

    A full-revision row retired into membership governance intentionally loses
    its raw-level ``logical_source_key``.  Its single retained membership key
    remains the authoritative identity, so use it before the legacy
    native-id/path fallback.  Shared raws can hold several membership keys;
    they have no one raw-level cohort and must keep that fallback instead of
    being arbitrarily assigned to one member.
    """
    return logical_head_cohort_sql(
        conn,
        raw_alias=raw_alias,
        has_memberships=table_exists(conn, "raw_session_memberships"),
    )


def _check_source_index_coverage(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_source_index_coverage_at_index_path(archive_root, _resolve_index_path(archive_root), sample_limit)


def _check_source_index_coverage_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Every logical source's head is indexed OR carries a typed refusal.

    Universe (polylogue-r4jiu, invariant I1): ``raw_sessions`` logical heads --
    the latest revision per
    ``(origin, COALESCE(logical_source_key, native_id, source_path))`` --
    which is a ground-truth table every acquired raw lands in, not a ledger a
    downstream reconciliation stage can silently omit rows from. A logical
    head counts as covered when *any* raw in its revision group is
    materialized into ``index.db``. An uncovered head must be typed as one
    of: ``parse_error`` (raw_sessions.parse_error set), ``non_session`` /
    ``census_failed`` (raw_membership_census recorded a terminal non-complete
    verdict), a content-bound byte-supersession receipt whose named twin is
    present in the candidate index, or ``quarantined``
    (raw_sessions.revision_authority -- reconciliation hasn't granted it
    authority to write yet, WARN-level evidence, not blocking). An uncovered
    head matching none of those is an
    *untyped* gap -- a materialization failure no other subsystem has
    explained -- and is the only ERROR-gating condition here besides orphans
    (index sessions whose raw_id doesn't exist in source.db at all).
    """
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    if not source_path.exists() or not index_path.exists():
        return _skip_check("source-index-coverage", "source.db or index.db not present")

    try:
        conn = _open_ro(source_path)
    except sqlite3.Error as exc:
        return _error_check("source-index-coverage", f"could not open source.db: {exc}", exc=exc)

    try:
        try:
            conn.execute("ATTACH DATABASE ? AS idx_tier", (f"file:{index_path}?mode=ro",))
        except sqlite3.Error as exc:
            return _error_check("source-index-coverage", f"could not attach index.db: {exc}", exc=exc)

        try:
            has_census = table_exists(conn, "raw_membership_census")
            census_expr = (
                "(SELECT c.status FROM raw_membership_census c WHERE c.raw_id = r.raw_id)" if has_census else "NULL"
            )
            valid_supersession_expr = _valid_byte_duplicate_supersession_expr(conn, raw_alias="r")
            logical_cohort_expr = _logical_head_cohort_expr(conn, raw_alias="r")

            # A read-only connection (``query_only=ON``, connection-wide, not
            # per-attached-db) cannot ``CREATE TEMP VIEW`` -- the temp schema
            # is still a write. Repeat the heads CTE per query instead; it is
            # a plain in-query derived table, not a persisted write.
            heads_cte = f"""
                WITH heads AS (
                    SELECT
                        r.raw_id,
                        r.blob_hash,
                        r.parse_error,
                        r.revision_authority,
                        r.logical_source_key,
                        {census_expr} AS census_status,
                        {valid_supersession_expr} AS valid_supersession,
                        MAX(EXISTS(SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = r.raw_id))
                            OVER (
                                PARTITION BY r.origin,
                                             {logical_cohort_expr}
                            ) AS any_indexed,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.origin,
                                         {logical_cohort_expr}
                            ORDER BY r.acquired_at_ms DESC, r.raw_id DESC
                        ) AS rn
                    FROM raw_sessions r
                )
            """
            untyped_predicate = """
                rn = 1 AND any_indexed = 0 AND parse_error IS NULL
                  AND COALESCE(census_status, '') NOT IN ('non_session', 'failed')
                  AND valid_supersession = 0
                  AND revision_authority != 'quarantined'
            """
            quarantined_predicate = """
                rn = 1 AND any_indexed = 0 AND parse_error IS NULL
                  AND COALESCE(census_status, '') NOT IN ('non_session', 'failed')
                  AND valid_supersession = 0
                  AND revision_authority = 'quarantined'
            """

            total_heads = int(conn.execute(f"{heads_cte} SELECT COUNT(*) FROM heads WHERE rn = 1").fetchone()[0])

            counts = conn.execute(
                f"""
                {heads_cte}
                SELECT
                  SUM(CASE WHEN any_indexed = 0 AND parse_error IS NOT NULL THEN 1 ELSE 0 END),
                  SUM(CASE WHEN any_indexed = 0 AND parse_error IS NULL
                            AND census_status = 'non_session' THEN 1 ELSE 0 END),
                  SUM(CASE WHEN any_indexed = 0 AND parse_error IS NULL
                            AND COALESCE(census_status, '') != 'non_session'
                            AND census_status = 'failed' THEN 1 ELSE 0 END),
                  SUM(CASE WHEN any_indexed = 0 AND valid_supersession = 1 THEN 1 ELSE 0 END),
                  SUM(CASE WHEN {quarantined_predicate.replace("rn = 1 AND ", "")} THEN 1 ELSE 0 END),
                  SUM(CASE WHEN {untyped_predicate.replace("rn = 1 AND ", "")} THEN 1 ELSE 0 END)
                FROM heads WHERE rn = 1
                """
            ).fetchone()
            parse_error_n, non_session_n, census_failed_n, superseded_n, quarantined_n, untyped_n = (
                int(value or 0) for value in counts
            )

            untyped_sample = [
                str(row[0])
                for row in conn.execute(
                    f"{heads_cte} SELECT raw_id FROM heads WHERE {untyped_predicate} LIMIT ?",
                    (sample_limit,),
                )
            ]
            quarantined_sample = [
                str(row[0])
                for row in conn.execute(
                    f"{heads_cte} SELECT raw_id FROM heads WHERE {quarantined_predicate} LIMIT ?",
                    (sample_limit,),
                )
            ]

            # polylogue-t0m73 (2026-08-03 correction to I1's framing): an
            # unindexed head whose bytes are byte-identical to an already-
            # indexed raw (same blob_hash, different raw_id -- e.g. a second
            # acquisition of content already captured) is not novel missing
            # content. The operator's live-run challenge measured 4,305 of
            # 7,200 unindexed heads (77%) in this bucket -- without it, the
            # check's own evidence overstates the gap as if every unindexed
            # head were a distinct materialization failure. This never
            # changes ``blocking`` (an untyped/orphan head is still an error
            # regardless of whether its bytes happen to be duplicated
            # elsewhere); it only prevents the report from repeating the
            # overstatement.
            byte_dup_of_indexed_n = int(
                conn.execute(
                    f"""
                    {heads_cte}
                    SELECT SUM(CASE WHEN {quarantined_predicate.replace("rn = 1 AND ", "")}
                                       OR {untyped_predicate.replace("rn = 1 AND ", "")}
                                       OR valid_supersession = 1
                                  THEN (
                                    EXISTS(
                                        SELECT 1 FROM raw_sessions dup
                                        JOIN idx_tier.sessions ds ON ds.raw_id = dup.raw_id
                                        WHERE dup.blob_hash = heads.blob_hash
                                    )
                                  ) ELSE 0 END)
                    FROM heads WHERE rn = 1
                    """
                ).fetchone()[0]
                or 0
            )

            orphan_count = conn.execute(
                """
                SELECT COUNT(*) FROM idx_tier.sessions s
                WHERE s.raw_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = s.raw_id)
                """
            ).fetchone()[0]
            orphan_sample = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT DISTINCT s.raw_id FROM idx_tier.sessions s
                    WHERE s.raw_id IS NOT NULL
                      AND NOT EXISTS (SELECT 1 FROM raw_sessions r WHERE r.raw_id = s.raw_id)
                    LIMIT ?
                    """,
                    (sample_limit,),
                )
            ]
        except sqlite3.Error as exc:
            return _error_check("source-index-coverage", f"could not read source/index tiers: {exc}", exc=exc)
    finally:
        conn.close()

    unindexed_head_count = parse_error_n + non_session_n + census_failed_n + superseded_n + quarantined_n + untyped_n
    blocking = untyped_n > 0 or int(orphan_count or 0) > 0
    warning = quarantined_n > 0

    if blocking:
        status = OutcomeStatus.ERROR
    elif warning:
        status = OutcomeStatus.WARNING
    else:
        status = OutcomeStatus.OK

    novel_unindexed_n = max(unindexed_head_count - byte_dup_of_indexed_n, 0)

    parts = [f"{total_heads:,} logical source head(s), {unindexed_head_count:,} unindexed"]
    if untyped_n:
        parts.append(f"untyped={untyped_n:,}")
    if quarantined_n:
        parts.append(f"quarantined={quarantined_n:,} (WARN evidence, not blocking)")
    if parse_error_n:
        parts.append(f"parse_error={parse_error_n:,}")
    if non_session_n or census_failed_n:
        parts.append(f"declared-non-session={non_session_n + census_failed_n:,}")
    if superseded_n:
        parts.append(f"superseded-byte-duplicate={superseded_n:,}")
    if byte_dup_of_indexed_n:
        parts.append(
            f"byte-dup-of-indexed={byte_dup_of_indexed_n:,} (novel={novel_unindexed_n:,} of {unindexed_head_count:,})"
        )
    if orphan_count:
        parts.append(f"orphans={int(orphan_count):,}")
    summary = "; ".join(parts)

    return ArchiveVerificationCheck(
        name="source-index-coverage",
        status=status,
        summary=summary,
        count=untyped_n + int(orphan_count or 0),
        details=[
            *(f"untyped:{raw_id}" for raw_id in untyped_sample),
            *(f"orphan:{raw_id}" for raw_id in orphan_sample),
        ],
        evidence={
            "logical_head_count": total_heads,
            "unindexed_head_count": unindexed_head_count,
            "untyped_count": untyped_n,
            "untyped_sample": untyped_sample,
            "parse_error_count": parse_error_n,
            "non_session_count": non_session_n,
            "census_failed_count": census_failed_n,
            "superseded_byte_duplicate_count": superseded_n,
            "quarantined_count": quarantined_n,
            "quarantined_sample": quarantined_sample,
            "byte_dup_of_indexed_count": byte_dup_of_indexed_n,
            "novel_unindexed_count": novel_unindexed_n,
            "orphan_count": int(orphan_count or 0),
            "orphan_sample": orphan_sample,
        },
    )


def _check_source_index_coverage_at_candidate(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    return _check_source_index_coverage_at_index_path(archive_root, index_path, sample_limit)


def archive_verification_owner_adapters(
    archive_root: Path,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    index_path_override: Path | None = None,
) -> tuple[BoundOutcomeOwner, BoundOutcomeOwner]:
    """Return two domain-owned checks through the shared callable/result seam.

    The index owner is only applicable to the live index path.  The source
    owner has a cross-tier candidate adapter, so candidate composition can
    omit the index owner without inventing a universal obligation taxonomy.
    This narrow adapter is retained for callers that need only these two
    source/index owners; the full route composes domain declarations below.
    """
    if index_path_override is not None:
        index_check = None

        def source_check() -> OutcomeCheck:
            return _check_source_index_coverage_at_candidate(archive_root, index_path_override, sample_limit)
    else:

        def index_check() -> OutcomeCheck:
            return _check_fts_parity(archive_root, sample_limit)

        def source_check() -> OutcomeCheck:
            return _check_source_index_coverage(archive_root, sample_limit)

    return (
        BoundOutcomeOwner(name="fts-parity", check=index_check),
        BoundOutcomeOwner(name="source-index-coverage", check=source_check),
    )


def _declared_owner(
    *,
    name: str,
    check: OutcomeCheckFn,
    semantic_owner: str,
    applicable_routes: frozenset[str],
    production_route: str,
    population: tuple[str, ...],
    owned_reference: str,
) -> BoundOutcomeOwner:
    return BoundOutcomeOwner(
        name=name,
        check=check,
        semantic_owner=semantic_owner,
        applicable_routes=applicable_routes,
        production_route=production_route,
        population=population,
        owned_reference=owned_reference,
    )


def archive_verification_migrated_owner_adapters(
    archive_root: Path,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    index_path_override: Path | None = None,
    active_index_context: Literal["required", "unavailable_for_candidate"] = "required",
) -> tuple[BoundOutcomeOwner, ...]:
    """Bind the migrated source/storage producers to the result seam.

    The tuple is deliberately grouped by owner domain rather than by a new
    universal obligation taxonomy: source materialization, hook topology,
    blob/attachment lifecycle, cursor convergence, and source fidelity each
    retain their own predicate and denominator.  Candidate bindings are only
    supplied where the production route needs them.
    """
    return (
        # Derived-index owner.  The callable remains bound here so all
        # callers receive the same FTS lifecycle oracle and its evidence.
        _declared_owner(
            name="fts-parity",
            semantic_owner="fts-derivation",
            applicable_routes=frozenset({_ROUTE_INDEX_CANDIDATE, _ROUTE_LIVE}),
            production_route="daemon FTS convergence",
            population=("index.db.messages", "index.db.blocks"),
            owned_reference="test_deleted_fts_row_trips_message_fts_parity",
            check=lambda: _check_fts_parity(archive_root, sample_limit, index_path=index_path_override),
        ),
        # Topology owner.  The two named projections below share the graph
        # loader/traversal in their implementation, while retaining their
        # distinct public law identities and candidate selection.
        _declared_owner(
            name="lineage-sanity",
            semantic_owner="topology",
            applicable_routes=frozenset({_ROUTE_INDEX_CANDIDATE, _ROUTE_LIVE}),
            production_route="index graph materialization",
            population=("index.db.session_links", "index.db.sessions", "index.db.messages"),
            owned_reference="test_dangling_resolved_dst_trips_lineage_sanity",
            check=lambda: _check_lineage_sanity(archive_root, sample_limit, index_path=index_path_override),
        ),
        _declared_owner(
            name="session-lineage-acyclic",
            semantic_owner="topology",
            applicable_routes=frozenset({_ROUTE_INDEX_CANDIDATE, _ROUTE_LIVE}),
            production_route="index graph materialization",
            population=("index.db.sessions",),
            owned_reference="test_parent_session_id_cycle_trips_session_lineage_acyclic",
            check=lambda: _check_session_lineage_acyclic(archive_root, sample_limit, index_path=index_path_override),
        ),
        _declared_owner(
            name="message-count-projection",
            semantic_owner="session-projection",
            applicable_routes=frozenset({_ROUTE_INDEX_CANDIDATE, _ROUTE_LIVE}),
            production_route="session materialization",
            population=("index.db.sessions", "index.db.messages"),
            owned_reference="test_drifted_message_count_trips_message_count_projection",
            check=lambda: _check_message_count_projection(archive_root, sample_limit, index_path=index_path_override),
        ),
        _declared_owner(
            name="session-fingerprint-stamps",
            semantic_owner="session-hashing",
            applicable_routes=frozenset({_ROUTE_INDEX_CANDIDATE, _ROUTE_LIVE}),
            production_route="canonical replay",
            population=("index.db.sessions",),
            owned_reference="test_reindex_acceptance_rejects_missing_semantic_stamp_coverage",
            check=lambda: _check_session_fingerprint_stamps(archive_root, sample_limit, index_path=index_path_override),
        ),
        # Planner statistics and convergence freshness are live daemon health
        # observations.  They are intentionally not candidate obligations.
        _declared_owner(
            name="planner-stats",
            semantic_owner="daemon-index-health",
            applicable_routes=frozenset({_ROUTE_HEALTH_MEDIUM, _ROUTE_LIVE}),
            production_route="daemon index health",
            population=("index.db.sqlite_stat1",),
            owned_reference="test_missing_sqlite_stat1_is_warning_not_error",
            check=lambda: _check_planner_stats(archive_root, sample_limit, index_path=index_path_override),
        ),
        _declared_owner(
            name="convergence-freshness",
            semantic_owner="daemon-convergence-health",
            applicable_routes=frozenset({_ROUTE_HEALTH_MEDIUM, _ROUTE_LIVE}),
            production_route="daemon convergence",
            population=("source.db.raw_sessions", "ops.db.ingest_attempts"),
            owned_reference="test_stalled_backlog_trips_convergence_freshness",
            check=lambda: _check_convergence_freshness(archive_root, sample_limit),
        ),
        _declared_owner(
            name="active-leaf-title-convergence",
            semantic_owner="title-derivation",
            applicable_routes=frozenset(
                {_ROUTE_HEALTH_MEDIUM, _ROUTE_INDEX_CANDIDATE, _ROUTE_CANARY_CANDIDATE, _ROUTE_LIVE}
            ),
            production_route="title derivation",
            population=("index.db.sessions", "index.db.messages"),
            owned_reference="test_second_active_leaf_trips_active_leaf_title_convergence",
            check=lambda: _check_active_leaf_title_convergence(
                archive_root, sample_limit, index_path=index_path_override
            ),
        ),
        _declared_owner(
            name="tier-schema",
            semantic_owner="storage-schema",
            applicable_routes=frozenset({_ROUTE_LIVE}),
            production_route="archive-readiness",
            population=("archive tiers",),
            owned_reference="ARCHIVE_TIER_SPECS",
            check=lambda: _check_tier_schema(archive_root, sample_limit),
        ),
        _declared_owner(
            name="pointer-coherence",
            semantic_owner="index-generation-lifecycle",
            applicable_routes=frozenset({_ROUTE_LIVE}),
            production_route="generation activation",
            population=("index pointer",),
            owned_reference="archive_identity",
            check=lambda: _check_pointer_coherence(archive_root, sample_limit),
        ),
        _declared_owner(
            name="enum-superset-check",
            semantic_owner="schema-generation",
            applicable_routes=frozenset({_ROUTE_INDEX_CANDIDATE, _ROUTE_LIVE}),
            production_route="schema generation",
            population=("sqlite_master",),
            owned_reference="test_missing_enum_value_trips_enum_superset_check",
            check=lambda: _check_enum_superset(archive_root, sample_limit),
        ),
        _declared_owner(
            name="embeddings-refs-liveness",
            semantic_owner="embedding-retention",
            applicable_routes=frozenset({_ROUTE_HEALTH_MEDIUM, _ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_LIVE}),
            production_route="embedding generation",
            population=("embeddings.db.message_embedding_refs", "index.db.messages"),
            owned_reference="test_orphaned_embedding_ref_trips_embeddings_refs_liveness",
            check=lambda: (
                _check_embeddings_refs_liveness_at_candidate(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_embeddings_refs_liveness(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="user-tier-refs",
            semantic_owner="durable-assertions",
            applicable_routes=frozenset({_ROUTE_HEALTH_MEDIUM, _ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_LIVE}),
            production_route="assertion rebinding",
            population=("user.db.assertions", "index.db.sessions", "index.db.messages"),
            owned_reference="test_dangling_assertion_target_trips_user_tier_refs",
            check=lambda: (
                _check_user_tier_refs_at_candidate(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_user_tier_refs(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="source-index-coverage",
            semantic_owner="source-materialization",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_LIVE}),
            production_route="source-to-index replay",
            population=("source.db.raw_sessions", "index.db.sessions"),
            owned_reference="test_source_index_coverage_census_deletion_does_not_hide_raw_head",
            check=lambda: (
                _check_source_index_coverage_at_candidate(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_source_index_coverage(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="hook-authority-topology-conflict",
            semantic_owner="topology",
            applicable_routes=frozenset({_ROUTE_INDEX_CANDIDATE, _ROUTE_LIVE}),
            production_route="hook reconciliation",
            population=("index.db.session_links",),
            owned_reference="test_contradiction_without_authoritative_winner_trips_the_check",
            check=lambda: _check_hook_authority_topology_conflict(archive_root, sample_limit),
        ),
        _declared_owner(
            name="blob-refs-liveness",
            semantic_owner="blob-lifecycle",
            applicable_routes=frozenset(
                {_ROUTE_HEALTH_MEDIUM, _ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_SOURCE_PREFLIGHT, _ROUTE_LIVE}
            ),
            production_route="blob liveness/GC",
            population=("source.db.blob_refs", "source.db.raw_sessions"),
            owned_reference="test_blob_ref_with_no_referent_trips_blob_refs_liveness",
            check=lambda: (
                _check_blob_refs_liveness_at_candidate(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_blob_refs_liveness(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="blob-integrity",
            semantic_owner="blob-lifecycle",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_SOURCE_PREFLIGHT, _ROUTE_LIVE}),
            production_route="source seal",
            population=("source.db.raw_sessions", "blob/"),
            owned_reference="test_full_blob_integrity_red_twin",
            check=lambda: (
                _check_blob_integrity_at_candidate(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_blob_integrity(archive_root, sample_limit, active_index_context=active_index_context)
            ),
        ),
        _declared_owner(
            name="attachment-coverage",
            semantic_owner="attachment-acquisition",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_SOURCE_PREFLIGHT, _ROUTE_LIVE}),
            production_route="attachment acquisition",
            population=("index.db.attachments", "index.db.attachment_refs", "blob/"),
            owned_reference="test_acquired_unreachable_attachment_debt_is_blocking",
            check=lambda: (
                _check_attachment_coverage_at_candidate(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_attachment_coverage(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="raw-failure-lifecycle",
            semantic_owner="source-failure-lifecycle",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_SOURCE_PREFLIGHT, _ROUTE_LIVE}),
            production_route="source admission",
            population=("source.db.raw_sessions", "source.db.raw_artifacts"),
            owned_reference="test_raw_failure_lifecycle_mutation_red_twin_for_validation_failures",
            check=lambda: (
                _check_raw_failure_lifecycle_at_candidate(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_raw_failure_lifecycle(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="blob-reference-closure",
            semantic_owner="blob-lifecycle",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_LIVE}),
            production_route="blob reference closure",
            population=("source.db.blob_refs", "index.db.attachment_refs"),
            owned_reference="test_blob_reference_closure_rejects_acquired_attachment_without_ref",
            check=lambda: (
                _check_blob_reference_closure_at_index_path(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_blob_reference_closure(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="excluded-cursor-vocabulary-honesty",
            semantic_owner="source-cursor",
            applicable_routes=frozenset({_ROUTE_HEALTH_MEDIUM, _ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_LIVE}),
            production_route="cursor writer",
            population=("ops.db.ingest_cursor",),
            owned_reference="test_excluded_cursor_with_live_next_retry_at_trips_vocabulary_honesty",
            check=lambda: _check_excluded_cursor_vocabulary_honesty(archive_root, sample_limit),
        ),
        _declared_owner(
            name="stalled-append-cursor-freshness",
            semantic_owner="source-cursor",
            applicable_routes=frozenset({_ROUTE_HEALTH_MEDIUM, _ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_LIVE}),
            production_route="append continuity",
            population=("ops.db.ingest_cursor",),
            owned_reference="test_stalled_append_cursor_trips_freshness_check",
            check=lambda: _check_stalled_append_cursor_freshness(archive_root, sample_limit),
        ),
        _declared_owner(
            name="chatgpt-content-conservation",
            semantic_owner="chatgpt-admission",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_LIVE, _ROUTE_CORPUS_FIDELITY}),
            production_route="ChatGPT source admission",
            population=("source.db.raw_sessions", "blob/", "index.db.messages"),
            owned_reference="test_dropped_raw_node_trips_chatgpt_content_conservation",
            check=lambda: (
                _check_chatgpt_content_conservation_at_index_path(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_chatgpt_content_conservation(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="corpus-absences",
            semantic_owner="candidate-corpus-fidelity",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_CORPUS_FIDELITY}),
            production_route="candidate corpus acceptance",
            population=("source.db.raw_session_memberships", "index.db.sessions"),
            owned_reference="test_corpus_absences_red_twin",
            check=lambda: (
                _check_corpus_absences_at_index_path(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_corpus_absences(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="corpus-attachment-fidelity",
            semantic_owner="candidate-attachment-fidelity",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_CORPUS_FIDELITY}),
            production_route="candidate attachment acceptance",
            population=("index.db.attachments", "index.db.attachment_refs"),
            owned_reference="test_corpus_attachment_fidelity_red_twin",
            check=lambda: (
                _check_corpus_attachment_fidelity_at_index_path(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_corpus_attachment_fidelity(archive_root, sample_limit)
            ),
        ),
        _declared_owner(
            name="corpus-revision-fidelity",
            semantic_owner="candidate-revision-fidelity",
            applicable_routes=frozenset({_ROUTE_CROSS_TIER_CANDIDATE, _ROUTE_CORPUS_FIDELITY}),
            production_route="candidate revision acceptance",
            population=("source.db.raw_session_memberships", "index.db.sessions", "index.db.messages"),
            owned_reference="test_corpus_revision_fidelity_red_twin",
            check=lambda: (
                _check_corpus_revision_fidelity_at_index_path(archive_root, index_path_override, sample_limit)
                if index_path_override is not None
                else _check_corpus_revision_fidelity(archive_root, sample_limit)
            ),
        ),
    )


# Route labels are consumed by the declarations below; route coverage is
# compiled from each owner's ``applicable_routes`` field.
_ROUTE_LIVE = "live-archive"
_ROUTE_SOURCE_PREFLIGHT = "reindex-source-preflight"
_ROUTE_INDEX_CANDIDATE = "reindex-index-candidate"
_ROUTE_CROSS_TIER_CANDIDATE = "reindex-cross-tier-candidate"
_ROUTE_CANARY_CANDIDATE = "reindex-canary-candidate"
_ROUTE_CORPUS_FIDELITY = "corpus-fidelity"
_ROUTE_HEALTH_MEDIUM = "health-medium"
_CANDIDATE_ROUTES = frozenset(
    {
        _ROUTE_INDEX_CANDIDATE,
        _ROUTE_CROSS_TIER_CANDIDATE,
        _ROUTE_CANARY_CANDIDATE,
        _ROUTE_CORPUS_FIDELITY,
    }
)


def archive_verification_domain_adapters(
    archive_root: Path,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    index_path_override: Path | None = None,
    active_index_context: Literal["required", "unavailable_for_candidate"] = "required",
) -> tuple[BoundOutcomeOwner, ...]:
    """Return executable declarations from the owners of each predicate."""
    owners = archive_verification_migrated_owner_adapters(
        archive_root,
        sample_limit=sample_limit,
        index_path_override=index_path_override,
        active_index_context=active_index_context,
    )
    if not all(owner.semantic_owner and owner.owned_reference and owner.applicable_routes for owner in owners):
        raise ValueError("an archive verification owner omitted its domain declaration")
    return tuple(
        replace(
            owner,
            candidate_check=(
                owner.check if index_path_override is not None and owner.applicable_routes & _CANDIDATE_ROUTES else None
            ),
        )
        for owner in owners
    )


@dataclass(frozen=True)
class ArchiveVerificationCoverage:
    """Disposable coverage evidence compiled from owner declarations."""

    route: str
    candidate_id: str | None
    declarations: tuple[BoundOutcomeOwner, ...]
    missing_production_routes: tuple[str, ...] = ()
    ownerless_checks: tuple[str, ...] = ()
    duplicate_checks: tuple[str, ...] = ()
    retirement_candidates: tuple[str, ...] = (
        "pathology-zoo-invariants",
        "counts-summary",
        "raw-quarantine-group-dedup",
    )

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "route": self.route,
                "candidate_id": self.candidate_id,
                "denominator": len(self.declarations),
                "declarations": [
                    {
                        "name": owner.name,
                        "semantic_owner": owner.semantic_owner,
                        "production_route": owner.production_route,
                        "population": list(owner.population),
                        "owned_reference": owner.owned_reference,
                        "candidate_runner": owner.candidate_check is not None,
                    }
                    for owner in self.declarations
                ],
                "missing_production_routes": list(self.missing_production_routes),
                "ownerless_checks": list(self.ownerless_checks),
                "duplicate_checks": list(self.duplicate_checks),
                "retirement_candidates": list(self.retirement_candidates),
            }
        )


def archive_verification_coverage(
    *,
    archive_root: Path,
    route: str = _ROUTE_LIVE,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    index_path_override: Path | None = None,
    active_index_context: Literal["required", "unavailable_for_candidate"] = "required",
) -> ArchiveVerificationCoverage:
    declarations = archive_verification_domain_adapters(
        archive_root,
        sample_limit=sample_limit,
        index_path_override=index_path_override,
        active_index_context=active_index_context,
    )
    selected = tuple(owner for owner in declarations if route in owner.applicable_routes)
    names = tuple(owner.name for owner in selected)
    return ArchiveVerificationCoverage(
        route=route,
        candidate_id=str(index_path_override) if index_path_override is not None else None,
        declarations=selected,
        missing_production_routes=tuple(owner.name for owner in selected if not owner.production_route),
        ownerless_checks=tuple(
            owner.name for owner in selected if not owner.semantic_owner or not owner.owned_reference
        ),
        duplicate_checks=tuple(sorted({name for name in names if names.count(name) > 1})),
    )


# ---------------------------------------------------------------------------
# Check 4: FTS parity (archive-wide)
# ---------------------------------------------------------------------------


def _check_fts_parity(
    archive_root: Path, sample_limit: int, *, index_path: Path | None = None
) -> ArchiveVerificationCheck:
    index_path = index_path or _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("fts-parity", "index.db not present")

    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("fts-parity", f"could not open index.db: {exc}", exc=exc)

    evidence: dict[str, Any] = {}
    problems: list[str] = []
    try:
        from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

        messages = fts_invariant_snapshot_sync(conn).messages
        expected, indexed = messages.source_rows, messages.indexed_rows
        gap = expected - indexed
        worst_sessions: list[dict[str, Any]] = []
        if gap and table_exists(conn, "blocks") and table_exists(conn, "messages_fts_docsize"):
            rows = conn.execute(
                """
                SELECT b.session_id,
                       COUNT(*) FILTER (WHERE b.search_text != '') AS expected,
                       COUNT(d.id) FILTER (WHERE b.search_text != '') AS indexed
                FROM blocks AS b
                LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                GROUP BY b.session_id
                HAVING expected != indexed
                ORDER BY (expected - indexed) DESC
                LIMIT ?
                """,
                (sample_limit,),
            ).fetchall()
            worst_sessions = [{"session_id": str(r[0]), "expected": int(r[1]), "indexed": int(r[2])} for r in rows]
        if messages.source_exists and not messages.exists:
            problems.append("messages_fts missing")
        if gap:
            problems.append(f"messages_fts gap={gap}")
        elif messages.missing_rows:
            problems.append(f"messages_fts missing_rows={messages.missing_rows}")
        if messages.excess_rows:
            problems.append(f"messages_fts excess_rows={messages.excess_rows}")
        if messages.duplicate_rows:
            problems.append(f"messages_fts duplicate_rows={messages.duplicate_rows}")
        if messages.identity_mismatch_rows:
            problems.append(f"messages_fts identity_mismatch_rows={messages.identity_mismatch_rows}")
        if messages.exists and not messages.triggers_present:
            problems.append("messages_fts triggers missing")
        evidence["messages_fts"] = {
            "expected": expected,
            "indexed": indexed,
            "gap": gap,
            "missing_rows": messages.missing_rows,
            "excess_rows": messages.excess_rows,
            "duplicate_rows": messages.duplicate_rows,
            "identity_mismatch_rows": messages.identity_mismatch_rows,
            "triggers_present": messages.triggers_present,
            "worst_sessions": worst_sessions,
        }

        if table_exists(conn, "blocks") and table_exists(conn, "blocks_command_trigram_docsize"):
            # blocks_command_trigram is an external-content FTS5 table
            # (content='blocks'): a bare, MATCH-less ``SELECT rowid FROM
            # blocks_command_trigram`` reads through to the content table's
            # rowids regardless of whether that rowid was ever indexed --
            # verified locally, an fts5 'delete' command removes the row from
            # ``blocks_command_trigram_docsize`` but a plain unfiltered
            # select against the virtual table itself still returns it. The
            # docsize shadow table (same convention messages_fts_docsize
            # uses above) is what actually reflects indexed state.
            row = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE b.block_type = 'tool_use' AND b.tool_detail_text != ' '),
                    COUNT(d.id) FILTER (WHERE b.block_type = 'tool_use' AND b.tool_detail_text != ' ')
                FROM blocks AS b
                LEFT JOIN blocks_command_trigram_docsize AS d ON d.id = b.rowid
                """
            ).fetchone()
            texpected, tindexed = int(row[0] or 0), int(row[1] or 0)
            tgap = texpected - tindexed
            evidence["blocks_command_trigram"] = {"expected": texpected, "indexed": tindexed, "gap": tgap}
            if tgap:
                problems.append(f"blocks_command_trigram gap={tgap}")
        else:
            evidence["blocks_command_trigram"] = None
    except sqlite3.Error as exc:
        return _error_check("fts-parity", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    status = OutcomeStatus.ERROR if problems else OutcomeStatus.OK
    summary = "; ".join(problems) if problems else "messages_fts and blocks_command_trigram exactly in sync"
    return ArchiveVerificationCheck(
        name="fts-parity",
        status=status,
        summary=summary,
        count=len(problems),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Check 5: lineage sanity
# ---------------------------------------------------------------------------


def _lineage_parent_graph(conn: sqlite3.Connection) -> tuple[dict[str, str], set[str]]:
    """Load and validate the stored parent graph once for topology checks."""
    parents = {
        str(row[0]): str(row[1])
        for row in conn.execute(
            "SELECT session_id, parent_session_id FROM sessions WHERE parent_session_id IS NOT NULL"
        ).fetchall()
    }
    resolved: set[str] = set()
    cycle_members: set[str] = set()
    for start in parents:
        if start in resolved:
            continue
        path: list[str] = []
        node = start
        while node in parents and node not in resolved:
            if node in path:
                cycle_members.update(path[path.index(node) :])
                break
            path.append(node)
            node = parents[node]
        resolved.update(path)
    return parents, cycle_members


def _check_lineage_sanity(
    archive_root: Path, sample_limit: int, *, index_path: Path | None = None
) -> ArchiveVerificationCheck:
    index_path = index_path or _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("lineage-sanity", "index.db not present")

    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("lineage-sanity", f"could not open index.db: {exc}", exc=exc)

    try:
        if not table_exists(conn, "session_links"):
            return _skip_check("lineage-sanity", "session_links table not present")

        dangling_dst_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM session_links AS sl
                WHERE sl.resolved_dst_session_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM sessions AS s WHERE s.session_id = sl.resolved_dst_session_id)
                """
            ).fetchone()[0]
        )
        dangling_dst_sample = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT sl.resolved_dst_session_id FROM session_links AS sl
                WHERE sl.resolved_dst_session_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM sessions AS s WHERE s.session_id = sl.resolved_dst_session_id)
                LIMIT ?
                """,
                (sample_limit,),
            )
        ]

        dangling_branch_point_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM session_links AS sl
                WHERE sl.branch_point_message_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM messages AS m WHERE m.message_id = sl.branch_point_message_id)
                """
            ).fetchone()[0]
        )
        dangling_branch_point_sample = [
            {
                "src_session_id": str(row[0]),
                "dst_origin": str(row[1]),
                "dst_native_id": str(row[2]),
                "branch_point_message_id": str(row[3]),
            }
            for row in conn.execute(
                """
                SELECT sl.src_session_id, sl.dst_origin, sl.dst_native_id, sl.branch_point_message_id
                FROM session_links AS sl
                WHERE sl.branch_point_message_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM messages AS m WHERE m.message_id = sl.branch_point_message_id)
                LIMIT ?
                """,
                (sample_limit,),
            )
        ]
    except sqlite3.Error as exc:
        return _error_check("lineage-sanity", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    problems: list[str] = []
    if dangling_dst_count:
        problems.append(f"dangling resolved_dst_session_id x{dangling_dst_count}")
    if dangling_branch_point_count:
        problems.append(f"dangling branch_point_message_id x{dangling_branch_point_count}")

    status = OutcomeStatus.ERROR if problems else OutcomeStatus.OK
    return ArchiveVerificationCheck(
        name="lineage-sanity",
        status=status,
        summary="; ".join(problems) if problems else "session_links lineage references resolve cleanly",
        count=dangling_dst_count + dangling_branch_point_count,
        evidence={
            "dangling_resolved_dst_count": dangling_dst_count,
            "dangling_resolved_dst_sample": dangling_dst_sample,
            "dangling_branch_point_count": dangling_branch_point_count,
            "dangling_branch_point_sample": dangling_branch_point_sample,
        },
    )


# ---------------------------------------------------------------------------
# Check: hook-authority-topology-conflict (polylogue-hook-authority-conflict-proof)
# ---------------------------------------------------------------------------


def _check_hook_authority_topology_conflict(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Census contradictions between hook evidence and transcript inference.

    A contradiction is not an error: acquired ``codex_thread_spawn_edge``
    evidence disagreeing with a parser-inferred parent is exactly the case the
    write path is designed to resolve, and the quarantined loser is the
    recorded proof that it did. What WOULD be a defect is a contradiction that
    left no authoritative winner, so that is the only condition reported as an
    error here -- the count itself is reported as observability.
    """
    index_path = _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("hook-authority-topology-conflict", "index.db not present")
    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("hook-authority-topology-conflict", f"could not open index.db: {exc}", exc=exc)
    try:
        if not table_exists(conn, "session_links"):
            return _skip_check("hook-authority-topology-conflict", "session_links table not present")
        contradicted_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM session_links WHERE method = ?",
                (HOOK_CONTRADICTED_LINK_METHOD,),
            ).fetchone()[0]
        )
        authoritative_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM session_links WHERE method = ?",
                (HOOK_AUTHORITATIVE_LINK_METHOD,),
            ).fetchone()[0]
        )
        # Every contradicted edge must have an authoritative sibling for the
        # same child: that sibling IS the resolution. A contradicted edge alone
        # means inference was overruled by nothing.
        unresolved_rows = conn.execute(
            """
            SELECT loser.src_session_id FROM session_links AS loser
            WHERE loser.method = ?
              AND NOT EXISTS (
                  SELECT 1 FROM session_links AS winner
                  WHERE winner.src_session_id = loser.src_session_id
                    AND winner.link_type = loser.link_type
                    AND winner.method = ?
              )
            LIMIT ?
            """,
            (HOOK_CONTRADICTED_LINK_METHOD, HOOK_AUTHORITATIVE_LINK_METHOD, sample_limit),
        ).fetchall()
        unresolved_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM session_links AS loser
                WHERE loser.method = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM session_links AS winner
                      WHERE winner.src_session_id = loser.src_session_id
                        AND winner.link_type = loser.link_type
                        AND winner.method = ?
                  )
                """,
                (HOOK_CONTRADICTED_LINK_METHOD, HOOK_AUTHORITATIVE_LINK_METHOD),
            ).fetchone()[0]
        )
        # "Some winner exists" is not the invariant -- "exactly one" is. A
        # revised hook claim lands at a DIFFERENT primary key, so without this
        # a child could carry two permanent authoritative edges and composition
        # would choose between them by arrival order, while a some-winner-exists
        # check reported OK. Zero such children exist on the live corpus today;
        # this is the latent case, gated before it can appear.
        multi_authoritative_rows = conn.execute(
            """
            SELECT src_session_id, link_type, COUNT(*) AS n
            FROM session_links
            WHERE method = ?
            GROUP BY src_session_id, link_type
            HAVING n > 1
            ORDER BY src_session_id
            LIMIT ?
            """,
            (HOOK_AUTHORITATIVE_LINK_METHOD, sample_limit),
        ).fetchall()
        multi_authoritative_count = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT src_session_id FROM session_links
                    WHERE method = ?
                    GROUP BY src_session_id, link_type
                    HAVING COUNT(*) > 1
                )
                """,
                (HOOK_AUTHORITATIVE_LINK_METHOD,),
            ).fetchone()[0]
        )
    except sqlite3.Error as exc:
        return _error_check("hook-authority-topology-conflict", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    problems: list[str] = []
    if unresolved_count:
        problems.append(f"contradicted edge without an authoritative winner x{unresolved_count}")
    if multi_authoritative_count:
        problems.append(f"child with more than one authoritative parent x{multi_authoritative_count}")
    status = OutcomeStatus.ERROR if problems else OutcomeStatus.OK
    summary = (
        "; ".join(problems)
        if problems
        else (
            f"{contradicted_count} hook/inference contradiction(s), each resolved to authoritative evidence; "
            f"{authoritative_count} authoritative edge(s)"
        )
    )
    return ArchiveVerificationCheck(
        name="hook-authority-topology-conflict",
        status=status,
        summary=summary,
        count=unresolved_count + multi_authoritative_count,
        evidence={
            "contradicted_count": contradicted_count,
            "authoritative_count": authoritative_count,
            "unresolved_contradiction_count": unresolved_count,
            "unresolved_contradiction_sample": [str(row[0]) for row in unresolved_rows],
            "multi_authoritative_count": multi_authoritative_count,
            "multi_authoritative_sample": [f"{row[0]}:{row[1]}={row[2]}" for row in multi_authoritative_rows],
        },
    )


# ---------------------------------------------------------------------------
# Check: enum-superset-CHECK (I2, polylogue-t0m73)
# ---------------------------------------------------------------------------

#: Column names that carry the ``Origin`` vocabulary and are generated via
#: ``check("origin", Origin)`` / ``check("dst_origin", Origin)`` at DDL-build
#: time (``archive_tiers/*.py``). Restricted to these two, word-boundary
#: anchored columns rather than every enum-backed column in the schema:
#: several other CHECK-constrained column names (``status``, ``kind``) are
#: reused across tables with *unrelated*, hand-written literal vocabularies,
#: so a blind column-name match would false-positive on those. ``origin`` is
#: the column this class of bug was actually found on live (2026-08-03:
#: ``Origin`` gained ``claude-design-session`` after several tables' DDL had
#: already been written to disk with the older literal list -- a CHECK
#: constraint is baked into the table at creation time and does not
#: retroactively track a later enum change).
_ORIGIN_CHECK_COLUMN_PATTERN = re.compile(
    r"[(,\s][\"'`\[]?(origin|dst_origin)[\"'`\]]?\s+TEXT[^,]*?CHECK\s*\([^)]*?\bIN\s*\(([^)]*)\)",
    re.IGNORECASE,
)


def _check_enum_superset(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    """Every live ``origin``/``dst_origin`` CHECK list is a superset of ``Origin``.

    A CHECK constraint is generated from :class:`polylogue.core.enums.Origin`
    at DDL-build time, but once a table exists on disk its CHECK text is
    frozen -- SQLite has no ``ALTER ... CHECK``. If ``Origin`` gains a member
    after a table was created (an additive-derived or additive-durable
    change that didn't trigger a full DDL rebuild for every affected table),
    the *live* archive's CHECK list silently falls behind the vocabulary
    production code can write, and the next insert with the new value either
    raises or -- if some path writes around it -- corrupts silently. This
    check reads ``sqlite_master`` on the live tiers and compares the CHECK
    text actually on disk against the *current* Python enum, so it is
    ground-truth against reality (the disk), not against the code that
    would generate a fresh table today.
    """
    from polylogue.core.enums import Origin

    # The generated archive-tier DDL is the canonical schema vocabulary.  Do
    # not maintain a second sqlite_master-derived list of expected values:
    # compare the live table declarations with the exact DDL that current
    # schema specs would create.
    canonical_ddls = {tier.value: ARCHIVE_TIER_SPECS[tier].ddl for tier in (ArchiveTier.SOURCE, ArchiveTier.INDEX)}
    origins = {o.value for o in Origin}
    bad: dict[str, Any] = {}
    examined_any = False
    for db_name in ("source.db", "index.db"):
        db_path = archive_root / db_name if db_name == "source.db" else _resolve_index_path(archive_root)
        if not db_path.exists():
            continue
        examined_any = True
        try:
            conn = _open_ro(db_path)
        except sqlite3.Error as exc:
            return _error_check("enum-superset-check", f"could not open {db_name}: {exc}", exc=exc)
        try:
            rows = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql LIKE '%origin%IN (%'"
            ).fetchall()
        except sqlite3.Error as exc:
            return _error_check("enum-superset-check", f"could not read {db_name}: {exc}", exc=exc)
        finally:
            conn.close()

        canonical_ddl = canonical_ddls[ArchiveTier.SOURCE.value if db_name == "source.db" else ArchiveTier.INDEX.value]
        for table_name, ddl in rows:
            for match in _ORIGIN_CHECK_COLUMN_PATTERN.finditer(ddl or ""):
                column, allowed_list = match.group(1), match.group(2)
                allowed = set(re.findall(r"'([^']*)'", allowed_list))
                canonical_matches = [
                    m
                    for m in _ORIGIN_CHECK_COLUMN_PATTERN.finditer(canonical_ddl)
                    if m.group(1).lower() == column.lower()
                ]
                canonical_allowed = (
                    set(re.findall(r"'([^']*)'", canonical_matches[0].group(2))) if canonical_matches else origins
                )
                missing = sorted(canonical_allowed - allowed)
                if missing:
                    bad[f"{db_name}:{table_name}.{column}"] = missing

    if not examined_any:
        return _skip_check("enum-superset-check", "neither source.db nor index.db present")
    if bad:
        return ArchiveVerificationCheck(
            name="enum-superset-check",
            status=OutcomeStatus.ERROR,
            summary=f"{len(bad)} CHECK list(s) missing current Origin member(s): {sorted(bad)}",
            count=len(bad),
            details=[f"{key}: missing {values}" for key, values in sorted(bad.items())],
            evidence={"missing_by_column": bad, "current_origin_vocabulary": sorted(origins)},
        )
    return ArchiveVerificationCheck(
        name="enum-superset-check",
        status=OutcomeStatus.OK,
        summary="every origin/dst_origin CHECK list on disk covers the current Origin enum",
        evidence={"current_origin_vocabulary": sorted(origins)},
    )


# ---------------------------------------------------------------------------
# Check: blob_refs join-liveness per ref_type (I3, polylogue-t0m73)
# ---------------------------------------------------------------------------

#: ``blob_refs.ref_type`` -> the source-tier table + PK column it must join
#: to. GC's oracle for "is this blob still referenced" is exactly this join
#: (a ``blob_refs`` row with no live referent is orphaned bookkeeping, not
#: proof the blob itself is unreferenced -- the join direction the schema
#: intends), so a ref_type this check doesn't recognize is itself a gap the
#: check surfaces via `_error_check`, not a silent skip.
_BLOB_REF_REFERENT_TABLES: dict[str, tuple[str, str]] = {
    ref_type: (referent_table, referent_column)
    for ref_type, referent_table, referent_column in validated_blob_ref_liveness_joins()
}


def _check_blob_refs_liveness(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Every ``blob_refs`` row resolves in its referent table for its ``ref_type``.

    ``blob_refs`` is the durable GC substrate: a blob is eligible for
    collection only once every referencing row is gone. An orphaned
    ``blob_refs`` row (its referent already deleted, e.g. by a cascading
    raw replace) is stale bookkeeping that either wastes retention (the blob
    looks referenced forever) or -- worse -- signals the delete path forgot
    to clean up ``blob_refs`` alongside the referent, which is a correctness
    bug in the GC liveness contract itself.
    """
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    if not source_path.exists():
        return _skip_check("blob-refs-liveness", "source.db not present")

    try:
        conn = _open_ro(source_path)
    except sqlite3.Error as exc:
        return _error_check("blob-refs-liveness", f"could not open source.db: {exc}", exc=exc)

    try:
        if not table_exists(conn, "blob_refs"):
            return _skip_check("blob-refs-liveness", "blob_refs table not present")

        try:
            ref_types = {str(row[0]) for row in conn.execute("SELECT DISTINCT ref_type FROM blob_refs")}
        except sqlite3.Error as exc:
            return _error_check("blob-refs-liveness", f"could not read blob_refs: {exc}", exc=exc)

        unknown_ref_types = sorted(ref_types - set(_BLOB_REF_REFERENT_TABLES))
        if unknown_ref_types:
            return _error_check(
                "blob-refs-liveness",
                f"blob_refs has ref_type(s) this check doesn't recognize: {unknown_ref_types} "
                "(update _BLOB_REF_REFERENT_TABLES)",
            )

        orphans_by_type: dict[str, int] = {}
        samples_by_type: dict[str, list[str]] = {}
        try:
            for ref_type, (referent_table, referent_column) in _BLOB_REF_REFERENT_TABLES.items():
                if not table_exists(conn, referent_table):
                    return _error_check(
                        "blob-refs-liveness",
                        f"referent table {referent_table!r} for ref_type={ref_type!r} does not exist",
                    )
                count = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM blob_refs b
                    WHERE b.ref_type = ? AND NOT EXISTS (
                        SELECT 1 FROM {referent_table} r WHERE r.{referent_column} = b.ref_id
                    )
                    """,
                    (ref_type,),
                ).fetchone()[0]
                orphans_by_type[ref_type] = int(count)
                if count:
                    samples_by_type[ref_type] = [
                        str(row[0])
                        for row in conn.execute(
                            f"""
                            SELECT DISTINCT b.ref_id FROM blob_refs b
                            WHERE b.ref_type = ? AND NOT EXISTS (
                                SELECT 1 FROM {referent_table} r WHERE r.{referent_column} = b.ref_id
                            )
                            LIMIT ?
                            """,
                            (ref_type, sample_limit),
                        )
                    ]
        except sqlite3.Error as exc:
            return _error_check("blob-refs-liveness", f"could not read source.db: {exc}", exc=exc)
    finally:
        conn.close()

    total_orphans = sum(orphans_by_type.values())
    status = OutcomeStatus.ERROR if total_orphans else OutcomeStatus.OK
    summary = (
        "; ".join(f"{ref_type} orphans={count:,}" for ref_type, count in orphans_by_type.items() if count)
        if total_orphans
        else "every blob_refs row resolves in its referent table"
    )
    return ArchiveVerificationCheck(
        name="blob-refs-liveness",
        status=status,
        summary=summary,
        count=total_orphans,
        details=[f"{ref_type}:{ref_id}" for ref_type, ids in samples_by_type.items() for ref_id in ids],
        evidence={"orphans_by_ref_type": orphans_by_type, "orphan_samples_by_ref_type": samples_by_type},
    )


def _check_blob_reference_closure_for_index(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Require acquired rows to retain their exact canonical references."""
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    if not source_path.exists() or not index_path.exists():
        return _skip_check("blob-reference-closure", "source.db or index.db not present")
    try:
        source_conn = _open_ro(source_path)
    except sqlite3.Error as exc:
        return _error_check("blob-reference-closure", f"could not open source/index tiers: {exc}", exc=exc)
    try:
        index_conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        source_conn.close()
        return _error_check("blob-reference-closure", f"could not open source/index tiers: {exc}", exc=exc)
    try:
        from polylogue.maintenance.blob_reference_closure import (
            closure_counts,
            raw_reference_closure_predicate,
        )

        counts = closure_counts(source_conn, index_conn)
        raw_sample = [
            str(row[0])
            for row in source_conn.execute(
                f"""
                SELECT r.raw_id FROM raw_sessions r
                WHERE {raw_reference_closure_predicate()}
                ORDER BY r.raw_id LIMIT ?
                """,
                (sample_limit,),
            )
        ]
        attachment_sample = [
            str(row[0])
            for row in index_conn.execute(
                """
                SELECT a.attachment_id FROM attachments a
                WHERE a.acquisition_status = 'acquired'
                  AND NOT EXISTS (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)
                ORDER BY a.attachment_id LIMIT ?
                """,
                (sample_limit,),
            )
        ]
    except sqlite3.Error as exc:
        return _error_check("blob-reference-closure", f"could not read source/index tiers: {exc}", exc=exc)
    finally:
        index_conn.close()
        source_conn.close()

    total = sum(counts.values())
    return ArchiveVerificationCheck(
        name="blob-reference-closure",
        status=OutcomeStatus.ERROR if total else OutcomeStatus.OK,
        summary=(
            f"{counts['raw_missing_exact_count']:,} raw session(s) and "
            f"{counts['acquired_attachment_missing_ref_count']:,} acquired attachment(s) lack canonical refs"
            if total
            else "every raw session and acquired attachment has canonical reference closure"
        ),
        count=total,
        details=[f"raw:{raw_id}" for raw_id in raw_sample]
        + [f"attachment:{attachment_id}" for attachment_id in attachment_sample],
        evidence={
            **counts,
            "raw_sample": raw_sample,
            "attachment_sample": attachment_sample,
        },
    )


def _check_blob_reference_closure(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_blob_reference_closure_for_index(archive_root, _resolve_index_path(archive_root), sample_limit)


def _check_blob_reference_closure_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    return _check_blob_reference_closure_for_index(archive_root, index_path, sample_limit)


# ---------------------------------------------------------------------------
# Check: raw failure lifecycle
# ---------------------------------------------------------------------------


def _check_raw_failure_lifecycle(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Require typed evidence for every retained raw parse/validation failure."""
    snapshot = read_raw_failure_lifecycle(_tier_path(archive_root, ArchiveTier.SOURCE), sample_limit=sample_limit)
    evidence = snapshot.to_dict()
    if not snapshot.available:
        return _error_check(
            "raw-failure-lifecycle",
            str(snapshot.reason or "raw failure lifecycle is unavailable"),
            evidence=evidence,
        )
    if snapshot.unexplained:
        return ArchiveVerificationCheck(
            name="raw-failure-lifecycle",
            status=OutcomeStatus.ERROR,
            summary=(
                f"{snapshot.unexplained:,} raw failure(s) lack typed deferred/terminal evidence; "
                "reindex is unsafe until each is classified"
            ),
            count=snapshot.unexplained,
            details=[
                f"{sample.get('origin', 'unknown')}:{sample.get('artifact_kind') or '<none>'}"
                for sample in snapshot.samples
                if sample.get("lifecycle") == "unexplained"
            ],
            evidence=evidence,
        )
    known = snapshot.deferred + snapshot.terminal
    # Terminal evidence is the completed state this invariant requires. Keep
    # it visible in counts/evidence, but do not emit a warning that strict
    # candidate acceptance interprets as a permanent veto. Deferred failures
    # remain warning-level because their source disposition is still open.
    status = OutcomeStatus.WARNING if snapshot.deferred else OutcomeStatus.OK
    summary = (
        f"{known:,} raw failure(s) classified ({snapshot.deferred:,} deferred, {snapshot.terminal:,} terminal)"
        if known
        else "no raw parse or validation failures"
    )
    return ArchiveVerificationCheck(
        name="raw-failure-lifecycle",
        status=status,
        summary=summary,
        count=known,
        evidence=evidence,
    )


def _check_raw_failure_lifecycle_at_candidate(
    archive_root: Path, _index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Run the source-only lifecycle gate for an inactive index candidate."""
    return _check_raw_failure_lifecycle(archive_root, sample_limit)


def _check_blob_refs_liveness_at_candidate(
    archive_root: Path, _index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Run the source-only blob check while a candidate index is selected."""
    return _check_blob_refs_liveness(archive_root, sample_limit)


def _check_blob_integrity(
    archive_root: Path,
    sample_limit: int,
    index_path: Path | None = None,
    *,
    active_index_context: Literal["required", "unavailable_for_candidate"] = "required",
) -> ArchiveVerificationCheck:
    """Require a complete physical scan of the durable blob namespace.

    The ordinary blob-health surface is intentionally bounded. Archive
    verification is the explicit point-in-time integrity receipt, so it uses
    the scanner's full traversal while retaining only bounded finding samples.
    Orphans are errors here as well as missing or corrupted referenced blobs:
    a clean/nonblocking archive must describe the physical namespace exactly.
    """
    scan_path = index_path or _resolve_index_path(archive_root)
    if not scan_path.exists():
        source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
        if source_path.exists():
            scan_path = source_path
        elif (archive_root / "blob").exists():
            return _error_check(
                "blob-integrity",
                "cannot verify the physical blob namespace without index.db or source.db reference evidence",
            )
        else:
            return _skip_check("blob-integrity", "no archive database or blob namespace present")
    try:
        report = scan_blob_integrity(
            scan_path,
            store=BlobStore(archive_root / "blob"),
            full=True,
            sample_size=sample_limit,
            configured_root=archive_root,
            active_index_context=active_index_context,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        return _error_check("blob-integrity", f"could not scan the physical blob namespace: {exc}", exc=exc)

    finding_counts = {finding.kind: finding.count for finding in report.findings}
    details = [f"{finding.kind}:{finding.count}" for finding in report.findings]
    return ArchiveVerificationCheck(
        name="blob-integrity",
        # The shared scanner labels aged orphans as warnings for ordinary
        # daemon health, but this point-in-time archive gate is the admission
        # boundary for rebuilds and promotion. Any physical finding therefore
        # blocks here; otherwise a candidate can pass while carrying bytes the
        # source authority cannot explain.
        status=OutcomeStatus.ERROR if report.findings else OutcomeStatus.OK,
        summary=(
            "; ".join(f"{kind}={count:,}" for kind, count in finding_counts.items())
            if finding_counts
            else f"full physical blob scan verified {report.scanned_blobs:,} blob(s) and {report.scanned_references:,} reference(s)"
        ),
        count=sum(finding_counts.values()),
        details=details,
        evidence={"scan": report.to_dict(), "finding_counts": finding_counts},
    )


def _check_blob_integrity_at_candidate(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Run the durable full blob scan against the candidate reference set."""
    return _check_blob_integrity(
        archive_root, sample_limit, index_path, active_index_context="unavailable_for_candidate"
    )


def _check_attachment_coverage(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_attachment_coverage_at_index_path(archive_root, _resolve_index_path(archive_root), sample_limit)


def _check_attachment_coverage_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Acquired attachment bytes must exist and remain reachable by a ref."""
    if not index_path.exists():
        return _skip_check("attachment-coverage", "index.db not present")
    try:
        report = scan_attachment_coverage(
            index_path,
            store=BlobStore(archive_root / "blob"),
            sample_size=sample_limit,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        return _error_check("attachment-coverage", f"could not scan attachment coverage: {exc}", exc=exc)

    missing = report.acquired_missing_blob_count
    unreachable = report.acquired_unreachable_count
    debt_count = missing + unreachable
    details = [
        *(f"acquired-missing-blob:{attachment_id}" for attachment_id in report.acquired_missing_blob_sample),
        *(f"acquired-unreachable:{attachment_id}" for attachment_id in report.acquired_unreachable_sample),
    ]
    return ArchiveVerificationCheck(
        name="attachment-coverage",
        status=OutcomeStatus.ERROR if debt_count else OutcomeStatus.OK,
        summary=(
            f"acquired attachment debt: missing_blob={missing:,}, unreachable={unreachable:,}"
            if debt_count
            else f"all {report.acquired_count:,} acquired attachment(s) have bytes and a live attachment reference"
        ),
        count=debt_count,
        details=details,
        evidence={"scan": report.to_dict(), "missing_blob_count": missing, "unreachable_count": unreachable},
    )


def _check_attachment_coverage_at_candidate(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    return _check_attachment_coverage_at_index_path(archive_root, index_path, sample_limit)


# ---------------------------------------------------------------------------
# Check: embeddings refs resolve in the index tier (I4, polylogue-t0m73)
# ---------------------------------------------------------------------------


def _check_embeddings_refs_liveness(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_embeddings_refs_liveness_at_index_path(archive_root, _resolve_index_path(archive_root), sample_limit)


def _check_embeddings_refs_liveness_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Every ``message_embedding_refs`` row resolves to a live index message.

    embeddings.db is a rebuildable derived tier keyed off index.db's message
    ids; an embedding ref outliving the message it was computed for (e.g. a
    session full-replace deleting messages without the embedding catch-up
    stage draining the corresponding refs) is exactly the feu0-class bug:
    dead vector rows that read back as "embedded" for content that no
    longer exists in the index.
    """
    embeddings_path = _tier_path(archive_root, ArchiveTier.EMBEDDINGS)
    if not embeddings_path.exists():
        return _skip_check("embeddings-refs-liveness", "embeddings.db not present")
    if not index_path.exists():
        return _skip_check("embeddings-refs-liveness", "index.db not present")

    try:
        conn = _open_ro(embeddings_path)
    except sqlite3.Error as exc:
        return _error_check("embeddings-refs-liveness", f"could not open embeddings.db: {exc}", exc=exc)

    try:
        if not table_exists(conn, "message_embedding_refs"):
            return _skip_check("embeddings-refs-liveness", "message_embedding_refs table not present")

        try:
            conn.execute("ATTACH DATABASE ? AS idx_tier", (f"file:{index_path}?mode=ro",))
        except sqlite3.Error as exc:
            return _error_check("embeddings-refs-liveness", f"could not attach index.db: {exc}", exc=exc)

        try:
            total, orphans = conn.execute(
                """
                SELECT COUNT(*), SUM(
                    CASE WHEN NOT EXISTS (
                        SELECT 1 FROM idx_tier.messages m WHERE m.message_id = r.message_id
                    ) THEN 1 ELSE 0 END
                )
                FROM message_embedding_refs r
                """
            ).fetchone()
            orphans = int(orphans or 0)
            orphan_sample = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT r.message_id FROM message_embedding_refs r
                    WHERE NOT EXISTS (SELECT 1 FROM idx_tier.messages m WHERE m.message_id = r.message_id)
                    LIMIT ?
                    """,
                    (sample_limit,),
                )
            ]
        except sqlite3.Error as exc:
            return _error_check("embeddings-refs-liveness", f"could not read embeddings/index tiers: {exc}", exc=exc)
    finally:
        conn.close()

    status = OutcomeStatus.ERROR if orphans else OutcomeStatus.OK
    return ArchiveVerificationCheck(
        name="embeddings-refs-liveness",
        status=status,
        summary=(
            f"{int(total or 0):,} embedding ref(s), {orphans:,} orphaned (no live index message)"
            if orphans
            else f"all {int(total or 0):,} embedding ref(s) resolve to a live index message"
        ),
        count=orphans,
        details=[f"orphan:{message_id}" for message_id in orphan_sample],
        evidence={"ref_count": int(total or 0), "orphan_count": orphans, "orphan_sample": orphan_sample},
    )


def _check_embeddings_refs_liveness_at_candidate(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    return _check_embeddings_refs_liveness_at_index_path(archive_root, index_path, sample_limit)


# ---------------------------------------------------------------------------
# Check: session parent-chain acyclicity (I5, polylogue-t0m73)
# ---------------------------------------------------------------------------


def _check_session_lineage_acyclic(
    archive_root: Path, _sample_limit: int, *, index_path: Path | None = None
) -> ArchiveVerificationCheck:
    """``sessions.parent_session_id`` never closes a cycle.

    ``lineage-sanity`` (above) already covers ``session_links``' dangling
    ``resolved_dst_session_id`` / ``branch_point_message_id`` references --
    this check adds the one I5 sub-invariant that has no existing coverage:
    the derived ``parent_session_id`` chain resolvers walk to recompose a
    session's inherited prefix must terminate. A cycle here would make
    prefix recomposition (see the lineage-normalization doc in
    ``storage/sqlite/archive_tiers/write.py``) loop forever or silently
    truncate depending on the walker's own defenses -- this check proves
    the graph itself is acyclic independent of any particular walker's
    resilience to being handed a cycle.
    """
    index_path = index_path or _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("session-lineage-acyclic", "index.db not present")

    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("session-lineage-acyclic", f"could not open index.db: {exc}", exc=exc)

    try:
        try:
            parents, cycle_members = _lineage_parent_graph(conn)
        except sqlite3.Error as exc:
            return _error_check("session-lineage-acyclic", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    status = OutcomeStatus.ERROR if cycle_members else OutcomeStatus.OK
    return ArchiveVerificationCheck(
        name="session-lineage-acyclic",
        status=status,
        summary=(
            f"{len(cycle_members)} session(s) on a parent_session_id cycle"
            if cycle_members
            else f"{len(parents):,} parent-linked session(s), parent chain is acyclic"
        ),
        count=len(cycle_members),
        details=sorted(cycle_members)[:10],
        evidence={"linked_session_count": len(parents), "cycle_member_sample": sorted(cycle_members)[:10]},
    )


# ---------------------------------------------------------------------------
# Check: sessions.message_count projection drift (I8, polylogue-t0m73)
# ---------------------------------------------------------------------------


def _check_message_count_projection(
    archive_root: Path, sample_limit: int, *, index_path: Path | None = None
) -> ArchiveVerificationCheck:
    """``sessions.message_count`` matches the actual materialized row count.

    ``message_count`` is a write-time projection (set when a session is
    written, not a generated column), so it can drift from the underlying
    ``messages`` rows if a partial write, a bug in the write path, or a
    manual repair touches one but not the other. Ground truth is
    ``COUNT(*)`` over ``messages`` itself, joined back to ``sessions``.
    """
    index_path = index_path or _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("message-count-projection", "index.db not present")

    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("message-count-projection", f"could not open index.db: {exc}", exc=exc)

    try:
        try:
            drift_count = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT s.session_id FROM sessions s
                    LEFT JOIN (SELECT session_id, COUNT(*) AS n FROM messages GROUP BY session_id) m
                        ON m.session_id = s.session_id
                    WHERE COALESCE(m.n, 0) != s.message_count
                )
                """
            ).fetchone()[0]
            drift_sample = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT s.session_id FROM sessions s
                    LEFT JOIN (SELECT session_id, COUNT(*) AS n FROM messages GROUP BY session_id) m
                        ON m.session_id = s.session_id
                    WHERE COALESCE(m.n, 0) != s.message_count
                    LIMIT ?
                    """,
                    (sample_limit,),
                )
            ]
        except sqlite3.Error as exc:
            return _error_check("message-count-projection", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    status = OutcomeStatus.ERROR if drift_count else OutcomeStatus.OK
    return ArchiveVerificationCheck(
        name="message-count-projection",
        status=status,
        summary=(
            f"{drift_count} session(s) with drifted message_count"
            if drift_count
            else "sessions.message_count matches COUNT(messages) for every session"
        ),
        count=drift_count,
        details=[f"session:{session_id}" for session_id in drift_sample],
        evidence={"drifted_session_count": drift_count, "drifted_session_sample": drift_sample},
    )


# ---------------------------------------------------------------------------
# Check: parser/lowering semantic stamp coverage (polylogue-xselt)
# ---------------------------------------------------------------------------


def _check_session_fingerprint_stamps(
    archive_root: Path, sample_limit: int, *, index_path: Path | None = None
) -> ArchiveVerificationCheck:
    """Require every candidate session to carry current, unmixed semantic stamps.

    ``sessions`` is the ground-truth candidate universe. The check compares
    each row to the current source-derived stamp rather than accepting a
    merely well-formed historical value, so a parser/lowering semantic change
    cannot promote a stale reindex candidate.
    """
    index_path = index_path or _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("session-fingerprint-stamps", "index.db not present")
    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("session-fingerprint-stamps", f"could not open index.db: {exc}", exc=exc)

    try:
        required_columns = {"parser_fingerprint", "lowering_fingerprint"}
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sessions)")}
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            return _error_check(
                "session-fingerprint-stamps",
                f"sessions is missing fingerprint column(s): {', '.join(missing_columns)}",
            )

        total = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        invalid_sql = "{column} IS NULL OR length({column}) != 64 OR {column} GLOB '*[^0-9a-f]*'"
        invalid_parser_count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE {invalid_sql.format(column='parser_fingerprint')}"
            ).fetchone()[0]
        )
        invalid_lowering_count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE {invalid_sql.format(column='lowering_fingerprint')}"
            ).fetchone()[0]
        )
        parser_rows = [
            (str(row[0]), str(row[1]), int(row[2]))
            for row in conn.execute(
                """
                SELECT origin, parser_fingerprint, COUNT(*)
                FROM sessions
                WHERE parser_fingerprint IS NOT NULL
                  AND length(parser_fingerprint) = 64
                  AND parser_fingerprint NOT GLOB '*[^0-9a-f]*'
                GROUP BY origin, parser_fingerprint
                ORDER BY origin, parser_fingerprint
                """
            )
        ]
        lowering_rows = [
            (str(row[0]), int(row[1]))
            for row in conn.execute(
                """
                SELECT lowering_fingerprint, COUNT(*)
                FROM sessions
                WHERE lowering_fingerprint IS NOT NULL
                  AND length(lowering_fingerprint) = 64
                  AND lowering_fingerprint NOT GLOB '*[^0-9a-f]*'
                GROUP BY lowering_fingerprint
                ORDER BY lowering_fingerprint
                """
            )
        ]
        session_stamp_rows = [
            (str(row[0]), row[1], row[2])
            for row in conn.execute("SELECT origin, parser_fingerprint, lowering_fingerprint FROM sessions")
        ]
        invalid_sample = [
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT session_id FROM sessions
                WHERE {invalid_sql.format(column="parser_fingerprint")}
                   OR {invalid_sql.format(column="lowering_fingerprint")}
                ORDER BY session_id
                LIMIT ?
                """,
                (sample_limit,),
            )
        ]
    except sqlite3.Error as exc:
        return _error_check("session-fingerprint-stamps", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    expected_parser_by_origin: dict[str, str] = {}
    parser_values_by_origin: dict[str, dict[str, int]] = {}
    for origin, fingerprint, count in parser_rows:
        parser_values_by_origin.setdefault(origin, {})[fingerprint] = count
    parser_stale_by_origin: dict[str, int] = {}
    parser_mixed_by_origin: dict[str, int] = {}
    for origin, values in parser_values_by_origin.items():
        try:
            expected = parser_fingerprint_for_origin(origin)
        except ValueError:
            parser_stale_by_origin[origin] = sum(values.values())
            continue
        expected_parser_by_origin[origin] = expected
        stale_count = sum(count for fingerprint, count in values.items() if fingerprint != expected)
        if stale_count:
            parser_stale_by_origin[origin] = stale_count
        if len(values) != 1:
            parser_mixed_by_origin[origin] = len(values)

    expected_lowering = lowering_fingerprint()
    lowering_stale_count = sum(count for fingerprint, count in lowering_rows if fingerprint != expected_lowering)
    lowering_distinct_count = len(lowering_rows)
    fully_current_count = sum(
        expected_parser_by_origin.get(origin) is not None
        and parser_fingerprint == expected_parser_by_origin[origin]
        and lowering_stamp == expected_lowering
        for origin, parser_fingerprint, lowering_stamp in session_stamp_rows
    )
    problems: list[str] = []
    if invalid_parser_count:
        problems.append(f"missing/invalid parser stamps={invalid_parser_count:,}")
    if invalid_lowering_count:
        problems.append(f"missing/invalid lowering stamps={invalid_lowering_count:,}")
    if parser_stale_by_origin:
        problems.append(f"stale parser stamps={parser_stale_by_origin}")
    if lowering_stale_count:
        problems.append(f"stale lowering stamps={lowering_stale_count:,}")
    if parser_mixed_by_origin:
        problems.append(f"mixed parser stamps={parser_mixed_by_origin}")
    if total and lowering_distinct_count != 1:
        problems.append(f"mixed lowering stamps={lowering_distinct_count}")

    return ArchiveVerificationCheck(
        name="session-fingerprint-stamps",
        status=OutcomeStatus.ERROR if problems else OutcomeStatus.OK,
        summary=("; ".join(problems) if problems else f"{total:,} session(s) carry current semantic stamps"),
        count=len(problems),
        details=[f"invalid:{session_id}" for session_id in invalid_sample],
        evidence={
            "session_count": total,
            "fully_current_session_count": fully_current_count,
            "invalid_parser_stamp_count": invalid_parser_count,
            "invalid_lowering_stamp_count": invalid_lowering_count,
            "expected_parser_by_origin": expected_parser_by_origin,
            "parser_values_by_origin": parser_values_by_origin,
            "parser_stale_by_origin": parser_stale_by_origin,
            "parser_mixed_by_origin": parser_mixed_by_origin,
            "expected_lowering_fingerprint": expected_lowering,
            "lowering_values": dict(lowering_rows),
            "lowering_stale_count": lowering_stale_count,
            "lowering_distinct_count": lowering_distinct_count,
            "invalid_sample": invalid_sample,
        },
    )


# ---------------------------------------------------------------------------
# Check 6: planner stats presence (polylogue-l3tk class)
# ---------------------------------------------------------------------------


def _check_planner_stats(
    archive_root: Path, _sample_limit: int, *, index_path: Path | None = None
) -> ArchiveVerificationCheck:
    index_path = index_path or _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("planner-stats", "index.db not present")

    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("planner-stats", f"could not open index.db: {exc}", exc=exc)

    try:
        if not table_exists(conn, "sqlite_stat1"):
            return ArchiveVerificationCheck(
                name="planner-stats",
                status=OutcomeStatus.WARNING,
                summary=(
                    "sqlite_stat1 is absent -- run ANALYZE before heavy replay/query load "
                    "(polylogue-l3tk class: unanalyzed fresh generations pick pathological plans)"
                ),
                count=1,
                evidence={"covered_tables": [], "missing_tables": list(_PLANNER_STATS_COVERED_TABLES)},
            )
        placeholders = ",".join("?" for _ in _PLANNER_STATS_COVERED_TABLES)
        analyzed = {
            str(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT tbl FROM sqlite_stat1 WHERE tbl IN ({placeholders})",
                _PLANNER_STATS_COVERED_TABLES,
            )
        }
    except sqlite3.Error as exc:
        return _error_check("planner-stats", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    missing = [table for table in _PLANNER_STATS_COVERED_TABLES if table not in analyzed]
    if missing:
        return ArchiveVerificationCheck(
            name="planner-stats",
            status=OutcomeStatus.WARNING,
            summary=(f"sqlite_stat1 missing coverage for: {', '.join(missing)} (polylogue-l3tk class)"),
            count=len(missing),
            evidence={"covered_tables": sorted(analyzed), "missing_tables": missing},
        )
    return ArchiveVerificationCheck(
        name="planner-stats",
        status=OutcomeStatus.OK,
        summary="sqlite_stat1 covers blocks/messages/action_pairs",
        evidence={"covered_tables": sorted(analyzed), "missing_tables": []},
    )


# ---------------------------------------------------------------------------
# Check: excluded-cursor vocabulary honesty (polylogue-ix5r)
# ---------------------------------------------------------------------------


def _check_excluded_cursor_vocabulary_honesty(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """An excluded (quarantined) cursor row must never carry a live ``next_retry_at``.

    Exclusion is permanent-until-file-replaced (or, since polylogue-ix5r, until
    the responsible parser's fingerprint changes) -- it never retries on a
    schedule. A row that is both ``excluded=1`` and carries a non-NULL
    ``next_retry_at`` is exactly the mislabeling shape the 2026-07-31 audit
    found in ``polylogued status`` output (excluded rows folded into a
    "retry due" count that would never actually come due for them): any
    consumer that queries "failing rows with a due retry timer" without
    remembering to filter ``excluded = 0`` would misreport this row as
    retryable. The production actuators (``mark_failed``'s exclusion branch)
    already null ``next_retry_at`` when they set ``excluded=1``; this check
    is the standing regression guard that invariant holds archive-wide, plus
    it surfaces the excluded population's size and oldest age so an operator
    can see at a glance how large and how stale the permanently-parked set
    is (previously only visible via ``polylogue ops status`` truncated to a
    50-row sample).
    """
    ops_path = _tier_path(archive_root, ArchiveTier.OPS)
    if not ops_path.exists():
        return _skip_check("excluded-cursor-vocabulary-honesty", "ops.db not present")

    try:
        conn = _open_ro(ops_path)
    except sqlite3.Error as exc:
        return _error_check("excluded-cursor-vocabulary-honesty", f"could not open ops.db: {exc}", exc=exc)

    try:
        if not table_exists(conn, "ingest_cursor"):
            return _skip_check("excluded-cursor-vocabulary-honesty", "ingest_cursor table not present")
        excluded_count = int(conn.execute("SELECT COUNT(*) FROM ingest_cursor WHERE excluded = 1").fetchone()[0])
        oldest_row = conn.execute("SELECT MIN(updated_at_ms) FROM ingest_cursor WHERE excluded = 1").fetchone()
        mislabeled_rows = conn.execute(
            """
            SELECT source_path FROM ingest_cursor
            WHERE excluded = 1 AND next_retry_at IS NOT NULL
            ORDER BY source_path
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
        mislabeled_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM ingest_cursor WHERE excluded = 1 AND next_retry_at IS NOT NULL"
            ).fetchone()[0]
        )
    except sqlite3.Error as exc:
        return _error_check("excluded-cursor-vocabulary-honesty", f"could not read ops.db: {exc}", exc=exc)
    finally:
        conn.close()

    oldest_age_s = None
    if oldest_row is not None and oldest_row[0] is not None:
        oldest_age_s = max(0.0, datetime.now(UTC).timestamp() - int(oldest_row[0]) / 1000.0)

    evidence: dict[str, Any] = {
        "excluded_count": excluded_count,
        "excluded_oldest_age_s": oldest_age_s,
        "mislabeled_count": mislabeled_count,
    }
    if mislabeled_count:
        return ArchiveVerificationCheck(
            name="excluded-cursor-vocabulary-honesty",
            status=OutcomeStatus.ERROR,
            summary=(
                f"{mislabeled_count:,} excluded cursor row(s) still carry a next_retry_at "
                "(would misreport as retry-due)"
            ),
            count=mislabeled_count,
            details=[str(row[0]) for row in mislabeled_rows],
            evidence=evidence,
        )
    return ArchiveVerificationCheck(
        name="excluded-cursor-vocabulary-honesty",
        status=OutcomeStatus.OK,
        summary=(
            f"{excluded_count:,} excluded cursor(s), none mislabeled as retry-due"
            + (f", oldest excluded {oldest_age_s / 3600.0:.1f}h ago" if oldest_age_s is not None else "")
        ),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Checks: corpus acceptance measures (polylogue-f1vg)
# ---------------------------------------------------------------------------


def _check_corpus_absences(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_corpus_absences_at_index_path(archive_root, _resolve_index_path(archive_root), sample_limit)


def _check_corpus_absences_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    if not source_path.exists() or not index_path.exists():
        return _skip_check("corpus-absences", "source.db or index.db not present")
    try:
        source = _open_ro(source_path)
        index = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("corpus-absences", f"could not open source/index tiers: {exc}", exc=exc)
    try:
        evidence = audit_absences(source, index, sample_limit=sample_limit)
    except sqlite3.Error as exc:
        return _error_check("corpus-absences", f"could not read source/index tiers: {exc}", exc=exc)
    finally:
        index.close()
        source.close()
    count = int(evidence["absent_total"]) + int(evidence["raws_without_attributable_identity"])
    details = [item for values in evidence["samples"].values() for item in values]
    details.extend(evidence["unattributable_sample"])
    return ArchiveVerificationCheck(
        name="corpus-absences",
        status=OutcomeStatus.ERROR if count else OutcomeStatus.OK,
        summary=(
            f"{evidence['absent_total']:,} absent document(s), "
            f"{evidence['raws_without_attributable_identity']:,} raw(s) without attributable identity"
            if count
            else f"all {evidence['documents_known']:,} source-backed document(s) are indexed"
        ),
        count=count,
        details=details[:sample_limit],
        evidence=evidence,
    )


def _check_corpus_attachment_fidelity(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    return _check_corpus_attachment_fidelity_at_index_path(
        archive_root, _resolve_index_path(archive_root), _sample_limit
    )


def _check_corpus_attachment_fidelity_at_index_path(
    _archive_root: Path, index_path: Path, _sample_limit: int
) -> ArchiveVerificationCheck:
    if not index_path.exists():
        return _skip_check("corpus-attachment-fidelity", "index.db not present")
    try:
        index = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("corpus-attachment-fidelity", f"could not open index.db: {exc}", exc=exc)
    try:
        evidence = audit_attachment_fidelity(index)
    except sqlite3.Error as exc:
        return _error_check("corpus-attachment-fidelity", f"could not read index.db: {exc}", exc=exc)
    finally:
        index.close()
    unfetched = int(evidence["refs_unfetched"])
    unprovenanced_unavailable = int(evidence["refs_unavailable_without_provenance"])
    blocking_count = unfetched + unprovenanced_unavailable
    return ArchiveVerificationCheck(
        name="corpus-attachment-fidelity",
        status=OutcomeStatus.ERROR if blocking_count else OutcomeStatus.OK,
        summary=(
            f"{unfetched:,} attachment reference(s) remain unfetched"
            if unfetched
            else f"{unprovenanced_unavailable:,} unavailable attachment reference(s) lack structured provenance"
            if unprovenanced_unavailable
            else f"all attachment references are acquired or typed unavailable ({evidence['refs_unavailable']:,} unavailable)"
        ),
        count=blocking_count,
        evidence=evidence,
    )


def _check_corpus_revision_fidelity(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_corpus_revision_fidelity_at_index_path(archive_root, _resolve_index_path(archive_root), sample_limit)


def _check_corpus_revision_fidelity_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    if not source_path.exists() or not index_path.exists():
        return _skip_check("corpus-revision-fidelity", "source.db or index.db not present")
    try:
        source = _open_ro(source_path)
        index = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("corpus-revision-fidelity", f"could not open source/index tiers: {exc}", exc=exc)
    try:
        evidence = audit_revision_fidelity(source, index, sample_limit=sample_limit)
    except sqlite3.Error as exc:
        return _error_check("corpus-revision-fidelity", f"could not read source/index tiers: {exc}", exc=exc)
    finally:
        index.close()
        source.close()
    shortfall = int(evidence["unexplained_shortfall"])
    return ArchiveVerificationCheck(
        name="corpus-revision-fidelity",
        status=OutcomeStatus.ERROR if shortfall else OutcomeStatus.OK,
        summary=(
            f"{shortfall:,} document(s) below best recorded evidence"
            if shortfall
            else f"indexed evidence meets every comparable recorded revision ({evidence['explained_by_event_reclassification']:,} event-reclassified)"
        ),
        count=shortfall,
        details=[str(item["session_id"]) for item in evidence["worst"]],
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Check: stalled append-cursor freshness (polylogue-2qrx)
# ---------------------------------------------------------------------------

#: How long a non-excluded cursor may sit with ``byte_offset < stat_size``
#: (content on disk, fully acquirable, but not yet caught up) before it is
#: flagged stale rather than merely "still catching up". Mirrors the
#: escalation age ``sources/live/watcher.py``'s
#: ``_STUCK_DEFERRED_APPEND_AGE_S`` already uses for the deferred-tail
#: sub-case of this same shape (polylogue-2qrx's root-caused stall
#: mechanism, PR #3650) -- reusing the same value here rather than picking a
#: second, undocumented threshold.
_STALLED_APPEND_CURSOR_STALE_AGE_S = 60.0 * 60.0


def _check_stalled_append_cursor_freshness(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Flag non-excluded cursors stuck behind their file's current size.

    A cursor with ``byte_offset < stat_size`` has content sitting on disk
    that is fully recoverable -- but only via ordinary acquisition catch-up;
    an index rebuild replays ``source.db`` and recovers none of it, since it
    was never acquired in the first place (polylogue-2qrx). A cursor briefly
    behind its file's size is normal (a writer mid-append); one behind for
    longer than :data:`_STALLED_APPEND_CURSOR_STALE_AGE_S` with the file
    otherwise cold is exactly the "329h stale, 94.8MB behind" shape the
    2026-07-31 audit found with zero durable signal -- this check is that
    signal.
    """
    ops_path = _tier_path(archive_root, ArchiveTier.OPS)
    if not ops_path.exists():
        return _skip_check("stalled-append-cursor-freshness", "ops.db not present")

    try:
        conn = _open_ro(ops_path)
    except sqlite3.Error as exc:
        return _error_check("stalled-append-cursor-freshness", f"could not open ops.db: {exc}", exc=exc)

    try:
        if not table_exists(conn, "ingest_cursor"):
            return _skip_check("stalled-append-cursor-freshness", "ingest_cursor table not present")
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        stale_before_ms = now_ms - int(_STALLED_APPEND_CURSOR_STALE_AGE_S * 1000)
        rows = conn.execute(
            """
            SELECT source_path, stat_size, byte_offset, updated_at_ms
            FROM ingest_cursor
            WHERE excluded = 0
              AND byte_offset IS NOT NULL AND stat_size IS NOT NULL
              AND byte_offset < stat_size
              AND updated_at_ms <= ?
            ORDER BY (stat_size - byte_offset) DESC
            """,
            (stale_before_ms,),
        ).fetchall()
    except sqlite3.Error as exc:
        return _error_check("stalled-append-cursor-freshness", f"could not read ops.db: {exc}", exc=exc)
    finally:
        conn.close()

    stalled_count = len(rows)
    total_lag_bytes = sum(int(row[1]) - int(row[2]) for row in rows)
    oldest_age_s = max((now_ms - int(row[3]) for row in rows), default=0) / 1000.0
    samples = [
        {
            "source_path": str(row[0]),
            "lag_bytes": int(row[1]) - int(row[2]),
            "stale_age_s": (now_ms - int(row[3])) / 1000.0,
        }
        for row in rows[:sample_limit]
    ]
    evidence: dict[str, Any] = {
        "stalled_count": stalled_count,
        "total_lag_bytes": total_lag_bytes,
        "oldest_stale_age_s": oldest_age_s if stalled_count else None,
        "stale_age_threshold_s": _STALLED_APPEND_CURSOR_STALE_AGE_S,
        "samples": samples,
    }
    if stalled_count:
        return ArchiveVerificationCheck(
            name="stalled-append-cursor-freshness",
            status=OutcomeStatus.WARNING,
            summary=(
                f"{stalled_count:,} cursor(s) stalled behind their file's current size "
                f"({total_lag_bytes:,} bytes lag, oldest {oldest_age_s / 3600.0:.1f}h)"
            ),
            count=stalled_count,
            details=[str(sample["source_path"]) for sample in samples],
            evidence=evidence,
        )
    return ArchiveVerificationCheck(
        name="stalled-append-cursor-freshness",
        status=OutcomeStatus.OK,
        summary="no non-excluded cursor is stalled behind its file's current size",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Check 7: convergence freshness (I6, polylogue-t0m73)
# ---------------------------------------------------------------------------

#: Window within which *some* daemon/convergence activity must have been
#: observed for an open unindexed backlog to read as "being worked" rather
#: than stalled. 24h matches the daemon's own quiet-window/catch-up cadence
#: (see ``daemon/convergence_stages.py``) -- shorter than that flags normal
#: async lag as false-positive stalls; much longer hides a genuinely dead
#: converger for days.
CONVERGENCE_FRESHNESS_WINDOW_MS = 24 * 3600 * 1000


def _unindexed_backlog_gap(conn: sqlite3.Connection) -> int:
    """Count of ``raw_sessions`` logical heads with no indexed materialization
    and no terminal typed refusal (parse_error / declared non-session) --
    the same universe :func:`_check_source_index_coverage` (I1) tallies as
    ``untyped_count + quarantined_count``, recomputed here so I6 doesn't
    need I1's full breakdown/sample evidence, only the scalar gap.
    """
    has_census = table_exists(conn, "raw_membership_census")
    census_expr = "(SELECT c.status FROM raw_membership_census c WHERE c.raw_id = r.raw_id)" if has_census else "NULL"
    valid_supersession_expr = _valid_byte_duplicate_supersession_expr(conn, raw_alias="r")
    logical_cohort_expr = _logical_head_cohort_expr(conn, raw_alias="r")
    row = conn.execute(
        f"""
        WITH heads AS (
            SELECT
                r.raw_id,
                r.parse_error,
                r.logical_source_key,
                {census_expr} AS census_status,
                {valid_supersession_expr} AS valid_supersession,
                MAX(EXISTS(SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = r.raw_id))
                    OVER (
                        PARTITION BY r.origin,
                                     {logical_cohort_expr}
                    ) AS any_indexed,
                ROW_NUMBER() OVER (
                    PARTITION BY r.origin,
                                 {logical_cohort_expr}
                    ORDER BY r.acquired_at_ms DESC, r.raw_id DESC
                ) AS rn
            FROM raw_sessions r
        )
        SELECT SUM(
            CASE WHEN rn = 1 AND any_indexed = 0 AND parse_error IS NULL
                      AND COALESCE(census_status, '') NOT IN ('non_session', 'failed')
                      AND valid_supersession = 0
                 THEN 1 ELSE 0 END
        )
        FROM heads WHERE rn = 1
        """
    ).fetchone()
    return int(row[0] or 0)


def _check_convergence_freshness(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    """An open unindexed backlog is only benign async lag if convergence is
    actively working it (I6, polylogue-t0m73).

    The prototype's original I6 criterion ("some daemon activity happened
    ever") was too weak: it passed on an archive with a 7,200-source gap
    and zero daemon stage events in the preceding 24h, because activity from
    weeks earlier still counted. The corrected criterion requires BOTH a
    non-zero gap (reusing I1's universe via :func:`_unindexed_backlog_gap`)
    AND *no* convergence activity inside :data:`CONVERGENCE_FRESHNESS_WINDOW_MS`
    across ``daemon_events``, ``daemon_stage_events``, and ``convergence_debt``
    (the last of these was itself missed by the original predicate --
    a debt row's own ``updated_at_ms`` is evidence of retry activity even
    when no daemon_events/stage_events row was emitted in the same window).
    A gap with recent activity is WARNING (backlog exists, still converging);
    a gap with no recent activity anywhere is ERROR (stalled, not lag).
    """
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    index_path = _resolve_index_path(archive_root)
    ops_path = _tier_path(archive_root, ArchiveTier.OPS)
    if not source_path.exists() or not index_path.exists():
        return _skip_check("convergence-freshness", "source.db or index.db not present")
    if not ops_path.exists():
        return _skip_check("convergence-freshness", "ops.db not present")

    try:
        conn = _open_ro(source_path)
    except sqlite3.Error as exc:
        return _error_check("convergence-freshness", f"could not open source.db: {exc}", exc=exc)
    try:
        try:
            conn.execute("ATTACH DATABASE ? AS idx_tier", (f"file:{index_path}?mode=ro",))
        except sqlite3.Error as exc:
            return _error_check("convergence-freshness", f"could not attach index.db: {exc}", exc=exc)
        try:
            gap = _unindexed_backlog_gap(conn)
        except sqlite3.Error as exc:
            return _error_check("convergence-freshness", f"could not read source/index tiers: {exc}", exc=exc)
    finally:
        conn.close()

    if gap == 0:
        return ArchiveVerificationCheck(
            name="convergence-freshness",
            status=OutcomeStatus.OK,
            summary="no unindexed backlog to converge",
            evidence={"unindexed_backlog_gap": 0},
        )

    try:
        ops_conn = _open_ro(ops_path)
    except sqlite3.Error as exc:
        return _error_check("convergence-freshness", f"could not open ops.db: {exc}", exc=exc)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    activity: dict[str, Any] = {}
    most_recent_ms: int | None = None
    try:
        if table_exists(ops_conn, "daemon_events"):
            row = ops_conn.execute("SELECT MAX(ts_ms) FROM daemon_events").fetchone()
            if row and row[0] is not None:
                activity["daemon_events_latest_ms"] = int(row[0])
                most_recent_ms = max(most_recent_ms or int(row[0]), int(row[0]))
        if table_exists(ops_conn, "daemon_stage_events"):
            row = ops_conn.execute("SELECT MAX(observed_at_ms) FROM daemon_stage_events").fetchone()
            if row and row[0] is not None:
                activity["daemon_stage_events_latest_ms"] = int(row[0])
                most_recent_ms = max(most_recent_ms or int(row[0]), int(row[0]))
        if table_exists(ops_conn, "convergence_debt"):
            row = ops_conn.execute("SELECT MAX(updated_at_ms), COUNT(*) FROM convergence_debt").fetchone()
            activity["convergence_debt_count"] = int(row[1] or 0) if row else 0
            if row and row[0] is not None:
                activity["convergence_debt_latest_ms"] = int(row[0])
                most_recent_ms = max(most_recent_ms or int(row[0]), int(row[0]))
    except sqlite3.Error as exc:
        return _error_check("convergence-freshness", f"could not read ops.db: {exc}", exc=exc)
    finally:
        ops_conn.close()

    age_ms = None if most_recent_ms is None else now_ms - most_recent_ms
    recent = age_ms is not None and age_ms <= CONVERGENCE_FRESHNESS_WINDOW_MS
    evidence = {"unindexed_backlog_gap": gap, "window_ms": CONVERGENCE_FRESHNESS_WINDOW_MS, **activity}
    if age_ms is not None:
        evidence["most_recent_activity_age_ms"] = age_ms

    if recent:
        return ArchiveVerificationCheck(
            name="convergence-freshness",
            status=OutcomeStatus.WARNING,
            summary=f"{gap:,} unindexed head(s) with convergence activity in the last "
            f"{CONVERGENCE_FRESHNESS_WINDOW_MS // 3_600_000}h -- still converging, not stalled",
            count=gap,
            evidence=evidence,
        )
    return ArchiveVerificationCheck(
        name="convergence-freshness",
        status=OutcomeStatus.ERROR,
        summary=f"{gap:,} unindexed head(s), no daemon/convergence activity in the last "
        f"{CONVERGENCE_FRESHNESS_WINDOW_MS // 3_600_000}h -- stalled, not just async lag",
        count=gap,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Check 9: user-tier reference liveness (I10, polylogue-t0m73)
# ---------------------------------------------------------------------------


def _check_user_tier_refs(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_user_tier_refs_at_index_path(archive_root, _resolve_index_path(archive_root), sample_limit)


def _check_user_tier_refs_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """``assertions.target_ref`` of kind ``session``/``message`` must resolve
    to a live row in ``index.db`` (I10, polylogue-t0m73).

    Universe is ``assertions`` itself -- the durable, irreplaceable user-tier
    ground truth (a mark/annotation/correction a human or agent made) -- not
    ``index.db``'s own bookkeeping of what it currently holds. An assertion
    whose target no longer resolves (the session/message it was about was
    deleted or never survived a rebuild) is a dangling reference: not fatal
    to read, but the annotation becomes silently unreachable from any
    normal target-scoped query.
    """
    user_path = _tier_path(archive_root, ArchiveTier.USER)
    if not user_path.exists() or not index_path.exists():
        return _skip_check("user-tier-refs", "user.db or index.db not present")

    try:
        conn = _open_ro(user_path)
    except sqlite3.Error as exc:
        return _error_check("user-tier-refs", f"could not open user.db: {exc}", exc=exc)

    try:
        try:
            conn.execute("ATTACH DATABASE ? AS idx_tier", (f"file:{index_path}?mode=ro",))
        except sqlite3.Error as exc:
            return _error_check("user-tier-refs", f"could not attach index.db: {exc}", exc=exc)
        try:
            if not table_exists(conn, "assertions"):
                return _skip_check("user-tier-refs", "assertions table not present")

            dangling_predicate = """
                (a.target_ref LIKE 'session:%' AND NOT EXISTS (
                    SELECT 1 FROM idx_tier.sessions s WHERE s.session_id = substr(a.target_ref, 9)
                ))
                OR (a.target_ref LIKE 'message:%' AND NOT EXISTS (
                    SELECT 1 FROM idx_tier.messages m WHERE m.message_id = substr(a.target_ref, 9)
                ))
            """
            total, dangling_sessions, dangling_messages = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN a.target_ref LIKE 'session:%' OR a.target_ref LIKE 'message:%' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN a.target_ref LIKE 'session:%' AND NOT EXISTS (
                        SELECT 1 FROM idx_tier.sessions s WHERE s.session_id = substr(a.target_ref, 9)
                    ) THEN 1 ELSE 0 END),
                    SUM(CASE WHEN a.target_ref LIKE 'message:%' AND NOT EXISTS (
                        SELECT 1 FROM idx_tier.messages m WHERE m.message_id = substr(a.target_ref, 9)
                    ) THEN 1 ELSE 0 END)
                FROM assertions a
                """
            ).fetchone()
            sample = [
                str(row[0])
                for row in conn.execute(
                    f"SELECT assertion_id FROM assertions a WHERE {dangling_predicate} LIMIT ?",
                    (sample_limit,),
                )
            ]
        except sqlite3.Error as exc:
            return _error_check("user-tier-refs", f"could not read user/index tiers: {exc}", exc=exc)
    finally:
        conn.close()

    total_n = int(total or 0)
    dangling_n = int(dangling_sessions or 0) + int(dangling_messages or 0)
    status = OutcomeStatus.ERROR if dangling_n else OutcomeStatus.OK
    summary = (
        f"{total_n:,} session/message-scoped assertion(s), {dangling_n:,} dangling"
        if dangling_n
        else f"{total_n:,} session/message-scoped assertion(s), all resolve"
    )
    return ArchiveVerificationCheck(
        name="user-tier-refs",
        status=status,
        summary=summary,
        count=dangling_n,
        details=[f"dangling:{assertion_id}" for assertion_id in sample],
        evidence={
            "total_scoped_assertion_count": total_n,
            "dangling_session_ref_count": int(dangling_sessions or 0),
            "dangling_message_ref_count": int(dangling_messages or 0),
            "dangling_sample": sample,
        },
    )


def _check_user_tier_refs_at_candidate(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    return _check_user_tier_refs_at_index_path(archive_root, index_path, sample_limit)


# ---------------------------------------------------------------------------
# Check: active-leaf and title convergence (polylogue-2hwl)
# ---------------------------------------------------------------------------


def _check_active_leaf_title_convergence(
    archive_root: Path, sample_limit: int, *, index_path: Path | None = None
) -> ArchiveVerificationCheck:
    return _check_active_leaf_title_convergence_at_index_path(
        index_path or _resolve_index_path(archive_root), sample_limit
    )


def _check_active_leaf_title_convergence_at_index_path(index_path: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Post-convergence session-row state the streaming merge path must reach.

    polylogue-2hwl's two defects both survive as *stored state*, which is why
    this is a census and not only a parser regression test:

    * ``merge_parsed_session_chunks`` recomputed ``is_active_leaf`` by
      comparing ``provider_message_id`` against the last message's id. Retries
      and regenerations legitimately reuse a native id at more than one
      position, so every matching position was flagged and a session ended up
      with several active leaves (103 sessions measured live, 2026-08-03).
    * the merge kept ``existing.title`` unless it equalled the provider session
      id, so a chunk-1 ``title=None`` was never replaced by a real title
      arriving in chunk 2, and a placeholder could be stored while still
      claiming ``title_source='origin'`` provenance.

    Three predicates, all read from ``index.db`` ground truth:

    1. at most one ``messages.is_active_leaf = 1`` row per session;
    2. ``sessions.active_leaf_message_id``, when set, resolves to a message of
       that same session -- a pointer into another session or into nothing is
       the same convergence failure seen from the session row;
    3. no session claims ``title_source = 'origin'`` (a real provider title)
       while its ``title`` is missing, blank, or literally its own
       ``native_id``. The bare-id fallback itself is legitimate and common
       (``sources/parsers/chatgpt.py`` stores it deliberately) -- what is never
       legitimate is stamping ORIGIN provenance onto it.

    **Scope limit, stated rather than implied:** arrival-order-independent
    winner selection is a property of the *merge function*, not of a snapshot.
    No census over stored rows can replay a different chunk arrival order, so
    that half of the invariant is owned by the parser-level regression tests
    2hwl shipped. What a wrong winner leaves behind in storage is exactly
    predicates 1--3, which is what this check measures.
    """
    if not index_path.exists():
        return _skip_check("active-leaf-title-convergence", "index.db not present")
    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("active-leaf-title-convergence", f"could not open index.db: {exc}", exc=exc)
    try:
        if not table_exists(conn, "sessions") or not table_exists(conn, "messages"):
            return _skip_check("active-leaf-title-convergence", "sessions/messages tables not present")
        multi_leaf_rows = conn.execute(
            """
            SELECT session_id, COUNT(*) AS leaf_count
            FROM messages
            WHERE is_active_leaf = 1
            GROUP BY session_id
            HAVING COUNT(*) > 1
            ORDER BY leaf_count DESC, session_id
            """
        ).fetchall()
        dangling_pointer_rows = conn.execute(
            """
            SELECT s.session_id, s.active_leaf_message_id
            FROM sessions AS s
            WHERE s.active_leaf_message_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM messages AS m
                  WHERE m.message_id = s.active_leaf_message_id
                    AND m.session_id = s.session_id
              )
            ORDER BY s.session_id
            """
        ).fetchall()
        placeholder_title_rows = conn.execute(
            """
            SELECT session_id, native_id, COALESCE(title, '')
            FROM sessions
            WHERE title_source = 'origin'
              AND (title IS NULL OR TRIM(title) = '' OR title = native_id)
            ORDER BY session_id
            """
        ).fetchall()
        # Measured population, so a green result is distinguishable from a
        # green-because-empty one both in the receipt and in the corpus test.
        (sessions_measured,) = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        (sessions_with_active_leaf,) = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM messages WHERE is_active_leaf = 1"
        ).fetchone()
        (origin_titled_sessions,) = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE title_source = 'origin'"
        ).fetchone()
    except sqlite3.Error as exc:
        return _error_check("active-leaf-title-convergence", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    multi_leaf_sessions = len(multi_leaf_rows)
    dangling_pointers = len(dangling_pointer_rows)
    placeholder_titles = len(placeholder_title_rows)
    offending = multi_leaf_sessions + dangling_pointers + placeholder_titles
    details = [f"multi-leaf:{row[0]}({row[1]})" for row in multi_leaf_rows[:sample_limit]]
    details.extend(f"dangling-active-leaf:{row[0]}" for row in dangling_pointer_rows[:sample_limit])
    details.extend(f"origin-titled-placeholder:{row[0]}" for row in placeholder_title_rows[:sample_limit])
    summary = (
        f"{multi_leaf_sessions:,} session(s) with multiple active leaves, "
        f"{dangling_pointers:,} unresolvable active-leaf pointer(s), "
        f"{placeholder_titles:,} origin-titled placeholder(s)"
        if offending
        else "every session has at most one active leaf, a resolvable pointer, and no origin-titled placeholder"
    )
    return ArchiveVerificationCheck(
        name="active-leaf-title-convergence",
        status=OutcomeStatus.ERROR if offending else OutcomeStatus.OK,
        summary=summary,
        count=offending,
        details=details[: sample_limit * 3],
        evidence={
            "sessions_measured": int(sessions_measured),
            "sessions_with_active_leaf": int(sessions_with_active_leaf),
            "origin_titled_session_count": int(origin_titled_sessions),
            "multi_active_leaf_session_count": multi_leaf_sessions,
            "worst_active_leaf_count": int(multi_leaf_rows[0][1]) if multi_leaf_rows else 0,
            "unresolvable_active_leaf_pointer_count": dangling_pointers,
            "origin_titled_placeholder_count": placeholder_titles,
            "multi_active_leaf_sample": [
                {"session_id": str(row[0]), "active_leaf_count": int(row[1])} for row in multi_leaf_rows[:sample_limit]
            ],
            "unresolvable_pointer_sample": [
                {"session_id": str(row[0]), "active_leaf_message_id": str(row[1])}
                for row in dangling_pointer_rows[:sample_limit]
            ],
            "origin_titled_placeholder_sample": [
                {"session_id": str(row[0]), "native_id": str(row[1]), "title": str(row[2])}
                for row in placeholder_title_rows[:sample_limit]
            ],
        },
    )


# ---------------------------------------------------------------------------
# Check: ChatGPT parse-boundary content conservation (polylogue-xofj)
# ---------------------------------------------------------------------------


def _check_chatgpt_content_conservation(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    return _check_chatgpt_content_conservation_at_index_path(
        archive_root, _resolve_index_path(archive_root), sample_limit
    )


def _check_chatgpt_content_conservation_at_index_path(
    archive_root: Path, index_path: Path, sample_limit: int
) -> ArchiveVerificationCheck:
    """Re-read acquired ChatGPT bytes and require every content unit to survive.

    The detector class polylogue-xofj's parent note called for and this
    prior route lacked. Every other conservation check here compares one indexed
    projection with another indexed relation, so a content unit the parser
    never emitted is outside all of their universes by construction: nothing
    counted it on the way in. This check's universe is the durable blob, so a
    ``content_type`` the parser has no branch for -- the exact shape of the
    ``tether_quote``/``sonic_webpage``/``system_error``/``citable_code_output``
    class xofj fixed -- shows up as a drop with its provider content type
    named, instead of as silence.
    """
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    blob_root = archive_root / "blob"
    if not source_path.exists() or not index_path.exists():
        return _skip_check("chatgpt-content-conservation", "source.db or index.db not present")
    if not blob_root.exists():
        return _skip_check("chatgpt-content-conservation", "blob namespace not present")
    try:
        source = _open_ro(source_path)
        index = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("chatgpt-content-conservation", f"could not open source/index tiers: {exc}", exc=exc)
    try:
        if not table_exists(source, "raw_sessions"):
            return _skip_check("chatgpt-content-conservation", "raw_sessions table not present")
        evidence = audit_chatgpt_content_conservation(
            source, index, BlobStore(blob_root).read_all, sample_limit=sample_limit
        )
    except sqlite3.Error as exc:
        return _error_check("chatgpt-content-conservation", f"could not read source/index tiers: {exc}", exc=exc)
    finally:
        index.close()
        source.close()

    dropped = int(evidence["content_units_dropped"])
    conserved = int(evidence["content_units_conserved"])
    if not evidence["source_rows_selected"]:
        return ArchiveVerificationCheck(
            name="chatgpt-content-conservation",
            status=OutcomeStatus.SKIP,
            summary="ChatGPT source population is absent; conservation is not applicable",
            evidence={**evidence, "outcome_reason": "no_chatgpt_population"},
        )
    if not evidence["blobs_readable"]:
        return ArchiveVerificationCheck(
            name="chatgpt-content-conservation",
            status=OutcomeStatus.ERROR,
            summary="ChatGPT source evidence exists but no selected blob is readable",
            count=int(evidence["blobs_missing"]),
            evidence={**evidence, "outcome_reason": "all_blobs_unreadable"},
        )
    if not evidence["documents_lowered"]:
        return ArchiveVerificationCheck(
            name="chatgpt-content-conservation",
            status=OutcomeStatus.ERROR,
            summary="ChatGPT source evidence lowered zero documents",
            count=int(evidence["blobs_readable"]),
            evidence={**evidence, "outcome_reason": "unsupported_or_malformed_envelopes"},
        )
    if not evidence["candidate_documents_matched"]:
        return ArchiveVerificationCheck(
            name="chatgpt-content-conservation",
            status=OutcomeStatus.ERROR,
            summary="ChatGPT documents were lowered but none overlap the candidate index",
            count=int(evidence["candidate_documents_absent"]),
            evidence={**evidence, "outcome_reason": "zero_candidate_overlap"},
        )
    if not evidence["content_units_enumerated"]:
        return ArchiveVerificationCheck(
            name="chatgpt-content-conservation",
            status=OutcomeStatus.ERROR,
            summary="ChatGPT documents overlap the candidate but contain no measurable content units",
            evidence={**evidence, "outcome_reason": "zero_content_units"},
        )
    summary = (
        f"{dropped:,} content unit(s) dropped at the parse boundary across "
        f"{evidence['documents_with_dropped_content']:,} document(s): "
        + ", ".join(f"{name}={count:,}" for name, count in evidence["dropped_by_content_type"].items())
        if dropped
        else (
            f"all {conserved:,} content unit(s) across {evidence['documents_measured']:,} "
            "chatgpt-export document(s) materialize"
        )
    )
    return ArchiveVerificationCheck(
        name="chatgpt-content-conservation",
        status=OutcomeStatus.ERROR if dropped else OutcomeStatus.OK,
        summary=summary,
        count=dropped,
        details=[
            f"{item['session_id']}:{item['provider_message_id']}[{item['content_type']}]"
            for item in evidence["dropped_sample"]
        ],
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Domain declarations + entrypoint
# ---------------------------------------------------------------------------


def verify_archive(
    archive_root: Path,
    *,
    checks: Sequence[str] | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    index_path_override: Path | None = None,
    active_index_context: Literal["required", "unavailable_for_candidate"] = "required",
) -> ArchiveVerificationReport:
    """Run selected owner declarations and return disposable evidence."""
    declarations = archive_verification_domain_adapters(
        archive_root,
        sample_limit=sample_limit,
        index_path_override=index_path_override,
        active_index_context=active_index_context,
    )
    by_name = {owner.name: owner for owner in declarations}
    selected_names = (
        tuple(dict.fromkeys(checks)) if checks is not None else archive_verification_names_for_route(_ROUTE_LIVE)
    )
    unknown = [name for name in selected_names if name not in by_name]
    if unknown:
        raise ValueError(
            f"unknown archive verification check(s): {', '.join(unknown)}; available: {', '.join(by_name)}"
        )
    selected = tuple(by_name[name] for name in selected_names)
    if index_path_override is not None:
        unsupported = [owner.name for owner in selected if owner.candidate_check is None]
        if unsupported:
            raise ValueError("candidate index verification is unsupported for check(s): " + ", ".join(unsupported))
    report = compose_outcome_checks(selected)
    coverage = ArchiveVerificationCoverage(
        route=("candidate-index" if index_path_override is not None else _ROUTE_LIVE),
        candidate_id=str(index_path_override) if index_path_override is not None else None,
        declarations=selected,
        missing_production_routes=tuple(owner.name for owner in selected if not owner.production_route),
        ownerless_checks=tuple(
            owner.name for owner in selected if not owner.semantic_owner or not owner.owned_reference
        ),
        duplicate_checks=tuple(sorted({name for name in selected_names if selected_names.count(name) > 1})),
    )
    return ArchiveVerificationReport(
        checks=report.checks,
        archive_root=str(archive_root),
        generated_at=datetime.now(UTC).isoformat(),
        coverage=coverage,
    )


def archive_verification_names_for_route(route: str) -> tuple[str, ...]:
    """Return names selected by a production route's owner declarations."""
    return tuple(
        owner.name for owner in archive_verification_domain_adapters(Path(".")) if route in owner.applicable_routes
    )


__all__ = [
    "ArchiveVerificationCoverage",
    "ArchiveVerificationCheck",
    "ArchiveVerificationReport",
    "DEFAULT_SAMPLE_LIMIT",
    "archive_verification_owner_adapters",
    "archive_verification_migrated_owner_adapters",
    "archive_verification_domain_adapters",
    "archive_verification_coverage",
    "archive_verification_names_for_route",
    "read_raw_failure_lifecycle",
    "passes_strict_acceptance",
    "strict_acceptance_failures",
    "verify_archive",
]
