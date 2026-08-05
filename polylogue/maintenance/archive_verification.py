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

New checks slot into :data:`ARCHIVE_VERIFICATION_CHECKS` without touching
:func:`verify_archive` or its callers -- the registry is the extension point
for future checks (blob-reference debt, cost rollups, ...).

**Registry contract rule (polylogue-in24n):** a check's universe must be a
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus
from polylogue.logging import get_logger
from polylogue.maintenance.corpus_fidelity import (
    audit_absences,
    audit_attachment_fidelity,
    audit_revision_fidelity,
)
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.storage.blob_gc import BLOB_REF_LIVENESS_JOIN
from polylogue.storage.introspection import table_exists
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

#: Default cap on per-check sample evidence (worst sessions, offending ids, ...).
DEFAULT_SAMPLE_LIMIT = 10


class ArchiveVerificationCheckClass(str, Enum):
    """Bug-class taxonomy each registry check is tagged with (polylogue-t0m73).

    Orthogonal to :class:`~polylogue.core.outcomes.OutcomeStatus` (pass/warn/
    fail/skip is the *outcome*; this is the *kind of invariant* being
    checked), the tag drives which of the two verification planes
    (polylogue-60gzo) a check is expected to run under on a schedule:
    ``liveness``/``freshness`` classes are the daemon health-tier home
    (Plane 2, production monitoring -- something must have happened
    recently); the rest are point-in-time coherence checks that make sense
    on any snapshot including a reindex candidate generation (both planes).
    """

    STATE_INVARIANT = "state-invariant"
    """A structural fact that must always hold on any coherent snapshot
    (pointer targets resolve, no cycles, no orphaned rows)."""

    LIVENESS = "liveness"
    """A reference/relationship that must remain live -- staleness here means
    something died without cleaning up after itself (GC bookkeeping,
    embedding refs outliving their source rows)."""

    FRESHNESS = "freshness"
    """Not a correctness fact but a recency fact: has required maintenance
    (ANALYZE, convergence passes) run recently enough to trust the archive's
    current operating characteristics."""

    COMPLEXITY = "complexity"
    """Archive-wide shape/size reporting -- not itself a pass/fail invariant,
    but the numbers an operator needs to judge whether other checks' drift is
    proportionally significant."""

    FIDELITY = "fidelity"
    """A derived/shadow representation must exactly mirror its source of
    truth (FTS index vs the blocks it indexes)."""

    CONSERVATION = "conservation"
    """A quantity computed two different ways must agree (a write-time
    projection vs a live COUNT over the rows it summarizes)."""

    CONFIG = "config"
    """The archive's on-disk schema/vocabulary configuration must be a
    superset of what current code can write (CHECK constraints vs the
    Python enum that generated them, tier schema versions vs the canonical
    spec)."""


@dataclass(frozen=True)
class ArchiveVerificationWaiver:
    """A known-red-on-live acknowledgement for one registry check.

    Per polylogue-t0m73's design: a waiver exists only to keep the
    *aggregate* gate (:attr:`ArchiveVerificationReport.blocking`) from
    tripping on a bug that is already tracked and being worked, not to hide
    the finding -- the underlying check still reports its true ``error``
    status and evidence; only the gate's blocking computation treats it as
    non-blocking. A waiver is a manual, reviewed acknowledgement, not an
    automatic bd query (this module never calls ``bd``): remove the entry in
    the same change that closes the bead, and if the check still reports
    red afterward, the removal makes the gate red again on the next run,
    which is the intended non-silent failure mode of an unwaived close.
    """

    bead_id: str
    reason: str


#: Checks with a currently-open, already-tracked live finding: the check
#: keeps reporting its true ``error`` status (evidence is never suppressed),
#: but :attr:`ArchiveVerificationReport.blocking` excludes it so an unrelated
#: gate (e.g. the reindex acceptance run) doesn't trip on a distinct,
#: separately-owned bug. Remove an entry only alongside closing its bead.
ARCHIVE_VERIFICATION_WAIVERS: dict[str, ArchiveVerificationWaiver] = {
    "embeddings-refs-liveness": ArchiveVerificationWaiver(
        bead_id="polylogue-feu0",
        reason=(
            "4,186 message_embedding_refs point at messages no longer in index.db "
            "(known, undrained catch-up debt as of 2026-08-03; embeddings convergence "
            "is a separate async lane from index materialization)"
        ),
    ),
}

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
    #: Bug-class tag copied from the owning :class:`ArchiveVerificationCheckSpec`
    #: by :func:`verify_archive` (the spec is the source of truth; this is a
    #: read-through convenience for JSON consumers that only see the check).
    check_class: str = ""
    #: Set by :func:`verify_archive` when this check's name has an entry in
    #: :data:`ARCHIVE_VERIFICATION_WAIVERS`. ``status`` is never altered by a
    #: waiver -- a waived check that is genuinely red still reports ``error``;
    #: only :attr:`ArchiveVerificationReport.blocking` treats it as non-blocking.
    waived_bead_id: str | None = None

    def to_json(self) -> JSONDocument:
        return archive_verification_check_json(self)


def archive_verification_check_json(check: OutcomeCheck) -> JSONDocument:
    """Return the JSON payload for one check, base attrs plus its evidence.

    Mirrors :func:`polylogue.schemas.audit.models.audit_check_json`: reads
    the shared :class:`OutcomeCheck` attrs directly and reaches for the
    subclass-only ``evidence``/``check_class``/``waived_bead_id`` fields via
    ``getattr`` so callers holding a plain ``OutcomeCheck``-typed reference
    (e.g. ``ArchiveVerificationReport.checks``, typed at its base-class
    element type) can still serialize a concrete :class:`ArchiveVerificationCheck`
    without a narrowing cast.
    """
    payload = dict(check.to_dict())
    payload["evidence"] = json_document(getattr(check, "evidence", {}))
    payload["check_class"] = getattr(check, "check_class", "")
    payload["waived_bead_id"] = getattr(check, "waived_bead_id", None)
    return payload


def _error_check(name: str, summary: str, *, exc: Exception | None = None) -> ArchiveVerificationCheck:
    evidence: dict[str, Any] = {"error": str(exc)} if exc is not None else {}
    return ArchiveVerificationCheck(name=name, status=OutcomeStatus.ERROR, summary=summary, count=1, evidence=evidence)


def _skip_check(name: str, summary: str) -> ArchiveVerificationCheck:
    return ArchiveVerificationCheck(name=name, status=OutcomeStatus.SKIP, summary=summary)


@dataclass
class ArchiveVerificationReport(OutcomeReport):
    """Full archive-verification report across every selected check."""

    archive_root: str = ""
    generated_at: str = ""

    @property
    def blocking(self) -> bool:
        """True when at least one *unwaived* check reports ``error``.

        A waived check (see :data:`ARCHIVE_VERIFICATION_WAIVERS`) still
        contributes to :attr:`error_count` -- the underlying finding is never
        hidden -- but is excluded from the gate condition itself, so an
        already-tracked, separately-owned bug doesn't block an unrelated
        acceptance run (e.g. a reindex candidate promotion).
        """
        return any(
            check.status is OutcomeStatus.ERROR and getattr(check, "waived_bead_id", None) is None
            for check in self.checks
        )

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "archive_root": self.archive_root,
                "generated_at": self.generated_at,
                "summary": self.summary_counts(include_skip=True),
                "blocking": self.blocking,
                "checks": [archive_verification_check_json(check) for check in self.checks],
            }
        )


ArchiveVerificationCheckFn = Callable[[Path, int], ArchiveVerificationCheck]
ArchiveVerificationCandidateCheckFn = Callable[[Path, Path, int], ArchiveVerificationCheck]


@dataclass(frozen=True)
class ArchiveVerificationCheckSpec:
    """One named, independently runnable archive-coherence check."""

    name: str
    description: str
    run: ArchiveVerificationCheckFn
    #: One of :class:`ArchiveVerificationCheckClass`'s values -- every spec in
    #: :data:`ARCHIVE_VERIFICATION_CHECKS` must set this (no default, so a new
    #: check added without a class tag is a construction-time TypeError, not a
    #: silent "" that would slip past classification).
    check_class: ArchiveVerificationCheckClass
    #: Optional candidate-index execution path for checks that combine the
    #: durable archive root with an inactive generation before promotion.
    candidate_run: ArchiveVerificationCandidateCheckFn | None = None


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


def _check_source_index_coverage(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Every logical source's head is indexed OR carries a typed refusal.

    Universe (polylogue-in24n, invariant I1): ``raw_sessions`` logical heads --
    the latest revision per ``(origin, COALESCE(native_id, source_path))`` --
    which is a ground-truth table every acquired raw lands in, not a ledger a
    downstream reconciliation stage can silently omit rows from. A logical
    head counts as covered when *any* raw in its revision group is
    materialized into ``index.db``. An uncovered head must be typed as one
    of: ``parse_error`` (raw_sessions.parse_error set), ``non_session`` /
    ``census_failed`` (raw_membership_census recorded a terminal non-complete
    verdict), or ``quarantined`` (raw_sessions.revision_authority --
    reconciliation hasn't granted it authority to write yet, WARN-level
    evidence, not blocking). An uncovered head matching none of those is an
    *untyped* gap -- a materialization failure no other subsystem has
    explained -- and is the only ERROR-gating condition here besides orphans
    (index sessions whose raw_id doesn't exist in source.db at all).
    """
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    index_path = _resolve_index_path(archive_root)
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
                        {census_expr} AS census_status,
                        MAX(EXISTS(SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = r.raw_id))
                            OVER (PARTITION BY r.origin, COALESCE(r.native_id, r.source_path)) AS any_indexed,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.origin, COALESCE(r.native_id, r.source_path)
                            ORDER BY r.acquired_at_ms DESC, r.raw_id DESC
                        ) AS rn
                    FROM raw_sessions r
                )
            """
            untyped_predicate = """
                rn = 1 AND any_indexed = 0 AND parse_error IS NULL
                  AND COALESCE(census_status, '') NOT IN ('non_session', 'failed')
                  AND revision_authority != 'quarantined'
            """
            quarantined_predicate = """
                rn = 1 AND any_indexed = 0 AND parse_error IS NULL
                  AND COALESCE(census_status, '') NOT IN ('non_session', 'failed')
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
                  SUM(CASE WHEN {quarantined_predicate.replace("rn = 1 AND ", "")} THEN 1 ELSE 0 END),
                  SUM(CASE WHEN {untyped_predicate.replace("rn = 1 AND ", "")} THEN 1 ELSE 0 END)
                FROM heads WHERE rn = 1
                """
            ).fetchone()
            parse_error_n, non_session_n, census_failed_n, quarantined_n, untyped_n = (
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

    unindexed_head_count = parse_error_n + non_session_n + census_failed_n + quarantined_n + untyped_n
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
            "quarantined_count": quarantined_n,
            "quarantined_sample": quarantined_sample,
            "byte_dup_of_indexed_count": byte_dup_of_indexed_n,
            "novel_unindexed_count": novel_unindexed_n,
            "orphan_count": int(orphan_count or 0),
            "orphan_sample": orphan_sample,
        },
    )


# ---------------------------------------------------------------------------
# Check 4: FTS parity (archive-wide)
# ---------------------------------------------------------------------------


def _check_fts_parity(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    index_path = _resolve_index_path(archive_root)
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


def _check_lineage_sanity(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    index_path = _resolve_index_path(archive_root)
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

        for table_name, ddl in rows:
            for match in _ORIGIN_CHECK_COLUMN_PATTERN.finditer(ddl or ""):
                column, allowed_list = match.group(1), match.group(2)
                allowed = set(re.findall(r"'([^']*)'", allowed_list))
                missing = sorted(origins - allowed)
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
    ref_type: (referent_table, referent_column) for ref_type, referent_table, referent_column in BLOB_REF_LIVENESS_JOIN
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
        index_conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("blob-reference-closure", f"could not open source/index tiers: {exc}", exc=exc)
    try:
        from polylogue.maintenance.blob_reference_closure import closure_counts

        counts = closure_counts(source_conn, index_conn)
        raw_sample = [
            str(row[0])
            for row in source_conn.execute(
                """
                SELECT r.raw_id FROM raw_sessions r
                WHERE (
                    SELECT COUNT(*) FROM blob_refs b
                    WHERE b.ref_type = 'raw_payload' AND b.ref_id = r.raw_id AND b.blob_hash = r.blob_hash
                ) != 1
                   OR (
                       SELECT COUNT(*) FROM blob_refs b
                       WHERE b.ref_type = 'raw_payload' AND b.ref_id = r.raw_id
                   ) != 1
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


def _check_pathology_zoo_invariants(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    """Run each production-owned pathology-zoo invariant when its corpus is present."""
    from polylogue.maintenance.pathology_zoo import (
        PATHOLOGY_ZOO_MANIFEST,
        PathologyZooMember,
        pathology_zoo_is_present,
    )

    if not pathology_zoo_is_present(archive_root):
        return ArchiveVerificationCheck(
            name="pathology-zoo-invariants",
            status=OutcomeStatus.OK,
            summary="no pathology-zoo corpus detected; zoo-specific invariants are not applicable",
            evidence={"active": False, "checked_member_ids": [], "failed_member_ids": []},
        )

    failed: list[PathologyZooMember] = []
    for member in PATHOLOGY_ZOO_MANIFEST:
        try:
            satisfied = member.invariant.is_satisfied(archive_root)
        except sqlite3.Error:
            satisfied = False
        if not satisfied:
            failed.append(member)
    return ArchiveVerificationCheck(
        name="pathology-zoo-invariants",
        status=OutcomeStatus.ERROR if failed else OutcomeStatus.OK,
        summary=(
            f"{len(failed)} pathology-zoo invariant(s) failed"
            if failed
            else f"all {len(PATHOLOGY_ZOO_MANIFEST)} pathology-zoo invariants hold"
        ),
        count=len(failed),
        details=[f"{member.member_id}: {member.invariant.condition}" for member in failed],
        evidence={
            "active": True,
            "checked_member_ids": [member.member_id for member in PATHOLOGY_ZOO_MANIFEST],
            "failed_member_ids": [member.member_id for member in failed],
        },
    )


def _check_pathology_zoo_invariants_at_index_path(
    archive_root: Path, index_path: Path, _sample_limit: int
) -> ArchiveVerificationCheck:
    """Run pathology-zoo invariants against an inactive index and live durable tiers."""
    from polylogue.maintenance.pathology_zoo import (
        PATHOLOGY_ZOO_MANIFEST,
        PathologyZooMember,
        pathology_zoo_is_present,
    )

    if not pathology_zoo_is_present(archive_root, index_path_override=index_path):
        return ArchiveVerificationCheck(
            name="pathology-zoo-invariants",
            status=OutcomeStatus.OK,
            summary="no pathology-zoo corpus detected; zoo-specific invariants are not applicable",
            evidence={"active": False, "checked_member_ids": [], "failed_member_ids": []},
        )

    failed: list[PathologyZooMember] = []
    for member in PATHOLOGY_ZOO_MANIFEST:
        try:
            satisfied = member.invariant.is_satisfied(archive_root, index_path_override=index_path)
        except sqlite3.Error:
            satisfied = False
        if not satisfied:
            failed.append(member)
    return ArchiveVerificationCheck(
        name="pathology-zoo-invariants",
        status=OutcomeStatus.ERROR if failed else OutcomeStatus.OK,
        summary=(
            f"{len(failed)} pathology-zoo invariant(s) failed"
            if failed
            else f"all {len(PATHOLOGY_ZOO_MANIFEST)} pathology-zoo invariants hold"
        ),
        count=len(failed),
        details=[f"{member.member_id}: {member.invariant.condition}" for member in failed],
        evidence={
            "active": True,
            "checked_member_ids": [member.member_id for member in PATHOLOGY_ZOO_MANIFEST],
            "failed_member_ids": [member.member_id for member in failed],
            "index_path": str(index_path),
            "durable_archive_root": str(archive_root),
        },
    )


# ---------------------------------------------------------------------------
# Check: embeddings refs resolve in the index tier (I4, polylogue-t0m73)
# ---------------------------------------------------------------------------


def _check_embeddings_refs_liveness(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Every ``message_embedding_refs`` row resolves to a live index message.

    embeddings.db is a rebuildable derived tier keyed off index.db's message
    ids; an embedding ref outliving the message it was computed for (e.g. a
    session full-replace deleting messages without the embedding catch-up
    stage draining the corresponding refs) is exactly the feu0-class bug:
    dead vector rows that read back as "embedded" for content that no
    longer exists in the index.
    """
    embeddings_path = _tier_path(archive_root, ArchiveTier.EMBEDDINGS)
    index_path = _resolve_index_path(archive_root)
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


# ---------------------------------------------------------------------------
# Check: session parent-chain acyclicity (I5, polylogue-t0m73)
# ---------------------------------------------------------------------------


def _check_session_lineage_acyclic(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
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
    index_path = _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("session-lineage-acyclic", "index.db not present")

    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("session-lineage-acyclic", f"could not open index.db: {exc}", exc=exc)

    try:
        try:
            parents: dict[str, str] = dict(
                conn.execute(
                    "SELECT session_id, parent_session_id FROM sessions WHERE parent_session_id IS NOT NULL"
                ).fetchall()
            )
        except sqlite3.Error as exc:
            return _error_check("session-lineage-acyclic", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    # Walk each node's parent chain once; nodes already proven part of some
    # walk (cyclic or not) are never re-walked, so this is O(n) total despite
    # the outer loop over every node.
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


def _check_message_count_projection(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """``sessions.message_count`` matches the actual materialized row count.

    ``message_count`` is a write-time projection (set when a session is
    written, not a generated column), so it can drift from the underlying
    ``messages`` rows if a partial write, a bug in the write path, or a
    manual repair touches one but not the other. Ground truth is
    ``COUNT(*)`` over ``messages`` itself, joined back to ``sessions``.
    """
    index_path = _resolve_index_path(archive_root)
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


def _check_session_fingerprint_stamps(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """Require every candidate session to carry current, unmixed semantic stamps.

    ``sessions`` is the ground-truth candidate universe. The check compares
    each row to the current source-derived stamp rather than accepting a
    merely well-formed historical value, so a parser/lowering semantic change
    cannot promote a stale reindex candidate.
    """
    index_path = _resolve_index_path(archive_root)
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


def _check_planner_stats(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    index_path = _resolve_index_path(archive_root)
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
# Check: raw quarantine group dedup (polylogue-zm4w8)
# ---------------------------------------------------------------------------


def _check_raw_quarantine_group_dedup(archive_root: Path, sample_limit: int) -> ArchiveVerificationCheck:
    """No raw_sessions row is an unindexed byte-identical duplicate of another sharing its source_path.

    polylogue-zm4w8: a raw_sessions row is flagged when it belongs to a
    ``(source_path, blob_hash)`` group of more than one quarantined row AND
    no member of that group (nor any other raw sharing that blob_hash
    anywhere) already has a materialized ``index.db`` session or a
    non-quarantined ``revision_authority``. This is the residual,
    genuinely-unresolved population: content acquired more than once that
    was never chosen as its group's representative and materialized --
    distinct from ``source-index-coverage``'s ``byte_dup_of_indexed_count``
    (which requires an *already-indexed* twin) and from what
    ``raw-byte-duplicate-supersession-apply`` (the actuator for that
    already-indexed case) can catch by construction. The one-shot
    ``raw-quarantine-group-dedup-apply`` actuator resolves a flagged group by
    materializing exactly one representative raw and marking the rest
    ``byte_proven`` -- once run, this check should report clean, and stays
    part of the registry as the standing regression guard against the
    pattern recurring.
    """
    from polylogue.storage.raw_quarantine_group_dedup import plan_raw_quarantine_group_dedup

    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    index_path = _resolve_index_path(archive_root)
    if not source_path.exists() or not index_path.exists():
        return _skip_check("raw-quarantine-group-dedup", "source.db or index.db not present")

    try:
        source_conn = _open_ro(source_path)
    except sqlite3.Error as exc:
        return _error_check("raw-quarantine-group-dedup", f"could not open source.db: {exc}", exc=exc)

    try:
        index_conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        source_conn.close()
        return _error_check("raw-quarantine-group-dedup", f"could not open index.db: {exc}", exc=exc)

    try:
        plan = plan_raw_quarantine_group_dedup(source_conn, index_conn)
    except sqlite3.Error as exc:
        return _error_check("raw-quarantine-group-dedup", f"could not read source/index tiers: {exc}", exc=exc)
    finally:
        index_conn.close()
        source_conn.close()

    group_count = len(plan.groups)
    duplicate_count = plan.duplicate_count
    sample = [
        {
            "source_path": group.source_path,
            "representative_raw_id": group.representative_raw_id,
            "duplicate_raw_ids": list(group.duplicate_raw_ids),
        }
        for group in plan.groups[:sample_limit]
    ]

    status = OutcomeStatus.ERROR if group_count else OutcomeStatus.OK
    summary = (
        f"{group_count:,} fully-quarantined duplicate group(s), {duplicate_count:,} unindexed duplicate row(s) "
        f"({_gib_str(plan.duplicate_bytes)}); run raw-quarantine-group-dedup-apply"
        if group_count
        else f"no fully-quarantined byte-identical duplicate groups ({plan.scanned_count:,} quarantined row(s) scanned)"
    )
    return ArchiveVerificationCheck(
        name="raw-quarantine-group-dedup",
        status=status,
        summary=summary,
        count=duplicate_count,
        details=[f"group:{group.source_path}" for group in plan.groups[:sample_limit]],
        evidence={
            "scanned_count": plan.scanned_count,
            "group_count": group_count,
            "duplicate_count": duplicate_count,
            "duplicate_bytes": plan.duplicate_bytes,
            "already_resolved_group_count": plan.already_resolved_group_count,
            "group_sample": sample,
        },
    )


def _gib_str(byte_count: int) -> str:
    return f"{byte_count / (1024**3):.2f} GiB"


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
# Check 7: counts summary
# ---------------------------------------------------------------------------


def _check_counts_summary(archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
    index_path = _resolve_index_path(archive_root)
    if not index_path.exists():
        return _skip_check("counts-summary", "index.db not present")

    try:
        conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("counts-summary", f"could not open index.db: {exc}", exc=exc)

    try:
        session_count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        message_count = int(conn.execute("SELECT COALESCE(SUM(message_count), 0) FROM sessions").fetchone()[0])
        block_count = (
            int(conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]) if table_exists(conn, "blocks") else 0
        )
        origin_breakdown = {
            str(row[0]): int(row[1])
            for row in conn.execute("SELECT origin, COUNT(*) FROM sessions GROUP BY origin ORDER BY origin")
        }
    except sqlite3.Error as exc:
        return _error_check("counts-summary", f"could not read index.db: {exc}", exc=exc)
    finally:
        conn.close()

    return ArchiveVerificationCheck(
        name="counts-summary",
        status=OutcomeStatus.OK,
        summary=f"{session_count:,} sessions, {message_count:,} messages, {block_count:,} blocks",
        breakdown=origin_breakdown,
        evidence={
            "session_count": session_count,
            "message_count": message_count,
            "block_count": block_count,
            "origin_breakdown": origin_breakdown,
        },
    )


# ---------------------------------------------------------------------------
# Check 8: convergence freshness (I6, polylogue-t0m73)
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
    row = conn.execute(
        f"""
        WITH heads AS (
            SELECT
                r.raw_id,
                r.parse_error,
                {census_expr} AS census_status,
                MAX(EXISTS(SELECT 1 FROM idx_tier.sessions s WHERE s.raw_id = r.raw_id))
                    OVER (PARTITION BY r.origin, COALESCE(r.native_id, r.source_path)) AS any_indexed,
                ROW_NUMBER() OVER (
                    PARTITION BY r.origin, COALESCE(r.native_id, r.source_path)
                    ORDER BY r.acquired_at_ms DESC, r.raw_id DESC
                ) AS rn
            FROM raw_sessions r
        )
        SELECT SUM(
            CASE WHEN rn = 1 AND any_indexed = 0 AND parse_error IS NULL
                      AND COALESCE(census_status, '') NOT IN ('non_session', 'failed')
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
    index_path = _resolve_index_path(archive_root)
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


# ---------------------------------------------------------------------------
# Registry + entrypoint
# ---------------------------------------------------------------------------

ARCHIVE_VERIFICATION_CHECKS: tuple[ArchiveVerificationCheckSpec, ...] = (
    ArchiveVerificationCheckSpec(
        "tier-schema",
        "Tier presence and PRAGMA user_version vs the canonical ARCHIVE_TIER_SPECS.",
        _check_tier_schema,
        ArchiveVerificationCheckClass.CONFIG,
    ),
    ArchiveVerificationCheckSpec(
        "pointer-coherence",
        "Conventional index.db path vs the active .index-active-pointer generation (polylogue-k8kj class).",
        _check_pointer_coherence,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
    ),
    ArchiveVerificationCheckSpec(
        "source-index-coverage",
        "Every raw_sessions logical head is indexed or typed (parse_error/non_session/quarantined); "
        "untyped gaps and index-orphans (raw_id ground truth, not the census ledger, polylogue-in24n) block.",
        _check_source_index_coverage,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
    ),
    ArchiveVerificationCheckSpec(
        "fts-parity",
        "messages_fts and blocks_command_trigram exactly cover their source rows, archive-wide.",
        _check_fts_parity,
        ArchiveVerificationCheckClass.FIDELITY,
    ),
    ArchiveVerificationCheckSpec(
        "lineage-sanity",
        "session_links.resolved_dst_session_id / branch_point_message_id resolve to real sessions/messages.",
        _check_lineage_sanity,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
    ),
    ArchiveVerificationCheckSpec(
        "enum-superset-check",
        "Live origin/dst_origin CHECK lists on disk are a superset of the current Origin enum (polylogue-t0m73 I2).",
        _check_enum_superset,
        ArchiveVerificationCheckClass.CONFIG,
    ),
    ArchiveVerificationCheckSpec(
        "blob-refs-liveness",
        "Every blob_refs row resolves in its ref_type's referent table -- the GC liveness oracle is a join, "
        "not membership in blob_refs itself (polylogue-t0m73 I3).",
        _check_blob_refs_liveness,
        ArchiveVerificationCheckClass.LIVENESS,
    ),
    ArchiveVerificationCheckSpec(
        "blob-reference-closure",
        "Every raw session has exactly one matching raw_payload ref and every acquired attachment is reachable by attachment_refs.",
        _check_blob_reference_closure,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
        _check_blob_reference_closure_at_index_path,
    ),
    ArchiveVerificationCheckSpec(
        "pathology-zoo-invariants",
        "Every present pathology-zoo member satisfies its production-owned invariant.",
        _check_pathology_zoo_invariants,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
        _check_pathology_zoo_invariants_at_index_path,
    ),
    ArchiveVerificationCheckSpec(
        "embeddings-refs-liveness",
        "message_embedding_refs resolve to live index.db messages (feu0-class dead-vector detection, "
        "polylogue-t0m73 I4).",
        _check_embeddings_refs_liveness,
        ArchiveVerificationCheckClass.LIVENESS,
    ),
    ArchiveVerificationCheckSpec(
        "session-lineage-acyclic",
        "sessions.parent_session_id chains never close a cycle (polylogue-t0m73 I5).",
        _check_session_lineage_acyclic,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
    ),
    ArchiveVerificationCheckSpec(
        "message-count-projection",
        "sessions.message_count matches COUNT(messages) per session (polylogue-t0m73 I8).",
        _check_message_count_projection,
        ArchiveVerificationCheckClass.CONSERVATION,
    ),
    ArchiveVerificationCheckSpec(
        "session-fingerprint-stamps",
        "Every indexed session carries current, unmixed parser and lowering semantic stamps (polylogue-xselt).",
        _check_session_fingerprint_stamps,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
    ),
    ArchiveVerificationCheckSpec(
        "planner-stats",
        "sqlite_stat1 covers blocks/messages/action_pairs (polylogue-l3tk class, warn-level).",
        _check_planner_stats,
        ArchiveVerificationCheckClass.FRESHNESS,
    ),
    ArchiveVerificationCheckSpec(
        "counts-summary",
        "Archive-wide session/message/block counts and origin breakdown (numbers-freeze starter).",
        _check_counts_summary,
        ArchiveVerificationCheckClass.COMPLEXITY,
    ),
    ArchiveVerificationCheckSpec(
        "convergence-freshness",
        "An open unindexed backlog (I1's universe) with no daemon/convergence activity in the last 24h is "
        "stalled, not async lag (polylogue-t0m73 I6).",
        _check_convergence_freshness,
        ArchiveVerificationCheckClass.FRESHNESS,
    ),
    ArchiveVerificationCheckSpec(
        "user-tier-refs",
        "assertions.target_ref of kind session/message resolves to a live index.db row (polylogue-t0m73 I10).",
        _check_user_tier_refs,
        ArchiveVerificationCheckClass.LIVENESS,
    ),
    ArchiveVerificationCheckSpec(
        "excluded-cursor-vocabulary-honesty",
        "No excluded ingest_cursor row carries a live next_retry_at (would misreport as retry-due, polylogue-ix5r).",
        _check_excluded_cursor_vocabulary_honesty,
        ArchiveVerificationCheckClass.LIVENESS,
    ),
    ArchiveVerificationCheckSpec(
        "stalled-append-cursor-freshness",
        "No non-excluded ingest_cursor row sits behind its file's current size for longer than "
        "the stall-escalation threshold (polylogue-2qrx).",
        _check_stalled_append_cursor_freshness,
        ArchiveVerificationCheckClass.LIVENESS,
    ),
    ArchiveVerificationCheckSpec(
        "raw-quarantine-group-dedup",
        "No raw_sessions row is an unindexed byte-identical duplicate of another sharing its "
        "source_path within a fully-quarantined group (polylogue-zm4w8).",
        _check_raw_quarantine_group_dedup,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
    ),
    ArchiveVerificationCheckSpec(
        "corpus-absences",
        "Every source-backed logical corpus document is represented by an indexed session (polylogue-f1vg).",
        _check_corpus_absences,
        ArchiveVerificationCheckClass.STATE_INVARIANT,
        _check_corpus_absences_at_index_path,
    ),
    ArchiveVerificationCheckSpec(
        "corpus-attachment-fidelity",
        "Every attachment reference is acquired or explicitly typed unavailable, bucketed by origin and upload origin (polylogue-f1vg).",
        _check_corpus_attachment_fidelity,
        ArchiveVerificationCheckClass.FIDELITY,
        _check_corpus_attachment_fidelity_at_index_path,
    ),
    ArchiveVerificationCheckSpec(
        "corpus-revision-fidelity",
        "Indexed comparable evidence is not smaller than the best recorded source revision (polylogue-f1vg).",
        _check_corpus_revision_fidelity,
        ArchiveVerificationCheckClass.FIDELITY,
        _check_corpus_revision_fidelity_at_index_path,
    ),
)

ARCHIVE_VERIFICATION_CHECK_NAMES: tuple[str, ...] = tuple(spec.name for spec in ARCHIVE_VERIFICATION_CHECKS)

#: The corpus-fidelity checks exposed by the post-rebuild acceptance command.
#: Keeping this selection beside the registry prevents a surface from
#: re-declaring the check names or drifting away from the production gate.
CORPUS_FIDELITY_CHECKS: tuple[str, ...] = (
    "corpus-absences",
    "corpus-attachment-fidelity",
    "corpus-revision-fidelity",
)

#: The subset of ground-truth checks a blue-green reindex candidate generation
#: is expected to satisfy before promotion (polylogue-t0m73's "reindex
#: acceptance gate"). Restricted to checks whose universe is satisfiable from
#: ``index.db`` alone -- a candidate generation directory holds only the new
#: ``index.db``, not the durable ``source.db``/``user.db``/``embeddings.db``
#: tiers (those live once at the archive root, not per-generation), so
#: cross-tier checks (``source-index-coverage``, ``blob-refs-liveness``,
#: ``embeddings-refs-liveness``, ``tier-schema``) would either report a false
#: ERROR (missing tiers they're not skip-tolerant of, e.g. tier-schema) or
#: only ever SKIP (uninformative). Intended use: ``verify_archive(generation_
#: root, checks=REINDEX_ACCEPTANCE_CHECKS)`` right before promotion, exactly
#: how :mod:`polylogue.maintenance.rebuild_index` already ran the ``fts-
#: parity`` singleton -- this constant widens that gate to the rest of the
#: ground-truth-eligible registry.
REINDEX_ACCEPTANCE_CHECKS: tuple[str, ...] = (
    "fts-parity",
    "lineage-sanity",
    "enum-superset-check",
    "session-lineage-acyclic",
    "message-count-projection",
    "session-fingerprint-stamps",
    "planner-stats",
)

#: Candidate checks that need the durable archive tiers as well as the
#: inactive generation's index. These are run with ``index_path_override`` so
#: they cannot silently fall back to the active/default index.
REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS: tuple[str, ...] = (
    "blob-reference-closure",
    "corpus-absences",
    "corpus-attachment-fidelity",
    "corpus-revision-fidelity",
    "pathology-zoo-invariants",
)

#: A partial reindex-canary candidate is intentionally not a complete source
#: replay, so full corpus-fidelity checks would reject every useful canary.
#: Its candidate-specific gate still runs the pathology-zoo contract against
#: the candidate index and the durable source tiers.
REINDEX_CANARY_ACCEPTANCE_CHECKS: tuple[str, ...] = ("pathology-zoo-invariants",)


def _select_check_specs(checks: Sequence[str] | None) -> tuple[ArchiveVerificationCheckSpec, ...]:
    if checks is None:
        return ARCHIVE_VERIFICATION_CHECKS
    selected_names = list(dict.fromkeys(checks))
    by_name = {spec.name: spec for spec in ARCHIVE_VERIFICATION_CHECKS}
    unknown = [name for name in selected_names if name not in by_name]
    if unknown:
        raise ValueError(
            f"unknown archive verification check(s): {', '.join(unknown)}; "
            f"available: {', '.join(ARCHIVE_VERIFICATION_CHECK_NAMES)}"
        )
    return tuple(by_name[name] for name in selected_names)


def verify_archive(
    archive_root: Path,
    *,
    checks: Sequence[str] | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    index_path_override: Path | None = None,
) -> ArchiveVerificationReport:
    """Run every selected archive-coherence check and return the aggregate report.

    Purely read-only: no check ever writes to the archive. Each check is
    independently wrapped so an unexpected exception -- including a locked
    or mid-rebuild tier file -- is reported as that check's own ``error``
    outcome rather than aborting the remaining checks. ``checks=None`` (the
    default) runs the full registry in :data:`ARCHIVE_VERIFICATION_CHECKS`
    order; a name not in the registry raises :class:`ValueError` immediately.
    ``index_path_override`` runs only checks with a candidate runner against
    that inactive generation's real index while retaining durable tiers from
    ``archive_root``.
    """
    specs = _select_check_specs(checks)
    if index_path_override is not None:
        unsupported = [spec.name for spec in specs if spec.candidate_run is None]
        if unsupported:
            raise ValueError("candidate index verification is unsupported for check(s): " + ", ".join(unsupported))
    # Typed at the OutcomeCheck base so this list assigns cleanly into
    # ArchiveVerificationReport.checks (inherited, invariant list[OutcomeCheck])
    # without a redundant narrower field redeclaration on the report dataclass.
    results: list[OutcomeCheck] = []
    for spec in specs:
        try:
            result = (
                spec.candidate_run(archive_root, index_path_override, sample_limit)
                if index_path_override is not None and spec.candidate_run is not None
                else spec.run(archive_root, sample_limit)
            )
        except Exception as exc:  # defense-in-depth: see module/function docstring
            logger.exception("archive verification check %s raised", spec.name)
            result = _error_check(spec.name, f"check raised {type(exc).__name__}: {exc}")
        # Stamp class + waiver metadata centrally so every check function
        # (and every _error_check/_skip_check escape hatch) picks it up
        # uniformly, rather than every one of the ~12 check bodies having to
        # remember to thread it through by hand.
        if isinstance(result, ArchiveVerificationCheck):
            result.check_class = spec.check_class.value
            if result.status is OutcomeStatus.ERROR:
                waiver = ARCHIVE_VERIFICATION_WAIVERS.get(spec.name)
                if waiver is not None:
                    result.waived_bead_id = waiver.bead_id
                    result.evidence.setdefault("waiver", {"bead_id": waiver.bead_id, "reason": waiver.reason})
        results.append(result)

    return ArchiveVerificationReport(
        checks=results,
        archive_root=str(archive_root),
        generated_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "ARCHIVE_VERIFICATION_CHECKS",
    "ARCHIVE_VERIFICATION_CHECK_NAMES",
    "ARCHIVE_VERIFICATION_WAIVERS",
    "CORPUS_FIDELITY_CHECKS",
    "REINDEX_CANARY_ACCEPTANCE_CHECKS",
    "REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS",
    "REINDEX_ACCEPTANCE_CHECKS",
    "ArchiveVerificationCheck",
    "ArchiveVerificationCheckClass",
    "ArchiveVerificationCheckSpec",
    "ArchiveVerificationReport",
    "ArchiveVerificationWaiver",
    "DEFAULT_SAMPLE_LIMIT",
    "verify_archive",
]
