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
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus
from polylogue.logging import get_logger
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.table_existence import table_exists

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
    subclass-only ``evidence`` field via ``getattr`` so callers holding a
    plain ``OutcomeCheck``-typed reference (e.g. ``ArchiveVerificationReport
    .checks``, typed at its base-class element type) can still serialize a
    concrete :class:`ArchiveVerificationCheck` without a narrowing cast.
    """
    payload = dict(check.to_dict())
    payload["evidence"] = json_document(getattr(check, "evidence", {}))
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
        """True when at least one check reports ``error`` -- the gate condition."""
        return self.error_count > 0

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


@dataclass(frozen=True)
class ArchiveVerificationCheckSpec:
    """One named, independently runnable archive-coherence check."""

    name: str
    description: str
    run: ArchiveVerificationCheckFn


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
    source_path = _tier_path(archive_root, ArchiveTier.SOURCE)
    index_path = _resolve_index_path(archive_root)
    if not source_path.exists() or not index_path.exists():
        return _skip_check("source-index-coverage", "source.db or index.db not present")

    try:
        source_conn = _open_ro(source_path)
    except sqlite3.Error as exc:
        return _error_check("source-index-coverage", f"could not open source.db: {exc}", exc=exc)
    try:
        if table_exists(source_conn, "raw_membership_census"):
            censused_complete = {
                str(row[0])
                for row in source_conn.execute(
                    "SELECT raw_id FROM raw_membership_census WHERE status = 'complete' AND member_count > 0"
                )
            }
        else:
            censused_complete = set()
        all_raw_ids = {str(row[0]) for row in source_conn.execute("SELECT raw_id FROM raw_sessions")}
    except sqlite3.Error as exc:
        return _error_check("source-index-coverage", f"could not read source.db: {exc}", exc=exc)
    finally:
        source_conn.close()

    try:
        index_conn = _open_ro(index_path)
    except sqlite3.Error as exc:
        return _error_check("source-index-coverage", f"could not open index.db: {exc}", exc=exc)
    try:
        session_raw_ids = {
            str(row[0]) for row in index_conn.execute("SELECT DISTINCT raw_id FROM sessions WHERE raw_id IS NOT NULL")
        }
    except sqlite3.Error as exc:
        return _error_check("source-index-coverage", f"could not read index.db: {exc}", exc=exc)
    finally:
        index_conn.close()

    missing_work = sorted(censused_complete - session_raw_ids)
    orphans = sorted(session_raw_ids - all_raw_ids)

    status = OutcomeStatus.ERROR if (missing_work or orphans) else OutcomeStatus.OK
    summary = (
        f"{len(censused_complete):,} complete-census raw(s), {len(session_raw_ids):,} raw-backed session(s); "
        f"missing_work={len(missing_work):,} orphans={len(orphans):,}"
    )
    return ArchiveVerificationCheck(
        name="source-index-coverage",
        status=status,
        summary=summary,
        count=len(missing_work) + len(orphans),
        details=[
            *(f"missing-work:{raw_id}" for raw_id in missing_work[:sample_limit]),
            *(f"orphan:{raw_id}" for raw_id in orphans[:sample_limit]),
        ],
        evidence={
            "censused_complete_raw_count": len(censused_complete),
            "raw_backed_session_count": len(session_raw_ids),
            "missing_work_count": len(missing_work),
            "missing_work_sample": missing_work[:sample_limit],
            "orphan_count": len(orphans),
            "orphan_sample": orphans[:sample_limit],
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
        if table_exists(conn, "blocks") and table_exists(conn, "messages_fts_docsize"):
            row = conn.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE b.search_text != ''),
                    COUNT(d.id) FILTER (WHERE b.search_text != '')
                FROM blocks AS b
                LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                """
            ).fetchone()
            expected, indexed = int(row[0] or 0), int(row[1] or 0)
            gap = expected - indexed
            worst_sessions: list[dict[str, Any]] = []
            if gap:
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
                problems.append(f"messages_fts gap={gap}")
            evidence["messages_fts"] = {
                "expected": expected,
                "indexed": indexed,
                "gap": gap,
                "worst_sessions": worst_sessions,
            }
        else:
            evidence["messages_fts"] = None

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
# Registry + entrypoint
# ---------------------------------------------------------------------------

ARCHIVE_VERIFICATION_CHECKS: tuple[ArchiveVerificationCheckSpec, ...] = (
    ArchiveVerificationCheckSpec(
        "tier-schema",
        "Tier presence and PRAGMA user_version vs the canonical ARCHIVE_TIER_SPECS.",
        _check_tier_schema,
    ),
    ArchiveVerificationCheckSpec(
        "pointer-coherence",
        "Conventional index.db path vs the active .index-active-pointer generation (polylogue-k8kj class).",
        _check_pointer_coherence,
    ),
    ArchiveVerificationCheckSpec(
        "source-index-coverage",
        "Complete-census raw sessions with no materialized index session, and index sessions with no backing raw.",
        _check_source_index_coverage,
    ),
    ArchiveVerificationCheckSpec(
        "fts-parity",
        "messages_fts and blocks_command_trigram exactly cover their source rows, archive-wide.",
        _check_fts_parity,
    ),
    ArchiveVerificationCheckSpec(
        "lineage-sanity",
        "session_links.resolved_dst_session_id / branch_point_message_id resolve to real sessions/messages.",
        _check_lineage_sanity,
    ),
    ArchiveVerificationCheckSpec(
        "planner-stats",
        "sqlite_stat1 covers blocks/messages/action_pairs (polylogue-l3tk class, warn-level).",
        _check_planner_stats,
    ),
    ArchiveVerificationCheckSpec(
        "counts-summary",
        "Archive-wide session/message/block counts and origin breakdown (numbers-freeze starter).",
        _check_counts_summary,
    ),
)

ARCHIVE_VERIFICATION_CHECK_NAMES: tuple[str, ...] = tuple(spec.name for spec in ARCHIVE_VERIFICATION_CHECKS)


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
) -> ArchiveVerificationReport:
    """Run every selected archive-coherence check and return the aggregate report.

    Purely read-only: no check ever writes to the archive. Each check is
    independently wrapped so an unexpected exception -- including a locked
    or mid-rebuild tier file -- is reported as that check's own ``error``
    outcome rather than aborting the remaining checks. ``checks=None`` (the
    default) runs the full registry in :data:`ARCHIVE_VERIFICATION_CHECKS`
    order; a name not in the registry raises :class:`ValueError` immediately.
    """
    specs = _select_check_specs(checks)
    # Typed at the OutcomeCheck base so this list assigns cleanly into
    # ArchiveVerificationReport.checks (inherited, invariant list[OutcomeCheck])
    # without a redundant narrower field redeclaration on the report dataclass.
    results: list[OutcomeCheck] = []
    for spec in specs:
        try:
            results.append(spec.run(archive_root, sample_limit))
        except Exception as exc:  # defense-in-depth: see module/function docstring
            logger.exception("archive verification check %s raised", spec.name)
            results.append(_error_check(spec.name, f"check raised {type(exc).__name__}: {exc}"))

    return ArchiveVerificationReport(
        checks=results,
        archive_root=str(archive_root),
        generated_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "ARCHIVE_VERIFICATION_CHECKS",
    "ARCHIVE_VERIFICATION_CHECK_NAMES",
    "ArchiveVerificationCheck",
    "ArchiveVerificationCheckSpec",
    "ArchiveVerificationReport",
    "DEFAULT_SAMPLE_LIMIT",
    "verify_archive",
]
