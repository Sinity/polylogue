"""Canonical column dispositions for the durable audit tier.

The DDL remains the denominator.  This module supplies the evidence fields
that a disposition review must carry for every declared audit column and
rejects copied or incomplete inventories before they can be treated as a
complete review.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL

Disposition = Literal["KEEP", "COMPLETE", "PURGE", "DERIVE", "TRANSITION"]


@dataclass(frozen=True, slots=True)
class AuditColumnDisposition:
    table: str
    column: str
    disposition: Disposition
    writer: str
    reader: str
    authority_role: str
    retention: str
    continuity_or_receipt: str
    live_denominator: str
    evidence: str
    owner_60i5: str | None = None

    @property
    def ref(self) -> str:
        return f"audit.{self.table}.{self.column}"


_TABLE_EVIDENCE: dict[str, tuple[str, str, str, str, str, str, str]] = {
    "archive_authority": (
        "operations/audit.py:ensure_archive_authority",
        "operations/audit.py:archive_authority",
        "archive identity authority",
        "irreplaceable; backup required",
        "archive identity used by receipts and restore",
        "archive_authority row count",
        "audit.py authority initialization and status readers",
    ),
    "operation_previews": (
        "operations/preview.py and operations/audit.py",
        "operations/audit.py preview readers and maintenance status",
        "pre-mutation plan authority",
        "irreplaceable; append-only lifecycle evidence",
        "preview reference for authorization and recovery receipts",
        "operation_previews row count",
        "preview creation, stale/consume checks, and status projections",
    ),
    "operation_preview_targets": (
        "operations/preview.py",
        "operations/audit.py target hydration and recovery inspection",
        "planned target and recovery-policy authority",
        "irreplaceable; retained with parent preview",
        "target receipt and recovery semantics",
        "operation_preview_targets row count",
        "preview target hydration and operation recovery paths",
    ),
    "operation_preview_capabilities": (
        "operations/preview.py",
        "operations/audit.py authorization capability checks",
        "capability requirement authority",
        "irreplaceable; retained with parent preview",
        "authorization admission evidence",
        "operation_preview_capabilities row count",
        "capability set reconstruction for authorization",
    ),
    "operation_authorizations": (
        "operations/audit.py authorization writer",
        "operations/audit.py authorization validation and recovery",
        "operator authorization authority",
        "irreplaceable; backup and legal deletion governed",
        "authorization receipt binds preview to actor and token",
        "operation_authorizations row count",
        "token/state/expiry validation and restore proof",
    ),
    "operation_authorization_capabilities": (
        "operations/audit.py authorization writer",
        "operations/audit.py capability admission readers",
        "authorized capability authority",
        "irreplaceable; retained with authorization",
        "authorization receipt capability set",
        "operation_authorization_capabilities row count",
        "capability hydration and audit replay",
    ),
    "operation_runs": (
        "operations/audit.py run writer and mutation transaction",
        "operations/audit.py run/status/recovery readers",
        "operation lifecycle authority",
        "irreplaceable; append-only lifecycle evidence",
        "run receipt and terminal recovery state",
        "operation_runs row count",
        "run lifecycle, idempotency, status, and public status projections",
    ),
    "operation_run_capabilities": (
        "operations/audit.py run writer",
        "operations/audit.py run capability readers",
        "run capability authority",
        "irreplaceable; retained with operation run",
        "run admission receipt",
        "operation_run_capabilities row count",
        "run hydration and recovery inspection",
    ),
    "operation_targets": (
        "operations/audit.py target writer",
        "operations/audit.py target/recovery readers",
        "per-target lifecycle authority",
        "irreplaceable; append-only target evidence",
        "target state and domain receipt linkage",
        "operation_targets row count",
        "target recovery, acknowledgement, and receipt verification",
    ),
    "operation_attempts": (
        "operations/audit.py attempt writer",
        "operations/audit.py attempt/recovery readers",
        "worker attempt and lease authority",
        "irreplaceable; retained for incident reconstruction",
        "attempt receipt and interrupted-work continuity",
        "operation_attempts row count",
        "lease, retry, interruption, and reconciliation readers",
    ),
    "operation_events": (
        "operations/audit.py event append writer",
        "operations/audit.py event timeline and recovery readers",
        "append-only operation timeline authority",
        "irreplaceable; append-only, backup required",
        "ordered incident and state-transition evidence",
        "operation_events row count",
        "timeline reconstruction and receipt verification",
    ),
    "audit_continuity_head": (
        "storage/sqlite/audit_continuity.py",
        "storage/sqlite/audit_continuity.py and migration/recovery readers",
        "audit hash-chain continuity authority",
        "irreplaceable; append-only continuity head",
        "head generation and hash verify every audit mutation",
        "audit_continuity_head singleton row",
        "continuity advance, startup recovery, backup, and restore proof",
    ),
}

# The two capability tables share the same row shape.  Keeping their policy
# explicit avoids a second hand-maintained column list while retaining a
# table-specific, DDL-derived denominator.


def canonical_audit_columns(ddl: str = AUDIT_DDL) -> tuple[tuple[str, str], ...]:
    """Return the exact table/column denominator from canonical audit DDL."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(ddl)
        return tuple(
            (table, str(row[1]))
            for (table,) in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        )
    finally:
        connection.close()


def audit_column_dispositions() -> tuple[AuditColumnDisposition, ...]:
    """Build all audit dispositions from DDL columns and table evidence."""
    rows: list[AuditColumnDisposition] = []
    for table, column in canonical_audit_columns():
        try:
            writer, reader, authority, retention, continuity, denominator, evidence = _TABLE_EVIDENCE[table]
        except KeyError as exc:
            raise ValueError(f"audit table has no disposition evidence: {table}") from exc
        rows.append(
            AuditColumnDisposition(
                table=table,
                column=column,
                disposition="KEEP",
                writer=writer,
                reader=reader,
                authority_role=authority,
                retention=retention,
                continuity_or_receipt=continuity,
                live_denominator=denominator,
                evidence=evidence,
            )
        )
    return tuple(rows)


def assert_complete_audit_disposition(
    rows: Sequence[AuditColumnDisposition],
    *,
    ddl: str = AUDIT_DDL,
) -> None:
    """Fail closed on omitted, duplicate, extra, or unresolved audit fields."""
    refs = [row.ref for row in rows]
    duplicates = sorted({ref for ref in refs if refs.count(ref) > 1})
    expected = {f"audit.{table}.{column}" for table, column in canonical_audit_columns(ddl)}
    actual = set(refs)
    problems: list[str] = []
    if duplicates:
        problems.append(f"duplicate columns: {', '.join(duplicates)}")
    if missing := sorted(expected - actual):
        problems.append(f"omitted columns: {', '.join(missing)}")
    if extra := sorted(actual - expected):
        problems.append(f"unclassified columns: {', '.join(extra)}")
    if any(row.disposition not in {"KEEP", "COMPLETE", "PURGE", "DERIVE", "TRANSITION"} for row in rows):
        problems.append("dispositions must use KEEP, COMPLETE, PURGE, DERIVE, or TRANSITION")
    for row in rows:
        if row.disposition in {"PURGE", "DERIVE", "TRANSITION"} and not row.owner_60i5:
            problems.append(f"{row.ref} lacks 60i5 copy-forward owner")
    if problems:
        raise ValueError("incomplete audit schema disposition: " + "; ".join(problems))


__all__ = [
    "AuditColumnDisposition",
    "Disposition",
    "assert_complete_audit_disposition",
    "audit_column_dispositions",
    "canonical_audit_columns",
]


@dataclass(frozen=True, slots=True)
class SchemaDisposition:
    """One code-owned decision for a canonical six-tier schema object."""

    object_ref: str
    tier: str
    object_type: str
    table_name: str
    name: str
    disposition: Disposition
    semantic_owner: str
    evidence: str
    tier_durability: str
    reindex_timing: str
    implementation_bead: str


_PURGE_TABLES = frozenset(
    {
        "raw_append_chain_backfill_receipts",
        "raw_authority_verdicts",
        "raw_byte_duplicate_supersession_receipts",
        "raw_failure_disposition_receipts",
        "raw_live_source_reconciliation_receipts",
        "raw_membership_writeback_receipts",
        "raw_non_session_duplicate_exclusion_receipts",
        "raw_quarantine_group_dedup_receipts",
        "raw_unknown_export_reclassification_receipts",
        "history_sidecars",
    }
)


def _object_decision(obj: object) -> SchemaDisposition:
    """Assign policy from object identity while leaving the denominator to DDL."""
    tier = obj.tier.value  # type: ignore[attr-defined]
    table_name = obj.table_name  # type: ignore[attr-defined]
    name = obj.name  # type: ignore[attr-defined]
    object_type = obj.object_type  # type: ignore[attr-defined]
    disposition: Disposition = "KEEP"
    owner = "canonical schema owner"
    evidence = "reachable canonical declaration; live counts are evidence only"
    bead = "polylogue-20eld"
    if table_name in _PURGE_TABLES:
        disposition = "PURGE"
        owner = "historical repair retirement"
        bead = "2x6xu"
        evidence = "positive retirement decision removes the historical repair receipt family"
    elif table_name == "excised_content":
        disposition = "TRANSITION"
        owner = "durable excision authority"
        bead = "1sb32"
        evidence = "audit-backed excision evidence is sole authority; source keeps only non-resurrection identity"
    elif tier == "index" and table_name == "threads" and name == "dominant_repo_id":
        disposition = "PURGE"
        owner = "thread repository projection"
        bead = "polylogue-20eld"
        evidence = (
            "no production writer, reader, hydration, or intended identity join; dominant_repo remains authoritative"
        )
    elif tier == "user" and table_name in {"result_set_holdout_policies", "holdout_access_receipts"}:
        disposition = "COMPLETE"
        owner = "holdout evaluation workflow"
        bead = "polylogue-arik"
        evidence = "wired production route exists; zero live rows is not retirement evidence"
    elif tier == "source" and table_name.startswith("sinex_publication_"):
        disposition = "COMPLETE"
        owner = "Sinex publication workflow"
        bead = "ofry"
        evidence = "unfinished wanted publication capability has a named production owner"
    elif object_type == "table" and obj.virtual:  # type: ignore[attr-defined]
        disposition = "DERIVE"
        owner = "derived index convergence"
        bead = "g0s6k.2"
        evidence = "virtual relation is regenerated from its canonical source relation"
    return SchemaDisposition(
        object_ref=obj.object_ref,  # type: ignore[attr-defined]
        tier=tier,
        object_type=object_type,
        table_name=table_name,
        name=name,
        disposition=disposition,
        semantic_owner=owner,
        evidence=evidence,
        tier_durability={
            "source": "durable",
            "index": "rebuildable",
            "embeddings": "expensive_rebuild",
            "user": "durable-irreplaceable",
            "audit": "durable-append-only",
            "ops": "disposable",
        }[tier],
        reindex_timing="authenticated durable train"
        if tier in {"source", "user", "audit"}
        else "next derived convergence",
        implementation_bead=bead,
    )


def schema_dispositions() -> tuple[SchemaDisposition, ...]:
    """Generate the complete six-tier disposition from canonical declarations."""
    from polylogue.storage.sqlite.archive_tiers.schema_inventory import canonical_schema_objects
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    return tuple(_object_decision(obj) for tier in ArchiveTier for obj in canonical_schema_objects(tier))


def assert_complete_schema_dispositions(rows: Sequence[SchemaDisposition]) -> None:
    """Fail closed if a candidate is built from an undeclared or unresolved object."""
    from polylogue.storage.sqlite.archive_tiers.schema_inventory import canonical_schema_objects
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    expected = {obj.object_ref for tier in ArchiveTier for obj in canonical_schema_objects(tier)}
    refs = [row.object_ref for row in rows]
    problems: list[str] = []
    if len(refs) != len(set(refs)):
        problems.append("duplicate schema objects")
    if missing := sorted(expected - set(refs)):
        problems.append(f"omitted schema objects: {', '.join(missing)}")
    if extra := sorted(set(refs) - expected):
        problems.append(f"undeclared schema objects: {', '.join(extra)}")
    if any(row.disposition not in {"KEEP", "COMPLETE", "PURGE", "DERIVE", "TRANSITION"} for row in rows):
        problems.append("UNCLEAR or unsupported schema disposition")
    if any(not row.semantic_owner or not row.implementation_bead for row in rows):
        problems.append("schema disposition lacks owner")
    if problems:
        raise ValueError("incomplete six-tier schema disposition: " + "; ".join(problems))


def schema_disposition_report() -> dict[str, object]:
    """Return a generated review projection without storing live or campaign state."""
    rows = schema_dispositions()
    assert_complete_schema_dispositions(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.disposition] = counts.get(row.disposition, 0) + 1
    return {
        "format": "polylogue-schema-disposition/v1",
        "complete": True,
        "object_count": len(rows),
        "disposition_counts": counts,
        "objects": [
            {
                "object_ref": row.object_ref,
                "tier": row.tier,
                "object_type": row.object_type,
                "table_name": row.table_name,
                "name": row.name,
                "disposition": row.disposition,
                "semantic_owner": row.semantic_owner,
                "evidence": row.evidence,
                "tier_durability": row.tier_durability,
                "reindex_timing": row.reindex_timing,
                "implementation_bead": row.implementation_bead,
            }
            for row in rows
        ],
    }


__all__ += [
    "SchemaDisposition",
    "assert_complete_schema_dispositions",
    "schema_disposition_report",
    "schema_dispositions",
]
