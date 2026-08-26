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

Disposition = Literal["KEEP-WIRED", "PURGE", "REPLACE-DERIVE", "UNCLEAR"]


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
                disposition="KEEP-WIRED",
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
    if any(row.disposition == "UNCLEAR" for row in rows):
        problems.append("UNCLEAR dispositions must transfer to 20eld before closure")
    for row in rows:
        if row.disposition in {"PURGE", "REPLACE-DERIVE"} and not row.owner_60i5:
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
