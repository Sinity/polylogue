"""Verify schema-evolution policy boundaries.

Background
----------

Polylogue has two schema-evolution regimes:

* Durable tiers (``source.db``, ``user.db``, and ``audit.db``) may use explicit additive SQL
  migrations with a backup gate.
* Derived/rebuildable tiers (``index.db`` and ``embeddings.db``) do not use
  migration chains. They are rebuilt or blue-green replaced from durable source
  evidence, except for explicitly declared, clone-validated SQL plans.

What this lint checks
---------------------

1. Fail when the current index schema version lacks a delta-class declaration.
   Durable-tier migrations
   must live under ``polylogue/storage/sqlite/migrations/{source,user}/`` as
   numbered SQL resources.

2. Validate every entry in the index-tier benign-DDL convergence registry
   (``polylogue.storage.sqlite.archive_tiers.index_convergence.
   INDEX_BENIGN_DDL_REGISTRY``, polylogue-jc1b): each entry's SQL must be
   exactly one of ``CREATE TABLE IF NOT EXISTS``, ``CREATE INDEX IF NOT
   EXISTS``, or ``DROP TABLE IF EXISTS`` -- idempotent and data-non-
   transforming by construction. Anything else (a bare ``CREATE``/``DROP``
   missing its guard clause, ``ALTER TABLE``, or a data-mutating statement)
   is the sanctioned same-version-open path being asked to do something a
   version bump should gate instead, and is rejected here.

The lint validates structured schema carriers and executable SQL shapes. It
does not infer architecture from Python function names.

Wired into ``devtools verify --lab`` rather than the fast default path
because the policy boundary is a lab/architectural concern, not a
per-edit gate.

**Out of scope:** this lint is keyed entirely to ``INDEX_SCHEMA_VERSION``.
Parser and lowering drift use the production fingerprints declared by
``polylogue.sources.origin_specs`` instead: archive rows, candidate metadata,
and live-proof receipts carry those fingerprints, and archive verification
rejects stale or mixed values. A green run of *this* lint alone is therefore
not evidence that no reparse is needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from devtools import repo_root as _get_root
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index_convergence import (
    INDEX_BENIGN_DDL_REGISTRY,
    BenignDDLEntry,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DurableMigrationClaim,
    durable_change_train_policy_report,
    durable_migration_claim_for_sql,
)
from polylogue.storage.sqlite.durable_change_train import (
    durable_migration_collision_report as _durable_migration_collision_report,
)
from polylogue.storage.sqlite.lifecycle import IndexDeltaDeclarationReport, index_delta_declaration_report

ROOT = _get_root()
MIGRATIONS_DIR = ROOT / "polylogue" / "storage" / "sqlite" / "migrations"
ALLOWED_MIGRATION_TIERS = {"source", "user", "audit"}

_DURABLE_MIGRATION_SQL_RE = re.compile(r"^\d{3,}_[a-z0-9_]+\.sql$")
_DURABLE_MIGRATION_SIDECAR_RE = re.compile(r"^\d{3,}\.train\.json$")


# Index-tier benign-DDL registry entries (polylogue-jc1b) must be exactly one
# of these idempotent shapes -- the whole point is that re-applying an entry
# on every same-version open is always a no-op past the first time.
_ALLOWED_BENIGN_DDL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)^\s*CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s"),
    re.compile(r"(?is)^\s*CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s"),
    re.compile(r"(?is)^\s*DROP\s+TABLE\s+IF\s+EXISTS\s"),
)

# Any of these appearing anywhere in an entry's SQL means it mutates row data
# (not merely schema) or alters an existing object in place -- both outside
# the "idempotent, data-non-transforming" class a same-version-open hook may
# apply without an INDEX_SCHEMA_VERSION bump.
_FORBIDDEN_BENIGN_DDL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)\bALTER\s+TABLE\b"),
    re.compile(r"(?is)\bINSERT\s+INTO\b"),
    re.compile(r"(?is)\bUPDATE\s+\S"),
    re.compile(r"(?is)\bDELETE\s+FROM\b"),
)


@dataclass(frozen=True, slots=True)
class BenignDDLViolation:
    entry_name: str
    reason: str


def _invalid_benign_ddl_entries(
    entries: Iterable[BenignDDLEntry] = INDEX_BENIGN_DDL_REGISTRY,
) -> list[BenignDDLViolation]:
    """Return every registry entry whose SQL is not an allowed idempotent shape."""
    violations: list[BenignDDLViolation] = []
    for entry in entries:
        sql = entry.sql.strip()
        # Reject multi-statement smuggling -- a registry entry is exactly one
        # DDL statement (an optional single trailing semicolon is fine).
        body = sql[:-1] if sql.endswith(";") else sql
        if ";" in body:
            violations.append(BenignDDLViolation(entry.name, "registry entry must be exactly one statement"))
            continue
        if not any(pattern.match(sql) for pattern in _ALLOWED_BENIGN_DDL_PATTERNS):
            violations.append(
                BenignDDLViolation(
                    entry.name,
                    "must be CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS / DROP TABLE IF EXISTS",
                )
            )
            continue
        for forbidden in _FORBIDDEN_BENIGN_DDL_PATTERNS:
            if forbidden.search(sql):
                violations.append(
                    BenignDDLViolation(entry.name, f"data-transforming or non-idempotent clause: {forbidden.pattern}")
                )
                break
    return violations


def _invalid_migration_paths() -> list[Path]:
    if not MIGRATIONS_DIR.exists():
        return []
    invalid: list[Path] = []
    for path in sorted(MIGRATIONS_DIR.rglob("*")):
        if path.is_dir() or path.name == "__init__.py":
            continue
        rel = path.relative_to(MIGRATIONS_DIR)
        if (
            len(rel.parts) != 2
            or rel.parts[0] not in ALLOWED_MIGRATION_TIERS
            or not (
                _DURABLE_MIGRATION_SQL_RE.fullmatch(rel.parts[1])
                or _DURABLE_MIGRATION_SIDECAR_RE.fullmatch(rel.parts[1])
            )
        ):
            invalid.append(path)
    return invalid


def _durable_migration_claims_on_disk() -> tuple[DurableMigrationClaim, ...]:
    """Build the shared migration-slot claims from checked-in SQL resources."""
    claims: list[DurableMigrationClaim] = []
    for tier_name in sorted(ALLOWED_MIGRATION_TIERS):
        tier = ArchiveTier(tier_name)
        tier_dir = MIGRATIONS_DIR / tier_name
        if not tier_dir.exists():
            continue
        for path in sorted(tier_dir.glob("*.sql")):
            if re.fullmatch(r"\d{3,}_[a-z0-9_]+\.sql", path.name) is None:
                continue
            claims.append(
                durable_migration_claim_for_sql(
                    tier,
                    path.relative_to(ROOT),
                    path.read_text(encoding="utf-8"),
                    owner_ref=str(path.relative_to(ROOT)),
                )
            )
    return tuple(claims)


def durable_migration_collision_report(
    claims: Iterable[DurableMigrationClaim],
) -> dict[str, object]:
    """Expose the shared slot report to schema-policy callers and tests."""
    return _durable_migration_collision_report(claims)


def _format_report(
    *,
    invalid_migrations: list[Path],
    delta_report: IndexDeltaDeclarationReport,
    benign_ddl_violations: list[BenignDDLViolation],
    durable_change_train_reports: dict[str, dict[str, object]] | None = None,
    durable_migration_collisions: dict[str, object] | None = None,
) -> str:
    durable_change_train_reports = durable_change_train_reports or {}
    durable_violations = [
        violation
        for report in durable_change_train_reports.values()
        for violation in cast(tuple[object, ...], report.get("violations", ()))
        if isinstance(violation, str)
    ]
    durable_reservations = [
        reservation
        for report in durable_change_train_reports.values()
        for reservation in cast(tuple[object, ...], report.get("reservations", ()))
    ]
    durable_migration_collisions = durable_migration_collisions or {}
    collision_entries = cast(tuple[object, ...], durable_migration_collisions.get("collisions", ()))
    lines = [
        f"invalid durable migration resources found: {len(invalid_migrations)}",
        f"durable change-train reservations found: {len(durable_reservations)}",
        f"durable change-train violations found: {len(durable_violations)}",
        f"durable migration slot collisions found: {len(collision_entries)}",
        f"undeclared index schema deltas found: {len(delta_report['missing_versions'])}",
        f"invalid index benign-DDL registry entries found: {len(benign_ddl_violations)}",
    ]
    if invalid_migrations:
        lines.append("")
        lines.append("Invalid migration resources:")
        for path in invalid_migrations:
            lines.append(f"  {path.relative_to(ROOT)}")
    if not bool(delta_report["ok"]):
        lines.append("")
        lines.append("Index fast-forward declaration drift:")
        lines.append(f"  compatibility floor: v{delta_report['compatibility_floor']}")
        lines.append(f"  missing: {list(delta_report['missing_versions'])}")
        lines.append(f"  duplicate: {list(delta_report['duplicate_versions'])}")
        lines.append(f"  invalid: {list(delta_report['invalid_versions'])}")
        lines.append("")
        lines.append("Policy violation: each index schema bump needs a declared delta class.")
    if benign_ddl_violations:
        lines.append("")
        lines.append("Invalid index benign-DDL registry entries:")
        for violation in benign_ddl_violations:
            lines.append(f"  {violation.entry_name}: {violation.reason}")
        lines.append("")
        lines.append(
            "Policy violation: index-tier same-version convergence entries must be idempotent, "
            "data-non-transforming CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS / "
            "DROP TABLE IF EXISTS statements."
        )
    if durable_reservations:
        lines.append("")
        lines.append("Durable change-train reservations:")
        lines.extend(f"  {reservation}" for reservation in durable_reservations)
    if durable_violations:
        lines.append("")
        lines.append("Durable change-train violations:")
        lines.extend(f"  {violation}" for violation in durable_violations)
    if collision_entries:
        lines.append("")
        lines.append("Durable migration slot collisions:")
        lines.extend(f"  {collision}" for collision in collision_entries)
    if (
        not invalid_migrations
        and bool(delta_report["ok"])
        and not benign_ddl_violations
        and not durable_violations
        and not collision_entries
    ):
        lines.append("")
        lines.append("Schema evolution policy intact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    invalid_migrations = _invalid_migration_paths()
    durable_change_train_reports = {
        tier.value: durable_change_train_policy_report(tier)
        for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
    }
    durable_migration_collisions = durable_migration_collision_report(_durable_migration_claims_on_disk())
    delta_report = index_delta_declaration_report(INDEX_SCHEMA_VERSION)
    benign_ddl_violations = _invalid_benign_ddl_entries()

    ok = (
        not invalid_migrations
        and bool(delta_report["ok"])
        and not benign_ddl_violations
        and all(bool(report.get("ok")) for report in durable_change_train_reports.values())
        and bool(durable_migration_collisions["ok"])
    )

    if args.json:
        payload = {
            "invalid_migration_resources": [str(path.relative_to(ROOT)) for path in invalid_migrations],
            "durable_change_trains": durable_change_train_reports,
            "durable_migration_collisions": durable_migration_collisions,
            "index_delta_declarations": delta_report,
            "invalid_benign_ddl_entries": [
                {"name": violation.entry_name, "reason": violation.reason} for violation in benign_ddl_violations
            ],
            "ok": ok,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            _format_report(
                invalid_migrations=invalid_migrations,
                delta_report=delta_report,
                benign_ddl_violations=benign_ddl_violations,
                durable_change_train_reports=durable_change_train_reports,
                durable_migration_collisions=durable_migration_collisions,
            )
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
