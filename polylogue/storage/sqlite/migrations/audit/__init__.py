"""Numbered additive migrations for the durable audit tier.

The audit tier is currently at the archive-format floor (``user_version=1``),
which is the fresh-bootstrap version.  Its first in-place change therefore
owns slot ``002``.  That slot is intentionally not reserved by this package:
the schema owner must first advance ``ARCHIVE_VERSION_BY_TIER[AUDIT]`` and land
the matching fresh DDL.  Until then, the migration runner rejects a requested
target above v1 instead of silently accepting an untracked audit schema.

When the route is opened, the migration must be accompanied by a
``002.train.json`` sidecar with ``requires_backup=true`` and a non-empty,
verified ``backup_plan_ref`` (audit.db is durable and irreplaceable).  Recovery
is the normal durable-train path: restore the verified full-evidence backup,
reconcile any interrupted train, then retry through ``maintenance migrate-tier
audit``.  Do not add an SQL file without the version-authority change and
sidecar; the runner's contiguous-chain and fresh-DDL parity checks are the
guard against a half-established route.
"""

AUDIT_FIRST_MIGRATION_SLOT = 2
AUDIT_MIGRATION_BACKUP_PROFILE = "full_evidence"

__all__ = ["AUDIT_FIRST_MIGRATION_SLOT", "AUDIT_MIGRATION_BACKUP_PROFILE"]
