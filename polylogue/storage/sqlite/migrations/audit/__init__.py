"""Future numbered additive migrations for the durable audit tier.

The current fresh archive format creates audit.db at version 1, with the
operation lookup index already present in canonical DDL. A future audit
migration starts at slot 002 and must advance the version authority alongside
fresh DDL so migrated and new audit files have the same inventory.

Every future route must have a matching train sidecar with
``requires_backup=true`` and a non-empty, verified ``backup_plan_ref``
(``audit.db`` is durable and irreplaceable). Recovery uses the normal
durable-train path. Do not add an SQL file without the version-authority
change and sidecar; contiguous-chain and fresh-DDL parity checks guard against
a half-established route.
"""

AUDIT_FIRST_MIGRATION_SLOT = 2
AUDIT_MIGRATION_BACKUP_PROFILE = "full_evidence"

__all__ = ["AUDIT_FIRST_MIGRATION_SLOT", "AUDIT_MIGRATION_BACKUP_PROFILE"]
