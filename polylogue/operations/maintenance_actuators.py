"""Maintenance actuators for blob-reference debt repair.

These live outside :mod:`polylogue.operations.mutation_actuators` on purpose.
That module is inside the derived-schema identity closure (an AST closure over
imported source), so an actuator there that reaches
:mod:`polylogue.storage.blob_integrity` would drag the whole blob-integrity
module into the closure and make every edit to it a schema-identity move.
These two actuators are reached only from
:mod:`polylogue.operations.daemon_mutations`, which is outside the closure,
so the repair routines stay ordinary code. ``devtools gate schema-closure``
enforces exactly this: it refused the addition when they lived next door.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    _FailClosedRecovery,
    build_plan,
    make_target_ref,
)

__all__ = [
    "BlobReferenceOrphanPruneActuator",
    "BlobReferenceOrphanPruneArgs",
    "BlobReferenceSourceReplaceActuator",
    "BlobReferenceSourceReplaceArgs",
]


@dataclass(frozen=True, slots=True)
class BlobReferenceSourceReplaceArgs:
    """Exact current-source replacement request for raw-backed blob-ref debt."""

    archive_root: Path
    manifest_path: Path
    max_count: int | None = None
    sample_size: int = 30


@dataclass(frozen=True, slots=True)
class BlobReferenceSourceReplaceActuator(_FailClosedRecovery):
    """Repoint raw-backed missing blob refs at current source-derived bytes.

    ``ops maintenance blob-reference-replace-from-source`` used to call
    ``replace_raw_backed_blob_reference_debt_from_source(dry_run=False)``
    inside the CLI process, which ``UPDATE``s ``raw_sessions`` and commits
    against the durable source tier. The storage routine asserts a write lease,
    but lease enforcement is only armed inside the daemon, so outside it the
    assertion returned ``None`` and the write proceeded beside a live daemon.

    ``prepare`` runs the same candidate resolution in dry-run mode, so it is
    read-only and re-runnable, and the plan records what APPLY expects to find.
    """

    operation: str = "mutate-replace-blob-refs-from-source"
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: BlobReferenceSourceReplaceArgs) -> MutationPlan:
        from polylogue.storage.blob_integrity import replace_raw_backed_blob_reference_debt_from_source

        report = replace_raw_backed_blob_reference_debt_from_source(
            args.archive_root / "source.db",
            dry_run=True,
            manifest_path=None,
            max_count=args.max_count,
            sample_size=args.sample_size,
        )
        return build_plan(
            operation=self.operation,
            destructive_class=self.destructive_class,
            target_refs=(make_target_ref("source", "blob-refs:replace-from-source"),),
            affected_tiers=("source", "audit"),
            reversible=False,
            context={
                "scanned_rows": report.scanned_rows,
                "candidate_rows": report.candidate_rows,
                "max_count": args.max_count,
                "manifest_path": str(args.manifest_path),
            },
        )

    def apply(self, plan: MutationPlan, args: BlobReferenceSourceReplaceArgs) -> MutationReceipt:
        from polylogue.storage.blob_integrity import replace_raw_backed_blob_reference_debt_from_source

        report = replace_raw_backed_blob_reference_debt_from_source(
            args.archive_root / "source.db",
            dry_run=False,
            manifest_path=args.manifest_path,
            max_count=args.max_count,
            sample_size=args.sample_size,
        )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if report.replaced_rows else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=report.replaced_rows,
            detail=None if report.replaced_rows else "no_replaceable_rows",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt=report.to_dict(),
        )


@dataclass(frozen=True, slots=True)
class BlobReferenceOrphanPruneArgs:
    """Exact quarantine-backed prune request for orphan blob-ref debt."""

    archive_root: Path
    quarantine_path: Path | None = None
    max_count: int | None = None
    sample_size: int = 30


@dataclass(frozen=True, slots=True)
class BlobReferenceOrphanPruneActuator(_FailClosedRecovery):
    """Quarantine and delete missing ``blob_refs`` with no surviving raw row.

    Same predecessor shape as :class:`BlobReferenceSourceReplaceActuator`:
    ``ops maintenance blob-reference-prune-orphans`` ran
    ``prune_orphan_blob_reference_debt(dry_run=False)`` in the CLI's own
    process, which ``DELETE``s from ``blob_refs`` and commits against the
    durable source tier. The storage routine writes the quarantine JSONL
    before the delete, and that ordering is preserved -- only the executor
    moves.
    """

    operation: str = "mutate-prune-orphan-blob-refs"
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: BlobReferenceOrphanPruneArgs) -> MutationPlan:
        from polylogue.storage.blob_integrity import prune_orphan_blob_reference_debt

        report = prune_orphan_blob_reference_debt(
            args.archive_root / "source.db",
            dry_run=True,
            quarantine_path=None,
            max_count=args.max_count,
            sample_size=args.sample_size,
        )
        return build_plan(
            operation=self.operation,
            destructive_class=self.destructive_class,
            target_refs=(make_target_ref("source", "blob-refs:orphan-prune"),),
            affected_tiers=("source", "audit"),
            reversible=False,
            context={
                "scanned_blob_refs": report.scanned_blob_refs,
                "missing_orphan_refs": report.missing_orphan_refs,
                "max_count": args.max_count,
                "quarantine_path": None if args.quarantine_path is None else str(args.quarantine_path),
            },
        )

    def apply(self, plan: MutationPlan, args: BlobReferenceOrphanPruneArgs) -> MutationReceipt:
        from polylogue.storage.blob_integrity import prune_orphan_blob_reference_debt

        report = prune_orphan_blob_reference_debt(
            args.archive_root / "source.db",
            dry_run=False,
            quarantine_path=args.quarantine_path,
            max_count=args.max_count,
            sample_size=args.sample_size,
        )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if report.pruned_refs else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=report.pruned_refs,
            detail=None if report.pruned_refs else "no_orphan_refs",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt=report.to_dict(),
        )
