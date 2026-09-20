"""Shared-kernel declarations for the maintenance command family.

``polylogue/cli/commands/maintenance/__init__.py`` used to carry a
hand-maintained four-tuple table -- ``(cli name, submodule, attribute, short
help)`` plus a separate ``_NESTED_GROUP_COMMANDS`` set -- whose entries were
resolved lazily by string at dispatch time.  A renamed or deleted command
function stayed invisible until someone actually ran that subcommand, and the
nested-group set was a second site to forget.

This module is now the single declaration site.  The CLI group derives its
registration table from :data:`MAINTENANCE_COMMAND_DECLARATIONS`, and every
declared handler symbol, owner path, output target, and example is resolved by
``devtools gate declaration-bindings`` against the live checkout, so the break
is reported once, with its repair command, instead of at dispatch.

The module deliberately imports nothing from the maintenance runtime: the CLI
group is imported on *any* ``ops maintenance ...`` invocation, including a bare
``--help`` listing, and must not pull in ``ArchiveStore``, ``blob_gc``, or the
migration runner (polylogue-sod7).  Declarations are strings and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
    validate_registry,
)

REPAIR_COMMAND: Final = "devtools test tests/unit/cli/test_maintenance_registration.py"

#: The Click group every declared command is registered on.
MAINTENANCE_CONSUMER: Final = "polylogue.cli.commands.maintenance.maintenance_group"

_PACKAGE: Final = "polylogue.cli.commands.maintenance"
_PACKAGE_PATH: Final = "polylogue/cli/commands/maintenance"


@dataclass(frozen=True, slots=True)
class MaintenanceCommandDeclaration:
    """One maintenance CLI command and its shared-kernel record."""

    kernel: DeclarationSpec
    cli_name: str
    submodule: str
    attribute: str
    short_help: str
    nested_group: bool = False

    @property
    def module(self) -> str:
        """Dotted module path the lazy Click command resolves at dispatch."""

        return f"{_PACKAGE}.{self.submodule}"

    @property
    def invocation(self) -> tuple[str, ...]:
        """The argv path that reaches this command through the real CLI."""

        return ("ops", "maintenance", self.cli_name)


def _command(
    cli_name: str,
    submodule: str,
    attribute: str,
    short_help: str,
    *,
    nested_group: bool = False,
) -> MaintenanceCommandDeclaration:
    owner_path = f"{_PACKAGE_PATH}/{submodule}.py"
    producer = f"{_PACKAGE}.{submodule}.{attribute}"
    kernel = DeclarationSpec(
        declaration_id=f"maintenance.command.{cli_name}",
        family_id="maintenance.command",
        public_name=f"ops maintenance {cli_name}",
        owner_path=owner_path,
        compatibility=CompatibilityKey(
            identity="maintenance-command",
            lifecycle="stable",
            authority="operator-cli",
            access_result_shape="click-command",
            durability="guarded-maintenance",
        ),
        producer=producer,
        role_gate="cli.ops.maintenance",
        schema_ref=f"click.Group:{attribute}" if nested_group else f"click.Command:{attribute}",
        discovery_text=short_help,
        repair_command=REPAIR_COMMAND,
        handlers=(
            HandlerBinding(
                surface="cli",
                owner_path=owner_path,
                symbol=attribute,
                binding_key=f"ops maintenance {cli_name}",
            ),
        ),
        outputs=(
            OutputSpec(
                name="cli-command",
                kind="click-group" if nested_group else "click-command",
                schema_ref="polylogue.cli.click_command_registration._NestedLazyGroup"
                if nested_group
                else "polylogue.cli.click_command_registration._LazyCommand",
                target_path=f"ops maintenance {cli_name}",
            ),
        ),
        # The example is the real argv path. ``tests/unit/cli/
        # test_maintenance_declarations.py`` runs every one of them through the
        # production Click tree, so an example naming a command the CLI does
        # not expose fails rather than documenting a command that never ran.
        examples=(
            ExampleSpec(
                name="help",
                summary=f"Resolve `ops maintenance {cli_name}` through the production CLI tree.",
                arguments=(("argv", ("ops", "maintenance", cli_name, "--help")),),
            ),
        ),
        completeness_edges=(
            CompletenessEdge(
                producer=producer,
                consumer=MAINTENANCE_CONSUMER,
                kind="click-registration",
                owner_path=f"{_PACKAGE_PATH}/__init__.py",
            ),
        ),
    )
    return MaintenanceCommandDeclaration(
        kernel=kernel,
        cli_name=cli_name,
        submodule=submodule,
        attribute=attribute,
        short_help=short_help,
        nested_group=nested_group,
    )


MAINTENANCE_COMMAND_DECLARATIONS: Final[tuple[MaintenanceCommandDeclaration, ...]] = (
    _command(
        "beads-origin-census",
        "_beads_origin_census",
        "beads_origin_census_command",
        "Read-only census and exact plan for retired Beads-origin evidence.",
    ),
    _command("archive-plan", "_archive_plan", "archive_plan_command", "Inspect readiness for the archive file set."),
    _command(
        "backup-plan",
        "_backup_plan",
        "backup_plan_command",
        "Inspect archive backup boundaries without copying data.",
    ),
    _command(
        "assertion-export",
        "_assertion_export",
        "assertion_export_command",
        "Export the durable assertion substrate from user.db.",
    ),
    _command("archive-read", "_archive_read", "archive_read_command", "Read index sessions from the archive."),
    _command(
        "archive-init",
        "_archive_plan",
        "archive_init_command",
        "Initialize the archive file set after explicit confirmation.",
    ),
    _command(
        "migrate-tier",
        "_migrate_tier",
        "migrate_tier_command",
        "Apply additive migrations for one durable archive tier.",
    ),
    _command(
        "raw-authority-frontier",
        "_raw_identity",
        "raw_authority_frontier_command",
        "Inspect and record the raw-authority frontier; plan application is daemon-owned.",
    ),
    _command(
        "raw-authority-blockers",
        "_raw_identity",
        "raw_authority_blockers_command",
        "List unresolved raw-authority blockers (frontier-judgment vs frontier-obligation). Read-only.",
    ),
    _command(
        "raw-authority-blocker-resolve",
        "_raw_identity",
        "raw_authority_blocker_resolve_command",
        "Resolve one durable frontier blocker against current source evidence.",
    ),
    _command(
        "operation-recovery",
        "_operation_recovery",
        "operation_recovery_command",
        "Inspect or adjudicate bounded interrupted-operation recovery evidence.",
    ),
    _command("blob-gc", "_blob_gc", "blob_gc_command", "Preview lease-safe blob garbage collection. Read-only."),
    _command(
        "blob-publications",
        "_blob_publications",
        "blob_publications_command",
        "Inspect publication receipts or explicitly abandon selected debt.",
    ),
    _command(
        "blob-reference-debt",
        "_blob_integrity",
        "blob_reference_debt_command",
        "Classify missing referenced blobs without mutating the archive.",
    ),
    _command(
        "blob-conservation",
        "_blob_conservation",
        "blob_conservation_command",
        "Verify both directions of blob/reference conservation without mutation.",
    ),
    _command(
        "blob-reference-recovery-plan",
        "_blob_integrity",
        "blob_reference_recovery_plan_command",
        "Plan recovery for raw-backed missing blobs without mutating archive state.",
    ),
    _command(
        "blob-reference-replace-from-source-preview",
        "_blob_integrity",
        "blob_reference_replace_from_source_preview_command",
        "Preview raw-backed blob-reference replacement without mutating the archive.",
    ),
    _command(
        "blob-reference-replace-from-source",
        "_blob_integrity",
        "blob_reference_replace_from_source_command",
        "Replace raw-backed missing blob refs with current source-derived bytes.",
    ),
    _command(
        "blob-reference-prune-orphans-preview",
        "_blob_integrity",
        "blob_reference_prune_orphans_preview_command",
        "Preview orphan blob_refs pruning without mutating the archive.",
    ),
    _command(
        "blob-reference-prune-orphans",
        "_blob_integrity",
        "blob_reference_prune_orphans_command",
        "Quarantine and prune missing blob_refs that no longer have raw rows.",
    ),
    _command(
        "embedding-orphan-reconcile",
        "_embeddings",
        "embedding_orphan_reconcile_command",
        "Inspect (default) or reconcile embeddings.db rows orphaned by an index rebuild.",
    ),
    _command(
        "gc-history", "_blob_gc", "gc_history_command", "Show recent blob-GC passes recorded in ``gc_generations``."
    ),
    _command(
        "gc-recover",
        "_blob_gc",
        "gc_recover_command",
        "Inspect or explicitly abandon a blocked pending blob-GC generation without unlinking blobs.",
    ),
    _command(
        "verify-archive",
        "_verify_archive",
        "verify_archive_command",
        "Prove the archive is coherent after a rebuild, restore, or promotion. Read-only.",
    ),
    _command(
        "wanted-sources",
        "_wanted_sources",
        "wanted_sources_command",
        "Freeze or authorize the private wanted-source denominator for the final rebuild.",
    ),
    _command(
        "blob-residue-compare",
        "_blob_residue_compare",
        "blob_residue_compare_command",
        "Compare present blob-residue candidates through the production parse route.",
    ),
    _command(
        "blob-disposition",
        "_blob_disposition",
        "blob_disposition_group",
        "Compile or consume the physical blob namespace disposition plan.",
        nested_group=True,
    ),
    _command(
        "embedding-preservation",
        "_embedding_preservation",
        "embedding_preservation_group",
        "Preserve, restore, prove, and discard embedding vectors across a rebuild.",
        nested_group=True,
    ),
)

MAINTENANCE_COMMAND_BY_NAME: Final[dict[str, MaintenanceCommandDeclaration]] = {
    declaration.cli_name: declaration for declaration in MAINTENANCE_COMMAND_DECLARATIONS
}
if len(MAINTENANCE_COMMAND_BY_NAME) != len(MAINTENANCE_COMMAND_DECLARATIONS):  # pragma: no cover - import guard
    raise RuntimeError("duplicate maintenance command declaration name")

MAINTENANCE_KERNEL_REGISTRY: Final[DeclarationRegistry] = DeclarationRegistry()
for _declaration in MAINTENANCE_COMMAND_DECLARATIONS:
    MAINTENANCE_KERNEL_REGISTRY.register(_declaration.kernel)

_DIAGNOSTICS = validate_registry(MAINTENANCE_KERNEL_REGISTRY)
if _DIAGNOSTICS:  # pragma: no cover - a structurally incomplete family must not import
    raise RuntimeError(
        "incomplete maintenance declaration registry: " + "; ".join(item.message for item in _DIAGNOSTICS)
    )


def declaration_for_command(cli_name: str) -> MaintenanceCommandDeclaration:
    """Resolve one maintenance command declaration or name the exact repair."""

    try:
        return MAINTENANCE_COMMAND_BY_NAME[cli_name]
    except KeyError as exc:
        raise KeyError(
            f"maintenance command {cli_name!r} has no declaration; add it to "
            f"MAINTENANCE_COMMAND_DECLARATIONS in polylogue/maintenance/declarations.py and run {REPAIR_COMMAND}"
        ) from exc


__all__ = [
    "MAINTENANCE_COMMAND_BY_NAME",
    "MAINTENANCE_COMMAND_DECLARATIONS",
    "MAINTENANCE_CONSUMER",
    "MAINTENANCE_KERNEL_REGISTRY",
    "REPAIR_COMMAND",
    "MaintenanceCommandDeclaration",
    "declaration_for_command",
]
