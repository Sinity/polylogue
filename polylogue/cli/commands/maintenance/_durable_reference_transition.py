"""``maintenance durable-reference-transition``: rebind durable refs across a rebuild.

The two phases are separate invocations because a rebuild happens between them.
``plan`` runs while both the retiring index generation and the candidate exist
and is read-only. ``apply`` runs offline, with the archive lease held, against
the exact plan it authorizes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.maintenance.durable_reference_transition import DurableReferenceTransition

_PLAN_SCHEMA = "polylogue.durable-reference-transition-plan.v1"


def _root(root: Path | None) -> Path:
    from polylogue.storage.archive_identity import ArchiveLocation

    return ArchiveLocation.resolve((root if root is not None else archive_root()).absolute()).configured_root


def _active_index(root: Path) -> Path:
    from polylogue.storage.archive_identity import ArchiveLocation

    return ArchiveLocation.resolve(root).active_index_path


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _build(
    root: Path,
    candidate_index: Path,
    predecessor_index: Path,
    *,
    producer: str,
    user_conn: sqlite3.Connection | None = None,
    audit_conn: sqlite3.Connection | None = None,
) -> DurableReferenceTransition:
    """Plan the transition from the archive's own files.

    ``user_conn``/``audit_conn`` are supplied by ``apply``, which already holds
    them read-write under the archive lease; the caller owns those.
    """
    from polylogue.maintenance.durable_reference_transition import (
        plan_durable_reference_transition,
        source_session_claims,
    )
    from polylogue.version import POLYLOGUE_VERSION

    with ExitStack() as stack:
        user = user_conn if user_conn is not None else stack.enter_context(closing(_readonly(root / "user.db")))
        audit = audit_conn if audit_conn is not None else stack.enter_context(closing(_readonly(root / "audit.db")))
        source = stack.enter_context(closing(_readonly(root / "source.db")))
        candidate = stack.enter_context(closing(_readonly(candidate_index)))
        predecessor = stack.enter_context(closing(_readonly(predecessor_index)))
        return plan_durable_reference_transition(
            user_conn=user,
            audit_conn=audit,
            candidate_index_conn=candidate,
            predecessor_index_conn=predecessor,
            source_claims=source_session_claims(source),
            package_version=POLYLOGUE_VERSION,
            producer=producer,
            candidate_index_path=str(candidate_index),
            predecessor_index_path=str(predecessor_index),
        )


@click.group("durable-reference-transition")
def durable_reference_transition_group() -> None:
    """Classify and rebind durable user/audit references across an index rebuild."""


@click.command("plan")
@click.option(
    "--candidate-index",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The rebuilt index generation the durable references must resolve against.",
)
@click.option(
    "--predecessor-index",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The retiring index generation (default: the archive's active index).",
)
@click.option("--archive-root", "root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--producer",
    default="maintenance:durable-reference-transition",
    show_default=True,
    help="Recorded author of the identity migration map.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the plan document that `apply` authorizes.",
)
@click.option("--output-format", "output_format", type=click.Choice(["plain", "json"]), default="plain")
def plan_command(
    candidate_index: Path,
    predecessor_index: Path | None,
    root: Path | None,
    producer: str,
    output_path: Path,
    output_format: str,
) -> None:
    """Classify every durable public reference against the candidate. Read-only."""
    from polylogue.maintenance.assertion_transition import ObjectRefReconciliationError

    resolved_root = _root(root)
    predecessor = predecessor_index if predecessor_index is not None else _active_index(resolved_root)
    try:
        transition = _build(resolved_root, candidate_index.absolute(), predecessor.absolute(), producer=producer)
    except (ObjectRefReconciliationError, sqlite3.Error) as exc:
        raise click.ClickException(str(exc)) from exc

    document = {
        "schema": _PLAN_SCHEMA,
        "archive_root": str(resolved_root),
        "producer": producer,
        "transition": transition.receipt(applied=False),
    }
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8")
    output_path.write_bytes(payload)
    authorization = hashlib.sha256(payload).hexdigest()

    if output_format == "json":
        click.echo(
            json.dumps(
                {**document, "plan_path": str(output_path), "authorize": authorization}, indent=2, sort_keys=True
            )
        )
        return
    click.echo(f"Plan:        {output_path}")
    click.echo(f"Authorize:   {authorization}")
    click.echo(f"Candidate:   {transition.candidate_index_path}")
    click.echo(f"Predecessor: {transition.predecessor_index_path}")
    click.echo(f"References:  {len(transition.plan.rows):,} classified from {len(transition.inventory):,} cell(s)")
    for disposition, count in sorted(transition.dispositions.items()):
        click.echo(f"  {disposition}: {count:,}")
    click.echo(f"Rewrites:    {len(transition.plan.forward):,}")
    if transition.is_blocked:
        click.echo("Blocked: the candidate is missing references the source tier still claims.", err=True)


@click.command("apply")
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Plan document produced by `plan`.",
)
@click.option("--authorize", "authorization", required=True, help="sha256 of the plan document being authorized.")
@click.option(
    "--backup-manifest",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Verified backup manifest covering user.db and audit.db.",
)
@click.option("--archive-root", "root", type=click.Path(path_type=Path), default=None)
@click.option("--output-format", "output_format", type=click.Choice(["plain", "json"]), default="plain")
def apply_command(
    plan_path: Path,
    authorization: str,
    backup_manifest: Path,
    root: Path | None,
    output_format: str,
) -> None:
    """Apply the authorized transition to user.db and audit.db, and record it."""
    from polylogue.maintenance.assertion_transition import ObjectRefReconciliationError
    from polylogue.maintenance.durable_reference_transition import (
        apply_durable_reference_transition,
        publish_transition_receipt,
    )
    from polylogue.operations.durable_change_train import ArchiveOwnershipError, acquire_durable_archive_ownership
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.migration_runner import MigrationError, validate_migration_backup_manifest

    payload = plan_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != authorization:
        raise click.ClickException("authorization does not match the plan document")
    document = json.loads(payload.decode("utf-8"))
    if document.get("schema") != _PLAN_SCHEMA:
        raise click.ClickException("plan document has an unexpected schema")

    resolved_root = _root(root)
    if document.get("archive_root") != str(resolved_root):
        raise click.ClickException(f"plan was produced for a different archive root: {document.get('archive_root')}")
    record = document["transition"]

    try:
        with acquire_durable_archive_ownership(resolved_root, owner_id=f"durable-reference-transition:{os.getpid()}"):
            user = sqlite3.connect(resolved_root / "user.db")
            audit = sqlite3.connect(resolved_root / "audit.db")
            try:
                validate_migration_backup_manifest(backup_manifest, ArchiveTier.USER, connection=user)
                validate_migration_backup_manifest(backup_manifest, ArchiveTier.AUDIT, connection=audit)
                # Recompute rather than deserialize: the plan is only an
                # authorization, and the durable tiers may have moved since.
                transition = _build(
                    resolved_root,
                    Path(str(record["candidate_index_path"])),
                    Path(str(record["predecessor_index_path"])),
                    producer=str(document["producer"]),
                    user_conn=user,
                    audit_conn=audit,
                )
                if transition.plan.digest() != record["plan_digest"]:
                    raise click.ClickException("archive state no longer matches the authorized plan")
                apply_durable_reference_transition(
                    user_conn=user,
                    audit_conn=audit,
                    transition=transition,
                    verified_backup=True,
                )
            finally:
                user.close()
                audit.close()
            receipt = publish_transition_receipt(resolved_root, transition, applied=True)
    except (ObjectRefReconciliationError, ArchiveOwnershipError, MigrationError, sqlite3.Error) as exc:
        raise click.ClickException(str(exc)) from exc

    result = {
        "applied": True,
        "receipt": receipt,
        "plan_digest": transition.plan.digest(),
        "rewrites": len(transition.plan.forward),
        "dispositions": transition.dispositions,
    }
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, sort_keys=True))
        return
    click.echo(f"Applied:   {len(transition.plan.forward):,} rewrite(s)")
    click.echo(f"Receipt:   .maintenance-state/durable-reference-transitions/{receipt}")
    for disposition, count in sorted(transition.dispositions.items()):
        click.echo(f"  {disposition}: {count:,}")


for _command in (plan_command, apply_command):
    durable_reference_transition_group.add_command(_command)

del _command

__all__ = ["durable_reference_transition_group"]
