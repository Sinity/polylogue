"""``maintenance durable-reference-transition``: rebind durable refs across a rebuild.

The two phases are separate invocations because a rebuild happens between them.
``plan`` runs while both the retiring index generation and the candidate exist
and is read-only. ``apply`` runs offline, with the archive lease held, against
the exact plan it authorizes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.maintenance.durable_reference_transition import DurableReferenceTransition

_PLAN_SCHEMA = "polylogue.durable-reference-transition-plan.v1"


def _location(root: Path | None) -> tuple[Path, Path]:
    from polylogue.operations.durable_reference_transition import resolve_archive_location

    return resolve_archive_location(root if root is not None else archive_root())


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
    from polylogue.operations.durable_reference_transition import DurableReferenceTransitionError, plan_transition

    resolved_root, active_index = _location(root)
    predecessor = (predecessor_index if predecessor_index is not None else active_index).absolute()
    try:
        transition = plan_transition(
            archive_root=resolved_root,
            candidate_index=candidate_index.absolute(),
            predecessor_index=predecessor,
            producer=producer,
        )
    except DurableReferenceTransitionError as exc:
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
    _echo_summary(transition)
    click.echo(f"Plan:        {output_path}")
    click.echo(f"Authorize:   {authorization}")
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
    from polylogue.operations.durable_reference_transition import DurableReferenceTransitionError, apply_transition

    payload = plan_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != authorization:
        raise click.ClickException("authorization does not match the plan document")
    document = json.loads(payload.decode("utf-8"))
    if document.get("schema") != _PLAN_SCHEMA:
        raise click.ClickException("plan document has an unexpected schema")

    resolved_root, _active_index = _location(root)
    if document.get("archive_root") != str(resolved_root):
        raise click.ClickException(f"plan was produced for a different archive root: {document.get('archive_root')}")
    record = document["transition"]

    try:
        transition, receipt = apply_transition(
            archive_root=resolved_root,
            candidate_index=Path(str(record["candidate_index_path"])),
            predecessor_index=Path(str(record["predecessor_index_path"])),
            producer=str(document["producer"]),
            backup_manifest=backup_manifest,
            expected_plan_digest=str(record["plan_digest"]),
        )
    except DurableReferenceTransitionError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "applied": True,
                    "receipt": receipt,
                    "plan_digest": transition.plan.digest(),
                    "rewrites": len(transition.plan.forward),
                    "dispositions": transition.dispositions,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    _echo_summary(transition)
    click.echo(f"Applied:     {len(transition.plan.forward):,} rewrite(s)")
    click.echo(f"Receipt:     .maintenance-state/durable-reference-transitions/{receipt}")


def _echo_summary(transition: DurableReferenceTransition) -> None:
    click.echo(f"Candidate:   {transition.candidate_index_path}")
    click.echo(f"Predecessor: {transition.predecessor_index_path}")
    click.echo(f"References:  {len(transition.plan.rows):,} classified from {len(transition.inventory):,} cell(s)")
    for disposition, count in sorted(transition.dispositions.items()):
        click.echo(f"  {disposition}: {count:,}")
    click.echo(f"Rewrites:    {len(transition.plan.forward):,}")


for _command in (plan_command, apply_command):
    durable_reference_transition_group.add_command(_command)

del _command

__all__ = ["durable_reference_transition_group"]
