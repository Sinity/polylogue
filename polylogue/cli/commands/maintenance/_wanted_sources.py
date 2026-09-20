"""``maintenance wanted-sources``: freeze and authorize the rebuild denominator.

The final rebuild's conservation proof needs a denominator that was fixed
*before* the build, not re-derived from whatever the source roots happen to
hold afterwards.  :mod:`polylogue.maintenance.source_manifest_continuity` owns
that receipt; this command is the operator route to it, and the only route
that produces one outside tests (polylogue-co2iz).

Two directions, one receipt:

``--freeze``
    Enumerate the configured declarations once under the campaign policy and
    atomically publish the private receipt into the archive's own maintenance
    state.

default
    Authorize a rebuild from the frozen receipt through
    ``require_rebuild_preflight``: integrity, policy identity, declaration
    digest, root availability and completeness are all recomputed, and a
    missing, tampered, incomplete or policy-revised receipt exits non-zero.
    There is no fallback that fresh-walks ambient source roots.

The receipt itself holds real roots and member coordinates and stays in
private operator state.  This command's output is deliberately digest- and
count-shaped: source ids, roles, root states and totals, never a path.
"""

from __future__ import annotations

import json

import click

from polylogue.paths import archive_root


def _root_state_rows(receipt: object) -> list[dict[str, object]]:
    states = getattr(receipt, "root_states", {})
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for member in getattr(receipt, "members", ()):
        counts[member.source_id] = counts.get(member.source_id, 0) + 1
        sizes[member.source_id] = sizes.get(member.source_id, 0) + member.size
    rows: list[dict[str, object]] = []
    for declaration in getattr(receipt, "declarations", ()):
        source_id = declaration.source_id
        state = states.get(source_id)
        rows.append(
            {
                "source_id": source_id,
                "role": declaration.role.value,
                "state": state.value if state is not None else "unknown",
                "item_count": counts.get(source_id, 0),
                "byte_count": sizes.get(source_id, 0),
            }
        )
    return rows


@click.command("wanted-sources")
@click.option(
    "--freeze",
    is_flag=True,
    help="Enumerate the configured declarations and publish a new private wanted-source receipt.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def wanted_sources_command(freeze: bool, output_format: str) -> None:
    """Freeze or authorize the frozen wanted-source denominator for a rebuild.

    Without ``--freeze`` this is read-only: it loads the private receipt,
    recomputes its integrity against the configured declarations, and prints
    the preflight authorization -- receipt digest, policy identity,
    declaration digest, frontier digest and the denominator totals.  A
    missing, tampered, incomplete, or policy-mismatched receipt is a refusal
    with exit 1; the rebuild is expected to stop there rather than walk the
    source roots again.
    """
    from polylogue.config import configured_source_declarations, resolve_runtime_config
    from polylogue.maintenance.source_manifest_continuity import (
        WantedSourceReceiptError,
        campaign_default_wanted_source_policy,
        require_rebuild_preflight,
        write_wanted_source_receipt,
    )

    root = archive_root()
    declarations = configured_source_declarations(resolve_runtime_config())
    policy = campaign_default_wanted_source_policy()

    payload: dict[str, object]
    try:
        if freeze:
            receipt = write_wanted_source_receipt(root, declarations, policy=policy)
            payload = {
                "outcome": "ok",
                "action": "freeze",
                "receipt_sha256": receipt.receipt_sha256,
                "policy_identity": receipt.policy_identity,
                "declaration_sha256": receipt.declaration_sha256,
                "frontier_sha256": receipt.frontier_sha256,
                "item_count": receipt.item_count,
                "byte_count": receipt.byte_count,
                "complete": receipt.complete,
                "blocker_count": len(receipt.blockers),
                "excluded_source_ids": list(receipt.excluded_source_ids),
                "roots": _root_state_rows(receipt),
            }
        else:
            preflight = require_rebuild_preflight(root, policy=policy, declarations=declarations)
            payload = {"action": "preflight", **preflight.as_dict()}
    except WantedSourceReceiptError as exc:
        refusal = {
            "outcome": "error",
            "action": "freeze" if freeze else "preflight",
            "reason": str(exc),
        }
        if output_format == "json":
            click.echo(json.dumps(refusal, indent=2, sort_keys=True))
        else:
            click.echo(f"Wanted sources: REFUSED ({exc})")
        raise click.exceptions.Exit(1) from exc

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Wanted sources: {'FROZEN' if freeze else 'AUTHORIZED'}")
    click.echo(f"Receipt: {payload['receipt_sha256']}")
    click.echo(f"Policy:  {payload['policy_identity']} ({policy.name} rev {policy.revision})")
    click.echo(f"Declarations: {payload['declaration_sha256']}")
    click.echo(f"Frontier:     {payload['frontier_sha256']}")
    click.echo(f"Denominator:  {int(payload['item_count']):,} items; {int(payload['byte_count']):,} bytes")
    for row in payload.get("roots", []):  # type: ignore[union-attr]
        click.echo(
            f"  {row['source_id']} [{row['role']}] {row['state']}: "
            f"{int(row['item_count']):,} items; {int(row['byte_count']):,} bytes"
        )


__all__ = ["wanted_sources_command"]
