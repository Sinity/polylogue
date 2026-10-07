"""``maintenance assertion-export``: export the durable assertion substrate from user.db."""

from __future__ import annotations

import json
from pathlib import Path

import click

from polylogue.core.enums import AssertionKind
from polylogue.paths import archive_root

# AssertionKind is imported directly from polylogue.core.enums (not
# polylogue.storage.sqlite.archive_tiers.user_write, which re-exports the
# same class) so click.Choice(...) below -- evaluated at decoration time,
# unlike the rest of this module's storage imports -- doesn't force the
# archive_tiers package's own eager DDL-import chain onto the `--help` path.
# See polylogue-sod7.


@click.command("assertion-export")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="jsonl",
    show_default=True,
    help="Export format for assertion rows.",
)
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None, help="Write export to this path.")
@click.option(
    "--kind",
    "kinds",
    multiple=True,
    type=click.Choice([kind.value for kind in AssertionKind]),
    help="Restrict export to one assertion kind; repeatable.",
)
@click.option("--status", "statuses", multiple=True, help="Restrict export to one assertion status; repeatable.")
@click.option("--limit", "-l", type=click.IntRange(min=0), default=None, help="Maximum assertion rows to export.")
def assertion_export_command(
    output_format: str,
    out_path: Path | None,
    kinds: tuple[str, ...],
    statuses: tuple[str, ...],
    limit: int | None,
) -> None:
    """Export the durable assertion substrate from user.db."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = archive_root()
    user_db_path = root / ARCHIVE_TIER_SPECS[ArchiveTier.USER].filename
    from polylogue.cli.operation_kernel import OperationKernelError, configured_read_operation
    from polylogue.config import load_polylogue_config

    try:
        result = configured_read_operation(
            load_polylogue_config(),
            "user.assertions.export",
            {"kinds": list(kinds) or None, "statuses": list(statuses) or None, "limit": limit},
        ).value
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        raise click.ClickException("user.assertions.export returned an invalid result")
    payload_rows = result["items"]

    if output_format == "json":
        content = (
            json.dumps(
                {
                    "ok": True,
                    "mode": "assertion_export",
                    "archive_root": str(root),
                    "user_db_path": str(user_db_path),
                    "count": len(payload_rows),
                    "assertions": payload_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload_rows)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        click.echo(f"Exported {len(payload_rows)} assertions to {out_path}")
        return

    click.echo(content, nl=False)
