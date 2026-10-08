"""``maintenance assertion-export``: export the durable assertion substrate from user.db."""

from __future__ import annotations

import json
import shutil
import tempfile
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

    # Stage the complete walk before exposing bytes. A refused continuation,
    # cancellation or missing original User authority leaves output untouched.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as staged:
        offset = 0
        epoch = None
        total = None
        count = 0
        while True:
            try:
                result = configured_read_operation(
                    load_polylogue_config(),
                    "user.assertions.export",
                    {
                        "kinds": list(kinds) or None,
                        "statuses": list(statuses) or None,
                        "limit": limit,
                        "offset": offset,
                        "page_size": 256,
                        "selection_epoch": epoch,
                    },
                ).value
            except OperationKernelError as exc:
                from polylogue.cli.render.outcome import exit_for_read_failure

                exit_for_read_failure(exc)
            if not isinstance(result, dict) or not isinstance(result.get("items"), list):
                raise click.ClickException("user.assertions.export returned an invalid page")
            items = result["items"]
            page_total = result.get("total")
            page_epoch = result.get("snapshot_epoch")
            next_offset = result.get("next_offset")
            if (
                not isinstance(page_total, int)
                or page_total < 0
                or not isinstance(page_epoch, str)
                or not page_epoch
                or result.get("offset") != offset
                or len(items) > 256
                or (total is not None and (total != page_total or epoch != page_epoch))
            ):
                raise click.ClickException("user.assertions.export returned an invalid page")
            if total is None:
                total, epoch = page_total, page_epoch
                if output_format == "json":
                    header = json.dumps(
                        {
                            "ok": True,
                            "mode": "assertion_export",
                            "archive_root": str(root),
                            "user_db_path": str(user_db_path),
                            "count": total,
                        },
                        sort_keys=True,
                    )
                    staged.write(header[:-1] + ', "assertions": [')
            for row in items:
                if not isinstance(row, dict):
                    raise click.ClickException("user.assertions.export returned an invalid item")
                if output_format == "json":
                    if count:
                        staged.write(",")
                    staged.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                else:
                    staged.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            if count > total:
                raise click.ClickException("user.assertions.export returned an invalid count")
            if next_offset is None:
                if count != total:
                    raise click.ClickException("user.assertions.export returned an incomplete page walk")
                break
            if (
                not isinstance(next_offset, int)
                or next_offset != count
                or next_offset <= offset
                or next_offset >= total
            ):
                raise click.ClickException("user.assertions.export returned an invalid continuation")
            offset = next_offset
        if output_format == "json":
            staged.write("]}\n")
        staged.seek(0)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as output:
                shutil.copyfileobj(staged, output)
            click.echo(f"Exported {count} assertions to {out_path}")
        else:
            while chunk := staged.read(64 * 1024):
                click.echo(chunk, nl=False)
