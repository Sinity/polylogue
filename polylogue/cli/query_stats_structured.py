"""Structured grouped-stats serialization helpers."""

from __future__ import annotations

import json

import click


def emit_structured_stats(
    *,
    output_format: str,
    dimension: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    multi_membership: bool = False,
) -> bool:
    if output_format == "json":
        click.echo(json.dumps(
            {
                "dimension": dimension,
                "multi_membership": multi_membership,
                "rows": rows,
                "summary": summary,
            },
            indent=2,
        ))
        return True

    if output_format == "yaml":
        import yaml

        click.echo(yaml.dump(
            {
                "dimension": dimension,
                "multi_membership": multi_membership,
                "rows": rows,
                "summary": summary,
            },
            default_flow_style=False,
            allow_unicode=True,
        ))
        return True

    if output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        fieldnames = list(summary.keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)
        click.echo(buf.getvalue().rstrip("\r\n"))
        return True

    return False


__all__ = ["emit_structured_stats"]
