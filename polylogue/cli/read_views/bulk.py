"""Bulk export read-view adapter."""

from __future__ import annotations

import io
from pathlib import Path

import click

from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def run_bulk_export_view(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str | None,
    fields: str | None,
    destination: str,
    out_path: str | None,
) -> None:
    """Bulk export all matched sessions."""

    from polylogue.cli.bulk_export import run_bulk_export

    fmt = output_format or "ndjson"
    bulk_fmt = "jsonl" if fmt == "ndjson" else fmt

    if destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")

        buf = io.StringIO()

        def _captured_echo_bulk(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_bulk  # type: ignore[assignment]
        try:
            run_bulk_export(env, request, output_format=bulk_fmt, fields=fields)
        finally:
            click.echo = _orig_echo
        Path(out_path).write_text(buf.getvalue(), encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
        return

    run_bulk_export(env, request, output_format=bulk_fmt, fields=fields)


__all__ = ["run_bulk_export_view"]
