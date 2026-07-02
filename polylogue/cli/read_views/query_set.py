"""Query-set read-view adapter."""

from __future__ import annotations

import io
from pathlib import Path

import click

from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def run_query_set_read_view(
    env: AppEnv,
    request: RootModeRequest,
    *,
    output_format: str | None,
    fields: str | None,
    destination: str,
    out_path: str | None,
) -> None:
    """Render all matched sessions through the query-set read path."""

    from polylogue.cli.query_set_read import run_query_set_read

    fmt = output_format or "ndjson"
    bulk_fmt = "jsonl" if fmt == "ndjson" else fmt

    if destination == "file":
        if not out_path:
            raise click.UsageError("--to file requires --out <path>.")

        buf = io.StringIO()

        def _captured_echo_read_set(message: object = None, **_kwargs: object) -> None:
            buf.write(str(message or "") + "\n")

        _orig_echo = click.echo
        click.echo = _captured_echo_read_set  # type: ignore[assignment]
        try:
            run_query_set_read(env, request, output_format=bulk_fmt, fields=fields)
        finally:
            click.echo = _orig_echo
        Path(out_path).write_text(buf.getvalue(), encoding="utf-8")
        env.ui.console.print(f"Wrote to {out_path}")
        return

    run_query_set_read(env, request, output_format=bulk_fmt, fields=fields)


__all__ = ["run_query_set_read_view"]
