"""Ingest command — schedule source files for ingestion via the daemon."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.operations.import_contracts import ImportOperation

_DEFAULT_DAEMON_URL = "http://127.0.0.1:8766"


def _daemon_url(env: AppEnv) -> str:
    return getattr(env, "daemon_url", None) or _DEFAULT_DAEMON_URL


@click.command("ingest")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--daemon-url",
    default=_DEFAULT_DAEMON_URL,
    show_default=True,
    help="Daemon API URL.",
)
@click.pass_obj
def ingest_command(
    env: AppEnv,
    path: Path,
    daemon_url: str,
) -> None:
    """Schedule a file or directory for ingestion by the daemon.

    Contacts the running polylogued daemon and requests ingestion of
    PATH. The daemon copies the file into its inbox and begins processing
    asynchronously. Use 'polylogue status' to monitor progress.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        fail("ingest", f"Path does not exist: {resolved}")

    body = json.dumps({"path": str(resolved)}).encode("utf-8")
    req = Request(
        f"{daemon_url}/api/ingest",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=5) as resp:
            raw = json.loads(resp.read())
    except OSError as exc:
        fail(
            "ingest",
            f"Could not reach daemon at {daemon_url}. Is polylogued running? ({exc})",
        )

    operation = ImportOperation.from_dict(raw)
    if operation.status not in ("failed", "error"):
        env.ui.console.print(
            f"[bold green]Scheduled:[/bold green] {operation.path}\n"
            f"  Operation: {operation.operation_id}\n"
            f"  {operation.message}"
        )
    else:
        fail("ingest", operation.error or "Unknown error")


__all__ = ["ingest_command"]
