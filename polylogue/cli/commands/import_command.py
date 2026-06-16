"""import command — schedule source files for import via the daemon.

Truthfulness contract (#1264 / #869 slice C):

* ``polylogue import PATH`` either really stages the file into the daemon
  inbox **and** confirms the daemon accepted scheduling, or it fails with
  an actionable message. There is no silent success path.

The three observable outcomes are:

1. **accepted + observable** — the file was copied into
   ``archive_root()/inbox`` and the running daemon returned an
   ``ImportAck`` with status ``pending``/``accepted``. The user sees the
   staged path, the operation id, and the next-step pointer
   (``polylogue status``) so they can watch the work converge.
2. **rejected (input)** — the supplied path does not exist, cannot be
   read, or cannot be staged into the inbox. Click rejects missing paths
   directly; staging errors raise a ``fail()`` with the offending path.
3. **rejected (daemon)** — the daemon is not reachable, returned an HTTP
   error, or returned a response with a failed status. The error message
   names the daemon URL and points the user at ``polylogued`` so they
   know which process to start or inspect.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.operations.import_contracts import ImportOperation
from polylogue.paths import archive_root

_DEFAULT_DAEMON_URL = "http://127.0.0.1:8766"

# Statuses that mean the daemon accepted scheduling and the work is now
# observable through the inbox / ``polylogue status`` surfaces.
_ACCEPTED_STATUSES = frozenset({"accepted", "pending", "scheduled", "queued"})


def _daemon_url(env: AppEnv) -> str:
    return getattr(env, "daemon_url", None) or _DEFAULT_DAEMON_URL


def _stage_for_daemon(path: Path, *, replace_existing: bool = False) -> Path:
    """Copy a local import target into the archive inbox for daemon pickup."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        fail("import", f"Path does not exist: {resolved}")

    inbox = archive_root() / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    dest = inbox / resolved.name

    if dest.exists() and resolved == dest.resolve():
        return dest

    try:
        if replace_existing and dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if resolved.is_dir():
            shutil.copytree(resolved, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(resolved, dest)
    except OSError as exc:
        fail("import", f"Could not stage {resolved} in daemon inbox: {exc}")

    return dest


def _materialize_demo_source() -> Path:
    """Write the approved deterministic demo fixture world to a local source dir."""
    from polylogue.scenarios import build_demo_corpus_specs
    from polylogue.schemas.synthetic import SyntheticCorpus

    source_root = archive_root() / "demo-fixture-world-source"
    if source_root.exists():
        shutil.rmtree(source_root)
    SyntheticCorpus.write_specs_artifacts(build_demo_corpus_specs(), source_root, prefix="demo", index_width=2)
    return source_root


def _daemon_unreachable_message(daemon_url: str, reason: str) -> str:
    """Build an actionable error when the daemon is unreachable."""
    return (
        f"Could not reach daemon at {daemon_url} ({reason}).\n"
        "  Is polylogued running? Start it with 'polylogued run' and re-try, "
        "or pass --daemon-url to point at the correct API endpoint."
    )


def _daemon_http_error_message(exc: HTTPError, *, daemon_url: str, staged: Path) -> str:
    detail = ""
    try:
        raw_body = exc.read()
    except OSError:
        raw_body = b""
    if raw_body:
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            body_detail = payload.get("detail")
            fragments: list[str] = []
            if isinstance(error, str) and error:
                fragments.append(error)
            if isinstance(body_detail, str) and body_detail:
                fragments.append(body_detail)
            detail = "\n  Daemon detail: " + " — ".join(fragments) if fragments else ""
    return (
        f"Daemon at {daemon_url} rejected /api/ingest with HTTP {exc.code}: {exc.reason}.{detail}\n"
        "  Check the daemon log for the cause; the staged inbox file was "
        f"left in place at {staged}."
    )


@click.command("import")
@click.argument("path", required=False, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--demo",
    is_flag=True,
    help="Generate and schedule the approved deterministic demo fixture world.",
)
@click.option(
    "--daemon-url",
    default=_DEFAULT_DAEMON_URL,
    show_default=True,
    help="Daemon API URL.",
)
@click.pass_obj
def import_command(
    env: AppEnv,
    path: Path | None,
    demo: bool,
    daemon_url: str,
) -> None:
    """Schedule a file or directory for import by the running daemon.

    Stages PATH into the archive inbox and asks the running polylogued
    daemon to schedule it for processing. The command is truthful: it
    either confirms the daemon accepted scheduling (with a pointer to
    'polylogue status' for observable progress) or fails with an
    actionable error. It never reports success without observable
    processing.
    """
    if demo:
        if path is not None:
            fail("import", "Use either PATH or --demo, not both.")
        source_path = _materialize_demo_source()
        staged = _stage_for_daemon(source_path, replace_existing=True)
    else:
        if path is None:
            fail("import", "Provide a source PATH or pass --demo.")
        staged = _stage_for_daemon(path)

    body = json.dumps({"path": str(staged)}).encode("utf-8")
    req = Request(
        f"{daemon_url}/api/ingest",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=5) as resp:
            raw = json.loads(resp.read())
    except HTTPError as exc:
        # Daemon responded but rejected the request. Surface the status
        # code so the operator knows it's a contract problem, not a
        # transport problem.
        fail("import", _daemon_http_error_message(exc, daemon_url=daemon_url, staged=staged))
    except URLError as exc:
        fail("import", _daemon_unreachable_message(daemon_url, str(exc.reason)))
    except OSError as exc:
        fail("import", _daemon_unreachable_message(daemon_url, str(exc)))

    operation = ImportOperation.from_dict(raw)

    if operation.status in ("failed", "error"):
        fail("import", operation.error or operation.message or "Unknown error")

    if operation.status not in _ACCEPTED_STATUSES:
        # The daemon returned something we don't recognize as accepted
        # *or* failed. Refuse to fabricate success.
        fail(
            "import",
            f"Daemon returned unexpected status {operation.status!r}; refusing to claim success.",
        )

    env.ui.console.print(
        f"[bold green]Scheduled:[/bold green] {operation.path or staged}\n"
        f"  Staged file:  {staged}\n"
        f"  Operation:    {operation.operation_id}\n"
        f"  Daemon:       {daemon_url}\n"
        f"  Next:         the daemon will process the staged file automatically.\n"
        f"                Check progress:  journalctl --user -u polylogued.service -f\n"
        f"                Verify ingested: polylogue stats"
    )


__all__ = ["import_command"]
