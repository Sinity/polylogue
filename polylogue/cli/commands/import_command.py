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
   (``polylogue ops status``) so they can watch the work converge.
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
import os
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.operations.import_contracts import ImportOperation
from polylogue.paths import archive_root
from polylogue.sources.import_explain import explain_import_path
from polylogue.sources.parsers import hermes_state
from polylogue.sources.sqlite_snapshot import snapshot_sqlite_database
from polylogue.surfaces.payloads import model_json_document

_DEFAULT_DAEMON_URL = "http://127.0.0.1:8766"

# Statuses that mean the daemon accepted scheduling and the work is now
# observable through the inbox / ``polylogue ops status`` surfaces.
_ACCEPTED_STATUSES = frozenset({"accepted", "pending", "scheduled", "queued"})
_DEMO_WAIT_POLL_INTERVAL_S = 0.25


def _default_daemon_url() -> str:
    return os.environ.get("POLYLOGUE_DAEMON_URL") or _DEFAULT_DAEMON_URL


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
        if hermes_state.looks_like_state_db_path(resolved):
            snapshot_sqlite_database(resolved, dest)
            return dest
        if resolved.is_dir():
            shutil.copytree(resolved, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(resolved, dest)
    except OSError as exc:
        fail("import", f"Could not stage {resolved} in daemon inbox: {exc}")

    return dest


def _materialize_demo_source() -> Path:
    """Write the approved deterministic demo fixture world to a local source dir."""
    from polylogue.demo import materialize_demo_source

    return materialize_demo_source(archive_root(), force=True)


def _wait_for_demo_archive_ready(*, timeout_s: float, require_overlays: bool = False) -> None:
    """Wait until the daemon-ingested demo archive passes semantic verification."""
    from polylogue.demo import verify_demo_archive

    deadline = time.monotonic() + timeout_s
    last_problems: tuple[str, ...] = ("verification did not run",)
    while time.monotonic() <= deadline:
        result = verify_demo_archive(
            archive_root(),
            require_overlays=require_overlays,
            check_source_path_leaks=False,
        )
        if result.ok:
            return
        last_problems = result.problems
        time.sleep(_DEMO_WAIT_POLL_INTERVAL_S)

    problem_text = "; ".join(last_problems) if last_problems else "semantic checks did not pass"
    fail(
        "import",
        f"Timed out waiting {timeout_s:g}s for demo archive convergence: {problem_text}",
    )


def _verify_demo_now(*, require_overlays: bool = False) -> None:
    """Run the demo verifier once and surface semantic failures through Click."""
    from polylogue.demo import verify_demo_archive

    result = verify_demo_archive(
        archive_root(),
        require_overlays=require_overlays,
        check_source_path_leaks=False,
    )
    if not result.ok:
        problem_text = "; ".join(result.problems) if result.problems else "semantic checks did not pass"
        fail("import", f"Demo archive verification failed: {problem_text}")


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
    default=_default_daemon_url,
    show_default=True,
    help="Daemon API URL (env: POLYLOGUE_DAEMON_URL).",
)
@click.option(
    "--explain",
    is_flag=True,
    help="Explain detector/parser decisions without scheduling daemon import.",
)
@click.option(
    "--wait",
    is_flag=True,
    help="With --demo, wait for daemon convergence and verify the demo archive.",
)
@click.option(
    "--timeout",
    "wait_timeout_s",
    type=click.FloatRange(min=0.001),
    default=30.0,
    show_default=True,
    help="Seconds to wait for --demo --wait convergence.",
)
@click.option(
    "--with-overlays",
    is_flag=True,
    help="With --demo --wait, seed deterministic user overlays after daemon ingest.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "ndjson"]),
    default="json",
    show_default=True,
    help="Machine-readable output format for --explain.",
)
@click.pass_obj
def import_command(
    env: AppEnv,
    path: Path | None,
    demo: bool,
    daemon_url: str,
    explain: bool,
    wait: bool,
    wait_timeout_s: float,
    with_overlays: bool,
    output_format: str,
) -> None:
    """Schedule a file or directory for import by the running daemon.

    Stages PATH into the archive inbox and asks the running polylogued
    daemon to schedule it for processing. The command is truthful: it
    either confirms the daemon accepted scheduling (with a pointer to
    'polylogue ops status' for observable progress) or fails with an
    actionable error. It never reports success without observable
    processing.
    """
    if explain:
        if demo:
            fail("import", "--explain requires a source PATH; --demo materializes and schedules generated fixtures.")
        if path is None:
            fail("import", "Provide a source PATH to explain.")
        payload = explain_import_path(path)
        if output_format == "ndjson":
            for entry in payload.entries:
                click.echo(json.dumps(model_json_document(entry, exclude_none=True), sort_keys=True))
            return
        click.echo(payload.to_json(exclude_none=True))
        return

    if wait and not demo:
        fail("import", "--wait is currently supported only with --demo.")
    if with_overlays and not demo:
        fail("import", "--with-overlays is currently supported only with --demo.")
    if with_overlays and not wait:
        fail("import", "--with-overlays requires --demo --wait so overlays attach to ingested sessions.")

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
        f"                Check progress:    journalctl --user -u polylogued.service -f\n"
        f"                Check convergence: polylogued status\n"
        f"                Verify archive:    polylogue status --full"
    )

    if wait:
        env.ui.console.print(f"[bold]Waiting:[/bold] demo archive convergence (timeout {wait_timeout_s:g}s)")
        _wait_for_demo_archive_ready(timeout_s=wait_timeout_s)
        if with_overlays:
            from polylogue.scenarios import seed_demo_user_overlays

            seed_demo_user_overlays(archive_root())
            _verify_demo_now(require_overlays=True)
        env.ui.console.print(
            "[bold green]Demo archive verified:[/bold green] "
            f"sessions=3 messages=19 overlays={'yes' if with_overlays else 'no'}"
        )


__all__ = ["import_command"]
