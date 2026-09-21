"""import command — schedule source files for import via the daemon.

Truthfulness contract (#1264 / #869 slice C):

* ``polylogue import PATH`` either really stages the file into the daemon
  inbox **and** confirms the daemon accepted scheduling, or it fails with
  an actionable message. There is no silent success path.

The three observable outcomes are:

1. **accepted + observable** — the file was copied into
   ``archive_root()/inbox`` and the running daemon returned an
   durable operation reference with status ``accepted``. The user sees the
   staged path, the operation id, and the next-step pointer
   (``polylogue ops status``) so they can watch the work converge.
2. **rejected (input)** — the supplied path does not exist, cannot be
   read, or cannot be staged into the inbox. Click rejects missing paths
   directly; staging errors raise a ``fail()`` with the offending path.
3. **rejected (daemon)** — the daemon is not running, refused the
   declared ``ingest`` operation, or returned an envelope this command
   cannot read as acceptance. The error message names the archive the
   submission was scoped to and points the user at ``polylogued run``:
   the resident daemon is the only writer, so there is no standalone
   mode to fall back to.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.cli.shared.helpers import DaemonRequiredError, fail
from polylogue.cli.shared.types import AppEnv
from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.demo import DemoVerifyResult

# Statuses that mean the daemon accepted scheduling and the work is now
# observable through the inbox / ``polylogue ops status`` surfaces.
_ACCEPTED_STATUSES = frozenset({"accepted", "pending", "scheduled", "queued"})
_DEMO_WAIT_POLL_INTERVAL_S = 0.25


def _default_daemon_url() -> str:
    from polylogue.config import load_polylogue_config

    return load_polylogue_config().daemon_url or "http://127.0.0.1:8766"


def _stage_for_daemon(path: Path, *, replace_existing: bool = False) -> Path:
    """Copy a local import target into the archive inbox for daemon pickup."""
    from polylogue.sources.parsers import antigravity, hermes_state
    from polylogue.sources.sqlite_snapshot import sqlite_staging_metadata_path, stage_sqlite_snapshot

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
        if hermes_state.looks_like_state_db_path(resolved) or antigravity.looks_like_trajectory_db_path(resolved):
            stage_sqlite_snapshot(resolved, dest)
            return dest
        sqlite_staging_metadata_path(dest).unlink(missing_ok=True)
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


def _wait_for_demo_archive_ready(*, timeout_s: float, require_overlays: bool = False) -> DemoVerifyResult:
    """Wait until the daemon-ingested demo archive reaches base ingest convergence.

    Deliberately skips the declared demo-construct minimums
    (``check_constructs=False``): several constructs (provider usage,
    synthetic embeddings, the canonical repo name) are populated by
    ``apply_demo_post_ingest_augmentation`` after this wait returns, not by
    ingest itself, so waiting on them here would never converge.
    """
    from polylogue.demo import verify_demo_archive

    deadline = time.monotonic() + timeout_s
    last_problems: tuple[str, ...] = ("verification did not run",)
    while time.monotonic() <= deadline:
        result = verify_demo_archive(
            archive_root(),
            require_overlays=require_overlays,
            check_source_path_leaks=False,
            check_constructs=False,
        )
        if result.ok:
            return result
        last_problems = result.problems
        time.sleep(_DEMO_WAIT_POLL_INTERVAL_S)

    problem_text = "; ".join(last_problems) if last_problems else "semantic checks did not pass"
    fail(
        "import",
        f"Timed out waiting {timeout_s:g}s for demo archive convergence: {problem_text}",
    )


def _verify_demo_now(*, require_overlays: bool = False) -> DemoVerifyResult:
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
    return result


def _request_demo_augmentation(env: AppEnv, *, with_overlays: bool) -> None:
    """Ask the daemon to apply demo-only writes under its writer lease."""
    from polylogue.cli.operation_kernel import (
        OperationIndeterminateError,
        OperationKernelError,
        OperationUnavailableError,
        configured_mutation_operation,
    )
    from polylogue.cli.shared.helpers import load_effective_config

    config = load_effective_config(env)
    try:
        result = configured_mutation_operation(
            config,
            "maintenance.demo.augment",
            {"with_overlays": with_overlays},
        )
    except OperationUnavailableError as exc:
        raise _daemon_required(config.archive_root, operation="maintenance.demo.augment") from exc
    except OperationIndeterminateError as exc:
        fail(
            "import",
            f"Demo augmentation reached the daemon and no receipt came back ({exc}); "
            "refusing to claim a verified archive. Inspect the daemon log before retrying.",
        )
    except OperationKernelError as exc:
        fail("import", f"Daemon rejected demo augmentation ({exc}); refusing to claim a verified archive.")

    if result.get("effect") == "indeterminate":
        fail("import", "Daemon reported an indeterminate demo augmentation; refusing to claim a verified archive.")


def _daemon_endpoint(config: object) -> str:
    """Return the archive-scoped daemon socket this CLI actually submits to."""
    from polylogue.daemon.socket_path import daemon_socket_path

    return str(daemon_socket_path(config.archive_root))  # type: ignore[attr-defined]


def _preflight_or_fail(staged: Path) -> None:
    """Refuse an inadmissible source before asking the daemon to schedule it.

    The daemon runs the same read-only admissibility check before it accepts
    an ingest.  Running it here too costs nothing and lets the operator see
    "this export shape is not parseable" without a scheduled operation that
    can only fail later.
    """
    from polylogue.operations.import_operations import import_source_admissibility

    preflight = import_source_admissibility(staged)
    if preflight.admissible:
        return
    fail(
        "import",
        f"{preflight.error_code}: {preflight.summary()}\n  The staged copy was left in place at {staged}.",
    )


def _submit_ingest(env: AppEnv, *, staged: Path, requested_source: Path) -> dict[str, object]:
    """Submit the declared ``ingest`` operation and report its acceptance.

    Returns an :class:`~polylogue.operations.import_contracts.ImportOperation`
    shaped mapping. Acceptance is decided exactly as the daemon's own ingest
    route decides it: a durable ``accepted_reference`` *and* an outcome that
    admits the work. Anything else is reported as a failure, never as success.
    """
    from polylogue.cli.operation_kernel import (
        OperationFailedError,
        OperationIndeterminateError,
        OperationKernelError,
        OperationUnavailableError,
        configured_accepted_operation,
    )
    from polylogue.cli.shared.helpers import load_effective_config

    config = load_effective_config(env)
    payload: dict[str, object] = {
        "path": str(staged),
        "source_path": str(requested_source.expanduser().resolve()),
        "idempotency_key": None,
    }
    try:
        envelope = configured_accepted_operation(config, "ingest", payload)
    except OperationUnavailableError as exc:
        raise _daemon_required(config.archive_root, operation="ingest") from exc
    except OperationIndeterminateError as exc:
        fail(
            "import",
            f"The ingest submission reached the daemon and no receipt came back ({exc}).\n"
            "  The daemon may already have admitted it: check `polylogued status` and the daemon "
            "log rather than re-running this command.\n"
            f"  Staged content is preserved at: {staged}",
        )
    except OperationFailedError as exc:
        fail(
            "import",
            f"Daemon refused the ingest operation ({exc.code}: {exc.detail}).\n"
            f"  The staged inbox entry was left in place at {staged}.",
        )
    except OperationKernelError as exc:
        fail(
            "import",
            f"Daemon returned an unusable ingest response ({exc}); refusing to claim success.\n"
            f"  The staged inbox entry was left in place at {staged}.",
        )

    reference = envelope.get("accepted_reference")
    outcome = envelope.get("outcome")
    accepted = reference is not None and outcome in {"accepted", "running", "completed", "indeterminate"}
    operation_id = ""
    if isinstance(reference, Mapping):
        operation_id = str(reference.get("request_id") or "")
    if not operation_id:
        operation_id = str(envelope.get("request_id") or "")
    return {
        "operation_id": operation_id,
        "kind": "import",
        "status": "accepted" if accepted else "failed",
        "path": str(staged),
        "error": None if accepted else f"daemon returned outcome {outcome!r} with no durable acceptance reference",
    }


def _daemon_required(archive: object, *, operation: str) -> DaemonRequiredError:
    """Build the typed refusal for a write no daemon is here to own.

    Typed rather than ``fail()``: ``fail`` raises ``SystemExit`` with a string,
    which ``machine_main`` turns into ``invalid_arguments`` -- so ``import
    --format json`` reported a daemon-absent archive as a malformed command
    line (polylogue-re6s3 AC4).
    """
    return DaemonRequiredError(
        f"No polylogued daemon is serving the archive at {archive}.\n"
        "  The resident daemon is the only writer: start it with 'polylogued run' and re-try.\n"
        "  There is no standalone import mode to fall back to.",
        operation=operation,
        archive_root=archive,
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
    # URL policy (mirrors ``polylogue status``, polylogue-2d8oq): the option and
    # ``POLYLOGUE_DAEMON_URL`` are accepted, but this command no longer speaks
    # the browser HTTP API — it submits the declared ``ingest`` operation over
    # the archive-scoped daemon socket, so the URL selects nothing here.
    del daemon_url

    if explain:
        from polylogue.sources.import_explain import explain_import_path
        from polylogue.surfaces.payloads import model_json_document

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
        requested_source = _materialize_demo_source()
        staged = _stage_for_daemon(requested_source, replace_existing=True)
    else:
        if path is None:
            fail("import", "Provide a source PATH or pass --demo.")
        requested_source = path
        staged = _stage_for_daemon(requested_source)

    from polylogue.cli.shared.helpers import load_effective_config

    _preflight_or_fail(staged)
    raw = _submit_ingest(env, staged=staged, requested_source=requested_source)

    from polylogue.operations.import_contracts import ImportOperation

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
        f"  Daemon:       {_daemon_endpoint(load_effective_config(env))}\n"
        f"  Next:         the daemon will process the staged file automatically.\n"
        f"                Check progress:    journalctl --user -u polylogued.service -f\n"
        f"                Check convergence: polylogued status\n"
        f"                Verify archive:    polylogue status --full"
    )

    if wait:
        env.ui.console.print(f"[bold]Waiting:[/bold] demo archive convergence (timeout {wait_timeout_s:g}s)")
        _wait_for_demo_archive_ready(timeout_s=wait_timeout_s)

        # Ingest alone (whichever path scheduled it) only produces the parsed
        # session/message tree. Demo-only enrichments -- provider usage,
        # insight materialization, the canonical repo name, synthetic
        # embeddings -- are layered on afterward so a daemon-ingested demo
        # archive matches ``polylogue demo seed``'s semantic contract exactly
        # (polylogue-z1c6). Idempotent: safe even if a prior --wait already
        # applied it against this archive root.
        _request_demo_augmentation(env, with_overlays=with_overlays)

        result = _verify_demo_now(require_overlays=with_overlays)
        env.ui.console.print(
            "[bold green]Demo archive verified:[/bold green] "
            f"sessions={result.session_count} messages={result.message_count} "
            f"overlays={'yes' if with_overlays else 'no'}"
        )


__all__ = ["import_command"]
