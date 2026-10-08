"""import command — schedule source files for import via the daemon.

Truthfulness contract (#1264 / #869 slice C):

* ``polylogue import PATH`` either really stages the file into the archive's
  import staging directory **and** confirms the daemon accepted scheduling,
  or it fails with an actionable message. There is no silent success path.
  The staging directory is not a watched source: the accepted ``ingest``
  operation is the only route that acquires a staged import.

The three observable outcomes are:

1. **accepted + observable** — the file was copied into
   ``archive_root()/import-staging`` and the running daemon returned an
   durable operation reference with status ``accepted``. The user sees the
   staged path, the operation id, and the next-step pointer
   (``polylogue ops status``) so they can watch the work converge.
2. **rejected (input)** — the supplied path does not exist, cannot be
   read, or cannot be staged. Click rejects missing paths
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
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import click

from polylogue.cli.shared.helpers import DaemonRequiredError, fail
from polylogue.cli.shared.types import AppEnv
from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.demo import DemoVerifyResult
    from polylogue.operations.import_operations import ImportSourceAdmission

# Statuses that mean the daemon accepted scheduling and the work is now
# observable through ``polylogue ops status``.
_ACCEPTED_STATUSES = frozenset({"accepted", "pending", "scheduled", "queued"})


def _stage_for_daemon(path: Path) -> Path:
    """Capture one private import slot with its immutable outside receipt."""
    from polylogue.operations.import_staging import stage_import_input

    try:
        return stage_import_input(path, archive_root(), check_stop=lambda: None)
    except (OSError, ValueError) as exc:
        fail("import", f"Could not stage {path} for import: {exc}")


def _materialize_demo_source() -> Path:
    """Write the approved deterministic demo fixture world to a local source dir."""
    from polylogue.demo import materialize_demo_source

    return materialize_demo_source(archive_root(), force=True)


def _wait_for_ingest(env: AppEnv, accepted: dict[str, object], *, timeout_s: float) -> None:
    """Wait for the accepted ingest's terminal receipt from the daemon.

    The ingest operation reaches ``completed`` only after its material is
    parsed, materialized and its session profiles converged, so its receipt is
    the convergence signal; no archive file is re-read on a timer. A
    ``degraded`` ingest committed its rows but stopped converging on a
    retryable target, so it is refused here rather than reported done. For
    ``--demo`` the demo-only constructs (provider usage, synthetic embeddings,
    the canonical repo name) are layered on by
    ``apply_demo_post_ingest_augmentation`` after this returns, which is why the
    demo verification runs after that.
    """
    from polylogue.cli.operation_kernel import OperationKernelError, configured_follow_operation
    from polylogue.cli.shared.helpers import load_effective_config

    try:
        receipt = configured_follow_operation(load_effective_config(env), "ingest", accepted, wait_s=timeout_s)
    except OperationKernelError as exc:
        fail("import", f"Lost the ingest before its receipt ({exc}); refusing to claim it finished.")
    outcome = receipt.get("outcome")
    if outcome == "completed":
        return
    if outcome == "indeterminate":
        fail(
            "import",
            f"Timed out waiting {timeout_s:g}s for the ingest to converge; it is still the daemon's "
            "work. Check `polylogued status`.",
        )
    if outcome == "degraded":
        fail(
            "import",
            "The ingest committed its sessions but did not finish converging their profiles and "
            "insights; the daemon's convergence continues it. Check `polylogued status`.",
        )
    error = receipt.get("error")
    detail = error.get("detail") if isinstance(error, dict) else None
    fail(
        "import",
        f"Ingest ended {outcome!r}{f': {detail}' if detail else ''}; refusing to claim it finished.",
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


def _preflight_or_fail(staged: Path) -> ImportSourceAdmission:
    """Refuse an inadmissible source before asking the daemon to schedule it.

    The daemon runs the same read-only admissibility check before it accepts
    an ingest.  Running it here too costs nothing and lets the operator see
    "this export shape is not parseable" without a scheduled operation that
    can only fail later.
    """
    from polylogue.operations.import_operations import prepare_import_source_admission

    try:
        admission = prepare_import_source_admission(staged)
    except (OSError, ValueError) as exc:
        fail("import", f"Staged input could not be authenticated: {exc}")
    preflight = admission.preflight
    if preflight.admissible:
        return admission
    fail(
        "import",
        f"{preflight.error_code}: {preflight.summary()}\n  The staged copy was left in place at {staged}.",
    )


def _submit_ingest(
    env: AppEnv, *, staged: Path, admission: ImportSourceAdmission
) -> tuple[dict[str, object], dict[str, object]]:
    """Submit the declared ``ingest`` operation and report its acceptance.

    Returns an :class:`~polylogue.operations.import_contracts.ImportOperation`
    shaped mapping and the accepted operation envelope ``--wait`` follows.
    Acceptance is decided exactly as the daemon's own ingest route decides it:
    a durable ``accepted_reference`` *and* an outcome that admits the work.
    Anything else is reported as a failure, never as success.
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
        "source_path": admission.request.source_path,
        "source_name": admission.request.source_name,
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
            f"  The staged entry was left in place at {staged}.",
        )
    except OperationKernelError as exc:
        fail(
            "import",
            f"Daemon returned an unusable ingest response ({exc}); refusing to claim success.\n"
            f"  The staged entry was left in place at {staged}.",
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
    }, envelope


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
    "--explain",
    is_flag=True,
    help="Explain detector/parser decisions without scheduling daemon import.",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for the daemon to finish the ingest; with --demo, also verify the demo archive.",
)
@click.option(
    "--timeout",
    "wait_timeout_s",
    type=click.FloatRange(min=0.001),
    default=30.0,
    show_default=True,
    help="Seconds to wait for --wait convergence.",
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
    explain: bool,
    wait: bool,
    wait_timeout_s: float,
    with_overlays: bool,
    output_format: str,
) -> None:
    """Schedule a file or directory for import by the running daemon.

    Stages PATH into the archive's import staging directory and asks the
    running polylogued daemon to ingest it. The command is truthful: it
    either confirms the daemon accepted scheduling (with a pointer to
    'polylogue ops status' for observable progress) or fails with an
    actionable error. It never reports success without observable
    processing.
    """
    # The declared ``ingest`` operation goes over the archive-scoped daemon
    # socket; the browser API URL (``POLYLOGUE_DAEMON_URL``) selects nothing
    # here, so the command takes no URL option.
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

    if with_overlays and not demo:
        fail("import", "--with-overlays is currently supported only with --demo.")
    if with_overlays and not wait:
        fail("import", "--with-overlays requires --demo --wait so overlays attach to ingested sessions.")

    if demo:
        if path is not None:
            fail("import", "Use either PATH or --demo, not both.")
        requested_source = _materialize_demo_source()
        staged = _stage_for_daemon(requested_source)
    else:
        if path is None:
            fail("import", "Provide a source PATH or pass --demo.")
        requested_source = path
        staged = _stage_for_daemon(requested_source)

    from polylogue.cli.shared.helpers import load_effective_config

    admission = _preflight_or_fail(staged)
    raw, accepted_envelope = _submit_ingest(env, staged=staged, admission=admission)

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

    if not wait:
        return
    env.ui.console.print(f"[bold]Waiting:[/bold] ingest convergence (timeout {wait_timeout_s:g}s)")
    _wait_for_ingest(env, accepted_envelope, timeout_s=wait_timeout_s)
    if not demo:
        env.ui.console.print(f"[bold green]Ingested:[/bold green] {staged}")
        return

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
