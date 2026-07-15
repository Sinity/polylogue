"""Daemon commands for the local browser-capture receiver."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import get_args

import click
from pydantic import ValidationError

from polylogue.browser_capture.launch_jobs import enqueue_launch_job
from polylogue.browser_capture.models import (
    BrowserLaunchAttachmentInput,
    BrowserLaunchJobRequest,
    BrowserPostCommandRequest,
    BrowserPostProvider,
    BrowserPostTarget,
)
from polylogue.browser_capture.receiver import (
    BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV,
    BROWSER_POST_ENABLED_ENV,
    BrowserPostDisabledError,
    browser_post_enabled,
    enqueue_post_command,
    load_or_mint_receiver_token,
    resolve_receiver_auth_token,
)
from polylogue.browser_capture.server import make_server
from polylogue.browser_capture.work_package import build_sol_pro_work_package
from polylogue.core.json import dumps
from polylogue.daemon.status import browser_capture_status_payload


@click.group("browser-capture")
def browser_capture_command() -> None:
    """Run and inspect the browser-capture receiver."""


@browser_capture_command.command("status")
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def status_command(spool_path: Path | None, output_format: str | None) -> None:
    """Show receiver configuration and capture-spool target."""
    payload = browser_capture_status_payload(spool_path, include_spool_path=True)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo("Browser capture receiver")
    click.echo(f"Spool: {'ready' if payload.get('spool_ready') else 'unavailable'}")
    origins = payload.get("allowed_origins", [])
    origin_text = ", ".join(str(item) for item in origins) if isinstance(origins, list) else str(origins)
    click.echo(f"Allowed origins: {origin_text}")


@browser_capture_command.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--auth-token", "auth_token", default=None, help="Bearer token; auto-minted/loaded if not given.")
@click.option(
    "--allow-no-auth",
    is_flag=True,
    default=False,
    envvar=BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV,
    help=(
        f"Serve with no bearer token at all (same effect as {BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV}=1). "
        "Any local process can then read/post to the receiver -- default OFF."
    ),
)
def serve_command(host: str, port: int, spool_path: Path | None, auth_token: str | None, allow_no_auth: bool) -> None:
    """Run the local browser-capture receiver.

    Requires a bearer token by default: an explicit ``--auth-token`` wins,
    otherwise one is auto-minted/loaded from a 0600 file (see
    ``browser-capture token show``). Pass ``--allow-no-auth`` to opt out.
    """
    resolved_token = resolve_receiver_auth_token(auth_token, allow_no_auth=allow_no_auth)
    server = make_server(host, port, spool_path=spool_path, auth_token=resolved_token)
    click.echo(f"Listening on http://{host}:{port}")
    click.echo(f"Writing captures to {server.config.spool_path}")
    if resolved_token is None:
        click.echo("WARNING: no bearer token configured -- any local process can read/post to this receiver")
    else:
        click.echo("Auth: bearer token required (run `polylogued browser-capture token show` to view/pair it)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping browser capture receiver")
    finally:
        server.server_close()


@browser_capture_command.group("token")
def token_group() -> None:
    """Manage the browser-capture receiver's local pairing token."""


@token_group.command("show")
@click.option("--rotate", is_flag=True, help="Mint a new token, invalidating the previous one.")
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def token_show(rotate: bool, output_format: str | None) -> None:
    """Print the receiver's bearer token, minting one if none exists yet.

    Paste this into the extension popup's "Receiver token" field to pair it
    with a receiver that requires authentication (the default posture).
    """
    token = load_or_mint_receiver_token(rotate=rotate)
    if output_format == "json":
        click.echo(dumps({"token": token, "rotated": rotate}))
        return
    click.echo(token)


@browser_capture_command.command("post")
@click.option("--provider", type=click.Choice(list(get_args(BrowserPostProvider))), required=True)
@click.option(
    "--conversation-id",
    "conversation_id",
    default="new",
    show_default=True,
    help='Provider-native conversation id, or "new" to request a fresh thread.',
)
@click.option("--project-ref", "project_ref", default=None, help="Optional provider project/workspace ref.")
@click.option("--text", required=True, help="Prompt text to post into the conversation.")
@click.option("--command-id", "command_id", default=None, help="Optional explicit command id.")
@click.option(
    "--submit/--no-submit",
    "submit",
    default=False,
    show_default=True,
    help="Request the extension to actually click send. Default is a dry-run that fills the composer only.",
)
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def post_command(
    provider: str,
    conversation_id: str,
    project_ref: str | None,
    text: str,
    command_id: str | None,
    submit: bool,
    spool_path: Path | None,
    output_format: str | None,
) -> None:
    """Enqueue an outbound post command for the extension to deliver.

    Safety: refused unless POLYLOGUE_BROWSER_POST_ENABLED=1 is set. The command
    is only queued here; the running receiver serves it to the extension via
    GET /v1/post-commands.
    """
    try:
        request = BrowserPostCommandRequest(
            provider=provider,  # type: ignore[arg-type]
            target=BrowserPostTarget(conversation_id=conversation_id, project_ref=project_ref),
            text=text,
            command_id=command_id,
            submit=submit,
        )
        command = enqueue_post_command(request, spool_path=spool_path)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from None
    except BrowserPostDisabledError:
        raise click.ClickException(
            f"Outbound posting is disabled. Set {BROWSER_POST_ENABLED_ENV}=1 to enable (default OFF safety guard)."
        ) from None
    payload = command.model_dump(mode="json", exclude_none=True)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo(f"Queued post command {command.command_id}")
    click.echo(f"Provider: {command.provider}  target: {command.target.conversation_id}")
    click.echo(f"Submit (click send): {command.submit}")
    if not browser_post_enabled():  # pragma: no cover - defensive; enqueue would have raised
        click.echo(f"WARNING: {BROWSER_POST_ENABLED_ENV} is not set")


@browser_capture_command.command("launch")
@click.option(
    "--prompt-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=True,
    help="Narrow per-job scope; the versioned Sol Pro worker contract is prepended automatically.",
)
@click.option(
    "--attachment",
    "attachment_paths",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    multiple=True,
    help="Targeted project input copied into receiver-owned storage (repeatable).",
)
@click.option(
    "--project-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    default=None,
    help="Build one deterministic targeted project pack rooted here.",
)
@click.option("--bead", "bead_ids", multiple=True, help="Full Beads record to include in the project pack.")
@click.option(
    "--source",
    "source_paths",
    type=click.Path(path_type=Path, exists=True, readable=True),
    multiple=True,
    help="File or directory to include under REPO/ in the project pack.",
)
@click.option(
    "--verification",
    "verification_paths",
    type=click.Path(path_type=Path, exists=True, readable=True),
    multiple=True,
    help="Verification receipt to include in the project pack.",
)
@click.option(
    "--full-worktree-fallback",
    is_flag=True,
    help="Explicitly include every tracked/unignored worktree file instead of only --source paths.",
)
@click.option(
    "--cadence",
    "cadence_minutes",
    type=click.Choice(["1", "5", "15", "30", "60"]),
    default="5",
    show_default=True,
)
@click.option("--title", "job_title", required=True, help="Readable mission title shown at the top of the chat.")
@click.option("--not-before", default=None, help="Optional ISO-8601 earliest launch time.")
@click.option("--job-id", default=None, help="Optional stable external job id.")
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def launch_command(
    prompt_file: Path,
    attachment_paths: tuple[Path, ...],
    project_root: Path | None,
    bead_ids: tuple[str, ...],
    source_paths: tuple[Path, ...],
    verification_paths: tuple[Path, ...],
    full_worktree_fallback: bool,
    cadence_minutes: str,
    job_title: str,
    not_before: str | None,
    job_id: str | None,
    spool_path: Path | None,
    output_format: str | None,
) -> None:
    """Queue one ordinary Chat · GPT-5.6 Sol · Pro work package.

    Enqueue is the operator authorization boundary. Every supplied file is
    copied and hash-pinned before an extension instance can lease the job.
    """
    try:
        scope_prompt = prompt_file.read_text(encoding="utf-8")
        attachments = [
            BrowserLaunchAttachmentInput(
                name=path.name,
                mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                content_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
            )
            for path in attachment_paths
        ]
        if project_root is not None:
            package = build_sol_pro_work_package(
                repo_root=project_root,
                job_title=job_title,
                scope_prompt=scope_prompt,
                bead_ids=bead_ids,
                source_paths=source_paths,
                verification_paths=verification_paths,
                full_worktree_fallback=full_worktree_fallback,
            )
            attachments.insert(
                0,
                BrowserLaunchAttachmentInput(
                    name=package.name,
                    mime_type="application/gzip",
                    content_base64=base64.b64encode(package.content).decode("ascii"),
                ),
            )
        elif bead_ids or source_paths or verification_paths or full_worktree_fallback:
            raise ValueError("--bead/--source/--verification/full fallback require --project-root")
        job = enqueue_launch_job(
            BrowserLaunchJobRequest(
                job_title=job_title,
                scope_prompt=scope_prompt,
                attachments=attachments,
                cadence_minutes=int(cadence_minutes),  # type: ignore[arg-type]
                not_before=not_before,
                job_id=job_id,
            ),
            spool_path=spool_path,
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from None
    payload = job.model_dump(mode="json", exclude_none=True)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo(f"Queued launch job {job.job_id}")
    click.echo("Target: Chat · GPT-5.6 Sol · Pro")
    click.echo(f"Cadence: {job.cadence_minutes}m  attachments: {len(job.attachments)}")


__all__ = ["browser_capture_command", "launch_command", "post_command", "serve_command", "status_command"]
