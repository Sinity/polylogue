"""Daemon commands for the local browser-capture receiver."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import get_args

import click
from pydantic import ValidationError

from polylogue.browser_capture.actions import (
    ACTION_ATTACHMENT_MAX_BYTES,
    ACTION_TOTAL_ATTACHMENT_MAX_BYTES,
    BrowserActionConflictError,
    BrowserActionQuotaError,
    enqueue_action,
)
from polylogue.browser_capture.models import (
    BrowserActionAttachmentInput,
    BrowserActionPresentation,
    BrowserActionProvider,
    BrowserActionRequest,
    BrowserActionTarget,
)
from polylogue.browser_capture.receiver import (
    BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV,
    BrowserCaptureReceiverConfig,
    load_or_mint_receiver_token,
    receiver_identity,
    resolve_receiver_auth_token,
)
from polylogue.browser_capture.server import make_server
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


@browser_capture_command.command("action")
@click.option("--provider", type=click.Choice(list(get_args(BrowserActionProvider))), required=True)
@click.option(
    "--operation",
    type=click.Choice(["conversation.create", "conversation.reply"]),
    default=None,
    help='Defaults to create for conversation-id "new", otherwise reply.',
)
@click.option(
    "--conversation-id",
    "conversation_id",
    default="new",
    show_default=True,
    help='Provider-native conversation id, or "new" to request a fresh thread.',
)
@click.option("--project-ref", "project_ref", default=None, help="Optional provider project/workspace ref.")
@click.option("--conversation-url", default=None, help="Exact first-party target URL, required for routed projects.")
@click.option("--text", default=None, help="Message text. Mutually exclusive with --prompt-file.")
@click.option(
    "--prompt-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    default=None,
    help="Read message text from this file.",
)
@click.option(
    "--attachment",
    "attachment_paths",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    multiple=True,
    help="Hash-pinned input file copied into receiver storage; repeatable.",
)
@click.option("--model-slug", required=True, help="Exact provider model slug advertised by capabilities.")
@click.option("--model-label", required=True, help="Exact provider model label visible at submit.")
@click.option("--effort-label", required=True, help="Exact provider effort label visible at submit.")
@click.option("--action-id", default=None, help="Optional stable action identity.")
@click.option("--idempotency-key", default=None, help="Stable caller retry identity.")
@click.option(
    "--submit/--stage-only",
    "submit",
    default=False,
    show_default=True,
    help="Submit once or only stage a verified provider draft.",
)
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def action_command(
    provider: str,
    operation: str | None,
    conversation_id: str,
    project_ref: str | None,
    conversation_url: str | None,
    text: str | None,
    prompt_file: Path | None,
    attachment_paths: tuple[Path, ...],
    model_slug: str,
    model_label: str,
    effort_label: str,
    action_id: str | None,
    idempotency_key: str | None,
    submit: bool,
    spool_path: Path | None,
    output_format: str | None,
) -> None:
    """Enqueue one provider-neutral action for a replaceable extension."""
    try:
        if (text is None) == (prompt_file is None):
            raise ValueError("provide exactly one of --text or --prompt-file")
        resolved_text = text if text is not None else (prompt_file or Path()).read_text(encoding="utf-8")
        resolved_operation = operation or ("conversation.create" if conversation_id == "new" else "conversation.reply")
        attachments: list[BrowserActionAttachmentInput] = []
        total = 0
        for path in attachment_paths:
            size = path.stat().st_size
            if size > ACTION_ATTACHMENT_MAX_BYTES:
                raise BrowserActionQuotaError(
                    f"browser action attachment exceeds {ACTION_ATTACHMENT_MAX_BYTES} bytes: {path.name}"
                )
            total += size
            if total > ACTION_TOTAL_ATTACHMENT_MAX_BYTES:
                raise BrowserActionQuotaError(
                    f"browser action attachments exceed {ACTION_TOTAL_ATTACHMENT_MAX_BYTES} total bytes"
                )
            attachments.append(
                BrowserActionAttachmentInput(
                    name=path.name,
                    mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    content_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
                )
            )
        request = BrowserActionRequest(
            action_id=action_id,
            idempotency_key=idempotency_key,
            provider=provider,  # type: ignore[arg-type]
            operation=resolved_operation,  # type: ignore[arg-type]
            target=BrowserActionTarget(
                conversation_id=conversation_id,
                conversation_url=conversation_url,
                project_ref=project_ref,
            ),
            text=resolved_text,
            attachments=attachments,
            presentation=BrowserActionPresentation(
                model_slug=model_slug,
                model_label=model_label,
                effort_label=effort_label,
            ),
            submit_policy="submit_once" if submit else "stage_only",
        )
        receiver_config = BrowserCaptureReceiverConfig(
            spool_path=spool_path or BrowserCaptureReceiverConfig.default().spool_path,
            auth_token=resolve_receiver_auth_token(None, allow_no_auth=False),
        )
        action = enqueue_action(
            request,
            receiver_id=receiver_identity(receiver_config),
            spool_path=spool_path,
        )
    except (OSError, ValueError, ValidationError, BrowserActionConflictError, BrowserActionQuotaError) as exc:
        raise click.ClickException(str(exc)) from None
    payload = action.model_dump(mode="json", exclude_none=True)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo(f"Queued browser action {action.action_id}")
    click.echo(f"Provider: {action.provider}  operation: {action.operation}")
    click.echo(f"Target: {action.target.conversation_id}  policy: {action.submit_policy}")


__all__ = ["action_command", "browser_capture_command", "serve_command", "status_command"]
