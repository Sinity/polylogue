"""Daemon commands for the local browser-capture receiver."""

from __future__ import annotations

import http.client
import json
import mimetypes
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import get_args

import click
from ijson.common import JSONError
from pydantic import ValidationError

from polylogue.browser_capture.actions import (
    BrowserActionConflictError,
    BrowserActionQuotaError,
    enqueue_action,
    store_action_attachment,
)
from polylogue.browser_capture.models import (
    BrowserActionAttachmentInput,
    BrowserActionPresentation,
    BrowserActionProvider,
    BrowserActionRequest,
    BrowserActionTarget,
    BrowserCaptureReceiverStatusChallengePayload,
    BrowserCaptureReceiverStatusPayload,
)
from polylogue.browser_capture.native_host import install_native_host
from polylogue.browser_capture.pairing import (
    PAIRING_CODE_DEFAULT_TTL_SECONDS,
    mint_pairing_code,
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


@click.group("browser-capture")
def browser_capture_command() -> None:
    """Run and inspect the browser-capture receiver."""


def _read_receiver_credential(path: Path, *, secret: bool) -> str:
    from polylogue.browser_capture.receiver import ReceiverCredentialError, read_receiver_credential

    try:
        return read_receiver_credential(path, secret=secret)
    except ReceiverCredentialError as exc:
        raise click.ClickException(str(exc)) from exc


@contextmanager
def _observed_receiver_status(
    host: str | None, port: int | None, allow_no_auth: bool | None
) -> Iterator[dict[str, object]]:
    """Own the selected peer and disk-backed status until its consumer finishes."""
    from polylogue.browser_capture.native_host import (
        ReceiverNetworkReadError,
        ReceiverObservationStorageError,
        ReceiverResponseAuthenticationError,
        ReceiverResponsePayloadError,
        _receiver_response_document,
    )
    from polylogue.browser_capture.receiver import receiver_status_proof
    from polylogue.config import resolve_runtime_config
    from polylogue.paths import browser_capture_receiver_identity_path, browser_capture_receiver_token_path

    config = resolve_runtime_config().settings
    host = config.browser_capture_host if host is None else host
    port = config.browser_capture_port if port is None else port
    allow_no_auth = config.browser_capture_allow_no_auth if allow_no_auth is None else allow_no_auth
    expected_identity = _read_receiver_credential(browser_capture_receiver_identity_path(), secret=False)
    token = None if allow_no_auth else _read_receiver_credential(browser_capture_receiver_token_path(), secret=True)
    connection = http.client.HTTPConnection(host, port, timeout=None)

    def response_for(
        method: str, path: str, *, body: str | None = None, headers: dict[str, str] | None = None
    ) -> http.client.HTTPResponse:
        try:
            connection.request(method, path, body=body, headers=headers or {})
            return connection.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            raise click.ClickException("receiver_unreachable") from exc

    try:
        status_auth: tuple[str, str, str] | None = None
        if not allow_no_auth:
            assert token is not None
            challenge_response = response_for(
                "GET", "/v1/receiver/status-challenge", headers={"Connection": "keep-alive"}
            )
            if challenge_response.status != 200:
                raise click.ClickException(f"receiver_status_refused_{challenge_response.status}")
            try:
                with _receiver_response_document(challenge_response) as document:
                    if not isinstance(document, dict):
                        raise ValueError("status challenge must be an object")
                    issued = BrowserCaptureReceiverStatusChallengePayload.model_validate(
                        {"receiver_id": document.get("receiver_id"), "challenge": document.get("challenge")}
                    )
                    challenge = issued.challenge
                    if issued.receiver_id != expected_identity:
                        raise click.ClickException("receiver_identity_mismatch")
            except (ValueError, ValidationError, JSONError) as exc:
                raise click.ClickException("receiver_status_invalid_payload") from exc
            if connection.sock is None:
                raise click.ClickException("receiver_authentication_failed")
            # Never deliver an authenticated request on a replacement connection.
            connection.auto_open = 0
            request = {
                "receiver_id": expected_identity,
                "challenge": challenge,
                "proof": receiver_status_proof(token, expected_identity, challenge),
            }
            status_auth = (token, expected_identity, challenge)
            response = response_for(
                "POST",
                "/v1/receiver/status-attest",
                body=json.dumps(request),
                headers={"Content-Type": "application/json"},
            )
        else:
            response = response_for("GET", "/v1/status")
        if response.status != 200:
            raise click.ClickException(f"receiver_status_refused_{response.status}")
        with _receiver_response_document(response, status_auth=status_auth) as document:
            try:
                if not isinstance(document, dict):
                    raise ValueError("status must be an object")
                origins = document.get("allowed_origins")
                if not isinstance(origins, list) or any(not isinstance(origin, str) for origin in origins):
                    raise ValueError("invalid allowed_origins")
                fields = {
                    key: document[key]
                    for key in BrowserCaptureReceiverStatusPayload.model_fields
                    if key != "allowed_origins" and key in document
                }
                payload = BrowserCaptureReceiverStatusPayload.model_validate({**fields, "allowed_origins": []})
                if payload.receiver_id != expected_identity:
                    raise click.ClickException("receiver_identity_mismatch")
                if payload.auth_required is allow_no_auth:
                    raise click.ClickException("receiver_authentication_policy_mismatch")
                observed = payload.model_dump(mode="json")
                observed["allowed_origins"] = origins
            except (ValueError, ValidationError, JSONError) as exc:
                raise click.ClickException("receiver_status_invalid_payload") from exc
            yield observed
    except ReceiverResponseAuthenticationError as exc:
        raise click.ClickException("receiver_authentication_failed") from exc
    except ReceiverResponsePayloadError as exc:
        raise click.ClickException("receiver_status_invalid_payload") from exc
    except ReceiverObservationStorageError as exc:
        raise click.ClickException("receiver_observation_storage_failed") from exc
    except ReceiverNetworkReadError as exc:
        raise click.ClickException("receiver_unreachable") from exc
    finally:
        connection.close()


@browser_capture_command.command("status")
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.option("--host", default=None, help="Receiver host; defaults to resolved settings.")
@click.option("--port", default=None, type=int, help="Receiver port; defaults to resolved settings.")
@click.option(
    "--allow-no-auth/--require-auth",
    is_flag=True,
    default=None,
    envvar=BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV,
    help="Override resolved authentication mode: no credential or required receiver attestation.",
)
def status_command(output_format: str | None, host: str | None, port: int | None, allow_no_auth: bool | None) -> None:
    """Show the bound standalone or daemon receiver's observed policy."""
    with _observed_receiver_status(host, port, allow_no_auth) as payload:
        if output_format == "json":
            for piece in json.JSONEncoder(ensure_ascii=True, allow_nan=False).iterencode(payload):
                click.echo(piece, nl=False)
            click.echo()
            return
        click.echo("Browser capture receiver")
        click.echo(f"Spool: {'ready' if payload.get('spool_ready') else 'unavailable'}")
        click.echo("Allowed origins: ", nl=False)
        origins = payload["allowed_origins"]
        assert isinstance(origins, list)
        for position, origin in enumerate(origins):
            click.echo((", " if position else "") + str(origin), nl=False)
        click.echo()
        from polylogue.core.json import json_document
        from polylogue.daemon.status import format_browser_capture_policy_lines

        # Only fixed policy scalars are needed by the shared formatter.
        for line in format_browser_capture_policy_lines(
            json_document({"auth_required": payload["auth_required"], "allow_remote": payload["allow_remote"]})
        ):
            click.echo(line)


@browser_capture_command.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
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
def serve_command(host: str, port: int, auth_token: str | None, allow_no_auth: bool) -> None:
    """Run the local browser-capture receiver.

    Requires a bearer token by default: an explicit ``--auth-token`` wins,
    otherwise one is auto-minted/loaded from a 0600 file (see
    ``browser-capture token show``). Pass ``--allow-no-auth`` to opt out.
    """
    resolved_token = resolve_receiver_auth_token(auth_token, allow_no_auth=allow_no_auth)
    from polylogue.config import resolve_runtime_config
    from polylogue.paths import browser_capture_receiver_token_path

    config = resolve_runtime_config().as_config()
    server = make_server(
        host,
        port,
        auth_token=resolved_token,
        auth_token_path=browser_capture_receiver_token_path() if resolved_token is not None else None,
        archive_root=config.archive_root,
        api_auth_token=config.api_auth_token,
        api_allow_no_auth=config.api_allow_no_auth,
    )
    click.echo(f"Listening on http://{host}:{port}")
    click.echo(f"Writing captures to {server.config.spool_path}")
    if resolved_token is None:
        click.echo("WARNING: no bearer token configured -- any local process can read/post to this receiver")
    else:
        click.echo("Auth: bearer token required (run `polylogued browser-capture token show` to view/pair it)")
    from polylogue.daemon.status_snapshot import configure_browser_capture_status

    configure_browser_capture_status(server.config)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping browser capture receiver")
    finally:
        configure_browser_capture_status(None)
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


@browser_capture_command.group("pairing")
def pairing_group() -> None:
    """Pair a fresh extension install without viewing or pasting the bearer token."""


@pairing_group.command("start")
@click.option(
    "--ttl-seconds",
    default=PAIRING_CODE_DEFAULT_TTL_SECONDS,
    show_default=True,
    type=int,
    help="How long the code remains redeemable.",
)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def pairing_start(ttl_seconds: int, output_format: str | None) -> None:
    """Mint a short-lived one-time pairing code.

    Enter the printed code into the extension popup's "Pairing code" field
    through the external HTTP client's pairing route within the TTL. The code
    is single-use. The browser extension uses native transport and never
    receives or stores the receiver bearer.
    """
    minted = mint_pairing_code(ttl_seconds=ttl_seconds)
    if output_format == "json":
        click.echo(
            dumps({"code": minted.code, "expires_at_ms": minted.expires_at_ms, "ttl_seconds": minted.ttl_seconds})
        )
        return
    click.echo(f"Pairing code: {minted.code}")
    click.echo(f"Expires in {minted.ttl_seconds}s -- redeem it through the external HTTP client pairing route.")


@browser_capture_command.group("native-host")
def native_host_group() -> None:
    """Install the browser-scoped authenticated receiver transport."""


@native_host_group.command("install")
@click.option("--extension-id", multiple=True, required=True, help="Exact packaged extension ID; repeatable.")
@click.option("--browser", type=click.Choice(["chrome", "firefox"]), default="chrome", show_default=True)
@click.option("--executable", default="polylogue-browser-capture-native-host", show_default=True)
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path), default=None)
def native_host_install(
    extension_id: tuple[str, ...], browser: str, executable: str, manifest_path: Path | None
) -> None:
    """Register native messaging for only the supplied extension IDs."""
    target = install_native_host(extension_id, executable=executable, browser=browser, destination=manifest_path)
    click.echo(f"Installed {target}")


@browser_capture_command.command("capture-health")
@click.option(
    "--limit", default=50, show_default=True, type=int, help="Maximum reports to show; -1 streams all reports."
)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
@click.option("--cursor", default=None, help="Continue a capture-health history snapshot.")
def capture_health_command(limit: int, output_format: str | None, cursor: str | None) -> None:
    """Stream extension-reported capture-health history from the ops tier."""
    from polylogue.core.errors import SchemaSkew
    from polylogue.daemon.events import (
        CAPTURE_HISTORY_PAGE_ROWS,
        CaptureHistoryCursorError,
        CaptureHistoryStorageError,
        capture_health_page,
    )

    if limit < -1:
        raise click.BadParameter("Use a nonnegative count or -1.", param_hint="--limit")
    if limit == 0:
        click.echo(
            dumps({"events": [], "next_cursor": cursor})
            if output_format == "json"
            else "No capture-health events recorded."
        )
        return
    remaining = limit
    emitted = 0
    first = True
    # A JSON document becomes public only after every requested page succeeds.
    # TemporaryFile is private, disk-backed and removed on every exit, including
    # interruption; plain text keeps its declared streaming output.
    try:
        with (
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") if output_format == "json" else nullcontext(None)
        ) as staged:
            page = capture_health_page(
                page_size=CAPTURE_HISTORY_PAGE_ROWS
                if remaining == -1
                else max(1, min(CAPTURE_HISTORY_PAGE_ROWS, remaining)),
                cursor=cursor,
            )
            if staged is not None:
                staged.write('{"events":[')
            while remaining != 0:
                for event in page["events"]:
                    if staged is not None:
                        staged.write(("" if first else ",") + dumps(event))
                    else:
                        raw_payload = event["payload"]
                        payload = raw_payload if isinstance(raw_payload, dict) else {}
                        click.echo(
                            f"{event['ts']}  {payload.get('event', '?')}  provider={payload.get('provider') or '-'}  session={payload.get('provider_session_id') or '-'}"
                        )
                    first = False
                    emitted += 1
                    if remaining > 0:
                        remaining -= 1
                cursor = page["next_cursor"]
                if cursor is None or remaining == 0:
                    break
                page = capture_health_page(
                    page_size=CAPTURE_HISTORY_PAGE_ROWS
                    if remaining == -1
                    else min(CAPTURE_HISTORY_PAGE_ROWS, remaining),
                    cursor=cursor,
                )
            if staged is not None:
                staged.write('],"next_cursor":' + dumps(cursor) + "}\n")
                staged.seek(0)
                shutil.copyfileobj(staged, sys.stdout)
            elif emitted == 0:
                click.echo("No capture-health events recorded.")

    except CaptureHistoryCursorError as exc:
        raise click.ClickException(str(exc)) from exc
    except SchemaSkew as exc:
        raise click.ClickException("schema_skew") from exc
    except CaptureHistoryStorageError as exc:
        raise click.ClickException(exc.code) from exc
    except OSError as exc:
        raise click.ClickException("capture_history_output_failed") from exc


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
    output_format: str | None,
) -> None:
    """Enqueue one provider-neutral action for a replaceable extension."""
    try:
        if (text is None) == (prompt_file is None):
            raise ValueError("provide exactly one of --text or --prompt-file")
        resolved_text = text if text is not None else (prompt_file or Path()).read_text(encoding="utf-8")
        resolved_operation = operation or ("conversation.create" if conversation_id == "new" else "conversation.reply")
        attachments: list[BrowserActionAttachmentInput] = []
        for path in attachment_paths:
            with path.open("rb") as source:
                reference = store_action_attachment(source.read, os.fstat(source.fileno()).st_size)
            attachments.append(
                BrowserActionAttachmentInput(
                    name=path.name,
                    mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    attachment_ref=reference,
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
            spool_path=BrowserCaptureReceiverConfig.default().spool_path,
        )
        action = enqueue_action(request, receiver_id=receiver_identity(receiver_config))
    except (OSError, ValueError, ValidationError, BrowserActionConflictError, BrowserActionQuotaError) as exc:
        raise click.ClickException(str(exc)) from None
    payload = action.model_dump(mode="json", exclude_none=True)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo(f"Queued browser action {action.action_id}")
    click.echo(f"Provider: {action.provider}  operation: {action.operation}")
    click.echo(f"Target: {action.target.conversation_id}  policy: {action.submit_policy}")


__all__ = [
    "action_command",
    "browser_capture_command",
    "capture_health_command",
    "pairing_start",
    "native_host_install",
    "serve_command",
    "status_command",
]
