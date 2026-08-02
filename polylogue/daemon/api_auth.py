"""Daemon HTTP API bearer-token lifecycle and CLI (polylogue-rzve).

Mirrors :mod:`polylogue.browser_capture.receiver`'s auto-mint/load/rotate
pattern for the receiver's pairing token. ``docs/daemon.md`` documents
``--api-auth-token`` as "auto-generated if not provided" but, before this
module existed, nothing actually minted one: the daemon simply started the
API with ``auth_token=None`` (fully open on loopback) whenever the flag was
omitted. That is the "ambiguous unauthenticated state" the owning bead
(polylogue-rzve) calls out -- a remotely-reachable, write-capable API must
not silently default to open. This module closes the gap by giving the API
the same explicit contract the browser-capture receiver already has: an
auto-minted, persisted 0600 token by default, with a loud, explicit
``--api-allow-no-auth`` opt-out for the rare setup that wants the previous
fully-open posture.
"""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

import click

from polylogue.core.json import dumps
from polylogue.logging import get_logger
from polylogue.paths import api_auth_token_path

logger = get_logger(__name__)

#: Entropy (bytes, pre-base64) for an auto-minted API bearer token. Matches
#: RECEIVER_TOKEN_ENTROPY_BYTES in polylogue.browser_capture.receiver.
API_AUTH_TOKEN_ENTROPY_BYTES = 32

#: Env flag that must equal ``"1"`` before the daemon API will start with no
#: bearer token at all. Default OFF, mirroring
#: BROWSER_CAPTURE_ALLOW_NO_AUTH_ENV: without a token any local process (not
#: just an authenticated client) can read/write through the API, so
#: auto-minting a token closes that hole by default and this flag is the
#: explicit, logged escape hatch.
API_ALLOW_NO_AUTH_ENV = "POLYLOGUE_API_ALLOW_NO_AUTH"


def _is_trusted_token_file(target: Path) -> bool:
    """Refuse to trust a token file this process does not exclusively own.

    polylogue-n6pz: the cross-uid audit finding this closes was exactly the
    mistake of assuming a filesystem boundary holds without checking it. Do
    not repeat that assumption for the token file itself -- an unmint'd file
    at ``target`` (planted before this process ever ran, restored from a
    backup with looser bits, or living under a misconfigured/shared config
    root) could hand out an attacker-known token while the operator believes
    the API is private. A symlink is refused outright (never follow it into
    another owner's file); a regular file must be owned by our own uid and
    carry no group/other permission bits.
    """
    try:
        info = target.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return False
    if info.st_uid != os.getuid():
        return False
    return stat.S_IMODE(info.st_mode) & 0o077 == 0


def load_or_mint_api_auth_token(path: Path | None = None, *, rotate: bool = False) -> str:
    """Return the daemon API's persisted bearer token, minting one on first use.

    Written atomically (mkstemp + ``fchmod(0o600)`` before any bytes land,
    then ``os.replace``) so the token is never briefly world-readable under a
    permissive umask -- identical idiom to
    :func:`polylogue.browser_capture.receiver.load_or_mint_receiver_token`.
    An existing file is trusted only if :func:`_is_trusted_token_file` passes;
    otherwise it is treated as absent and a fresh token is minted in its
    place (logging so the operator can investigate why a stray/loose file was
    sitting at the token path).
    """
    target = path if path is not None else api_auth_token_path()
    if not rotate and target.exists():
        if _is_trusted_token_file(target):
            existing = target.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        else:
            logger.warning(
                "daemon_api.token_file_untrusted",
                path=str(target),
                action="reminting",
                reason="existing token file is not an owner-only regular file we exclusively own",
            )
    token = secrets.token_urlsafe(API_AUTH_TOKEN_ENTROPY_BYTES)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        os.replace(tmp_path, target)
    except BaseException:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
    return token


def resolve_api_auth_token(
    explicit_token: str | None,
    *,
    allow_no_auth: bool = False,
    token_path: Path | None = None,
) -> str | None:
    """Return the bearer token the daemon API should require before serving.

    An explicitly configured token (CLI flag / config layer) always wins.
    Otherwise the daemon auto-mints/loads a persisted 0600 token so the API
    is never started ambiguously open. ``allow_no_auth`` is the explicit,
    loudly-logged opt-out for a deployment that wants the fully-open posture
    (e.g. a firewalled loopback-only dev box).
    """
    if explicit_token:
        return explicit_token
    if allow_no_auth:
        logger.warning(
            "daemon_api.auth_disabled",
            reason="allow_no_auth explicitly set",
            risk="any local process can read/write through the daemon API",
        )
        return None
    return load_or_mint_api_auth_token(token_path)


@click.group("api")
def api_command() -> None:
    """Manage the daemon HTTP API's auto-minted bearer token."""


@api_command.group("token")
def token_group() -> None:
    """Manage the daemon API's local bearer token."""


@token_group.command("show")
@click.option("--rotate", is_flag=True, help="Mint a new token, invalidating the previous one.")
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def token_show(rotate: bool, output_format: str | None) -> None:
    """Print the daemon API's bearer token, minting one if none exists yet.

    Use this value for ``Authorization: Bearer <token>`` when calling the
    daemon API from a script, or when configuring ``daemon.api.auth_token`` /
    ``POLYLOGUE_API_AUTH_TOKEN`` for a client (``polylogue ops rebuild-index``
    and friends already read this from config automatically).
    """
    token = load_or_mint_api_auth_token(rotate=rotate)
    if output_format == "json":
        click.echo(dumps({"token": token, "rotated": rotate}))
        return
    click.echo(token)


__all__ = [
    "API_ALLOW_NO_AUTH_ENV",
    "API_AUTH_TOKEN_ENTROPY_BYTES",
    "api_command",
    "load_or_mint_api_auth_token",
    "resolve_api_auth_token",
]
