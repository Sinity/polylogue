"""Sign a browser into the daemon's WebUI when the CLI opens it.

Loopback is not identity, so the daemon serves shell HTML only to the owner's
credential. The CLI can read the owner-only bearer token; it exchanges that
for a one-time sign-in ticket and hands the ticket to the browser in the URL
fragment, which never reaches a server log or a ``Referer`` header. The
browser's sign-in page redeems it for the first-party cookie.
"""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from polylogue.cli.shared.types import AppEnv

_TICKET_TIMEOUT_S = 5.0


def _api_token(env: AppEnv) -> str | None:
    from polylogue.cli.shared.helpers import load_effective_config
    from polylogue.daemon.api_auth import resolve_api_auth_token

    config = load_effective_config(env)
    return resolve_api_auth_token(
        getattr(config, "api_auth_token", None),
        allow_no_auth=getattr(config, "api_allow_no_auth", False),
    )


def signed_in_web_url(env: AppEnv, daemon_url: str, web_url: str) -> str:
    """Return *web_url* carrying a one-time sign-in ticket, when one can be minted.

    Without a configured token the daemon serves the shell openly, so the URL
    is returned unchanged. If the daemon cannot mint a ticket (not running,
    refused), the plain URL is returned: the browser then shows the sign-in
    page, which says how to sign in, rather than a CLI failure.
    """

    token = _api_token(env)
    if not token:
        return web_url
    request = Request(
        f"{daemon_url.rstrip('/')}/api/web-auth/ticket",
        data=b"",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlopen(request, timeout=_TICKET_TIMEOUT_S) as response:
            payload = json.load(response)
    except (URLError, OSError, ValueError):
        return web_url
    ticket = payload.get("ticket") if isinstance(payload, dict) else None
    if not isinstance(ticket, str) or not ticket:
        return web_url
    return f"{web_url}#polylogue-ticket={quote(ticket, safe='')}"


__all__ = ["signed_in_web_url"]
