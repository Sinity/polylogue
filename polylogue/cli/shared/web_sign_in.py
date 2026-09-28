"""Sign a browser into the daemon's WebUI when the CLI opens it.

Loopback is not identity, so the daemon serves shell HTML only to the owner's
credential. The CLI can read the owner-only bearer token; it exchanges that
for a one-time sign-in ticket and hands the ticket to the browser in the URL
fragment, which never reaches a server log or a ``Referer`` header. The
ticket always opens the daemon's exchange page (``/web-auth/sign-in``), which
consumes and clears the fragment even when the browser already holds a
credential, then continues to the requested page.

The ticketed URL is never handed directly to ``webbrowser.open``: on Linux
its backends spawn the browser (or an ``xdg-open``-style helper) with the
full URL in ``argv``, which another local uid can read from
``/proc/<pid>/cmdline`` during the ticket's lifetime. Instead the ticketed
URL is written into an owner-only-readable local redirect file
(``tempfile.mkstemp`` mode 0600) and a ``file://`` URL to *that* is returned;
the browser process's argv then carries only a filesystem path, and the
ticket travels solely inside that file's content, read by the browser after
the process is already running as the invoking user.
"""

from __future__ import annotations

import json
import os
import tempfile
from html import escape as _html_escape
from urllib.error import URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from polylogue.cli.shared.types import AppEnv

_TICKET_TIMEOUT_S = 5.0


def _local_redirect_url(target: str) -> str:
    """Write an owner-only redirect page to *target* and return its ``file://`` URL."""
    fd, path = tempfile.mkstemp(prefix="polylogue-signin-", suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                '<!doctype html><meta charset="utf-8">'
                f"<script>location.replace({json.dumps(target)});</script>"
                f'<a href="{_html_escape(target)}">Continue</a>'
            )
    except BaseException:
        os.unlink(path)
        raise
    return f"file://{path}"


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
    target = urlsplit(web_url)
    next_path = target.path or "/"
    if target.query:
        next_path = f"{next_path}?{target.query}"
    exchange = urlunsplit((target.scheme, target.netloc, "/web-auth/sign-in", f"next={quote(next_path, safe='/')}", ""))
    ticketed = f"{exchange}#polylogue-ticket={quote(ticket, safe='')}"
    return _local_redirect_url(ticketed)


__all__ = ["signed_in_web_url"]
