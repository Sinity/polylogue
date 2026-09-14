"""The webui asset-unavailable event names a route, never a query value.

When ``WebUIAssetBundle.discover`` fails, every webui page route emits
``daemon.webui.assets_unavailable`` at ERROR. The ``route`` field is built
from the raw request target, so a request whose query string carries a
mistyped secret (``/webui/search?q=<secret>&token=<secret>``) wrote that
secret into the daemon event log, where it outlives the request and is read
by operators and log shippers.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from polylogue.daemon.http import DaemonAPIHandler
from polylogue.daemon.webui import WebUIAssetError

_SECRET = "sk-test-not-a-real-credential-9f2c"
_REQUEST_PATH_QUERY = f"?q={_SECRET}&token={_SECRET}"


def _handler(path: str) -> DaemonAPIHandler:
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.path = path
    handler.server = SimpleNamespace(webui_dist_root=None)  # type: ignore[assignment]
    return handler


_ROUTES: list[tuple[str, str, Callable[[DaemonAPIHandler], None]]] = [
    ("archive_overview", "/webui", lambda h: h._serve_webui_archive_overview()),
    ("session_list", "/webui/sessions", lambda h: h._serve_webui_session_list({})),
    ("session_read", "/webui/session", lambda h: h._serve_webui_session_read("codex-session:s1")),
    ("search", "/webui/search", lambda h: h._serve_webui_search({})),
    ("observability", "/webui/observability", lambda h: h._serve_webui_observability()),
    ("cost", "/webui/cost", lambda h: h._serve_webui_cost()),
]


@pytest.mark.parametrize(("name", "route", "invoke"), _ROUTES, ids=[row[0] for row in _ROUTES])
def test_asset_discovery_failure_logs_route_without_query_values(
    name: str,
    route: str,
    invoke: Callable[[DaemonAPIHandler], None],
) -> None:
    """The emitted ``route`` is the path alone and the query never appears.

    Anti-vacuity: the request path deliberately carries a query string whose
    values are secret-shaped. Restoring ``route=self.path`` at this emit site
    puts ``?q=<secret>&token=<secret>`` into the captured event and both
    assertions below go red. A path with no query would prove nothing, so the
    test also asserts the query is present in what the handler received.
    """
    from polylogue.logging import capture

    request_path = route + _REQUEST_PATH_QUERY
    handler = _handler(request_path)
    assert "?" in handler.path, "request target must carry a query or this test is vacuous"

    sent: list[tuple[HTTPStatus, str]] = []

    def record_html(_self: Any, status: HTTPStatus, body: str) -> None:
        sent.append((status, body))

    with patch(
        "polylogue.daemon.webui.WebUIAssetBundle.discover",
        side_effect=WebUIAssetError("Vite manifest is missing from the packaged WebUI"),
    ):
        with patch.object(DaemonAPIHandler, "_send_webui_html", record_html):
            with capture() as records:
                invoke(handler)

    assert sent and sent[0][0] == HTTPStatus.SERVICE_UNAVAILABLE
    events = [r for r in records if r["event"] == "daemon.webui.assets_unavailable"]
    assert len(events) == 1, f"{name} did not emit exactly one asset-unavailable event"
    event = events[0]
    assert event["route"] == route
    assert "?" not in str(event["route"])
    # Nothing else in the serialized event may carry the query value either.
    assert _SECRET not in repr(event)
