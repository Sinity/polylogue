"""The browser process forwards to the daemon without acquiring archive authority."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from polylogue.daemon.browser_host import create_browser_app


@pytest.mark.asyncio
async def test_browser_host_forwards_auth_query_and_cookie_without_leaking_backend_origin() -> None:
    observed: list[httpx.Request] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/api/web-auth/session":
            return httpx.Response(
                200,
                json={"ok": True},
                headers={"Set-Cookie": "polylogue_web=secret; HttpOnly; SameSite=Strict; Path=/"},
            )
        return httpx.Response(200, json={"items": [], "continuation": "opaque-next", "outcome": {"state": "empty"}})

    app = create_browser_app(
        "http://127.0.0.1:8766",
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        credential = await browser.post(
            "/api/web-auth/session",
            headers={"Origin": "http://127.0.0.1:8767", "X-Polylogue-Web-Client": "1"},
        )
        page = await browser.get(
            "/api/query-units?continuation=q2.opaque%2Ftoken",
            headers={"Authorization": "Bearer caller-token", "Cookie": "polylogue_web=secret"},
        )

    assert credential.status_code == 200
    assert credential.headers["set-cookie"] == "polylogue_web=secret; HttpOnly; SameSite=Strict; Path=/"
    assert observed[0].headers["host"] == "127.0.0.1:8766"
    assert observed[0].headers["origin"] == "http://127.0.0.1:8766"
    assert observed[1].url.query == b"continuation=q2.opaque%2Ftoken"
    assert observed[1].headers["authorization"] == "Bearer caller-token"
    assert observed[1].headers["cookie"] == "polylogue_web=secret"
    assert page.json()["continuation"] == "opaque-next"


@pytest.mark.asyncio
async def test_browser_host_rejects_foreign_host_and_origin_before_daemon() -> None:
    calls = 0

    async def upstream(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    app = create_browser_app("http://127.0.0.1:8766", upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        foreign_host = await browser.get("/api/status", headers={"Host": "attacker.example"})
        foreign_origin = await browser.post(
            "/api/web-auth/session",
            headers={"Origin": "http://attacker.example"},
        )
    assert foreign_host.status_code == 403
    assert foreign_origin.status_code == 403
    assert calls == 0


@pytest.mark.asyncio
async def test_browser_host_wildcard_bind_accepts_loopback_host_only() -> None:
    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    app = create_browser_app(
        "http://127.0.0.2:8766", bind_host="0.0.0.0", upstream_transport=httpx.MockTransport(upstream)
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        admitted = await browser.get("/api/status")
        denied = await browser.get("/api/status", headers={"Host": "attacker.example"})
    assert admitted.status_code == 200
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_browser_host_reports_daemon_loss_without_database_fallback() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon stopped", request=request)

    app = create_browser_app("http://127.0.0.1:8766", upstream_transport=httpx.MockTransport(unavailable))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        live = await browser.get("/healthz/live")
        status = await browser.get("/api/status")
    assert live.status_code == 200
    assert status.status_code == 503
    assert status.json()["error"] == "daemon_unavailable"


@pytest.mark.asyncio
async def test_browser_host_preserves_sse_frames() -> None:
    frames = b'id: 42\nevent: update\ndata: {"outcome":"degraded"}\n\n'

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=frames, headers={"Content-Type": "text/event-stream"})

    app = create_browser_app("http://127.0.0.1:8766", upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        response = await browser.get("/api/events")
    assert response.status_code == 200
    assert response.content == frames
    assert response.headers["content-type"] == "text/event-stream"


@pytest.mark.asyncio
async def test_browser_host_closes_upstream_client_when_send_is_cancelled() -> None:
    class WaitingTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.closed = False

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.entered.set()
            await asyncio.Event().wait()
            raise AssertionError("request was not cancelled")

        async def aclose(self) -> None:
            self.closed = True

    upstream = WaitingTransport()
    app = create_browser_app("http://127.0.0.1:8766", upstream_transport=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        pending = asyncio.create_task(browser.get("/api/status"))
        await upstream.entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert upstream.closed


@pytest.mark.asyncio
async def test_browser_host_does_not_finish_partial_json_as_success() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"partial":'
            raise httpx.ReadError("upstream disappeared")

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream(), headers={"Content-Type": "application/json"})

    app = create_browser_app("http://127.0.0.1:8766", upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        with pytest.raises(httpx.ReadError):
            await browser.get("/api/status")


@pytest.mark.asyncio
async def test_browser_host_marks_midstream_sse_failure_as_unavailable() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'event: update\ndata: {"id":1}\n\n'
            raise httpx.ReadError("upstream disappeared")

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream(), headers={"Content-Type": "text/event-stream"})

    app = create_browser_app("http://127.0.0.1:8766", upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        response = await browser.get("/api/events")

    assert response.content.endswith(b'event: error\ndata: {"error":"daemon_unavailable"}\n\n')


@pytest.mark.asyncio
async def test_browser_assets_survive_daemon_loss(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"index.ts": {"isEntry": True, "name": "archive-overview", "file": "app-a1b2c3d4.js"}}),
        encoding="utf-8",
    )
    (tmp_path / "app-a1b2c3d4.js").write_bytes(b"export {};")
    app = create_browser_app("http://127.0.0.1:8766", webui_dist_root=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1:8767") as browser:
        asset = await browser.get("/assets/app-a1b2c3d4.js")
        unchanged = await browser.get("/assets/app-a1b2c3d4.js", headers={"If-None-Match": asset.headers["etag"]})
    assert asset.status_code == 200
    assert asset.content == b"export {};"
    assert unchanged.status_code == 304
