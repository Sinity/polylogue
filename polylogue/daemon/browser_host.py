"""Isolated ASGI browser transport over the daemon's existing HTTP authority."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from polylogue.core.loopback import is_loopback_host
from polylogue.daemon.web_auth import exact_origin_allowed

_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_METHODS = ("GET", "HEAD", "POST", "DELETE", "PUT", "PATCH", "OPTIONS")


def _error(code: str, status: int) -> JSONResponse:
    return JSONResponse({"ok": False, "error": code, "detail": None, "field": None}, status_code=status)


def _daemon_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or not is_loopback_host(parsed.hostname):
        raise ValueError("browser host daemon origin must be loopback HTTP")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("browser host daemon origin must have no credentials, path, or query")
    return f"http://{parsed.netloc}"


def _public_host_allowed(host_header: str, bind_host: str, port: int) -> bool:
    try:
        parsed = urlsplit(f"//{host_header}")
        hostname = parsed.hostname
        header_port = parsed.port
    except ValueError:
        return False
    if hostname is None or header_port not in {None, port}:
        return False
    if bind_host in {"0.0.0.0", "::"}:
        return is_loopback_host(hostname)
    aliases = {bind_host.lower()}
    if bind_host in {"127.0.0.1", "localhost", "::1"}:
        aliases.update({"127.0.0.1", "localhost", "::1"})
    return hostname.lower() in aliases


def _backend_headers(request: Request, origin: str) -> list[tuple[str, str]]:
    headers = [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in request.scope["headers"]
        if name.decode("latin-1").lower() not in _HOP_HEADERS | {"host", "origin", "referer"}
    ]
    headers.append(("Host", urlsplit(origin).netloc))
    external_origin = request.headers.get("origin")
    if external_origin:
        headers.append(("Origin", origin))
    referer = request.headers.get("referer")
    if referer:
        parsed = urlsplit(referer)
        headers.append(
            ("Referer", urlunsplit(("http", urlsplit(origin).netloc, parsed.path, parsed.query, parsed.fragment)))
        )
    return headers


def _safe_external_request(request: Request, bind_host: str, port: int) -> bool:
    host = request.headers.get("host", "")
    if not _public_host_allowed(host, bind_host, port):
        return False
    if not exact_origin_allowed(request.headers.get("origin", ""), host):
        return False
    referer = request.headers.get("referer")
    if referer:
        parsed = urlsplit(referer)
        if not exact_origin_allowed(f"{parsed.scheme}://{parsed.netloc}", host):
            return False
    return True


def create_browser_app(
    daemon_origin: str,
    *,
    bind_host: str = "127.0.0.1",
    port: int = 8767,
    webui_dist_root: Path | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    """Serve immutable browser assets locally and forward data to the daemon.

    The host has no archive path or writer. Each request owns its HTTP client,
    so a Set-Cookie response cannot enter a shared proxy cookie jar.
    """

    origin = _daemon_origin(daemon_origin)

    async def live(request: Request) -> Response:
        if not _safe_external_request(request, bind_host, port):
            return _error("host_not_allowed", 403)
        return JSONResponse({"ok": True, "status": "healthy", "surface": "browser"})

    async def asset(request: Request) -> Response:
        if not _safe_external_request(request, bind_host, port):
            return _error("host_not_allowed", 403)
        from polylogue.daemon.webui import WebUIAssetBundle, WebUIAssetError

        try:
            selected = WebUIAssetBundle.discover(webui_dist_root).read_asset(request.path_params["name"])
        except FileNotFoundError:
            return _error("not_found", 404)
        except WebUIAssetError:
            return _error("webui_unavailable", 503)
        headers = {
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": selected.etag,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cross-Origin-Resource-Policy": "same-origin",
        }
        if request.headers.get("if-none-match") == selected.etag:
            return Response(status_code=304, headers=headers)
        return Response(selected.body, media_type=selected.content_type, headers=headers)

    async def proxy(request: Request) -> Response:
        if not _safe_external_request(request, bind_host, port):
            return _error("host_not_allowed", 403)
        raw_path_bytes = request.scope.get("raw_path")
        raw_path = raw_path_bytes.decode("ascii") if raw_path_bytes is not None else request.url.path
        query = request.scope.get("query_string", b"")
        target = f"{origin}{raw_path}"
        if query:
            target += "?" + query.decode("ascii")
        length_header = request.headers.get("content-length")
        if length_header is not None:
            try:
                if int(length_header) < 0:
                    raise ValueError
            except ValueError:
                return _error("invalid_request", 400)
        content: bytes | AsyncIterable[bytes] | None = None
        if request.method not in {"GET", "HEAD"}:
            if length_header is None:
                buffered = bytearray()
                async for chunk in request.stream():
                    buffered.extend(chunk)
                    if len(buffered) > 1024 * 1024:
                        return _error("request_length_required", 411)
                content = bytes(buffered)
            else:
                content = request.stream()
        client = httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(None, connect=5.0),
            transport=upstream_transport,
        )
        try:
            outgoing = client.build_request(
                request.method,
                target,
                headers=_backend_headers(request, origin),
                content=content,
            )
            upstream = await client.send(outgoing, stream=True)
        except (httpx.RequestError, ValueError):
            await client.aclose()
            return _error("daemon_unavailable", 503)
        except BaseException:
            await client.aclose()
            raise

        async def close() -> None:
            await upstream.aclose()
            await client.aclose()

        async def body() -> AsyncIterator[bytes]:
            try:
                # HTTPX responses constructed with content already consumed their stream.
                if upstream.is_stream_consumed:
                    yield upstream.content
                else:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
            except httpx.RequestError:
                if upstream.headers.get("content-type", "").startswith("text/event-stream"):
                    yield b'event: error\ndata: {"error":"daemon_unavailable"}\n\n'
                else:
                    raise
            finally:
                await close()

        response = StreamingResponse(body(), status_code=upstream.status_code, background=BackgroundTask(close))
        sse = upstream.headers.get("content-type", "").startswith("text/event-stream")
        response.raw_headers = [
            (name, value)
            for name, value in upstream.headers.raw
            if name.decode("latin-1").lower() not in _HOP_HEADERS
            and not (sse and name.decode("latin-1").lower() == "content-length")
        ]
        return response

    return Starlette(
        routes=[
            Route("/healthz/live", live, methods=["GET"]),
            Route("/assets/{name}", asset, methods=["GET", "HEAD"]),
            Route("/{path:path}", proxy, methods=list(_METHODS)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Polylogue browser host")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--daemon-origin", required=True)
    arguments = parser.parse_args()
    app = create_browser_app(arguments.daemon_origin, bind_host=arguments.host, port=arguments.port)
    uvicorn.run(app, host=arguments.host, port=arguments.port, access_log=False)


if __name__ == "__main__":
    main()
