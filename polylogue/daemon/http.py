"""Daemon HTTP API server for the Polylogue local daemon."""

from __future__ import annotations

import asyncio
import functools
import json
import os
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from polylogue.daemon.events import emit_daemon_event
from polylogue.daemon.status import daemon_status_payload
from polylogue.errors import PolylogueError
from polylogue.logging import get_logger
from polylogue.paths import db_path

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)


def _json_bytes(payload: object) -> bytes:
    import orjson

    return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)


def daemon_safe_handler(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that discriminates PolylogueError types to HTTP status codes.

    PolylogueError subclasses carry ``http_status_code`` — use it.
    Unexpected exceptions map to 500 and are logged.
    """

    @functools.wraps(fn)
    def wrapper(self: DaemonAPIHandler, *args: object, **kwargs: object) -> None:
        try:
            fn(self, *args, **kwargs)
        except PolylogueError as exc:
            status = (
                HTTPStatus(exc.http_status_code)
                if 100 <= exc.http_status_code <= 599
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            self._send_json(status, {"ok": False, "error": type(exc).__name__, "detail": str(exc)})
        except Exception:
            logger.exception("unhandled error in %s", fn.__name__)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "internal_error"})

    return wrapper


def _get_or_create_polylogue() -> Polylogue:
    from polylogue.api import Polylogue as _Polylogue

    return _Polylogue()


class DaemonAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the daemon API server.

    Runs async archive operations via ``asyncio.run()`` in a thread pool
    worker. This is safe because each request runs in its own thread.
    """

    server: DaemonAPIHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        raw = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, status: HTTPStatus, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: HTTPStatus, code: str) -> None:
        self._send_json(status, {"ok": False, "error": code})

    def _parse_path(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        path = parsed.path.strip("/").split("/")
        params = parse_qs(parsed.query)
        return path, params

    def _get_param(self, params: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
        values = params.get(key)
        if values:
            return values[0]
        return default

    def _get_int(self, params: dict[str, list[str]], key: str, default: int = 0) -> int:
        val = self._get_param(params, key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
        return default

    # ------------------------------------------------------------------
    # Async operation runner
    # ------------------------------------------------------------------

    async def _run_archive_query(self, handler: Callable) -> object:  # type: ignore[type-arg]
        from polylogue.api import Polylogue

        async with Polylogue() as polylogue:
            return await handler(polylogue)

    def _sync_run(self, handler: Callable) -> object:  # type: ignore[type-arg]
        return asyncio.run(self._run_archive_query(handler))

    # ------------------------------------------------------------------
    # OPTIONS (CORS preflight)
    # ------------------------------------------------------------------

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT.value)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    # ------------------------------------------------------------------
    # Route dispatch
    # ------------------------------------------------------------------

    def _dispatch_get(self, path: list[str], params: dict[str, list[str]]) -> None:
        """Dispatch GET requests via route table."""
        # Static routes
        if path == [""]:
            self._serve_web_shell()
        elif path == ["api", "health"]:
            self._handle_health()
        elif path == ["api", "status"]:
            self._handle_status()
        elif path == ["api", "conversations"]:
            self._handle_list_conversations(params)
        elif path == ["api", "facets"]:
            self._handle_facets(params)
        elif path == ["api", "sources"]:
            self._handle_sources()
        elif len(path) == 3 and path[:2] == ["api", "conversations"] and path[2]:
            self._handle_get_conversation(path[2])
        elif len(path) == 5 and path[:2] == ["api", "conversations"] and path[3] == "messages":
            self._handle_get_messages(path[2], params)
        elif len(path) == 4 and path[:3] == ["api", "raw_artifacts"]:
            self._handle_get_raw_artifact(path[3])
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    def do_GET(self) -> None:
        path, params = self._parse_path()
        self._dispatch_get(path, params)

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        path, params = self._parse_path()
        if path == ["api", "reset"]:
            self._handle_reset()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

    # ------------------------------------------------------------------
    # Web shell
    # ------------------------------------------------------------------

    def _serve_web_shell(self) -> None:
        from polylogue.daemon.web_shell import WEB_SHELL_HTML

        self._send_html(HTTPStatus.OK, WEB_SHELL_HTML)

    # ------------------------------------------------------------------
    # Handlers: health
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_health(self) -> None:
        dbp = db_path()
        db_size = dbp.stat().st_size if dbp.exists() else 0
        wal_size = 0
        wal = dbp.with_suffix(".db-wal")
        if wal.exists():
            wal_size = wal.stat().st_size
        disk_free = 0
        try:
            st = os.statvfs(str(dbp.parent))
            disk_free = st.f_frsize * st.f_bavail
        except OSError:
            pass

        quick_check_ok = True
        try:
            from polylogue.config import Config
            from polylogue.paths import archive_root, render_root
            from polylogue.readiness import get_readiness

            cfg = Config(archive_root=archive_root(), render_root=render_root(), sources=[])
            report = get_readiness(cfg, deep=False, probe_only=False)
            quick_check_ok = report.counts().ok > 0
        except Exception:
            quick_check_ok = False

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": quick_check_ok,
                "db_size_bytes": db_size,
                "wal_size_bytes": wal_size,
                "disk_free_bytes": disk_free,
                "blob_dir_size_bytes": 0,
                "quick_check": "pass" if quick_check_ok else "error",
                "quick_check_age_s": None,
            },
        )

    # ------------------------------------------------------------------
    # Handlers: status
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_status(self) -> None:
        status = daemon_status_payload()
        self._send_json(HTTPStatus.OK, status)

    # ------------------------------------------------------------------
    # Handlers: list conversations
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_list_conversations(self, params: dict[str, list[str]]) -> None:
        limit = self._get_int(params, "limit", 50)
        provider = self._get_param(params, "provider")
        since = self._get_param(params, "since")

        def _list(poly: Polylogue) -> object:
            return asyncio.run(self._do_list(poly, limit, provider, since))

        result = self._sync_run(_list)
        self._send_json(HTTPStatus.OK, result)

    async def _do_list(self, poly: Polylogue, limit: int, provider: str | None, since: str | None) -> object:
        summaries = await poly.list_conversations(provider=provider, limit=limit)
        hits: list[dict[str, object]] = []
        for summary in summaries:
            hits.append(
                {
                    "id": str(summary.id),
                    "title": summary.display_title,
                    "provider": str(summary.provider) if summary.provider else None,
                    "created_at": summary.created_at.isoformat() if summary.created_at else None,
                    "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
                    "message_count": getattr(summary, "message_count", None),
                }
            )
        return {"hits": hits, "total": len(hits)}

    # ------------------------------------------------------------------
    # Handlers: get conversation
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_conversation(self, conv_id: str) -> None:
        def _get(poly: Polylogue) -> object:
            return asyncio.run(self._do_get_conversation(poly, conv_id))

        result = self._sync_run(_get)
        if result is None:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_conversation(self, poly: Polylogue, conv_id: str) -> object:
        conv = await poly.get_conversation(conv_id)
        if conv is None:
            return None
        return {
            "id": str(conv.id),
            "title": conv.title,
            "provider": str(conv.provider) if conv.provider else None,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "messages": [
                {
                    "id": str(msg.id),
                    "role": str(msg.role),
                    "text": msg.text,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                }
                for msg in conv.messages
            ],
            "total": len(conv.messages),
        }

    # ------------------------------------------------------------------
    # Handlers: get messages
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_messages(self, conv_id: str, params: dict[str, list[str]]) -> None:
        limit = self._get_int(params, "limit", 50)
        offset = self._get_int(params, "offset", 0)

        def _get(poly: Polylogue) -> object:
            return asyncio.run(self._do_get_messages(poly, conv_id, limit, offset))

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_messages(self, poly: Polylogue, conv_id: str, limit: int, offset: int) -> object:
        messages, total = await poly.get_messages_paginated(conv_id, limit=limit, offset=offset)
        return {
            "messages": [
                {
                    "id": str(msg.id),
                    "role": str(msg.role),
                    "text": msg.text,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                }
                for msg in messages
            ],
            "total": total,
        }

    # ------------------------------------------------------------------
    # Handlers: get raw artifact
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_get_raw_artifact(self, artifact_id: str) -> None:
        def _get(poly: Polylogue) -> object:
            return asyncio.run(self._do_get_raw_artifacts(poly, artifact_id))

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_get_raw_artifacts(self, poly: Polylogue, artifact_id: str) -> object:
        raw_items = await poly.get_raw_artifacts_for_conversation(artifact_id)
        return {"raw_artifacts": raw_items}

    # ------------------------------------------------------------------
    # Handlers: facets
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_facets(self, params: dict[str, list[str]]) -> None:
        def _get(poly: Polylogue) -> object:
            return asyncio.run(self._do_facets(poly))

        result = self._sync_run(_get)
        self._send_json(HTTPStatus.OK, result)

    async def _do_facets(self, poly: Polylogue) -> object:
        stats = await poly.stats()
        return {
            "providers": stats.providers,
            "total_conversations": stats.conversation_count,
            "total_messages": stats.message_count,
        }

    # ------------------------------------------------------------------
    # Handlers: sources
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_sources(self) -> None:
        from polylogue.sources.live.watcher import default_sources

        sources = default_sources()
        self._send_json(
            HTTPStatus.OK,
            {"sources": [{"name": s.name, "root": str(s.root), "exists": s.exists()} for s in sources]},
        )

    # ------------------------------------------------------------------
    # Handlers: reset
    # ------------------------------------------------------------------

    @daemon_safe_handler
    def _handle_reset(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        body_text = body_raw.decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        scope = body.get("scope", "all")
        conv_id = body.get("conversation_id")

        if scope == "conversation" and not conv_id:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return

        op_id = f"reset-{scope}-{conv_id[:16] if conv_id else 'all'}"

        def _do_reset() -> dict[str, object]:
            poly = _get_or_create_polylogue()
            if scope == "conversation" and conv_id:
                deleted = poly.delete_conversation(conv_id)
                return {"deleted": deleted, "conversation_id": conv_id}
            return {"ok": True}

        result = self._sync_run(lambda p: _do_reset())

        emit_daemon_event("reset", operation_id=op_id, payload=result if isinstance(result, dict) else None)
        self._send_json(HTTPStatus.OK, result)


class DaemonAPIHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server for the daemon API."""

    allow_reuse_address = True
    daemon_threads = True
