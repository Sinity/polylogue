"""Daemon HTTP API server for the Polylogue local daemon."""

from __future__ import annotations

import asyncio
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from polylogue.daemon.events import emit_daemon_event
from polylogue.daemon.status import daemon_status_payload
from polylogue.logging import get_logger
from polylogue.paths import db_path

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)


def _get_or_create_polylogue() -> Polylogue:
    from polylogue.api import Polylogue as _Polylogue

    return _Polylogue()


def _json_bytes(payload: object) -> bytes:
    import orjson

    return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)


_SAFE_ERROR_CODES: dict[str, str] = {
    "not_found": "not_found",
    "method_not_allowed": "method_not_allowed",
    "invalid_request": "invalid_request",
    "internal_error": "internal_error",
    "not_implemented": "not_implemented",
}


class DaemonAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the daemon API server.

    Runs async archive operations via ``asyncio.run()`` in a thread pool
    worker. This is safe because each request runs in its own thread.
    """

    server: DaemonAPIHTTPServer  # type: ignore[assignment]

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

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        safe = _SAFE_ERROR_CODES.get(message, "internal_error")
        self._send_json(status, {"ok": False, "error": safe})

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

    async def _run_archive_query(self, handler) -> object:  # type: ignore[no-untyped-def]
        """Run an async operation against the archive and return the result."""
        from polylogue.api import Polylogue

        async with Polylogue() as polylogue:
            return await handler(polylogue)

    def _sync_run(self, handler) -> object:  # type: ignore[no-untyped-def]
        """Run an async archive query synchronously in the current thread."""
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
    # GET
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        path, params = self._parse_path()

        if path == [""]:
            self._serve_web_shell()
            return

        if path == ["api", "health"]:
            self._handle_health()
            return

        if path == ["api", "status"]:
            self._handle_status()
            return

        if path == ["api", "conversations"]:
            self._handle_list_conversations(params)
            return

        if len(path) == 3 and path[:2] == ["api", "conversations"] and path[2] and len(path) == 3:
            conv_id = path[2]
            self._handle_get_conversation(conv_id)
            return

        if len(path) == 5 and path[:2] == ["api", "conversations"] and path[3] == "messages":
            conv_id = path[2]
            self._handle_get_messages(conv_id, params)
            return

        if len(path) == 4 and path[:3] == ["api", "raw_artifacts"]:
            artifact_id = path[3]
            self._handle_get_raw_artifact(artifact_id)
            return

        if path == ["api", "facets"]:
            self._handle_facets(params)
            return

        if path == ["api", "sources"]:
            self._handle_sources()
            return

        self._send_error(HTTPStatus.NOT_FOUND, "not_found")

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
        """Serve the single-page web shell at GET /."""
        from polylogue.daemon.web_shell import WEB_SHELL_HTML

        self._send_html(HTTPStatus.OK, WEB_SHELL_HTML)

    # ------------------------------------------------------------------
    # Handlers: health
    # ------------------------------------------------------------------

    def _handle_health(self) -> None:
        try:
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

            # Quick check via readiness
            quick_check_ok = True
            quick_check_detail = ""
            try:
                report = self._sync_run(lambda p: p.health_check())
                errors = [c for c in report.checks if c.status.value == "error"]
                quick_check_ok = len(errors) == 0
                quick_check_detail = "ok" if quick_check_ok else f"{len(errors)} check(s) in error state"
            except Exception as exc:
                quick_check_ok = False
                quick_check_detail = str(exc)

            payload = {
                "ok": quick_check_ok,
                "db_size_bytes": db_size,
                "wal_size_bytes": wal_size,
                "disk_free_bytes": disk_free,
                "quick_check": quick_check_detail,
            }
            status = HTTPStatus.OK if quick_check_ok else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(status, payload)

        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")
            self._log_error(f"health check failed: {exc}")

    # ------------------------------------------------------------------
    # Handlers: status
    # ------------------------------------------------------------------

    def _handle_status(self) -> None:
        try:
            payload = daemon_status_payload()
            # Add runtime fields
            dbp = db_path()
            if dbp.exists():
                payload["db_path"] = str(dbp)
                payload["db_size_bytes"] = dbp.stat().st_size
            payload["browser_capture_active"] = True  # receiver is running if this daemon is up
            self._send_json(HTTPStatus.OK, payload)
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Handlers: list conversations
    # ------------------------------------------------------------------

    def _handle_list_conversations(self, params: dict[str, list[str]]) -> None:
        try:
            provider = self._get_param(params, "provider")
            tag = self._get_param(params, "tag")
            since = self._get_param(params, "since")
            query = self._get_param(params, "query")
            limit = self._get_int(params, "limit", 50)
            offset = self._get_int(params, "offset", 0)

            if query:
                result = self._sync_run(lambda p: p.search(query, limit=limit, since=since))
                conversations = [
                    {
                        "id": str(h.conversation_id or getattr(h, "id", "")),
                        "title": h.title or "",
                        "provider": h.provider,
                        "created_at": h.created_at,
                        "updated_at": getattr(h, "updated_at", None),
                        "message_count": getattr(h, "message_count", 0) or 0,
                        "word_count": getattr(h, "word_count", 0) or 0,
                        "snippet": getattr(h, "snippet", None),
                    }
                    for h in (result.hits if result else [])
                ]
            else:
                conversations = self._sync_run(
                    lambda p: p.query_conversations(
                        provider=provider,
                        tag=tag,
                        since=since,
                        limit=limit,
                        offset=offset,
                    )
                )

            self._send_json(HTTPStatus.OK, {"ok": True, "conversations": conversations})
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Handlers: get conversation
    # ------------------------------------------------------------------

    def _handle_get_conversation(self, conv_id: str) -> None:
        try:
            conversation = self._sync_run(lambda p: p.get_conversation(conv_id))
            if conversation is None:
                self._send_error(HTTPStatus.NOT_FOUND, "not_found")
                return

            payload = {
                "ok": True,
                "conversation": {
                    "id": str(getattr(conversation, "id", conv_id)),
                    "title": getattr(conversation, "title", ""),
                    "provider": getattr(conversation, "provider", ""),
                    "created_at": getattr(conversation, "created_at", None),
                    "updated_at": getattr(conversation, "updated_at", None),
                    "message_count": getattr(conversation, "message_count", 0) or 0,
                    "word_count": getattr(conversation, "word_count", 0) or 0,
                },
            }
            self._send_json(HTTPStatus.OK, payload)
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Handlers: get messages
    # ------------------------------------------------------------------

    def _handle_get_messages(self, conv_id: str, params: dict[str, list[str]]) -> None:
        try:
            limit = self._get_int(params, "limit", 50)
            offset = self._get_int(params, "offset", 0)

            messages, total = self._sync_run(lambda p: p.get_messages_paginated(conv_id, limit=limit, offset=offset))

            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "messages": [
                        {
                            "id": m.id,
                            "role": str(m.role),
                            "text": m.text or "",
                            "message_type": m.message_type.value,
                        }
                        for m in messages
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                },
            )
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Handlers: get raw artifact
    # ------------------------------------------------------------------

    def _handle_get_raw_artifact(self, artifact_id: str) -> None:
        try:
            raw_artifacts, total = self._sync_run(lambda p: p.get_raw_artifacts_for_conversation(artifact_id, limit=1))
            if not raw_artifacts:
                self._send_error(HTTPStatus.NOT_FOUND, "not_found")
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "artifact": raw_artifacts[0]})
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Handlers: facets
    # ------------------------------------------------------------------

    def _handle_facets(self, params: dict[str, list[str]]) -> None:
        try:
            stats = self._sync_run(lambda p: p.stats())
            facets = {
                "total_conversations": getattr(stats, "conversation_count", 0),
                "total_messages": getattr(stats, "message_count", 0),
                "total_words": getattr(stats, "word_count", 0),
                "providers": dict(getattr(stats, "providers", {})),
                "tags": dict(getattr(stats, "tags", {})),
            }
            self._send_json(HTTPStatus.OK, {"ok": True, "facets": facets})
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Handlers: sources
    # ------------------------------------------------------------------

    def _handle_sources(self) -> None:
        try:
            from polylogue.daemon.status import live_source_status_payload
            from polylogue.sources.live.watcher import default_sources

            sources = default_sources()
            payload = live_source_status_payload(sources)
            self._send_json(HTTPStatus.OK, {"ok": True, "sources": payload})
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Handlers: reset
    # ------------------------------------------------------------------

    def _handle_reset(self) -> None:
        """Reset archive data, then trigger daemon re-convergence."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = {}
            if length > 0:
                body = json.loads(self.rfile.read(length))

            scope = body.get("scope", "")
            conv_id = body.get("id", "")

            if not scope and not conv_id:
                self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
                return

            op_id = f"reset-{scope}-{conv_id[:16] if conv_id else 'all'}"

            # Actually perform the reset via the Polylogue facade
            def _do_reset() -> dict:
                poly = _get_or_create_polylogue()
                if scope == "conversation" and conv_id:
                    deleted = poly.delete_conversation(conv_id)
                    return {"deleted": deleted, "conversation_id": conv_id}
                elif scope == "source":
                    from polylogue.cli.commands.reset import _tombstone_conversations

                    poly._ensure_archive()
                    _tombstone_conversations(poly.repository, source_path=conv_id)
                    return {"tombstoned": True, "source": conv_id}
                return {"ok": True}

            result = self._sync_run(lambda p: _do_reset())

            emit_daemon_event("reset", operation_id=op_id, payload={"scope": scope, "id": conv_id, "result": result})

            # Trigger re-convergence: the daemon watcher will detect missing
            # derived state on next scan and re-ingest from raw blobs.
            self._send_json(
                HTTPStatus.OK, {"ok": True, "reset": {"scope": scope, "id": conv_id, "convergence": "scheduled"}}
            )
        except (json.JSONDecodeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        except Exception:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_error(self, msg: str) -> None:
        """Log to stderr without exposing paths in API responses."""
        import sys

        print(f"[polylogued] {msg}", file=sys.stderr)


class DaemonAPIHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server for the daemon API."""

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, DaemonAPIHandler)


def make_daemon_api_server(host: str, port: int) -> DaemonAPIHTTPServer:
    """Create a configured daemon API server."""
    return DaemonAPIHTTPServer((host, port))


__all__ = ["DaemonAPIHandler", "DaemonAPIHTTPServer", "make_daemon_api_server"]
