"""Reader smoke tests — page structure, API contracts, degraded states (#865)."""

from __future__ import annotations

import json
from http.server import HTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def _start_server(tmp_path: Path, seeded: bool = True) -> tuple[HTTPServer, str]:
    """Start a DaemonAPIHTTPServer on a random port with optional seeded data."""
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    if seeded:
        _seed_test_db(tmp_path)

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"


def _seed_test_db(tmp_path: Path) -> None:
    """Seed a test archive with 3 conversations."""
    import sqlite3

    from polylogue.storage.sqlite.schema_ddl_archive import ARCHIVE_STORAGE_DDL

    db = tmp_path / "polylogue.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(ARCHIVE_STORAGE_DDL)
    for cid, prov in [("c1", "claude-code"), ("c2", "chatgpt"), ("c3", "claude-ai")]:
        conn.execute(
            "INSERT INTO conversations(conversation_id, provider_name, provider_conversation_id, title) VALUES(?,?,?,?)",
            (cid, prov, f"p-{cid}", f"Test {prov}"),
        )
        conn.execute(
            "INSERT INTO messages(message_id, conversation_id, role, text, provider_name) VALUES(?,?,?,?,?)",
            (f"m-{cid}", cid, "user", "Hello", prov),
        )
    conn.commit()
    conn.close()


def test_web_shell_returns_html(tmp_path: Path) -> None:
    """GET / returns text/html with expected DOM structure."""
    server, base_url = _start_server(tmp_path)
    try:
        req = Request(f"{base_url}/")
        with urlopen(req) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            assert "text/html" in content_type
            body = resp.read().decode()
            assert "<!DOCTYPE html>" in body or "<html" in body
            assert len(body) > 100
    finally:
        server.shutdown()


def test_api_status_returns_daemon_fields(tmp_path: Path) -> None:
    """GET /api/status returns daemon_liveness and component_state."""
    server, base_url = _start_server(tmp_path)
    try:
        req = Request(f"{base_url}/api/status")
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "daemon_liveness" in data
            assert "component_state" in data
    finally:
        server.shutdown()


def test_api_conversations_returns_paginated_envelope(tmp_path: Path) -> None:
    """GET /api/conversations returns items, total, limit, offset."""
    server, base_url = _start_server(tmp_path)
    try:
        req = Request(f"{base_url}/api/conversations")
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "items" in data
            assert "total" in data
            assert data["total"] == 3
    finally:
        server.shutdown()


def test_api_facets_returns_scoped_flag(tmp_path: Path) -> None:
    """GET /api/facets returns scoped_to_query, providers, tags."""
    server, base_url = _start_server(tmp_path)
    try:
        req = Request(f"{base_url}/api/facets")
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "scoped_to_query" in data
            assert "providers" in data
    finally:
        server.shutdown()


def test_empty_archive_returns_zero(tmp_path: Path) -> None:
    """Empty archive returns items=[] with total=0."""
    server, base_url = _start_server(tmp_path, seeded=False)
    try:
        req = Request(f"{base_url}/api/conversations")
        with urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["total"] == 0
            assert data["items"] == []
    finally:
        server.shutdown()
