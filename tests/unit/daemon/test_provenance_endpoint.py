"""Per-session provenance endpoint contracts (#1125).

The provenance read surface returns the source artifact and ingest
metadata that produced a given session. The raw payload preview
within the same envelope is opt-in and bounded server-side regardless
of what a client requests.

Tests use the in-process handler pattern from
``tests/unit/daemon/test_daemon_http_contracts.py`` — no real daemon,
no socket listener — so the endpoint can be exercised against a freshly
seeded SQLite archive and a real blob store rooted at the temporary
workspace.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon.provenance import (
    RAW_PREVIEW_MAX_BYTES,
    _display_source_path,
    build_provenance_payload,
)
from polylogue.paths import active_index_db_path
from polylogue.storage.blob_store import get_blob_store

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)
    archive_query_admission = threading.BoundedSemaphore(64)  # generous: not under test


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str, *, body: bytes = b"") -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": str(len(body))}
    handler.headers = cast("Message[str, str]", _MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


def _index_db() -> Path:
    """Native index.db path the provenance reader resolves to."""
    return active_index_db_path()


def _seed_raw_blob(payload: bytes) -> str:
    """Write *payload* into the content-addressed blob store; return blob hash."""
    raw_id, _size = get_blob_store().write_from_bytes(payload)
    return raw_id


def _native_session_id(origin: str, native_id: str) -> str:
    from polylogue.core.identity_law import session_id as archive_session_id

    return archive_session_id(origin, native_id)


def _session_parts(session_id: str, origin: str) -> tuple[str, str]:
    prefix = f"{origin}:"
    native_id = session_id[len(prefix) :] if session_id.startswith(prefix) else session_id
    return native_id, _native_session_id(origin, native_id)


def _seed_archive_provenance(
    *,
    session_id: str,
    origin: str = "claude-code-session",
    raw_id: str | None,
    raw_blob_id: str | None = None,
    source_path: str,
    blob_size: int | None,
    acquired_at_ms: int = 1_767_225_600_000,
    file_mtime_ms: int | None = 1_767_225_601_000,
    parsed_at_ms: int | None = 1_767_225_602_000,
    parse_error: str | None = None,
    validated_at_ms: int | None = 1_767_225_603_000,
    validation_status: str | None = "passed",
    validation_error: str | None = None,
    content_hash: bytes = b"x" * 32,
    write_source_tier: bool = True,
) -> str:
    """Seed a index.db session row + source.db raw_sessions row.

    Mirrors the archive source/index tiers the provenance reader joins
    (``polylogue/daemon/provenance.py:_fetch_archive_provenance_row``).
    """
    archive_db = _index_db()
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    native_id, archive_session_id = _session_parts(session_id, origin)
    # The blob store keys raw blobs by their SHA-256 hash; ``raw_id`` from
    # ``_seed_raw_blob`` IS that hash, so default the source-tier blob_hash
    # reference to it when not given explicitly.
    if raw_blob_id is None and raw_id is not None:
        raw_blob_id = raw_id
    if raw_id is not None and write_source_tier:
        with sqlite3.connect(archive_db.with_name("source.db")) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_sessions (
                    raw_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    blob_hash BLOB NOT NULL,
                    blob_size INTEGER NOT NULL,
                    acquired_at_ms INTEGER NOT NULL,
                    file_mtime_ms INTEGER,
                    parsed_at_ms INTEGER,
                    parse_error TEXT,
                    validated_at_ms INTEGER,
                    validation_status TEXT,
                    validation_error TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, source_path, blob_hash, blob_size,
                    acquired_at_ms, file_mtime_ms, parsed_at_ms, parse_error,
                    validated_at_ms, validation_status, validation_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    raw_id,
                    origin,
                    source_path,
                    bytes.fromhex(raw_blob_id) if raw_blob_id else b"",
                    blob_size if blob_size is not None else 0,
                    acquired_at_ms,
                    file_mtime_ms,
                    None if parse_error else parsed_at_ms,
                    parse_error,
                    None if validation_status is None else validated_at_ms,
                    validation_status,
                    validation_error,
                ),
            )
            conn.commit()
    with sqlite3.connect(archive_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, content_hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(origin, native_id) DO UPDATE SET
                raw_id = excluded.raw_id,
                content_hash = excluded.content_hash
            """,
            (native_id, origin, raw_id, content_hash),
        )
        conn.commit()
    return archive_session_id


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_display_source_path_returns_none_for_falsy() -> None:
    assert _display_source_path(None) is None
    assert _display_source_path("") is None


def test_display_source_path_replaces_home_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/alice")
    assert _display_source_path("/home/alice/.claude/projects/x.jsonl") == "~/.claude/projects/x.jsonl"


def test_display_source_path_preserves_non_home_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/alice")
    assert _display_source_path("/var/data/exports/file.json") == "/var/data/exports/file.json"


# ---------------------------------------------------------------------------
# Provenance payload contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestProvenancePayload:
    """``build_provenance_payload`` behavior independent of HTTP routing."""

    def test_returns_none_for_missing_session(self, workspace_env: dict[str, Path]) -> None:
        assert build_provenance_payload("ghost") is None

    def test_returns_metadata_envelope_without_raw_preview(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b'{"hello": "world"}'
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c1",
            raw_id=raw_id,
            source_path="/home/example/exports/c1.json",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload(session_id)
        assert result is not None
        assert result["session_id"] == session_id
        assert result["raw_id"] == raw_id
        assert result["content_hash"]
        assert result["blob_size_bytes"] == len(payload_bytes)
        assert result["origin"] == "claude-code-session"
        assert result["raw_preview_included"] is False
        assert "raw_preview" not in result
        assert result["raw_preview_cap_bytes"] == RAW_PREVIEW_MAX_BYTES
        assert result["quarantined"] is False

    def test_reads_archive_file_set_from_archive_tiers(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b'{"archive": "current"}'
        raw_blob_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="codex-session:v1",
            origin="codex-session",
            raw_id="raw-v1",
            raw_blob_id=raw_blob_id,
            source_path="/home/example/.codex/sessions/v1.jsonl",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload(session_id, include_raw=True)

        assert result is not None
        assert result["session_id"] == "codex-session:v1"
        assert result["origin"] == "codex-session"
        assert result["raw_id"] == "raw-v1"
        assert result["origin"] == "codex-session"
        assert result["blob_size_bytes"] == len(payload_bytes)
        assert result["validation_status"] == "passed"
        assert result["quarantined"] is False
        assert result["raw_preview_included"] is True
        raw_preview = result["raw_preview"]
        assert isinstance(raw_preview, dict)
        assert raw_preview["available"] is True
        assert raw_preview["text"] == payload_bytes.decode()

    def test_archive_session_survives_missing_source_tier(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_archive_provenance(
            session_id="codex-session:missing-source",
            origin="codex-session",
            raw_id="raw-missing",
            raw_blob_id=None,
            source_path="",
            blob_size=None,
            write_source_tier=False,
        )

        result = build_provenance_payload(session_id, include_raw=True)

        assert result is not None
        assert result["session_id"] == "codex-session:missing-source"
        assert result["origin"] == "codex-session"
        assert result["raw_id"] == "raw-missing"
        assert result["origin"] == "codex-session"
        assert result["source_path_display"] is None
        assert result["blob_size_bytes"] is None
        assert result["quarantined"] is False
        assert result["raw_preview_included"] is True
        raw_preview = result["raw_preview"]
        assert isinstance(raw_preview, dict)
        assert raw_preview["available"] is False
        assert raw_preview["reason"] == "no_raw_artifact"

    def test_raw_preview_is_bounded_to_server_cap(self, workspace_env: dict[str, Path]) -> None:
        """A client asking for a billion bytes still gets at most the cap."""
        payload_bytes = b"a" * (RAW_PREVIEW_MAX_BYTES * 4)
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c-big",
            raw_id=raw_id,
            source_path="/tmp/large.bin",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload(
            session_id,
            include_raw=True,
            requested_bytes=10**9,
        )
        assert result is not None
        preview = result["raw_preview"]
        assert isinstance(preview, dict)
        assert preview["available"] is True
        assert preview["max_bytes"] == RAW_PREVIEW_MAX_BYTES
        assert preview["bytes_returned"] <= RAW_PREVIEW_MAX_BYTES
        assert preview["truncated"] is True

    def test_raw_preview_decodes_utf8_when_possible(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b'{"greeting": "hello"}'
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c-utf",
            raw_id=raw_id,
            source_path="/tmp/x.json",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload(session_id, include_raw=True)
        assert result is not None
        preview = result["raw_preview"]
        assert isinstance(preview, dict)
        assert preview["encoding"] == "utf-8"
        assert preview["text"] == payload_bytes.decode("utf-8")
        assert preview["truncated"] is False

    def test_raw_preview_falls_back_to_base64_for_binary(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b"\xff\xfe\x00\x01\x02non-utf-8\x80"
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c-bin",
            raw_id=raw_id,
            source_path="/tmp/x.bin",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload(session_id, include_raw=True)
        assert result is not None
        preview = result["raw_preview"]
        assert isinstance(preview, dict)
        assert preview["encoding"] == "base64"
        assert "base64" in preview
        assert "text" not in preview

    def test_quarantine_surfaces_when_parse_error(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b"corrupt"
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c-q",
            raw_id=raw_id,
            source_path="/tmp/x.json",
            blob_size=len(payload_bytes),
            parse_error="json: malformed input",
            validation_status=None,
        )

        result = build_provenance_payload(session_id)
        assert result is not None
        assert result["quarantined"] is True
        assert result["quarantine_reason"] == "parse_error"
        assert result["parse_error"] == "json: malformed input"

    def test_quarantine_surfaces_when_validation_failed(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b"{}"
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c-v",
            raw_id=raw_id,
            source_path="/tmp/x.json",
            blob_size=len(payload_bytes),
            validation_status="failed",
        )

        result = build_provenance_payload(session_id)
        assert result is not None
        assert result["quarantined"] is True
        assert result["quarantine_reason"] == "validation_failed"

    def test_quarantine_surfaces_when_no_raw_artifact(self, workspace_env: dict[str, Path]) -> None:
        session_id = _seed_archive_provenance(
            session_id="c-orphan",
            raw_id=None,
            source_path="",
            blob_size=0,
            validation_status=None,
        )

        result = build_provenance_payload(session_id, include_raw=True)
        assert result is not None
        assert result["quarantined"] is True
        assert result["quarantine_reason"] == "no_raw_artifact"
        assert result["raw_id"] is None
        preview = result["raw_preview"]
        assert isinstance(preview, dict)
        assert preview["available"] is False
        assert preview["reason"] == "no_raw_artifact"

    def test_source_path_display_sanitizes_home(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", "/home/example")
        payload_bytes = b"{}"
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c-home",
            raw_id=raw_id,
            source_path="/home/example/exports/c-home.json",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload(session_id)
        assert result is not None
        assert result["source_path_display"] == "~/exports/c-home.json"
        assert result["source_path_contains_home"] is True
        # The raw home-prefixed path must not appear anywhere in the
        # serialized payload.
        assert "/home/example/exports/c-home.json" not in json.dumps(result)


# ---------------------------------------------------------------------------
# HTTP endpoint contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestProvenanceEndpoint:
    """``GET /api/sessions/{id}/provenance`` HTTP contract."""

    def test_missing_session_returns_404(self, workspace_env: dict[str, Path]) -> None:
        handler = _make_handler("GET", "/api/sessions/ghost/provenance")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()

        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"
        send_json.assert_not_called()

    def test_default_request_omits_raw_preview(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b"hello"
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c1",
            raw_id=raw_id,
            source_path="/tmp/c1.json",
            blob_size=len(payload_bytes),
        )

        handler = _make_handler("GET", f"/api/sessions/{session_id}/provenance")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert isinstance(payload, dict)
        assert payload["raw_preview_included"] is False
        assert "raw_preview" not in payload
        # No raw bytes from the source artifact appear in the response.
        assert "hello" not in json.dumps(payload)

    def test_include_raw_query_param_attaches_preview(self, workspace_env: dict[str, Path]) -> None:
        payload_bytes = b'{"x": 1}'
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c1",
            raw_id=raw_id,
            source_path="/tmp/c1.json",
            blob_size=len(payload_bytes),
        )

        handler = _make_handler("GET", f"/api/sessions/{session_id}/provenance?include_raw=1")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        send_json.assert_called_once()
        _, payload = send_json.call_args.args
        assert payload["raw_preview_included"] is True
        preview = payload["raw_preview"]
        assert preview["available"] is True
        assert preview["text"] == '{"x": 1}'

    def test_client_cannot_widen_raw_preview_cap(self, workspace_env: dict[str, Path]) -> None:
        """A ``bytes=`` parameter larger than the cap is clamped server-side.

        This is the security-bounded-payload contract: a malicious or
        confused client must not be able to exfiltrate the full blob by
        asking for billions of bytes.
        """
        payload_bytes = b"X" * (RAW_PREVIEW_MAX_BYTES * 8)
        raw_id = _seed_raw_blob(payload_bytes)
        session_id = _seed_archive_provenance(
            session_id="c-big",
            raw_id=raw_id,
            source_path="/tmp/big.bin",
            blob_size=len(payload_bytes),
        )

        handler = _make_handler(
            "GET",
            f"/api/sessions/{session_id}/provenance?include_raw=1&bytes=999999999",
        )
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        preview = payload["raw_preview"]
        assert preview["max_bytes"] == RAW_PREVIEW_MAX_BYTES
        assert preview["bytes_returned"] <= RAW_PREVIEW_MAX_BYTES
        # Sanity: the full payload size is reported so the operator can
        # see how much was withheld.
        assert preview["total_size_bytes"] == len(payload_bytes)
        assert preview["truncated"] is True
