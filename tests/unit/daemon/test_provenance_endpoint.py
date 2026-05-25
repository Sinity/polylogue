"""Per-conversation provenance endpoint contracts (#1125).

The provenance read surface returns the source artifact and ingest
metadata that produced a given conversation. The raw payload preview
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
from polylogue.paths import db_path
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.sqlite.schema_ddl_archive import (
    ARCHIVE_STORAGE_DDL,
    MESSAGE_FTS_DDL,
    RAW_ARCHIVE_DDL,
    RECALL_PACKS_DDL,
    SAVED_VIEWS_DDL,
    USER_ANNOTATIONS_DDL,
    USER_MARKS_DDL,
)

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"


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


def _bootstrap_schema(dbp: Path) -> None:
    dbp.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(dbp))
    try:
        conn.executescript(RAW_ARCHIVE_DDL)
        conn.executescript(ARCHIVE_STORAGE_DDL)
        conn.executescript(MESSAGE_FTS_DDL)
        conn.executescript(USER_MARKS_DDL)
        conn.executescript(USER_ANNOTATIONS_DDL)
        conn.executescript(SAVED_VIEWS_DDL)
        conn.executescript(RECALL_PACKS_DDL)
        conn.commit()
    finally:
        conn.close()


def _seed_raw_blob(payload: bytes) -> str:
    """Write *payload* into the content-addressed blob store; return raw_id."""
    raw_id, _size = get_blob_store().write_from_bytes(payload)
    return raw_id


def _seed_conversation(
    dbp: Path,
    *,
    conversation_id: str,
    raw_id: str | None,
    source_path: str,
    blob_size: int,
    acquired_at: str = "2026-05-17T00:00:00+00:00",
    parse_error: str | None = None,
    validation_status: str | None = "passed",
    content_hash: str = "abc123" * 10,
    source_name: str = "claude-code",
) -> None:
    conn = sqlite3.connect(str(dbp))
    try:
        if raw_id is not None:
            conn.execute(
                """
                INSERT INTO raw_conversations(
                    raw_id, source_name, source_path, source_name,
                    blob_size, acquired_at, parsed_at, parse_error,
                    validated_at, validation_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    raw_id,
                    source_name,
                    source_path,
                    Path(source_path).name,
                    blob_size,
                    acquired_at,
                    None if parse_error else acquired_at,
                    parse_error,
                    None if validation_status is None else acquired_at,
                    validation_status,
                ),
            )
        conn.execute(
            """
            INSERT INTO conversations(
                conversation_id, source_name, provider_conversation_id,
                title, content_hash, version, raw_id
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                conversation_id,
                source_name,
                f"p-{conversation_id}",
                f"Title for {conversation_id}",
                content_hash,
                1,
                raw_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


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

    def test_returns_none_for_missing_conversation(self, workspace_env: dict[str, Path]) -> None:
        _bootstrap_schema(db_path())
        assert build_provenance_payload("ghost") is None

    def test_returns_metadata_envelope_without_raw_preview(self, workspace_env: dict[str, Path]) -> None:
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b'{"hello": "world"}'
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c1",
            raw_id=raw_id,
            source_path="/home/example/exports/c1.json",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload("c1")
        assert result is not None
        assert result["conversation_id"] == "c1"
        assert result["raw_id"] == raw_id
        assert result["content_hash"]
        assert result["blob_size_bytes"] == len(payload_bytes)
        assert result["source_name"] == "c1.json"
        assert result["raw_preview_included"] is False
        assert "raw_preview" not in result
        assert result["raw_preview_cap_bytes"] == RAW_PREVIEW_MAX_BYTES
        assert result["quarantined"] is False

    def test_raw_preview_is_bounded_to_server_cap(self, workspace_env: dict[str, Path]) -> None:
        """A client asking for a billion bytes still gets at most the cap."""
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b"a" * (RAW_PREVIEW_MAX_BYTES * 4)
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c-big",
            raw_id=raw_id,
            source_path="/tmp/large.bin",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload(
            "c-big",
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
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b'{"greeting": "hello"}'
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c-utf",
            raw_id=raw_id,
            source_path="/tmp/x.json",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload("c-utf", include_raw=True)
        assert result is not None
        preview = result["raw_preview"]
        assert isinstance(preview, dict)
        assert preview["encoding"] == "utf-8"
        assert preview["text"] == payload_bytes.decode("utf-8")
        assert preview["truncated"] is False

    def test_raw_preview_falls_back_to_base64_for_binary(self, workspace_env: dict[str, Path]) -> None:
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b"\xff\xfe\x00\x01\x02non-utf-8\x80"
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c-bin",
            raw_id=raw_id,
            source_path="/tmp/x.bin",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload("c-bin", include_raw=True)
        assert result is not None
        preview = result["raw_preview"]
        assert isinstance(preview, dict)
        assert preview["encoding"] == "base64"
        assert "base64" in preview
        assert "text" not in preview

    def test_quarantine_surfaces_when_parse_error(self, workspace_env: dict[str, Path]) -> None:
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b"corrupt"
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c-q",
            raw_id=raw_id,
            source_path="/tmp/x.json",
            blob_size=len(payload_bytes),
            parse_error="json: malformed input",
            validation_status=None,
        )

        result = build_provenance_payload("c-q")
        assert result is not None
        assert result["quarantined"] is True
        assert result["quarantine_reason"] == "parse_error"
        assert result["parse_error"] == "json: malformed input"

    def test_quarantine_surfaces_when_validation_failed(self, workspace_env: dict[str, Path]) -> None:
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b"{}"
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c-v",
            raw_id=raw_id,
            source_path="/tmp/x.json",
            blob_size=len(payload_bytes),
            validation_status="failed",
        )

        result = build_provenance_payload("c-v")
        assert result is not None
        assert result["quarantined"] is True
        assert result["quarantine_reason"] == "validation_failed"

    def test_quarantine_surfaces_when_no_raw_artifact(self, workspace_env: dict[str, Path]) -> None:
        dbp = db_path()
        _bootstrap_schema(dbp)
        _seed_conversation(
            dbp,
            conversation_id="c-orphan",
            raw_id=None,
            source_path="",
            blob_size=0,
            validation_status=None,
        )

        result = build_provenance_payload("c-orphan", include_raw=True)
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
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b"{}"
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c-home",
            raw_id=raw_id,
            source_path="/home/example/exports/c-home.json",
            blob_size=len(payload_bytes),
        )

        result = build_provenance_payload("c-home")
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
    """``GET /api/conversations/{id}/provenance`` HTTP contract."""

    def test_missing_conversation_returns_404(self, workspace_env: dict[str, Path]) -> None:
        _bootstrap_schema(db_path())
        handler = _make_handler("GET", "/api/conversations/ghost/provenance")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()

        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"
        send_json.assert_not_called()

    def test_default_request_omits_raw_preview(self, workspace_env: dict[str, Path]) -> None:
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b"hello"
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c1",
            raw_id=raw_id,
            source_path="/tmp/c1.json",
            blob_size=len(payload_bytes),
        )

        handler = _make_handler("GET", "/api/conversations/c1/provenance")
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
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b'{"x": 1}'
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c1",
            raw_id=raw_id,
            source_path="/tmp/c1.json",
            blob_size=len(payload_bytes),
        )

        handler = _make_handler("GET", "/api/conversations/c1/provenance?include_raw=1")
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
        dbp = db_path()
        _bootstrap_schema(dbp)
        payload_bytes = b"X" * (RAW_PREVIEW_MAX_BYTES * 8)
        raw_id = _seed_raw_blob(payload_bytes)
        _seed_conversation(
            dbp,
            conversation_id="c-big",
            raw_id=raw_id,
            source_path="/tmp/big.bin",
            blob_size=len(payload_bytes),
        )

        handler = _make_handler(
            "GET",
            "/api/conversations/c-big/provenance?include_raw=1&bytes=999999999",
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
