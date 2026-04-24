from __future__ import annotations

import json
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import cast

from click.testing import CliRunner

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    capture_artifact_path,
    existing_capture_state,
    write_capture_envelope,
)
from polylogue.browser_capture.server import make_server
from polylogue.cli.click_app import cli


def _payload(provider: str = "chatgpt", session_id: str = "conv-123") -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-123",
            "page_title": "ChatGPT - Work plan",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
        },
        "session": {
            "provider": provider,
            "provider_session_id": session_id,
            "title": "Work plan",
            "turns": [{"provider_turn_id": "u1", "role": "user", "text": "Draft"}],
        },
    }


def test_capture_artifact_path_is_deterministic(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload(session_id="c/with spaces"))

    first = capture_artifact_path(envelope, tmp_path)
    second = capture_artifact_path(envelope, tmp_path)

    assert first == second
    assert first.parent.name == "chatgpt"
    assert "with-spaces" in first.name


def test_write_capture_envelope_replaces_same_artifact(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())

    first = write_capture_envelope(envelope, inbox_path=tmp_path)
    second = write_capture_envelope(envelope, inbox_path=tmp_path)

    assert first.path == second.path
    assert first.replaced is False
    assert second.replaced is True
    assert json.loads(first.path.read_text(encoding="utf-8"))["session"]["provider_session_id"] == "conv-123"


def test_existing_capture_state_reports_written_artifact(tmp_path: Path) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    write_capture_envelope(envelope, inbox_path=tmp_path)

    state = existing_capture_state("chatgpt", "conv-123", inbox_path=tmp_path)

    assert state["captured"] is True
    assert state["provider"] == "chatgpt"


def test_browser_capture_status_cli_json(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["browser-capture", "status", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["receiver"] == "polylogue-browser-capture"


def test_receiver_http_accepts_capture_and_rejects_unknown_origin(tmp_path: Path) -> None:
    server = make_server("127.0.0.1", 0, inbox_path=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])

    try:
        conn = HTTPConnection(host, port)
        conn.request("GET", "/v1/status", headers={"Origin": "https://evil.example"})
        response = conn.getresponse()
        assert response.status == HTTPStatus.FORBIDDEN
        response.read()
        conn.close()

        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/browser-captures",
            body=json.dumps(_payload()),
            headers={"Content-Type": "application/json", "Origin": "https://chatgpt.com"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        assert response.status == HTTPStatus.ACCEPTED
        assert body["ok"] is True
        assert Path(body["artifact_path"]).exists()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
