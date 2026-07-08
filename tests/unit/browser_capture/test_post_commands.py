"""Tests for the browser-capture outbound post-command queue and routes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from pathlib import Path
from threading import Thread
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.models import (
    BrowserPostCommandAckRequest,
    BrowserPostCommandRequest,
    BrowserPostTarget,
)
from polylogue.browser_capture.receiver import (
    BROWSER_POST_ENABLED_ENV,
    BrowserPostCommandConflictError,
    BrowserPostCommandStateError,
    BrowserPostDisabledError,
    ack_post_command,
    browser_post_enabled,
    enqueue_post_command,
    poll_post_commands,
    post_command_queue_root,
)
from polylogue.browser_capture.route_contracts import (
    BROWSER_CAPTURE_ROUTE_CONTRACTS,
    browser_capture_route_contract_for,
)
from polylogue.browser_capture.server import make_server
from polylogue.daemon.cli import main as daemon_cli

_EXTENSION_ORIGIN = "chrome-extension://polylogue-test"


@pytest.fixture
def post_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BROWSER_POST_ENABLED_ENV, "1")


@pytest.fixture
def post_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BROWSER_POST_ENABLED_ENV, raising=False)


def _request_obj(provider: str = "chatgpt", conversation_id: str = "conv-1") -> BrowserPostCommandRequest:
    return BrowserPostCommandRequest(
        provider=provider,  # type: ignore[arg-type]
        target=BrowserPostTarget(conversation_id=conversation_id),
        text="Please research X.",
    )


# ---- queue-level lifecycle ------------------------------------------------


def test_enqueue_requires_env_flag(tmp_path: Path, post_disabled: None) -> None:
    assert browser_post_enabled() is False
    with pytest.raises(BrowserPostDisabledError):
        enqueue_post_command(_request_obj(), spool_path=tmp_path)
    assert not post_command_queue_root(tmp_path).exists()


def test_enqueue_poll_ack_lifecycle(tmp_path: Path, post_enabled: None) -> None:
    command = enqueue_post_command(_request_obj(), spool_path=tmp_path)
    assert command.status == "pending"
    assert command.submit is False

    # Poll claims the command: pending -> dispatched.
    polled = poll_post_commands(provider="chatgpt", spool_path=tmp_path)
    assert [c.command_id for c in polled] == [command.command_id]
    assert polled[0].status == "dispatched"

    # A second poll does not re-dispatch a claimed command.
    assert poll_post_commands(provider="chatgpt", spool_path=tmp_path) == []

    acked = ack_post_command(
        command.command_id,
        BrowserPostCommandAckRequest(
            status="submitted", detail="dry_run_filled_not_sent", observed_url="https://chatgpt.com/c/conv-1"
        ),
        spool_path=tmp_path,
    )
    assert acked is not None
    assert acked.status == "submitted"
    assert acked.observed_url == "https://chatgpt.com/c/conv-1"


def test_poll_returns_empty_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BROWSER_POST_ENABLED_ENV, "1")
    enqueue_post_command(_request_obj(), spool_path=tmp_path)
    # Even with a queued command on disk, disabling posting hides it.
    monkeypatch.delenv(BROWSER_POST_ENABLED_ENV, raising=False)
    assert poll_post_commands(spool_path=tmp_path) == []


def test_poll_filters_by_provider(tmp_path: Path, post_enabled: None) -> None:
    enqueue_post_command(_request_obj(provider="chatgpt"), spool_path=tmp_path)
    enqueue_post_command(_request_obj(provider="claude"), spool_path=tmp_path)
    claude = poll_post_commands(provider="claude", spool_path=tmp_path)
    assert [c.provider for c in claude] == ["claude"]


def test_ack_unknown_command(tmp_path: Path, post_enabled: None) -> None:
    assert (
        ack_post_command("does-not-exist", BrowserPostCommandAckRequest(status="failed"), spool_path=tmp_path) is None
    )


def test_enqueue_rejects_duplicate_normalized_command_id(tmp_path: Path, post_enabled: None) -> None:
    enqueue_post_command(_request_obj().model_copy(update={"command_id": "manual/id"}), spool_path=tmp_path)

    with pytest.raises(BrowserPostCommandConflictError):
        enqueue_post_command(_request_obj().model_copy(update={"command_id": "manual id"}), spool_path=tmp_path)


def test_enqueue_over_file_count_quota_raises(
    tmp_path: Path, post_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-command queue has its own bound, independent of the capture
    spool -- a runaway enqueue loop must not grow the queue unbounded
    (polylogue-gnie)."""
    import polylogue.browser_capture.receiver as receiver_mod
    from polylogue.browser_capture.receiver import SpoolQuotaExceededError

    monkeypatch.setattr(receiver_mod, "POST_COMMAND_QUEUE_MAX_FILES", 1)
    enqueue_post_command(_request_obj(conversation_id="conv-1"), spool_path=tmp_path)

    with pytest.raises(SpoolQuotaExceededError):
        enqueue_post_command(_request_obj(conversation_id="conv-2"), spool_path=tmp_path)


def test_enqueue_over_byte_quota_raises(tmp_path: Path, post_enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.browser_capture.receiver as receiver_mod
    from polylogue.browser_capture.receiver import SpoolQuotaExceededError

    enqueue_post_command(_request_obj(conversation_id="conv-1"), spool_path=tmp_path)
    monkeypatch.setattr(receiver_mod, "POST_COMMAND_QUEUE_MAX_BYTES", 1)

    with pytest.raises(SpoolQuotaExceededError):
        enqueue_post_command(_request_obj(conversation_id="conv-2"), spool_path=tmp_path)


def test_enqueue_under_quota_succeeds(tmp_path: Path, post_enabled: None) -> None:
    command = enqueue_post_command(_request_obj(), spool_path=tmp_path)
    assert command.status == "pending"


def test_ack_requires_dispatched_state(tmp_path: Path, post_enabled: None) -> None:
    command = enqueue_post_command(_request_obj(), spool_path=tmp_path)

    with pytest.raises(BrowserPostCommandStateError):
        ack_post_command(command.command_id, BrowserPostCommandAckRequest(status="submitted"), spool_path=tmp_path)


def test_ack_terminal_state_is_idempotent(tmp_path: Path, post_enabled: None) -> None:
    command = enqueue_post_command(_request_obj(), spool_path=tmp_path)
    [dispatched] = poll_post_commands(provider="chatgpt", spool_path=tmp_path)
    acked = ack_post_command(
        dispatched.command_id,
        BrowserPostCommandAckRequest(status="submitted", detail="first", observed_url="https://chatgpt.com/c/conv-1"),
        spool_path=tmp_path,
    )
    repeated = ack_post_command(
        command.command_id,
        BrowserPostCommandAckRequest(status="failed", detail="second", observed_url="https://chatgpt.com/c/conv-2"),
        spool_path=tmp_path,
    )

    assert acked is not None
    assert repeated is not None
    assert repeated.status == "submitted"
    assert repeated.detail == "first"
    assert repeated.observed_url == "https://chatgpt.com/c/conv-1"


# ---- HTTP route surface ---------------------------------------------------


@contextmanager
def _running_receiver(tmp_path: Path) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(
    host: str, port: int, method: str, path: str, *, body: object | None = None
) -> tuple[int, dict[str, object]]:
    conn = HTTPConnection(host, port)
    headers = {"Origin": _EXTENSION_ORIGIN}
    payload: str | None = None
    if body is not None:
        payload = json.dumps(body)
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    response: HTTPResponse = conn.getresponse()
    raw = response.read()
    conn.close()
    parsed = json.loads(raw) if raw else {}
    return response.status, parsed


def test_route_contracts_registered() -> None:
    kinds = {c.kind for c in BROWSER_CAPTURE_ROUTE_CONTRACTS}
    assert {"post_command_enqueue", "post_command_poll", "post_command_ack"} <= kinds
    assert browser_capture_route_contract_for("POST", "/v1/post-commands") is not None
    assert browser_capture_route_contract_for("GET", "/v1/post-commands") is not None
    assert browser_capture_route_contract_for("POST", "/v1/post-commands/abc123/ack") is not None


def test_http_enqueue_disabled_returns_403(tmp_path: Path, post_disabled: None) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(host, port, "POST", "/v1/post-commands", body={"provider": "chatgpt", "text": "hi"})
    assert status == HTTPStatus.FORBIDDEN
    assert body["error"] == "post_disabled"


def test_http_enqueue_returns_429_without_writing_when_queue_quota_exceeded(
    tmp_path: Path, post_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.browser_capture.receiver as receiver_mod

    monkeypatch.setattr(receiver_mod, "POST_COMMAND_QUEUE_MAX_FILES", 1)
    enqueue_post_command(_request_obj(conversation_id="conv-1"), spool_path=tmp_path)
    before = sorted(post_command_queue_root(tmp_path).glob("*.json"))

    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(
            host,
            port,
            "POST",
            "/v1/post-commands",
            body={"provider": "chatgpt", "target": {"conversation_id": "conv-2"}, "text": "hi"},
        )

    assert status == HTTPStatus.TOO_MANY_REQUESTS
    assert body["error"] == "post_command_quota_exceeded"
    after = sorted(post_command_queue_root(tmp_path).glob("*.json"))
    assert after == before


def test_http_post_command_full_lifecycle(tmp_path: Path, post_enabled: None) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, enq = _request(
            host,
            port,
            "POST",
            "/v1/post-commands",
            body={
                "provider": "chatgpt",
                "target": {"conversation_id": "conv-9"},
                "text": "Research X",
                "submit": False,
            },
        )
        assert status == HTTPStatus.ACCEPTED
        command_id = cast(str, enq["command_id"])
        assert enq["status"] == "pending"

        status, poll = _request(host, port, "GET", "/v1/post-commands?provider=chatgpt")
        assert status == HTTPStatus.OK
        assert poll["post_enabled"] is True
        commands = cast(list[dict[str, object]], poll["commands"])
        assert [c["command_id"] for c in commands] == [command_id]
        assert commands[0]["status"] == "dispatched"

        status, ack = _request(
            host,
            port,
            "POST",
            f"/v1/post-commands/{command_id}/ack",
            body={"status": "submitted", "detail": "dry_run_filled_not_sent"},
        )
        assert status == HTTPStatus.OK
        assert ack["status"] == "submitted"

        # The claimed+acked command is no longer pending for the extension.
        status, poll2 = _request(host, port, "GET", "/v1/post-commands?provider=chatgpt")
        assert cast(list[object], poll2["commands"]) == []


def test_http_poll_empty_when_disabled(tmp_path: Path, post_disabled: None) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(host, port, "GET", "/v1/post-commands")
    assert status == HTTPStatus.OK
    assert body["post_enabled"] is False
    assert body["commands"] == []


def test_http_ack_unknown_command_404(tmp_path: Path, post_enabled: None) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, body = _request(host, port, "POST", "/v1/post-commands/nope/ack", body={"status": "failed"})
    assert status == HTTPStatus.NOT_FOUND
    assert body["error"] == "unknown_command"


def test_http_ack_pending_command_returns_conflict(tmp_path: Path, post_enabled: None) -> None:
    with _running_receiver(tmp_path) as (host, port):
        status, enq = _request(
            host,
            port,
            "POST",
            "/v1/post-commands",
            body={"provider": "chatgpt", "target": {"conversation_id": "conv-9"}, "text": "Research X"},
        )
        assert status == HTTPStatus.ACCEPTED
        command_id = cast(str, enq["command_id"])

        status, body = _request(
            host,
            port,
            "POST",
            f"/v1/post-commands/{command_id}/ack",
            body={"status": "submitted"},
        )

    assert status == HTTPStatus.CONFLICT
    assert body["error"] == "invalid_post_command_state"


def test_daemon_post_command_reports_empty_text_cleanly(tmp_path: Path, post_enabled: None) -> None:
    result = CliRunner().invoke(
        daemon_cli,
        [
            "browser-capture",
            "post",
            "--provider",
            "chatgpt",
            "--text",
            "   ",
            "--spool",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "post command text must not be empty" in result.output
