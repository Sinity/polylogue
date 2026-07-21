"""One-time pairing-code bootstrap (polylogue-gnie) and capture-health telemetry (polylogue-3v1).

polylogue-gnie's recommended design is an installer-managed native-messaging
bootstrap; this repo cannot install an OS-level native-messaging host or
exercise a real browser extension identity headlessly, so that path is not
covered here (see the PR body's AC matrix). What *is* covered, over real
HTTP against the threaded receiver server (same pattern as
``test_receiver_token.py``), is the design's explicit fallback: a
short-lived single-use pairing code that lets a fresh install adopt the
receiver's bearer token without the operator ever viewing, copying, or
pasting the token itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.pairing import (
    PAIRING_CODE_MAX_ATTEMPTS,
    PairingCodeAlreadyUsedError,
    PairingCodeError,
    PairingCodeExpiredError,
    PairingCodeInvalidError,
    PairingCodeRateLimitedError,
    mint_pairing_code,
    redeem_pairing_code,
)
from polylogue.browser_capture.receiver import load_or_mint_receiver_token
from polylogue.browser_capture.server import make_server
from polylogue.daemon.cli import main as daemon_cli

_EXTENSION_ORIGIN = "chrome-extension://polylogue-test"
_CHATGPT_ORIGIN = "https://chatgpt.com"


@contextmanager
def _running_receiver(spool_path: Path, *, auth_token: str | None) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=spool_path, auth_token=auth_token)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# polylogue.browser_capture.pairing unit coverage
# ---------------------------------------------------------------------------


def test_mint_and_redeem_pairing_code_returns_current_token(tmp_path: Path) -> None:
    state_path = tmp_path / "pairing"
    token_path = tmp_path / "token"
    expected_token = load_or_mint_receiver_token(token_path)

    minted = mint_pairing_code(path=state_path)
    # Plaintext code must never be persisted to disk, only its hash.
    assert minted.code not in state_path.read_text(encoding="utf-8")

    resolved = redeem_pairing_code(minted.code, path=state_path, token_path=token_path)

    assert resolved == expected_token


def test_redeem_pairing_code_is_single_use(tmp_path: Path) -> None:
    state_path = tmp_path / "pairing"
    token_path = tmp_path / "token"
    minted = mint_pairing_code(path=state_path)

    redeem_pairing_code(minted.code, path=state_path, token_path=token_path)

    with pytest.raises((PairingCodeInvalidError, PairingCodeAlreadyUsedError)):
        redeem_pairing_code(minted.code, path=state_path, token_path=token_path)


def test_redeem_pairing_code_rejects_wrong_code(tmp_path: Path) -> None:
    state_path = tmp_path / "pairing"
    mint_pairing_code(path=state_path)

    with pytest.raises(PairingCodeInvalidError):
        redeem_pairing_code("WRONGCODE", path=state_path)


def test_redeem_pairing_code_rejects_expired_code(tmp_path: Path) -> None:
    state_path = tmp_path / "pairing"
    minted = mint_pairing_code(ttl_seconds=1, path=state_path)
    # Force the persisted expiry into the past instead of sleeping.
    record = json.loads(state_path.read_text(encoding="utf-8"))
    record["expires_at_ms"] = 0
    state_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(PairingCodeExpiredError):
        redeem_pairing_code(minted.code, path=state_path)


def test_redeem_pairing_code_with_no_pending_code_is_invalid(tmp_path: Path) -> None:
    state_path = tmp_path / "pairing"

    with pytest.raises(PairingCodeInvalidError):
        redeem_pairing_code("ANYCODE1", path=state_path)


def test_redeem_pairing_code_locks_out_after_max_wrong_attempts(tmp_path: Path) -> None:
    state_path = tmp_path / "pairing"
    minted = mint_pairing_code(path=state_path)

    for _ in range(PAIRING_CODE_MAX_ATTEMPTS):
        with pytest.raises(PairingCodeInvalidError):
            redeem_pairing_code("WRONGCODE", path=state_path)

    # The *correct* code is now also rejected -- the pending code is burned,
    # not just individual wrong guesses.
    with pytest.raises((PairingCodeRateLimitedError, PairingCodeError)):
        redeem_pairing_code(minted.code, path=state_path)


def test_mint_pairing_code_replaces_any_prior_pending_code(tmp_path: Path) -> None:
    state_path = tmp_path / "pairing"
    first = mint_pairing_code(path=state_path)
    second = mint_pairing_code(path=state_path)

    assert first.code != second.code
    with pytest.raises(PairingCodeInvalidError):
        redeem_pairing_code(first.code, path=state_path)
    # The newer code still works.
    redeem_pairing_code(second.code, path=state_path)


# ---------------------------------------------------------------------------
# Real-HTTP receiver route coverage
# ---------------------------------------------------------------------------


def test_pairing_redeem_route_pairs_without_a_preexisting_token(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    """A fresh install has no bearer token configured at all (`auth_token`
    would be `None` on an unpaired client) -- the redeem call itself must
    succeed unauthenticated, gated only by extension origin + the code."""
    minted = mint_pairing_code()
    expected_token = load_or_mint_receiver_token()

    with _running_receiver(tmp_path / "spool", auth_token=expected_token) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/pairing/redeem",
            body=json.dumps({"code": minted.code}),
            headers={"Origin": _EXTENSION_ORIGIN, "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()

    assert response.status == HTTPStatus.OK
    assert body["auth_token"] == expected_token
    assert body["receiver_id"]


def test_pairing_redeem_route_rejects_wrong_code(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    mint_pairing_code()
    token = load_or_mint_receiver_token()

    with _running_receiver(tmp_path / "spool", auth_token=token) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/pairing/redeem",
            body=json.dumps({"code": "WRONGCODE"}),
            headers={"Origin": _EXTENSION_ORIGIN, "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.UNAUTHORIZED


def test_pairing_redeem_route_rejects_disallowed_origin(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    """An arbitrary webpage cannot forge the extension's origin and must not
    be able to redeem a pairing code even if it somehow learned one."""
    minted = mint_pairing_code()
    token = load_or_mint_receiver_token()

    with _running_receiver(tmp_path / "spool", auth_token=token) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/pairing/redeem",
            body=json.dumps({"code": minted.code}),
            headers={"Origin": _CHATGPT_ORIGIN, "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.FORBIDDEN


def test_pairing_redeem_route_is_single_use_over_http(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    minted = mint_pairing_code()
    token = load_or_mint_receiver_token()

    with _running_receiver(tmp_path / "spool", auth_token=token) as (host, port):

        def _redeem() -> int:
            conn = HTTPConnection(host, port)
            conn.request(
                "POST",
                "/v1/pairing/redeem",
                body=json.dumps({"code": minted.code}),
                headers={"Origin": _EXTENSION_ORIGIN, "Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response.read()
            conn.close()
            return response.status

        assert _redeem() == HTTPStatus.OK
        assert _redeem() == HTTPStatus.UNAUTHORIZED


# ---------------------------------------------------------------------------
# Capture-health telemetry (polylogue-3v1)
# ---------------------------------------------------------------------------


def test_capture_health_report_requires_bearer_token(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    token = load_or_mint_receiver_token()

    with _running_receiver(tmp_path / "spool", auth_token=token) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/capture-health",
            body=json.dumps({"event": "capture_gap", "provider": "chatgpt"}),
            headers={"Origin": _EXTENSION_ORIGIN, "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.UNAUTHORIZED


def test_capture_health_report_is_queryable_via_get(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    token = load_or_mint_receiver_token()

    with _running_receiver(tmp_path / "spool", auth_token=token) as (host, port):
        report = HTTPConnection(host, port)
        report.request(
            "POST",
            "/v1/capture-health",
            body=json.dumps(
                {
                    "event": "capture_gap",
                    "provider": "chatgpt",
                    "provider_session_id": "conv-123",
                    "extension_instance_id": "ext-1",
                    "visible_count": 4,
                    "captured_count": 2,
                }
            ),
            headers={
                "Origin": _EXTENSION_ORIGIN,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        report_response = report.getresponse()
        report_body = json.loads(report_response.read())
        report.close()

        listing = HTTPConnection(host, port)
        listing.request(
            "GET",
            "/v1/capture-health",
            headers={"Origin": _EXTENSION_ORIGIN, "Authorization": f"Bearer {token}"},
        )
        listing_response = listing.getresponse()
        listing_body = json.loads(listing_response.read())
        listing.close()

    assert report_response.status == HTTPStatus.ACCEPTED
    assert report_body["event_id"]
    assert listing_response.status == HTTPStatus.OK
    assert listing_body["ok"] is True
    events = listing_body["events"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["event"] == "capture_gap"
    assert payload["provider"] == "chatgpt"
    assert payload["provider_session_id"] == "conv-123"
    assert payload["visible_count"] == 4
    assert payload["captured_count"] == 2


def test_capture_health_report_rejects_invalid_event_kind(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    token = load_or_mint_receiver_token()

    with _running_receiver(tmp_path / "spool", auth_token=token) as (host, port):
        conn = HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/capture-health",
            body=json.dumps({"event": "not_a_real_kind"}),
            headers={
                "Origin": _EXTENSION_ORIGIN,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == HTTPStatus.BAD_REQUEST


# ---------------------------------------------------------------------------
# `polylogued browser-capture pairing start` / `capture-health` CLI
# ---------------------------------------------------------------------------


def test_pairing_start_cli_mints_a_redeemable_code(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()

    result = runner.invoke(
        daemon_cli, ["browser-capture", "pairing", "start", "--format", "json"], catch_exceptions=False
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["code"]) == 8
    assert payload["ttl_seconds"] > 0
    # The code the CLI just printed must actually redeem.
    redeem_pairing_code(payload["code"])


def test_pairing_start_cli_human_output_does_not_leak_the_bearer_token(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()

    result = runner.invoke(daemon_cli, ["browser-capture", "pairing", "start"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Pairing code:" in result.output
    # The receiver's bearer token must never be minted/echoed by this command
    # -- only the short-lived code is operator-facing.
    assert not (cli_workspace["archive_root"] / "browser-capture-receiver-token").exists()


def test_capture_health_cli_lists_reported_events(cli_workspace: dict[str, Path]) -> None:
    from polylogue.daemon.events import emit_daemon_event

    emit_daemon_event(
        "browser_capture_health",
        operation_id="ext-1",
        payload={"event": "capture_gap", "provider": "chatgpt", "provider_session_id": "conv-1"},
    )
    runner = CliRunner()

    result = runner.invoke(
        daemon_cli, ["browser-capture", "capture-health", "--format", "json"], catch_exceptions=False
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["events"]) == 1
    assert payload["events"][0]["payload"]["event"] == "capture_gap"


def test_capture_health_cli_reports_no_events_when_empty(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()

    result = runner.invoke(daemon_cli, ["browser-capture", "capture-health"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "No capture-health events recorded." in result.output
