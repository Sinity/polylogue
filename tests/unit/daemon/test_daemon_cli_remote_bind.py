"""Remote-bind enforcement for the daemon API (#868-A4, #868-A5).

The daemon must refuse to start an HTTP API on a non-loopback address
unless the operator explicitly opts in (``--insecure-allow-remote``)
*and* configures a bearer token (``--api-auth-token``). The logic lives
at the top of ``run_daemon_services`` and is the gate that prevents
accidental exposure of the local archive over the network.

Two refusal conditions, each tested:

1. Non-loopback bind without ``--insecure-allow-remote`` → UsageError.
2. Non-loopback bind with ``--insecure-allow-remote`` but no token →
   UsageError.

A passing positive case (loopback bind, no token required) would
require mocking the entire daemon startup chain, which is out of scope
for a focused security test. The pure-logic refusal lives at the top
of the function and fires before any heavyweight setup.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from polylogue.daemon.cli import main, run_daemon_services


def _run(coro: object) -> None:
    """Drive an async function until it raises or returns."""
    asyncio.run(coro)  # type: ignore[arg-type]


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("api_host", ["0.0.0.0", "192.168.1.1", "10.0.0.1"])
def test_non_loopback_bind_without_allow_remote_refuses(api_host: str) -> None:
    """Refusal carries the operator-actionable message naming the flag."""
    with pytest.raises(click.UsageError, match="not a loopback"):
        _run(
            run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                browser_capture_allow_remote=False,
                browser_capture_auth_token=None,
                browser_capture_extra_origins=(),
                enable_api=True,
                api_host=api_host,
                api_port=8766,
                api_auth_token="some-token",
            )
        )


@pytest.mark.parametrize("api_host", ["0.0.0.0", "192.168.1.1"])
def test_non_loopback_bind_with_allow_remote_but_no_token_refuses(
    api_host: str,
) -> None:
    """Even with the explicit risk-acknowledgement flag, a token is required."""
    with pytest.raises(click.UsageError, match="requires --api-auth-token"):
        _run(
            run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                browser_capture_allow_remote=True,
                browser_capture_auth_token=None,
                browser_capture_extra_origins=(),
                enable_api=True,
                api_host=api_host,
                api_port=8766,
                api_auth_token=None,
            )
        )


@pytest.mark.parametrize("api_host", ["0.0.0.0", "192.168.1.1"])
def test_non_loopback_bind_with_allow_remote_and_empty_token_refuses(
    api_host: str,
) -> None:
    """Empty string token is treated as no token (truthy check)."""
    with pytest.raises(click.UsageError, match="requires --api-auth-token"):
        _run(
            run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                browser_capture_allow_remote=True,
                browser_capture_auth_token=None,
                browser_capture_extra_origins=(),
                enable_api=True,
                api_host=api_host,
                api_port=8766,
                api_auth_token="",
            )
        )


@pytest.mark.parametrize(
    ("api_host", "receiver_host"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("localhost", "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1"),
    ],
)
def test_api_and_browser_capture_same_socket_refuses(api_host: str, receiver_host: str) -> None:
    """A static host/port collision fails before either HTTP server binds."""
    with pytest.raises(click.UsageError, match="conflicts with browser-capture receiver"):
        _run(
            run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=True,
                browser_capture_host=receiver_host,
                browser_capture_port=8766,
                browser_capture_spool_path=None,
                browser_capture_allow_remote=True,
                browser_capture_auth_token="receiver-token",
                browser_capture_extra_origins=(),
                enable_api=True,
                api_host=api_host,
                api_port=8766,
                api_auth_token="api-token",
            )
        )


@pytest.mark.timeout(30)
def test_loopback_bind_passes_remote_check() -> None:
    """Loopback bind does not trip the remote-bind refusal.

    Tests only the security gate at the top of ``run_daemon_services``;
    we patch the post-gate startup chain to a fast no-op so the test
    exits the moment the gate decides "allow." A regression that broadens
    the gate to apply to loopback would surface as a UsageError here.
    """
    from unittest.mock import patch

    api_port = _unused_loopback_port()
    configured: list[dict[str, object]] = []

    def record_runtime_components(**kwargs: object) -> None:
        configured.append(kwargs)

    with (
        patch(
            "polylogue.daemon.cli._run_startup_fts_readiness",
            side_effect=RuntimeError("post-gate sentinel"),
        ),
        patch("polylogue.daemon.status_snapshot.configure_runtime_components", side_effect=record_runtime_components),
    ):
        with pytest.raises(RuntimeError, match="post-gate sentinel"):
            _run(
                run_daemon_services(
                    sources=(),
                    debounce_s=1.0,
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                    browser_capture_allow_remote=False,
                    browser_capture_auth_token=None,
                    browser_capture_extra_origins=(),
                    enable_api=True,
                    api_host="127.0.0.1",
                    api_port=api_port,
                    api_auth_token=None,
                )
            )

    assert configured == [
        {
            "api_enabled": True,
            "watcher_enabled": False,
            "watcher_roots": (),
            "browser_capture_enabled": False,
            "browser_capture_spool_path": None,
        }
    ]


@pytest.mark.timeout(30)
def test_api_disabled_skips_remote_check() -> None:
    """If the API is not enabled at all, the remote-bind check should
    not fire — the operator hasn't asked for an API server, so even a
    non-loopback ``api_host`` value is irrelevant.
    """
    from unittest.mock import patch

    with patch(
        "polylogue.daemon.cli._run_startup_fts_readiness",
        side_effect=RuntimeError("post-gate sentinel"),
    ):
        with pytest.raises(RuntimeError, match="post-gate sentinel"):
            _run(
                run_daemon_services(
                    sources=(),
                    debounce_s=1.0,
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                    browser_capture_spool_path=None,
                    browser_capture_allow_remote=False,
                    browser_capture_auth_token=None,
                    browser_capture_extra_origins=(),
                    enable_api=False,
                    api_host="0.0.0.0",  # would trip if gate fired
                    api_port=8766,
                    api_auth_token=None,
                )
            )


def test_run_command_applies_configured_remote_api_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOML/env network policy feeds the same remote-bind safety gate as CLI flags."""
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("POLYLOGUE_API_HOST", "0.0.0.0")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_ALLOW_REMOTE", "true")

    result = CliRunner().invoke(main, ["run", "--no-watch", "--no-browser-capture"])

    assert result.exit_code != 0
    assert "requires --api-auth-token" in result.output


def test_run_command_passes_effective_config_to_daemon_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup-bound TOML/env values are honored without requiring duplicate CLI flags."""
    cfg = tmp_path / "polylogue.toml"
    spool = tmp_path / "capture-spool"
    cfg.write_text(
        f"""
[daemon.api]
host = "0.0.0.0"
port = 9901
auth_token = "api-secret"

[daemon.browser_capture]
host = "0.0.0.0"
port = 9902
allow_remote = true
auth_token = "browser-secret"
allowed_origins = "https://workbench.example"
spool_path = "{spool}"

[daemon.watch]
debounce_s = 0.25
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(tmp_path / "absent-site.toml"))

    recorded: dict[str, object] = {}

    async def fake_run_daemon_services(**kwargs: object) -> None:
        recorded.update(kwargs)

    with patch("polylogue.daemon.cli.run_daemon_services", side_effect=fake_run_daemon_services):
        result = CliRunner().invoke(main, ["run", "--no-watch"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert recorded["api_host"] == "0.0.0.0"
    assert recorded["api_port"] == 9901
    assert recorded["api_auth_token"] == "api-secret"
    assert recorded["browser_capture_host"] == "0.0.0.0"
    assert recorded["browser_capture_port"] == 9902
    assert recorded["browser_capture_allow_remote"] is True
    assert recorded["browser_capture_auth_token"] == "browser-secret"
    assert recorded["browser_capture_extra_origins"] == ("https://workbench.example",)
    assert recorded["browser_capture_spool_path"] == spool
    assert recorded["debounce_s"] == 0.25
