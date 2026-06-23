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

import click
import pytest

from polylogue.daemon.cli import run_daemon_services


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
            "polylogue.daemon.cli._ensure_fts_startup_readiness",
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
        "polylogue.daemon.cli._ensure_fts_startup_readiness",
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
