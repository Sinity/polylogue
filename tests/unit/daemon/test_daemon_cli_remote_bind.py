"""Remote-bind enforcement for the daemon API (#868-A4, #868-A5, polylogue-rzve).

The daemon must refuse to start an HTTP API on a non-loopback address
unless the operator explicitly opts in (``--insecure-allow-remote``)
*and* an auth token is in effect (``--api-auth-token`` is an "auto"
default -- polylogue-rzve made that real: a token is auto-minted/loaded
unless the operator explicitly opts all the way out with
``--api-allow-no-auth``). The logic lives at the top of
``run_daemon_services`` and is the gate that prevents accidental
exposure of the local archive over the network.

Refusal conditions, each tested:

1. Non-loopback bind without ``--insecure-allow-remote`` → UsageError.
2. Non-loopback bind with ``--insecure-allow-remote`` *and*
   ``--api-allow-no-auth`` (the only way to reach "no token in effect"
   now that a token auto-mints by default) → UsageError.

Passing bind-policy cases use the production policy helper through the
resident-core service harness, so they do not start archive convergence.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from polylogue.daemon.cli import main, run_daemon_services
from polylogue.daemon.services import ServiceCapability, ServiceProfile
from tests.infra.daemon_service_harness import ServiceHarness


def _run(coro: object) -> None:
    """Drive an async function until it raises or returns."""
    asyncio.run(coro)  # type: ignore[arg-type]


@pytest.mark.parametrize("api_host", ["0.0.0.0", "192.168.1.1", "10.0.0.1"])
def test_non_loopback_bind_without_allow_remote_refuses(api_host: str) -> None:
    """Refusal carries the operator-actionable message naming the flag."""
    with pytest.raises(click.UsageError, match="not a loopback"):
        _run(
            run_daemon_services(
                sources=(),
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
def test_non_loopback_bind_with_allow_remote_and_allow_no_auth_refuses(
    api_host: str,
) -> None:
    """Even with the explicit risk-acknowledgement flag, an explicit
    ``--api-allow-no-auth`` opt-out cannot be combined with a remote bind --
    remote exposure without any credential in effect is never supported."""
    with pytest.raises(click.UsageError, match="requires --api-auth-token"):
        _run(
            run_daemon_services(
                sources=(),
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
                api_allow_no_auth=True,
            )
        )


@pytest.mark.parametrize("api_host", ["0.0.0.0", "192.168.1.1"])
def test_non_loopback_bind_with_allow_remote_and_no_explicit_token_auto_mints(
    api_host: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote bind with no explicit ``--api-auth-token`` and no
    ``--api-allow-no-auth`` opt-out succeeds -- the gate is satisfied by a
    real auto-minted token, matching the browser-capture receiver's
    already-shipped contract (polylogue-rzve). Removing the auto-mint call
    from ``resolve_api_auth_token`` makes this raise UsageError instead.

    The production bind-policy helper is exercised through the resident-core
    service harness, without starting archive convergence or opening sockets.
    """
    harness = ServiceHarness(
        profile=ServiceProfile.SURFACES,
        capabilities={ServiceCapability.API},
    )
    harness.require_selected("api_server")
    assert "raw_observation_convergence" not in harness.selected_names

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    from polylogue.daemon.cli import resolve_api_auth_token

    token = resolve_api_auth_token(None)
    assert token
    harness.validate_api_bind(enabled=True, host=api_host, allow_remote=True, auth_token=token)
    from polylogue.paths import api_auth_token_path

    assert api_auth_token_path().exists()


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


def test_browser_host_requires_api_and_distinct_port() -> None:
    with pytest.raises(click.UsageError, match="requires the daemon API"):
        _run(
            run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                browser_port=8767,
                enable_api=False,
            )
        )
    with pytest.raises(click.UsageError, match="must differ"):
        _run(
            run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                browser_port=8767,
                enable_api=True,
                api_port=8767,
            )
        )


def test_browser_host_requires_loopback_reachable_api_bind() -> None:
    """A remote-specific API address gives the browser proxy no loopback upstream."""
    with pytest.raises(click.UsageError, match="reachable on loopback"):
        _run(
            run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
                enable_api=True,
                api_host="192.0.2.1",
                api_port=8766,
                browser_port=8767,
            )
        )


@pytest.mark.uses_real_clock("times ten sequential production-profile network policy fixtures")
def test_loopback_bind_passes_remote_check() -> None:
    """Loopback bind does not trip the remote-bind refusal.

    The focused production profile is selected and the production policy
    helper decides the bind without entering archive startup.
    """
    elapsed_runs: list[float] = []
    for _ in range(10):
        started = time.monotonic()
        harness = ServiceHarness(
            profile=ServiceProfile.SURFACES,
            capabilities={ServiceCapability.API},
        )
        harness.require_selected("api_server")
        assert "raw_observation_convergence" not in harness.selected_names
        harness.validate_api_bind(enabled=True, host="127.0.0.1", allow_remote=False, auth_token="token")
        elapsed_runs.append(time.monotonic() - started)
        assert elapsed_runs[-1] < 10.0, elapsed_runs


@pytest.mark.uses_real_clock("times ten sequential production-profile API-disabled fixtures")
def test_api_disabled_skips_remote_check() -> None:
    """If the API is not enabled at all, the remote-bind check should
    not fire — the operator hasn't asked for an API server, so even a
    non-loopback ``api_host`` value is irrelevant.
    """
    elapsed_runs: list[float] = []
    for _ in range(10):
        started = time.monotonic()
        harness = ServiceHarness(profile=ServiceProfile.RESIDENT_CORE)
        harness.require_selected("api_server", selected=False)
        assert "raw_observation_convergence" not in harness.selected_names
        harness.validate_api_bind(enabled=False, host="0.0.0.0", allow_remote=False, auth_token=None)
        elapsed_runs.append(time.monotonic() - started)
        assert elapsed_runs[-1] < 10.0, elapsed_runs


def test_run_command_applies_configured_remote_api_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOML/env network policy feeds the same remote-bind safety gate as CLI flags.

    A remote bind alone no longer fails closed by itself -- an auth token
    now auto-mints by default (polylogue-rzve) -- so this also sets the
    explicit ``POLYLOGUE_API_ALLOW_NO_AUTH`` opt-out to reach the "no token
    in effect" state the remote-bind gate refuses.
    """
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("POLYLOGUE_API_HOST", "0.0.0.0")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_ALLOW_REMOTE", "true")
    monkeypatch.setenv("POLYLOGUE_API_ALLOW_NO_AUTH", "true")

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
    assert recorded["browser_port"] is None
    assert recorded["api_auth_token"] == "api-secret"
    assert recorded["browser_capture_host"] == "0.0.0.0"
    assert recorded["browser_capture_port"] == 9902
    assert recorded["browser_capture_allow_remote"] is True
    assert recorded["browser_capture_auth_token"] == "browser-secret"
    assert recorded["browser_capture_extra_origins"] == ("https://workbench.example",)
    assert recorded["browser_capture_spool_path"] == spool


def test_browser_port_is_opt_in_and_reaches_the_service_composition() -> None:
    recorded: dict[str, object] = {}

    async def fake_run_daemon_services(**kwargs: object) -> None:
        recorded.update(kwargs)

    with patch("polylogue.daemon.cli.run_daemon_services", side_effect=fake_run_daemon_services):
        result = CliRunner().invoke(main, ["run", "--no-watch", "--no-browser-capture", "--browser-port", "8767"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert recorded["browser_port"] == 8767
