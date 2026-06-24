"""Contract tests for ``polylogue.core.loopback``.

The shared helper is consumed by ``polylogue/daemon/http.py`` (HTTP API
trust boundary), ``polylogue/daemon/cli.py`` (remote-bind enforcement),
and ``polylogue/browser_capture/server.py`` (capture receiver). Pin the
RFC 5735 contract so a future regression in any consumer fails here
first instead of at the network surface.
"""

from __future__ import annotations

import pytest

from polylogue.core.loopback import bind_hosts_overlap, is_loopback_host, is_loopback_origin


class TestIsLoopbackHost:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "127.0.0.2",
            "127.1.2.3",
            "127.255.255.254",
            "::1",
            "localhost",
        ],
    )
    def test_loopback_accepted(self, host: str) -> None:
        assert is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "",
            "0.0.0.0",
            "10.0.0.1",
            "192.168.1.1",
            "8.8.8.8",
            "128.0.0.1",
            "::",
            "fe80::1",
            "not-an-address",
            "evil.example.com",
        ],
    )
    def test_non_loopback_rejected(self, host: str) -> None:
        assert is_loopback_host(host) is False


class TestBindHostsOverlap:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("127.0.0.1", "127.0.0.1"),
            ("127.0.0.1", "localhost"),
            ("0.0.0.0", "127.0.0.1"),
            ("::", "127.0.0.1"),
            ("", "192.168.1.10"),
        ],
    )
    def test_overlapping_bind_hosts(self, left: str, right: str) -> None:
        assert bind_hosts_overlap(left, right) is True

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("127.0.0.1", "192.168.1.10"),
            ("10.0.0.1", "192.168.1.10"),
            ("::1", "192.168.1.10"),
        ],
    )
    def test_non_overlapping_bind_hosts(self, left: str, right: str) -> None:
        assert bind_hosts_overlap(left, right) is False


class TestIsLoopbackOrigin:
    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost:8765",
            "https://localhost:8766",
            "http://127.0.0.1:8765",
            "https://127.0.0.1:8765",
            "http://127.0.0.2:8765",
            "http://127.1.2.3:9999",
            "http://[::1]:8765",
            "https://[::1]:8765/path/segment",
        ],
    )
    def test_loopback_origins(self, origin: str) -> None:
        assert is_loopback_origin(origin) is True

    @pytest.mark.parametrize(
        "origin",
        [
            "",
            "http://example.com",
            "https://192.168.1.1:8765",
            "http://0.0.0.0:8765",
            "ftp://localhost:8765",
            "chrome-extension://abcdef",
            "http://[fe80::1]:8765",
            "http://[::]:8765",
            "http://[::1",  # malformed bracket
        ],
    )
    def test_non_loopback_origins(self, origin: str) -> None:
        assert is_loopback_origin(origin) is False
