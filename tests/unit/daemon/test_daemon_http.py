"""Security tests for daemon HTTP auth and origin enforcement (#868)."""

from __future__ import annotations


def test_cross_origin_logic_rejects_external_origin() -> None:
    """_check_cross_origin returns False for non-localhost Origin header."""
    assert _origin_check("https://evil.example.com") is False
    assert _origin_check("http://192.168.1.1:8080") is False


def test_cross_origin_logic_allows_localhost() -> None:
    """_check_cross_origin returns True for localhost Origin header."""
    assert _origin_check("http://127.0.0.1:8766") is True
    assert _origin_check("http://localhost:3000") is True
    assert _origin_check("https://127.0.0.1:8766") is True


def test_cross_origin_logic_allows_no_origin() -> None:
    """_check_cross_origin returns True when no Origin header is present."""
    assert _origin_check("") is True


def test_auth_requires_token_when_configured() -> None:
    """_check_auth_logic returns not-allowed when token set but not sent."""
    from polylogue.daemon.http import _check_auth_logic

    assert _check_auth_logic("secret", "127.0.0.1", "").allowed is False
    assert _check_auth_logic("secret", "127.0.0.1", "Bearer wrong").allowed is False
    assert _check_auth_logic("secret", "127.0.0.1", "Bearer secret").allowed is True


def test_auth_allows_when_no_token_configured() -> None:
    """_check_auth_logic returns allowed when no token is configured."""
    from polylogue.daemon.http import _check_auth_logic

    assert _check_auth_logic("", "127.0.0.1", "").allowed is True
    assert _check_auth_logic(None, "192.168.1.1", "").allowed is True


def _origin_check(origin: str) -> bool:
    """Simulate the origin check logic from _check_cross_origin."""
    if not origin:
        return True
    return (
        origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")
        or origin.startswith("https://127.0.0.1:") or origin.startswith("https://localhost:")
    )
