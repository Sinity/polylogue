"""Auto-minted daemon HTTP API bearer token (polylogue-rzve).

``docs/daemon.md`` documented ``--api-auth-token`` as "auto-generated if not
provided" while ``run_daemon_services`` merely stored ``auth_token=None`` --
no minting code existed anywhere. These tests cover the mint/load/rotate/
resolve primitives in ``polylogue.daemon.api_auth`` (mirroring the
browser-capture receiver's already-shipped token contract exactly) and the
CLI ``polylogued api token show`` surface.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.daemon.api_auth import (
    api_command,
    load_or_mint_api_auth_token,
    resolve_api_auth_token,
)
from polylogue.paths import api_auth_token_path


def test_load_or_mint_api_auth_token_creates_0600_file_and_persists(tmp_path: Path) -> None:
    token_path = tmp_path / "api-auth-token"

    first = load_or_mint_api_auth_token(token_path)
    second = load_or_mint_api_auth_token(token_path)

    assert first == second
    assert token_path.read_text(encoding="utf-8").strip() == first
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_load_or_mint_api_auth_token_rotate_changes_the_value(tmp_path: Path) -> None:
    token_path = tmp_path / "api-auth-token"

    original = load_or_mint_api_auth_token(token_path)
    rotated = load_or_mint_api_auth_token(token_path, rotate=True)
    reloaded = load_or_mint_api_auth_token(token_path)

    assert rotated != original
    assert reloaded == rotated


def test_default_token_path_is_scoped_to_archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two POLYLOGUE_ARCHIVE_ROOT values must never resolve to the same
    default token file -- a scratch archive must not silently authenticate
    against the real daemon's token (mirrors the receiver-token test)."""
    root_a = tmp_path / "archive-a"
    root_b = tmp_path / "archive-b"

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root_a))
    path_a = api_auth_token_path()
    token_a = load_or_mint_api_auth_token()

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root_b))
    path_b = api_auth_token_path()
    token_b = load_or_mint_api_auth_token()

    assert path_a != path_b
    assert token_a != token_b
    assert root_a in path_a.parents
    assert root_b in path_b.parents


def test_resolve_api_auth_token_prefers_explicit_token(tmp_path: Path) -> None:
    token_path = tmp_path / "api-auth-token"

    resolved = resolve_api_auth_token("explicit-secret", token_path=token_path)

    assert resolved == "explicit-secret"
    assert not token_path.exists()


def test_resolve_api_auth_token_allow_no_auth_returns_none(tmp_path: Path) -> None:
    token_path = tmp_path / "api-auth-token"

    resolved = resolve_api_auth_token(None, allow_no_auth=True, token_path=token_path)

    assert resolved is None
    assert not token_path.exists()


def test_resolve_api_auth_token_default_mints_and_persists(tmp_path: Path) -> None:
    """The startup-without-token path: no explicit token, no opt-out ->
    a real token is minted AND persisted to disk. This is the exact
    behavior that was documented but never implemented (polylogue-rzve);
    removing the mint call from ``resolve_api_auth_token`` makes this fail
    (the file would never be created and ``resolved`` would be ``None``)."""
    token_path = tmp_path / "api-auth-token"

    resolved = resolve_api_auth_token(None, token_path=token_path)

    assert resolved is not None
    assert token_path.exists()
    assert token_path.read_text(encoding="utf-8").strip() == resolved
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_resolve_api_auth_token_default_is_stable_across_calls(tmp_path: Path) -> None:
    """A second daemon start (or client resolving the same config) must
    load the same persisted token rather than minting a fresh one each
    time -- otherwise every restart would invalidate every existing
    client's credential."""
    token_path = tmp_path / "api-auth-token"

    first = resolve_api_auth_token(None, token_path=token_path)
    second = resolve_api_auth_token(None, token_path=token_path)

    assert first == second


def test_api_token_show_command_mints_and_prints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))

    result = CliRunner().invoke(api_command, ["token", "show"])

    assert result.exit_code == 0, (result.output, result.exception)
    token = result.output.strip()
    assert token
    assert api_auth_token_path().read_text(encoding="utf-8").strip() == token


def test_api_token_show_command_rotate_changes_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))

    first = CliRunner().invoke(api_command, ["token", "show"]).output.strip()
    rotated = CliRunner().invoke(api_command, ["token", "show", "--rotate"]).output.strip()

    assert rotated != first
