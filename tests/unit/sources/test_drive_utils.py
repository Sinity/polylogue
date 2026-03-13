"""Compact contracts for Drive utility parsing and path resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.sources import DriveAuthError
from polylogue.sources.drive_client import (
    DEFAULT_DRIVE_RETRIES,
    DEFAULT_DRIVE_RETRY_BASE,
    _build_folder_lookup_query,
    _looks_like_id,
    _parse_modified_time,
    _parse_size,
    _resolve_credentials_path,
    _resolve_retries,
    _resolve_retry_base,
    _resolve_token_path,
    default_credentials_path,
    default_token_path,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("not a date", None),
        ("12345", None),
        ("2024-13-45T99:99:99Z", None),
        ("2024-01-15T10:30:45Z", datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc).timestamp()),
        ("2024-01-15T10:30:45+00:00", datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc).timestamp()),
    ],
)
def test_parse_modified_time_contract(raw: str | None, expected: float | None) -> None:
    result = _parse_modified_time(raw)
    if expected is None:
        assert result is None
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (0, 0),
        (1024, 1024),
        (-1, -1),
        ("123", 123),
        ("  456  ", 456),
        ("0", 0),
        ("not a number", None),
        ("12.34", None),
        ("12a", None),
        ("", None),
    ],
)
def test_parse_size_contract(raw: str | int | None, expected: int | None) -> None:
    assert _parse_size(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", False),
        ("hello world", False),
        (" test", False),
        ("test ", False),
        ("file.txt", False),
        ("file@home", False),
        ("a/b", False),
        ("abc-123-def", True),
        ("file_1_test", True),
        ("abc123", True),
        ("123", True),
        ("a", True),
        ("---", True),
    ],
)
def test_looks_like_id_contract(value: str, expected: bool) -> None:
    assert _looks_like_id(value) is expected


@pytest.mark.parametrize(
    ("folder_ref", "expected"),
    [
        ("Inbox", "name = 'Inbox' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"),
        ("O'Hare Folder", "name = 'O\\'Hare Folder' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"),
    ],
)
def test_build_folder_lookup_query_contract(folder_ref: str, expected: str) -> None:
    assert _build_folder_lookup_query(folder_ref) == expected


@pytest.mark.parametrize(
    ("explicit", "config_value", "env_value", "expected"),
    [
        (5, None, None, 5),
        (0, None, None, 0),
        (-5, None, None, 0),
        (None, 7, None, 7),
        (None, -2, None, 0),
        (None, None, "9", 9),
        (None, None, "-3", 0),
        (None, None, "not_a_number", DEFAULT_DRIVE_RETRIES),
        (None, None, None, DEFAULT_DRIVE_RETRIES),
        (10, 5, "20", 10),
        (None, 5, "20", 5),
    ],
)
def test_resolve_retries_precedence_contract(
    monkeypatch: pytest.MonkeyPatch,
    explicit: int | None,
    config_value: int | None,
    env_value: str | None,
    expected: int,
) -> None:
    if env_value is None:
        monkeypatch.delenv("POLYLOGUE_DRIVE_RETRIES", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", env_value)
    config = None if config_value is None else MagicMock(retry_count=config_value)
    assert _resolve_retries(value=explicit, config=config) == expected


@pytest.mark.parametrize(
    ("explicit", "env_value", "expected"),
    [
        (1.5, None, 1.5),
        (0.1, None, 0.1),
        (-0.5, None, 0.0),
        (None, "2.5", 2.5),
        (None, "not_a_float", DEFAULT_DRIVE_RETRY_BASE),
        (None, None, DEFAULT_DRIVE_RETRY_BASE),
    ],
)
def test_resolve_retry_base_contract(
    monkeypatch: pytest.MonkeyPatch,
    explicit: float | None,
    env_value: str | None,
    expected: float,
) -> None:
    if env_value is None:
        monkeypatch.delenv("POLYLOGUE_DRIVE_RETRY_BASE", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRY_BASE", env_value)
    assert _resolve_retry_base(explicit) == expected


@pytest.mark.parametrize(
    ("attr_name", "default_fn", "configured", "expected"),
    [
        ("credentials_path", default_credentials_path, None, None),
        ("credentials_path", default_credentials_path, "/custom/creds.json", Path("/custom/creds.json")),
        ("token_path", default_token_path, None, None),
        ("token_path", default_token_path, "/custom/token.json", Path("/custom/token.json")),
    ],
)
def test_default_path_helpers_contract(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
    default_fn,
    configured: str | None,
    expected: Path | None,
) -> None:
    sentinel = Path(f"/tmp/sentinel-{attr_name}.json")
    patch_target = (
        "polylogue.sources.drive_client.drive_credentials_path"
        if attr_name == "credentials_path"
        else "polylogue.sources.drive_client.drive_token_path"
    )
    monkeypatch.setattr(patch_target, lambda: sentinel)
    config = None if configured is None else MagicMock(**{attr_name: configured})
    result = default_fn(config=config)
    assert result == (expected or sentinel)


@pytest.mark.parametrize(
    ("config_path", "env_path", "default_exists", "ui_plain", "ui_response", "expected_kind"),
    [
        ("/cfg/creds.json", None, False, True, None, "config"),
        (None, "~/creds.json", False, True, None, "env"),
        (None, None, True, True, None, "default"),
        (None, None, False, False, "user", "interactive"),
        (None, None, False, False, None, "error"),
        (None, None, False, True, None, "error"),
    ],
)
def test_resolve_credentials_path_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: str | None,
    env_path: str | None,
    default_exists: bool,
    ui_plain: bool,
    ui_response: str | None,
    expected_kind: str,
) -> None:
    default_path = tmp_path / "default" / "creds.json"
    if default_exists:
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text('{"default": true}', encoding="utf-8")
    user_path = tmp_path / "user" / "creds.json"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text('{"user": true}', encoding="utf-8")

    monkeypatch.setattr("polylogue.sources.drive_client.default_credentials_path", lambda config: default_path)
    if env_path is None:
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", env_path)

    config = None if config_path is None else MagicMock(credentials_path=config_path)
    ui = MagicMock()
    ui.plain = ui_plain
    ui.input.return_value = str(user_path) if ui_response == "user" else ui_response

    if expected_kind == "error":
        with pytest.raises(DriveAuthError, match="credentials"):
            _resolve_credentials_path(ui=None if ui_plain else ui, config=config)
        return

    result = _resolve_credentials_path(ui=None if ui_plain else ui, config=config)
    if expected_kind == "config":
        assert result == Path(config_path)
    elif expected_kind == "env":
        assert result == Path(env_path).expanduser()
    elif expected_kind == "default":
        assert result == default_path
    else:
        assert result == default_path
        assert default_path.read_text(encoding="utf-8") == '{"user": true}'
        ui.input.assert_called_once()


@pytest.mark.parametrize(
    ("config_path", "env_path", "expected"),
    [
        ("/cfg/token.json", None, Path("/cfg/token.json")),
        (None, "~/token.json", Path("~/token.json").expanduser()),
        (None, None, Path("/tmp/default-token.json")),
    ],
)
def test_resolve_token_path_contract(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str | None,
    env_path: str | None,
    expected: Path,
) -> None:
    monkeypatch.setattr("polylogue.sources.drive_client.default_token_path", lambda config: Path("/tmp/default-token.json"))
    if env_path is None:
        monkeypatch.delenv("POLYLOGUE_TOKEN_PATH", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", env_path)
    config = None if config_path is None else MagicMock(token_path=config_path)
    assert _resolve_token_path(config=config) == expected
