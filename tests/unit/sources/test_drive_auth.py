"""Contract tests for DriveAuthManager — credential lifecycle and path resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from polylogue.sources.drive_auth import (
    DriveAuthManager,
    DriveAuthPrompter,
    UIAuthPrompter,
    _resolve_credentials_path,
    _resolve_token_path,
    default_credentials_path,
    default_token_path,
)
from polylogue.sources.drive_types import DriveAuthError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _creds(
    *,
    valid: bool,
    expired: bool,
    refresh_token: str | None = "refresh-token",
    token_json: str = '{"token":"fresh"}',
):
    creds = MagicMock()
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.to_json.return_value = token_json
    return creds


class FakeAuthPrompter:
    """Minimal DriveAuthPrompter implementation for tests."""

    def __init__(self, code: str | None) -> None:
        self.code = code
        self.announced_urls: list[str] = []

    def announce_auth_url(self, url: str) -> None:
        self.announced_urls.append(url)

    def request_authorization_code(self) -> str | None:
        return self.code or None


def _auth_manager_with_prompter(
    prompter: DriveAuthPrompter | None,
    *,
    token_path: Path | None = None,
    credentials_path: Path | None = None,
) -> DriveAuthManager:
    mgr = DriveAuthManager(token_path=token_path, credentials_path=credentials_path)
    mgr._prompter = prompter
    return mgr


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


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
        "polylogue.sources.drive_auth.drive_credentials_path"
        if attr_name == "credentials_path"
        else "polylogue.sources.drive_auth.drive_token_path"
    )
    monkeypatch.setattr(patch_target, lambda: sentinel)
    config = None if configured is None else MagicMock(**{attr_name: configured})
    result = default_fn(config=config)
    assert result == (expected or sentinel)


@pytest.mark.parametrize(
    ("config_path", "env_path", "default_exists", "has_prompter", "prompter_response", "expected_kind"),
    [
        ("/cfg/creds.json", None, False, False, None, "config"),
        (None, "~/creds.json", False, False, None, "env"),
        (None, None, True, False, None, "default"),
        (None, None, False, True, "user", "interactive"),
        (None, None, False, True, None, "error"),
        (None, None, False, False, None, "error"),
    ],
)
def test_resolve_credentials_path_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: str | None,
    env_path: str | None,
    default_exists: bool,
    has_prompter: bool,
    prompter_response: str | None,
    expected_kind: str,
) -> None:
    default_path = tmp_path / "default" / "creds.json"
    if default_exists:
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text('{"default": true}', encoding="utf-8")
    user_path = tmp_path / "user" / "creds.json"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text('{"user": true}', encoding="utf-8")

    monkeypatch.setattr("polylogue.sources.drive_auth.default_credentials_path", lambda config: default_path)
    if env_path is None:
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", env_path)

    config = None if config_path is None else MagicMock(credentials_path=config_path)
    ui = None
    if has_prompter:
        ui = MagicMock()
        ui.plain = False
        ui.input.return_value = str(user_path) if prompter_response == "user" else prompter_response

    if expected_kind == "error":
        with pytest.raises(DriveAuthError, match="credentials"):
            _resolve_credentials_path(ui=ui, config=config)
        return

    result = _resolve_credentials_path(ui=ui, config=config)
    if expected_kind == "config":
        assert result == Path(config_path)
    elif expected_kind == "env":
        assert result == Path(env_path).expanduser()
    elif expected_kind == "default":
        assert result == default_path
    else:
        assert result == default_path
        assert default_path.read_text(encoding="utf-8") == '{"user": true}'


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
    monkeypatch.setattr(
        "polylogue.sources.drive_auth.default_token_path", lambda config: Path("/tmp/default-token.json")
    )
    if env_path is None:
        monkeypatch.delenv("POLYLOGUE_TOKEN_PATH", raising=False)
    else:
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", env_path)
    config = None if config_path is None else MagicMock(token_path=config_path)
    assert _resolve_token_path(config=config) == expected


# ---------------------------------------------------------------------------
# Manual auth flow contracts
# ---------------------------------------------------------------------------


class _StubFlow:
    def __init__(self, *, credentials: object, fetch_error: Exception | None = None) -> None:
        self.credentials = credentials
        self.fetch_error = fetch_error
        self.fetch_codes: list[str] = []
        self.authorization_calls: list[tuple[str, str]] = []

    def authorization_url(self, *, prompt: str, access_type: str) -> tuple[str, None]:
        self.authorization_calls.append((prompt, access_type))
        return ("https://accounts.example/authorize", None)

    def fetch_token(self, *, code: str) -> None:
        self.fetch_codes.append(code)
        if self.fetch_error is not None:
            raise self.fetch_error


@pytest.mark.parametrize(
    ("code", "fetch_error", "expected_message"),
    [
        ("auth-code-123", None, None),
        (None, None, "Drive authorization cancelled"),
        ("bad-code", RuntimeError("bad authorization code"), "Drive authorization failed"),
    ],
    ids=["success", "cancelled", "fetch-error"],
)
def test_run_manual_auth_flow_contract(
    tmp_path: Path,
    code: str | None,
    fetch_error: Exception | None,
    expected_message: str | None,
) -> None:
    creds = object()
    flow = _StubFlow(credentials=creds, fetch_error=fetch_error)
    prompter = FakeAuthPrompter(code)
    mgr = _auth_manager_with_prompter(prompter, token_path=tmp_path / "token.json")

    if expected_message is None:
        result = mgr._run_manual_auth_flow(flow)
        assert result is creds
        assert flow.fetch_codes == ["auth-code-123"]
    else:
        with pytest.raises(DriveAuthError, match=expected_message):
            mgr._run_manual_auth_flow(flow)

    assert flow.authorization_calls == [("consent", "offline")]
    assert "https://accounts.example/authorize" in prompter.announced_urls


def test_run_manual_auth_flow_no_prompter_cancels(tmp_path: Path) -> None:
    flow = _StubFlow(credentials=object())
    mgr = _auth_manager_with_prompter(None, token_path=tmp_path / "token.json")

    with pytest.raises(DriveAuthError, match="Drive authorization cancelled"):
        mgr._run_manual_auth_flow(flow)


# ---------------------------------------------------------------------------
# UIAuthPrompter adapter
# ---------------------------------------------------------------------------


def test_ui_auth_prompter_announces_url() -> None:
    ui = MagicMock()
    ui.console = MagicMock()
    prompter = UIAuthPrompter(ui)
    prompter.announce_auth_url("https://auth.example/flow")
    ui.console.print.assert_any_call("https://auth.example/flow")


def test_ui_auth_prompter_silent_without_console() -> None:
    ui = MagicMock()
    ui.console = None
    prompter = UIAuthPrompter(ui)
    # Should not raise
    prompter.announce_auth_url("https://auth.example/flow")


def test_ui_auth_prompter_requests_code() -> None:
    ui = MagicMock()
    ui.input = MagicMock(return_value="my-code")
    prompter = UIAuthPrompter(ui)
    assert prompter.request_authorization_code() == "my-code"


def test_ui_auth_prompter_returns_none_for_empty_code() -> None:
    ui = MagicMock()
    ui.input = MagicMock(return_value="")
    prompter = UIAuthPrompter(ui)
    assert prompter.request_authorization_code() is None


# ---------------------------------------------------------------------------
# Credential loading state machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthLoadCase:
    name: str
    token_store_value: str | None
    token_file_value: str | None
    info_side_effect: object | None = None
    file_side_effect: object | None = None
    creds_factory: object | None = None
    expect_message: str | None = None
    refreshes: bool = False


AUTH_LOAD_CASES = [
    AuthLoadCase(
        name="token-store preferred",
        token_store_value='{"token":"from-store"}',
        token_file_value='{"token":"stale"}',
        creds_factory=lambda: _creds(valid=True, expired=False),
    ),
    AuthLoadCase(
        name="invalid store falls back to file",
        token_store_value="not-json",
        token_file_value='{"token":"from-file"}',
        info_side_effect=json.JSONDecodeError("bad", "x", 0),
        creds_factory=lambda: _creds(valid=True, expired=False, token_json='{"token":"from-file"}'),
    ),
    AuthLoadCase(
        name="refresh expired token",
        token_store_value='{"token":"stale"}',
        token_file_value='{"token":"stale"}',
        creds_factory=lambda: _creds(valid=False, expired=True),
        refreshes=True,
    ),
    AuthLoadCase(
        name="invalid non-refreshable token fails",
        token_store_value=None,
        token_file_value='{"token":"stale"}',
        creds_factory=lambda: _creds(valid=False, expired=True, refresh_token=None),
        expect_message="cannot be refreshed",
    ),
    AuthLoadCase(
        name="corrupt token file fails in plain mode",
        token_store_value=None,
        token_file_value="not json",
        file_side_effect=ValueError("bad token json"),
        expect_message="invalid or expired",
    ),
]


@pytest.mark.parametrize("case", AUTH_LOAD_CASES, ids=lambda case: case.name)
def test_load_credentials_state_machine(case: AuthLoadCase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    if case.token_file_value is not None:
        token_path.write_text(case.token_file_value, encoding="utf-8")

    credentials_cls = MagicMock()
    creds = case.creds_factory() if case.creds_factory is not None else None
    if case.info_side_effect is not None:
        credentials_cls.from_authorized_user_info.side_effect = case.info_side_effect
    elif creds is not None:
        credentials_cls.from_authorized_user_info.return_value = creds
    if case.file_side_effect is not None:
        credentials_cls.from_authorized_user_file.side_effect = case.file_side_effect
    elif creds is not None:
        credentials_cls.from_authorized_user_file.return_value = creds

    request_cls = MagicMock(return_value=SimpleNamespace(session=SimpleNamespace(timeout=None)))

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        if name == "google.auth.transport.requests":
            return SimpleNamespace(Request=request_cls)
        raise AssertionError(name)

    mgr = DriveAuthManager(ui=None, token_path=token_path)
    mgr._token_store = MagicMock()
    mgr._token_store.load.return_value = case.token_store_value
    monkeypatch.setattr("polylogue.sources.drive_auth._import_auth_module", fake_import)

    if case.refreshes and creds is not None:

        def refresh(_request) -> None:
            creds.valid = True
            creds.expired = False

        creds.refresh.side_effect = refresh

    if case.expect_message is not None:
        with pytest.raises(DriveAuthError, match=case.expect_message):
            mgr.load_credentials()
        return

    result = mgr.load_credentials()
    assert result is creds
    mgr._token_store.save.assert_called_once_with("drive_token", creds.to_json())
    if case.refreshes and creds is not None:
        creds.refresh.assert_called_once()


def test_load_cached_credentials_prefers_store_then_file(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token":"from-file"}', encoding="utf-8")
    credentials_cls = MagicMock()
    from_store = _creds(valid=True, expired=False, token_json='{"token":"from-store"}')
    credentials_cls.from_authorized_user_info.return_value = from_store
    mgr = DriveAuthManager(ui=None, token_path=token_path)
    mgr._token_store = MagicMock()
    mgr._token_store.load.return_value = '{"token":"from-store"}'

    state = mgr._load_cached_credentials(credentials_cls, token_path)

    assert state.creds is from_store
    assert state.had_invalid_token_path is False
    credentials_cls.from_authorized_user_info.assert_called_once()
    credentials_cls.from_authorized_user_file.assert_not_called()


def test_refresh_credentials_if_needed_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = tmp_path / "token.json"
    mgr = DriveAuthManager(ui=None, token_path=token_path)
    mgr._token_store = MagicMock()
    creds = _creds(valid=False, expired=True, token_json='{"token":"fresh"}')
    request = SimpleNamespace(session=SimpleNamespace(timeout=None))

    def refresh(_request) -> None:
        creds.valid = True
        creds.expired = False

    creds.refresh.side_effect = refresh
    monkeypatch.setattr(
        "polylogue.sources.drive_auth._import_auth_module",
        lambda name: (
            SimpleNamespace(Request=lambda: request)
            if name == "google.auth.transport.requests"
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )

    result = mgr._refresh_credentials_if_needed(creds, token_path)

    assert result is creds
    creds.refresh.assert_called_once()
    mgr._token_store.save.assert_called_once_with("drive_token", '{"token":"fresh"}')


def test_load_credentials_uses_manual_flow_when_local_server_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    credentials_path = tmp_path / "client.json"
    credentials_path.write_text('{"installed":{}}', encoding="utf-8")
    token_path = tmp_path / "token.json"
    flow = MagicMock()
    flow.run_local_server.side_effect = OSError("port unavailable")
    installed_app_flow_cls = MagicMock()
    installed_app_flow_cls.from_client_secrets_file.return_value = flow
    manual_creds = _creds(valid=True, expired=False, token_json='{"token":"manual"}')

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=MagicMock())
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=installed_app_flow_cls)
        raise AssertionError(name)

    mgr = DriveAuthManager(credentials_path=credentials_path, token_path=token_path)
    mgr._prompter = FakeAuthPrompter("manual-code")
    mgr._token_store = MagicMock()
    mgr._token_store.load.return_value = None
    mgr._run_manual_auth_flow = MagicMock(return_value=manual_creds)
    monkeypatch.setattr("polylogue.sources.drive_auth._import_auth_module", fake_import)

    result = mgr.load_credentials()

    assert result is manual_creds
    mgr._run_manual_auth_flow.assert_called_once_with(flow)
    mgr._token_store.save.assert_called_once_with("drive_token", '{"token":"manual"}')


def test_load_credentials_returns_local_server_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    credentials_path = tmp_path / "client.json"
    credentials_path.write_text('{"installed":{}}', encoding="utf-8")
    token_path = tmp_path / "token.json"
    server_creds = _creds(valid=True, expired=False, token_json='{"token":"server"}')
    flow = MagicMock()
    flow.run_local_server.return_value = server_creds
    installed_app_flow_cls = MagicMock()
    installed_app_flow_cls.from_client_secrets_file.return_value = flow

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=MagicMock())
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=installed_app_flow_cls)
        raise AssertionError(name)

    mgr = DriveAuthManager(credentials_path=credentials_path, token_path=token_path)
    mgr._prompter = FakeAuthPrompter("should-not-be-used")
    mgr._token_store = MagicMock()
    mgr._token_store.load.return_value = None
    mgr._run_manual_auth_flow = MagicMock(side_effect=AssertionError("manual flow should not run"))
    monkeypatch.setattr("polylogue.sources.drive_auth._import_auth_module", fake_import)

    result = mgr.load_credentials()

    assert result is server_creds
    flow.run_local_server.assert_called_once_with(open_browser=False, port=0)
    mgr._token_store.save.assert_called_once_with("drive_token", '{"token":"server"}')


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------


def test_revoke_deletes_token_store_and_file(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token":"old"}', encoding="utf-8")
    mgr = DriveAuthManager(token_path=token_path)
    mgr._token_store = MagicMock()

    mgr.revoke()

    mgr._token_store.delete.assert_called_once_with("drive_token")
    assert not token_path.exists()


def test_revoke_without_token_file_is_harmless(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    mgr = DriveAuthManager(token_path=token_path)
    mgr._token_store = MagicMock()

    mgr.revoke()

    mgr._token_store.delete.assert_called_once_with("drive_token")
    assert not token_path.exists()
