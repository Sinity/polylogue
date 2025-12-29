from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import polylogue.drive as drive_module
from polylogue.drive_client import DriveClient
from polylogue.drive import _run_console_flow, get_drive_service, snapshot_drive_metrics


class DummyUI:
    def __init__(self):
        self.plain = True
        self.console = self

    def print(self, *args, **kwargs):  # pragma: no cover - diagnostics not needed
        pass

    def banner(self, *args, **kwargs):  # pragma: no cover - plain mode skips banners
        pass


@pytest.fixture(autouse=True)
def _reset_drive_metrics():
    snapshot_drive_metrics(reset=True)
    yield
    snapshot_drive_metrics(reset=True)


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    monkeypatch.setattr("polylogue.drive.time.sleep", lambda _t: None)


def test_drive_client_uses_existing_credentials(tmp_path, monkeypatch):
    client = DriveClient(DummyUI())
    target_path = tmp_path / "stored.json"
    target_path.write_text(json.dumps({"installed": {"client_id": "demo"}}), encoding="utf-8")
    monkeypatch.setattr(client, "_credentials_path", target_path, raising=False)

    resolved = client.ensure_credentials()
    assert resolved == target_path
    assert target_path.exists()
    assert json.loads(target_path.read_text(encoding="utf-8")) == {"installed": {"client_id": "demo"}}


def test_drive_client_respects_env_path(tmp_path, monkeypatch):
    target_path = tmp_path / "env-creds" / "credentials.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps({"installed": {"client_id": "env-demo"}}), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(target_path))
    monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(tmp_path / "env-creds" / "token.json"))

    client = DriveClient(DummyUI())

    assert client.credentials_path == target_path
    assert client.ensure_credentials() == target_path
    assert client.token_path == tmp_path / "env-creds" / "token.json"


def test_drive_client_explicit_paths_override_env(tmp_path, monkeypatch):
    env_creds = tmp_path / "env-creds" / "credentials.json"
    env_creds.parent.mkdir(parents=True, exist_ok=True)
    env_creds.write_text(json.dumps({"installed": {"client_id": "env"}}), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(env_creds))
    monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(tmp_path / "env-creds" / "token.json"))

    explicit_creds = tmp_path / "explicit" / "credentials.json"
    explicit_token = tmp_path / "explicit" / "token.json"
    explicit_creds.parent.mkdir(parents=True, exist_ok=True)
    explicit_creds.write_text(json.dumps({"installed": {"client_id": "explicit"}}), encoding="utf-8")
    explicit_token.write_text(json.dumps({"token": "explicit"}), encoding="utf-8")

    client = DriveClient(DummyUI(), credentials_path=explicit_creds, token_path=explicit_token)

    assert client.credentials_path == explicit_creds
    assert client.token_path == explicit_token


def test_drive_client_falls_back_from_legacy_folder_name(monkeypatch):
    client = DriveClient(DummyUI())
    monkeypatch.setattr(client, "service", lambda: object())
    seen: list[str] = []

    def fake_find(_svc, name, notifier=None):
        seen.append(name)
        if name == "Google AI Studio":
            return "folder-123"
        return None

    monkeypatch.setattr("polylogue.drive_client.find_folder_id", fake_find)

    resolved = client.resolve_folder_id("AI Studio", None)

    assert resolved == "folder-123"
    assert seen == ["AI Studio", "Google AI Studio"]


def test_run_console_flow_assigns_redirect(monkeypatch):
    class DummySession:
        def __init__(self):
            self.redirect_uri = None

        def authorization_url(self, auth_uri, **kwargs):
            assert self.redirect_uri == "http://localhost"
            assert kwargs.get("redirect_uri") == "http://localhost"
            return "http://auth.local", "state-token"

    class DummyFlow:
        def __init__(self):
            self.client_config = {"redirect_uris": ["http://localhost"]}
            self.redirect_uri = None
            self.oauth2session = DummySession()
            self.credentials = SimpleNamespace(token=None)

        def authorization_url(self, **kwargs):
            return self.oauth2session.authorization_url("https://accounts.example", **kwargs)

        def fetch_token(self, code):
            assert code == "code-123"
            self.credentials = SimpleNamespace(token="new-token")

    monkeypatch.setattr("builtins.input", lambda _prompt="": "code-123")
    monkeypatch.setattr("polylogue.drive.colorize", lambda text, _color: text)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    flow = DummyFlow()
    creds = _run_console_flow(flow, verbose=False)

    assert flow.redirect_uri == "http://localhost"
    assert flow.oauth2session.redirect_uri == "http://localhost"
    assert creds.token == "new-token"


def test_get_drive_service_missing_credentials(monkeypatch, tmp_path, capsys):
    import polylogue.drive as drive

    monkeypatch.setattr(drive, "HAS_GOOGLE", True)
    cred_path = tmp_path / "missing_client.json"

    session = get_drive_service(cred_path, verbose=False)

    assert session is None
    captured = capsys.readouterr().out
    assert "Missing credentials" in captured


def test_plain_mode_credentials_error_mentions_env(monkeypatch, tmp_path):
    client = DriveClient(DummyUI())
    target_path = tmp_path / "creds" / "credentials.json"
    monkeypatch.setattr(client, "_credentials_path", target_path, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        client.ensure_credentials()

    message = str(excinfo.value)
    assert "POLYLOGUE_CREDENTIAL_PATH" in message
    assert str(target_path) in message


def test_get_drive_service_with_cached_token(monkeypatch, tmp_path):
    import json
    import polylogue.drive as drive

    class DummyCreds:
        def __init__(self):
            self.valid = True

    class DummyCredentials:
        @staticmethod
        def from_authorized_user_file(path, scopes):  # noqa: ARG002
            return DummyCreds()

    cred_path = tmp_path / "client.json"
    cred_path.write_text("{}", encoding="utf-8")
    token_path = cred_path.parent / "token.json"
    token_path.write_text(json.dumps({"token": "abc"}), encoding="utf-8")

    monkeypatch.setattr(drive, "HAS_GOOGLE", True)
    monkeypatch.setattr(drive, "Credentials", DummyCredentials)
    monkeypatch.setattr(drive, "_authorized_session", lambda creds: ("session", creds))

    session = get_drive_service(cred_path, verbose=False)

    assert isinstance(session, tuple)
    assert session[0] == "session"
    assert isinstance(session[1], DummyCreds)


def test_drive_metrics_records_retries(monkeypatch):
    class DummyResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

        def close(self):  # pragma: no cover - compatibility
            pass

    class DummySession:
        def __init__(self):
            self.calls = 0

        def get(self, _url, params=None, timeout=0, stream=False):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return DummyResp(500, {"error": {"message": "fail"}})
            return DummyResp(200, {"files": []})

    drive_module.list_children(DummySession(), "folder")
    stats = snapshot_drive_metrics(reset=True)
    assert stats["requests"] == 1
    assert stats["retries"] == 1
    assert stats["failures"] == 0
    assert stats["operations"]["metadata"]["retries"] == 1


def test_drive_metrics_records_failures(monkeypatch):
    class DummyResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

        def iter_content(self, chunk_size=1024):  # pragma: no cover - unused
            yield from ()

        def close(self):  # pragma: no cover
            pass

    class DummySession:
        def get(self, *_args, **_kwargs):
            return DummyResp(500, {"error": {"message": "fail"}})

    data = drive_module.download_file(DummySession(), "file-1")
    assert data is None
    stats = snapshot_drive_metrics(reset=True)
    assert stats["requests"] == 1
    assert stats["failures"] == 1
    assert stats["retries"] >= 1
    assert stats["lastError"]
    assert stats["operations"]["download"]["failures"] == 1
