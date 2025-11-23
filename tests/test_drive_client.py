from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import polylogue.drive as drive_module
from polylogue.drive_client import DRIVE_CREDENTIAL_ENV, DriveClient
from polylogue.drive import _run_console_flow, get_drive_service, snapshot_drive_metrics


class DummyUI:
    def __init__(self):
        self.plain = True
        self.console = self

    def print(self, *args, **kwargs):  # pragma: no cover - diagnostics not needed
        pass

    def banner(self, *args, **kwargs):  # pragma: no cover - plain mode skips banners
        pass


@pytest.fixture
def temp_credentials(tmp_path, monkeypatch):
    cred_path = tmp_path / "env-creds.json"
    cred_path.write_text(json.dumps({"installed": {"client_id": "demo"}}), encoding="utf-8")
    monkeypatch.setenv(DRIVE_CREDENTIAL_ENV, str(cred_path))
    return cred_path


@pytest.fixture(autouse=True)
def _reset_drive_metrics():
    snapshot_drive_metrics(reset=True)
    yield
    snapshot_drive_metrics(reset=True)


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    monkeypatch.setattr("polylogue.drive.time.sleep", lambda _t: None)


def test_drive_client_uses_env_credentials(temp_credentials, tmp_path, monkeypatch):
    client = DriveClient(DummyUI())
    target_path = tmp_path / "stored.json"
    monkeypatch.setattr(client, "_credentials_path", target_path, raising=False)

    resolved = client.ensure_credentials()
    assert resolved == target_path
    assert target_path.exists()
    assert json.loads(target_path.read_text(encoding="utf-8")) == {"installed": {"client_id": "demo"}}


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

    token_path = tmp_path / "token.json"
    token_path.write_text(json.dumps({"token": "abc"}), encoding="utf-8")
    cred_path = tmp_path / "client.json"
    cred_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))
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
