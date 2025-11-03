from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.drive_client import DRIVE_CREDENTIAL_ENV, DriveClient


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


def test_drive_client_uses_env_credentials(temp_credentials, tmp_path, monkeypatch):
    client = DriveClient(DummyUI())
    target_path = tmp_path / "stored.json"
    monkeypatch.setattr(client, "_credentials_path", target_path, raising=False)

    resolved = client.ensure_credentials()
    assert resolved == target_path
    assert target_path.exists()
    assert json.loads(target_path.read_text(encoding="utf-8")) == {"installed": {"client_id": "demo"}}
