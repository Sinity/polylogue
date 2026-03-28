from __future__ import annotations

import importlib

import pytest

import polylogue.drive_client as drive_client
from polylogue.drive_client import DriveAuthError, DriveClient


def test_drive_client_reports_missing_dependency(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name: str):
        if name.startswith("googleapiclient"):
            raise ModuleNotFoundError(name)
        return real_import(name)

    client = DriveClient(ui=None)
    monkeypatch.setattr(client, "_load_credentials", lambda: object())
    monkeypatch.setattr(drive_client.importlib, "import_module", fake_import)

    with pytest.raises(DriveAuthError, match="Drive dependencies"):
        client._service_handle()
