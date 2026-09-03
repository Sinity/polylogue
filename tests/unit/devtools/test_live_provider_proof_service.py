from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools import live_provider_proof_service


class _FakeServer:
    server_address = ("127.0.0.1", 49120)

    def serve_forever(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def server_close(self) -> None:
        return None


class _FakeThread:
    def __init__(self, **_kwargs: object) -> None:
        return None

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        del timeout


def test_live_provider_timeout_terminates_group_and_becomes_typed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(live_provider_proof_service, "require_declared_operation_context", lambda _operation: "unit")
    bound: list[object] = []

    def fake_make_server(_host: str, port: int, **_kwargs: object) -> _FakeServer:
        bound.append(port)
        return _FakeServer()

    monkeypatch.setattr(live_provider_proof_service, "make_server", fake_make_server)
    monkeypatch.setattr(live_provider_proof_service, "Thread", _FakeThread)
    process = SimpleNamespace(
        communicate=lambda **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(["node"], 120)),
        returncode=None,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    terminated: list[object] = []
    monkeypatch.setattr(live_provider_proof_service, "terminate_process_group", terminated.append)

    assert live_provider_proof_service.main(["--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "TimeoutExpired"
    assert "120" in payload["error"]["message"]
    assert terminated == [process, process]
    assert bound == [0]
