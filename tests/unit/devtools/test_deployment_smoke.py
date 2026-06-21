from __future__ import annotations

import json
import subprocess
import urllib.error
from email.message import Message

import pytest

from devtools import deployment_smoke


class _FakeResponse:
    def __init__(self, status: int, payload: dict[str, object]) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_deployment_smoke_json_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(deployment_smoke, "_resolve_command", lambda name, *, path: f"/bin/{name}")
    monkeypatch.setattr(deployment_smoke, "_repo_head", lambda: "abc123def456")

    def fake_run(
        command: list[str],
        *,
        path: str,
        timeout_s: float,
    ) -> subprocess.CompletedProcess[str]:
        del path, timeout_s
        if command == ["polylogued", "--version"]:
            return subprocess.CompletedProcess(command, 2, "", "no such option")
        return subprocess.CompletedProcess(command, 0, "polylogue, version test", "")

    def fake_open_url(url: str, *, timeout_s: float) -> _FakeResponse:
        del timeout_s
        if url.endswith("/api/read-view-profiles"):
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return _FakeResponse(200, {"ok": True})

    monkeypatch.setattr(deployment_smoke, "_run_command", fake_run)
    monkeypatch.setattr(deployment_smoke, "_open_url", fake_open_url)

    exit_code = deployment_smoke.main(["--json", "--path", "/bin", "--timeout-s", "1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["repo_head"] == "abc123def456"
    assert "command:polylogued --version:2" in payload["failures"]
    assert any("read-view-profiles" in failure for failure in payload["failures"])
    assert "deployed polylogued predates the --version option" in payload["diagnostics"]["likely_causes"]
    assert any("read-view profiles route" in action for action in payload["diagnostics"]["next_actions"])


def test_deployment_smoke_command_succeeds_when_all_probes_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(deployment_smoke, "_resolve_command", lambda name, *, path: f"/bin/{name}")
    monkeypatch.setattr(deployment_smoke, "_repo_head", lambda: "abc123def456")
    monkeypatch.setattr(
        deployment_smoke,
        "_run_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "version ok", ""),
    )
    monkeypatch.setattr(
        deployment_smoke,
        "_open_url",
        lambda url, *, timeout_s: _FakeResponse(200, {"ok": True, "url": url}),
    )

    exit_code = deployment_smoke.main(["--path", "/bin", "--timeout-s", "1"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Polylogue deployment smoke" in output
    assert "repo HEAD: abc123def456" in output
    assert "status: ok" in output
