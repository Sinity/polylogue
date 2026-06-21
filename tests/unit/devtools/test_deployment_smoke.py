from __future__ import annotations

import json
import subprocess
import urllib.error
from email.message import Message
from pathlib import Path

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
    tmp_path: Path,
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

    def fake_completion(
        *,
        path: str,
        timeout_s: float,
        comp_words: str,
    ) -> subprocess.CompletedProcess[str]:
        del path, timeout_s
        if comp_words == "polylogue find id:abc t":
            stdout = "plain,then\n"
        elif comp_words == "polylogue find id:abc then s":
            stdout = "plain,select\n"
        elif comp_words == "polylogue find id:abc then read --view ":
            stdout = "plain,summary\nplain,messages\nplain,raw\nplain,context-pack\n"
        else:
            stdout = "plain,json\nplain,ndjson\nplain,text\n"
        return subprocess.CompletedProcess(["polylogue"], 0, stdout, "")

    def fake_open_url(url: str, *, timeout_s: float) -> _FakeResponse:
        del timeout_s
        if url.endswith("/api/read-view-profiles"):
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return _FakeResponse(200, {"ok": True})

    monkeypatch.setattr(deployment_smoke, "_run_command", fake_run)
    monkeypatch.setattr(deployment_smoke, "_run_completion_command", fake_completion)
    monkeypatch.setattr(deployment_smoke, "_open_url", fake_open_url)

    exit_code = deployment_smoke.main(["--json", "--path", "/bin", "--archive-root", str(tmp_path), "--timeout-s", "1"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["repo_head"] == "abc123def456"
    assert "command:polylogued --version:2" in payload["failures"]
    assert any("read-view-profiles" in failure for failure in payload["failures"])
    assert payload["completions"][0]["ok"] is True
    assert "deployed polylogued predates the --version option" in payload["diagnostics"]["likely_causes"]
    assert any("read-view profiles route" in action for action in payload["diagnostics"]["next_actions"])


def test_deployment_smoke_command_succeeds_when_all_probes_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
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
        "_run_completion_command",
        lambda **kwargs: subprocess.CompletedProcess(
            ["polylogue"],
            0,
            {
                "polylogue find id:abc t": "plain,then\n",
                "polylogue find id:abc then s": "plain,select\n",
                "polylogue find id:abc then read --view ": (
                    "plain,summary\nplain,messages\nplain,raw\nplain,context-pack\n"
                ),
                "polylogue find id:abc then read --view messages --format ": "plain,json\nplain,ndjson\nplain,text\n",
            }[str(kwargs["comp_words"])],
            "",
        ),
    )
    monkeypatch.setattr(
        deployment_smoke,
        "_open_url",
        lambda url, *, timeout_s: _FakeResponse(200, {"ok": True, "url": url}),
    )

    exit_code = deployment_smoke.main(["--path", "/bin", "--archive-root", str(tmp_path), "--timeout-s", "1"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Polylogue deployment smoke" in output
    assert "repo HEAD: abc123def456" in output
    assert "status: ok" in output
    assert "Completions:" in output
    assert "Browser capture archive:" in output


def test_deployment_smoke_reports_missing_completion_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        "_run_completion_command",
        lambda **_kwargs: subprocess.CompletedProcess(["polylogue"], 0, "plain,wrong\n", ""),
    )
    monkeypatch.setattr(
        deployment_smoke,
        "_open_url",
        lambda url, *, timeout_s: _FakeResponse(200, {"ok": True, "url": url}),
    )

    report = deployment_smoke.build_report(
        path="/bin",
        daemon_base_url="http://daemon",
        receiver_base_url="http://receiver",
        archive_root=tmp_path,
        timeout_s=1,
    )

    assert report.ok is False
    assert "completion:query-then-connector:missing=then" in report.failures
    assert report.completions[0].missing == ["then"]


def test_deployment_smoke_reports_facets_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        "_run_completion_command",
        lambda **kwargs: subprocess.CompletedProcess(
            ["polylogue"],
            0,
            {
                "polylogue find id:abc t": "plain,then\n",
                "polylogue find id:abc then s": "plain,select\n",
                "polylogue find id:abc then read --view ": (
                    "plain,summary\nplain,messages\nplain,raw\nplain,context-pack\n"
                ),
                "polylogue find id:abc then read --view messages --format ": "plain,json\nplain,ndjson\nplain,text\n",
            }[str(kwargs["comp_words"])],
            "",
        ),
    )

    def fake_open_url(url: str, *, timeout_s: float) -> _FakeResponse:
        del timeout_s
        if url.endswith("/api/facets"):
            raise TimeoutError("facets exceeded budget")
        return _FakeResponse(200, {"ok": True, "url": url})

    monkeypatch.setattr(deployment_smoke, "_open_url", fake_open_url)

    report = deployment_smoke.build_report(
        path="/bin",
        daemon_base_url="http://daemon",
        receiver_base_url="http://receiver",
        archive_root=tmp_path,
        timeout_s=1,
    )

    assert report.ok is False
    assert any(failure.startswith("route:http://daemon/api/facets:") for failure in report.failures)
    assert "web-shell facets route exceeds the deployed smoke timeout" in report.diagnostics["likely_causes"]
    assert any("facet aggregation" in action for action in report.diagnostics["next_actions"])


def test_deployment_smoke_reports_spooled_browser_capture_without_raw_rows(tmp_path: Path) -> None:
    capture_dir = tmp_path / "browser-capture" / "chatgpt"
    capture_dir.mkdir(parents=True)
    (capture_dir / "capture.json").write_text("{}", encoding="utf-8")

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.spooled_count == 1
    assert probe.raw_rows is None
    assert probe.error == "source_db_missing"
