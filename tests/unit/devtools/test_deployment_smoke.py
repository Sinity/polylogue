from __future__ import annotations

import json
import sqlite3
import stat
import subprocess
import urllib.error
from email.message import Message
from pathlib import Path
from typing import cast

import pytest

from devtools import deployment_smoke


def _create_browser_source_db(
    path: Path,
    *,
    file_mtime_ms: int | None = None,
    raw_id: str = "raw-capture",
    native_id: str = "capture",
    capture_path: Path | None = None,
) -> None:
    with sqlite3.connect(path / "source.db") as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                file_mtime_ms INTEGER
            )
            """
        )
        if file_mtime_ms is not None:
            conn.execute(
                """
                INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, file_mtime_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    raw_id,
                    "chatgpt-export",
                    native_id,
                    str(capture_path or path / "browser-capture" / "chatgpt" / "capture.json"),
                    file_mtime_ms,
                ),
            )


def _write_spooled_capture(path: Path, *, provider_session_id: str = "capture") -> Path:
    capture_dir = path / "browser-capture" / "chatgpt"
    capture_dir.mkdir(parents=True)
    capture = capture_dir / f"{provider_session_id}.json"
    capture.write_text(
        json.dumps(
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "provenance": {
                    "source_url": f"https://chatgpt.com/c/{provider_session_id}",
                    "page_title": "Deployment smoke capture",
                    "captured_at": "2026-06-21T20:10:00+00:00",
                    "adapter_name": "deployment-smoke-test",
                },
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": provider_session_id,
                    "title": "Deployment smoke capture",
                    "turns": [{"provider_turn_id": "u1", "role": "user", "text": "probe"}],
                },
            }
        ),
        encoding="utf-8",
    )
    return capture


def _write_list_wrapped_spooled_capture(path: Path, *, provider_session_id: str = "capture") -> Path:
    capture = _write_spooled_capture(path, provider_session_id=provider_session_id)
    payload = json.loads(capture.read_text(encoding="utf-8"))
    capture.write_text(json.dumps([{"ignored": True}, payload]), encoding="utf-8")
    return capture


def _create_browser_index_db(
    path: Path,
    *,
    raw_id: str = "raw-capture",
    native_id: str = "capture",
    message_count: int = 1,
) -> None:
    with sqlite3.connect(path / "index.db") as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT,
                raw_id TEXT,
                native_id TEXT,
                title TEXT,
                message_count INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (session_id, raw_id, native_id, title, message_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"chatgpt-export:{native_id}",
                raw_id,
                native_id,
                "Deployment smoke capture",
                message_count,
            ),
        )


class _FakeResponse:
    def __init__(
        self,
        status: int,
        payload: dict[str, object] | str,
        *,
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self._payload = payload
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self._payload, str):
            return self._payload.encode()
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
    assert "Runtime evidence:" in output


def test_deployment_smoke_runtime_evidence_includes_effective_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(deployment_smoke, "_resolve_command", lambda name, *, path: f"/bin/{name}")
    monkeypatch.setattr(deployment_smoke, "_repo_head", lambda: "abc123def456")
    secret = "auth-LEAKME"

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command == ["polylogue", "config", "--format", "json"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "values": {
                            "api_auth_token": {
                                "value": "<set>",
                                "secret": True,
                                "secret_present": True,
                                "source_layer": "env",
                            }
                        }
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "polylogue, version 0.1.0+abc123d", "")

    monkeypatch.setattr(deployment_smoke, "_run_command", fake_run)
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

    report = deployment_smoke.build_report(
        path="/bin",
        daemon_base_url="http://daemon",
        receiver_base_url="http://receiver",
        archive_root=tmp_path,
        timeout_s=1,
    )

    evidence = report.runtime_evidence
    assert evidence["archive_root"] == str(tmp_path)
    assert evidence["daemon_base_url"] == "http://daemon"
    assert evidence["receiver_base_url"] == "http://receiver"
    config_probe = evidence["effective_config"]
    assert config_probe["ok"] is True
    assert config_probe["payload"]["values"]["api_auth_token"]["value"] == "<set>"
    assert secret not in json.dumps(evidence)


def test_deployment_smoke_includes_effective_config_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(deployment_smoke, "_resolve_command", lambda name, *, path: f"/bin/{name}")
    monkeypatch.setattr(deployment_smoke, "_repo_head", lambda: "abc123def456")
    monkeypatch.setattr(
        deployment_smoke,
        "_resource_limit_signals",
        lambda: {"cgroup": {"available": True, "cpu_max": "max 100000", "memory_high": "2G"}},
    )
    monkeypatch.setattr(
        deployment_smoke,
        "_run_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, f"{' '.join(command)} version", ""),
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

    report = deployment_smoke.build_report(
        path="/bin",
        daemon_base_url="http://daemon/",
        receiver_base_url="http://receiver/",
        archive_root=tmp_path,
        timeout_s=1,
    )

    evidence = report.runtime_evidence
    assert evidence["archive_root"] == str(tmp_path)
    assert evidence["daemon_base_url"] == "http://daemon/"
    assert evidence["receiver_base_url"] == "http://receiver/"
    command_versions = cast(dict[str, dict[str, object]], evidence["command_versions"])
    assert command_versions["polylogue --version"]["path"] == "/bin/polylogue"
    assert command_versions["polylogued --version"]["stdout"] == "polylogued --version version"
    assert evidence["resource_limits"]["cgroup"]["memory_high"] == "2G"


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


def test_deployment_smoke_keeps_facets_timeout_optional(
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

    assert report.ok is True
    assert not any(failure.startswith("route:http://daemon/api/facets:") for failure in report.failures)
    assert "optional web-shell facets route exceeds the deployed smoke timeout" in report.diagnostics["likely_causes"]
    assert any("deferred /api/facets" in action for action in report.diagnostics["next_actions"])


def test_deployment_smoke_accepts_html_root_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment_smoke,
        "_open_url",
        lambda url, *, timeout_s: _FakeResponse(
            200,
            "<!doctype html><title>Polylogue</title>",
            content_type="text/html; charset=utf-8",
        ),
    )

    probe = deployment_smoke._probe_route("http://daemon/", timeout_s=1)

    assert probe.ok is True
    assert probe.status == 200
    assert probe.content_type == "text/html; charset=utf-8"
    assert probe.body_bytes == len("<!doctype html><title>Polylogue</title>")
    assert probe.payload == {"body_preview": "<!doctype html><title>Polylogue</title>"}


def test_deployment_smoke_browser_render_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment_smoke,
        "_resolve_browser_executable",
        lambda path, executable: "/bin/google-chrome",
    )

    def fake_run(
        command: list[str],
        *,
        path: str,
        timeout_s: float,
    ) -> subprocess.CompletedProcess[str]:
        del path, timeout_s
        assert not any(arg.startswith("--timeout=") for arg in command)
        assert not any(arg.startswith("--virtual-time-budget=") for arg in command)
        screenshot_arg = next(arg for arg in command if arg.startswith("--screenshot="))
        Path(screenshot_arg.split("=", 1)[1]).write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0, "<html>Polylogue</html>", "")

    monkeypatch.setattr(deployment_smoke, "_run_browser_command", fake_run)

    probe = deployment_smoke._probe_browser_render(
        "http://daemon/",
        path="/bin",
        timeout_s=1,
        executable=None,
    )

    assert probe.ok is True
    assert probe.executable == "/bin/google-chrome"
    assert probe.dom_bytes == len("<html>Polylogue</html>")
    assert probe.screenshot_bytes == 3
    assert probe.error is None


def test_deployment_smoke_resolves_browser_from_candidate_path(tmp_path: Path) -> None:
    chrome = tmp_path / "chromium"
    chrome.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chrome.chmod(chrome.stat().st_mode | stat.S_IXUSR)

    resolved = deployment_smoke._resolve_browser_executable(str(tmp_path), None)

    assert resolved == str(chrome)


def test_deployment_smoke_rejects_missing_explicit_browser_path(tmp_path: Path) -> None:
    missing = tmp_path / "not-chrome"

    resolved = deployment_smoke._resolve_browser_executable(str(tmp_path), str(missing))
    diagnostics = deployment_smoke._browser_executable_resolution(str(tmp_path), str(missing))

    assert resolved is None
    assert diagnostics["ok"] is False
    assert diagnostics["requested"] == str(missing)
    assert diagnostics["error"] == "explicit_browser_executable_not_found_or_not_executable"


def test_deployment_smoke_browser_render_probe_reports_missing_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deployment_smoke, "_resolve_browser_executable", lambda path, executable: None)

    probe = deployment_smoke._probe_browser_render(
        "http://daemon/",
        path="/bin",
        timeout_s=1,
        executable=None,
    )

    assert probe.ok is False
    assert probe.error == "chrome_not_found"
    assert probe.executable is None


def test_deployment_smoke_browser_render_probe_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment_smoke,
        "_resolve_browser_executable",
        lambda path, executable: "/bin/google-chrome",
    )

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd=["google-chrome"],
            timeout=1,
            output="<html>",
            stderr="still loading",
        )

    monkeypatch.setattr(deployment_smoke, "_run_browser_command", fake_run)

    probe = deployment_smoke._probe_browser_render(
        "http://daemon/",
        path="/bin",
        timeout_s=1,
        executable=None,
    )

    assert probe.ok is False
    assert probe.error == "browser_timeout"
    assert probe.dom_bytes == len("<html>")
    assert "still loading" in probe.stderr_tail


def test_deployment_smoke_browser_render_accepts_timeout_after_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deployment_smoke,
        "_resolve_browser_executable",
        lambda path, executable: "/bin/google-chrome",
    )

    def fake_run(
        command: list[str],
        *,
        path: str,
        timeout_s: float,
    ) -> subprocess.CompletedProcess[str]:
        del path, timeout_s
        screenshot_arg = next(arg for arg in command if arg.startswith("--screenshot="))
        Path(screenshot_arg.split("=", 1)[1]).write_bytes(b"png")
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=1,
            output="<html>Polylogue</html>",
            stderr="DevTools listening",
        )

    monkeypatch.setattr(deployment_smoke, "_run_browser_command", fake_run)

    probe = deployment_smoke._probe_browser_render(
        "http://daemon/",
        path="/bin",
        timeout_s=1,
        executable=None,
    )

    assert probe.ok is True
    assert probe.dom_bytes == len("<html>Polylogue</html>")
    assert probe.screenshot_bytes == 3
    assert probe.error is None
    assert probe.caveats == ("browser_timeout_after_capture",)


def test_deployment_smoke_report_includes_browser_render_failure(
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
    monkeypatch.setattr(
        deployment_smoke,
        "_open_url",
        lambda url, *, timeout_s: _FakeResponse(200, {"ok": True, "url": url}),
    )
    monkeypatch.setattr(
        deployment_smoke,
        "_probe_browser_render",
        lambda *args, **kwargs: deployment_smoke.BrowserRenderProbe(
            url="http://daemon/",
            executable="/bin/google-chrome",
            exit_code=None,
            ok=False,
            error="browser_timeout",
        ),
    )

    report = deployment_smoke.build_report(
        path="/bin",
        daemon_base_url="http://daemon",
        receiver_base_url="http://receiver",
        archive_root=tmp_path,
        timeout_s=1,
        browser=True,
    )

    assert report.ok is False
    assert "browser-render:browser_timeout" in report.failures
    assert report.browser_render is not None
    assert report.browser_render.error == "browser_timeout"
    assert any(
        "web-shell browser render does not reach DOM/screenshot" in cause
        for cause in report.diagnostics["likely_causes"]
    )


def test_deployment_smoke_reports_spooled_browser_capture_without_raw_rows(tmp_path: Path) -> None:
    _write_spooled_capture(tmp_path)
    _create_browser_source_db(tmp_path)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.spooled_count == 1
    assert probe.raw_rows == 0
    assert probe.error == "spooled_without_raw_rows"


def test_deployment_smoke_reports_spooled_browser_capture_newer_than_raw_row(tmp_path: Path) -> None:
    capture = _write_spooled_capture(tmp_path)
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000) - 1000,
        capture_path=capture,
    )

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.spooled_count == 1
    assert probe.raw_rows == 1
    assert probe.latest_spooled_mtime_ms is not None
    assert probe.latest_raw_file_mtime_ms is not None
    assert probe.latest_spooled_mtime_ms > probe.latest_raw_file_mtime_ms
    assert probe.error == "spooled_newer_than_raw_rows"


def test_deployment_smoke_reports_latest_browser_capture_missing_index_row(tmp_path: Path) -> None:
    capture = _write_spooled_capture(tmp_path)
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        capture_path=capture,
    )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT,
                raw_id TEXT,
                native_id TEXT,
                title TEXT,
                message_count INTEGER
            )
            """
        )

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.raw_rows == 1
    assert probe.latest_raw_id == "raw-capture"
    assert probe.error == "latest_spooled_not_indexed"


def test_deployment_smoke_reports_latest_browser_capture_indexed_without_messages(tmp_path: Path) -> None:
    capture = _write_spooled_capture(tmp_path)
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        capture_path=capture,
    )
    _create_browser_index_db(tmp_path, message_count=0)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.raw_rows == 1
    assert probe.latest_indexed_message_count == 0
    assert probe.error == "latest_spooled_indexed_without_messages"


def test_deployment_smoke_accepts_materialized_browser_capture_at_latest_mtime(tmp_path: Path) -> None:
    capture = _write_spooled_capture(tmp_path)
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        capture_path=capture,
    )
    _create_browser_index_db(tmp_path)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is True
    assert probe.spooled_count == 1
    assert probe.raw_rows == 1
    assert probe.latest_capture_provider_session_id == "capture"
    assert probe.latest_capture_turn_count == 1
    assert probe.latest_raw_id == "raw-capture"
    assert probe.latest_indexed_native_id == "capture"
    assert probe.latest_indexed_message_count == 1
    assert probe.error is None


def test_deployment_smoke_accepts_legacy_spooled_id_normalized_by_index(tmp_path: Path) -> None:
    legacy_id = "chatgpt:6a232355-ac3c-83eb-a93d-9c70697bfc18:9f658806"
    native_id = "6a232355-ac3c-83eb-a93d-9c70697bfc18"
    capture = _write_spooled_capture(tmp_path, provider_session_id=legacy_id)
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        capture_path=capture,
        native_id=native_id,
    )
    _create_browser_index_db(tmp_path, native_id=native_id)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is True
    assert probe.latest_capture_provider_session_id == legacy_id
    assert probe.latest_indexed_native_id == native_id
    assert probe.error is None


def test_deployment_smoke_accepts_list_wrapped_browser_capture_envelope(tmp_path: Path) -> None:
    capture = _write_list_wrapped_spooled_capture(tmp_path)
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        capture_path=capture,
    )
    _create_browser_index_db(tmp_path)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is True
    assert probe.latest_capture_provider_session_id == "capture"
    assert probe.latest_capture_turn_count == 1


def test_deployment_smoke_matches_latest_browser_capture_by_stable_suffix(tmp_path: Path) -> None:
    capture = _write_spooled_capture(tmp_path)
    source_path = Path("/alternate/archive/browser-capture/chatgpt/capture.json")
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        capture_path=source_path,
    )
    _create_browser_index_db(tmp_path)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is True
    assert probe.latest_raw_source_path == str(source_path)
    assert probe.latest_indexed_message_count == 1


def test_deployment_smoke_matches_repeated_browser_capture_by_identity(tmp_path: Path) -> None:
    capture = _write_spooled_capture(tmp_path, provider_session_id="capture")
    source_path = Path("/alternate/archive/browser-capture/chatgpt/chatgpt-capture-oldhash.json")
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        capture_path=source_path,
        native_id="capture",
    )
    _create_browser_index_db(tmp_path, native_id="capture")

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is True
    assert probe.latest_spooled_path == str(capture)
    assert probe.latest_raw_source_path == str(source_path)
    assert probe.latest_indexed_native_id == "capture"
    assert probe.latest_indexed_message_count == 1


def test_deployment_smoke_escapes_latest_suffix_like_wildcards(tmp_path: Path) -> None:
    capture = _write_spooled_capture(tmp_path, provider_session_id="cap_100%")
    _create_browser_source_db(
        tmp_path,
        file_mtime_ms=int(capture.stat().st_mtime * 1000),
        raw_id="raw-wrong",
        native_id="wrong",
        capture_path=Path("/alternate/archive/browser-capture/chatgpt/capx100z.json"),
    )
    _create_browser_index_db(tmp_path, raw_id="raw-wrong", native_id="wrong")

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.error == "latest_spooled_without_raw_row"


def test_deployment_smoke_accepts_current_receiver_archive_state_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _write_spooled_capture(tmp_path, provider_session_id="capture-ok")
    archive_probe = deployment_smoke.BrowserCaptureArchiveProbe(
        spool_path=str(tmp_path / "browser-capture"),
        source_db_path=str(tmp_path / "source.db"),
        spooled_count=1,
        latest_spooled_path=str(capture),
        latest_spooled_mtime=capture.stat().st_mtime,
        latest_spooled_mtime_ms=int(capture.stat().st_mtime * 1000),
        raw_rows=1,
        latest_raw_file_mtime_ms=int(capture.stat().st_mtime * 1000),
        ok=True,
    )

    def fake_open_receiver_url(url: str, *, timeout_s: float) -> _FakeResponse:
        del timeout_s
        assert "provider=chatgpt" in url
        assert "provider_session_id=capture-ok" in url
        return _FakeResponse(
            200,
            {
                "provider": "chatgpt",
                "provider_session_id": "capture-ok",
                "state": "archived",
                "lifecycle": "archived",
                "captured": True,
                "artifact_ref": "chatgpt/capture-ok.json",
                "raw_row_exists": True,
                "raw_id": "raw-capture",
                "indexed_session_exists": True,
                "indexed_session_id": "chatgpt-export:capture-ok",
                "indexed_message_count": 1,
            },
        )

    monkeypatch.setattr(deployment_smoke, "_open_receiver_url", fake_open_receiver_url)

    probe = deployment_smoke._probe_browser_capture_receiver_archive_state(
        receiver_base_url="http://receiver",
        browser_capture_archive=archive_probe,
        timeout_s=1,
    )

    assert probe.ok is True
    assert probe.provider == "chatgpt"
    assert probe.provider_session_id == "capture-ok"
    assert probe.error is None


def test_deployment_smoke_reports_receiver_archive_state_absolute_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _write_spooled_capture(tmp_path, provider_session_id="capture-stale")
    archive_probe = deployment_smoke.BrowserCaptureArchiveProbe(
        spool_path=str(tmp_path / "browser-capture"),
        source_db_path=str(tmp_path / "source.db"),
        spooled_count=1,
        latest_spooled_path=str(capture),
        latest_spooled_mtime=capture.stat().st_mtime,
        latest_spooled_mtime_ms=int(capture.stat().st_mtime * 1000),
        raw_rows=0,
        latest_raw_file_mtime_ms=None,
        ok=False,
        error="spooled_without_raw_rows",
    )

    monkeypatch.setattr(
        deployment_smoke,
        "_open_receiver_url",
        lambda url, *, timeout_s: _FakeResponse(
            200,
            {
                "provider": "chatgpt",
                "provider_session_id": "capture-stale",
                "state": "archived",
                "captured": True,
                "artifact_path": str(capture),
            },
        ),
    )

    probe = deployment_smoke._probe_browser_capture_receiver_archive_state(
        receiver_base_url="http://receiver",
        browser_capture_archive=archive_probe,
        timeout_s=1,
    )

    assert probe.ok is False
    assert probe.error == "receiver_archive_state_absolute_artifact_path"


def test_deployment_smoke_rejects_receiver_archive_state_without_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _write_spooled_capture(tmp_path, provider_session_id="capture-no-state")
    archive_probe = deployment_smoke.BrowserCaptureArchiveProbe(
        spool_path=str(tmp_path / "browser-capture"),
        source_db_path=str(tmp_path / "source.db"),
        spooled_count=1,
        latest_spooled_path=str(capture),
        latest_spooled_mtime=capture.stat().st_mtime,
        latest_spooled_mtime_ms=int(capture.stat().st_mtime * 1000),
        raw_rows=1,
        latest_raw_file_mtime_ms=int(capture.stat().st_mtime * 1000),
        ok=True,
    )

    monkeypatch.setattr(
        deployment_smoke,
        "_open_receiver_url",
        lambda url, *, timeout_s: _FakeResponse(
            200,
            {
                "provider": "chatgpt",
                "provider_session_id": "capture-no-state",
                "captured": True,
                "artifact_ref": "chatgpt/capture-no-state.json",
            },
        ),
    )

    probe = deployment_smoke._probe_browser_capture_receiver_archive_state(
        receiver_base_url="http://receiver",
        browser_capture_archive=archive_probe,
        timeout_s=1,
    )

    assert probe.ok is False
    assert probe.error == "receiver_archive_state_missing_lifecycle"


def test_deployment_smoke_rejects_receiver_archive_state_without_index_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capture = _write_spooled_capture(tmp_path, provider_session_id="capture-no-index")
    archive_probe = deployment_smoke.BrowserCaptureArchiveProbe(
        spool_path=str(tmp_path / "browser-capture"),
        source_db_path=str(tmp_path / "source.db"),
        spooled_count=1,
        latest_spooled_path=str(capture),
        latest_spooled_mtime=capture.stat().st_mtime,
        latest_spooled_mtime_ms=int(capture.stat().st_mtime * 1000),
        raw_rows=1,
        latest_raw_file_mtime_ms=int(capture.stat().st_mtime * 1000),
        ok=True,
    )

    monkeypatch.setattr(
        deployment_smoke,
        "_open_receiver_url",
        lambda url, *, timeout_s: _FakeResponse(
            200,
            {
                "provider": "chatgpt",
                "provider_session_id": "capture-no-index",
                "state": "archived",
                "captured": True,
                "artifact_ref": "chatgpt/capture-no-index.json",
                "raw_row_exists": True,
                "indexed_session_exists": False,
            },
        ),
    )

    probe = deployment_smoke._probe_browser_capture_receiver_archive_state(
        receiver_base_url="http://receiver",
        browser_capture_archive=archive_probe,
        timeout_s=1,
    )

    assert probe.ok is False
    assert probe.error == "receiver_archive_state_missing_index_evidence"


@pytest.mark.parametrize(
    "receiver_error",
    [
        "receiver_archive_state_missing_index_evidence",
        "receiver_archive_state_not_captured",
    ],
)
def test_deployment_smoke_diagnoses_receiver_archive_state_without_query_visibility(
    receiver_error: str,
) -> None:
    diagnostics = deployment_smoke._diagnose(
        commands=[
            deployment_smoke.CommandProbe(
                name="polylogue --version",
                path="/bin/polylogue",
                exit_code=0,
                ok=True,
                stdout="polylogue, version 0.1.0+abc123",
                stderr="",
            ),
            deployment_smoke.CommandProbe(
                name="polylogued --version",
                path="/bin/polylogued",
                exit_code=0,
                ok=True,
                stdout="polylogued, version 0.1.0+abc123",
                stderr="",
            ),
        ],
        routes=[],
        repo_head="abc123def456",
        browser_render=None,
        browser_executable_resolution={},
        browser_capture_archive=deployment_smoke.BrowserCaptureArchiveProbe(
            spool_path="/tmp/browser-capture",
            source_db_path="/tmp/source.db",
            spooled_count=1,
            latest_spooled_path="/tmp/browser-capture/chatgpt/capture-no-index.json",
            latest_spooled_mtime=1.0,
            latest_spooled_mtime_ms=1000,
            raw_rows=1,
            latest_raw_file_mtime_ms=1000,
            ok=True,
        ),
        browser_capture_receiver_archive_state=deployment_smoke.BrowserCaptureReceiverArchiveStateProbe(
            url="http://receiver/v1/archive-state",
            provider="chatgpt",
            provider_session_id="capture-no-index",
            status=200,
            payload={
                "state": "archived",
                "captured": True,
                "artifact_ref": "chatgpt/capture-no-index.json",
                "raw_row_exists": True,
                "indexed_session_exists": False,
            },
            ok=False,
            error=receiver_error,
        ),
    )

    assert (
        "deployed browser-capture receiver archive-state DTO does not prove query visibility"
        in diagnostics["likely_causes"]
    )
    assert any("verify source/index rows" in action for action in diagnostics["next_actions"])
