from __future__ import annotations

import json
import sqlite3
import subprocess
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from devtools import deployment_smoke


def _create_browser_source_db(path: Path, *, file_mtime_ms: int | None = None) -> None:
    with sqlite3.connect(path / "source.db") as conn:
        conn.execute("CREATE TABLE raw_sessions (source_path TEXT, file_mtime_ms INTEGER)")
        if file_mtime_ms is not None:
            conn.execute(
                "INSERT INTO raw_sessions (source_path, file_mtime_ms) VALUES (?, ?)",
                (str(path / "browser-capture" / "chatgpt" / "capture.json"), file_mtime_ms),
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


def test_deployment_smoke_reports_spooled_browser_capture_without_raw_rows(tmp_path: Path) -> None:
    capture_dir = tmp_path / "browser-capture" / "chatgpt"
    capture_dir.mkdir(parents=True)
    (capture_dir / "capture.json").write_text("{}", encoding="utf-8")
    _create_browser_source_db(tmp_path)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.spooled_count == 1
    assert probe.raw_rows == 0
    assert probe.error == "spooled_without_raw_rows"


def test_deployment_smoke_reports_spooled_browser_capture_newer_than_raw_row(tmp_path: Path) -> None:
    capture_dir = tmp_path / "browser-capture" / "chatgpt"
    capture_dir.mkdir(parents=True)
    capture = capture_dir / "capture.json"
    capture.write_text("{}", encoding="utf-8")
    _create_browser_source_db(tmp_path, file_mtime_ms=int(capture.stat().st_mtime * 1000) - 1000)

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is False
    assert probe.spooled_count == 1
    assert probe.raw_rows == 1
    assert probe.latest_spooled_mtime_ms is not None
    assert probe.latest_raw_file_mtime_ms is not None
    assert probe.latest_spooled_mtime_ms > probe.latest_raw_file_mtime_ms
    assert probe.error == "spooled_newer_than_raw_rows"


def test_deployment_smoke_accepts_browser_capture_raw_row_at_latest_mtime(tmp_path: Path) -> None:
    capture_dir = tmp_path / "browser-capture" / "chatgpt"
    capture_dir.mkdir(parents=True)
    capture = capture_dir / "capture.json"
    capture.write_text("{}", encoding="utf-8")
    _create_browser_source_db(tmp_path, file_mtime_ms=int(capture.stat().st_mtime * 1000))

    probe = deployment_smoke._probe_browser_capture_archive(archive_root=tmp_path)

    assert probe.ok is True
    assert probe.spooled_count == 1
    assert probe.raw_rows == 1
    assert probe.error is None


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
                "captured": True,
                "artifact_ref": "chatgpt/capture-ok.json",
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
