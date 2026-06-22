from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from devtools import dev_loop


def test_system_service_status_reports_active_unit_archive_root(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], *, timeout_s: float = 2.0) -> dev_loop.CommandResult:
        assert args[:3] == ["systemctl", "--user", "show"]
        return dev_loop.CommandResult(
            exit_code=0,
            stdout="\n".join(
                [
                    "ActiveState=active",
                    "SubState=running",
                    "MainPID=1234",
                    "FragmentPath=/home/sinity/.config/systemd/user/polylogued.service",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(dev_loop, "_run_command", fake_run)
    monkeypatch.setattr(
        dev_loop, "_read_environ", lambda pid: {"POLYLOGUE_ARCHIVE_ROOT": "/archive"} if pid == 1234 else {}
    )

    payload = dev_loop.system_service_status()

    assert payload["available"] is True
    assert payload["active"] is True
    assert payload["main_pid"] == 1234
    assert payload["archive_root"] == "/archive"


def test_port_status_reports_owner_and_archive_root(monkeypatch: pytest.MonkeyPatch) -> None:
    ss_output = (
        'LISTEN 0 4096 127.0.0.1:8766 0.0.0.0:* users:(("python",pid=2222,fd=8))\n'
        'LISTEN 0 4096 127.0.0.1:8765 0.0.0.0:* users:(("python",pid=3333,fd=9))'
    )

    def fake_run(args: list[str], *, timeout_s: float = 2.0) -> dev_loop.CommandResult:
        assert args == ["ss", "-H", "-ltnp"]
        return dev_loop.CommandResult(exit_code=0, stdout=ss_output, stderr="")

    monkeypatch.setattr(dev_loop, "_run_command", fake_run)
    monkeypatch.setattr(dev_loop, "_socket_connectable", lambda port: port == 8766)
    monkeypatch.setattr(
        dev_loop,
        "_read_environ",
        lambda pid: {"POLYLOGUE_ARCHIVE_ROOT": f"/archive/{pid}"},
    )

    payload = dev_loop.port_status(8766)

    assert payload["connectable"] is True
    assert payload["owner_count"] == 1
    owner = payload["owners"][0]
    assert owner["pid"] == 2222
    assert owner["archive_root"] == "/archive/2222"


def test_build_dev_loop_status_uses_branch_local_paths_and_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)

    monkeypatch.setattr(
        dev_loop,
        "system_service_status",
        lambda: {
            "unit": "polylogued.service",
            "available": True,
            "active": True,
            "active_state": "active",
            "main_pid": 123,
            "archive_root": "/prod",
        },
    )
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 1 if port == 9999 else 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    payload = dev_loop.build_dev_loop_status(repo_root=repo, api_port=9999, browser_capture_port=9998, prepare=True)

    assert payload["branch"] == "feature/dev-loop"
    assert payload["commit"] == "abc1234"
    assert payload["run_id"] == "feature-dev-loop-abc1234-api9999-capture9998"
    assert payload["prepared"] is True
    assert payload["preflight_json_written"] is True
    assert Path(str(payload["dev_archive_root"])).is_dir()
    assert Path(str(payload["log_dir"])).is_dir()
    assert payload["run_log_dir"] == str(repo / ".cache" / "dev-loop" / "feature-dev-loop-abc1234-api9999-capture9998")
    assert Path(str(payload["run_log_dir"])).is_dir()
    assert Path(str(payload["artifacts"]["browser_dir"])).is_dir()
    assert Path(str(payload["artifacts"]["terminal_dir"])).is_dir()
    assert Path(str(payload["artifacts"]["tui_dir"])).is_dir()
    assert Path(str(payload["artifacts"]["preflight_json"])).is_file()
    assert payload["artifacts"]["daemon_log"].endswith(
        ".cache/dev-loop/feature-dev-loop-abc1234-api9999-capture9998/polylogued.log"
    )
    assert payload["artifacts"]["dev_events"].endswith(
        ".cache/dev-loop/feature-dev-loop-abc1234-api9999-capture9998/dev-loop.events.jsonl"
    )
    assert f"PYTHONPATH={repo}" in payload["commands"]["run_daemon"]
    assert "from polylogue.daemon.cli import main; main()" in payload["commands"]["run_daemon"]
    assert "run --api-port 9999 --port 9998" in payload["commands"]["run_daemon"]
    assert "polylogue ops status" in payload["commands"]["capture_cli_status"]
    assert payload["commands"]["capture_cli_status"].endswith("terminal/polylogue-ops-status.typescript")
    assert payload["commands"]["capture_tui_placeholder"].endswith(
        "tui; use the local terminal-control surface or VHS when visual playback is needed"
    )
    assert payload["suggested_env"]["POLYLOGUE_ARCHIVE_ROOT"] == str(repo / ".local" / "dev-archive")
    assert payload["suggested_env"]["POLYLOGUE_DEV_LOOP_RUN_ID"] == payload["run_id"]
    assert payload["warnings"] == [
        "systemwide polylogued.service is active; stop it or use isolated ports before branch-local runs",
        "api port 9999 already has a listener",
    ]


def test_build_dev_loop_status_marks_isolated_ports_without_service_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)

    monkeypatch.setattr(
        dev_loop,
        "system_service_status",
        lambda: {
            "unit": "polylogued.service",
            "available": True,
            "active": True,
            "active_state": "active",
            "main_pid": 123,
            "archive_root": "/prod",
        },
    )
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    payload = dev_loop.build_dev_loop_status(
        repo_root=repo,
        api_port=9911,
        browser_capture_port=9912,
        prepare=True,
        port_selection="isolated",
    )

    assert payload["port_selection"] == "isolated"
    assert payload["run_id"] == "feature-dev-loop-abc1234-api9911-capture9912"
    assert payload["warnings"] == []
    assert payload["suggested_env"]["POLYLOGUE_API_PORT"] == "9911"
    assert payload["suggested_env"]["POLYLOGUE_BROWSER_CAPTURE_PORT"] == "9912"
    assert "--api-port 9911 --browser-capture-port 9912 --prepare" in payload["commands"]["prepare"]
    assert payload["commands"]["prepare_isolated"] == "devtools workspace dev-loop --isolated-ports --prepare"
    assert "--api-port 9911 --browser-capture-port 9912 --json" in payload["commands"]["save_preflight"]


def test_main_isolated_ports_allocates_free_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": True})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )
    monkeypatch.setattr(dev_loop, "allocate_isolated_ports", lambda: (9921, 9922))

    assert (
        dev_loop.main(
            [
                "--json",
                "--isolated-ports",
                "--log-dir",
                str(tmp_path / "dev-loop-logs"),
                "--archive-root",
                str(tmp_path / "archive"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["port_selection"] == "isolated"
    assert payload["ports"]["api"]["port"] == 9921
    assert payload["ports"]["browser_capture"]["port"] == 9922
    assert payload["warnings"] == []
    assert payload["commands"]["open_web_shell"] == "http://127.0.0.1:9921/"


def test_main_json_outputs_machine_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        dev_loop,
        "build_dev_loop_status",
        lambda **kwargs: {
            "repo_root": str(tmp_path),
            "branch": "feature/dev-loop",
            "commit": "abc1234",
            "run_id": "feature-dev-loop-abc1234-api8766-capture8765",
            "prepared": kwargs["prepare"],
            "preflight_json_written": kwargs["prepare"],
            "dev_archive_root": str(tmp_path / "archive"),
            "log_dir": str(tmp_path / "logs"),
            "run_log_dir": str(tmp_path / "logs" / "run"),
            "artifacts": {},
            "system_service": {"active": False},
            "ports": {},
            "suggested_env": {},
            "commands": {},
            "warnings": [],
        },
    )

    assert dev_loop.main(["--json", "--prepare"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prepared"] is True
    assert payload["branch"] == "feature/dev-loop"


def test_receiver_smoke_proves_auth_rejection_and_acceptance(tmp_path: Path) -> None:
    payload = dev_loop.run_receiver_smoke(spool_path=tmp_path / "spool")

    assert payload["ok"] is True
    assert payload["unauthenticated_status"] == 401
    assert payload["unauthenticated_error"] == "unauthorized"
    assert payload["authenticated_status"] == 202
    assert payload["artifact_ref"] == "chatgpt/dev-loop-smoke-e368c8af2a6b.json"
    assert Path(str(payload["artifact_path"])).is_file()


def test_receiver_smoke_cli_outputs_combined_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    assert dev_loop.main(["--json", "--receiver-smoke", "--archive-root", str(tmp_path / "archive")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["preflight"]["prepared"] is True
    assert payload["preflight"]["preflight_json_written"] is True
    assert payload["receiver_smoke"]["ok"] is True


def test_extension_smoke_runs_background_worker_against_receiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    assert (
        dev_loop.main(
            [
                "--json",
                "--log-dir",
                str(tmp_path / "dev-loop-logs"),
                "--archive-root",
                str(tmp_path / "archive"),
                "--extension-smoke",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    smoke = payload["extension_smoke"]
    assert smoke["ok"] is True
    assert smoke["exit_code"] == 0
    assert smoke["artifact_ref"] == "chatgpt/dev-loop-extension-smoke-17055bf60522.json"
    assert Path(smoke["artifact_path"]).is_file()
    artifacts = smoke["artifacts"]
    assert Path(artifacts["summary"]).is_file()
    assert Path(artifacts["stdout"]).read_text(encoding="utf-8").strip().startswith("{")
    event_path = Path(payload["preflight"]["artifacts"]["dev_events"])
    event_rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    extension_events = event_rows[-2:]
    assert [row["event_type"] for row in extension_events] == [
        "extension_smoke_requested",
        "extension_smoke_finished",
    ]
    assert extension_events[0]["surface"] == "browser_extension"
    assert extension_events[1]["status"] == "ok"
    assert extension_events[1]["payload"]["artifact_ref"] == smoke["artifact_ref"]


def test_browser_plan_writes_local_extension_launch_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN", "dev-token")

    assert (
        dev_loop.main(
            [
                "--json",
                "--log-dir",
                str(tmp_path / "dev-loop-logs"),
                "--archive-root",
                str(tmp_path / "archive"),
                "--api-port",
                "9876",
                "--browser-capture-port",
                "9875",
                "--browser-plan",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    plan = payload["browser_plan"]
    assert plan["ok"] is True
    assert plan["receiver_url"] == "http://127.0.0.1:9875"
    assert plan["web_shell_url"] == "http://127.0.0.1:9876/"
    assert plan["receiver_auth_configured"] is True
    assert Path(plan["profile_dir"]).is_dir()
    assert Path(plan["screenshot_dir"]).is_dir()
    assert Path(plan["downloads_dir"]).is_dir()
    assert Path(plan["extension_root"]).name == "browser-extension"
    chrome = plan["commands"]["chrome"]
    assert chrome[0] == "google-chrome-stable"
    assert f"--load-extension={plan['extension_root']}" in chrome
    assert f"--user-data-dir={plan['profile_dir']}" in chrome

    artifacts = plan["artifacts"]
    json_plan = Path(artifacts["json"])
    markdown_plan = Path(artifacts["markdown"])
    assert json_plan.is_file()
    assert markdown_plan.is_file()
    assert "POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN" in markdown_plan.read_text(encoding="utf-8")
    event_rows = [json.loads(line) for line in Path(artifacts["events"]).read_text(encoding="utf-8").splitlines()]
    assert event_rows[-1]["event_type"] == "browser_plan_written"
    assert event_rows[-1]["surface"] == "browser"
    assert event_rows[-1]["payload"]["receiver_url"] == "http://127.0.0.1:9875"


def test_browser_plan_and_extension_smoke_can_share_one_dev_loop_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    assert (
        dev_loop.main(
            [
                "--json",
                "--log-dir",
                str(tmp_path / "dev-loop-logs"),
                "--archive-root",
                str(tmp_path / "archive"),
                "--api-port",
                "9876",
                "--browser-capture-port",
                "9875",
                "--browser-plan",
                "--extension-smoke",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["browser_plan"]["ok"] is True
    assert payload["extension_smoke"]["ok"] is True
    browser_artifacts = payload["browser_plan"]["artifacts"]
    smoke_artifacts = payload["extension_smoke"]["artifacts"]
    assert Path(browser_artifacts["json"]).is_file()
    assert Path(smoke_artifacts["summary"]).is_file()

    event_rows = [
        json.loads(line)
        for line in Path(payload["preflight"]["artifacts"]["dev_events"]).read_text(encoding="utf-8").splitlines()
    ]
    event_types = [row["event_type"] for row in event_rows]
    assert "browser_plan_written" in event_types
    assert event_types[-3:] == ["extension_smoke_requested", "extension_smoke_finished", "browser_plan_written"]


def test_browser_smoke_records_real_chrome_extension_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    class Completed:
        returncode = 0
        stdout = '{"ok":true}\n'
        stderr = ""

    def fake_run(*args: object, **kwargs: object) -> Completed:
        env_obj = kwargs.get("env")
        if not isinstance(env_obj, dict) or "POLYLOGUE_BROWSER_SMOKE_OUT" not in env_obj:
            return Completed()
        env = env_obj
        output_path = Path(env["POLYLOGUE_BROWSER_SMOKE_OUT"])
        artifact_ref = "chatgpt/dev-loop-browser-smoke.json"
        artifact_path = output_path.parent / "browser-smoke-spool" / artifact_ref
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text('{"ok": true}\n', encoding="utf-8")
        output_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "extension_id": "extension-id",
                    "manifest": {"manifest_version": 3, "name": "Polylogue Browser Capture", "version": "0.1.0"},
                    "capture": {"body": {"artifact_ref": artifact_ref}},
                }
            ),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr("devtools.dev_loop.subprocess.run", fake_run)

    assert (
        dev_loop.main(
            [
                "--json",
                "--log-dir",
                str(tmp_path / "dev-loop-logs"),
                "--archive-root",
                str(tmp_path / "archive"),
                "--browser-smoke",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    smoke = payload["browser_smoke"]
    assert smoke["ok"] is True
    assert smoke["extension_id"] == "extension-id"
    assert smoke["artifact_ref"] == "chatgpt/dev-loop-browser-smoke.json"
    assert Path(smoke["artifact_path"]).is_file()
    artifacts = smoke["artifacts"]
    assert Path(artifacts["summary"]).is_file()
    assert Path(artifacts["profile"]).is_dir()
    event_rows = [
        json.loads(line)
        for line in Path(payload["preflight"]["artifacts"]["dev_events"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event_type"] for row in event_rows[-2:]] == ["browser_smoke_requested", "browser_smoke_finished"]
    assert event_rows[-1]["surface"] == "browser"
    assert event_rows[-1]["payload"]["extension_id"] == "extension-id"


def test_tui_plan_writes_visual_inspection_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    assert (
        dev_loop.main(
            [
                "--json",
                "--log-dir",
                str(tmp_path / "dev-loop-logs"),
                "--archive-root",
                str(tmp_path / "archive"),
                "--api-port",
                "9876",
                "--browser-capture-port",
                "9875",
                "--tui-plan",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    plan = payload["tui_plan"]
    assert plan["ok"] is True
    assert plan["env"]["POLYLOGUE_ARCHIVE_ROOT"] == str(tmp_path / "archive")
    assert plan["env"]["POLYLOGUE_FORCE_PLAIN"] == "0"
    assert "script -q -c" in plan["commands"]["script_status"]
    assert plan["commands"]["vhs_render"].startswith("vhs ")
    artifacts = plan["artifacts"]
    assert Path(artifacts["json"]).is_file()
    assert Path(artifacts["markdown"]).is_file()
    assert Path(artifacts["vhs_tape"]).is_file()
    assert Path(artifacts["screenshots"]).is_dir()
    assert "polylogue ops status" in Path(artifacts["vhs_tape"]).read_text(encoding="utf-8")
    event_rows = [json.loads(line) for line in Path(artifacts["events"]).read_text(encoding="utf-8").splitlines()]
    assert event_rows[-1]["event_type"] == "tui_plan_written"
    assert event_rows[-1]["surface"] == "tui"
    assert event_rows[-1]["payload"]["artifacts"]["vhs_tape"] == artifacts["vhs_tape"]


def test_cli_capture_runs_command_with_branch_local_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )
    monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "should-not-be-written")
    monkeypatch.setenv("POLYLOGUE_NOTIFICATION_EMAIL_PASSWORD", "should-not-be-written")

    command = [
        sys.executable,
        "-c",
        "import os; print(os.environ['POLYLOGUE_DEV_LOOP_RUN_ID'])",
    ]
    assert (
        dev_loop.main(
            [
                "--json",
                "--log-dir",
                str(tmp_path / "dev-loop-logs"),
                "--archive-root",
                str(tmp_path / "archive"),
                "--capture-cli",
                "--",
                *command,
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    capture = payload["cli_capture"]
    run_id = payload["preflight"]["run_id"]

    assert capture["ok"] is True
    assert capture["exit_code"] == 0
    artifacts = capture["artifacts"]
    stdout_path = Path(artifacts["stdout"])
    transcript_path = Path(artifacts["transcript"])
    env_path = Path(artifacts["env"])
    summary_path = Path(artifacts["summary"])

    assert stdout_path.read_text(encoding="utf-8").strip() == run_id
    assert f"run_id={run_id}" in transcript_path.read_text(encoding="utf-8")
    env_payload = json.loads(env_path.read_text(encoding="utf-8"))
    assert env_payload["POLYLOGUE_ARCHIVE_ROOT"] == str(tmp_path / "archive")
    assert env_payload["POLYLOGUE_API_AUTH_TOKEN"] == "[redacted]"
    assert env_payload["POLYLOGUE_NOTIFICATION_EMAIL_PASSWORD"] == "[redacted]"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["exit_code"] == 0
    event_path = Path(payload["preflight"]["artifacts"]["dev_events"])
    event_rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    capture_events = event_rows[-2:]
    assert [row["event_type"] for row in capture_events] == ["cli_capture_requested", "cli_capture_finished"]
    assert capture_events[0]["surface"] == "cli"
    assert capture_events[0]["payload"]["command"] == command
    assert capture_events[1]["status"] == "ok"
    assert capture_events[1]["payload"]["exit_code"] == 0
    assert capture_events[1]["payload"]["artifacts"]["transcript"] == str(transcript_path)


def test_inspect_run_summarizes_dev_loop_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "dev-loop" / "feature-dev-loop-abc1234-api8766-capture8765"
    terminal_dir = run_dir / "terminal"
    browser_dir = run_dir / "browser"
    tui_dir = run_dir / "tui"
    terminal_dir.mkdir(parents=True)
    browser_dir.mkdir()
    tui_dir.mkdir()
    preflight = {
        "run_id": run_dir.name,
        "repo_root": "/repo",
        "dev_archive_root": "/repo/.local/dev-archive",
    }
    (run_dir / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    events = [
        {"surface": "daemon", "event_type": "launch_requested", "status": "starting"},
        {
            "surface": "cli",
            "event_type": "cli_capture_finished",
            "status": "ok",
            "payload": {"duration_ms": 120},
        },
        {
            "surface": "browser_extension",
            "event_type": "extension_smoke_finished",
            "status": "failed",
            "payload": {
                "duration_ms": 450,
                "artifacts": {"stdout": "/tmp/ext.stdout", "stderr": "/tmp/ext.stderr"},
            },
        },
    ]
    (run_dir / "dev-loop.events.jsonl").write_text(
        "\n".join([json.dumps(events[0]), "{not-json", *(json.dumps(row) for row in events[1:])]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "polylogued.log").write_text("daemon log\n", encoding="utf-8")
    (run_dir / "polylogued.launch.json").write_text(json.dumps({"ok": True, "pid": 1234}), encoding="utf-8")
    (browser_dir / "browser-plan.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (browser_dir / "extension-smoke.json").write_text(
        json.dumps({"ok": False, "exit_code": 1, "duration_ms": 450, "artifacts": {"stderr": "/tmp/ext.stderr"}}),
        encoding="utf-8",
    )
    (tui_dir / "tui-plan.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (terminal_dir / "polylogue.summary.json").write_text(
        json.dumps({"ok": False, "exit_code": 2, "duration_ms": 80, "artifacts": {"stderr": "/tmp/cli.stderr"}}),
        encoding="utf-8",
    )

    assert dev_loop.main(["--json", "--inspect-run", str(run_dir)]) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is False
    assert payload["run_id"] == run_dir.name
    assert payload["event_count"] == 3
    assert payload["event_status_counts"] == {"failed": 1, "ok": 1, "starting": 1}
    assert payload["event_surface_counts"]["browser_extension"] == 1
    assert payload["problem_events"][0]["event_type"] == "extension_smoke_finished"
    assert payload["malformed_event_lines"][0]["line"] == 2
    assert payload["malformed_event_lines"][0]["text"] == "{not-json"
    assert payload["slowest_events"][0] == {
        "duration_ms": 450,
        "event_type": "extension_smoke_finished",
        "status": "failed",
        "surface": "browser_extension",
    }
    assert payload["summaries"]["daemon_launch"]["pid"] == 1234
    assert payload["summaries"]["tui_plan"]["ok"] is True
    assert payload["failed_summaries"] == [
        {
            "name": "extension_smoke",
            "exit_code": 1,
            "duration_ms": 450,
            "artifacts": {"stderr": "/tmp/ext.stderr"},
        }
    ]
    assert payload["terminal_capture_count"] == 1
    assert payload["terminal_captures"][0]["exit_code"] == 2
    assert payload["failed_terminal_captures"] == [
        {
            "command_text": None,
            "exit_code": 2,
            "duration_ms": 80,
            "artifacts": {"stderr": "/tmp/cli.stderr"},
        }
    ]
    assert payload["artifact_index"]["extension_smoke"]["stderr"] == "/tmp/ext.stderr"
    assert payload["artifact_index"]["terminal_captures"][0]["stderr"] == "/tmp/cli.stderr"
    assert payload["missing_artifacts"] == []
    assert payload["warnings"] == [
        "1 malformed dev-loop event row(s) were skipped",
        "1 event(s) have failed/blocked status",
        "1 run summary file(s) report failure",
        "1 terminal capture(s) report failure",
    ]


def test_cli_capture_rejects_missing_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        dev_loop.main(["--capture-cli"])
    assert exc.value.code == 2
    assert "capture command must not be empty" in capsys.readouterr().err


def test_cli_capture_treats_forwarded_trailing_json_as_wrapper_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    assert (
        dev_loop.main(
            [
                "--archive-root",
                str(tmp_path / "archive"),
                "--capture-cli",
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1:])",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    stdout_path = Path(payload["cli_capture"]["artifacts"]["stdout"])
    assert stdout_path.read_text(encoding="utf-8").strip() == "[]"


def test_daemon_launch_writes_branch_local_process_artifacts(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = workspace_env["state_dir"] / "repo"
    repo.mkdir(parents=True)
    archive_root = workspace_env["archive_root"]
    monkeypatch.setattr(dev_loop, "_repo_root", lambda: repo)
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": False, "owner_count": 0, "owners": []},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )
    monkeypatch.setattr(dev_loop, "_socket_connectable", lambda port: True)

    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 4242

        def poll(self) -> int | None:
            return None

    def fake_start_daemon_process(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_file: object,
    ) -> FakeProcess:
        launched["command"] = command
        launched["cwd"] = cwd
        launched["env"] = env
        launched["log_file"] = log_file
        return FakeProcess()

    monkeypatch.setattr(dev_loop, "_start_daemon_process", fake_start_daemon_process)

    assert (
        dev_loop.main(
            [
                "--json",
                "--archive-root",
                str(archive_root),
                "--api-port",
                "9876",
                "--browser-capture-port",
                "9875",
                "--launch-daemon",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    launch = payload["daemon_launch"]
    command = launch["command"]
    assert command[:4] == [
        sys.executable,
        "-c",
        "from polylogue.daemon.cli import main; main()",
        "run",
    ]
    assert "--no-watch" in command
    assert command[command.index("--api-port") : command.index("--api-port") + 2] == ["--api-port", "9876"]
    assert command[command.index("--port") : command.index("--port") + 2] == ["--port", "9875"]
    assert launch["pid"] == 4242
    assert launch["api_ready"] is True
    assert launch["browser_capture_ready"] is True

    env = launched["env"]
    assert isinstance(env, dict)
    assert env["POLYLOGUE_ARCHIVE_ROOT"] == str(archive_root)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(Path(payload["preflight"]["repo_root"]))
    assert launched["cwd"] == Path(payload["preflight"]["repo_root"])

    artifacts = launch["artifacts"]
    assert Path(artifacts["pid"]).read_text(encoding="utf-8") == "4242\n"
    assert json.loads(Path(artifacts["summary"]).read_text(encoding="utf-8"))["pid"] == 4242
    assert json.loads(Path(artifacts["env"]).read_text(encoding="utf-8"))["POLYLOGUE_API_PORT"] == "9876"
    assert Path(artifacts["log"]).read_text(encoding="utf-8").startswith("\n# dev-loop launch")
    event_rows = [
        json.loads(line) for line in Path(artifacts["events"]).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [row["event_type"] for row in event_rows] == [
        "launch_requested",
        "process_spawned",
        "readiness_succeeded",
    ]
    assert event_rows[0]["run_id"] == payload["preflight"]["run_id"]
    assert event_rows[0]["archive_root"] == str(archive_root)
    assert event_rows[0]["payload"]["api_port"] == 9876
    assert event_rows[1]["payload"]["pid"] == 4242
    assert event_rows[2]["status"] == "ok"
    assert event_rows[2]["payload"]["api_ready"] is True


def test_daemon_launch_rejects_occupied_branch_local_ports(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = workspace_env["state_dir"] / "repo"
    repo.mkdir(parents=True)
    archive_root = workspace_env["archive_root"]
    monkeypatch.setattr(dev_loop, "_repo_root", lambda: repo)
    monkeypatch.setattr(dev_loop, "system_service_status", lambda: {"active": False})
    monkeypatch.setattr(
        dev_loop,
        "port_status",
        lambda port: {"port": port, "connectable": True, "owner_count": 1, "owners": [{"pid": 1234}]},
    )
    monkeypatch.setattr(
        dev_loop, "_git_value", lambda args, *, cwd: "feature/dev-loop" if args[0] == "branch" else "abc1234"
    )

    with pytest.raises(SystemExit) as exc:
        dev_loop.main(["--archive-root", str(archive_root), "--launch-daemon"])

    assert exc.value.code == 2
    assert "selected branch-local ports already have listeners" in capsys.readouterr().err
    run_log_dir = repo / ".cache" / "dev-loop" / "feature-dev-loop-abc1234-api8766-capture8765"
    event_path = run_log_dir / "dev-loop.events.jsonl"
    event_rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in event_rows] == ["launch_rejected"]
    assert event_rows[0]["status"] == "blocked"
    assert event_rows[0]["payload"]["occupied_ports"] == ["api port 8766", "browser_capture port 8765"]
