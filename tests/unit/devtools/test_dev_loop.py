from __future__ import annotations

import json
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
    repo.mkdir()

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
    assert "polylogued run --api-port 9999 --port 9998" in payload["commands"]["run_daemon"]
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

    command = [
        sys.executable,
        "-c",
        "import os; print(os.environ['POLYLOGUE_DEV_LOOP_RUN_ID'])",
    ]
    assert (
        dev_loop.main(
            [
                "--json",
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
    assert json.loads(summary_path.read_text(encoding="utf-8"))["exit_code"] == 0


def test_cli_capture_rejects_missing_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        dev_loop.main(["--capture-cli"])
    assert exc.value.code == 2
    assert "capture command must not be empty" in capsys.readouterr().err


def test_daemon_launch_writes_branch_local_process_artifacts(
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
                str(tmp_path / "archive"),
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
    assert command[:2] == ["polylogued", "run"]
    assert "--no-watch" in command
    assert command[command.index("--api-port") : command.index("--api-port") + 2] == ["--api-port", "9876"]
    assert command[command.index("--port") : command.index("--port") + 2] == ["--port", "9875"]
    assert launch["pid"] == 4242
    assert launch["api_ready"] is True
    assert launch["browser_capture_ready"] is True

    env = launched["env"]
    assert isinstance(env, dict)
    assert env["POLYLOGUE_ARCHIVE_ROOT"] == str(tmp_path / "archive")
    assert launched["cwd"] == Path(payload["preflight"]["repo_root"])

    artifacts = launch["artifacts"]
    assert Path(artifacts["pid"]).read_text(encoding="utf-8") == "4242\n"
    assert json.loads(Path(artifacts["summary"]).read_text(encoding="utf-8"))["pid"] == 4242
    assert json.loads(Path(artifacts["env"]).read_text(encoding="utf-8"))["POLYLOGUE_API_PORT"] == "9876"
    assert Path(artifacts["log"]).read_text(encoding="utf-8").startswith("\n# dev-loop launch")


def test_daemon_launch_rejects_occupied_branch_local_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        dev_loop.main(["--archive-root", str(tmp_path / "archive"), "--launch-daemon"])

    assert exc.value.code == 2
    assert "selected branch-local ports already have listeners" in capsys.readouterr().err
