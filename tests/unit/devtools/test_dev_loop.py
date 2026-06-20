from __future__ import annotations

import json
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
    assert Path(str(payload["artifacts"]["preflight_json"])).is_file()
    assert payload["artifacts"]["daemon_log"].endswith(
        ".cache/dev-loop/feature-dev-loop-abc1234-api9999-capture9998/polylogued.log"
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
