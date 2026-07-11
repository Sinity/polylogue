"""Behavioral proofs for the managed pytest ownership boundary."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
import tomllib

import devtools.pytest_supervisor as pytest_supervisor
from devtools.pytest_supervisor import (
    CGROUP_MODE_ENV,
    SupervisorLaunch,
    build_supervisor_launch,
    read_receipt,
    signal_process_identity,
    termination_request_path,
    user_systemd_available,
    write_termination_request,
)
from devtools.verify import PYTEST_CONTAINMENT_PATH, PYTEST_REPORT_PATH, ROOT, _run
from devtools.verify_runs import VerifyRun


def _wait_for(predicate: Callable[[], bool], *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate()


def _wait_for_receipt(path: Path, *, status: str | None = None) -> dict[str, object]:
    payload: dict[str, object] | None = None

    def _ready() -> bool:
        nonlocal payload
        candidate = read_receipt(path)
        if candidate is None:
            return False
        payload = candidate
        return status is None or candidate.get("status") == status

    _wait_for(_ready)
    assert payload is not None
    return payload


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _same_process_alive(identity: tuple[int, int | None]) -> bool:
    pid, start_ticks = identity
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        state = fields[0]
        current = int(fields[19])
    except (OSError, IndexError, ValueError):
        return False
    return state != "Z" and current == start_ticks


def _cgroup_processes(cgroup_path: str) -> set[int]:
    path = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "cgroup.procs"
    try:
        return {int(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()}
    except OSError:
        return set()


def _session_processes(pgid: int, sid: int) -> set[int]:
    processes: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
            state = fields[0]
            process_pgid = int(fields[2])
            process_sid = int(fields[3])
        except (OSError, IndexError, ValueError):
            continue
        if state != "Z" and process_pgid == pgid and process_sid == sid:
            processes.add(int(entry.name))
    return processes


def _kill_owned_scope(*, unit: str | None, pgid: int | None) -> None:
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if unit is not None:
        subprocess.run(
            ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", unit],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )


def _write_hanging_xdist_test(path: Path) -> None:
    path.write_text(
        "import json\n"
        "import os\n"
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n"
        "def test_hangs_with_signal_resistant_descendant():\n"
        "    child_code = (\n"
        "        'import json, os, signal, time\\n'\n"
        "        'from pathlib import Path\\n'\n"
        "        'for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):\\n'\n"
        "        '    signal.signal(sig, signal.SIG_IGN)\\n'\n"
        "        \"Path(os.environ['POLYLOGUE_STUBBORN_READY']).write_text(\"\n"
        "        \"json.dumps({'pid': os.getpid()}))\\n\"\n"
        "        'time.sleep(60)\\n'\n"
        "    )\n"
        "    subprocess.Popen(\n"
        "        [sys.executable, '-c', child_code],\n"
        "        env=os.environ.copy(),\n"
        "        stdout=subprocess.DEVNULL,\n"
        "        stderr=subprocess.DEVNULL,\n"
        "    )\n"
        "    deadline = time.monotonic() + 5\n"
        "    ready = Path(os.environ['POLYLOGUE_STUBBORN_READY'])\n"
        "    while not ready.exists() and time.monotonic() < deadline:\n"
        "        time.sleep(0.02)\n"
        "    assert ready.exists()\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )


def _write_streaming_xdist_test(path: Path, ready_path: Path) -> None:
    path.write_text(
        "import json\n"
        "import os\n"
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n"
        "def test_streams_forever():\n"
        "    child_code = (\n"
        "        'import signal, time\\n'\n"
        "        'for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):\\n'\n"
        "        '    signal.signal(sig, signal.SIG_IGN)\\n'\n"
        "        'time.sleep(60)\\n'\n"
        "    )\n"
        "    child = subprocess.Popen([sys.executable, '-c', child_code], start_new_session=True)\n"
        f"    Path({str(ready_path)!r}).write_text(\n"
        "        json.dumps({'worker_pid': os.getpid(), 'escaped_pid': child.pid})\n"
        "    )\n"
        "    while True:\n"
        "        print('fallback-controller-still-running', flush=True)\n"
        "        time.sleep(0.02)\n",
        encoding="utf-8",
    )


def _xdist_controller_cmd(path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:testmon",
        "-p",
        "no:randomly",
        "-p",
        "devtools.pytest_progress_plugin",
        f"--rootdir={ROOT}",
        "-n",
        "2",
        "-q",
        str(path),
    ]


def test_repository_pytest_timeout_policy_is_bounded_and_installed() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dev_dependencies = config["project"]["optional-dependencies"]["dev"]
    pytest_config = config["tool"]["pytest"]["ini_options"]

    assert any(dependency.startswith("pytest-timeout") for dependency in dev_dependencies)
    assert pytest_config["timeout"] == 300
    assert pytest_config["timeout_method"] == "signal"


def test_identity_signal_rejects_reused_pid_identity() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        start_ticks = _process_start_ticks(process.pid)
        assert start_ticks is not None
        assert signal_process_identity(process.pid, start_ticks + 1, signal.SIGKILL) is False
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=2)


def test_identity_signal_rechecks_after_opening_pidfd(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    observations = iter((41, 42))
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(pytest_supervisor, "_process_start_ticks", lambda _pid: next(observations))
    monkeypatch.setattr(os, "pidfd_open", lambda _pid, _flags: read_fd)
    monkeypatch.setattr(
        signal,
        "pidfd_send_signal",
        lambda pidfd, sig: sent.append((pidfd, sig)),
    )
    try:
        assert signal_process_identity(1234, 41, signal.SIGKILL) is False
        assert sent == []
    finally:
        os.close(write_fd)


def test_non_linux_platform_refuses_unsafe_group_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="requires Linux process identities"):
        build_supervisor_launch(
            [sys.executable, "-c", "pass"],
            owner_pid=os.getpid(),
            timeout_s=1,
            term_grace_s=0.1,
            receipt_path=tmp_path / "containment.json",
            run_id="non-linux",
        )


def test_supervisor_rejects_owner_identity_captured_before_launch(tmp_path: Path) -> None:
    owner_start_ticks = _process_start_ticks(os.getpid())
    if owner_start_ticks is None:
        pytest.skip("kernel process start identity is unavailable")
    receipt_path = tmp_path / "owner-reuse-containment.json"
    env = os.environ.copy()
    env[CGROUP_MODE_ENV] = "off"
    launch = build_supervisor_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        owner_pid=os.getpid(),
        timeout_s=10,
        term_grace_s=0.1,
        receipt_path=receipt_path,
        run_id="owner-reuse",
        env=env,
    )
    argv = list(launch.argv)
    ticks_index = argv.index("--owner-start-ticks") + 1
    assert int(argv[ticks_index]) == owner_start_ticks
    argv[ticks_index] = str(owner_start_ticks + 1)

    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _stdout, _stderr = process.communicate(timeout=3)
    final = _wait_for_receipt(receipt_path, status="terminated")

    assert process.returncode == 125
    assert final["owner_start_ticks"] == owner_start_ticks + 1
    assert final["termination_reason"] == f"pytest runner owner pid {os.getpid()} exited"
    assert final["controller_group_alive"] is False


def test_supervisor_kills_controller_when_receipt_publication_fails(tmp_path: Path) -> None:
    ready_path = tmp_path / "setup-controller.json"
    receipt_path = tmp_path / "unpublished-containment.json"
    controller_code = (
        "import json, os, time\n"
        "from pathlib import Path\n"
        "fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()\n"
        f"Path({str(ready_path)!r}).write_text(json.dumps({{'pid': os.getpid(), 'start_ticks': int(fields[19])}}))\n"
        "time.sleep(60)\n"
    )
    supervisor_code = (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "import devtools.pytest_supervisor as supervisor\n"
        f"ready = Path({str(ready_path)!r})\n"
        "def fail_write(_path, _payload):\n"
        "    deadline = time.monotonic() + 2\n"
        "    while not ready.exists() and time.monotonic() < deadline:\n"
        "        time.sleep(0.01)\n"
        "    raise OSError('injected receipt failure')\n"
        "supervisor._write_json = fail_write\n"
        f"command = [sys.executable, '-c', {controller_code!r}]\n"
        "raise SystemExit(supervisor.supervise(\n"
        "    command,\n"
        f"    receipt_path=Path({str(receipt_path)!r}),\n"
        "    owner_pid=os.getppid(),\n"
        "    owner_start_ticks=supervisor._process_start_ticks(os.getppid()),\n"
        "    timeout_s=10,\n"
        "    term_grace_s=0.1,\n"
        "    mode='process-group',\n"
        "    unit=None,\n"
        "    runtime_cap_s=None,\n"
        "))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    process = subprocess.run(
        [sys.executable, "-c", supervisor_code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    controller = json.loads(ready_path.read_text(encoding="utf-8"))
    controller_identity = (int(controller["pid"]), int(controller["start_ticks"]))
    assert process.returncode != 0
    assert "injected receipt failure" in process.stderr
    _wait_for(lambda: not _same_process_alive(controller_identity))


def test_managed_runner_retains_responsible_node_for_per_test_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hanging_test = tmp_path / "test_deliberate_per_test_timeout.py"
    hanging_test.write_text(
        "import time\n\ndef test_deliberate_per_test_timeout():\n    time.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "10")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "10")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TERM_GRACE_S", "0.2")
    monkeypatch.setenv("POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S", "0")
    run = VerifyRun(tier="containment-per-test", argv=[str(hanging_test)], git_head=None, root=tmp_path)
    report_path = tmp_path / PYTEST_REPORT_PATH
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "devtools.pytest_progress_plugin",
        "-p",
        "no:testmon",
        "-p",
        "no:randomly",
        "--timeout=0.2",
        "--timeout-method=signal",
        f"--rootdir={ROOT}",
        "--json-report",
        f"--json-report-file={report_path}",
        "-n",
        "0",
        "-q",
        str(hanging_test),
    ]

    rc, _elapsed, metadata = _run("pytest per-test timeout", cmd, cwd=str(ROOT), run=run)

    artifacts = run.run_dir / "steps" / "01-pytest-per-test-timeout"
    output = (artifacts / "output.log").read_text(encoding="utf-8")
    containment = json.loads((artifacts / "containment.json").read_text(encoding="utf-8"))
    current_containment = json.loads((tmp_path / PYTEST_CONTAINMENT_PATH).read_text(encoding="utf-8"))
    assert rc != 0
    assert "test_deliberate_per_test_timeout" in output
    assert "Timeout" in output
    assert containment["status"] == "finished"
    assert containment["controller_returncode"] == rc
    assert current_containment == containment
    assert metadata["containment_mode"] in {"process-group", "systemd-scope"}
    assert metadata["report_status"] == "present"


def test_auto_mode_retries_when_systemd_scope_launch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_systemd_run = tmp_path / "systemd-run"
    fake_systemd_run.write_text("#!/bin/sh\necho 'scope denied' >&2\nexit 42\n", encoding="utf-8")
    fake_systemd_run.chmod(0o755)
    real_which = shutil.which

    def _which(name: str, path: str | None = None) -> str | None:
        if name == "systemd-run":
            return str(fake_systemd_run)
        return real_which(name, path=path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pytest_supervisor, "user_systemd_available", lambda _env=None: True)
    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "3")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "3")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TERM_GRACE_S", "0.1")
    monkeypatch.setenv("POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S", "0")

    rc, _elapsed, metadata = _run(
        "pytest auto fallback",
        [sys.executable, "-c", "print('fallback-controller-finished')"],
    )

    output = (tmp_path / ".cache/verify/current-pytest-output.log").read_text(encoding="utf-8")
    receipt = json.loads((tmp_path / PYTEST_CONTAINMENT_PATH).read_text(encoding="utf-8"))
    assert rc == 0
    assert metadata["containment_mode"] == "process-group"
    assert receipt["mode"] == "process-group"
    assert "scope denied" in output
    assert "systemd scope launch failed; retrying" in output
    assert "fallback-controller-finished" in output


def test_whole_run_deadline_bounds_stalled_supervisor_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_systemd_run = tmp_path / "systemd-run"
    fake_systemd_run.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
    fake_systemd_run.chmod(0o755)
    real_which = shutil.which

    def _which(name: str, path: str | None = None) -> str | None:
        if name == "systemd-run":
            return str(fake_systemd_run)
        return real_which(name, path=path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pytest_supervisor, "user_systemd_available", lambda _env=None: True)
    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "0.2")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "10")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TERM_GRACE_S", "0.1")
    monkeypatch.setenv("POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S", "0")

    rc, elapsed, metadata = _run(
        "pytest stalled startup",
        [sys.executable, "-c", "raise AssertionError('controller must not start')"],
    )

    receipt = json.loads((tmp_path / PYTEST_CONTAINMENT_PATH).read_text(encoding="utf-8"))
    output = (tmp_path / ".cache/verify/current-pytest-output.log").read_text(encoding="utf-8")
    assert rc == 124
    assert elapsed < 2
    assert metadata["termination_reason"] == "pytest runtime exceeded 0.2s"
    assert receipt["status"] == "terminated"
    assert receipt["startup_failure"] is True
    assert receipt["runner_forced_cleanup"] is True
    assert receipt["controller_command"][-1] == "raise AssertionError('controller must not start')"
    assert "pytest runtime exceeded 0.2s" in output


@pytest.mark.load_sensitive
def test_runner_deadline_cleans_group_when_fallback_supervisor_is_sigkilled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_path = tmp_path / "fallback-worker.json"
    hanging_test = tmp_path / "test_fallback_supervisor_death.py"
    _write_streaming_xdist_test(hanging_test, ready_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CGROUP_MODE_ENV, "off")
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.05")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "4")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "10")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TERM_GRACE_S", "0.1")
    monkeypatch.setenv("POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S", "0")
    run = VerifyRun(tier="fallback-supervisor-death", argv=[str(hanging_test)], git_head=None, root=tmp_path)
    receipt_path = run.run_dir / "steps" / "01-pytest-fallback-supervisor-death" / "containment.json"
    observed_group: tuple[int, int] | None = None
    owned_identities: set[tuple[int, int | None]] = set()
    errors: list[BaseException] = []

    def _kill_supervisor() -> None:
        nonlocal observed_group, owned_identities
        try:
            running = _wait_for_receipt(receipt_path, status="running")
            _wait_for(ready_path.exists)
            controller_pid = running["controller_pid"]
            controller_pgid = running["controller_pgid"]
            controller_sid = running["controller_sid"]
            supervisor_pid = running["supervisor_pid"]
            supervisor_start_ticks = running["supervisor_start_ticks"]
            assert isinstance(controller_pid, int)
            assert isinstance(controller_pgid, int)
            assert isinstance(controller_sid, int)
            assert isinstance(supervisor_pid, int)
            assert isinstance(supervisor_start_ticks, int)
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            worker_pid = int(ready["worker_pid"])
            escaped_pid = int(ready["escaped_pid"])
            owned_pids = _session_processes(controller_pgid, controller_sid)
            assert {controller_pid, worker_pid} <= owned_pids
            assert escaped_pid not in owned_pids
            assert len(owned_pids) >= 3
            observed_group = (controller_pgid, controller_sid)
            owned_identities = {(pid, _process_start_ticks(pid)) for pid in owned_pids} | {
                (escaped_pid, _process_start_ticks(escaped_pid))
            }
            assert signal_process_identity(supervisor_pid, supervisor_start_ticks, signal.SIGKILL)
        except BaseException as exc:
            errors.append(exc)

    killer = threading.Thread(target=_kill_supervisor, daemon=True)
    killer.start()
    cmd = _xdist_controller_cmd(hanging_test)
    cmd.insert(-1, "-s")
    rc, elapsed, metadata = _run("pytest fallback supervisor death", cmd, cwd=str(ROOT), run=run)
    killer.join(timeout=5)

    assert not killer.is_alive()
    assert errors == []
    assert rc == 124
    assert elapsed < 7
    assert metadata["termination_reason"] == "pytest runtime exceeded 4s"
    assert metadata["containment_mode"] == "process-group"
    final = _wait_for_receipt(receipt_path, status="terminated")
    assert final["runner_forced_cleanup"] is True
    assert final["exit_code"] == 124
    _wait_for(lambda: not any(_same_process_alive(identity) for identity in owned_identities))
    assert observed_group is not None
    controller_pgid, controller_sid = observed_group
    _wait_for(lambda: not _session_processes(controller_pgid, controller_sid))
    output = (run.run_dir / "steps" / "01-pytest-fallback-supervisor-death" / "output.log").read_text(encoding="utf-8")
    events = (run.run_dir / "steps" / "01-pytest-fallback-supervisor-death" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "test_streams_forever" in events
    assert "pytest runtime exceeded 4s" in output


@pytest.mark.load_sensitive
def test_controller_sigkill_clears_exact_owned_xdist_cgroup(tmp_path: Path) -> None:
    if not user_systemd_available():
        pytest.skip("user systemd is unavailable; process-group fallback is covered by timeout tests")

    ready_path = tmp_path / "stubborn-child.json"
    hanging_test = tmp_path / "test_deliberate_controller_kill.py"
    _write_hanging_xdist_test(hanging_test)
    receipt_path = tmp_path / "containment.json"
    env = os.environ.copy()
    env[CGROUP_MODE_ENV] = "require"
    env["POLYLOGUE_STUBBORN_READY"] = str(ready_path)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    controller_cmd = _xdist_controller_cmd(hanging_test)
    launch = build_supervisor_launch(
        controller_cmd,
        owner_pid=os.getpid(),
        timeout_s=30,
        term_grace_s=0.25,
        receipt_path=receipt_path,
        run_id="sigkill-regression",
        env=env,
    )
    process = subprocess.Popen(
        launch.argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    unrelated_identity = (unrelated.pid, _process_start_ticks(unrelated.pid))
    controller_pgid: int | None = None
    try:
        running = _wait_for_receipt(receipt_path, status="running")
        controller_pid_raw = running["controller_pid"]
        controller_pgid_raw = running["controller_pgid"]
        assert isinstance(controller_pid_raw, int)
        assert isinstance(controller_pgid_raw, int)
        controller_pid = controller_pid_raw
        controller_pgid = controller_pgid_raw
        cgroup_path = str(running["cgroup_path"])
        unit_state = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                str(launch.unit),
                "--property=ControlGroup",
                "--property=KillMode",
                "--property=RuntimeMaxUSec",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
        assert "KillMode=control-group" in unit_state
        assert "RuntimeMaxUSec=" in unit_state and "RuntimeMaxUSec=infinity" not in unit_state
        assert f"ControlGroup={cgroup_path}" in unit_state
        _wait_for(ready_path.exists)
        stubborn_pid = int(json.loads(ready_path.read_text(encoding="utf-8"))["pid"])
        _wait_for(lambda: stubborn_pid in _cgroup_processes(cgroup_path))
        owned_pids = _cgroup_processes(cgroup_path)
        assert {process.pid, controller_pid, stubborn_pid} <= owned_pids
        assert len(owned_pids) >= 5  # supervisor + controller + two xdist workers + stubborn child
        owned_identities = {(pid, _process_start_ticks(pid)) for pid in owned_pids}

        cleanup_started_at = time.monotonic()
        cleanup_deadline = cleanup_started_at + 5.0
        os.kill(controller_pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=max(0.001, cleanup_deadline - time.monotonic()))
        _wait_for(
            lambda: not any(_same_process_alive(identity) for identity in owned_identities),
            timeout_s=max(0.0, cleanup_deadline - time.monotonic()),
        )
        _wait_for(
            lambda: not _cgroup_processes(cgroup_path),
            timeout_s=max(0.0, cleanup_deadline - time.monotonic()),
        )
        assert time.monotonic() <= cleanup_deadline

        final = _wait_for_receipt(receipt_path, status="terminated")
        assert process.returncode == 137
        assert final["controller_returncode"] == -signal.SIGKILL
        assert final["signals_sent"] == ["SIGTERM", "SIGKILL"]
        assert final["escalated_to_sigkill"] is True
        assert final["controller_group_alive"] is False
        assert _same_process_alive(unrelated_identity)
        assert unrelated.poll() is None
        assert "pytest controller exited by signal SIGKILL" in stderr
        assert stdout == ""
    finally:
        if process.poll() is None:
            write_termination_request(launch.request_path, reason="regression cleanup", exit_code=125)
            os.kill(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        _kill_owned_scope(unit=launch.unit, pgid=controller_pgid)
        if unrelated.poll() is None:
            unrelated.kill()
        unrelated.wait(timeout=2)


@pytest.mark.load_sensitive
def test_owner_sigkill_clears_exact_owned_xdist_cgroup(tmp_path: Path) -> None:
    if not user_systemd_available():
        pytest.skip("user systemd is unavailable; owner-watch fallback is Linux process-group only")

    ready_path = tmp_path / "owner-stubborn-child.json"
    hanging_test = tmp_path / "test_deliberate_owner_kill.py"
    receipt_path = tmp_path / "owner-containment.json"
    _write_hanging_xdist_test(hanging_test)
    env = os.environ.copy()
    env[CGROUP_MODE_ENV] = "require"
    env["POLYLOGUE_STUBBORN_READY"] = str(ready_path)
    env["POLYLOGUE_OWNER_RECEIPT"] = str(receipt_path)
    env["POLYLOGUE_OWNER_COMMAND"] = json.dumps(_xdist_controller_cmd(hanging_test))
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    owner_script = (
        "import json, os, subprocess, time\n"
        "from pathlib import Path\n"
        "from devtools.pytest_supervisor import build_supervisor_launch\n"
        "command = json.loads(os.environ['POLYLOGUE_OWNER_COMMAND'])\n"
        "receipt = Path(os.environ['POLYLOGUE_OWNER_RECEIPT'])\n"
        "launch = build_supervisor_launch(command, owner_pid=os.getpid(), timeout_s=30, term_grace_s=0.25, "
        "receipt_path=receipt, run_id='owner-sigkill-regression', env=os.environ)\n"
        "subprocess.Popen(launch.argv, cwd=os.environ['POLYLOGUE_REPO_ROOT'], env=os.environ.copy(), "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "time.sleep(60)\n"
    )
    env["POLYLOGUE_REPO_ROOT"] = str(ROOT)
    owner = subprocess.Popen([sys.executable, "-c", owner_script], cwd=ROOT, env=env, start_new_session=True)
    unit: str | None = None
    controller_pgid: int | None = None
    try:
        running = _wait_for_receipt(receipt_path, status="running")
        unit = str(running["unit"])
        cgroup_path = str(running["cgroup_path"])
        controller_pgid_raw = running["controller_pgid"]
        assert isinstance(controller_pgid_raw, int)
        controller_pgid = controller_pgid_raw
        _wait_for(ready_path.exists)
        stubborn_pid = int(json.loads(ready_path.read_text(encoding="utf-8"))["pid"])
        _wait_for(lambda: stubborn_pid in _cgroup_processes(cgroup_path))
        owned_identities = {(pid, _process_start_ticks(pid)) for pid in _cgroup_processes(cgroup_path)}
        assert len(owned_identities) >= 4

        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=2)
        final = _wait_for_receipt(receipt_path, status="terminated")
        _wait_for(lambda: not any(_same_process_alive(identity) for identity in owned_identities))
        _wait_for(lambda: not _cgroup_processes(cgroup_path))

        assert final["termination_reason"] == f"pytest runner owner pid {owner.pid} exited"
        assert final["exit_code"] == 125
        assert final["signals_sent"] == ["SIGTERM", "SIGKILL"]
        assert final["controller_group_alive"] is False
    finally:
        if owner.poll() is None:
            os.kill(owner.pid, signal.SIGKILL)
            owner.wait(timeout=2)
        _kill_owned_scope(unit=unit, pgid=controller_pgid)


def test_runner_bounds_post_supervisor_pipe_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_path = tmp_path / "escaped-pipe-child.json"

    def _fake_launch(
        controller_cmd: Sequence[str],
        *,
        owner_pid: int,
        timeout_s: float,
        term_grace_s: float,
        receipt_path: Path,
        run_id: str | None,
        env: Mapping[str, str] | None = None,
    ) -> SupervisorLaunch:
        del controller_cmd, owner_pid, timeout_s, term_grace_s, run_id, env
        receipt_path = receipt_path.absolute()
        child_code = (
            "import json, os, signal, time\n"
            "from pathlib import Path\n"
            "for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, signal.SIG_IGN)\n"
            "fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()\n"
            f"Path({str(ready_path)!r}).write_text(json.dumps({{'pid': os.getpid(), 'start_ticks': int(fields[19])}}))\n"
            "time.sleep(60)\n"
        )
        fake_supervisor = (
            "import json, os, subprocess, sys\n"
            "from pathlib import Path\n"
            f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True)\n"
            "fields = Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ', 1)[1].split()\n"
            "self_fields = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()\n"
            "payload = {'schema_version': 1, 'status': 'finished', 'supervisor_pid': os.getpid(), "
            "'supervisor_start_ticks': int(self_fields[19]), 'controller_pid': child.pid, "
            "'controller_start_ticks': int(fields[19]), 'controller_pgid': int(fields[2]), "
            "'controller_sid': int(fields[3]), 'controller_returncode': 0, 'exit_code': 0, "
            "'termination_reason': None, 'signals_sent': [], 'escalated_to_sigkill': False, "
            "'controller_group_alive': True, 'mode': 'process-group', 'unit': None, "
            "'cgroup_path': None, 'cgroup_owned': False}\n"
            f"Path({str(receipt_path)!r}).parent.mkdir(parents=True, exist_ok=True)\n"
            f"Path({str(receipt_path)!r}).write_text(json.dumps(payload))\n"
        )
        return SupervisorLaunch(
            [sys.executable, "-c", fake_supervisor],
            receipt_path,
            termination_request_path(receipt_path),
            "process-group",
            None,
            None,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "10")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "0")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TERM_GRACE_S", "0.1")
    monkeypatch.setenv("POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S", "0")
    monkeypatch.setattr("devtools.verify.build_supervisor_launch", _fake_launch)

    child_identity: tuple[int, int] | None = None
    try:
        rc, elapsed, metadata = _run("pytest escaped pipe", [sys.executable, "-c", "pass"])

        child = json.loads(ready_path.read_text(encoding="utf-8"))
        child_identity = (int(child["pid"]), int(child["start_ticks"]))
        assert rc == 125
        assert elapsed < 3
        assert metadata["termination_reason"] == "pytest supervisor exited while owned output pipes remained open"
        assert metadata["containment_mode"] == "process-group"
        assert not _same_process_alive(child_identity)
    finally:
        if child_identity is None and ready_path.exists():
            child = json.loads(ready_path.read_text(encoding="utf-8"))
            child_identity = (int(child["pid"]), int(child["start_ticks"]))
        if child_identity is not None:
            signal_process_identity(child_identity[0], child_identity[1], signal.SIGKILL)
