"""External lifecycle supervisor for pytest controller process groups.

The verify process is not a sufficient ownership boundary: if it or pytest is
killed with SIGKILL, Python cleanup handlers cannot run.  This module is
launched as a separate process for every managed pytest step.  It owns the
pytest controller's process group, watches the invoking devtools process via a
pidfd, enforces the whole-run deadline, and persists a containment receipt.

On Sinnix, the supervisor itself runs in a unique transient user-systemd scope
under the configured build slice.  ``RuntimeMaxSec`` and
``KillMode=control-group`` provide the final cgroup boundary if both ordinary
Python cleanup layers are lost.  Other Linux hosts retain the external
supervisor and process-group guarantees without claiming cgroup isolation.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CGROUP_MODE_ENV = "POLYLOGUE_VERIFY_PYTEST_CGROUP"
_SYSTEMD_MODES = frozenset({"auto", "off", "require"})
_POLL_INTERVAL_S = 0.05
_PR_SET_CHILD_SUBREAPER = 36


@dataclass(frozen=True)
class SupervisorLaunch:
    """Command and ownership metadata for one external pytest supervisor."""

    argv: list[str]
    receipt_path: Path
    request_path: Path
    mode: str
    unit: str | None
    runtime_cap_s: float | None
    fallback_argv: list[str] | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_receipt(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def update_receipt(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_receipt(path) or {}
    payload.update(updates)
    _write_json(path, payload)
    return payload


def termination_request_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.name}.terminate")


def write_termination_request(path: Path, *, reason: str, exit_code: int = 124) -> None:
    _write_json(
        path,
        {
            "requested_at": utc_now(),
            "requested_by_pid": os.getpid(),
            "reason": reason,
            "exit_code": int(exit_code),
        },
    )


def _read_termination_request(path: Path) -> dict[str, Any] | None:
    return read_receipt(path)


def _cgroup_path(pid: int) -> str | None:
    try:
        rows = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for row in rows:
        parts = row.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return parts[2]
    return None


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _linux_proc_available() -> bool:
    return sys.platform == "linux" and Path("/proc/self/stat").is_file()


def signal_process_identity(pid: int, start_ticks: int | None, sig: signal.Signals) -> bool:
    """Signal one process only when its recorded kernel identity still matches."""
    if start_ticks is None or _process_start_ticks(pid) != start_ticks:
        return False
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is not None and pidfd_send_signal is not None:
        try:
            pidfd = int(pidfd_open(pid, 0))
        except OSError:
            return False
        try:
            # pidfd_open pins an identity, but the PID may have been reused
            # between the first /proc read and opening it. Validate once more
            # after the pin; subsequent reuse cannot retarget this pidfd.
            if _process_start_ticks(pid) != start_ticks:
                return False
            pidfd_send_signal(pidfd, sig)
            return True
        except ProcessLookupError:
            return False
        finally:
            with contextlib.suppress(OSError):
                os.close(pidfd)
    if _process_start_ticks(pid) != start_ticks:
        return False
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def _process_group_identities(
    *,
    pgid: int,
    sid: int,
    leader_start_ticks: int | None,
) -> list[tuple[int, int]]:
    """Snapshot an owned session/group as PID identities, never process names."""
    if not _linux_proc_available():
        return []
    current_leader_ticks = _process_start_ticks(sid)
    if current_leader_ticks is not None and current_leader_ticks != leader_start_ticks:
        return []
    identities: list[tuple[int, int]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
            state = fields[0]
            process_pgid = int(fields[2])
            process_sid = int(fields[3])
            start_ticks = int(fields[19])
        except (OSError, IndexError, ValueError):
            continue
        if state != "Z" and process_pgid == pgid and process_sid == sid:
            identities.append((int(entry.name), start_ticks))
    return identities


def signal_process_group_identities(
    *,
    pgid: int,
    sid: int,
    leader_start_ticks: int | None,
    sig: signal.Signals,
) -> int:
    """Signal the exact current members of a recorded process group via pidfds."""
    return sum(
        signal_process_identity(pid, start_ticks, sig)
        for pid, start_ticks in _process_group_identities(
            pgid=pgid,
            sid=sid,
            leader_start_ticks=leader_start_ticks,
        )
    )


def signal_owned_process_group(
    *,
    pgid: int,
    sid: int,
    leader_start_ticks: int | None,
    sig: signal.Signals,
) -> int:
    """Signal exact members of an owned Linux process group via pidfds."""
    if not _linux_proc_available():
        return 0
    return signal_process_group_identities(
        pgid=pgid,
        sid=sid,
        leader_start_ticks=leader_start_ticks,
        sig=sig,
    )


def _descendant_identities(root_pid: int) -> list[tuple[int, int]]:
    if not _linux_proc_available():
        return []
    rows: dict[int, tuple[int, str, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
            rows[int(entry.name)] = (int(fields[1]), fields[0], int(fields[19]))
        except (OSError, IndexError, ValueError):
            continue
    descendants: set[int] = set()
    frontier = {root_pid}
    while frontier:
        children = {pid for pid, (ppid, _state, _start) in rows.items() if ppid in frontier and pid not in descendants}
        descendants.update(children)
        frontier = children
    return [(pid, rows[pid][2]) for pid in descendants if rows[pid][1] != "Z"]


def _owned_identities(
    *,
    supervisor_pid: int,
    pgid: int,
    sid: int,
    leader_start_ticks: int | None,
) -> list[tuple[int, int]]:
    return sorted(
        set(_descendant_identities(supervisor_pid))
        | set(
            _process_group_identities(
                pgid=pgid,
                sid=sid,
                leader_start_ticks=leader_start_ticks,
            )
        )
    )


def enable_child_subreaper() -> bool:
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return int(libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)) == 0
    except (AttributeError, OSError):
        return False


def reap_exited_children() -> None:
    """Reap controller descendants adopted through the Linux subreaper."""
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def signal_descendant_identities(root_pid: int, sig: signal.Signals) -> int:
    """Signal the exact live descendants of a Linux subreaper via pidfds."""
    return sum(signal_process_identity(pid, start_ticks, sig) for pid, start_ticks in _descendant_identities(root_pid))


def _systemd_mode(env: Mapping[str, str]) -> str:
    mode = env.get(CGROUP_MODE_ENV, "auto").strip().lower() or "auto"
    if mode not in _SYSTEMD_MODES:
        return "auto"
    return mode


def user_systemd_available(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    if _systemd_mode(values) == "off" or sys.platform != "linux":
        return False
    runtime_dir = values.get("XDG_RUNTIME_DIR")
    return bool(
        runtime_dir
        and Path(runtime_dir, "systemd", "private").is_socket()
        and shutil.which("systemd-run", path=values.get("PATH"))
    )


def _runtime_inventory_path(env: Mapping[str, str]) -> Path | None:
    configured = env.get("SINNIX_RUNTIME_INVENTORY_FILE")
    candidates = (
        Path(configured) if configured else None,
        Path("/etc/sinnix/runtime-inventory.json"),
        Path("/run/current-system/etc/sinnix/runtime-inventory.json"),
    )
    return next((path for path in candidates if path is not None and path.is_file()), None)


def _sinnix_build_scope(env: Mapping[str, str]) -> tuple[str | None, list[str]]:
    """Read the existing Sinnix build class instead of duplicating its policy."""
    path = _runtime_inventory_path(env)
    if path is None:
        return None, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        build = payload["commandClasses"]["build"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None, []
    if not isinstance(build, dict):
        return None, []
    raw_slice = build.get("slice")
    slice_name = (
        raw_slice
        if isinstance(raw_slice, str) and re.fullmatch(r"[A-Za-z0-9_.@:-]+\.slice", raw_slice) and "/" not in raw_slice
        else None
    )
    raw_properties = build.get("systemdProperties")
    properties: list[str] = []
    if isinstance(raw_properties, dict):
        for name, value in sorted(raw_properties.items()):
            if (
                name in {"KillMode", "RuntimeMaxSec", "SendSIGKILL", "TimeoutStopSec"}
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(name)) is None
            ):
                continue
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (int, float, str)):
                rendered = str(value)
            else:
                continue
            properties.append(f"--property={name}={rendered}")
    return slice_name, properties


def _unit_name(run_id: str | None) -> str:
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id or "run").strip("-.") or "run"
    safe_run = safe_run[:80]
    return f"polylogue-pytest-{safe_run}-{os.getpid()}-{time.monotonic_ns()}.scope"


def build_supervisor_launch(
    controller_cmd: Sequence[str],
    *,
    owner_pid: int,
    timeout_s: float,
    term_grace_s: float,
    receipt_path: Path,
    run_id: str | None,
    env: Mapping[str, str] | None = None,
) -> SupervisorLaunch:
    """Build the Linux supervisor command and optional owned cgroup scope."""
    values = dict(os.environ if env is None else env)
    if sys.platform != "linux":
        raise RuntimeError("managed pytest containment requires Linux process identities")
    receipt_path = receipt_path.absolute()
    request_path = termination_request_path(receipt_path)
    owner_start_ticks = _process_start_ticks(owner_pid)
    systemd_available = user_systemd_available(values)
    requested_mode = _systemd_mode(values)
    if requested_mode == "require" and not systemd_available:
        raise RuntimeError("user systemd is required for pytest containment but is unavailable")

    mode = "systemd-scope" if systemd_available else "process-group"
    unit = _unit_name(run_id) if systemd_available else None
    runtime_cap_s = timeout_s + term_grace_s + 5.0 if systemd_available and timeout_s > 0 else None

    def _supervisor_argv(
        launch_mode: str,
        launch_unit: str | None,
        launch_runtime_cap_s: float | None,
    ) -> list[str]:
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--receipt",
            str(receipt_path),
            "--owner-pid",
            str(owner_pid),
        ]
        if owner_start_ticks is not None:
            argv.extend(("--owner-start-ticks", str(owner_start_ticks)))
        argv.extend(
            (
                "--timeout-s",
                f"{timeout_s:g}",
                "--term-grace-s",
                f"{term_grace_s:g}",
                "--mode",
                launch_mode,
            )
        )
        if launch_unit is not None:
            argv.extend(("--unit", launch_unit))
        if launch_runtime_cap_s is not None:
            argv.extend(("--runtime-cap-s", f"{launch_runtime_cap_s:g}"))
        argv.extend(("--", *controller_cmd))
        return argv

    supervisor_argv = _supervisor_argv(mode, unit, runtime_cap_s)
    if not systemd_available:
        return SupervisorLaunch(supervisor_argv, receipt_path, request_path, mode, None, None)

    systemd_run = shutil.which("systemd-run", path=values.get("PATH"))
    assert systemd_run is not None
    slice_name, inherited_properties = _sinnix_build_scope(values)
    systemd_argv = [
        systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--same-dir",
        f"--unit={unit}",
        "--property=KillMode=control-group",
        "--property=SendSIGKILL=yes",
        f"--property=TimeoutStopSec={term_grace_s:g}s",
        *inherited_properties,
    ]
    if runtime_cap_s is not None:
        systemd_argv.append(f"--property=RuntimeMaxSec={runtime_cap_s:g}s")
    if slice_name is not None:
        systemd_argv.append(f"--slice={slice_name}")
    systemd_argv.extend(("--", *supervisor_argv))
    fallback_argv = _supervisor_argv("process-group", None, None) if requested_mode == "auto" else None
    return SupervisorLaunch(
        systemd_argv,
        receipt_path,
        request_path,
        mode,
        unit,
        runtime_cap_s,
        fallback_argv,
    )


class _OwnerWatch:
    def __init__(self, pid: int, start_ticks: int | None) -> None:
        self.pid = pid
        self.start_ticks = start_ticks
        self.pidfd: int | None = None
        self.poller: select.poll | None = None
        self.identity_mismatch = start_ticks is not None and _process_start_ticks(pid) != start_ticks
        if self.identity_mismatch:
            return
        pidfd_open = getattr(os, "pidfd_open", None)
        if pidfd_open is not None:
            with contextlib.suppress(OSError):
                self.pidfd = int(pidfd_open(pid, 0))
                if start_ticks is not None and _process_start_ticks(pid) != start_ticks:
                    os.close(self.pidfd)
                    self.pidfd = None
                    self.identity_mismatch = True
                    return
                self.poller = select.poll()
                self.poller.register(self.pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)

    def gone(self) -> bool:
        if self.identity_mismatch:
            return True
        if self.poller is not None and self.poller.poll(0):
            return True
        if self.start_ticks is not None:
            return _process_start_ticks(self.pid) != self.start_ticks
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def close(self) -> None:
        if self.pidfd is not None:
            with contextlib.suppress(OSError):
                os.close(self.pidfd)


def _owned_processes_exist(
    *,
    supervisor_pid: int,
    pgid: int,
    sid: int,
    leader_start_ticks: int | None,
) -> bool:
    if not _linux_proc_available():
        return False
    return bool(
        _owned_identities(
            supervisor_pid=supervisor_pid,
            pgid=pgid,
            sid=sid,
            leader_start_ticks=leader_start_ticks,
        )
    )


def _signal_owned_processes(
    *,
    supervisor_pid: int,
    pgid: int,
    sid: int,
    leader_start_ticks: int | None,
    sig: signal.Signals,
) -> int:
    if not _linux_proc_available():
        return 0
    return sum(
        signal_process_identity(pid, start_ticks, sig)
        for pid, start_ticks in _owned_identities(
            supervisor_pid=supervisor_pid,
            pgid=pgid,
            sid=sid,
            leader_start_ticks=leader_start_ticks,
        )
    )


def _terminate_controller_group(
    process: subprocess.Popen[bytes],
    *,
    supervisor_pid: int,
    pgid: int,
    sid: int,
    leader_start_ticks: int | None,
    grace_s: float,
) -> tuple[list[str], bool]:
    sent: list[str] = []
    if _signal_owned_processes(
        supervisor_pid=supervisor_pid,
        pgid=pgid,
        sid=sid,
        leader_start_ticks=leader_start_ticks,
        sig=signal.SIGTERM,
    ):
        sent.append("SIGTERM")
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        process.poll()
        if not _owned_processes_exist(
            supervisor_pid=supervisor_pid,
            pgid=pgid,
            sid=sid,
            leader_start_ticks=leader_start_ticks,
        ):
            break
        time.sleep(min(_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic())))
    process.poll()
    escalated = _owned_processes_exist(
        supervisor_pid=supervisor_pid,
        pgid=pgid,
        sid=sid,
        leader_start_ticks=leader_start_ticks,
    )
    if escalated and _signal_owned_processes(
        supervisor_pid=supervisor_pid,
        pgid=pgid,
        sid=sid,
        leader_start_ticks=leader_start_ticks,
        sig=signal.SIGKILL,
    ):
        sent.append("SIGKILL")
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=max(0.2, grace_s))
    reap_deadline = time.monotonic() + max(0.5, grace_s)
    while (
        _owned_processes_exist(
            supervisor_pid=supervisor_pid,
            pgid=pgid,
            sid=sid,
            leader_start_ticks=leader_start_ticks,
        )
        and time.monotonic() < reap_deadline
    ):
        time.sleep(_POLL_INTERVAL_S)
    reap_exited_children()
    return sent, escalated


def _signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return str(-returncode)


def supervise(
    controller_cmd: Sequence[str],
    *,
    receipt_path: Path,
    owner_pid: int,
    owner_start_ticks: int | None,
    timeout_s: float,
    term_grace_s: float,
    mode: str,
    unit: str | None,
    runtime_cap_s: float | None,
) -> int:
    """Run one pytest controller and clean every process in its owned group."""
    request_path = termination_request_path(receipt_path)
    for stale in (receipt_path, request_path):
        with contextlib.suppress(FileNotFoundError):
            stale.unlink()
    started_at = utc_now()
    started = time.monotonic()
    subreaper_enabled = enable_child_subreaper()
    owner_watch = _OwnerWatch(owner_pid, owner_start_ticks)
    received_signal: list[int] = []

    def _handle_signal(signum: int, _frame: object) -> None:
        if not received_signal:
            received_signal.append(signum)

    previous_handlers = {
        signum: signal.signal(signum, _handle_signal) for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    process = subprocess.Popen(list(controller_cmd), start_new_session=True)
    controller_pid = process.pid
    controller_pgid = controller_pid
    controller_sid = controller_pid
    controller_start_ticks = _process_start_ticks(controller_pid)
    try:
        controller_pgid = os.getpgid(controller_pid)
        controller_sid = os.getsid(controller_pid)
        base_receipt: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "started_at": started_at,
            "supervisor_pid": os.getpid(),
            "supervisor_start_ticks": _process_start_ticks(os.getpid()),
            "owner_pid": owner_pid,
            "owner_start_ticks": owner_start_ticks,
            "controller_pid": controller_pid,
            "controller_start_ticks": controller_start_ticks,
            "controller_pgid": controller_pgid,
            "controller_sid": controller_sid,
            "controller_command": list(controller_cmd),
            "mode": mode,
            "unit": unit,
            "cgroup_path": _cgroup_path(os.getpid()),
            "cgroup_owned": mode == "systemd-scope",
            "timeout_s": timeout_s,
            "term_grace_s": term_grace_s,
            "runtime_cap_s": runtime_cap_s,
            "subreaper_enabled": subreaper_enabled,
        }
        _write_json(receipt_path, base_receipt)
    except BaseException:
        # The controller already owns a new session here. If publishing the
        # ownership receipt fails, kill that still-pinned group before the
        # supervisor can exit and strand it without an addressable receipt.
        _signal_owned_processes(
            supervisor_pid=os.getpid(),
            pgid=controller_pgid,
            sid=controller_sid,
            leader_start_ticks=controller_start_ticks,
            sig=signal.SIGKILL,
        )
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)
        owner_watch.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        raise

    termination_reason: str | None = None
    requested_exit_code: int | None = None
    controller_returncode: int | None = None
    signals_sent: list[str] = []
    escalated = False
    try:
        while True:
            controller_returncode = process.poll()
            if controller_returncode is not None:
                if controller_returncode < 0:
                    termination_reason = f"pytest controller exited by signal {_signal_name(controller_returncode)}"
                break
            if owner_watch.gone():
                termination_reason = f"pytest runner owner pid {owner_pid} exited"
                requested_exit_code = 125
                break
            if received_signal:
                request = _read_termination_request(request_path)
                signal_name = signal.Signals(received_signal[0]).name
                if request is not None and isinstance(request.get("reason"), str):
                    termination_reason = str(request["reason"])
                    raw_exit = request.get("exit_code")
                    requested_exit_code = int(raw_exit) if isinstance(raw_exit, int) else 124
                else:
                    termination_reason = f"pytest supervisor received {signal_name}"
                    requested_exit_code = 128 + received_signal[0]
                break
            if timeout_s > 0 and time.monotonic() - started >= timeout_s:
                termination_reason = f"pytest runtime exceeded {timeout_s:g}s"
                requested_exit_code = 124
                break
            time.sleep(_POLL_INTERVAL_S)
    finally:
        signals_sent, escalated = _terminate_controller_group(
            process,
            supervisor_pid=os.getpid(),
            pgid=controller_pgid,
            sid=controller_sid,
            leader_start_ticks=controller_start_ticks,
            grace_s=term_grace_s,
        )
        if controller_returncode is None:
            controller_returncode = process.poll()
        owner_watch.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if requested_exit_code is not None:
        exit_code = requested_exit_code
    elif controller_returncode is None:
        exit_code = 125
    elif controller_returncode < 0:
        exit_code = 128 + (-controller_returncode)
    else:
        exit_code = controller_returncode
    final_receipt = {
        **base_receipt,
        "status": "terminated" if termination_reason is not None else "finished",
        "finished_at": utc_now(),
        "duration_s": round(time.monotonic() - started, 4),
        "exit_code": exit_code,
        "controller_returncode": controller_returncode,
        "termination_reason": termination_reason,
        "signals_sent": signals_sent,
        "escalated_to_sigkill": escalated,
        "controller_group_alive": _owned_processes_exist(
            supervisor_pid=os.getpid(),
            pgid=controller_pgid,
            sid=controller_sid,
            leader_start_ticks=controller_start_ticks,
        ),
    }
    _write_json(receipt_path, final_receipt)
    if termination_reason is not None:
        sys.stderr.write(
            f"pytest supervisor: {termination_reason}; owned pgid={controller_pgid} "
            f"mode={mode} signals={','.join(signals_sent) or 'none'}\n"
        )
        sys.stderr.flush()
    return exit_code


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        separator = list(argv).index("--")
    except ValueError as exc:
        raise SystemExit("pytest supervisor: missing '--' before controller command") from exc
    option_argv = list(argv[:separator])
    controller_cmd = list(argv[separator + 1 :])
    if not controller_cmd:
        raise SystemExit("pytest supervisor: controller command is empty")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-start-ticks", type=int)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--term-grace-s", type=float, required=True)
    parser.add_argument("--mode", choices=("process-group", "systemd-scope"), required=True)
    parser.add_argument("--unit")
    parser.add_argument("--runtime-cap-s", type=float)
    return parser.parse_args(option_argv), controller_cmd


def main(argv: Sequence[str] | None = None) -> int:
    args, controller_cmd = _parse_args(sys.argv[1:] if argv is None else argv)
    return supervise(
        controller_cmd,
        receipt_path=args.receipt,
        owner_pid=args.owner_pid,
        owner_start_ticks=args.owner_start_ticks,
        timeout_s=max(0.0, args.timeout_s),
        term_grace_s=max(0.0, args.term_grace_s),
        mode=args.mode,
        unit=args.unit,
        runtime_cap_s=args.runtime_cap_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
