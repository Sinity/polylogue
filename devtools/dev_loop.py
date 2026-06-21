"""Branch-local daemon/web/browser-capture development preflight."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any, TextIO

from devtools import repo_root as _repo_root
from polylogue.browser_capture.server import make_server

DEFAULT_API_PORT = 8766
DEFAULT_BROWSER_CAPTURE_PORT = 8765
_RECEIVER_SMOKE_ORIGIN = "chrome-extension://polylogue-dev-loop"
_RECEIVER_SMOKE_TOKEN = "polylogue-dev-loop-token"
_SENSITIVE_ENV_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASS|KEY|CREDENTIAL|AUTH)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


def _run_command(args: list[str], *, timeout_s: float = 2.0) -> CommandResult:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(exit_code=127, stdout="", stderr=str(exc))
    return CommandResult(exit_code=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def _safe_artifact_name(command: list[str]) -> str:
    raw = "-".join(Path(part).name if index == 0 else part for index, part in enumerate(command))
    name = _RUN_ID_SAFE_RE.sub("-", raw).strip("-").lower()
    return name[:80] or "command"


def _dev_loop_env_snapshot(env: dict[str, str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in sorted(env):
        if not (key.startswith("POLYLOGUE_") or key in {"PATH", "PYTHONPATH"}):
            continue
        snapshot[key] = "[redacted]" if _SENSITIVE_ENV_NAME_RE.search(key) else env[key]
    return snapshot


def _git_value(args: list[str], *, cwd: Path) -> str | None:
    result = _run_command(["git", "-C", str(cwd), *args], timeout_s=2.0)
    if result.exit_code != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _read_environ(pid: int) -> dict[str, str]:
    path = Path("/proc") / str(pid) / "environ"
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, _, value = item.partition(b"=")
        try:
            env[key.decode()] = value.decode(errors="replace")
        except UnicodeDecodeError:
            continue
    return env


def _socket_connectable(port: int, *, host: str = "127.0.0.1", timeout_s: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def system_service_status(unit: str = "polylogued.service") -> dict[str, Any]:
    result = _run_command(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState,SubState,MainPID,FragmentPath",
            "--no-pager",
        ]
    )
    if result.exit_code != 0:
        return {
            "unit": unit,
            "available": False,
            "active": False,
            "error": result.stderr or result.stdout or "systemctl query failed",
        }
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            fields[key] = value
    pid = int(fields.get("MainPID") or 0)
    env = _read_environ(pid) if pid else {}
    return {
        "unit": unit,
        "available": True,
        "active": fields.get("ActiveState") == "active",
        "active_state": fields.get("ActiveState") or "unknown",
        "sub_state": fields.get("SubState") or "unknown",
        "main_pid": pid,
        "archive_root": env.get("POLYLOGUE_ARCHIVE_ROOT"),
        "fragment_path": fields.get("FragmentPath") or None,
    }


_PID_RE = re.compile(r"pid=(?P<pid>\d+)")
_RUN_ID_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _parse_ss_port_owners(stdout: str, port: int) -> list[dict[str, Any]]:
    owners: list[dict[str, Any]] = []
    port_token = f":{port}"
    for line in stdout.splitlines():
        if port_token not in line:
            continue
        match = _PID_RE.search(line)
        pid = int(match.group("pid")) if match else None
        env = _read_environ(pid) if pid is not None else {}
        owners.append(
            {
                "pid": pid,
                "line": line.strip(),
                "archive_root": env.get("POLYLOGUE_ARCHIVE_ROOT"),
            }
        )
    return owners


def port_status(port: int) -> dict[str, Any]:
    result = _run_command(["ss", "-H", "-ltnp"])
    owners = _parse_ss_port_owners(result.stdout, port) if result.exit_code == 0 else []
    return {
        "port": port,
        "connectable": _socket_connectable(port),
        "owner_count": len(owners),
        "owners": owners,
        "probe_error": None if result.exit_code == 0 else (result.stderr or result.stdout or "ss query failed"),
    }


def _browser_capture_smoke_payload() -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/dev-loop-smoke",
            "page_title": "Polylogue dev-loop smoke",
            "captured_at": "2026-06-20T00:00:00+00:00",
            "adapter_name": "dev-loop-smoke",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "dev-loop-smoke",
            "title": "Polylogue dev-loop smoke",
            "turns": [{"provider_turn_id": "turn-1", "role": "user", "text": "smoke"}],
        },
    }


def _receiver_post(
    *,
    host: str,
    port: int,
    body: object,
    auth_token: str | None,
) -> tuple[int, dict[str, object]]:
    headers = {
        "Content-Type": "application/json",
        "Origin": _RECEIVER_SMOKE_ORIGIN,
    }
    if auth_token is not None:
        headers["Authorization"] = f"Bearer {auth_token}"
    conn = HTTPConnection(host, port, timeout=5)
    try:
        conn.request("POST", "/v1/browser-captures", body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        response_body = json.loads(response.read().decode("utf-8"))
        return response.status, dict(response_body)
    finally:
        conn.close()


def run_receiver_smoke(*, spool_path: Path) -> dict[str, object]:
    spool_path.mkdir(parents=True, exist_ok=True)
    server = make_server(
        "127.0.0.1",
        0,
        spool_path=spool_path,
        auth_token=_RECEIVER_SMOKE_TOKEN,
        extra_origins=(_RECEIVER_SMOKE_ORIGIN,),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        payload = _browser_capture_smoke_payload()
        rejected_status, rejected_body = _receiver_post(host=str(host), port=int(port), body=payload, auth_token=None)
        accepted_status, accepted_body = _receiver_post(
            host=str(host),
            port=int(port),
            body=payload,
            auth_token=_RECEIVER_SMOKE_TOKEN,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    artifact_ref = accepted_body.get("artifact_ref")
    artifact_path = spool_path / str(artifact_ref) if isinstance(artifact_ref, str) else None
    return {
        "ok": rejected_status == 401
        and accepted_status == 202
        and artifact_path is not None
        and artifact_path.exists(),
        "host": str(host),
        "port": int(port),
        "spool_path": str(spool_path),
        "unauthenticated_status": rejected_status,
        "unauthenticated_error": rejected_body.get("error"),
        "authenticated_status": accepted_status,
        "artifact_ref": artifact_ref,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "bytes_written": accepted_body.get("bytes_written"),
    }


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _start_daemon_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: TextIO,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _write_dev_loop_event(
    path: Path,
    *,
    preflight: dict[str, Any],
    surface: str,
    event_type: str,
    status: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_id": str(preflight["run_id"]),
        "surface": surface,
        "event_type": event_type,
        "status": status,
        "repo_root": str(preflight["repo_root"]),
        "branch": preflight.get("branch"),
        "commit": preflight.get("commit"),
        "archive_root": str(preflight["dev_archive_root"]),
        "payload": payload or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def run_cli_capture(
    *,
    preflight: dict[str, Any],
    command: list[str],
    timeout_s: float,
) -> dict[str, object]:
    """Run a branch-local CLI command and persist debug artifacts."""

    if not command:
        raise ValueError("capture command must not be empty")
    if command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("capture command must not be empty")

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    terminal_dir = Path(str(artifacts["terminal_dir"]))
    terminal_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    suggested_env = preflight.get("suggested_env")
    if isinstance(suggested_env, dict):
        env.update({str(key): str(value) for key, value in suggested_env.items()})
    env.setdefault("POLYLOGUE_FORCE_PLAIN", "1")

    artifact_name = _safe_artifact_name(command)
    stdout_path = terminal_dir / f"{artifact_name}.stdout"
    stderr_path = terminal_dir / f"{artifact_name}.stderr"
    transcript_path = terminal_dir / f"{artifact_name}.transcript.txt"
    env_path = terminal_dir / f"{artifact_name}.env.json"
    summary_path = terminal_dir / f"{artifact_name}.summary.json"

    started_at = datetime.now(UTC)
    started = time.monotonic()
    timed_out = False
    try:
        result = subprocess.run(
            command,
            cwd=str(preflight["repo_root"]),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        exit_code = int(result.returncode)
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr)
        stderr = (stderr + "\n" if stderr else "") + f"command timed out after {timeout_s:.1f}s\n"
    except OSError as exc:
        exit_code = 127
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}\n"
    duration_ms = int((time.monotonic() - started) * 1000)
    ended_at = datetime.now(UTC)

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    transcript = (
        f"$ {shlex.join(command)}\n"
        f"# cwd={preflight['repo_root']}\n"
        f"# run_id={preflight['run_id']}\n"
        f"# started_at={started_at.isoformat()}\n\n"
        "[stdout]\n"
        f"{stdout}"
        "\n[stderr]\n"
        f"{stderr}"
        f"\n[exit {exit_code}]\n"
    )
    transcript_path.write_text(transcript, encoding="utf-8")
    env_snapshot = _dev_loop_env_snapshot(env)
    env_path.write_text(json.dumps(env_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload: dict[str, object] = {
        "ok": exit_code == 0,
        "command": command,
        "command_text": shlex.join(command),
        "cwd": str(preflight["repo_root"]),
        "run_id": str(preflight["run_id"]),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "transcript": str(transcript_path),
            "env": str(env_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def launch_branch_daemon(
    *,
    preflight: dict[str, Any],
    readiness_timeout_s: float,
) -> dict[str, object]:
    """Launch a branch-local polylogued and persist process/debug artifacts."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    run_log_dir = Path(str(preflight["run_log_dir"]))
    run_log_dir.mkdir(parents=True, exist_ok=True)
    daemon_log = Path(str(artifacts["daemon_log"]))
    daemon_log.parent.mkdir(parents=True, exist_ok=True)
    event_path = Path(str(artifacts.get("dev_events", run_log_dir / "dev-loop.events.jsonl")))
    env_path = run_log_dir / "polylogued.env.json"
    pid_path = run_log_dir / "polylogued.pid"
    summary_path = run_log_dir / "polylogued.launch.json"

    ports = preflight.get("ports")
    if not isinstance(ports, dict):
        raise ValueError("preflight payload is missing port status")
    occupied_ports: list[str] = []
    for name in ("api", "browser_capture"):
        status = ports.get(name)
        if isinstance(status, dict) and int(status.get("owner_count") or 0) > 0:
            occupied_ports.append(f"{name} port {status.get('port')}")
    if occupied_ports:
        _write_dev_loop_event(
            event_path,
            preflight=preflight,
            surface="daemon",
            event_type="launch_rejected",
            status="blocked",
            payload={"occupied_ports": occupied_ports, "ports": ports},
        )
        raise ValueError(
            "selected branch-local ports already have listeners: "
            + ", ".join(occupied_ports)
            + "; stop the deployed service or choose isolated ports"
        )

    env = os.environ.copy()
    suggested_env = preflight.get("suggested_env")
    if isinstance(suggested_env, dict):
        env.update({str(key): str(value) for key, value in suggested_env.items()})
    env.setdefault("POLYLOGUE_FORCE_PLAIN", "1")

    api_status = ports["api"]
    receiver_status = ports["browser_capture"]
    assert isinstance(api_status, dict)
    assert isinstance(receiver_status, dict)
    api_port = int(api_status["port"])
    receiver_port = int(receiver_status["port"])
    spool_path = run_log_dir / "browser-capture-spool"
    spool_path.mkdir(parents=True, exist_ok=True)
    command = [
        "polylogued",
        "run",
        "--no-watch",
        "--api-port",
        str(api_port),
        "--port",
        str(receiver_port),
        "--spool",
        str(spool_path),
    ]

    env_snapshot = _dev_loop_env_snapshot(env)
    env_path.write_text(json.dumps(env_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started_at = datetime.now(UTC)
    started = time.monotonic()
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="daemon",
        event_type="launch_requested",
        status="starting",
        payload={
            "command": command,
            "cwd": str(preflight["repo_root"]),
            "api_port": api_port,
            "browser_capture_port": receiver_port,
            "spool_path": str(spool_path),
            "log_path": str(daemon_log),
        },
    )
    with daemon_log.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n# dev-loop launch {started_at.isoformat()} run_id={preflight['run_id']}\n$ {shlex.join(command)}\n"
        )
        log_file.flush()
        process = _start_daemon_process(
            command,
            cwd=Path(str(preflight["repo_root"])),
            env=env,
            log_file=log_file,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="daemon",
        event_type="process_spawned",
        status="running",
        payload={"pid": process.pid},
    )

    deadline = time.monotonic() + max(0.0, readiness_timeout_s)
    api_ready = False
    receiver_ready = False
    exit_code: int | None = None
    while time.monotonic() <= deadline:
        exit_code = process.poll()
        api_ready = _socket_connectable(api_port)
        receiver_ready = _socket_connectable(receiver_port)
        if exit_code is not None or (api_ready and receiver_ready):
            break
        time.sleep(0.1)
    if exit_code is None:
        exit_code = process.poll()
    duration_ms = int((time.monotonic() - started) * 1000)
    ended_at = datetime.now(UTC)
    launch_ok = exit_code is None and api_ready and receiver_ready
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="daemon",
        event_type="readiness_succeeded" if launch_ok else "readiness_failed",
        status="ok" if launch_ok else "failed",
        payload={
            "pid": process.pid,
            "exit_code": exit_code,
            "api_ready": api_ready,
            "browser_capture_ready": receiver_ready,
            "duration_ms": duration_ms,
            "readiness_timeout_s": readiness_timeout_s,
        },
    )

    payload: dict[str, object] = {
        "ok": launch_ok,
        "pid": process.pid,
        "command": command,
        "command_text": shlex.join(command),
        "cwd": str(preflight["repo_root"]),
        "run_id": str(preflight["run_id"]),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "api_ready": api_ready,
        "browser_capture_ready": receiver_ready,
        "readiness_timeout_s": readiness_timeout_s,
        "spool_path": str(spool_path),
        "artifacts": {
            "log": str(daemon_log),
            "env": str(env_path),
            "pid": str(pid_path),
            "summary": str(summary_path),
            "events": str(event_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _dev_loop_run_id(*, branch: str | None, commit: str | None, api_port: int, browser_capture_port: int) -> str:
    branch_part = _RUN_ID_SAFE_RE.sub("-", branch or "detached").strip("-") or "detached"
    commit_part = _RUN_ID_SAFE_RE.sub("-", commit or "unknown").strip("-") or "unknown"
    return f"{branch_part}-{commit_part}-api{api_port}-capture{browser_capture_port}"


def build_dev_loop_status(
    *,
    repo_root: Path | None = None,
    api_port: int = DEFAULT_API_PORT,
    browser_capture_port: int = DEFAULT_BROWSER_CAPTURE_PORT,
    archive_root: Path | None = None,
    log_dir: Path | None = None,
    prepare: bool = False,
) -> dict[str, Any]:
    root = repo_root or _repo_root()
    archive = archive_root or root / ".local" / "dev-archive"
    logs = log_dir or root / ".cache" / "dev-loop"

    service = system_service_status()
    api = port_status(api_port)
    receiver = port_status(browser_capture_port)
    branch = _git_value(["branch", "--show-current"], cwd=root)
    commit = _git_value(["rev-parse", "--short", "HEAD"], cwd=root)
    run_id = _dev_loop_run_id(
        branch=branch,
        commit=commit,
        api_port=api_port,
        browser_capture_port=browser_capture_port,
    )
    run_log_dir = logs / run_id
    daemon_log = run_log_dir / "polylogued.log"
    browser_artifact_dir = run_log_dir / "browser"
    terminal_artifact_dir = run_log_dir / "terminal"
    tui_artifact_dir = run_log_dir / "tui"
    preflight_json = run_log_dir / "preflight.json"
    dev_events = run_log_dir / "dev-loop.events.jsonl"
    if prepare:
        archive.mkdir(parents=True, exist_ok=True)
        browser_artifact_dir.mkdir(parents=True, exist_ok=True)
        terminal_artifact_dir.mkdir(parents=True, exist_ok=True)
        tui_artifact_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if service.get("active"):
        warnings.append(
            "systemwide polylogued.service is active; stop it or use isolated ports before branch-local runs"
        )
    for name, status in (("api", api), ("browser_capture", receiver)):
        if int(status.get("owner_count") or 0) > 0:
            warnings.append(f"{name} port {status['port']} already has a listener")

    payload = {
        "repo_root": str(root),
        "branch": branch,
        "commit": commit,
        "run_id": run_id,
        "prepared": prepare,
        "preflight_json_written": False,
        "dev_archive_root": str(archive),
        "log_dir": str(logs),
        "run_log_dir": str(run_log_dir),
        "artifacts": {
            "daemon_log": str(daemon_log),
            "browser_dir": str(browser_artifact_dir),
            "terminal_dir": str(terminal_artifact_dir),
            "tui_dir": str(tui_artifact_dir),
            "preflight_json": str(preflight_json),
            "dev_events": str(dev_events),
        },
        "system_service": service,
        "ports": {
            "api": api,
            "browser_capture": receiver,
        },
        "suggested_env": {
            "POLYLOGUE_ARCHIVE_ROOT": str(archive),
            "POLYLOGUE_API_PORT": str(api_port),
            "POLYLOGUE_BROWSER_CAPTURE_PORT": str(browser_capture_port),
            "POLYLOGUE_DEV_LOOP_RUN_ID": run_id,
            "POLYLOGUE_DEV_LOOP_LOG_DIR": str(run_log_dir),
        },
        "commands": {
            "stop_system_service": "systemctl --user stop polylogued.service",
            "prepare": "devtools workspace dev-loop --prepare",
            "save_preflight": f"mkdir -p {run_log_dir} && devtools workspace dev-loop --json > {preflight_json}",
            "receiver_smoke": "devtools workspace dev-loop --receiver-smoke --json",
            "run_daemon": (
                f"env POLYLOGUE_ARCHIVE_ROOT={archive} "
                f"POLYLOGUE_API_PORT={api_port} "
                f"POLYLOGUE_BROWSER_CAPTURE_PORT={browser_capture_port} "
                f"POLYLOGUE_DEV_LOOP_RUN_ID={run_id} "
                f"POLYLOGUE_DEV_LOOP_LOG_DIR={run_log_dir} "
                f"polylogued run --api-port {api_port} --port {browser_capture_port} 2>&1 | tee {daemon_log}"
            ),
            "open_web_shell": f"http://127.0.0.1:{api_port}/",
            "receiver_status": f"curl -sf http://127.0.0.1:{browser_capture_port}/v1/status",
            "capture_cli_status": (
                "script -q -c "
                f"'env POLYLOGUE_ARCHIVE_ROOT={archive} polylogue ops status' "
                f"{terminal_artifact_dir / 'polylogue-ops-status.typescript'}"
            ),
            "capture_tui_placeholder": (
                "Record branch-local TUI/terminal runs into "
                f"{tui_artifact_dir}; use the local terminal-control surface or VHS when visual playback is needed"
            ),
        },
        "warnings": warnings,
    }
    if prepare:
        payload["preflight_json_written"] = True
        preflight_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _print_human(payload: dict[str, Any]) -> None:
    print("Polylogue dev loop")
    print(f"  repo:    {payload['repo_root']}")
    print(f"  branch:  {payload.get('branch') or 'unknown'} @ {payload.get('commit') or 'unknown'}")
    print(f"  run id:  {payload['run_id']}")
    print(f"  archive: {payload['dev_archive_root']}")
    print(f"  logs:    {payload['log_dir']}")
    print(f"  run log: {payload['run_log_dir']}")
    service = payload["system_service"]
    assert isinstance(service, dict)
    print(
        f"  service: {service.get('unit')} {service.get('active_state', 'unavailable')} pid={service.get('main_pid') or '-'}"
    )
    ports = payload["ports"]
    assert isinstance(ports, dict)
    for name in ("api", "browser_capture"):
        status = ports[name]
        assert isinstance(status, dict)
        print(f"  port {name}: {status['port']} listeners={status['owner_count']} connectable={status['connectable']}")
    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    commands = payload["commands"]
    assert isinstance(commands, dict)
    print("\nCommands:")
    for name, command in commands.items():
        print(f"  {name}: {command}")


def _print_receiver_smoke(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    smoke = payload["receiver_smoke"]
    assert isinstance(preflight, dict)
    assert isinstance(smoke, dict)
    print("Polylogue dev-loop receiver smoke")
    print(f"  run id:  {preflight['run_id']}")
    print(f"  ok:      {smoke['ok']}")
    print(f"  reject:  {smoke['unauthenticated_status']} {smoke.get('unauthenticated_error')}")
    print(f"  accept:  {smoke['authenticated_status']}")
    print(f"  artifact: {smoke.get('artifact_ref')}")


def _print_cli_capture(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    capture = payload["cli_capture"]
    assert isinstance(preflight, dict)
    assert isinstance(capture, dict)
    artifacts = capture["artifacts"]
    assert isinstance(artifacts, dict)
    print("Polylogue dev-loop CLI capture")
    print(f"  run id:    {preflight['run_id']}")
    print(f"  command:   {capture['command_text']}")
    print(f"  exit:      {capture['exit_code']}")
    print(f"  duration:  {capture['duration_ms']} ms")
    print(f"  transcript: {artifacts['transcript']}")


def _print_daemon_launch(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    launch = payload["daemon_launch"]
    assert isinstance(preflight, dict)
    assert isinstance(launch, dict)
    artifacts = launch["artifacts"]
    assert isinstance(artifacts, dict)
    print("Polylogue dev-loop daemon launch")
    print(f"  run id:   {preflight['run_id']}")
    print(f"  pid:      {launch['pid']}")
    print(f"  command:  {launch['command_text']}")
    print(f"  api:      ready={launch['api_ready']}")
    print(f"  capture:  ready={launch['browser_capture_ready']}")
    print(f"  log:      {artifacts['log']}")
    print(f"  summary:  {artifacts['summary']}")


def main(argv: list[str] | None = None) -> int:
    original_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--api-port", type=int, default=int(os.environ.get("POLYLOGUE_API_PORT", DEFAULT_API_PORT)))
    parser.add_argument(
        "--browser-capture-port",
        type=int,
        default=int(os.environ.get("POLYLOGUE_BROWSER_CAPTURE_PORT", DEFAULT_BROWSER_CAPTURE_PORT)),
    )
    parser.add_argument("--archive-root", type=Path, help="Branch-local archive root to report/use.")
    parser.add_argument("--log-dir", type=Path, help="Branch-local dev-loop log directory to report/use.")
    parser.add_argument("--prepare", action="store_true", help="Create the reported archive/log directories.")
    parser.add_argument(
        "--receiver-smoke",
        action="store_true",
        help="Run a deterministic in-process browser-capture receiver smoke.",
    )
    parser.add_argument(
        "--capture-timeout-s",
        type=float,
        default=30.0,
        help="Timeout for --capture-cli commands.",
    )
    parser.add_argument(
        "--capture-cli",
        action="store_true",
        help="Run a branch-local CLI command and write stdout/stderr/transcript artifacts.",
    )
    parser.add_argument(
        "--launch-daemon",
        action="store_true",
        help="Launch branch-local polylogued with --no-watch and write PID/log/summary artifacts.",
    )
    parser.add_argument(
        "--daemon-ready-timeout-s",
        type=float,
        default=10.0,
        help="Readiness wait for --launch-daemon.",
    )
    parser.add_argument("capture_command", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    capture_command = list(args.capture_command)
    if args.capture_cli and not args.json and "--" not in original_argv and capture_command[-1:] == ["--json"]:
        args.json = True
        capture_command = capture_command[:-1]
    payload = build_dev_loop_status(
        api_port=args.api_port,
        browser_capture_port=args.browser_capture_port,
        archive_root=args.archive_root,
        log_dir=args.log_dir,
        prepare=args.prepare or args.receiver_smoke or args.launch_daemon,
    )
    if args.receiver_smoke:
        smoke_payload: dict[str, Any] = {
            "preflight": payload,
            "receiver_smoke": run_receiver_smoke(spool_path=Path(str(payload["run_log_dir"])) / "receiver-smoke-spool"),
        }
        if args.json:
            json.dump(smoke_payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_receiver_smoke(smoke_payload)
        return 0 if smoke_payload["receiver_smoke"]["ok"] else 1
    if args.capture_cli:
        try:
            capture_payload = run_cli_capture(
                preflight=payload,
                command=capture_command,
                timeout_s=args.capture_timeout_s,
            )
        except ValueError as exc:
            parser.error(str(exc))
        combined_payload: dict[str, Any] = {
            "preflight": payload,
            "cli_capture": capture_payload,
        }
        if args.json:
            json.dump(combined_payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_cli_capture(combined_payload)
        capture_exit = capture_payload["exit_code"]
        if not isinstance(capture_exit, int):
            capture_exit = 1
        return 0 if capture_payload.get("ok") is True else capture_exit
    if args.launch_daemon:
        try:
            launch_payload = launch_branch_daemon(
                preflight=payload,
                readiness_timeout_s=args.daemon_ready_timeout_s,
            )
        except ValueError as exc:
            parser.error(str(exc))
        combined_payload = {
            "preflight": payload,
            "daemon_launch": launch_payload,
        }
        if args.json:
            json.dump(combined_payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_daemon_launch(combined_payload)
        return 0 if launch_payload.get("ok") is True else 1
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
