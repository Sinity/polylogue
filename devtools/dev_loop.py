"""Branch-local daemon/web/browser-capture development preflight."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devtools import repo_root as _repo_root

DEFAULT_API_PORT = 8766
DEFAULT_BROWSER_CAPTURE_PORT = 8765


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
    preflight_json = run_log_dir / "preflight.json"
    if prepare:
        archive.mkdir(parents=True, exist_ok=True)
        browser_artifact_dir.mkdir(parents=True, exist_ok=True)
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
            "preflight_json": str(preflight_json),
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
            "run_daemon": (
                f"env POLYLOGUE_ARCHIVE_ROOT={archive} "
                f"POLYLOGUE_API_PORT={api_port} "
                f"POLYLOGUE_BROWSER_CAPTURE_PORT={browser_capture_port} "
                f"POLYLOGUE_DEV_LOOP_RUN_ID={run_id} "
                f"POLYLOGUE_DEV_LOOP_LOG_DIR={run_log_dir} "
                f"polylogued run 2>&1 | tee {daemon_log}"
            ),
            "open_web_shell": f"http://127.0.0.1:{api_port}/",
            "receiver_status": f"curl -sf http://127.0.0.1:{browser_capture_port}/v1/status",
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


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)
    payload = build_dev_loop_status(
        api_port=args.api_port,
        browser_capture_port=args.browser_capture_port,
        archive_root=args.archive_root,
        log_dir=args.log_dir,
        prepare=args.prepare,
    )
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
