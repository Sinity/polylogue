"""Branch-local daemon/web/browser-capture development preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any, TextIO, cast
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from devtools import repo_root as _repo_root
from polylogue.browser_capture.server import make_server
from polylogue.storage.sqlite.archive_tiers.archive_init import (
    ArchiveInitBlockedError,
    initialize_archive_tier_files,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

DEFAULT_API_PORT = 8766
DEFAULT_BROWSER_CAPTURE_PORT = 8765
_RECEIVER_SMOKE_ORIGIN = "chrome-extension://polylogue-dev-loop"
_RECEIVER_SMOKE_TOKEN = "polylogue-dev-loop-token"
_SENSITIVE_ENV_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASS|KEY|CREDENTIAL|AUTH)", re.IGNORECASE)
_LOG = logging.getLogger(__name__)

_DEV_LOOP_AUTHORITIES: dict[str, dict[str, object]] = {
    "preflight": {
        "label": "source-only-preflight",
        "description": "Inspect checkout, branch-local paths, deployed service state, and selected ports without starting browsers or using live profiles.",
        "source_safe": True,
        "cloud_safe": True,
        "requires_browser": False,
        "requires_copied_profile": False,
        "local_only": False,
    },
    "launch_daemon": {
        "label": "source-only-branch-daemon",
        "description": "Start polylogued from this checkout with explicit branch-local archive, ports, run id, and log directory.",
        "source_safe": True,
        "cloud_safe": True,
        "requires_browser": False,
        "requires_copied_profile": False,
        "local_only": False,
    },
    "capture_cli": {
        "label": "source-only-cli-capture",
        "description": "Run a CLI command with branch-local POLYLOGUE_* environment and persisted stdout/stderr/transcript artifacts.",
        "source_safe": True,
        "cloud_safe": True,
        "requires_browser": False,
        "requires_copied_profile": False,
        "local_only": False,
    },
    "receiver_smoke": {
        "label": "source-only-deterministic-receiver-smoke",
        "description": "Run an in-process loopback receiver auth/spool smoke with synthetic capture payloads only.",
        "source_safe": True,
        "cloud_safe": True,
        "requires_browser": False,
        "requires_copied_profile": False,
        "local_only": False,
    },
    "extension_smoke": {
        "label": "source-only-deterministic-extension-smoke",
        "description": "Exercise the extension background worker against a temporary receiver with a Chrome API mock and synthetic payloads.",
        "source_safe": True,
        "cloud_safe": True,
        "requires_browser": False,
        "requires_copied_profile": False,
        "local_only": False,
    },
    "browser_smoke": {
        "label": "local-browser-deterministic-extension-smoke",
        "description": "Load the unpacked extension in headless Chrome/Chromium against a temporary receiver without live cookies.",
        "source_safe": True,
        "cloud_safe": False,
        "requires_browser": True,
        "requires_copied_profile": False,
        "local_only": True,
    },
    "browser_provider_smoke": {
        "label": "local-browser-deterministic-provider-smoke",
        "description": "Load deterministic ChatGPT/Claude fixture pages in headless Chrome/Chromium and verify content-script capture without live cookies.",
        "source_safe": True,
        "cloud_safe": False,
        "requires_browser": True,
        "requires_copied_profile": False,
        "local_only": True,
    },
    "browser_provider_live_follow": {
        "label": "local-browser-deterministic-live-follow-proof",
        "description": "Launch branch-local daemon, capture deterministic provider pages through the extension, and prove receiver output reaches archive/API reads.",
        "source_safe": True,
        "cloud_safe": False,
        "requires_browser": True,
        "requires_copied_profile": False,
        "local_only": True,
    },
    "browser_plan": {
        "label": "local-browser-control-handoff-plan",
        "description": "Write exact local browser launch/configuration artifacts without claiming to control or certify the operator browser.",
        "source_safe": True,
        "cloud_safe": False,
        "requires_browser": True,
        "requires_copied_profile": False,
        "local_only": True,
    },
    "browser_live_proof": {
        "label": "operator-local-copied-profile-live-proof",
        "description": "Visible local Chrome/Chromium proof using an operator-approved copied profile; never CI/cloud evidence and never a source-only claim.",
        "source_safe": False,
        "cloud_safe": False,
        "requires_browser": True,
        "requires_copied_profile": True,
        "local_only": True,
    },
    "tui_plan": {
        "label": "source-owned-local-terminal-plan",
        "description": "Write branch-local terminal/TUI recording commands and artifact paths; terminal control remains local.",
        "source_safe": True,
        "cloud_safe": True,
        "requires_browser": False,
        "requires_copied_profile": False,
        "local_only": False,
    },
    "inspect_run": {
        "label": "source-only-run-artifact-inspection",
        "description": "Summarize persisted run-local artifacts and report missing or failed surfaces without rerunning browser/profile proofs.",
        "source_safe": True,
        "cloud_safe": True,
        "requires_browser": False,
        "requires_copied_profile": False,
        "local_only": False,
    },
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


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


def _terminate_process_tree(process: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        try:
            if sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()
        except OSError:
            return


def _terminate_pid_tree(pid: int, *, grace_s: float = 3.0) -> dict[str, object]:
    def _reaped_or_exited() -> bool:
        try:
            waited_pid, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        else:
            if waited_pid == pid:
                return True
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return True
        parts = stat.split()
        return len(parts) > 2 and parts[2] == "Z"

    signals_sent: list[str] = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if _reaped_or_exited():
            return {"ok": True, "pid": pid, "signals_sent": signals_sent, "state": "exited"}
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return {"ok": True, "pid": pid, "signals_sent": signals_sent, "state": "exited"}
        except (PermissionError, OSError) as exc:
            return {
                "ok": False,
                "pid": pid,
                "signals_sent": signals_sent,
                "state": "signal_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        signals_sent.append(signal.Signals(sig).name)
        deadline = time.monotonic() + grace_s
        while time.monotonic() <= deadline:
            if _reaped_or_exited():
                return {"ok": True, "pid": pid, "signals_sent": signals_sent, "state": "exited"}
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return {"ok": True, "pid": pid, "signals_sent": signals_sent, "state": "exited"}
            except PermissionError:
                break
            time.sleep(0.05)
    return {"ok": False, "pid": pid, "signals_sent": signals_sent, "state": "still_running"}


def _run_process_tree(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
) -> ProcessResult:
    """Run a subprocess tree and kill the whole tree on timeout.

    Browser smokes spawn Chrome through Node. Stdout/stderr are captured via
    files rather than pipes so Chrome grandchildren cannot keep a pipe open and
    wedge the parent after the Node process has exited.
    """

    with TemporaryDirectory(prefix="polylogue-process-") as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.txt"
        stderr_path = Path(temp_dir) / "stderr.txt"
        with (
            stdout_path.open("w+", encoding="utf-8") as stdout_file,
            stderr_path.open(
                "w+",
                encoding="utf-8",
            ) as stderr_file,
        ):
            try:
                process = subprocess.Popen(
                    args,
                    cwd=str(cwd),
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                return ProcessResult(exit_code=127, stdout="", stderr=f"{type(exc).__name__}: {exc}\n", timed_out=False)
            timed_out = False
            try:
                exit_code = int(process.wait(timeout=timeout_s))
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_tree(process, signal.SIGTERM)
                try:
                    exit_code = int(process.wait(timeout=5))
                except subprocess.TimeoutExpired:
                    _terminate_process_tree(process, signal.SIGKILL)
                    try:
                        exit_code = int(process.wait(timeout=5))
                    except subprocess.TimeoutExpired:
                        exit_code = 124
            stdout_file.flush()
            stderr_file.flush()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
            if timed_out and exit_code != 124:
                exit_code = 124
            return ProcessResult(exit_code=exit_code, stdout=stdout, stderr=stderr, timed_out=timed_out)


def _safe_artifact_name(command: list[str]) -> str:
    raw = "-".join(Path(part).name if index == 0 else part for index, part in enumerate(command))
    name = _RUN_ID_SAFE_RE.sub("-", raw).strip("-").lower()
    return name[:80] or "command"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redact_live_provider_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "unparseable-url"
    if not parsed.scheme or not parsed.netloc:
        return "unparseable-url"
    path_parts = [part for part in parsed.path.split("/") if part]
    redacted_parts: list[str] = []
    for index, part in enumerate(path_parts):
        if index == 0 and part in {"c", "chat"}:
            redacted_parts.append(part)
        elif re.search(r"[A-Za-z0-9_-]{10,}", part):
            redacted_parts.append(f"<sha256:{_hash_text(part)[:12]}>")
        else:
            redacted_parts.append(part)
    redacted_path = "/" + "/".join(redacted_parts) if redacted_parts else "/"
    return urlunsplit((parsed.scheme, parsed.netloc, redacted_path, "", ""))


def _dev_loop_env_snapshot(env: dict[str, str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in sorted(env):
        if not (
            key.startswith("POLYLOGUE_")
            or key in {"PATH", "PYTHONPATH", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"}
        ):
            continue
        if _SENSITIVE_ENV_NAME_RE.search(key):
            snapshot[key] = "[redacted]"
        elif key in {"POLYLOGUE_LIVE_PROOF_CHATGPT_URL", "POLYLOGUE_LIVE_PROOF_CLAUDE_URL"}:
            snapshot[key] = _redact_live_provider_url(env[key])
            snapshot[f"{key}_SHA256"] = _hash_text(env[key])
        else:
            snapshot[key] = env[key]
    return snapshot


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    existing = env.get("PYTHONPATH")
    prefix = str(path)
    if existing:
        parts = existing.split(os.pathsep)
        if parts and parts[0] == prefix:
            return
        env["PYTHONPATH"] = os.pathsep.join([prefix, *[part for part in parts if part != prefix]])
    else:
        env["PYTHONPATH"] = prefix


def _dev_loop_suggested_env(
    *,
    archive: Path,
    run_log_dir: Path,
    api_port: int,
    browser_capture_port: int,
    run_id: str,
) -> dict[str, str]:
    return {
        "POLYLOGUE_ARCHIVE_ROOT": str(archive),
        "POLYLOGUE_API_PORT": str(api_port),
        "POLYLOGUE_BROWSER_CAPTURE_PORT": str(browser_capture_port),
        "POLYLOGUE_DAEMON_URL": f"http://127.0.0.1:{api_port}",
        "POLYLOGUE_DEV_LOOP_RUN_ID": run_id,
        "POLYLOGUE_DEV_LOOP_LOG_DIR": str(run_log_dir),
        "XDG_CACHE_HOME": str(run_log_dir / "xdg-cache"),
        "XDG_DATA_HOME": str(run_log_dir / "xdg-data"),
        "XDG_STATE_HOME": str(run_log_dir / "xdg-state"),
    }


def _preflight_suggested_env(preflight: dict[str, Any]) -> dict[str, str]:
    suggested_env = preflight.get("suggested_env")
    if isinstance(suggested_env, dict):
        return {str(key): str(value) for key, value in suggested_env.items()}
    ports = preflight.get("ports")
    if not isinstance(ports, dict):
        raise ValueError("preflight payload is missing port status")
    api_status = ports.get("api")
    receiver_status = ports.get("browser_capture")
    if not isinstance(api_status, dict) or not isinstance(receiver_status, dict):
        raise ValueError("preflight payload is missing API/browser-capture port status")
    return _dev_loop_suggested_env(
        archive=Path(str(preflight["dev_archive_root"])),
        run_log_dir=Path(str(preflight["run_log_dir"])),
        api_port=int(api_status["port"]),
        browser_capture_port=int(receiver_status["port"]),
        run_id=str(preflight["run_id"]),
    )


def _shell_env_prefix(env: dict[str, str]) -> str:
    return "env " + " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


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


def _allocate_loopback_port(*, reserved: set[int] | None = None) -> int:
    reserved_ports = reserved or set()
    for _attempt in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in reserved_ports and not _socket_connectable(port):
            return port
    raise RuntimeError("could not allocate an unused loopback port for the dev-loop")


def allocate_isolated_ports() -> tuple[int, int]:
    api_port = _allocate_loopback_port()
    browser_capture_port = _allocate_loopback_port(reserved={api_port})
    return api_port, browser_capture_port


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


def _http_get_json_url(url: str, *, timeout_s: float = 5.0) -> tuple[int, dict[str, object]]:
    parts = urlsplit(url)
    if parts.scheme != "http":
        raise ValueError(f"unsupported dev-loop URL scheme: {parts.scheme or '<missing>'}")
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 80
    path = urlunsplit(("", "", parts.path or "/", parts.query, ""))
    conn = HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"error": raw}
        return response.status, dict(body) if isinstance(body, dict) else {"body": body}
    finally:
        conn.close()


def _poll_archive_state(
    *,
    receiver_url: str,
    provider: str,
    provider_session_id: str,
    timeout_s: float,
    interval_s: float,
) -> dict[str, object]:
    query = urlencode({"provider": provider, "provider_session_id": provider_session_id})
    url = f"{receiver_url.rstrip('/')}/v1/archive-state?{query}"
    deadline = time.monotonic() + max(0.0, timeout_s)
    attempts = 0
    last_status = 0
    last_payload: dict[str, object] = {}
    while time.monotonic() <= deadline:
        attempts += 1
        try:
            last_status, last_payload = _http_get_json_url(url, timeout_s=5.0)
        except OSError as exc:
            last_status = 0
            last_payload = {"error": f"{type(exc).__name__}: {exc}"}
        if (
            last_status == 200
            and last_payload.get("raw_row_exists") is True
            and last_payload.get("indexed_session_exists") is True
        ):
            break
        time.sleep(max(0.05, interval_s))
    return {
        "ok": last_status == 200
        and last_payload.get("raw_row_exists") is True
        and last_payload.get("indexed_session_exists") is True,
        "status": last_status,
        "attempts": attempts,
        "url": url,
        "provider": provider,
        "provider_session_id": provider_session_id,
        "state": last_payload,
    }


def _session_id_for_provider(provider: str, provider_session_id: str) -> str:
    origin = {
        "chatgpt": "chatgpt-export",
        "claude-ai": "claude-ai-export",
    }.get(provider, provider)
    return f"{origin}:{provider_session_id}"


def _fetch_api_messages(
    *,
    api_url: str,
    session_id: str,
    limit: int = 5,
) -> dict[str, object]:
    path_session_id = quote(session_id, safe="")
    url = f"{api_url.rstrip('/')}/api/sessions/{path_session_id}/messages?{urlencode({'limit': limit})}"
    try:
        status, payload = _http_get_json_url(url, timeout_s=5.0)
    except OSError as exc:
        return {"ok": False, "status": 0, "url": url, "error": f"{type(exc).__name__}: {exc}"}
    messages = payload.get("messages")
    count = len(messages) if isinstance(messages, list) else 0
    return {
        "ok": status == 200 and count > 0,
        "status": status,
        "url": url,
        "session_id": session_id,
        "message_count": count,
        "payload_keys": sorted(payload.keys()),
    }


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
        "authority": _authority("receiver_smoke"),
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


def run_extension_smoke(*, preflight: dict[str, Any]) -> dict[str, object]:
    """Run the browser-extension background worker against a real receiver."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    browser_dir = Path(str(artifacts["browser_dir"])).resolve()
    browser_dir.mkdir(parents=True, exist_ok=True)
    event_path = Path(
        str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl"))
    ).resolve()
    extension_root = (Path(str(preflight["repo_root"])) / "browser-extension").resolve()
    spool_path = browser_dir / "extension-smoke-spool"
    summary_path = browser_dir / "extension-smoke.json"
    result_path = browser_dir / "extension-smoke-result.json"
    stdout_path = browser_dir / "extension-smoke.stdout"
    stderr_path = browser_dir / "extension-smoke.stderr"
    env_path = browser_dir / "extension-smoke.env.json"

    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_extension",
        event_type="extension_smoke_requested",
        status="starting",
        payload={
            "extension_root": str(extension_root),
            "spool_path": str(spool_path),
        },
    )

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
    if isinstance(host, bytes):
        host = host.decode("ascii")
    receiver_url = f"http://{host}:{port}"
    env = os.environ.copy()
    env.update(
        {
            "POLYLOGUE_EXTENSION_RECEIVER_URL": receiver_url,
            "POLYLOGUE_EXTENSION_RECEIVER_TOKEN": _RECEIVER_SMOKE_TOKEN,
            "POLYLOGUE_EXTENSION_SMOKE_OUT": str(result_path),
        }
    )
    env_path.write_text(json.dumps(_dev_loop_env_snapshot(env), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["npm", "run", "dev-loop-smoke", "--silent"],
            cwd=str(extension_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr)
        result_payload: dict[str, object] | None = None
        exit_code = 124
        stderr = (stderr + "\n" if stderr else "") + "extension smoke timed out after 20s\n"
    except OSError as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}\n"
        result_payload = None
        exit_code = 127
    else:
        stdout = result.stdout
        stderr = result.stderr
        exit_code = int(result.returncode)
        result_payload = None
        if result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    artifact_ref = None
    artifact_path: Path | None = None
    if isinstance(result_payload, dict):
        capture = result_payload.get("capture")
        if isinstance(capture, dict):
            artifact_ref = capture.get("artifact_ref")
            if isinstance(artifact_ref, str):
                artifact_path = spool_path / artifact_ref
    ok = exit_code == 0 and artifact_path is not None and artifact_path.exists()
    payload: dict[str, object] = {
        "ok": ok,
        "authority": _authority("extension_smoke"),
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "extension_root": str(extension_root),
        "receiver_url": receiver_url,
        "spool_path": str(spool_path),
        "artifact_ref": artifact_ref,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "result": result_payload,
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
            "result": str(result_path),
            "env": str(env_path),
            "spool": str(spool_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_extension",
        event_type="extension_smoke_finished",
        status="ok" if ok else "failed",
        payload={
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "artifact_ref": artifact_ref,
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


def run_browser_smoke(*, preflight: dict[str, Any]) -> dict[str, object]:
    """Run real Chrome with the unpacked extension against a local receiver."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    browser_dir = Path(str(artifacts["browser_dir"])).resolve()
    browser_dir.mkdir(parents=True, exist_ok=True)
    event_path = Path(
        str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl"))
    ).resolve()
    extension_root = (Path(str(preflight["repo_root"])) / "browser-extension").resolve()
    spool_path = browser_dir / "browser-smoke-spool"
    profile_dir = browser_dir / "browser-smoke-profile"
    summary_path = browser_dir / "browser-smoke.json"
    result_path = browser_dir / "browser-smoke-result.json"
    stdout_path = browser_dir / "browser-smoke.stdout"
    stderr_path = browser_dir / "browser-smoke.stderr"
    env_path = browser_dir / "browser-smoke.env.json"

    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser",
        event_type="browser_smoke_requested",
        status="starting",
        payload={
            "extension_root": str(extension_root),
            "profile_dir": str(profile_dir),
            "spool_path": str(spool_path),
        },
    )

    spool_path.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    server = make_server(
        "127.0.0.1",
        0,
        spool_path=spool_path,
        auth_token=_RECEIVER_SMOKE_TOKEN,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    if isinstance(host, bytes):
        host = host.decode("ascii")
    receiver_url = f"http://{host}:{port}"
    env = os.environ.copy()
    env.update(
        {
            "POLYLOGUE_BROWSER_SMOKE_EXTENSION_ROOT": str(extension_root),
            "POLYLOGUE_BROWSER_SMOKE_KEEP_PROFILE": "1",
            "POLYLOGUE_BROWSER_SMOKE_OUT": str(result_path),
            "POLYLOGUE_BROWSER_SMOKE_PROFILE_DIR": str(profile_dir),
            "POLYLOGUE_BROWSER_SMOKE_RECEIVER_TOKEN": _RECEIVER_SMOKE_TOKEN,
            "POLYLOGUE_BROWSER_SMOKE_RECEIVER_URL": receiver_url,
        }
    )
    env_path.write_text(json.dumps(_dev_loop_env_snapshot(env), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        result = _run_process_tree(
            ["node", "scripts/dev-loop-browser-smoke.mjs"],
            cwd=extension_root,
            env=env,
            timeout_s=35,
        )
        stdout = result.stdout
        stderr = result.stderr
        result_payload: dict[str, object] | None = None
        exit_code = int(result.exit_code)
        if result.timed_out:
            stderr = (stderr + "\n" if stderr else "") + "browser smoke timed out after 35s\n"
        elif result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    artifact_ref = None
    artifact_path: Path | None = None
    extension_id = None
    manifest = None
    if isinstance(result_payload, dict):
        extension_id = result_payload.get("extension_id")
        raw_manifest = result_payload.get("manifest")
        manifest = raw_manifest if isinstance(raw_manifest, dict) else None
        capture = result_payload.get("capture")
        if isinstance(capture, dict):
            body = capture.get("body")
            if isinstance(body, dict):
                artifact_ref = body.get("artifact_ref")
                if isinstance(artifact_ref, str):
                    artifact_path = spool_path / artifact_ref
    ok = exit_code == 0 and artifact_path is not None and artifact_path.exists()
    payload: dict[str, object] = {
        "ok": ok,
        "authority": _authority("browser_smoke"),
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "extension_id": extension_id,
        "extension_root": str(extension_root),
        "manifest": manifest,
        "profile_dir": str(profile_dir),
        "receiver_url": receiver_url,
        "spool_path": str(spool_path),
        "artifact_ref": artifact_ref,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "result": result_payload,
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
            "result": str(result_path),
            "env": str(env_path),
            "profile": str(profile_dir),
            "spool": str(spool_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser",
        event_type="browser_smoke_finished",
        status="ok" if ok else "failed",
        payload={
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "extension_id": extension_id,
            "artifact_ref": artifact_ref,
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


def _provider_smoke_artifacts(
    result_payload: dict[str, object] | None,
    *,
    spool_path: Path,
) -> tuple[dict[str, bool], dict[str, str], dict[str, str]]:
    provider_statuses: dict[str, bool] = {}
    artifact_refs: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    if not isinstance(result_payload, dict):
        return provider_statuses, artifact_refs, artifact_paths
    providers = result_payload.get("providers")
    if not isinstance(providers, dict):
        return provider_statuses, artifact_refs, artifact_paths
    for provider_name, provider_payload in providers.items():
        name = str(provider_name)
        if not isinstance(provider_payload, dict):
            provider_statuses[name] = False
            continue
        provider_statuses[name] = provider_payload.get("ok") is True
        capture_result = provider_payload.get("capture_result")
        if not isinstance(capture_result, dict):
            continue
        artifact_ref = capture_result.get("artifact_ref")
        if not isinstance(artifact_ref, str):
            continue
        artifact_refs[name] = artifact_ref
        artifact_paths[name] = str(spool_path / artifact_ref)
    return provider_statuses, artifact_refs, artifact_paths


_LIVE_PROOF_PROVIDERS = ("chatgpt", "claude")


def _parse_provider_csv(raw: str) -> list[str]:
    providers = [part.strip().lower() for part in raw.split(",") if part.strip()]
    deduped = list(dict.fromkeys(providers))
    invalid = [provider for provider in deduped if provider not in _LIVE_PROOF_PROVIDERS]
    if invalid:
        raise ValueError(f"unsupported browser live provider(s): {', '.join(invalid)}")
    if not deduped:
        raise ValueError("at least one browser live provider is required")
    return deduped


def _default_live_profile_dir(preflight: dict[str, Any]) -> Path:
    return Path(str(preflight["repo_root"])) / ".local" / "browser-profiles" / f"{preflight['run_id']}-chrome-user-data"


def _default_live_provider_urls() -> dict[str, str]:
    return {
        "chatgpt": "https://chatgpt.com/c/<conversation-id>",
        "claude": "https://claude.ai/chat/<conversation-id>",
    }


def run_browser_provider_smoke(
    *,
    preflight: dict[str, Any],
    receiver_url_override: str | None = None,
    spool_path_override: Path | None = None,
    session_id: str | None = None,
    reader_base_url: str | None = None,
    reader_session_id: str | None = None,
) -> dict[str, object]:
    """Run real Chrome content scripts against deterministic provider fixture pages."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    browser_dir = Path(str(artifacts["browser_dir"])).resolve()
    browser_dir.mkdir(parents=True, exist_ok=True)
    event_path = Path(
        str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl"))
    ).resolve()
    extension_root = (Path(str(preflight["repo_root"])) / "browser-extension").resolve()
    spool_path = (
        spool_path_override if spool_path_override is not None else browser_dir / "browser-provider-smoke-spool"
    )
    profile_dir = browser_dir / "browser-provider-smoke-profile"
    summary_path = browser_dir / "browser-provider-smoke.json"
    result_path = browser_dir / "browser-provider-smoke-result.json"
    popup_screenshot_path = browser_dir / "browser-provider-smoke-popup.png"
    stdout_path = browser_dir / "browser-provider-smoke.stdout"
    stderr_path = browser_dir / "browser-provider-smoke.stderr"
    env_path = browser_dir / "browser-provider-smoke.env.json"
    generated_paths: tuple[Path, ...]
    if spool_path_override is None:
        generated_paths = (spool_path, profile_dir, result_path, popup_screenshot_path)
    else:
        generated_paths = (profile_dir, result_path, popup_screenshot_path)
    for generated_path in generated_paths:
        if generated_path.exists():
            if generated_path.is_dir():
                shutil.rmtree(generated_path)
            else:
                generated_path.unlink()

    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_provider",
        event_type="browser_provider_smoke_requested",
        status="starting",
        payload={
            "extension_root": str(extension_root),
            "profile_dir": str(profile_dir),
            "spool_path": str(spool_path),
            "providers": ["chatgpt", "claude"],
        },
    )

    spool_path.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    server = None
    thread: Thread | None = None
    receiver_url = receiver_url_override
    if receiver_url is None:
        server = make_server(
            "127.0.0.1",
            0,
            spool_path=spool_path,
            auth_token=_RECEIVER_SMOKE_TOKEN,
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        if isinstance(host, bytes):
            host = host.decode("ascii")
        receiver_url = f"http://{host}:{port}"
    env = os.environ.copy()
    env.update(
        {
            "POLYLOGUE_PROVIDER_SMOKE_EXTENSION_ROOT": str(extension_root),
            "POLYLOGUE_PROVIDER_SMOKE_KEEP_PROFILE": "1",
            "POLYLOGUE_PROVIDER_SMOKE_OUT": str(result_path),
            "POLYLOGUE_PROVIDER_SMOKE_TIMEOUT_MS": os.environ.get(
                "POLYLOGUE_PROVIDER_SMOKE_TIMEOUT_MS",
                "25000" if reader_base_url is not None and reader_session_id is not None else "10000",
            ),
            "POLYLOGUE_PROVIDER_SMOKE_PROFILE_DIR": str(profile_dir),
            "POLYLOGUE_PROVIDER_SMOKE_RECEIVER_TOKEN": _RECEIVER_SMOKE_TOKEN,
            "POLYLOGUE_PROVIDER_SMOKE_RECEIVER_URL": receiver_url,
            "POLYLOGUE_PROVIDER_SMOKE_SPOOL_DIR": str(spool_path),
        }
    )
    if session_id is not None:
        env["POLYLOGUE_PROVIDER_SMOKE_SESSION_ID"] = session_id
    if reader_base_url is not None and reader_session_id is not None:
        env["POLYLOGUE_PROVIDER_SMOKE_READER_BASE_URL"] = reader_base_url
        env["POLYLOGUE_PROVIDER_SMOKE_READER_SESSION_ID"] = reader_session_id
    env_path.write_text(json.dumps(_dev_loop_env_snapshot(env), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        result = _run_process_tree(
            ["node", "scripts/dev-loop-provider-smoke.mjs"],
            cwd=extension_root,
            env=env,
            timeout_s=45,
        )
        stdout = result.stdout
        stderr = result.stderr
        result_payload: dict[str, object] | None = None
        exit_code = int(result.exit_code)
        if result.timed_out:
            stderr = (stderr + "\n" if stderr else "") + "browser provider smoke timed out after 45s\n"
        elif result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    provider_statuses, artifact_refs, artifact_paths = _provider_smoke_artifacts(
        result_payload,
        spool_path=spool_path,
    )
    extension_id = None
    manifest = None
    privacy_posture = None
    popup_status = None
    reader_status = None
    if isinstance(result_payload, dict):
        extension_id = result_payload.get("extension_id")
        raw_manifest = result_payload.get("manifest")
        manifest = raw_manifest if isinstance(raw_manifest, dict) else None
        raw_privacy_posture = result_payload.get("privacy_posture")
        privacy_posture = raw_privacy_posture if isinstance(raw_privacy_posture, str) else None
        raw_popup = result_payload.get("popup")
        if isinstance(raw_popup, dict):
            inspection = raw_popup.get("inspection")
            if isinstance(inspection, dict):
                popup_status = {
                    "ok": bool(inspection.get("ok")),
                    "debug_log_count": inspection.get("debugLogCount"),
                    "receiver_event_count": inspection.get("receiverEventCount"),
                    "capture_log_count": inspection.get("captureLogCount"),
                    "has_raw_payload_leak": bool(inspection.get("hasRawPayloadLeak")),
                }
        raw_reader = result_payload.get("reader")
        if isinstance(raw_reader, dict):
            raw_inspection = raw_reader.get("inspection")
            inspection = raw_inspection if isinstance(raw_inspection, dict) else {}
            reader_status = {
                "ok": bool(raw_reader.get("ok")),
                "url": raw_reader.get("url"),
                "selected_session_id": inspection.get("selectedConvId"),
                "message_row_count": inspection.get("messageRowCount"),
                "has_user_turn": bool(inspection.get("hasUserTurn")),
                "has_assistant_turn": bool(inspection.get("hasAssistantTurn")),
                "has_live_chip": bool(inspection.get("hasLiveChip")),
                "has_raw_private_paths": bool(inspection.get("hasRawPrivatePaths")),
            }
    ok = (
        exit_code == 0
        and bool(provider_statuses)
        and all(provider_statuses.values())
        and bool(artifact_paths)
        and all(Path(path).exists() for path in artifact_paths.values())
        and (popup_status is None or (popup_status["ok"] is True and popup_status["has_raw_payload_leak"] is False))
        and (
            reader_status is None
            or (
                reader_status["ok"] is True
                and reader_status["has_user_turn"] is True
                and reader_status["has_assistant_turn"] is True
                and reader_status["has_live_chip"] is True
                and reader_status["has_raw_private_paths"] is False
            )
        )
    )
    payload: dict[str, object] = {
        "ok": ok,
        "authority": _authority("browser_provider_smoke"),
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "extension_id": extension_id,
        "extension_root": str(extension_root),
        "manifest": manifest,
        "privacy_posture": privacy_posture,
        "popup_status": popup_status,
        "reader_status": reader_status,
        "profile_dir": str(profile_dir),
        "receiver_url": receiver_url,
        "spool_path": str(spool_path),
        "provider_statuses": provider_statuses,
        "artifact_refs": artifact_refs,
        "artifact_paths": artifact_paths,
        "result": result_payload,
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
            "result": str(result_path),
            "popup_screenshot": str(popup_screenshot_path) if popup_screenshot_path.exists() else None,
            "env": str(env_path),
            "profile": str(profile_dir),
            "spool": str(spool_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_provider",
        event_type="browser_provider_smoke_finished",
        status="ok" if ok else "failed",
        payload={
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "extension_id": extension_id,
            "provider_statuses": provider_statuses,
            "popup_status": popup_status,
            "reader_status": reader_status,
            "artifact_refs": artifact_refs,
            "artifact_paths": artifact_paths,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


def run_browser_live_proof(
    *,
    preflight: dict[str, Any],
    profile_dir: Path,
    providers: list[str],
    chatgpt_url: str | None,
    claude_url: str | None,
    wait_s: float,
) -> dict[str, object]:
    """Run an operator-local visible browser proof against live provider pages."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    if not profile_dir.exists() or not profile_dir.is_dir():
        raise ValueError("--browser-live-profile-dir must point at an existing ignored local copied Chrome profile")
    browser_dir = Path(str(artifacts["browser_dir"]))
    browser_dir.mkdir(parents=True, exist_ok=True)
    event_path = Path(str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl")))
    extension_root = Path(str(preflight["repo_root"])) / "browser-extension"
    spool_path = browser_dir / "browser-live-proof-spool"
    summary_path = browser_dir / "browser-live-proof.json"
    result_path = browser_dir / "browser-live-proof-result.json"
    stdout_path = browser_dir / "browser-live-proof.stdout"
    stderr_path = browser_dir / "browser-live-proof.stderr"
    env_path = browser_dir / "browser-live-proof.env.json"

    selected_urls = {
        "chatgpt": chatgpt_url or "https://chatgpt.com/",
        "claude": claude_url or "https://claude.ai/",
    }
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_live",
        event_type="browser_live_proof_requested",
        status="starting",
        payload={
            "extension_root": str(extension_root),
            "profile_dir": str(profile_dir),
            "spool_path": str(spool_path),
            "providers": providers,
            "wait_s": wait_s,
        },
    )

    spool_path.mkdir(parents=True, exist_ok=True)
    server = make_server(
        "127.0.0.1",
        0,
        spool_path=spool_path,
        auth_token=_RECEIVER_SMOKE_TOKEN,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    if isinstance(host, bytes):
        host = host.decode("ascii")
    receiver_url = f"http://{host}:{port}"
    env = os.environ.copy()
    env.update(
        {
            "POLYLOGUE_LIVE_PROOF_EXTENSION_ROOT": str(extension_root),
            "POLYLOGUE_LIVE_PROOF_OUT": str(result_path),
            "POLYLOGUE_LIVE_PROOF_PROFILE_DIR": str(profile_dir),
            "POLYLOGUE_LIVE_PROOF_PROVIDERS": ",".join(providers),
            "POLYLOGUE_LIVE_PROOF_RECEIVER_TOKEN": _RECEIVER_SMOKE_TOKEN,
            "POLYLOGUE_LIVE_PROOF_RECEIVER_URL": receiver_url,
            "POLYLOGUE_LIVE_PROOF_SPOOL_DIR": str(spool_path),
            "POLYLOGUE_LIVE_PROOF_WAIT_MS": str(max(0, int(wait_s * 1000))),
            "POLYLOGUE_LIVE_PROOF_TIMEOUT_MS": str(max(30_000, int(wait_s * 1000) + 75_000)),
        }
    )
    if "chatgpt" in providers:
        env["POLYLOGUE_LIVE_PROOF_CHATGPT_URL"] = selected_urls["chatgpt"]
    if "claude" in providers:
        env["POLYLOGUE_LIVE_PROOF_CLAUDE_URL"] = selected_urls["claude"]
    env_path.write_text(json.dumps(_dev_loop_env_snapshot(env), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        result = _run_process_tree(
            ["node", "scripts/dev-loop-live-provider-proof.mjs"],
            cwd=extension_root,
            env=env,
            timeout_s=max(45.0, wait_s + 90.0),
        )
        stdout = result.stdout
        stderr = result.stderr
        result_payload: dict[str, object] | None = None
        exit_code = int(result.exit_code)
        if result.timed_out:
            stderr = (stderr + "\n" if stderr else "") + f"browser live proof timed out after {wait_s + 90.0:.1f}s\n"
        elif result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    provider_statuses, artifact_refs, artifact_paths = _provider_smoke_artifacts(
        result_payload,
        spool_path=spool_path,
    )
    extension_id = None
    manifest = None
    privacy_posture = (
        "operator-local copied-profile proof; summary omits raw turn text; local spool may contain raw capture content"
    )
    if isinstance(result_payload, dict):
        extension_id = result_payload.get("extension_id")
        raw_manifest = result_payload.get("manifest")
        manifest = raw_manifest if isinstance(raw_manifest, dict) else None
        raw_privacy_posture = result_payload.get("privacy_posture")
        if isinstance(raw_privacy_posture, str):
            privacy_posture = raw_privacy_posture
    ok = (
        exit_code == 0
        and set(provider_statuses) == set(providers)
        and all(provider_statuses.get(provider) is True for provider in providers)
        and bool(artifact_paths)
        and all(Path(path).exists() for path in artifact_paths.values())
    )
    payload: dict[str, object] = {
        "ok": ok,
        "authority": _authority("browser_live_proof"),
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "extension_id": extension_id,
        "extension_root": str(extension_root),
        "manifest": manifest,
        "privacy_posture": privacy_posture,
        "profile_dir": str(profile_dir),
        "providers": providers,
        "provider_urls_redacted": {
            provider: _redact_live_provider_url(selected_urls[provider]) for provider in providers
        },
        "provider_url_sha256": {provider: _hash_text(selected_urls[provider]) for provider in providers},
        "receiver_url": receiver_url,
        "spool_path": str(spool_path),
        "provider_statuses": provider_statuses,
        "artifact_refs": artifact_refs,
        "artifact_paths": artifact_paths,
        "result": result_payload,
        "artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
            "result": str(result_path),
            "env": str(env_path),
            "profile": str(profile_dir),
            "spool": str(spool_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_live",
        event_type="browser_live_proof_finished",
        status="ok" if ok else "failed",
        payload={
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "extension_id": extension_id,
            "providers": providers,
            "provider_statuses": provider_statuses,
            "artifact_refs": artifact_refs,
            "artifact_paths": artifact_paths,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


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


def build_browser_plan(*, preflight: dict[str, Any]) -> dict[str, object]:
    """Write a browser-control plan for loading the unpacked extension locally."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    browser_dir = Path(str(artifacts["browser_dir"]))
    browser_dir.mkdir(parents=True, exist_ok=True)
    event_path = Path(str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl")))

    extension_root = Path(str(preflight["repo_root"])) / "browser-extension"
    profile_dir = browser_dir / "chrome-profile"
    screenshot_dir = browser_dir / "screenshots"
    downloads_dir = browser_dir / "downloads"
    live_profile_root = Path(str(preflight["repo_root"])) / ".local" / "browser-profiles"
    live_profile_dir = _default_live_profile_dir(preflight)
    live_spool_dir = browser_dir / "browser-live-proof-spool"
    live_env_example_path = browser_dir / "browser-live-proof.env.example"
    live_checklist_path = browser_dir / "browser-live-proof-checklist.md"
    for path in (profile_dir, screenshot_dir, downloads_dir, live_profile_root, live_spool_dir):
        path.mkdir(parents=True, exist_ok=True)

    ports = preflight.get("ports")
    if not isinstance(ports, dict):
        raise ValueError("preflight payload is missing port status")
    api_status = ports.get("api")
    receiver_status = ports.get("browser_capture")
    if not isinstance(api_status, dict) or not isinstance(receiver_status, dict):
        raise ValueError("preflight payload is missing API/browser-capture port status")
    api_port = int(api_status["port"])
    receiver_port = int(receiver_status["port"])
    receiver_url = f"http://127.0.0.1:{receiver_port}"
    web_shell_url = f"http://127.0.0.1:{api_port}/"
    receiver_token = os.environ.get("POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN", "")

    chromium_command = [
        "chromium",
        f"--user-data-dir={profile_dir}",
        "--enable-unsafe-extension-debugging",
        f"--unsafely-treat-insecure-origin-as-secure={receiver_url}",
        f"--disable-extensions-except={extension_root}",
        f"--load-extension={extension_root}",
        "--new-window",
        web_shell_url,
    ]
    google_chrome_command = ["google-chrome-stable", *chromium_command[1:]]
    live_launch_command = [
        "chromium",
        f"--user-data-dir={live_profile_dir}",
        "--enable-unsafe-extension-debugging",
        f"--unsafely-treat-insecure-origin-as-secure={receiver_url}",
        f"--disable-extensions-except={extension_root}",
        f"--load-extension={extension_root}",
        "--new-window",
        "https://chatgpt.com/",
        "https://claude.ai/",
    ]
    live_urls = _default_live_provider_urls()
    run_live_proof_command = [
        "devtools",
        "workspace",
        "dev-loop",
        "--archive-root",
        str(preflight["dev_archive_root"]),
        "--log-dir",
        str(preflight["log_dir"]),
        "--api-port",
        str(api_port),
        "--browser-capture-port",
        str(receiver_port),
        "--browser-live-proof",
        "--browser-live-profile-dir",
        str(live_profile_dir),
        "--browser-live-chatgpt-url",
        live_urls["chatgpt"],
        "--browser-live-claude-url",
        live_urls["claude"],
    ]
    copy_profile_command = (
        ': "${POLYLOGUE_SOURCE_CHROME_USER_DATA_DIR:?set this to a Chrome/Chromium user-data-dir source}" && '
        f"mkdir -p {shlex.quote(str(live_profile_root))} && "
        "rsync -a --delete "
        "--exclude='Singleton*' --exclude='Crashpad' --exclude='Crash Reports' "
        "--exclude='GrShaderCache' --exclude='ShaderCache' --exclude='GPUCache' --exclude='Code Cache' "
        '"${POLYLOGUE_SOURCE_CHROME_USER_DATA_DIR%/}/" '
        f"{shlex.quote(str(live_profile_dir))}/"
    )
    configure_steps = [
        "Open the extension popup.",
        f"Set Local receiver URL to {receiver_url}.",
        (
            "Leave receiver token empty."
            if not receiver_token
            else "Set receiver token from POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN in your local shell."
        ),
        "Open a supported ChatGPT or Claude.ai page with an operator-approved profile/cookie copy.",
        "Use the browser/DevTools control plane to inspect popup state, service-worker logs, network requests, and screenshots.",
    ]
    plan_artifacts = {
        "json": str(browser_dir / "browser-plan.json"),
        "markdown": str(browser_dir / "browser-plan.md"),
        "events": str(event_path),
        "live_proof_checklist": str(live_checklist_path),
        "live_proof_env_example": str(live_env_example_path),
        "live_proof_summary": str(browser_dir / "browser-live-proof.json"),
        "live_proof_result": str(browser_dir / "browser-live-proof-result.json"),
        "live_proof_spool": str(live_spool_dir),
        "live_profile_root": str(live_profile_root),
    }
    live_proof = {
        "profile_copy_root": str(live_profile_root),
        "suggested_profile_dir": str(live_profile_dir),
        "summary_path": str(browser_dir / "browser-live-proof.json"),
        "result_path": str(browser_dir / "browser-live-proof-result.json"),
        "spool_path": str(live_spool_dir),
        "providers": ["chatgpt", "claude"],
        "provider_url_templates": live_urls,
        "commands": {
            "copy_profile_template": copy_profile_command,
            "visible_launch_copied_profile": live_launch_command,
            "run_live_proof": run_live_proof_command,
            "inspect_run": ["devtools", "workspace", "dev-loop", "--inspect-run", str(preflight["run_log_dir"])],
        },
        "operator_boundary": {
            "profile_copy_policy": "use an operator-approved copied Chrome user-data-dir under an ignored local path",
            "ci_policy": "the live proof command refuses CI/cloud runs by default; use deterministic smokes in CI",
            "summary_policy": "proof summaries redact URLs/session ids and omit raw turn text; receiver spool remains ignored local state",
        },
    }
    payload: dict[str, object] = {
        "ok": True,
        "run_id": str(preflight["run_id"]),
        "extension_root": str(extension_root),
        "profile_dir": str(profile_dir),
        "screenshot_dir": str(screenshot_dir),
        "downloads_dir": str(downloads_dir),
        "receiver_url": receiver_url,
        "receiver_auth_configured": bool(receiver_token),
        "web_shell_url": web_shell_url,
        "commands": {
            "preferred": chromium_command,
            "chromium": chromium_command,
            "google_chrome": google_chrome_command,
        },
        "supported_probe_urls": {
            "chatgpt": "https://chatgpt.com/",
            "claude": "https://claude.ai/",
        },
        "configure_steps": configure_steps,
        "live_provider_proof": live_proof,
        "safety": {
            "profile_dir_policy": "ignored local dev-loop artifact; never commit copied cookies or profile state",
            "ci_policy": "not used in CI or cloud agents",
            "repo_boundary": "Polylogue writes launch/config artifacts; browser control and visual inspection are local agent/operator capabilities",
        },
        "artifacts": plan_artifacts,
    }
    json_path = Path(plan_artifacts["json"])
    markdown_path = Path(plan_artifacts["markdown"])
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    live_env_example = "\n".join(
        [
            "# Polylogue live provider proof template.",
            "# Edit the source path and provider URLs before running. Keep profile copies local and ignored.",
            'export POLYLOGUE_SOURCE_CHROME_USER_DATA_DIR="${HOME}/.config/google-chrome"',
            copy_profile_command,
            "",
            shlex.join([str(part) for part in run_live_proof_command]),
            "",
        ]
    )
    live_env_example_path.write_text(live_env_example, encoding="utf-8")
    live_checklist = "\n".join(
        [
            "# Polylogue Live Provider Proof Checklist",
            "",
            f"- Run id: `{payload['run_id']}`",
            f"- Copied profile target: `{live_profile_dir}`",
            f"- Extension root: `{extension_root}`",
            f"- Receiver proof spool: `{live_spool_dir}`",
            "",
            "## Before running",
            "",
            "1. Close Chrome/Chromium for the source profile or make a fresh operator-approved copy.",
            "2. Copy the user-data-dir into the suggested ignored path with `Singleton*` lock files excluded.",
            "3. Replace the placeholder ChatGPT/Claude URLs with conversation pages that are already visible in the copied profile.",
            "4. Keep screenshots, downloads, receiver spool, and copied profile data under the run-local ignored directories.",
            "",
            "## Proof command",
            "",
            "```bash",
            shlex.join([str(part) for part in run_live_proof_command]),
            "```",
            "",
            "The command opens a visible local Chrome, configures the unpacked extension, waits for the operator-controlled pages to settle, asks the content scripts to capture, and writes a redacted summary. Raw captured content, if any, is only in the ignored receiver spool.",
            "",
            "## Closeout evidence",
            "",
            f"- Summary: `{browser_dir / 'browser-live-proof.json'}`",
            f"- Result: `{browser_dir / 'browser-live-proof-result.json'}`",
            f"- Stdout/stderr: `{browser_dir / 'browser-live-proof.stdout'}`, `{browser_dir / 'browser-live-proof.stderr'}`",
            f"- Receiver spool: `{live_spool_dir}`",
            f"- Inspect run: `devtools workspace dev-loop --inspect-run {preflight['run_log_dir']}`",
            "",
        ]
    )
    live_checklist_path.write_text(live_checklist, encoding="utf-8")
    markdown = "\n".join(
        [
            "# Polylogue Browser Dev-Loop Plan",
            "",
            f"- Run id: `{payload['run_id']}`",
            f"- Extension root: `{payload['extension_root']}`",
            f"- Browser profile: `{payload['profile_dir']}`",
            f"- Receiver URL: `{payload['receiver_url']}`",
            f"- Web shell: `{payload['web_shell_url']}`",
            "",
            "## Launch",
            "",
            "Preferred automated browser:",
            "",
            "```bash",
            shlex.join(chromium_command),
            "```",
            "",
            "Branded Chrome fallback for manual/visible runs:",
            "",
            "```bash",
            shlex.join(google_chrome_command),
            "```",
            "",
            "## Configure",
            "",
            *[f"{index}. {step}" for index, step in enumerate(configure_steps, start=1)],
            "",
            "## Operator-Local Live Provider Proof",
            "",
            "Use this only with an operator-approved copied browser profile. It is not a CI/cloud test and it does not write raw conversation text into the JSON summary.",
            "",
            "Copy profile template:",
            "",
            "```bash",
            copy_profile_command,
            "```",
            "",
            "Run proof template:",
            "",
            "```bash",
            shlex.join([str(part) for part in run_live_proof_command]),
            "```",
            "",
            f"Checklist: `{live_checklist_path}`",
            f"Environment template: `{live_env_example_path}`",
            "",
            "## Artifact Paths",
            "",
            f"- Screenshots: `{payload['screenshot_dir']}`",
            f"- Downloads: `{payload['downloads_dir']}`",
            f"- JSON plan: `{json_path}`",
            f"- Live proof summary: `{plan_artifacts['live_proof_summary']}`",
            f"- Live proof spool: `{plan_artifacts['live_proof_spool']}`",
            "",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser",
        event_type="browser_plan_written",
        status="ok",
        payload={
            "extension_root": str(extension_root),
            "profile_dir": str(profile_dir),
            "receiver_url": receiver_url,
            "web_shell_url": web_shell_url,
            "live_provider_proof": live_proof,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


def build_tui_plan(*, preflight: dict[str, Any]) -> dict[str, object]:
    """Write a local terminal/TUI visual-inspection plan for this dev-loop run."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    tui_dir = Path(str(artifacts["tui_dir"]))
    terminal_dir = Path(str(artifacts["terminal_dir"]))
    tui_dir.mkdir(parents=True, exist_ok=True)
    terminal_dir.mkdir(parents=True, exist_ok=True)
    event_path = Path(str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl")))

    env_exports = {**_preflight_suggested_env(preflight), "POLYLOGUE_FORCE_PLAIN": "0"}
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_exports.items())
    typescript_path = tui_dir / "polylogue-ops-status.typescript"
    vhs_tape_path = tui_dir / "polylogue-status.tape"
    vhs_gif_path = tui_dir / "polylogue-status.gif"
    screenshot_dir = tui_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    plan_artifacts = {
        "json": str(tui_dir / "tui-plan.json"),
        "markdown": str(tui_dir / "tui-plan.md"),
        "typescript": str(typescript_path),
        "vhs_tape": str(vhs_tape_path),
        "vhs_gif": str(vhs_gif_path),
        "screenshots": str(screenshot_dir),
        "events": str(event_path),
    }
    script_command = (
        f"script -q -c {shlex.quote(env_prefix + ' polylogue ops status')} {shlex.quote(str(typescript_path))}"
    )
    vhs_tape = "\n".join(
        [
            f"Output {shlex.quote(str(vhs_gif_path))}",
            'Set Shell "zsh"',
            "Set FontSize 18",
            "Set Width 1200",
            "Set Height 720",
            f"Type {shlex.quote(env_prefix + ' polylogue ops status')}",
            "Enter",
            "Sleep 2s",
            "",
        ]
    )
    vhs_tape_path.write_text(vhs_tape, encoding="utf-8")
    payload: dict[str, object] = {
        "ok": True,
        "run_id": str(preflight["run_id"]),
        "env": env_exports,
        "commands": {
            "script_status": script_command,
            "vhs_render": f"vhs {shlex.quote(str(vhs_tape_path))}",
            "manual_tui_dir": str(tui_dir),
        },
        "artifacts": plan_artifacts,
        "notes": [
            "Use script output for ordinary CLI transcript evidence.",
            "Use VHS or the local terminal-control surface for full-screen TUI/visual playback.",
            "Keep generated recordings under the ignored run-local tui directory.",
        ],
    }
    json_path = Path(plan_artifacts["json"])
    markdown_path = Path(plan_artifacts["markdown"])
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = "\n".join(
        [
            "# Polylogue TUI Dev-Loop Plan",
            "",
            f"- Run id: `{payload['run_id']}`",
            f"- TUI artifacts: `{tui_dir}`",
            f"- Terminal artifacts: `{terminal_dir}`",
            "",
            "## Environment",
            "",
            "```bash",
            *(f"export {key}={shlex.quote(value)}" for key, value in env_exports.items()),
            "```",
            "",
            "## Script Capture",
            "",
            "```bash",
            script_command,
            "```",
            "",
            "## VHS",
            "",
            f"Edit `{vhs_tape_path}` if the command needs to drive a richer flow, then run:",
            "",
            "```bash",
            f"vhs {shlex.quote(str(vhs_tape_path))}",
            "```",
            "",
            "## Artifact Paths",
            "",
            f"- Typescript: `{typescript_path}`",
            f"- VHS tape: `{vhs_tape_path}`",
            f"- VHS GIF: `{vhs_gif_path}`",
            f"- Screenshots: `{screenshot_dir}`",
            "",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="tui",
        event_type="tui_plan_written",
        status="ok",
        payload={
            "artifacts": payload["artifacts"],
            "commands": payload["commands"],
        },
    )
    return payload


def _read_json_file(path: Path) -> object | None:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None


def _read_event_rows(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    malformed: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows, malformed
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            preview = line[:500]
            _LOG.warning("Skipping malformed dev-loop event row in %s:%s: %r", path, line_number, preview)
            malformed.append({"line": line_number, "error": str(exc), "text": preview})
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, malformed


def _payload_artifacts(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    artifacts = payload.get("artifacts")
    return artifacts if isinstance(artifacts, dict) else {}


def _payload_duration_ms(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    duration = payload.get("duration_ms")
    if isinstance(duration, int):
        return duration
    if isinstance(duration, float):
        return int(duration)
    return None


def _timed_event_duration(row: dict[str, object]) -> int:
    duration = row.get("duration_ms")
    return duration if isinstance(duration, int) else 0


def summarize_dev_loop_run(run_dir: Path) -> dict[str, object]:
    """Summarize one dev-loop run directory from persisted local artifacts."""

    run_dir = run_dir.resolve()
    preflight_path = run_dir / "preflight.json"
    events_path = run_dir / "dev-loop.events.jsonl"
    preflight = _read_json_file(preflight_path)
    events, malformed_event_lines = _read_event_rows(events_path)
    event_status_counts: dict[str, int] = {}
    event_surface_counts: dict[str, int] = {}
    problem_events: list[dict[str, object]] = []
    timed_events: list[dict[str, object]] = []
    for row in events:
        status = str(row.get("status") or "unknown")
        surface = str(row.get("surface") or "unknown")
        event_status_counts[status] = event_status_counts.get(status, 0) + 1
        event_surface_counts[surface] = event_surface_counts.get(surface, 0) + 1
        if status in {"failed", "blocked"}:
            problem_events.append(row)
        duration_ms = _payload_duration_ms(row.get("payload"))
        if duration_ms is not None:
            timed_events.append(
                {
                    "surface": surface,
                    "event_type": str(row.get("event_type") or "unknown"),
                    "status": status,
                    "duration_ms": duration_ms,
                }
            )

    summary_files = {
        "daemon_launch": run_dir / "polylogued.launch.json",
        "browser_plan": run_dir / "browser" / "browser-plan.json",
        "browser_smoke": run_dir / "browser" / "browser-smoke.json",
        "browser_provider_smoke": run_dir / "browser" / "browser-provider-smoke.json",
        "browser_provider_live_follow": run_dir / "browser" / "browser-provider-live-follow.json",
        "browser_live_proof": run_dir / "browser" / "browser-live-proof.json",
        "extension_smoke": run_dir / "browser" / "extension-smoke.json",
        "tui_plan": run_dir / "tui" / "tui-plan.json",
    }
    loaded_summaries = {name: _read_json_file(path) for name, path in summary_files.items() if path.exists()}
    terminal_summaries = sorted((run_dir / "terminal").glob("*.summary.json"))
    loaded_terminal = [
        payload for payload in (_read_json_file(path) for path in terminal_summaries) if isinstance(payload, dict)
    ]
    failed_summaries = [
        {
            "name": name,
            "exit_code": payload.get("exit_code"),
            "duration_ms": payload.get("duration_ms"),
            "artifacts": _payload_artifacts(payload),
        }
        for name, payload in loaded_summaries.items()
        if isinstance(payload, dict) and payload.get("ok") is False
    ]
    failed_terminal = [
        {
            "command_text": payload.get("command_text"),
            "exit_code": payload.get("exit_code"),
            "duration_ms": payload.get("duration_ms"),
            "artifacts": _payload_artifacts(payload),
        }
        for payload in loaded_terminal
        if payload.get("ok") is False
    ]
    artifact_index: dict[str, object] = {}
    for name, payload in loaded_summaries.items():
        artifacts = _payload_artifacts(payload)
        if artifacts:
            artifact_index[name] = artifacts
    if loaded_terminal:
        artifact_index["terminal_captures"] = [
            _payload_artifacts(payload) for payload in loaded_terminal if _payload_artifacts(payload)
        ]
    slowest_events = sorted(timed_events, key=_timed_event_duration, reverse=True)[:10]
    missing_artifacts = [
        str(path)
        for path in (
            preflight_path,
            events_path,
            run_dir / "polylogued.log",
        )
        if not path.exists()
    ]
    warnings: list[str] = []
    if preflight is None:
        warnings.append("preflight.json is missing or unreadable")
    if not events:
        warnings.append("dev-loop.events.jsonl is missing or empty")
    if malformed_event_lines:
        warnings.append(f"{len(malformed_event_lines)} malformed dev-loop event row(s) were skipped")
    if problem_events:
        warnings.append(f"{len(problem_events)} event(s) have failed/blocked status")
    if failed_summaries:
        warnings.append(f"{len(failed_summaries)} run summary file(s) report failure")
    if failed_terminal:
        warnings.append(f"{len(failed_terminal)} terminal capture(s) report failure")
    return {
        "ok": not warnings,
        "run_dir": str(run_dir),
        "run_id": preflight.get("run_id") if isinstance(preflight, dict) else run_dir.name,
        "preflight": preflight if isinstance(preflight, dict) else None,
        "event_count": len(events),
        "event_status_counts": event_status_counts,
        "event_surface_counts": event_surface_counts,
        "last_event": events[-1] if events else None,
        "problem_events": problem_events,
        "malformed_event_lines": malformed_event_lines,
        "slowest_events": slowest_events,
        "summaries": loaded_summaries,
        "failed_summaries": failed_summaries,
        "terminal_captures": loaded_terminal,
        "terminal_capture_count": len(loaded_terminal),
        "failed_terminal_captures": failed_terminal,
        "artifact_index": artifact_index,
        "missing_artifacts": missing_artifacts,
        "warnings": warnings,
    }


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
    event_path = Path(str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl")))

    env = os.environ.copy()
    suggested_env = preflight.get("suggested_env")
    if isinstance(suggested_env, dict):
        env.update({str(key): str(value) for key, value in suggested_env.items()})
    env.setdefault("POLYLOGUE_FORCE_PLAIN", "1")
    repo_root = Path(str(preflight["repo_root"]))
    _prepend_pythonpath(env, repo_root)

    artifact_name = _safe_artifact_name(command)
    stdout_path = terminal_dir / f"{artifact_name}.stdout"
    stderr_path = terminal_dir / f"{artifact_name}.stderr"
    transcript_path = terminal_dir / f"{artifact_name}.transcript.txt"
    env_path = terminal_dir / f"{artifact_name}.env.json"
    summary_path = terminal_dir / f"{artifact_name}.summary.json"

    started_at = datetime.now(UTC)
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="cli",
        event_type="cli_capture_requested",
        status="starting",
        payload={
            "command": command,
            "command_text": shlex.join(command),
            "artifact_name": artifact_name,
            "timeout_s": timeout_s,
        },
    )
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
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="cli",
        event_type="cli_capture_finished",
        status="ok" if exit_code == 0 else "failed",
        payload={
            "command": command,
            "command_text": shlex.join(command),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


def launch_branch_daemon(
    *,
    preflight: dict[str, Any],
    readiness_timeout_s: float,
    full_source_catchup: bool = False,
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
    repo_root = Path(str(preflight["repo_root"]))
    _prepend_pythonpath(env, repo_root)
    archive_preparation = _prepare_archive_for_daemon_launch(preflight)

    api_status = ports["api"]
    receiver_status = ports["browser_capture"]
    assert isinstance(api_status, dict)
    assert isinstance(receiver_status, dict)
    api_port = int(api_status["port"])
    receiver_port = int(receiver_status["port"])
    data_home = Path(
        str(env.get("XDG_DATA_HOME", repo_root / ".cache" / "dev-loop" / str(preflight["run_id"]) / "xdg-data"))
    )
    spool_path = data_home / "polylogue" / "browser-capture"
    spool_path.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-c",
        "from polylogue.daemon.cli import main; main()",
        "run",
        "--spool",
        str(spool_path),
        "--api-port",
        str(api_port),
        "--port",
        str(receiver_port),
    ]
    if not full_source_catchup:
        command.extend(["--root", str(spool_path), "--no-source-catchup"])

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
            "cwd": str(repo_root),
            "api_port": api_port,
            "browser_capture_port": receiver_port,
            "spool_path": str(spool_path),
            "log_path": str(daemon_log),
            "archive_preparation": archive_preparation,
            "full_source_catchup": full_source_catchup,
        },
    )
    with daemon_log.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n# dev-loop launch {started_at.isoformat()} run_id={preflight['run_id']}\n$ {shlex.join(command)}\n"
        )
        log_file.flush()
        process = _start_daemon_process(
            command,
            cwd=repo_root,
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
        "cwd": str(repo_root),
        "run_id": str(preflight["run_id"]),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "archive_preparation": archive_preparation,
        "api_ready": api_ready,
        "browser_capture_ready": receiver_ready,
        "readiness_timeout_s": readiness_timeout_s,
        "full_source_catchup": full_source_catchup,
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


def run_browser_provider_live_follow(
    *,
    preflight: dict[str, Any],
    readiness_timeout_s: float,
    archive_timeout_s: float,
    full_source_catchup: bool = False,
) -> dict[str, object]:
    """Prove deterministic browser captures flow through a branch daemon into reads."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("preflight payload is missing artifact paths")
    browser_dir = Path(str(artifacts["browser_dir"]))
    browser_dir.mkdir(parents=True, exist_ok=True)
    event_path = Path(str(artifacts.get("dev_events", Path(str(preflight["run_log_dir"])) / "dev-loop.events.jsonl")))
    summary_path = browser_dir / "browser-provider-live-follow.json"
    session_id = f"polylogue-dev-loop-live-follow-{preflight['run_id']}"
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_provider_live_follow",
        event_type="browser_provider_live_follow_requested",
        status="starting",
        payload={
            "session_id": session_id,
            "readiness_timeout_s": readiness_timeout_s,
            "archive_timeout_s": archive_timeout_s,
            "full_source_catchup": full_source_catchup,
        },
    )

    started = time.monotonic()
    launch_payload: dict[str, object] | None = None
    smoke_payload: dict[str, object] | None = None
    archive_states: dict[str, object] = {}
    api_messages: dict[str, object] = {}
    stop_payload: dict[str, object] | None = None
    error_payload: dict[str, object] = {}
    try:
        launch_payload = launch_branch_daemon(
            preflight=preflight,
            readiness_timeout_s=readiness_timeout_s,
            full_source_catchup=full_source_catchup,
        )
        if launch_payload.get("ok") is not True:
            raise RuntimeError("branch daemon did not become ready")
        ports = preflight.get("ports")
        if not isinstance(ports, dict):
            raise RuntimeError("preflight payload is missing ports")
        api_status = ports.get("api")
        receiver_status = ports.get("browser_capture")
        if not isinstance(api_status, dict) or not isinstance(receiver_status, dict):
            raise RuntimeError("preflight payload has malformed ports")
        receiver_url = f"http://127.0.0.1:{int(receiver_status['port'])}"
        api_url = f"http://127.0.0.1:{int(api_status['port'])}"
        smoke_payload = run_browser_provider_smoke(
            preflight=preflight,
            receiver_url_override=receiver_url,
            spool_path_override=Path(str(launch_payload["spool_path"])),
            session_id=session_id,
            reader_base_url=api_url,
            reader_session_id=_session_id_for_provider("chatgpt", session_id),
        )
        result_payload = smoke_payload.get("result")
        provider_rows = result_payload.get("providers") if isinstance(result_payload, dict) else None
        if isinstance(provider_rows, dict):
            for provider_key, provider_payload in provider_rows.items():
                if not isinstance(provider_payload, dict):
                    continue
                provider = provider_payload.get("provider")
                provider_session_id = provider_payload.get("provider_session_id")
                if not isinstance(provider, str) or not isinstance(provider_session_id, str):
                    continue
                archive_states[str(provider_key)] = _poll_archive_state(
                    receiver_url=receiver_url,
                    provider=provider,
                    provider_session_id=provider_session_id,
                    timeout_s=archive_timeout_s,
                    interval_s=0.25,
                )
                api_messages[str(provider_key)] = _fetch_api_messages(
                    api_url=api_url,
                    session_id=_session_id_for_provider(provider, provider_session_id),
                    limit=5,
                )
    except Exception as exc:
        error_payload = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if isinstance(launch_payload, dict):
            pid = launch_payload.get("pid")
            if isinstance(pid, int):
                stop_payload = _terminate_pid_tree(pid)

    duration_ms = int((time.monotonic() - started) * 1000)
    provider_statuses: dict[str, bool] = {}
    if isinstance(smoke_payload, dict):
        raw_provider_statuses = smoke_payload.get("provider_statuses")
        if isinstance(raw_provider_statuses, dict):
            provider_statuses = {str(key): value is True for key, value in raw_provider_statuses.items()}
    archive_ok = bool(archive_states) and all(
        isinstance(state, dict) and state.get("ok") is True for state in archive_states.values()
    )
    api_ok = bool(api_messages) and all(
        isinstance(state, dict) and state.get("ok") is True for state in api_messages.values()
    )
    reader_status = smoke_payload.get("reader_status") if isinstance(smoke_payload, dict) else None
    reader_ok = isinstance(reader_status, dict) and reader_status.get("ok") is True
    ok = (
        not error_payload
        and isinstance(launch_payload, dict)
        and launch_payload.get("ok") is True
        and isinstance(smoke_payload, dict)
        and smoke_payload.get("ok") is True
        and archive_ok
        and api_ok
        and reader_ok
    )
    payload: dict[str, object] = {
        "ok": ok,
        "authority": _authority("browser_provider_live_follow"),
        "session_id": session_id,
        "duration_ms": duration_ms,
        "provider_statuses": provider_statuses,
        "archive_ok": archive_ok,
        "api_ok": api_ok,
        "reader_ok": reader_ok,
        "error": error_payload or None,
        "daemon_launch": launch_payload,
        "browser_provider_smoke": smoke_payload,
        "archive_states": archive_states,
        "api_messages": api_messages,
        "reader": reader_status,
        "daemon_stop": stop_payload,
        "artifacts": {
            "summary": str(summary_path),
            "events": str(event_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dev_loop_event(
        event_path,
        preflight=preflight,
        surface="browser_provider_live_follow",
        event_type="browser_provider_live_follow_finished",
        status="ok" if ok else "failed",
        payload={
            "duration_ms": duration_ms,
            "session_id": session_id,
            "provider_statuses": provider_statuses,
            "archive_ok": archive_ok,
            "api_ok": api_ok,
            "reader_ok": reader_ok,
            "error": error_payload or None,
            "artifacts": payload["artifacts"],
        },
    )
    return payload


def _dev_loop_run_id(*, branch: str | None, commit: str | None, api_port: int, browser_capture_port: int) -> str:
    branch_part = _RUN_ID_SAFE_RE.sub("-", branch or "detached").strip("-") or "detached"
    commit_part = _RUN_ID_SAFE_RE.sub("-", commit or "unknown").strip("-") or "unknown"
    return f"{branch_part}-{commit_part}-api{api_port}-capture{browser_capture_port}"


def _authority(name: str) -> dict[str, object]:
    authority = _DEV_LOOP_AUTHORITIES[name]
    return dict(authority)


def _dev_loop_cli_args(
    *,
    api_port: int,
    browser_capture_port: int,
    archive: Path,
    logs: Path,
    extra_args: list[str],
) -> list[str]:
    return [
        "devtools",
        "workspace",
        "dev-loop",
        "--api-port",
        str(api_port),
        "--browser-capture-port",
        str(browser_capture_port),
        "--archive-root",
        str(archive),
        "--log-dir",
        str(logs),
        *extra_args,
    ]


def _artifact_state(
    path: Path,
    *,
    kind: str,
    expected_after: str,
    required_now: bool = False,
) -> dict[str, object]:
    exists = path.exists()
    state = "present" if exists else "degraded" if required_now else "planned"
    return {
        "path": str(path),
        "kind": kind,
        "exists": exists,
        "state": state,
        "expected_after": expected_after,
    }


def _archive_status(archive: Path) -> dict[str, object]:
    index_db = archive / "index.db"
    tier_statuses: dict[str, dict[str, object]] = {}
    all_tiers_ready = True
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        tier_path = archive / spec.filename
        user_version = _read_sqlite_user_version(tier_path) if tier_path.exists() else None
        ready = user_version == spec.version
        all_tiers_ready = all_tiers_ready and ready
        tier_statuses[tier.value] = {
            "path": str(tier_path),
            "exists": tier_path.exists(),
            "user_version": user_version,
            "expected_user_version": spec.version,
            "ready": ready,
            "durability": spec.durability,
        }
    status: dict[str, object] = {
        "archive_root": str(archive),
        "archive_root_exists": archive.exists(),
        "index_db": str(index_db),
        "index_db_exists": index_db.exists(),
        "schema_ready": False,
        "sessions_table_exists": False,
        "session_count": None,
        "message_count": None,
        "user_version": None,
        "tiers": tier_statuses,
        "error": None,
    }
    if not index_db.exists():
        status["error"] = "index.db missing"
        return status
    try:
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            status["user_version"] = user_version
            sessions_exists = (
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()
                is not None
            )
            status["sessions_table_exists"] = sessions_exists
            status["schema_ready"] = all_tiers_ready and user_version > 0 and sessions_exists
            if sessions_exists:
                status["session_count"] = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or 0)
            messages_exists = (
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'").fetchone()
                is not None
            )
            if messages_exists:
                status["message_count"] = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0)
    except sqlite3.Error as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _read_sqlite_user_version(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    except sqlite3.Error:
        return None


def _default_dev_archive_root(repo_root: Path) -> Path:
    return repo_root / ".local" / "dev-archive"


def _is_default_dev_archive(repo_root: Path, archive_root: Path) -> bool:
    try:
        return archive_root.expanduser().resolve() == _default_dev_archive_root(repo_root).expanduser().resolve()
    except OSError:
        return archive_root.expanduser().absolute() == _default_dev_archive_root(repo_root).expanduser().absolute()


def _prepare_archive_for_daemon_launch(preflight: dict[str, Any]) -> dict[str, object]:
    """Ensure the branch-local daemon archive is usable before launch.

    The repo-default ``.local/dev-archive`` is disposable dev-loop state, so the
    launcher may replace stale tier files there. Custom archive roots are not
    implicitly replaced; those may point at operator data and must be repaired by
    an explicit archive command.
    """
    archive_root = Path(str(preflight["dev_archive_root"]))
    repo_root = Path(str(preflight["repo_root"]))
    before = _archive_status(archive_root)
    if before.get("schema_ready") is True:
        return {"action": "unchanged", "archive_status": before}
    if not _is_default_dev_archive(repo_root, archive_root):
        not_ready = [
            name
            for name, tier in cast(dict[str, dict[str, object]], before.get("tiers", {})).items()
            if tier.get("ready") is not True
        ]
        raise ValueError(
            "branch-local daemon archive is not schema-ready for custom archive root "
            f"{archive_root}; not-ready tiers: {', '.join(not_ready) or 'unknown'}"
        )
    try:
        result = initialize_archive_tier_files(archive_root=archive_root, replace_existing=True)
    except ArchiveInitBlockedError as exc:
        raise ValueError(f"branch-local dev archive initialization blocked: {exc}") from exc
    after = _archive_status(archive_root)
    initialized = [
        {
            "tier": tier.tier,
            "action": tier.action.value,
            "path": str(tier.path),
            "backup_path": str(tier.backup_path) if tier.backup_path is not None else None,
        }
        for tier in result.tier_results
    ]
    preflight["archive_status"] = after
    return {
        "action": "initialized_default_dev_archive",
        "archive_status": after,
        "initialized": initialized,
    }


def _dev_loop_artifact_status(
    *,
    archive: Path,
    logs: Path,
    run_log_dir: Path,
    daemon_log: Path,
    browser_artifact_dir: Path,
    terminal_artifact_dir: Path,
    tui_artifact_dir: Path,
    preflight_json: Path,
    dev_events: Path,
    prepare: bool,
) -> dict[str, dict[str, object]]:
    return {
        "archive_root": _artifact_state(
            archive,
            kind="directory",
            expected_after="--prepare",
            required_now=prepare,
        ),
        "log_dir": _artifact_state(logs, kind="directory", expected_after="--prepare", required_now=prepare),
        "run_log_dir": _artifact_state(
            run_log_dir,
            kind="directory",
            expected_after="--prepare or any event-producing dev-loop action",
            required_now=prepare,
        ),
        "browser_dir": _artifact_state(
            browser_artifact_dir,
            kind="directory",
            expected_after="--prepare, --browser-plan, or any browser smoke/proof",
            required_now=prepare,
        ),
        "terminal_dir": _artifact_state(
            terminal_artifact_dir,
            kind="directory",
            expected_after="--prepare or --capture-cli",
            required_now=prepare,
        ),
        "tui_dir": _artifact_state(
            tui_artifact_dir,
            kind="directory",
            expected_after="--prepare or --tui-plan",
            required_now=prepare,
        ),
        "preflight_json": _artifact_state(
            preflight_json,
            kind="file",
            expected_after="--prepare",
            required_now=prepare,
        ),
        "dev_events": _artifact_state(
            dev_events,
            kind="jsonl",
            expected_after="--launch-daemon, --capture-cli, --browser-plan, --tui-plan, or a smoke/proof action",
        ),
        "daemon_log": _artifact_state(daemon_log, kind="log", expected_after="--launch-daemon"),
    }


def _dev_loop_action(
    *,
    name: str,
    purpose: str,
    command: list[str],
    artifact_dir: Path,
    authority_name: str,
    writes: list[Path] | None = None,
) -> dict[str, object]:
    authority = _authority(authority_name)
    return {
        "name": name,
        "purpose": purpose,
        "command": command,
        "command_text": shlex.join(command),
        "artifact_dir": str(artifact_dir),
        "writes": [str(path) for path in writes or []],
        "authority": authority,
    }


def _build_operator_plan(
    *,
    root: Path,
    archive: Path,
    logs: Path,
    run_id: str,
    run_log_dir: Path,
    daemon_log: Path,
    browser_artifact_dir: Path,
    terminal_artifact_dir: Path,
    tui_artifact_dir: Path,
    preflight_json: Path,
    dev_events: Path,
    api_port: int,
    browser_capture_port: int,
    prepared: bool,
) -> dict[str, object]:
    browser_plan_json = browser_artifact_dir / "browser-plan.json"
    tui_plan_json = tui_artifact_dir / "tui-plan.json"
    live_profile_dir = root / ".local" / "browser-profiles" / f"{run_id}-chrome-user-data"
    prepare_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--prepare"],
    )
    launch_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--launch-daemon", "--json"],
    )
    receiver_smoke_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--receiver-smoke", "--json"],
    )
    extension_smoke_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--extension-smoke", "--json"],
    )
    browser_smoke_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--browser-smoke", "--json"],
    )
    browser_provider_smoke_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--browser-provider-smoke", "--json"],
    )
    browser_provider_live_follow_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--browser-provider-live-follow", "--json"],
    )
    browser_plan_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--browser-plan", "--json"],
    )
    live_proof_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=[
            "--browser-live-proof",
            "--browser-live-profile-dir",
            str(live_profile_dir),
            "--browser-live-chatgpt-url",
            "https://chatgpt.com/c/<conversation-id>",
            "--browser-live-claude-url",
            "https://claude.ai/chat/<conversation-id>",
            "--json",
        ],
    )
    tui_plan_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--tui-plan", "--json"],
    )
    inspect_command = ["devtools", "workspace", "dev-loop", "--inspect-run", str(run_log_dir), "--json"]
    capture_cli_command = _dev_loop_cli_args(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive=archive,
        logs=logs,
        extra_args=["--capture-cli", "--", "polylogue", "ops", "status"],
    )

    actions = [
        _dev_loop_action(
            name="prepare",
            purpose="create the branch-local archive/run artifact directories and write preflight.json",
            command=prepare_command,
            artifact_dir=run_log_dir,
            authority_name="preflight",
            writes=[preflight_json, browser_artifact_dir, terminal_artifact_dir, tui_artifact_dir],
        ),
        _dev_loop_action(
            name="launch_daemon",
            purpose="start the branch-local daemon from source with selected ports/archive/logs",
            command=launch_command,
            artifact_dir=run_log_dir,
            authority_name="launch_daemon",
            writes=[daemon_log, run_log_dir / "polylogued.launch.json", run_log_dir / "polylogued.pid", dev_events],
        ),
        _dev_loop_action(
            name="receiver_smoke",
            purpose="prove receiver auth rejection and synthetic capture spool acceptance without browser/profile state",
            command=receiver_smoke_command,
            artifact_dir=run_log_dir / "receiver-smoke-spool",
            authority_name="receiver_smoke",
        ),
        _dev_loop_action(
            name="extension_smoke",
            purpose="exercise the extension background-worker receiver path with deterministic synthetic data",
            command=extension_smoke_command,
            artifact_dir=browser_artifact_dir,
            authority_name="extension_smoke",
            writes=[browser_artifact_dir / "extension-smoke.json", dev_events],
        ),
        _dev_loop_action(
            name="browser_smoke",
            purpose="load the unpacked extension in local headless Chrome/Chromium against a temporary receiver",
            command=browser_smoke_command,
            artifact_dir=browser_artifact_dir,
            authority_name="browser_smoke",
            writes=[browser_artifact_dir / "browser-smoke.json", dev_events],
        ),
        _dev_loop_action(
            name="browser_provider_smoke",
            purpose="verify deterministic provider fixture content-script capture in local headless Chrome/Chromium",
            command=browser_provider_smoke_command,
            artifact_dir=browser_artifact_dir,
            authority_name="browser_provider_smoke",
            writes=[browser_artifact_dir / "browser-provider-smoke.json", dev_events],
        ),
        _dev_loop_action(
            name="browser_provider_live_follow",
            purpose="launch branch daemon, capture deterministic provider fixture pages, and prove archive/API live-follow convergence",
            command=browser_provider_live_follow_command,
            artifact_dir=browser_artifact_dir,
            authority_name="browser_provider_live_follow",
            writes=[
                run_log_dir / "polylogued.launch.json",
                browser_artifact_dir / "browser-provider-live-follow.json",
                dev_events,
            ],
        ),
        _dev_loop_action(
            name="browser_plan",
            purpose="write local browser-control handoff commands, profile dirs, and copied-profile checklist",
            command=browser_plan_command,
            artifact_dir=browser_artifact_dir,
            authority_name="browser_plan",
            writes=[browser_plan_json, browser_artifact_dir / "browser-plan.md", dev_events],
        ),
        _dev_loop_action(
            name="browser_live_proof",
            purpose="operator-local visible copied-profile proof against live provider pages; never source-only/cloud evidence",
            command=live_proof_command,
            artifact_dir=browser_artifact_dir,
            authority_name="browser_live_proof",
            writes=[browser_artifact_dir / "browser-live-proof.json", dev_events],
        ),
        _dev_loop_action(
            name="capture_cli_status",
            purpose="record a branch-local CLI transcript with stdout/stderr/env/summary artifacts",
            command=capture_cli_command,
            artifact_dir=terminal_artifact_dir,
            authority_name="capture_cli",
            writes=[terminal_artifact_dir],
        ),
        _dev_loop_action(
            name="tui_plan",
            purpose="write local terminal/TUI recording commands and artifact paths",
            command=tui_plan_command,
            artifact_dir=tui_artifact_dir,
            authority_name="tui_plan",
            writes=[tui_plan_json, tui_artifact_dir / "tui-plan.md", dev_events],
        ),
        _dev_loop_action(
            name="inspect_run",
            purpose="summarize the run-local artifacts after any daemon/CLI/browser action",
            command=inspect_command,
            artifact_dir=run_log_dir,
            authority_name="inspect_run",
        ),
    ]
    checks = {
        str(action["name"]): action
        for action in actions
        if action["name"]
        in {
            "receiver_smoke",
            "extension_smoke",
            "browser_smoke",
            "browser_provider_smoke",
            "browser_provider_live_follow",
            "browser_live_proof",
        }
    }
    next_action = actions[1] if prepared else actions[0]
    return {
        "run_id": run_id,
        "artifact_dir": str(run_log_dir),
        "next_command_name": next_action["name"],
        "next_command": next_action["command_text"],
        "next_artifact_dir": next_action["artifact_dir"],
        "ports": {"api": api_port, "browser_capture": browser_capture_port},
        "urls": {
            "api_url": f"http://127.0.0.1:{api_port}",
            "web_url": f"http://127.0.0.1:{api_port}/",
            "receiver_url": f"http://127.0.0.1:{browser_capture_port}",
        },
        "actions": actions,
        "checks": checks,
        "authority_levels": {name: _authority(name) for name in sorted(_DEV_LOOP_AUTHORITIES)},
    }


def build_dev_loop_status(
    *,
    repo_root: Path | None = None,
    api_port: int = DEFAULT_API_PORT,
    browser_capture_port: int = DEFAULT_BROWSER_CAPTURE_PORT,
    archive_root: Path | None = None,
    log_dir: Path | None = None,
    prepare: bool = False,
    port_selection: str = "explicit",
) -> dict[str, Any]:
    root = repo_root or _repo_root()
    archive = archive_root or root / ".local" / "dev-archive"
    logs = log_dir or root / ".cache" / "dev-loop"

    service = system_service_status()
    api = port_status(api_port)
    receiver = port_status(browser_capture_port)
    archive_status = _archive_status(archive)
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
    suggested_env = _dev_loop_suggested_env(
        archive=archive,
        run_log_dir=run_log_dir,
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        run_id=run_id,
    )
    if prepare:
        archive.mkdir(parents=True, exist_ok=True)
        browser_artifact_dir.mkdir(parents=True, exist_ok=True)
        terminal_artifact_dir.mkdir(parents=True, exist_ok=True)
        tui_artifact_dir.mkdir(parents=True, exist_ok=True)
        Path(suggested_env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(suggested_env["XDG_DATA_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(suggested_env["XDG_STATE_HOME"]).mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if service.get("active") and port_selection != "isolated":
        warnings.append(
            "systemwide polylogued.service is active; stop it or use isolated ports before branch-local runs"
        )
    for name, status in (("api", api), ("browser_capture", receiver)):
        if int(status.get("owner_count") or 0) > 0:
            warnings.append(f"{name} port {status['port']} already has a listener")
    if not archive_status["schema_ready"]:
        warnings.append(
            "branch-local archive is not initialized; run the prepared launch/import path before using its counts"
        )
    env_archive = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    if env_archive and Path(env_archive).expanduser() != archive.expanduser():
        warnings.append(f"current POLYLOGUE_ARCHIVE_ROOT points at {env_archive}; dev-loop commands will use {archive}")
    env_daemon_url = os.environ.get("POLYLOGUE_DAEMON_URL")
    if env_daemon_url and env_daemon_url != suggested_env["POLYLOGUE_DAEMON_URL"]:
        warnings.append(
            "current POLYLOGUE_DAEMON_URL points at "
            f"{env_daemon_url}; dev-loop commands will use {suggested_env['POLYLOGUE_DAEMON_URL']}"
        )

    api_url = f"http://127.0.0.1:{api_port}"
    web_url = f"{api_url}/"
    receiver_url = f"http://127.0.0.1:{browser_capture_port}"
    artifact_status = _dev_loop_artifact_status(
        archive=archive,
        logs=logs,
        run_log_dir=run_log_dir,
        daemon_log=daemon_log,
        browser_artifact_dir=browser_artifact_dir,
        terminal_artifact_dir=terminal_artifact_dir,
        tui_artifact_dir=tui_artifact_dir,
        preflight_json=preflight_json,
        dev_events=dev_events,
        prepare=prepare,
    )
    if prepare:
        artifact_status["preflight_json"]["exists"] = True
        artifact_status["preflight_json"]["state"] = "present"
    operator_plan = _build_operator_plan(
        root=root,
        archive=archive,
        logs=logs,
        run_id=run_id,
        run_log_dir=run_log_dir,
        daemon_log=daemon_log,
        browser_artifact_dir=browser_artifact_dir,
        terminal_artifact_dir=terminal_artifact_dir,
        tui_artifact_dir=tui_artifact_dir,
        preflight_json=preflight_json,
        dev_events=dev_events,
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        prepared=prepare,
    )
    plan_actions = {str(action["name"]): action for action in cast(list[dict[str, object]], operator_plan["actions"])}
    plan_checks = cast(dict[str, dict[str, object]], operator_plan["checks"])
    shell_env_prefix = f"{_shell_env_prefix(suggested_env)} PYTHONPATH={root}${{PYTHONPATH:+:${{PYTHONPATH}}}}"
    commands = {
        "stop_system_service": "systemctl --user stop polylogued.service",
        "prepare": str(plan_actions["prepare"]["command_text"]),
        "prepare_isolated": "devtools workspace dev-loop --isolated-ports --prepare",
        "save_preflight": (
            f"mkdir -p {run_log_dir} && devtools workspace dev-loop --api-port {api_port} "
            f"--browser-capture-port {browser_capture_port} --archive-root {archive} "
            f"--log-dir {logs} --json > {preflight_json}"
        ),
        "launch_daemon": str(plan_actions["launch_daemon"]["command_text"]),
        "receiver_smoke": str(plan_checks["receiver_smoke"]["command_text"]),
        "extension_smoke": str(plan_checks["extension_smoke"]["command_text"]),
        "browser_smoke": str(plan_checks["browser_smoke"]["command_text"]),
        "browser_provider_smoke": str(plan_checks["browser_provider_smoke"]["command_text"]),
        "browser_provider_live_follow": str(plan_checks["browser_provider_live_follow"]["command_text"]),
        "browser_plan": str(plan_actions["browser_plan"]["command_text"]),
        "browser_live_proof": str(plan_checks["browser_live_proof"]["command_text"]),
        "tui_plan": str(plan_actions["tui_plan"]["command_text"]),
        "inspect_run": str(plan_actions["inspect_run"]["command_text"]),
        "run_daemon": (
            f"{shell_env_prefix} "
            f"{shlex.quote(sys.executable)} -c 'from polylogue.daemon.cli import main; main()' "
            f"run --api-port {api_port} --port {browser_capture_port} 2>&1 | tee {daemon_log}"
        ),
        "open_web_shell": web_url,
        "receiver_status": f"curl -sf {receiver_url}/v1/status",
        "capture_cli_status": (
            "script -q -c "
            f"'{shell_env_prefix} polylogue ops status' "
            f"{terminal_artifact_dir / 'polylogue-ops-status.typescript'}"
        ),
        "terminal_tui_plan": str(plan_actions["tui_plan"]["command_text"]),
    }

    payload = {
        "repo_root": str(root),
        "branch": branch,
        "commit": commit,
        "run_id": run_id,
        "port_selection": port_selection,
        "prepared": prepare,
        "preflight_json_written": False,
        "dev_archive_root": str(archive),
        "log_dir": str(logs),
        "run_log_dir": str(run_log_dir),
        "artifact_dir": str(run_log_dir),
        "api_url": api_url,
        "web_url": web_url,
        "receiver_url": receiver_url,
        "artifacts": {
            "daemon_log": str(daemon_log),
            "browser_dir": str(browser_artifact_dir),
            "terminal_dir": str(terminal_artifact_dir),
            "tui_dir": str(tui_artifact_dir),
            "preflight_json": str(preflight_json),
            "dev_events": str(dev_events),
        },
        "system_service": service,
        "archive_status": archive_status,
        "ports": {
            "api": api,
            "browser_capture": receiver,
        },
        "suggested_env": suggested_env,
        "commands": commands,
        "artifact_status": artifact_status,
        "plan": operator_plan,
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
    archive_status = payload.get("archive_status")
    if isinstance(archive_status, dict):
        ready = archive_status.get("schema_ready")
        sessions = archive_status.get("session_count")
        messages = archive_status.get("message_count")
        detail = f"ready={ready}"
        if sessions is not None:
            detail += f" sessions={sessions}"
        if messages is not None:
            detail += f" messages={messages}"
        error = archive_status.get("error")
        if error:
            detail += f" error={error}"
        print(f"  archive status: {detail}")
    print(f"  logs:    {payload['log_dir']}")
    print(f"  run log: {payload['run_log_dir']}")
    print(f"  artifacts: {payload.get('artifact_dir', payload['run_log_dir'])}")
    if payload.get("api_url"):
        print(f"  api:     {payload['api_url']}")
    if payload.get("web_url"):
        print(f"  web:     {payload['web_url']}")
    if payload.get("receiver_url"):
        print(f"  receiver:{payload['receiver_url']}")
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
    plan = payload.get("plan")
    if isinstance(plan, dict):
        print("\nPlan:")
        print(f"  next:      {plan.get('next_command_name')}")
        print(f"  command:   {plan.get('next_command')}")
        print(f"  artifacts: {plan.get('next_artifact_dir')}")
        checks = plan.get("checks")
        if isinstance(checks, dict):
            print("  checks:")
            for name in (
                "receiver_smoke",
                "extension_smoke",
                "browser_smoke",
                "browser_provider_smoke",
                "browser_provider_live_follow",
                "browser_live_proof",
            ):
                check = checks.get(name)
                if not isinstance(check, dict):
                    continue
                authority = check.get("authority")
                label = authority.get("label") if isinstance(authority, dict) else "unknown"
                cloud_safe = authority.get("cloud_safe") if isinstance(authority, dict) else "unknown"
                print(f"    - {name}: {label} cloud_safe={cloud_safe} artifacts={check.get('artifact_dir')}")
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


def _print_extension_smoke(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    smoke = payload["extension_smoke"]
    assert isinstance(preflight, dict)
    assert isinstance(smoke, dict)
    artifacts = smoke["artifacts"]
    assert isinstance(artifacts, dict)
    print("Polylogue dev-loop browser extension smoke")
    print(f"  run id:   {preflight['run_id']}")
    print(f"  ok:       {smoke['ok']}")
    print(f"  receiver: {smoke['receiver_url']}")
    print(f"  artifact: {smoke.get('artifact_ref')}")
    print(f"  summary:  {artifacts['summary']}")


def _print_browser_smoke(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    smoke = payload["browser_smoke"]
    assert isinstance(preflight, dict)
    assert isinstance(smoke, dict)
    artifacts = smoke["artifacts"]
    assert isinstance(artifacts, dict)
    print("Polylogue dev-loop real browser smoke")
    print(f"  run id:    {preflight['run_id']}")
    print(f"  ok:        {smoke['ok']}")
    print(f"  receiver:  {smoke['receiver_url']}")
    print(f"  extension: {smoke.get('extension_id')}")
    print(f"  artifact:  {smoke.get('artifact_ref')}")
    print(f"  profile:   {artifacts['profile']}")
    print(f"  summary:   {artifacts['summary']}")


def _print_browser_provider_smoke(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    smoke = payload["browser_provider_smoke"]
    assert isinstance(preflight, dict)
    assert isinstance(smoke, dict)
    artifacts = smoke["artifacts"]
    assert isinstance(artifacts, dict)
    provider_statuses = smoke.get("provider_statuses")
    if not isinstance(provider_statuses, dict):
        provider_statuses = {}
    providers = ", ".join(f"{name}={status}" for name, status in sorted(provider_statuses.items())) or "none"
    print("Polylogue dev-loop provider page smoke")
    print(f"  run id:    {preflight['run_id']}")
    print(f"  ok:        {smoke['ok']}")
    print(f"  receiver:  {smoke['receiver_url']}")
    print(f"  extension: {smoke.get('extension_id')}")
    print(f"  providers: {providers}")
    print(f"  profile:   {artifacts['profile']}")
    print(f"  summary:   {artifacts['summary']}")


def _print_browser_provider_live_follow(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    proof = payload["browser_provider_live_follow"]
    assert isinstance(preflight, dict)
    assert isinstance(proof, dict)
    artifacts = proof["artifacts"]
    assert isinstance(artifacts, dict)
    provider_statuses = proof.get("provider_statuses")
    if not isinstance(provider_statuses, dict):
        provider_statuses = {}
    providers = ", ".join(f"{name}={status}" for name, status in sorted(provider_statuses.items())) or "none"
    print("Polylogue dev-loop provider live-follow proof")
    print(f"  run id:      {preflight['run_id']}")
    print(f"  ok:          {proof['ok']}")
    print(f"  providers:   {providers}")
    print(f"  archive ok:  {proof.get('archive_ok')}")
    print(f"  api ok:      {proof.get('api_ok')}")
    print(f"  reader ok:   {proof.get('reader_ok')}")
    print(f"  session id:  {proof.get('session_id')}")
    print(f"  summary:     {artifacts['summary']}")


def _print_browser_live_proof(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    proof = payload["browser_live_proof"]
    assert isinstance(preflight, dict)
    assert isinstance(proof, dict)
    artifacts = proof["artifacts"]
    assert isinstance(artifacts, dict)
    provider_statuses = proof.get("provider_statuses")
    if not isinstance(provider_statuses, dict):
        provider_statuses = {}
    providers = ", ".join(f"{name}={status}" for name, status in sorted(provider_statuses.items())) or "none"
    print("Polylogue dev-loop live provider proof")
    print(f"  run id:    {preflight['run_id']}")
    print(f"  ok:        {proof['ok']}")
    print(f"  receiver:  {proof['receiver_url']}")
    print(f"  extension: {proof.get('extension_id')}")
    print(f"  providers: {providers}")
    print(f"  profile:   {artifacts['profile']}")
    print(f"  summary:   {artifacts['summary']}")


def _print_browser_plan(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    plan = payload["browser_plan"]
    assert isinstance(preflight, dict)
    assert isinstance(plan, dict)
    artifacts = plan["artifacts"]
    assert isinstance(artifacts, dict)
    commands = plan["commands"]
    assert isinstance(commands, dict)
    chrome = commands["preferred"]
    assert isinstance(chrome, list)
    print("Polylogue dev-loop browser plan")
    print(f"  run id:    {preflight['run_id']}")
    print(f"  receiver:  {plan['receiver_url']}")
    print(f"  web shell: {plan['web_shell_url']}")
    print(f"  profile:   {plan['profile_dir']}")
    print(f"  extension: {plan['extension_root']}")
    print(f"  command:   {shlex.join([str(part) for part in chrome])}")
    print(f"  plan:      {artifacts['markdown']}")


def _print_tui_plan(payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    plan = payload["tui_plan"]
    assert isinstance(preflight, dict)
    assert isinstance(plan, dict)
    artifacts = plan["artifacts"]
    assert isinstance(artifacts, dict)
    commands = plan["commands"]
    assert isinstance(commands, dict)
    print("Polylogue dev-loop TUI plan")
    print(f"  run id:     {preflight['run_id']}")
    print(f"  script:     {commands['script_status']}")
    print(f"  vhs:        {commands['vhs_render']}")
    print(f"  plan:       {artifacts['markdown']}")
    print(f"  artifacts:  {artifacts['screenshots']}")


def _print_run_summary(payload: dict[str, object]) -> None:
    print("Polylogue dev-loop run summary")
    print(f"  run id:     {payload.get('run_id')}")
    print(f"  run dir:    {payload.get('run_dir')}")
    print(f"  ok:         {payload.get('ok')}")
    print(f"  events:     {payload.get('event_count')}")
    print(f"  terminals:  {payload.get('terminal_capture_count')}")
    slowest = payload.get("slowest_events")
    if isinstance(slowest, list) and slowest:
        first = slowest[0]
        if isinstance(first, dict):
            print(f"  slowest:    {first.get('surface')}/{first.get('event_type')} {first.get('duration_ms')} ms")
    warnings = payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    last_event = payload.get("last_event")
    if isinstance(last_event, dict):
        print(
            "  last event: "
            f"{last_event.get('surface')}/{last_event.get('event_type')} status={last_event.get('status')}"
        )
    artifacts = payload.get("artifact_index")
    if isinstance(artifacts, dict) and artifacts:
        print("  artifacts:")
        for name in sorted(artifacts):
            print(f"    - {name}")


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
    parser.add_argument(
        "--isolated-ports",
        action="store_true",
        help="Pick currently-free loopback API and browser-capture ports for this dev-loop run.",
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
        help="Launch branch-local polylogued for browser-capture/API proof and write PID/log/summary artifacts.",
    )
    parser.add_argument(
        "--full-source-catchup",
        action="store_true",
        help="With --launch-daemon, watch configured sources instead of constraining the proof to browser-capture spool.",
    )
    parser.add_argument(
        "--extension-smoke",
        action="store_true",
        help="Run the browser-extension background worker against a temporary local receiver.",
    )
    parser.add_argument(
        "--browser-smoke",
        action="store_true",
        help="Launch real Chrome headless with the unpacked extension against a temporary receiver.",
    )
    parser.add_argument(
        "--browser-provider-smoke",
        action="store_true",
        help="Launch real Chrome headless against deterministic ChatGPT/Claude fixture pages and verify content-script capture.",
    )
    parser.add_argument(
        "--browser-provider-live-follow",
        action="store_true",
        help="Launch branch-local daemon, capture deterministic provider fixture pages, and verify archive/API live-follow.",
    )
    parser.add_argument(
        "--browser-plan",
        action="store_true",
        help="Write branch-local browser profile, extension load, receiver, and inspection plan artifacts.",
    )
    parser.add_argument(
        "--browser-live-proof",
        action="store_true",
        help="Run a visible operator-local copied-profile proof against live ChatGPT/Claude pages.",
    )
    parser.add_argument(
        "--browser-live-profile-dir",
        type=Path,
        help="Ignored local copied Chrome/Chromium user-data-dir for --browser-live-proof.",
    )
    parser.add_argument(
        "--browser-live-providers",
        default="chatgpt,claude",
        help="Comma-separated live providers for --browser-live-proof: chatgpt,claude.",
    )
    parser.add_argument(
        "--browser-live-chatgpt-url",
        help="Live ChatGPT conversation URL to open during --browser-live-proof.",
    )
    parser.add_argument(
        "--browser-live-claude-url",
        help="Live Claude conversation URL to open during --browser-live-proof.",
    )
    parser.add_argument(
        "--browser-live-wait-s",
        type=float,
        default=45.0,
        help="Visible-browser wait before capturing live provider pages.",
    )
    parser.add_argument(
        "--tui-plan",
        action="store_true",
        help="Write branch-local terminal/TUI visual-inspection plan artifacts.",
    )
    parser.add_argument(
        "--inspect-run",
        nargs="?",
        const=Path("."),
        type=Path,
        help="Summarize an existing dev-loop run directory instead of building a new preflight.",
    )
    parser.add_argument(
        "--daemon-ready-timeout-s",
        type=float,
        default=10.0,
        help="Readiness wait for --launch-daemon.",
    )
    parser.add_argument("capture_command", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inspect_run is not None:
        summary = summarize_dev_loop_run(args.inspect_run)
        if args.json:
            json.dump(summary, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_run_summary(summary)
        return 0 if summary["ok"] is True else 1
    capture_command = list(args.capture_command)
    if args.capture_cli and not args.json and "--" not in original_argv and capture_command[-1:] == ["--json"]:
        args.json = True
        capture_command = capture_command[:-1]
    api_port = args.api_port
    browser_capture_port = args.browser_capture_port
    port_selection = "explicit"
    if args.isolated_ports:
        api_port, browser_capture_port = allocate_isolated_ports()
        port_selection = "isolated"
    payload = build_dev_loop_status(
        api_port=api_port,
        browser_capture_port=browser_capture_port,
        archive_root=args.archive_root,
        log_dir=args.log_dir,
        prepare=args.prepare
        or args.receiver_smoke
        or args.launch_daemon
        or args.extension_smoke
        or args.browser_smoke
        or args.browser_provider_smoke
        or args.browser_provider_live_follow
        or args.browser_live_proof
        or args.browser_plan
        or args.tui_plan,
        port_selection=port_selection,
    )
    if args.capture_cli:
        try:
            capture_payload = run_cli_capture(
                preflight=payload,
                command=capture_command,
                timeout_s=args.capture_timeout_s,
            )
        except ValueError as exc:
            parser.error(str(exc))
        cli_payload: dict[str, Any] = {
            "preflight": payload,
            "cli_capture": capture_payload,
        }
        if args.json:
            json.dump(cli_payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_cli_capture(cli_payload)
        capture_exit = capture_payload["exit_code"]
        if not isinstance(capture_exit, int):
            capture_exit = 1
        return 0 if capture_payload.get("ok") is True else capture_exit
    if args.launch_daemon:
        try:
            launch_payload = launch_branch_daemon(
                preflight=payload,
                readiness_timeout_s=args.daemon_ready_timeout_s,
                full_source_catchup=args.full_source_catchup,
            )
        except ValueError as exc:
            parser.error(str(exc))
        daemon_payload = {
            "preflight": payload,
            "daemon_launch": launch_payload,
        }
        if args.json:
            json.dump(daemon_payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            _print_daemon_launch(daemon_payload)
        return 0 if launch_payload.get("ok") is True else 1

    combined_payload: dict[str, Any] = {"preflight": payload}
    combined_ok = True
    if args.receiver_smoke:
        receiver_smoke = run_receiver_smoke(spool_path=Path(str(payload["run_log_dir"])) / "receiver-smoke-spool")
        combined_payload["receiver_smoke"] = receiver_smoke
        combined_ok = combined_ok and receiver_smoke.get("ok") is True
    if args.extension_smoke:
        smoke_payload = run_extension_smoke(preflight=payload)
        combined_payload["extension_smoke"] = smoke_payload
        combined_ok = combined_ok and smoke_payload.get("ok") is True
    if args.browser_smoke:
        smoke_payload = run_browser_smoke(preflight=payload)
        combined_payload["browser_smoke"] = smoke_payload
        combined_ok = combined_ok and smoke_payload.get("ok") is True
    if args.browser_provider_smoke:
        provider_smoke_payload = run_browser_provider_smoke(preflight=payload)
        combined_payload["browser_provider_smoke"] = provider_smoke_payload
        combined_ok = combined_ok and provider_smoke_payload.get("ok") is True
    if args.browser_provider_live_follow:
        try:
            live_follow_payload = run_browser_provider_live_follow(
                preflight=payload,
                readiness_timeout_s=args.daemon_ready_timeout_s,
                archive_timeout_s=30.0,
                full_source_catchup=args.full_source_catchup,
            )
        except ValueError as exc:
            parser.error(str(exc))
        combined_payload["browser_provider_live_follow"] = live_follow_payload
        combined_ok = combined_ok and live_follow_payload.get("ok") is True
    if args.browser_live_proof:
        if args.browser_live_profile_dir is None:
            parser.error(
                "--browser-live-proof requires --browser-live-profile-dir pointing at an ignored local copied profile"
            )
        try:
            live_providers = _parse_provider_csv(str(args.browser_live_providers))
            live_proof_payload = run_browser_live_proof(
                preflight=payload,
                profile_dir=args.browser_live_profile_dir,
                providers=live_providers,
                chatgpt_url=args.browser_live_chatgpt_url,
                claude_url=args.browser_live_claude_url,
                wait_s=args.browser_live_wait_s,
            )
        except ValueError as exc:
            parser.error(str(exc))
        combined_payload["browser_live_proof"] = live_proof_payload
        combined_ok = combined_ok and live_proof_payload.get("ok") is True
    if args.browser_plan:
        try:
            browser_plan = build_browser_plan(preflight=payload)
        except ValueError as exc:
            parser.error(str(exc))
        combined_payload["browser_plan"] = browser_plan
        combined_ok = combined_ok and browser_plan.get("ok") is True
    if args.tui_plan:
        try:
            tui_plan = build_tui_plan(preflight=payload)
        except ValueError as exc:
            parser.error(str(exc))
        combined_payload["tui_plan"] = tui_plan
        combined_ok = combined_ok and tui_plan.get("ok") is True
    if len(combined_payload) > 1:
        if args.json:
            json.dump(combined_payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            if "receiver_smoke" in combined_payload:
                _print_receiver_smoke(combined_payload)
            if "extension_smoke" in combined_payload:
                _print_extension_smoke(combined_payload)
            if "browser_smoke" in combined_payload:
                _print_browser_smoke(combined_payload)
            if "browser_provider_smoke" in combined_payload:
                _print_browser_provider_smoke(combined_payload)
            if "browser_provider_live_follow" in combined_payload:
                _print_browser_provider_live_follow(combined_payload)
            if "browser_live_proof" in combined_payload:
                _print_browser_live_proof(combined_payload)
            if "browser_plan" in combined_payload:
                _print_browser_plan(combined_payload)
            if "tui_plan" in combined_payload:
                _print_tui_plan(combined_payload)
        return 0 if combined_ok else 1
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
