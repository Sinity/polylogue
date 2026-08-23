"""Fixed AgentCTL-owned service proof for Polylogue browser capture.

AgentCTL injects the descriptor-declared ports and owns the enclosing systemd
service, deadline, cancellation, lease, and result artifact. This module owns
only Polylogue semantics inside that fixed execution boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import quote, urlencode

from devtools.sinnixd_service_context import require_declared_service_context, terminate_process_group
from polylogue.browser_capture.server import make_server
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_API_PORT_ENV = "POLYLOGUE_API_PORT"
_BROWSER_CAPTURE_PORT_ENV = "POLYLOGUE_BROWSER_CAPTURE_PORT"
_DECLARED_PORTS = {
    _API_PORT_ENV: (48800, 48863),
    _BROWSER_CAPTURE_PORT_ENV: (48864, 48927),
}
_MAX_ERROR_MESSAGE = 512
_RECEIVER_ORIGIN = "chrome-extension://polylogue-agentctl-proof"
_RECEIVER_TOKEN = "polylogue-agentctl-proof-token"
_SHARED_CHROME_TIMEOUT_S = 30


def _leased_port(name: str, bounds: tuple[int, int]) -> int:
    """Read one expected service-context port, never a fallback."""
    raw = os.environ.get(name)
    if raw is None:
        raise ValueError(f"{name} must be present for the fixed dev-loop service context")
    try:
        port = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer lease port") from error
    if not bounds[0] <= port <= bounds[1]:
        raise ValueError(f"{name} is outside the fixed dev-loop service port range")
    return port


def _require_agentctl_service_context() -> dict[str, int]:
    """Reject accidental shell execution outside the expected service context.

    These checkout-local environment checks are deliberately not authorization
    or admission. Sinnixd validates the registered workspace, exact head,
    declared operation, lease, and service cgroup before it invokes this
    module. This guard only fails closed for ordinary accidental invocation.
    """
    require_declared_service_context("dev_loop_proof")
    ports = {name: _leased_port(name, bounds) for name, bounds in _DECLARED_PORTS.items()}
    if len(set(ports.values())) != len(ports):
        raise ValueError("fixed dev-loop service context contains duplicate ports")
    return ports


def _service_paths() -> tuple[Path, Path]:
    """Place disposable proof state under AgentCTL's per-job scratch root."""
    scratch = os.environ.get("TMPDIR")
    if not scratch:
        raise ValueError("TMPDIR must be injected by AgentCTL for the declared nvme scratch contract")
    root = Path(scratch).resolve() / "polylogue-dev-loop-proof"
    return root / "archive", root / "artifacts"


def _proof_environment(*, archive_root: Path, artifact_root: Path, api_port: int, capture_port: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
            "POLYLOGUE_API_PORT": str(api_port),
            "POLYLOGUE_BROWSER_CAPTURE_PORT": str(capture_port),
            "POLYLOGUE_DAEMON_URL": f"http://127.0.0.1:{api_port}",
            "XDG_CACHE_HOME": str(artifact_root / "xdg-cache"),
            "XDG_DATA_HOME": str(artifact_root / "xdg-data"),
            "XDG_STATE_HOME": str(artifact_root / "xdg-state"),
        }
    )
    return environment


def _http_get_json(url: str, *, timeout_s: float = 5.0) -> tuple[int, dict[str, object]]:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    connection = HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=timeout_s)
    try:
        connection.request("GET", parts.path + (f"?{parts.query}" if parts.query else ""))
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body if isinstance(body, dict) else {"body": body}
    finally:
        connection.close()


def _receiver_payload(*, provider: str = "chatgpt", session_id: str = "polylogue-agentctl-proof") -> dict[str, object]:
    source_url = {
        "chatgpt": f"https://chatgpt.com/c/{session_id}",
        "claude-ai": f"https://claude.ai/chat/{session_id}",
    }.get(provider)
    if source_url is None:
        raise ValueError(f"unsupported deterministic provider: {provider}")
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": source_url,
            "page_title": "Polylogue AgentCTL proof",
            "captured_at": "2026-08-23T00:00:00+00:00",
            "adapter_name": "agentctl-proof",
            "extension_instance_id": "agentctl-proof-instance",
        },
        "session": {
            "provider": provider,
            "provider_session_id": session_id,
            "title": "Polylogue AgentCTL proof",
            "turns": [{"provider_turn_id": "turn-1", "role": "user", "text": "proof"}],
        },
    }


def _receiver_post(*, port: int, body: object, token: str | None) -> tuple[int, dict[str, object]]:
    headers = {"Content-Type": "application/json", "Origin": _RECEIVER_ORIGIN}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("POST", "/v1/browser-captures", body=json.dumps(body), headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload if isinstance(payload, dict) else {"body": payload}
    finally:
        connection.close()


def run_receiver_smoke(*, spool_path: Path) -> dict[str, object]:
    """Keep the deterministic, in-process receiver-auth smoke product-owned."""
    spool_path.mkdir(parents=True, exist_ok=True)
    server = make_server(
        "127.0.0.1",
        0,
        spool_path=spool_path,
        auth_token=_RECEIVER_TOKEN,
        extra_origins=(_RECEIVER_ORIGIN,),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _host, port = server.server_address[:2]
        rejected_status, _rejected = _receiver_post(port=port, body=_receiver_payload(), token=None)
        accepted_status, accepted = _receiver_post(port=port, body=_receiver_payload(), token=_RECEIVER_TOKEN)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    artifact_ref = accepted.get("artifact_ref")
    return {
        "ok": rejected_status == 401 and accepted_status == 202 and isinstance(artifact_ref, str),
        "rejected_status": rejected_status,
        "accepted_status": accepted_status,
        "artifact_ref": artifact_ref,
    }


def _await_api(*, base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "API did not answer"
    while time.monotonic() <= deadline:
        try:
            status, payload = _http_get_json(f"{base_url}/healthz/live", timeout_s=2.0)
        except OSError as error:
            last_error = f"{type(error).__name__}: {error}"
        else:
            if status == 200 and payload.get("status") == "alive":
                return
            last_error = f"HTTP {status}"
        time.sleep(0.1)
    raise RuntimeError(f"Polylogue API convergence did not complete: {last_error}")


def _start_daemon(
    *, repo_root: Path, environment: dict[str, str], artifact_root: Path, api_port: int, capture_port: int
) -> subprocess.Popen[Any]:
    """Start the fixed product daemon as a child of AgentCTL's service cgroup.

    The dedicated child process group is terminated locally on every proof
    exit. Sinnixd retains lifecycle authority and is the outer cleanup net.
    """
    spool = artifact_root / "browser-capture"
    spool.mkdir(parents=True, exist_ok=True)
    log_path = artifact_root / "polylogued.log"
    command = [
        sys.executable,
        "-c",
        "from polylogue.daemon.cli import main; main()",
        "run",
        "--spool",
        str(spool),
        "--api-port",
        str(api_port),
        "--port",
        str(capture_port),
        "--root",
        str(spool),
        "--browser-capture-auth-token",
        _RECEIVER_TOKEN,
        "--no-source-catchup",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def _run_shared_chrome_control(*, repo_root: Path, timeout_s: float = _SHARED_CHROME_TIMEOUT_S) -> None:
    """Exercise the existing Chrome only through Sinnix's owned control boundary."""
    extension_root = repo_root / "browser-extension"
    environment = os.environ.copy()
    environment.update(
        {
            "POLYLOGUE_DEV_LOOP_EXTENSION_ROOT": str(extension_root),
        }
    )
    process = subprocess.Popen(
        ["node", "scripts/dev_loop_shared_chrome_proof.mjs"],
        cwd=str(extension_root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        terminate_process_group(process)
        process.communicate()
        raise RuntimeError("shared-Chrome control proof timed out") from error
    finally:
        terminate_process_group(process)
    completed = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        raise RuntimeError("shared-Chrome control proof did not produce a successful result")
    payload = json.loads(stdout)
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("shared-Chrome control proof reported failure")


def _submit_deterministic_captures(*, capture_port: int, session_id: str) -> dict[str, dict[str, str]]:
    """Keep receiver, archive, and API convergence deterministic and browser-free."""
    captures: dict[str, dict[str, str]] = {}
    for provider in ("chatgpt", "claude-ai"):
        provider_session_id = f"{session_id}-{provider}"
        status, _accepted = _receiver_post(
            port=capture_port,
            body=_receiver_payload(provider=provider, session_id=provider_session_id),
            token=_RECEIVER_TOKEN,
        )
        if status != 202:
            raise RuntimeError(f"deterministic {provider} capture was not accepted")
        captures[provider] = {"provider": provider, "provider_session_id": provider_session_id}
    return captures


def _poll_archive_state(*, receiver_url: str, provider: str, provider_session_id: str, timeout_s: float) -> bool:
    query = urlencode({"provider": provider, "provider_session_id": provider_session_id})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() <= deadline:
        try:
            status, payload = _http_get_json(f"{receiver_url}/v1/archive-state?{query}")
        except OSError:
            status, payload = 0, {}
        if status == 200 and payload.get("raw_row_exists") is True and payload.get("indexed_session_exists") is True:
            return True
        time.sleep(0.25)
    return False


def _session_id_for_provider(provider: str, provider_session_id: str) -> str:
    return (
        {"chatgpt": "chatgpt-export", "claude-ai": "claude-ai-export"}.get(provider, provider)
        + ":"
        + provider_session_id
    )


def _fetch_api_messages(*, api_url: str, session_id: str) -> bool:
    status, payload = _http_get_json(f"{api_url}/api/sessions/{quote(session_id, safe='')}/messages?limit=5")
    return status == 200 and isinstance(payload.get("messages"), list) and bool(payload["messages"])


def run_proof(*, repo_root: Path | None = None, readiness_timeout_s: float = 45.0) -> dict[str, object]:
    """Run the bounded Polylogue semantics behind the AgentCTL service lease."""
    checkout = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    ports = _require_agentctl_service_context()
    api_port = ports[_API_PORT_ENV]
    capture_port = ports[_BROWSER_CAPTURE_PORT_ENV]
    archive_root, artifact_root = _service_paths()
    artifact_root.mkdir(parents=True, exist_ok=True)
    initialize_active_archive_root(archive_root)
    environment = _proof_environment(
        archive_root=archive_root,
        artifact_root=artifact_root,
        api_port=api_port,
        capture_port=capture_port,
    )
    receiver_auth = run_receiver_smoke(spool_path=artifact_root / "receiver-auth")
    if receiver_auth.get("ok") is not True:
        raise RuntimeError("receiver authentication proof failed")
    daemon = _start_daemon(
        repo_root=checkout,
        environment=environment,
        artifact_root=artifact_root,
        api_port=api_port,
        capture_port=capture_port,
    )
    api_url = f"http://127.0.0.1:{api_port}"
    receiver_url = f"http://127.0.0.1:{capture_port}"
    try:
        _await_api(base_url=api_url, timeout_s=readiness_timeout_s)
        session_id = f"polylogue-agentctl-proof-{api_port}-{capture_port}"
        _run_shared_chrome_control(repo_root=checkout)
        providers = _submit_deterministic_captures(capture_port=capture_port, session_id=session_id)
        archive_ok = False
        api_ok = False
        if isinstance(providers, dict):
            archive_rows: list[bool] = []
            api_rows: list[bool] = []
            for item in providers.values():
                if not isinstance(item, dict):
                    continue
                provider = item.get("provider")
                provider_session_id = item.get("provider_session_id")
                if not isinstance(provider, str) or not isinstance(provider_session_id, str):
                    continue
                archive_rows.append(
                    _poll_archive_state(
                        receiver_url=receiver_url,
                        provider=provider,
                        provider_session_id=provider_session_id,
                        timeout_s=readiness_timeout_s,
                    )
                )
                api_rows.append(
                    _fetch_api_messages(
                        api_url=api_url,
                        session_id=_session_id_for_provider(provider, provider_session_id),
                    )
                )
            archive_ok = bool(archive_rows) and all(archive_rows)
            api_ok = bool(api_rows) and all(api_rows)
        if not archive_ok or not api_ok:
            raise RuntimeError("archive/API convergence proof failed")
        return {
            "ok": True,
            "ports": {"api": api_port, "browser_capture": capture_port},
            "receiver_auth": {"ok": True},
            "shared_chrome": {"ok": True},
            "provider_capture": {
                "providers": sorted(str(name) for name in providers),
                "archive_converged": archive_ok,
                "api_converged": api_ok,
            },
        }
    finally:
        terminate_process_group(daemon)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Polylogue's fixed AgentCTL dev-loop proof.")
    parser.add_argument("--json", action="store_true", help="Emit the bounded AgentCTL result object.")
    parser.parse_args(argv)
    try:
        payload: dict[str, Any] = run_proof()
    except Exception as error:
        payload = {
            "ok": False,
            "error": {"type": type(error).__name__, "message": str(error)[:_MAX_ERROR_MESSAGE]},
        }
        exit_code = 1
    else:
        exit_code = 0
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
