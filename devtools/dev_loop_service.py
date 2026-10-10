"""Fixed AgentCTL-owned proof for Polylogue browser capture.

AgentCTL owns the enclosing systemd service, deadline, cancellation, and result
artifact. This module owns Polylogue semantics inside that boundary, including
its own loopback ports: it binds them free and publishes them in the result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import quote, urlencode

from devtools.agentctl_service_context import require_declared_operation_context, terminate_process_group
from devtools.isolated_environment import isolated_home_environment
from devtools.native_transport_proof import scoped_native_transport_proof
from devtools.shared_chrome_lock import shared_chrome_extension_lock
from polylogue.browser_capture.server import make_server
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_MAX_ERROR_MESSAGE = 512
_RECEIVER_ORIGIN = "chrome-extension://polylogue-agentctl-proof"
_RECEIVER_TOKEN = "polylogue-agentctl-proof-token"
_API_TOKEN = "polylogue-agentctl-proof-api-token"
_SHARED_CHROME_TIMEOUT_S = 150
_CHILD_ERROR_TAIL_CHARS = 384
_DETERMINISTIC_PROVIDERS = ("chatgpt", "claude-ai")


def _require_agentctl_operation_context() -> None:
    """Reject accidental shell execution outside the declared operation context.

    These checkout-local environment checks are deliberately not authorization
    or admission. The runtime validates the registered workspace, exact head,
    declared operation, and job cgroup before it invokes this module. This
    guard only fails closed for ordinary accidental invocation.
    """
    require_declared_operation_context("dev_loop_proof")


def _service_paths() -> tuple[Path, Path]:
    """Place disposable proof state under the per-job temporary root."""
    root = Path(tempfile.gettempdir()).resolve() / "polylogue-dev-loop-proof"
    return root / "archive", root / "artifacts"


def _share_source_fingerprint_memo(home: Path) -> None:
    """Let the isolated daemon read the host's source-fingerprint memo.

    The memo holds digests of Polylogue's own source closures, keyed by their
    bytes, and no host data. Without it the empty home recomputes every
    parser closure cold (about 90 s under load) before the daemon can
    converge one capture, which outlasts the proof's readiness wait.
    """
    from polylogue.sources.origin_specs import _source_memo_root

    host_memo = _source_memo_root()
    if host_memo is None:
        return
    isolated = home / ".cache" / "polylogue" / "source-fingerprints"
    isolated.parent.mkdir(parents=True, exist_ok=True)
    if not isolated.is_symlink():
        isolated.symlink_to(host_memo, target_is_directory=True)


def _proof_environment(*, archive_root: Path, artifact_root: Path) -> dict[str, str]:
    """The proof daemon's environment, isolated from the host's sources.

    The daemon watches every origin at its canonical location under ``HOME``,
    so the proof runs in an empty home of its own: inheriting the host's
    ``HOME`` or XDG roots would ingest the operator's real transcripts.
    """
    home = artifact_root / "home"
    home.mkdir(parents=True, exist_ok=True)
    _share_source_fingerprint_memo(home)
    environment = isolated_home_environment(os.environ, home=home)
    environment.pop("POLYLOGUE_DAEMON_URL", None)
    environment.update(
        {
            "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
            "POLYLOGUE_API_PORT": "0",
            "POLYLOGUE_BROWSER_CAPTURE_PORT": "0",
        }
    )
    return environment


def _http_get_json(
    url: str,
    *,
    timeout_s: float = 5.0,
    bearer_token: str | None = None,
) -> tuple[int, dict[str, object]]:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    connection = HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=timeout_s)
    try:
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token is not None else {}
        connection.request("GET", parts.path + (f"?{parts.query}" if parts.query else ""), headers=headers)
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


def _await_api(*, base_url: str, timeout_s: float, daemon: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "API did not answer"
    while time.monotonic() <= deadline:
        _require_daemon_alive(daemon)
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


def _require_daemon_alive(daemon: subprocess.Popen[Any]) -> None:
    exit_code = daemon.poll()
    if exit_code is not None:
        raise RuntimeError(f"proof daemon exited during startup: {exit_code}")


def _await_listener_ports(
    *, daemon: subprocess.Popen[Any], listener_info_path: Path, timeout_s: float
) -> tuple[int, int]:
    """Read only this child's atomically published bound listener identities."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() <= deadline:
        _require_daemon_alive(daemon)
        try:
            payload = json.loads(listener_info_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("pid"), int)
            or isinstance(payload.get("pid"), bool)
            or payload["pid"] != daemon.pid
        ):
            raise RuntimeError("listener readback does not belong to the proof daemon")
        listeners = payload.get("listeners")
        if not isinstance(listeners, dict):
            raise RuntimeError("proof daemon listener readback is malformed")
        ports: list[int] = []
        for name in ("api", "browser_capture"):
            address = listeners.get(name)
            if not isinstance(address, dict) or address.get("host") != "127.0.0.1":
                raise RuntimeError(f"proof daemon {name} listener is not loopback")
            port = address.get("port")
            if not isinstance(port, int) or isinstance(port, bool) or not 0 < port <= 65535:
                raise RuntimeError(f"proof daemon {name} listener port is malformed")
            ports.append(port)
        if ports[0] == ports[1]:
            raise RuntimeError("proof daemon listener ports overlap")
        return ports[0], ports[1]
    raise RuntimeError("proof daemon did not publish bound listeners")


def _start_daemon(
    *, repo_root: Path, environment: dict[str, str], artifact_root: Path, listener_info_path: Path
) -> subprocess.Popen[Any]:
    """Start the fixed product daemon as a child of AgentCTL's service cgroup.

    The dedicated child process group is terminated locally on every proof
    exit. The runtime retains lifecycle authority and is the outer cleanup net.
    """
    log_path = artifact_root / "polylogued.log"
    command = [
        sys.executable,
        "-c",
        "from polylogue.daemon.commands import main; main()",
        "run",
        "--api-port",
        "0",
        "--port",
        "0",
        "--listener-info-path",
        str(listener_info_path),
        "--browser-capture-auth-token",
        _RECEIVER_TOKEN,
        "--api-auth-token",
        _API_TOKEN,
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


def _run_shared_chrome_control(
    *, repo_root: Path, proof_environment: dict[str, str], timeout_s: float = _SHARED_CHROME_TIMEOUT_S
) -> dict[str, object]:
    with shared_chrome_extension_lock(timeout_s=timeout_s):
        return _run_shared_chrome_control_locked(
            repo_root=repo_root, proof_environment=proof_environment, timeout_s=timeout_s
        )


def _run_shared_chrome_control_locked(
    *, repo_root: Path, proof_environment: dict[str, str], timeout_s: float
) -> dict[str, object]:
    """Exercise the existing Chrome only through Sinnix's owned control boundary."""
    extension_root = repo_root / "browser-extension"
    environment = os.environ.copy()
    environment.update(proof_environment)
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
        detail = " ".join(stderr.split())[-_CHILD_ERROR_TAIL_CHARS:] or f"exit {completed.returncode}"
        raise RuntimeError(f"shared-Chrome control proof failed: {detail}")
    payload = json.loads(stdout)
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("shared-Chrome control proof reported failure")
    native = payload.get("native_transport")
    if not isinstance(native, dict) or native.get("cancellation") is not True:
        raise RuntimeError("native transport proof missing")
    if native.get("response_sha256") != proof_environment["POLYLOGUE_DEV_LOOP_ATTACHMENT_SHA256"]:
        raise RuntimeError("native response bytes differ")
    return payload


def _submit_deterministic_captures(*, capture_port: int, session_id: str) -> dict[str, dict[str, str]]:
    """Keep receiver, archive, and API convergence deterministic and browser-free."""
    captures: dict[str, dict[str, str]] = {}
    for provider in _DETERMINISTIC_PROVIDERS:
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


def _poll_archive_state(
    *, receiver_url: str, provider: str, provider_session_id: str, timeout_s: float
) -> dict[str, object] | None:
    query = urlencode({"provider": provider, "provider_session_id": provider_session_id})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() <= deadline:
        try:
            status, payload = _http_get_json(f"{receiver_url}/v1/archive-state?{query}", bearer_token=_RECEIVER_TOKEN)
        except OSError:
            status, payload = 0, {}
        if status == 200 and payload.get("raw_row_exists") is True and payload.get("indexed_session_exists") is True:
            return payload
        time.sleep(0.25)
    return None


def _fetch_api_messages(*, api_url: str, session_id: str) -> bool:
    status, payload = _http_get_json(
        f"{api_url}/api/sessions/{quote(session_id, safe='')}/messages?limit=5",
        bearer_token=_API_TOKEN,
    )
    messages = payload.get("messages")
    if status != 200 or payload.get("session_id") != session_id or not isinstance(messages, list) or not messages:
        return False
    return all(
        isinstance(message, dict)
        and isinstance(message.get("id"), str)
        and bool(message["id"])
        and isinstance(message.get("role"), str)
        and bool(message["role"])
        and "text" in message
        and (message["text"] is None or isinstance(message["text"], str))
        and isinstance(message.get("target_ref"), dict)
        and message["target_ref"]
        == {
            "target_type": "message",
            "target_id": message["id"],
            "session_id": session_id,
            "message_id": message["id"],
            "identity_key": f"message:{session_id}:{message['id']}",
        }
        for message in messages
    )


def _validated_provider_captures(captures: object) -> dict[str, dict[str, str]]:
    """Validate the complete deterministic capture contract before polling."""
    if not isinstance(captures, dict):
        raise RuntimeError("deterministic provider captures were not a provider mapping")
    expected = set(_DETERMINISTIC_PROVIDERS)
    actual = set(captures)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected_count = len(actual - expected)
        raise RuntimeError(
            "deterministic provider capture set mismatch: "
            + json.dumps({"missing": missing, "unexpected_count": unexpected_count}, sort_keys=True)
        )
    validated: dict[str, dict[str, str]] = {}
    invalid: list[str] = []
    for provider in _DETERMINISTIC_PROVIDERS:
        item = captures[provider]
        if (
            not isinstance(item, dict)
            or set(item) != {"provider", "provider_session_id"}
            or item.get("provider") != provider
            or not isinstance(item.get("provider_session_id"), str)
            or not item["provider_session_id"]
        ):
            invalid.append(provider)
            continue
        validated[provider] = {"provider": provider, "provider_session_id": item["provider_session_id"]}
    if invalid:
        raise RuntimeError("deterministic provider capture entries were malformed: " + ", ".join(invalid))
    return validated


def _redacted_convergence(convergence: dict[str, dict[str, object]]) -> dict[str, dict[str, bool]]:
    return {
        provider: {
            "archive": row.get("archive") is True,
            "api": row.get("api") is True,
            "indexed_session_id_present": isinstance(row.get("indexed_session_id"), str),
        }
        for provider, row in convergence.items()
    }


def run_proof(*, repo_root: Path | None = None, readiness_timeout_s: float = 45.0) -> dict[str, object]:
    """Run the bounded Polylogue semantics inside the AgentCTL job boundary."""
    checkout = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    _require_agentctl_operation_context()
    archive_root, artifact_root = _service_paths()
    artifact_root.mkdir(parents=True, exist_ok=True)
    initialize_active_archive_root(archive_root)
    receiver_auth = run_receiver_smoke(spool_path=artifact_root / "receiver-auth")
    if receiver_auth.get("ok") is not True:
        raise RuntimeError("receiver authentication proof failed")
    listener_info_path = artifact_root / f"listeners-{uuid.uuid4().hex}.json"
    environment = _proof_environment(archive_root=archive_root, artifact_root=artifact_root)
    daemon = _start_daemon(
        repo_root=checkout,
        environment=environment,
        artifact_root=artifact_root,
        listener_info_path=listener_info_path,
    )
    try:
        api_port, capture_port = _await_listener_ports(
            daemon=daemon, listener_info_path=listener_info_path, timeout_s=readiness_timeout_s
        )
        api_url = f"http://127.0.0.1:{api_port}"
        receiver_url = f"http://127.0.0.1:{capture_port}"
        _await_api(base_url=api_url, timeout_s=readiness_timeout_s, daemon=daemon)
        session_id = f"polylogue-agentctl-proof-{api_port}-{capture_port}"
        with scoped_native_transport_proof(
            repo_root=checkout,
            scratch=artifact_root,
            environment=environment,
            endpoint=receiver_url,
        ) as native_environment:
            chrome_proof = _run_shared_chrome_control(repo_root=checkout, proof_environment=native_environment)
        providers = _validated_provider_captures(
            _submit_deterministic_captures(capture_port=capture_port, session_id=session_id)
        )
        archive_ok = False
        api_ok = False
        convergence: dict[str, dict[str, object]] = {}
        for provider in _DETERMINISTIC_PROVIDERS:
            item = providers[provider]
            provider_session_id = item["provider_session_id"]
            archive_state = _poll_archive_state(
                receiver_url=receiver_url,
                provider=provider,
                provider_session_id=provider_session_id,
                timeout_s=readiness_timeout_s,
            )
            indexed_session_id = archive_state.get("indexed_session_id") if archive_state is not None else None
            provider_api_ok = isinstance(indexed_session_id, str) and _fetch_api_messages(
                api_url=api_url,
                session_id=indexed_session_id,
            )
            convergence[provider] = {
                "archive": archive_state is not None,
                "api": provider_api_ok,
                "indexed_session_id": indexed_session_id,
            }
        archive_ok = bool(convergence) and all(row["archive"] is True for row in convergence.values())
        api_ok = bool(convergence) and all(row["api"] is True for row in convergence.values())
        if not archive_ok or not api_ok:
            raise RuntimeError(
                "archive/API convergence proof failed: "
                + json.dumps(_redacted_convergence(convergence), sort_keys=True)
            )
        return {
            "ok": True,
            "ports": {"api": api_port, "browser_capture": capture_port},
            "receiver_auth": {"ok": True},
            "shared_chrome": chrome_proof,
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
