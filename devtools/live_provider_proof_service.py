"""Declared AgentCTL entrypoint for the shared-Chrome provider proof.

The providers and receiver are fixed by this operation. The Node workflow uses
the Sinnix shared-Chrome control boundary, which opens and parks proof-owned
windows in the existing authenticated browser. The receiver binds a free
loopback port, passes it to the Node workflow, and publishes it in the result.
The runtime (agentctl) remains the authority for admission and exact-head binding.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from threading import Thread
from typing import Any

from devtools.agentctl_service_context import require_declared_operation_context, terminate_process_group
from polylogue.browser_capture.server import make_server

_RECEIVER_PORT_ENV = "POLYLOGUE_LIVE_PROVIDER_RECEIVER_PORT"
_NODE_PROOF_TIMEOUT_S = 120
_MAX_ERROR_MESSAGE = 512


def run_proof(*, repo_root: Path | None = None) -> dict[str, object]:
    """Run the shared-Chrome workflow against a self-bound loopback receiver."""
    require_declared_operation_context("live_provider_proof")
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    extension_root = root / "browser-extension"
    scratch = Path(tempfile.gettempdir()).resolve() / "polylogue-live-provider-proof"
    spool = scratch / "browser-capture"
    spool.mkdir(parents=True, exist_ok=True)
    receiver_token = secrets.token_urlsafe(32)
    server = make_server("127.0.0.1", 0, spool_path=spool, auth_token=receiver_token)
    receiver_port = int(server.server_address[1])
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = os.environ.copy()
    environment["POLYLOGUE_LIVE_PROVIDER_RECEIVER_TOKEN"] = receiver_token
    environment[_RECEIVER_PORT_ENV] = str(receiver_port)
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            ["node", "scripts/live_provider_proof.mjs"],
            cwd=extension_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert process is not None
        try:
            stdout, _stderr = process.communicate(timeout=_NODE_PROOF_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            terminate_process_group(process)
            with suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=2)
            raise subprocess.TimeoutExpired(
                ["node", "scripts/live_provider_proof.mjs"], _NODE_PROOF_TIMEOUT_S
            ) from error
        if process.returncode != 0:
            raise RuntimeError("live provider proof failed")
        result = json.loads(stdout)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError("live provider proof reported an unsuccessful result")
        return {
            "ok": True,
            "ports": {"browser_capture": receiver_port},
            "providers": sorted(result.get("providers", {})),
        }
    finally:
        if process is not None:
            terminate_process_group(process)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one bounded JSON result.")
    parser.parse_args(argv)
    try:
        payload: dict[str, Any] = run_proof()
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        payload = {"ok": False, "error": {"type": type(error).__name__, "message": str(error)[:_MAX_ERROR_MESSAGE]}}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
