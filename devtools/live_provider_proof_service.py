"""Declared AgentCTL entrypoint for the private copied-profile provider proof.

The profile path, Chrome resolution, providers, receiver, output, and leased
ports are fixed by this operation. Sinnixd remains the authority for admission,
exact-head binding, and leases. This module verifies it is still inside that
specific transient unit before it can start Node or Chrome.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
from pathlib import Path
from threading import Thread
from typing import Any

from devtools.sinnixd_service_context import require_declared_service_context, terminate_process_group
from polylogue.browser_capture.server import make_server

_CDP_PORT_ENV = "POLYLOGUE_LIVE_PROVIDER_CDP_PORT"
_RECEIVER_PORT_ENV = "POLYLOGUE_LIVE_PROVIDER_RECEIVER_PORT"
_CDP_PORT_RANGE = (49056, 49119)
_RECEIVER_PORT_RANGE = (49120, 49183)


def _leased_port(name: str, bounds: tuple[int, int]) -> int:
    raw = os.environ.get(name)
    try:
        port = int(raw) if raw is not None else None
    except ValueError as error:
        raise ValueError(f"{name} must be an integer declared-service port") from error
    if port is None or not bounds[0] <= port <= bounds[1]:
        raise ValueError(f"{name} is outside its fixed live-provider service port range")
    return port


def _service_context_ports() -> tuple[int, int]:
    require_declared_service_context("live_provider_proof")
    return (_leased_port(_CDP_PORT_ENV, _CDP_PORT_RANGE), _leased_port(_RECEIVER_PORT_ENV, _RECEIVER_PORT_RANGE))


def run_proof(*, repo_root: Path | None = None) -> dict[str, object]:
    """Run only the fixed copied-profile implementation under its service lease."""
    cdp_port, receiver_port = _service_context_ports()
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    extension_root = root / "browser-extension"
    scratch = Path(os.environ["TMPDIR"]).resolve() / "polylogue-live-provider-proof"
    spool = scratch / "browser-capture"
    spool.mkdir(parents=True, exist_ok=True)
    receiver_token = secrets.token_urlsafe(32)
    server = make_server("127.0.0.1", receiver_port, spool_path=spool, auth_token=receiver_token)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = os.environ.copy()
    environment["POLYLOGUE_LIVE_PROVIDER_RECEIVER_TOKEN"] = receiver_token
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
        stdout, stderr = process.communicate(timeout=180)
        if process.returncode != 0:
            raise RuntimeError((stderr or stdout or "live provider proof failed")[:512])
        result = json.loads(stdout)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError("live provider proof reported an unsuccessful result")
        return {
            "ok": True,
            "ports": {"browser_cdp": cdp_port, "browser_capture": receiver_port},
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
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        payload = {"ok": False, "error": str(error)[:512]}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
