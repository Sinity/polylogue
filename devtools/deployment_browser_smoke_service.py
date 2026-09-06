"""Fixed AgentCTL-owned shared-Chrome render proof for the deployed web root."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from devtools.agentctl_service_context import require_declared_operation_context, terminate_process_group

_NODE_PROOF_TIMEOUT_S = 90.0
_MAX_ERROR_MESSAGE = 512


def _proof_environment() -> dict[str, str]:
    """Pass runtime essentials without accepting private-browser configuration."""
    environment = os.environ.copy()
    for name in tuple(environment):
        if "CDP" in name or "PROFILE" in name or "BROWSER_EXECUTABLE" in name:
            environment.pop(name)
    return environment


def run_smoke(*, repo_root: Path | None = None, timeout_s: float = _NODE_PROOF_TIMEOUT_S) -> dict[str, object]:
    """Render the deployed root in one proof-owned parked Chrome target."""
    require_declared_operation_context("deployment_browser_smoke")
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    process = subprocess.Popen(
        ["node", "scripts/deployment_shared_chrome_smoke.mjs"],
        cwd=root / "browser-extension",
        env=_proof_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        terminate_process_group(process)
        raise RuntimeError("shared-Chrome deployment render proof timed out") from error
    finally:
        terminate_process_group(process)
    if process.returncode != 0:
        raise RuntimeError("shared-Chrome deployment render proof failed")
    payload = json.loads(stdout)
    render = payload.get("render") if isinstance(payload, dict) else None
    if payload.get("ok") is not True or not isinstance(render, dict):
        raise RuntimeError("shared-Chrome deployment render proof reported an unsuccessful result")
    if (
        render.get("url") != "http://127.0.0.1:8766/"
        or not isinstance(render.get("dom_bytes"), int)
        or isinstance(render.get("dom_bytes"), bool)
        or render["dom_bytes"] <= 0
        or not isinstance(render.get("screenshot_bytes"), int)
        or isinstance(render.get("screenshot_bytes"), bool)
        or render["screenshot_bytes"] <= 0
        or render.get("target_closed") is not True
    ):
        raise RuntimeError("shared-Chrome deployment render proof returned an invalid result")
    return {"ok": True, "render": render}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one bounded JSON result.")
    parser.parse_args(argv)
    payload: dict[str, object]
    try:
        payload = run_smoke()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        payload = {"ok": False, "error": str(error)[:_MAX_ERROR_MESSAGE]}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
