"""Fixed AgentCTL-owned browser-render smoke for the deployed Polylogue web root.

Sinnixd is the admission and lifecycle authority. This private module performs
the product probe after the declared operation has created the service cgroup
and leased its CDP port.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from devtools import deployment_smoke

_CDP_PORT_ENV = "POLYLOGUE_DEPLOYMENT_BROWSER_CDP_PORT"
_CDP_PORT_RANGE = (48992, 49055)


def _service_context_port() -> int:
    """Reject ordinary shell calls that lack the expected fixed service context.

    This is defense in depth only. Its environment values are not a capability;
    Sinnixd authorizes the job, binds the checkout, leases the port, and owns
    the transient systemd cgroup.
    """
    if (
        os.environ.get("SINNIXD_PROJECT_ID") != "polylogue"
        or os.environ.get("SINNIXD_OPERATION") != "deployment_browser_smoke"
    ):
        raise ValueError("deployment browser smoke rejects execution outside its fixed service context")
    raw_port = os.environ.get(_CDP_PORT_ENV)
    try:
        port = int(raw_port) if raw_port is not None else None
    except ValueError as error:
        raise ValueError(f"{_CDP_PORT_ENV} must be an integer fixed-service port") from error
    if port is None or not _CDP_PORT_RANGE[0] <= port <= _CDP_PORT_RANGE[1]:
        raise ValueError(f"{_CDP_PORT_ENV} is outside the fixed deployment-browser port range")
    return port


def run_smoke(*, timeout_s: float = 90.0) -> deployment_smoke.BrowserRenderProbe:
    """Render the fixed deployed web root and require Chrome to exit cleanly."""
    port = _service_context_port()
    return deployment_smoke._probe_browser_render(
        "http://127.0.0.1:8766/",
        path=deployment_smoke.SYSTEMWIDE_PATH,
        timeout_s=timeout_s,
        executable=None,
        debugging_port=port,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit one bounded JSON result.")
    parser.parse_args(argv)
    try:
        payload = asdict(run_smoke())
    except (OSError, ValueError) as error:
        payload = {"ok": False, "error": str(error)[:512]}
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
