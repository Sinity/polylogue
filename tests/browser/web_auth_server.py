"""Launch the production daemon over a deterministic demo archive for Playwright."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.daemon.web_auth import WebCredentialRegistry
from polylogue.demo.seed import seed_demo_archive


def _configure_archive(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    state_root = root.parent / "state"
    data_root = root.parent / "data"
    state_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(root)
    os.environ["XDG_STATE_HOME"] = str(state_root)
    os.environ["XDG_DATA_HOME"] = str(data_root)
    os.environ["POLYLOGUE_SCHEMA_VALIDATION"] = "off"


def main() -> int:
    archive_root_raw = os.environ.get("POLYLOGUE_BROWSER_TEST_ARCHIVE_ROOT")
    if not archive_root_raw:
        raise RuntimeError("POLYLOGUE_BROWSER_TEST_ARCHIVE_ROOT is required")
    archive_root = Path(archive_root_raw).expanduser().resolve()
    _configure_archive(archive_root)
    seed = asyncio.run(seed_demo_archive(archive_root, force=True, with_overlays=True))

    ttl_s = int(os.environ.get("POLYLOGUE_BROWSER_TEST_CREDENTIAL_TTL_S", "4"))
    server = DaemonAPIHTTPServer(
        ("127.0.0.1", 0),
        DaemonAPIHandler,
        auth_token="playwright-machine-token",
        api_host="127.0.0.1",
        web_credentials=WebCredentialRegistry(ttl_s=ttl_s),
    )
    port = int(server.server_address[1])

    def stop(_signum: int, _frame: FrameType | None) -> None:
        threading.Thread(target=server.shutdown, name="playwright-server-stop", daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        json.dumps(
            {
                "kind": "ready",
                "base_url": f"http://127.0.0.1:{port}",
                "session_count": seed.session_count,
                "message_count": seed.message_count,
                "credential_ttl_s": ttl_s,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"kind": "fatal", "error": type(exc).__name__}), file=sys.stderr, flush=True)
        raise
