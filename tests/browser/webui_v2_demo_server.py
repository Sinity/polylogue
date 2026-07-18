"""Launch the production daemon over a demo archive for webui-v2 Playwright journeys.

Unlike ``web_auth_server.py`` (which exercises the legacy shell's credential
flows against a bearer token), this script boots the daemon with NO auth
token — the "local dev default" per ``DaemonAPIHandler._check_auth`` — so
``/app/*`` SSR pages and their ``/api/*`` island fetches are open, matching
how a developer runs the daemon locally against `/app`.

The demo archive alone has no session large enough to exercise the
session-read pagination/deep-link machinery (polylogue-07g6), so this script
additionally writes one large synthetic session directly via ``ArchiveStore``
after seeding, deterministic and independent of demo-corpus content changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.demo.seed import seed_demo_archive
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

LARGE_SESSION_PROVIDER_ID = "webui-v2-e2e-large-session"
LARGE_SESSION_MESSAGE_COUNT = 90
LARGE_SESSION_TITLE = "webui-v2 e2e pagination fixture"


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


def _write_large_session(archive_root: Path) -> str:
    with ArchiveStore(archive_root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=LARGE_SESSION_PROVIDER_ID,
                title=LARGE_SESSION_TITLE,
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m{i}",
                        role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                        text=f"webui-v2 e2e fixture message body {i}",
                        position=i,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"webui-v2 e2e fixture message body {i}")],
                    )
                    for i in range(LARGE_SESSION_MESSAGE_COUNT)
                ],
            )
        )


def main() -> int:
    archive_root_raw = os.environ.get("POLYLOGUE_BROWSER_TEST_ARCHIVE_ROOT")
    if not archive_root_raw:
        raise RuntimeError("POLYLOGUE_BROWSER_TEST_ARCHIVE_ROOT is required")
    archive_root = Path(archive_root_raw).expanduser().resolve()
    _configure_archive(archive_root)
    seed = asyncio.run(seed_demo_archive(archive_root, force=True, with_overlays=False))
    large_session_id = _write_large_session(archive_root)

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler, api_host="127.0.0.1")
    port = int(server.server_address[1])

    def stop(_signum: int, _frame: FrameType | None) -> None:
        threading.Thread(target=server.shutdown, name="webui-v2-e2e-server-stop", daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        json.dumps(
            {
                "kind": "ready",
                "base_url": f"http://127.0.0.1:{port}",
                "session_count": seed.session_count + 1,
                "message_count": seed.message_count + LARGE_SESSION_MESSAGE_COUNT,
                "demo_session_ids": list(seed.session_ids),
                "large_session_id": large_session_id,
                "large_session_message_count": LARGE_SESSION_MESSAGE_COUNT,
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
