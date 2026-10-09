"""A daemon on an archive whose active Index carries a stale derived identity."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig
from polylogue.logging import capture
from polylogue.paths import browser_capture_spool_root
from polylogue.sources.live import WatchSource


def test_stale_index_identity_keeps_daemon_up_schema_blocked(tmp_path: Path) -> None:
    """A stale Index identity parks derived work; it never ends the daemon.

    Reverting the ``schema_blocked`` guard around ``daemon.cold_build.probe``
    in ``run_daemon_services`` makes the writable probe open raise
    ``SchemaSkewError`` out of ``asyncio.run`` instead of the listener stop.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.paths import archive_root
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = Path(archive_root())
    with ArchiveStore(root, initialize=True, read_only=False):
        pass
    with sqlite3.connect(root / "index.db") as connection:
        connection.execute("UPDATE schema_identity SET identity = 'stale' WHERE tier = 'index'")

    withheld = threading.Event()
    original_emit = cast(Any, daemon_cli).emit

    def observed_emit(event: str, *args: Any, **kwargs: Any) -> None:
        original_emit(event, *args, **kwargs)
        if event == "daemon.cold_build.withheld":
            withheld.set()

    class FakeServer:
        @property
        def config(self) -> BrowserCaptureReceiverConfig:
            return BrowserCaptureReceiverConfig(spool_path=browser_capture_spool_root())

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            # The listener stays up until startup has passed the cold-build
            # decision, then stops the daemon the ordinary way.
            withheld.wait()
            raise RuntimeError("server stopped")

        def shutdown(self) -> None:
            withheld.set()

        def server_close(self) -> None:
            return None

    with (
        patch.object(daemon_cli, "make_server", return_value=FakeServer()),
        patch.object(cast(Any, daemon_cli), "emit", observed_emit),
        capture() as records,
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=tmp_path / "codex"),),
                enable_watch=True,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )

    withheld_records = [record for record in records if record["event"] == "daemon.cold_build.withheld"]
    assert len(withheld_records) == 1
    assert withheld_records[0]["outcome"] == "refused"
    assert withheld_records[0]["reason"] == "schema_blocked"
