"""A stale disposable event ledger reconverges before startup can emit events."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.events import emit_daemon_event, query_events_since
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.stale_ops import make_ops_event_schema_stale


def test_production_startup_reconverges_stale_ops_then_restart_keeps_event_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the startup replacement, the first event is a SchemaSkew refusal.

    Stop at the real startup preflight after ownership, migrations, resets and
    ops admission. The checkpoint emits through the production ledger; a
    second complete startup admission must retain that idempotent event.
    """
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    make_ops_event_schema_stale(root / "ops.db")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    protected = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in root.glob("*.db") if p.name != "ops.db"}

    class StartupCheckpointError(Exception):
        pass

    current_ops_digest: bytes | None = None

    def reached_preflight() -> None:
        if current_ops_digest is not None:
            assert hashlib.sha256((root / "ops.db").read_bytes()).digest() == current_ops_digest
        emit_daemon_event("synthetic_startup", archive_root_path=root, idempotency_key="same-startup")
        raise StartupCheckpointError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", reached_preflight)
    for _restart in range(2):
        with pytest.raises(StartupCheckpointError):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                )
            )
        current_ops_digest = hashlib.sha256((root / "ops.db").read_bytes()).digest()
        events = query_events_since(0, kinds=("synthetic_startup",)).events
        assert len(events) == 1
        assert {
            p.name: hashlib.sha256(p.read_bytes()).digest() for p in root.glob("*.db") if p.name != "ops.db"
        } == protected
