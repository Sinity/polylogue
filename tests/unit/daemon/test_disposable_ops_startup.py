"""A stale disposable event ledger reconverges before startup can emit events."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.write_lease import write_lease
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.events import emit_daemon_event, query_events_since
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.stale_ops import (
    custody_file_inventory,
    durable_sql_inventory,
    make_ops_event_schema_stale,
    seed_custody_files,
)


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
    custody = seed_custody_files(root)
    durable = durable_sql_inventory(root)
    identities = {p.name: (p.stat().st_dev, p.stat().st_ino) for p in root.glob("*.db") if p.name != "ops.db"}
    # Existing staged-reset audit inspection selects WAL mode on its first
    # open. Only that physical journal header may change; durable SQL may not.
    protected = {
        p.name: hashlib.sha256(p.read_bytes()).digest()
        for p in root.glob("*.db")
        if p.name not in {"ops.db", "audit.db"}
    }

    class StartupCheckpointError(Exception):
        pass

    current_ops_digest: bytes | None = None

    def reached_preflight() -> None:
        if current_ops_digest is not None:
            assert hashlib.sha256((root / "ops.db").read_bytes()).digest() == current_ops_digest

        def emit() -> None:
            with write_lease("daemon.startup.test_event", archive_root=root):
                emit_daemon_event("synthetic_startup", archive_root_path=root, idempotency_key="same-startup")

        # The preflight hook runs on the daemon's event loop; the synchronous lease may not block it.
        run_off_event_loop(emit)
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
        events = query_events_since(None, kinds=("synthetic_startup",)).events
        assert len(events) == 1
        assert {
            p.name: hashlib.sha256(p.read_bytes()).digest()
            for p in root.glob("*.db")
            if p.name not in {"ops.db", "audit.db"}
        } == protected
        assert durable_sql_inventory(root) == durable
        assert {
            p.name: (p.stat().st_dev, p.stat().st_ino) for p in root.glob("*.db") if p.name != "ops.db"
        } == identities
        assert custody_file_inventory(root) == custody


@pytest.mark.parametrize(
    "failure", [PermissionError("synthetic permission fault"), sqlite3.OperationalError("database is locked")]
)
def test_ops_inspection_fault_does_not_authorize_disposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A failed inspection must leave even a stale ops tier byte-for-byte intact."""
    from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
    from polylogue.operations.mutation_replay import reconverge_disposable_ops_on_startup
    from polylogue.operations.reset_safety import archive_tiers_closed

    initialize_active_archive_root(tmp_path)
    path = tmp_path / "ops.db"
    make_ops_event_schema_stale(path)
    before = path.read_bytes()
    owner = acquire_durable_archive_ownership(tmp_path, owner_id="synthetic-startup")

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_readonly_connection", fail_read)
    try:
        with write_lease("daemon.startup.test", archive_root=tmp_path), archive_tiers_closed(tmp_path):
            with pytest.raises(type(failure)):
                reconverge_disposable_ops_on_startup(tmp_path, archive_owner=owner)
    finally:
        owner.release()
    assert path.read_bytes() == before


def test_ops_reconvergence_refuses_outside_closed_owned_startup(tmp_path: Path) -> None:
    """Holding the lease alone cannot unlink a tier another handle may retain."""
    from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
    from polylogue.operations.mutation_replay import reconverge_disposable_ops_on_startup
    from polylogue.operations.reset_safety import LiveArchiveTierResetError

    initialize_active_archive_root(tmp_path)
    before = (tmp_path / "ops.db").read_bytes()
    owner = acquire_durable_archive_ownership(tmp_path, owner_id="synthetic-startup")
    try:
        with write_lease("daemon.startup.test", archive_root=tmp_path):
            with pytest.raises(LiveArchiveTierResetError):
                reconverge_disposable_ops_on_startup(tmp_path, archive_owner=owner)
    finally:
        owner.release()
    assert (tmp_path / "ops.db").read_bytes() == before


def test_owned_ops_reconvergence_preserves_every_unrelated_tier_byte(tmp_path: Path) -> None:
    """The replacement itself cannot change even an unrelated journal header."""
    from polylogue.operations.durable_change_train import acquire_durable_archive_ownership
    from polylogue.operations.mutation_replay import reconverge_disposable_ops_on_startup
    from polylogue.operations.reset_safety import archive_tiers_closed

    initialize_active_archive_root(tmp_path)
    make_ops_event_schema_stale(tmp_path / "ops.db")
    custody = seed_custody_files(tmp_path)
    protected = {
        p.name: (p.stat().st_dev, p.stat().st_ino, p.read_bytes()) for p in tmp_path.glob("*.db") if p.name != "ops.db"
    }
    owner = acquire_durable_archive_ownership(tmp_path, owner_id="synthetic-startup")
    try:
        with write_lease("daemon.startup.test", archive_root=tmp_path), archive_tiers_closed(tmp_path):
            assert reconverge_disposable_ops_on_startup(tmp_path, archive_owner=owner)
    finally:
        owner.release()
    assert {
        p.name: (p.stat().st_dev, p.stat().st_ino, p.read_bytes()) for p in tmp_path.glob("*.db") if p.name != "ops.db"
    } == protected
    assert custody_file_inventory(tmp_path) == custody


@pytest.mark.parametrize("failed_member", ["ops.db-wal", "ops.db"])
def test_ops_disposal_fault_preserves_primary_and_refuses_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_member: str
) -> None:
    """A permission fault cannot become absent state followed by fresh DDL."""
    from polylogue.operations.reset_safety import archive_tiers_closed, discard_closed_derived_tier

    initialize_active_archive_root(tmp_path)
    primary = tmp_path / "ops.db"
    before = primary.read_bytes()
    wal = tmp_path / "ops.db-wal"
    wal.write_bytes(b"synthetic old sidecar")
    unlink = Path.unlink
    exists = Path.exists

    def masked_existence(path: Path) -> bool:
        # Python 3.14 reports permission-denied stat as absence. Reverting to
        # an existence probe would skip this member and swallow the fault.
        return False if path == tmp_path / failed_member else exists(path)

    def fail_member(path: Path, missing_ok: bool = False) -> None:
        if path == tmp_path / failed_member:
            raise PermissionError("synthetic disposal fault")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "exists", masked_existence)
    monkeypatch.setattr(Path, "unlink", fail_member)
    with archive_tiers_closed(tmp_path), pytest.raises(PermissionError):
        discard_closed_derived_tier(tmp_path, primary)
    assert primary.read_bytes() == before
    if failed_member == "ops.db-wal":
        assert wal.read_bytes() == b"synthetic old sidecar"
