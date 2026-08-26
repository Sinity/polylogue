"""Typed hook topology and physical-carrier identity laws."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.sources.hooks import (
    HookSpoolSourceSpec,
    HookSpoolTopologyError,
    hook_spool_sources,
    validate_hook_spool_topology,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent, HookEventConflictError


def test_hook_topology_requires_one_flat_primary_and_read_only_legacy_roots(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    legacy = tmp_path / "legacy"
    primary.mkdir()
    legacy.mkdir()
    specs = validate_hook_spool_topology(
        [
            HookSpoolSourceSpec("primary", "primary-writable", primary),
            HookSpoolSourceSpec("legacy", "legacy-read-only", legacy),
        ],
        require_existing=True,
    )
    assert specs[0].root == primary.resolve()
    with pytest.raises(HookSpoolTopologyError, match="nested"):
        validate_hook_spool_topology(
            [
                HookSpoolSourceSpec("primary", "primary-writable", primary),
                HookSpoolSourceSpec("legacy", "legacy-read-only", primary / "nested"),
            ]
        )
    with pytest.raises(HookSpoolTopologyError, match="primary-writable"):
        validate_hook_spool_topology([HookSpoolSourceSpec("legacy", "legacy-read-only", legacy)])


def test_default_hook_topology_enumerates_primary_and_declared_legacy(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    legacy = tmp_path / "legacy"
    primary.mkdir()
    legacy.mkdir()

    specs = hook_spool_sources(primary_root=primary, legacy_roots=(legacy,))

    assert [(spec.source_id, spec.role, spec.root) for spec in specs] == [
        ("primary-hook-spool", "primary-writable", primary.resolve()),
        ("legacy-hook-spool-0", "legacy-read-only", legacy.resolve()),
    ]


def test_hook_carriers_preserve_two_coordinates_and_reject_coordinate_conflict(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    event = ArchiveHookEvent(
        hook_event_id="hook:carrier-law",
        origin=Origin.CODEX_SESSION,
        source_path="/first/representative.json",
        event_type="PostToolUse",
        payload={"event": "PostToolUse", "n": 1},
        observed_at_ms=1,
        native_id="native",
        session_native_id="session",
    )
    with ArchiveStore(archive_root) as archive:
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=b'{"event":"PostToolUse","n":1}',
            source_path=event.source_path,
            acquired_at_ms=1,
            hook_event=event,
            carrier_source_id="primary",
            carrier_relative_path="a.json",
        )
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=b'{"event":"PostToolUse","n":1}',
            source_path=event.source_path,
            acquired_at_ms=1,
            hook_event=event,
            carrier_source_id="legacy",
            carrier_relative_path="b.json",
            carrier_role="legacy-read-only",
        )
        with pytest.raises(HookEventConflictError):
            archive.write_hook_event(
                provider=Provider.CODEX,
                payload=b"different",
                source_path="/second/representative.json",
                acquired_at_ms=1,
                hook_event=event,
                carrier_source_id="primary",
                carrier_relative_path="a.json",
            )
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM hook_event_carriers").fetchone() == (2,)
        assert conn.execute("SELECT source_path FROM raw_hook_events").fetchone() == (event.source_path,)
