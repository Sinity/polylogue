"""polylogue-ioz7 (direct fix for polylogue-b508): purge agent-*.meta.json
subagent-sidecar phantom sessions.

Builds a fixture archive with:

* two ``agent-<hash>.meta`` sidecar phantom sessions (message_count=0,
  raw source_path ending in ``.meta.json``) -- the exact target shape;
* the corresponding real subagent transcript sessions (non-empty,
  ``agent-<hash>`` native_id, no ``.meta`` suffix, different raw source_path)
  that must survive untouched;
* a genuinely empty top-level session whose raw source_path is a plain
  ``.jsonl`` (not ``.meta.json``) -- must also survive untouched, proving the
  classifier does not fall back to a blanket "no messages" predicate.

Proves:

* the read-only classifier (``scan_agent_meta_sidecar_sessions``) matches
  exactly the two sidecar phantoms, never the real transcripts or the
  legitimately-empty session;
* dry-run (the default) never mutates anything;
* --apply deletes exactly the matched sessions, leaves raw_sessions/blobs in
  source.db untouched, and writes one immutable receipt per purged row;
* applying without a backup manifest is refused before anything is touched;
* applying is refused if a candidate's native_id fails the agent-meta shape
  cross-check (defense in depth beyond the bead's own source_path-only
  predicate).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.maintenance.agent_meta_sidecar_purge_apply import (
    TOOL_VERSION,
    AgentMetaSidecarPurgeApplyError,
    apply_agent_meta_sidecar_purge,
)
from polylogue.storage.agent_meta_sidecar_sweep import scan_agent_meta_sidecar_sessions
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_ACQUIRED_AT_MS = 1_700_000_000_000


def _write_raw(archive: ArchiveStore, *, raw_id: str, source_path: str) -> None:
    archive.write_raw_payload(
        provider=Provider.CLAUDE_CODE,
        payload=f"raw:{raw_id}".encode(),
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=_ACQUIRED_AT_MS,
        raw_id=raw_id,
    )


def _insert_session(
    conn: sqlite3.Connection,
    *,
    native_id: str,
    origin: str,
    raw_id: str,
    message_count: int,
) -> None:
    content_hash = f"hash-{native_id}".encode().ljust(32, b"\x00")[:32]
    conn.execute(
        "INSERT INTO sessions (native_id, origin, raw_id, message_count, content_hash) VALUES (?, ?, ?, ?, ?)",
        (native_id, origin, raw_id, message_count, content_hash),
    )


def _build_fixture_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        # Two agent-meta.json sidecar phantoms -- the target shape.
        _write_raw(
            archive,
            raw_id="raw-meta-1",
            source_path="/home/x/.claude/projects/p/s1/subagents/agent-aaa111.meta.json",
        )
        _write_raw(
            archive,
            raw_id="raw-meta-2",
            source_path="/home/x/.claude/projects/p/s1/subagents/agent-bbb222.meta.json",
        )
        # Real subagent transcripts (correctly ingested, non-empty, no .meta suffix).
        _write_raw(
            archive,
            raw_id="raw-real-1",
            source_path="/home/x/.claude/projects/p/s1/subagents/agent-aaa111.jsonl",
        )
        _write_raw(
            archive,
            raw_id="raw-real-2",
            source_path="/home/x/.claude/projects/p/s1/subagents/agent-bbb222.jsonl",
        )
        # A legitimately-empty top-level session -- must never be swept up by
        # a blanket "no messages" predicate.
        _write_raw(
            archive,
            raw_id="raw-empty-legit",
            source_path="/home/x/.claude/projects/p/s2.jsonl",
        )
        archive.commit()

    conn = sqlite3.connect(archive_root / "index.db")
    try:
        _insert_session(
            conn,
            native_id="agent-aaa111.meta",
            origin=Origin.CLAUDE_CODE_SESSION.value,
            raw_id="raw-meta-1",
            message_count=0,
        )
        _insert_session(
            conn,
            native_id="agent-bbb222.meta",
            origin=Origin.CLAUDE_CODE_SESSION.value,
            raw_id="raw-meta-2",
            message_count=0,
        )
        _insert_session(
            conn,
            native_id="agent-aaa111",
            origin=Origin.CLAUDE_CODE_SESSION.value,
            raw_id="raw-real-1",
            message_count=5,
        )
        _insert_session(
            conn,
            native_id="agent-bbb222",
            origin=Origin.CLAUDE_CODE_SESSION.value,
            raw_id="raw-real-2",
            message_count=3,
        )
        _insert_session(
            conn,
            native_id="s2-uuid",
            origin=Origin.CLAUDE_CODE_SESSION.value,
            raw_id="raw-empty-legit",
            message_count=0,
        )
        conn.commit()
    finally:
        conn.close()

    return archive_root


def _session_ids(archive_root: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT session_id FROM sessions").fetchall()}
    finally:
        conn.close()


def _raw_ids(archive_root: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT raw_id FROM raw_sessions").fetchall()}
    finally:
        conn.close()


def _receipt_rows(archive_root: Path) -> dict[str, str]:
    conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT session_id, tool_version FROM agent_meta_sidecar_purge_receipts").fetchall()
    finally:
        conn.close()
    return dict(rows)


def test_classifier_matches_only_the_meta_sidecar_phantoms(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = scan_agent_meta_sidecar_sessions(conn, archive_root / "source.db")
    finally:
        conn.close()

    assert {c.session_id for c in plan.candidates} == {
        "claude-code-session:agent-aaa111.meta",
        "claude-code-session:agent-bbb222.meta",
    }
    assert plan.shape_mismatch_count == 0


def test_dry_run_makes_zero_mutations(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before_sessions = _session_ids(archive_root)
    before_raws = _raw_ids(archive_root)

    report = apply_agent_meta_sidecar_purge(archive_root, dry_run=True)

    assert report.applied is False
    assert report.purged_count == 2
    assert report.purged_session_ids == ()

    assert _session_ids(archive_root) == before_sessions
    assert _raw_ids(archive_root) == before_raws
    assert _receipt_rows(archive_root) == {}


def test_apply_purges_only_the_meta_sidecar_phantoms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)

    validated: list[tuple[Path, object]] = []

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.agent_meta_sidecar_purge_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    before_raws = _raw_ids(archive_root)
    manifest = tmp_path / "verified-backup" / "manifest.json"

    report = apply_agent_meta_sidecar_purge(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.purged_count == 2
    assert set(report.purged_session_ids) == {
        "claude-code-session:agent-aaa111.meta",
        "claude-code-session:agent-bbb222.meta",
    }
    assert report.backup_manifest == manifest
    assert validated == [(manifest, ArchiveTier.INDEX)]

    remaining = _session_ids(archive_root)
    assert remaining == {
        "claude-code-session:agent-aaa111",
        "claude-code-session:agent-bbb222",
        "claude-code-session:s2-uuid",
    }
    # source.db is never touched -- raw rows and blobs for the purged
    # sessions, including the deleted ones, are all retained.
    assert _raw_ids(archive_root) == before_raws

    receipts = _receipt_rows(archive_root)
    assert set(receipts) == {
        "claude-code-session:agent-aaa111.meta",
        "claude-code-session:agent-bbb222.meta",
    }
    assert all(tool_version == TOOL_VERSION for tool_version in receipts.values())


def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before_sessions = _session_ids(archive_root)

    with pytest.raises(AgentMetaSidecarPurgeApplyError, match="backup manifest"):
        apply_agent_meta_sidecar_purge(archive_root, backup_manifest=None, dry_run=False)

    assert _session_ids(archive_root) == before_sessions
    assert _receipt_rows(archive_root) == {}


def test_apply_refuses_when_backup_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before_sessions = _session_ids(archive_root)

    def _reject(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        raise ValueError("backup manifest does not match live index.db")

    monkeypatch.setattr(
        "polylogue.maintenance.agent_meta_sidecar_purge_apply.validate_migration_backup_manifest",
        _reject,
    )

    manifest = tmp_path / "stale-backup" / "manifest.json"
    with pytest.raises(ValueError, match="does not match"):
        apply_agent_meta_sidecar_purge(archive_root, backup_manifest=manifest, dry_run=False)

    assert _session_ids(archive_root) == before_sessions
    assert _receipt_rows(archive_root) == {}


def test_apply_refuses_on_native_id_shape_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)

    # Corrupt one candidate's native_id so it no longer matches the expected
    # 'agent-<hash>.meta' shape, even though its raw source_path still ends
    # in '.meta.json' -- the actuator must refuse rather than delete it.
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        conn.execute("UPDATE sessions SET native_id = 'unexpected-shape' WHERE raw_id = 'raw-meta-1'")
        conn.commit()
    finally:
        conn.close()

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.agent_meta_sidecar_purge_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    before_sessions = _session_ids(archive_root)
    manifest = tmp_path / "verified-backup" / "manifest.json"

    with pytest.raises(AgentMetaSidecarPurgeApplyError, match="native_id shape"):
        apply_agent_meta_sidecar_purge(archive_root, backup_manifest=manifest, dry_run=False)

    assert _session_ids(archive_root) == before_sessions
    assert _receipt_rows(archive_root) == {}
