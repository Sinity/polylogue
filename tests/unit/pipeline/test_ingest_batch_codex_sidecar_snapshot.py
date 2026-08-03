"""Codex sidecar acquisition is a frozen, persisted snapshot (polylogue-ih67 AC#3/4).

The daemon's canonical raw-record ingest used to call
``CodexAssemblySpec.discover_sidecars`` directly against the live filesystem
on every reprocess. That meant an ambient edit to a live ``history.jsonl``
(the operator typing a new prompt, or the file being rotated/truncated)
could silently change the title a *previously acquired* raw record resolves
to on a later reprocess pass -- the replay was not actually deterministic
over the acquired bytes.

``_resolve_codex_sidecar_snapshots`` (``ingest_batch/_core.py``) fixes this:
it runs in the main process before raw records are dispatched to the
ProcessPoolExecutor, resolving each Codex raw record's sidecar data once and
persisting it in ``history_sidecars`` (source.db, keyed by
``(origin, source_path)``). Once persisted, all later calls replay the
frozen snapshot regardless of what the live file now contains.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_batch._core import _resolve_codex_sidecar_snapshots
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _codex_runtime_root(tmp_path: Path, session_id: str) -> Path:
    codex_dir = tmp_path / ".codex"
    rollout = codex_dir / "sessions" / "2026" / f"rollout-{session_id}.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text("{}\n", encoding="utf-8")
    return rollout


def _record(source_path: str) -> RawSessionRecord:
    return RawSessionRecord(
        raw_id=f"raw-{abs(hash(source_path))}",
        source_name="codex",
        source_path=source_path,
        payload_provider=Provider.CODEX,
        source_index=None,
        blob_size=2,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )


@pytest.fixture
def archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    return root


def test_first_resolution_persists_snapshot_and_reads_disk(tmp_path: Path, archive_root: Path) -> None:
    session_id = "aaaa1111-2222-3333-4444-555566667777"
    rollout = _codex_runtime_root(tmp_path, session_id)
    (rollout.parents[2] / "history.jsonl").write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "original title"}) + "\n",
        encoding="utf-8",
    )

    record = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record], archive_root=archive_root)

    assert record.sidecar_snapshot is not None
    assert record.sidecar_snapshot["history_titles"][session_id] == "original title"

    with sqlite3.connect(str(archive_root / "source.db")) as conn:
        rows = conn.execute("SELECT payload_json FROM history_sidecars").fetchall()
    assert len(rows) == 1
    persisted = json.loads(rows[0][0])
    assert persisted["history_titles"][session_id] == "original title"


def test_ambient_edit_after_first_resolution_does_not_alter_replay(tmp_path: Path, archive_root: Path) -> None:
    """Anti-vacuity target: a live history.jsonl edit must not change a frozen replay."""
    session_id = "bbbb1111-2222-3333-4444-555566667777"
    rollout = _codex_runtime_root(tmp_path, session_id)
    history_path = rollout.parents[2] / "history.jsonl"
    history_path.write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "original title"}) + "\n",
        encoding="utf-8",
    )

    record_first = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record_first], archive_root=archive_root)
    assert record_first.sidecar_snapshot is not None
    assert record_first.sidecar_snapshot["history_titles"][session_id] == "original title"

    # Ambient edit: the operator's Codex client rewrites history.jsonl with a
    # different title for the same session id (e.g. edited/retried prompt).
    history_path.write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "AMBIENT EDIT title"}) + "\n",
        encoding="utf-8",
    )

    # Replay: a second raw record sharing the same acquisition source_path
    # (e.g. a later reprocess of the same raw bytes) must resolve to the
    # frozen snapshot, not the now-changed disk content.
    record_replay = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record_replay], archive_root=archive_root)

    assert record_replay.sidecar_snapshot is not None
    assert record_replay.sidecar_snapshot["history_titles"][session_id] == "original title"

    # Still exactly one persisted snapshot row for this path -- resolution
    # did not re-discover and did not grow an unbounded history.
    with sqlite3.connect(str(archive_root / "source.db")) as conn:
        rows = conn.execute("SELECT payload_json FROM history_sidecars").fetchall()
    assert len(rows) == 1


def test_transient_discovery_failure_does_not_freeze_an_empty_snapshot(
    tmp_path: Path,
    archive_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for polylogue-azf7.

    Before the fix, ``_resolve_codex_sidecar_snapshots`` swallowed any
    exception from ``discover_sidecars`` and persisted the resulting empty
    ``{}`` via ``write_history_sidecar``. Because
    ``read_earliest_history_sidecar_for_path`` replays the *first* persisted
    row forever (polylogue-ih67 AC#3/4), a one-off transient error (disk
    hiccup, a sidecar file mid-write) permanently froze an empty snapshot:
    every subsequent ingest of that ``source_path`` -- even long after the
    transient condition cleared -- would keep replaying the empty result and
    enrichment could never recover.
    """
    from polylogue.sources import assembly_codex

    session_id = "dddd1111-2222-3333-4444-555566667777"
    rollout = _codex_runtime_root(tmp_path, session_id)
    (rollout.parents[2] / "history.jsonl").write_text(
        json.dumps({"session_id": session_id, "ts": 1, "text": "real title"}) + "\n",
        encoding="utf-8",
    )

    original_discover = assembly_codex.CodexAssemblySpec.discover_sidecars
    call_count = {"n": 0}

    def flaky_discover(self: object, paths: object) -> object:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated transient disk error during discovery")
        return original_discover(self, paths)  # type: ignore[arg-type]

    monkeypatch.setattr(assembly_codex.CodexAssemblySpec, "discover_sidecars", flaky_discover)

    # First attempt: discovery raises. This ingest's record gets an empty
    # (unenriched) snapshot for its own materialization, but nothing may be
    # persisted as the frozen "first" row.
    record_first = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record_first], archive_root=archive_root)
    assert record_first.sidecar_snapshot == {}

    with sqlite3.connect(str(archive_root / "source.db")) as conn:
        rows = conn.execute("SELECT payload_json FROM history_sidecars").fetchall()
    assert rows == [], "a transient discovery failure must not persist a frozen empty snapshot"

    # Second attempt for the SAME source_path: discovery now succeeds. This
    # must become the real first-ever persisted snapshot and must be read
    # back correctly, not the empty result from the failed first attempt.
    record_second = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record_second], archive_root=archive_root)

    assert record_second.sidecar_snapshot is not None
    assert record_second.sidecar_snapshot["history_titles"][session_id] == "real title"

    with sqlite3.connect(str(archive_root / "source.db")) as conn:
        rows = conn.execute("SELECT payload_json FROM history_sidecars").fetchall()
    assert len(rows) == 1
    persisted = json.loads(rows[0][0])
    assert persisted["history_titles"][session_id] == "real title"


def _write_codex_thread_title_hook_event(archive_root: Path, *, thread_id: str, title: str) -> None:
    """Write a real ``codex_thread_title`` raw_hook_events row (polylogue-0jf4
    shape), the same one ``sources/live/batch.py::_write_codex_thread_state_evidence``
    produces from an acquired ``state_5.sqlite``."""
    import json as _json

    from polylogue.core.enums import Origin
    from polylogue.storage.sqlite.archive_tiers.source_write import (
        ArchiveHookEvent,
        write_source_hook_event,
    )

    payload = {"thread_id": thread_id, "title": title}
    encoded = _json.dumps(payload).encode("utf-8")
    with sqlite3.connect(str(archive_root / "source.db")) as conn:
        write_source_hook_event(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="synthetic:state-db",
            payload=encoded,
            acquired_at_ms=1_000,
            raw_id="raw-codex-thread-title",
            hook_event=ArchiveHookEvent(
                hook_event_id=f"codex-thread-title:{thread_id}",
                origin=Origin.CODEX_SESSION,
                source_path="synthetic:state-db",
                event_type="codex_thread_title",
                payload=payload,
                observed_at_ms=1_000,
                native_id=f"{thread_id}:codex_thread_title",
                session_native_id=thread_id,
            ),
        )


def test_hook_event_titles_merged_from_acquired_evidence(tmp_path: Path, archive_root: Path) -> None:
    """bd polylogue-foee: a codex_thread_title hook event acquired via
    polylogue-0jf4's state-db snapshot (never the JSONL rollout the parser
    reads) surfaces as ``hook_event_titles`` in the sidecar snapshot handed
    to ``CodexAssemblySpec.enrich_session``, even though the rollout itself
    carries no session_index.jsonl/history.jsonl/state_5.sqlite of its own.
    """
    session_id = "cccc1111-2222-3333-4444-555566667777"
    rollout = _codex_runtime_root(tmp_path, session_id)
    _write_codex_thread_title_hook_event(archive_root, thread_id=session_id, title="Durable curated title")

    record = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record], archive_root=archive_root)

    assert record.sidecar_snapshot is not None
    assert record.sidecar_snapshot["hook_event_titles"][session_id] == "Durable curated title"
    # Never a live-read title source -- nothing else discovered one.
    assert not record.sidecar_snapshot.get("state_titles")

    # Never frozen into the persisted history_sidecars snapshot: it is read
    # fresh from raw_hook_events on every batch (see the _core.py comment),
    # so a hook event acquired AFTER a source_path's first-ever resolution
    # still surfaces on the next batch.
    with sqlite3.connect(str(archive_root / "source.db")) as conn:
        rows = conn.execute("SELECT payload_json FROM history_sidecars").fetchall()
    assert len(rows) == 1
    persisted = json.loads(rows[0][0])
    assert "hook_event_titles" not in persisted


def test_hook_event_titles_surface_on_replay_even_when_acquired_later(tmp_path: Path, archive_root: Path) -> None:
    """The hook event can be acquired AFTER a source_path's frozen sidecar
    snapshot already exists; since hook_event_titles is never part of that
    frozen payload, a later batch still sees it."""
    session_id = "eeee1111-2222-3333-4444-555566667777"
    rollout = _codex_runtime_root(tmp_path, session_id)

    record_first = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record_first], archive_root=archive_root)
    assert record_first.sidecar_snapshot is not None
    assert "hook_event_titles" not in record_first.sidecar_snapshot

    _write_codex_thread_title_hook_event(archive_root, thread_id=session_id, title="Acquired after first parse")

    record_second = _record(str(rollout))
    _resolve_codex_sidecar_snapshots([record_second], archive_root=archive_root)
    assert record_second.sidecar_snapshot is not None
    assert record_second.sidecar_snapshot["hook_event_titles"][session_id] == "Acquired after first parse"


def test_non_codex_records_are_left_unresolved(tmp_path: Path, archive_root: Path) -> None:
    record = RawSessionRecord(
        raw_id="raw-claude",
        source_name="claude-code",
        source_path=str(tmp_path / "whatever.jsonl"),
        payload_provider=Provider.CLAUDE_CODE,
        source_index=None,
        blob_size=2,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )
    _resolve_codex_sidecar_snapshots([record], archive_root=archive_root)
    assert record.sidecar_snapshot is None
