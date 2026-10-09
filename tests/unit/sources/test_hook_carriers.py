"""Production-route tests for append-only hook-event carriers."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from polylogue.sources.hook_producer import append_carrier_line, carrier_path, compact_legacy_spool
from polylogue.sources.hook_producer import main as hook_producer_main
from polylogue.sources.hooks import (
    HookSpoolRecordError,
    append_hook_event,
    find_carrier_event,
    hook_carrier_dir,
    hook_carrier_provider_dir,
    read_hook_carrier,
    validated_hook_record,
)
from polylogue.sources.live.watcher import LiveWatcher, WatchSource
from polylogue.sources.parsers.hermes_lifecycle import DURABLE_FINALIZE, PER_TURN_END
from polylogue.sources.source_layout import export_drop_layout
from tests.infra.hook_carriers import (
    acquire_hook_carriers,
    hook_event_count,
    materialize_acquired_hook_carriers,
    materialize_hook_carriers,
)
from tests.infra.raw_owner_routes import live_owner_set

_TIMESTAMP = "2026-09-16T00:00:00Z"


def _scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """An isolated archive root and its declared hook spool root.

    The hook topology resolves its root from the archive root, so a test that
    puts its carriers anywhere else is testing a directory no production
    acquisition would ever read.
    """

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(archive_root / "polylogue.toml"))
    return archive_root, archive_root / "hooks"


def _carriers(spool_root: Path) -> list[Path]:
    return sorted(hook_carrier_dir(spool_root).rglob("*.ndjson"))


# ── producing ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "event_type", "session_id", "expected_origin"),
    [
        ("claude-code", "PostToolUse", "claude-session", "claude-code-session"),
        ("codex", "PostToolUse", "codex-session", "codex-session"),
        ("hermes", "tool_finish", "hermes-session", "hermes-session"),
    ],
)
def test_one_appended_line_materializes_one_hook_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    event_type: str,
    session_id: str,
    expected_origin: str,
) -> None:
    """The whole route: one append, one carrier, one acquisition, one event.

    Anti-vacuity: publish the event without materializing it and the row count
    is zero; drop the origin mapping and the stored origin changes.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    carrier = append_hook_event(
        event_type=event_type,
        session_id=session_id,
        provider=provider,
        timestamp=_TIMESTAMP,
        payload={"tool_name": "exec"},
        root=spool_root,
        event_id="e" * 32,
    )
    assert carrier.parent.parent.name == provider
    assert carrier.read_bytes().endswith(b"\n")

    assert materialize_hook_carriers(archive_root) == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT origin, session_native_id, event_type FROM raw_hook_events").fetchall() == [
            (expected_origin, session_id, event_type)
        ]


@pytest.mark.asyncio
async def test_hook_derivation_publication_uses_the_daemon_writer_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon's production hook callback bridges only the publisher write.

    Anti-vacuity: omitting its stage admission makes the source-tier write
    fail the daemon's enforced writer lease instead of materializing the event.
    """
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.stage_admission import stage_write_admission
    from polylogue.daemon.convergence import _DerivationAdmission
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
    from polylogue.logging import propagate
    from polylogue.operations.hook_event_derivation import converge_hook_carriers, discover_pending_hook_carriers

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    append_hook_event(
        event_type="PostToolUse",
        session_id="coordinator-session",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={"tool_name": "exec"},
        root=spool_root,
        event_id="f" * 32,
    )
    assert acquire_hook_carriers(archive_root) == 1
    pending = discover_pending_hook_carriers(archive_root, 1)
    assert len(pending) == 1

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    admission = _DerivationAdmission(
        DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
        loop_thread_id=threading.get_ident(),
    )
    try:
        with stage_write_admission(admission.stage_write):
            submitted = compute.submit(
                propagate(
                    functools.partial(
                        converge_hook_carriers,
                        archive_root,
                        raw_ids=(pending[0][0],),
                        limit=1,
                    )
                ),
                admission_class="incremental-background",
            )
        report = await asyncio.wrap_future(submitted.future)
        assert report.done == 1
        with sqlite3.connect(archive_root / "source.db") as conn:
            assert conn.execute(
                "SELECT session_native_id FROM raw_hook_events WHERE session_native_id = ?",
                ("coordinator-session",),
            ).fetchone() == ("coordinator-session",)
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_hook_derivation_rechecks_binding_after_writer_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A competing publication while queued makes the stale replacement a no-op."""
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.core.stage_admission import stage_write_admission
    from polylogue.daemon.convergence import _DerivationAdmission
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
    from polylogue.logging import propagate
    from polylogue.operations.hook_event_derivation import discover_pending_hook_carriers
    from polylogue.sources.live.archive_open import _open_archive_for_live_write
    from polylogue.storage.derived.hook_events import HookEventsDerivation

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    append_hook_event(
        event_type="PostToolUse",
        session_id="binding-race-session",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={"tool_name": "exec"},
        root=spool_root,
        event_id="e" * 32,
    )
    assert acquire_hook_carriers(archive_root) == 1
    pending = discover_pending_hook_carriers(archive_root, 1)
    assert len(pending) == 1

    derivation = HookEventsDerivation(archive_root)
    frame = SimpleNamespace(
        archive_root=archive_root,
        recipe_version=lambda domain: derivation.recipe_version if domain == derivation.domain else None,
    )
    replacement = derivation.compute(frame, pending[0][0])
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    loop = asyncio.get_running_loop()
    writer_admission = _DerivationAdmission(
        DaemonWriteThreadBridge(coordinator, loop),
        loop_thread_id=threading.get_ident(),
    )
    archive_open_count = 0

    def admit_after_competing_publication(actor: str, work: Callable[[], bool]) -> bool:
        def publish_competitor() -> bool:
            nonlocal archive_open_count
            archive_open_count += 1
            store = _open_archive_for_live_write(archive_root)
            with store as archive:
                archive.write_hook_events_from_carrier(
                    carrier_source_id=replacement.identity.source_id,
                    carrier_relative_path=replacement.identity.relative_path,
                    carrier_role=replacement.identity.role,
                    carrier_blob_hash=bytes.fromhex(replacement.blob_hash),
                    carrier_source_path=replacement.source_path,
                    events=replacement.payload,
                    acquired_at_ms=replacement.acquired_at_ms,
                )
                archive.commit()
            return work()

        return writer_admission.stage_write(actor, publish_competitor)

    try:
        with stage_write_admission(admit_after_competing_publication):
            submitted = compute.submit(
                propagate(functools.partial(derivation.publish, frame, replacement)),
                admission_class="incremental-background",
            )
        assert await asyncio.wrap_future(submitted.future) is False
        assert archive_open_count == 1
        with sqlite3.connect(archive_root / "source.db") as conn:
            assert conn.execute(
                "SELECT session_native_id FROM raw_hook_events WHERE session_native_id = ?",
                ("binding-race-session",),
            ).fetchall() == [("binding-race-session",)]
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


def test_a_carrier_never_mints_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hook event is evidence within a session, never a session (polylogue-31r1).

    Anti-vacuity: route materialization through the session writer instead of
    ``write_hook_events_from_carrier`` and ``sessions`` gains a row per event.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(4):
        append_hook_event(
            event_type="PreToolUse",
            session_id="parent-session",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    assert materialize_hook_carriers(archive_root) == 4
    with sqlite3.connect(archive_root / "source.db") as conn:
        # One raw row for the carrier itself, none for any event.
        carriers = conn.execute(
            "SELECT COUNT(*) FROM raw_artifacts WHERE artifact_kind = 'hook_event_carrier'"
        ).fetchone()[0]
        assert carriers == 1
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == carriers


def test_concurrent_producers_never_interleave_a_line(tmp_path: Path) -> None:
    """200 concurrent producers append 200 whole, parseable lines.

    Threads share this process's pid and therefore its carrier, which is the
    hard case: the single ``O_APPEND`` write is what keeps one 8 KiB payload
    from landing inside another. Anti-vacuity: replace the single write with
    a seek-and-write, or with two writes for the body and the newline, and
    lines tear.
    """

    spool_root = tmp_path / "hooks"
    target = carrier_path(spool_root, "codex")
    target.parent.mkdir(parents=True, exist_ok=True)
    barrier = threading.Barrier(200)

    def produce(index: int) -> None:
        barrier.wait()
        append_carrier_line(
            target,
            validated_hook_record(
                {
                    "event_id": f"{index:032x}",
                    "event_type": "PostToolUse",
                    "session_id": f"session-{index}",
                    "timestamp": _TIMESTAMP,
                    "provider": "codex",
                    # Well over PIPE_BUF: the bound the retired design said an
                    # append-only journal could not honour.
                    "payload": {"tool_output_preview": "x" * 8192},
                }
            ),
        )

    threads = [threading.Thread(target=produce, args=(index,)) for index in range(200)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines, refusals = read_hook_carrier(target.read_bytes())
    assert refusals == ()
    assert len(lines) == 200
    assert sorted(str(line.record["event_id"]) for line in lines) == sorted(f"{index:032x}" for index in range(200))
    # Offsets are real byte positions, not ordinals.
    assert [line.byte_offset for line in lines] == sorted(line.byte_offset for line in lines)
    assert lines[-1].byte_offset + lines[-1].line_bytes == target.stat().st_size


def test_a_partially_written_tail_line_is_not_materialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unterminated tail is a write in flight, not a malformed record.

    Anti-vacuity: admit the tail and the next acquisition, which sees the line
    completed, materializes a second event for the same append.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    carrier = append_hook_event(
        event_type="Stop",
        session_id="s",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={},
        root=spool_root,
        event_id="a" * 32,
    )
    with carrier.open("ab") as handle:
        handle.write(b'{"event_id":"bbbb","event_ty')

    lines, refusals = read_hook_carrier(carrier.read_bytes())
    assert [line.record["event_id"] for line in lines] == ["a" * 32]
    assert refusals == ()


def test_a_malformed_line_does_not_strand_its_neighbours(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad line is a counted refusal; the carrier still materializes.

    Anti-vacuity: raise on the malformed line instead of counting it and the
    two good events never reach the archive; treat the carrier as permanently
    stale and the derivation re-computes it forever.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    carrier = append_hook_event(
        event_type="Stop",
        session_id="s",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={},
        root=spool_root,
        event_id="a" * 32,
    )
    with carrier.open("ab") as handle:
        handle.write(b"this is not json at all\n")
    append_hook_event(
        event_type="Stop",
        session_id="s",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={},
        root=spool_root,
        event_id="c" * 32,
    )

    lines, refusals = read_hook_carrier(carrier.read_bytes())
    assert len(lines) == 2
    assert [refusal.reason.split(":")[0] for refusal in refusals] == ["JSONDecodeError"]
    assert materialize_hook_carriers(archive_root) == 2
    # The carrier reaches a settled verdict rather than being re-offered.
    from polylogue.operations.hook_event_derivation import discover_pending_hook_carriers

    assert discover_pending_hook_carriers(archive_root, 16) == ()


def test_materialization_is_idempotent_across_repeated_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running the whole route publishes no second copy of anything.

    Every identity is content-derived -- the event id from the producer, the
    carrier coordinate from the line's byte offset -- so a replay writes the
    same rows. Anti-vacuity: key the coordinate on a line ordinal and a
    re-acquired carrier duplicates every event after the first.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(6):
        append_hook_event(
            event_type="PreToolUse",
            session_id="s",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={"sequence": index},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    assert materialize_hook_carriers(archive_root) == 6
    assert materialize_hook_carriers(archive_root) == 6
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM hook_event_carriers").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(DISTINCT blob_hash) FROM hook_event_carriers").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'hook_payload'").fetchone()[0] == 6


def test_an_appended_carrier_materializes_only_its_new_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A growing carrier is acquired and materialized incrementally.

    Anti-vacuity: drop the coordinate comparison from ``inspect`` and the
    second pass reports the carrier valid with its new events unmaterialized.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(3):
        append_hook_event(
            event_type="PreToolUse",
            session_id="s",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    assert materialize_hook_carriers(archive_root) == 3
    for index in range(3, 7):
        append_hook_event(
            event_type="PostToolUse",
            session_id="s",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    assert materialize_hook_carriers(archive_root) == 7
    assert len(_carriers(spool_root)) == 1


def test_grown_carrier_retains_only_an_append_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A carrier growth stores its tail once, under the carrier's physical chain.

    Anti-vacuity: route a grown carrier through ordinary artifact admission
    without revision binding and both rows are unclassified full captures,
    including a second blob whose size is the entire grown carrier.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(3):
        append_hook_event(
            event_type="PreToolUse",
            session_id="s",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={"sequence": index},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    assert acquire_hook_carriers(archive_root) == 1
    carrier = _carriers(spool_root)[0]
    initial_size = carrier.stat().st_size

    for index in range(3, 7):
        append_hook_event(
            event_type="PostToolUse",
            session_id="s",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={"sequence": index},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    grown_size = carrier.stat().st_size
    source = WatchSource(
        name="codex-hooks",
        root=hook_carrier_provider_dir("codex", spool_root),
        layout=export_drop_layout((".ndjson",)),
        source_id="primary-hook-spool:codex",
        role="primary-writable",
    )

    async def grow() -> Any:
        async with live_owner_set(archive_root) as owners:
            watcher = LiveWatcher(
                SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=archive_root / "index.db")),
                (source,),
                **owners.watcher_kwargs(),
            )
            return await watcher._ingest_files([carrier])

    metrics = asyncio.run(grow())
    assert (metrics.append_file_count, metrics.full_file_count) == (1, 0)

    with sqlite3.connect(archive_root / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT origin, source_path, blob_size, revision_kind,
                   append_start_offset, append_end_offset
            FROM raw_sessions
            WHERE source_path = ?
            ORDER BY acquired_at_ms, raw_id
            """,
            (str(carrier),),
        ).fetchall()

    assert rows == [
        ("codex-session", str(carrier), initial_size, "full", None, None),
        ("codex-session", str(carrier), grown_size - initial_size, "append", initial_size, grown_size),
    ], rows


def test_carrier_coordinates_are_byte_offsets_not_ordinals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The durable carrier coordinate names a byte position in a named file.

    Anti-vacuity: an ordinal coordinate makes these values 0..2 and a carrier
    that gains a line ahead of them re-resolves durable references onto other
    events.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(3):
        append_hook_event(
            event_type="PreToolUse",
            session_id="s",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    assert materialize_hook_carriers(archive_root) == 3
    carrier = _carriers(spool_root)[0]
    lines, _refusals = read_hook_carrier(carrier.read_bytes())
    relative = carrier.relative_to(hook_carrier_dir(spool_root)).as_posix()
    with sqlite3.connect(archive_root / "source.db") as conn:
        recorded = sorted(row[0] for row in conn.execute("SELECT relative_path FROM hook_event_carriers").fetchall())
        sources = {row[0] for row in conn.execute("SELECT DISTINCT source_id FROM hook_event_carriers").fetchall()}
    assert sources == {"primary-hook-spool"}
    assert recorded == sorted(f"{relative}#{line.byte_offset:012d}:hook:{line.record['event_id']}" for line in lines)


def test_events_written_as_files_again_are_never_acquired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The anti-vacuity condition for the whole redesign (k3ahm AC5).

    A file-per-event spool is no longer an ingest surface. If a producer
    regresses to writing ``pending/<day>/<id>.json``, nothing acquires it and
    no carrier appears -- which is what this asserts, so the assertion goes
    red the moment a second ingest surface is quietly restored.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    pending = spool_root / "pending" / "2026-09-16"
    pending.mkdir(parents=True)
    (pending / "regressed.json").write_text(
        json.dumps(
            {
                "event_id": "f" * 32,
                "event_type": "Stop",
                "session_id": "s",
                "timestamp": _TIMESTAMP,
                "provider": "codex",
                "payload": {},
                "observed_at_ms": 0,
            }
        ),
        encoding="utf-8",
    )

    assert materialize_hook_carriers(archive_root) == 0
    assert _carriers(spool_root) == []


# ── the one-shot legacy fold (polylogue-k8wv) ─────────────────────────────


def test_compact_carrier_census_streams_wide_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F141: compact counts every carrier without collecting a directory first.

    Anti-vacuity: restoring sorted(scandir(...)) requests a second entry before
    inspecting the first, and the instrumented real directory iterator refuses.
    """
    spool_root = tmp_path / "hooks"
    leaf = hook_carrier_dir(spool_root) / "codex" / "2026-09-16"
    leaf.mkdir(parents=True)
    for name in ("one.ndjson", "two.ndjson", "ignore.txt"):
        (leaf / name).write_bytes(b"")
    (leaf / "link.ndjson").symlink_to(leaf / "one.ndjson")
    real_scandir = os.scandir
    scans: list[TrackedScan] = []

    class TrackedEntry:
        def __init__(self, entry: os.DirEntry[str], scan: TrackedScan) -> None:
            self.entry = entry
            self.scan = scan
            self.name = entry.name
            self.path = entry.path

        def is_dir(self, *, follow_symlinks: bool = True) -> bool:
            self.scan.inspected = True
            return self.entry.is_dir(follow_symlinks=follow_symlinks)

        def is_file(self, *, follow_symlinks: bool = True) -> bool:
            self.scan.inspected = True
            return self.entry.is_file(follow_symlinks=follow_symlinks)

    class TrackedScan:
        def __init__(self) -> None:
            self.entries = real_scandir(leaf)
            self.inspected = True
            self.closed = False

        def __iter__(self) -> TrackedScan:
            return self

        def __next__(self) -> TrackedEntry:
            assert self.inspected, "carrier census accumulated entries before inspecting them"
            entry = next(self.entries)
            self.inspected = False
            return TrackedEntry(entry, self)

        def __enter__(self) -> TrackedScan:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.close()

        def close(self) -> None:
            self.entries.close()
            self.closed = True

    def tracked_scandir(path: str | os.PathLike[str]) -> object:
        if Path(path) != leaf:
            return real_scandir(path)
        scan = TrackedScan()
        scans.append(scan)
        return scan

    monkeypatch.setattr(os, "scandir", tracked_scandir)
    receipt = compact_legacy_spool(spool_root)

    assert receipt["carrier_scope"] == {"before": {"file_count": 2}, "after": {"file_count": 2}}
    assert len(scans) == 2
    assert all(scan.closed for scan in scans)


def test_compact_folds_the_retired_spool_into_carriers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--compact`` is the one bridge from the retired spool to the carriers.

    Anti-vacuity: retire the originals before the carriers are fsynced and an
    interrupted fold loses events; fold the journal mirrors and the archive
    gains a second identity for events it already holds.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    pending = spool_root / "pending" / "2026-09-14"
    pending.mkdir(parents=True)
    for index in range(20):
        (pending / f"{index:032x}.json").write_text(
            json.dumps(
                {
                    "event_id": f"{index:032x}",
                    "event_type": "PreToolUse",
                    "session_id": f"legacy-session-{index % 3}",
                    "timestamp": "2026-09-14T08:00:00Z",
                    "provider": "claude-code",
                    "payload": {"tool_name": "Bash"},
                }
            ),
            encoding="utf-8",
        )
    (pending / ".0123.json.tmpsuffix").write_text("{}", encoding="utf-8")
    (pending / "empty.json").write_text("", encoding="utf-8")
    (spool_root / "claude-code-some-session.jsonl").write_text("{}\n", encoding="utf-8")

    summary = compact_legacy_spool(spool_root)
    assert summary["folded"] == 20
    assert summary["refused"] == {
        "hidden atomic-write tempname is not a published envelope": 1,
        "per-session journal mirror is not an ingest surface (docs/hooks.md)": 1,
        "zero-byte file carries no record": 1,
    }
    # Folded envelopes are retired; refused ones stay exactly where they were.
    assert sorted(path.name for path in pending.iterdir()) == [".0123.json.tmpsuffix", "empty.json"]
    assert (spool_root / "claude-code-some-session.jsonl").exists()

    assert materialize_hook_carriers(archive_root) == 20
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(DISTINCT session_native_id) FROM raw_hook_events").fetchone()[0] == 3


def test_compact_quiesces_carrier_producer_and_defers_mid_drain_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producer racing the drain is blocked, then admitted on the next pass."""

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    pending = spool_root / "pending" / "2026-09-14"
    pending.mkdir(parents=True)
    (pending / f"{0:032x}.json").write_text(_envelope(0), encoding="utf-8")

    from polylogue.sources.hook_producer import _CompactionSink

    entered = threading.Event()
    release = threading.Event()
    original_append = _CompactionSink.append

    def pause_once(self: _CompactionSink, record: dict[str, object]) -> Path:
        entered.set()
        release.wait(timeout=5)
        return original_append(self, record)

    monkeypatch.setattr(_CompactionSink, "append", pause_once)
    result: dict[str, object] = {}

    drain = threading.Thread(target=lambda: result.update(compact_legacy_spool(spool_root)))
    drain.start()
    assert entered.wait(timeout=5)

    appended = threading.Event()

    def append_during_drain() -> None:
        append_hook_event(
            event_type="PostToolUse",
            session_id="live-session",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={},
            root=spool_root,
            event_id="f" * 32,
        )
        appended.set()

    producer = threading.Thread(target=append_during_drain)
    producer.start()
    # Anti-vacuity: restore the shared producer lock and this hook blocks
    # behind a large legacy drain instead of admitting the event promptly.
    assert appended.wait(timeout=0.1)
    release.set()
    drain.join(timeout=5)
    producer.join(timeout=5)

    assert result["carrier_compaction_serialized"] is True
    policy = result["carrier_producer_policy"]
    assert policy == "hook producers do not wait for the legacy drain lock"
    # Anti-vacuity: restoring the old sorted path array grows the receipt with
    # every carrier filename instead of keeping a fixed-size count.
    scope = result["carrier_scope"]
    assert isinstance(scope, dict)
    assert set(scope) == {"before", "after"}
    assert all(set(value) == {"file_count"} for value in scope.values())
    assert result["conservation_reconciliation"] == (
        "event_id basename; acknowledged day shard is destination metadata"
    )
    assert appended.is_set()
    assert materialize_hook_carriers(archive_root) == 2


def _envelope(index: int, *, session: str = "legacy-session") -> str:
    return json.dumps(
        {
            "event_id": f"{index:032x}",
            "event_type": "PreToolUse",
            "session_id": session,
            "timestamp": "2026-09-14T08:00:00Z",
            "provider": "claude-code",
            "payload": {"tool_name": "Bash"},
        }
    )


def test_compact_checkpoints_so_an_interrupted_fold_repeats_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted fold re-folds one checkpoint, not the whole spool.

    The legacy spool holds ~716k envelopes. Retiring only after the final
    carrier fsync means any interruption -- a signal, an OOM, a reboot --
    retires nothing, so the next run re-folds every envelope and appends a
    second full copy of every event to the carriers, on the filesystem that
    also has to hold the rebuild's headroom.

    Anti-vacuity: move the retirement back to a single pass after the loop and
    the interrupted run below retires nothing, so the re-fold folds all 30
    envelopes again and the carriers hold 55 lines instead of 35.
    """

    from polylogue.sources.hook_producer import _CompactionSink

    _archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    pending = spool_root / "pending" / "2026-09-14"
    pending.mkdir(parents=True)
    for index in range(30):
        (pending / f"{index:032x}.json").write_text(_envelope(index), encoding="utf-8")

    real_append = _CompactionSink.append
    appended = 0

    def interrupt_after_25(self: _CompactionSink, record: dict[str, object]) -> Path:
        nonlocal appended
        appended += 1
        if appended > 25:
            raise KeyboardInterrupt("simulated interruption mid-batch")
        return real_append(self, record)

    monkeypatch.setattr(_CompactionSink, "append", interrupt_after_25)
    with pytest.raises(KeyboardInterrupt):
        compact_legacy_spool(spool_root, checkpoint_events=10)
    monkeypatch.undo()

    # Two checkpoints completed; the third batch never reached one.
    assert len(list(pending.glob("*.json"))) == 10

    resumed = compact_legacy_spool(spool_root, checkpoint_events=10)
    assert resumed["folded"] == 10
    assert resumed["retired"] == 10
    assert list(pending.glob("*.json")) == []

    # 25 lines from the interrupted run, 10 from the resumed one: the five
    # envelopes folded into a carrier but not yet retired are the one
    # checkpoint's worth of duplicate work this bound allows. The drain
    # deduplicates them by ``event_id``; what is bounded here is the carrier
    # bytes a repeated fold writes to the filesystem.
    lines = sum(len(carrier.read_bytes().splitlines()) for carrier in _carriers(spool_root))
    assert lines == 35


def test_compact_accounts_for_every_member_by_count_and_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """k8wv AC4/AC6: every inspected member lands in exactly one outcome.

    Conservation against the frozen manifest closes on counts *and* bytes, so
    the fold reports both. A member that silently vanishes from the tally --
    which is what a non-regular spool entry used to do -- makes the identity
    below false.

    Anti-vacuity: drop the ``is not a regular file`` refusal and the dangling
    symlink stops being counted, so ``scanned`` exceeds
    ``folded + sum(refused)``.
    """

    _archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    pending = spool_root / "pending" / "2026-09-14"
    pending.mkdir(parents=True)
    for index in range(5):
        (pending / f"{index:032x}.json").write_text(_envelope(index), encoding="utf-8")
    (pending / ".ffff.json.tmpsuffix").write_text("{}", encoding="utf-8")
    (pending / "empty.json").write_text("", encoding="utf-8")
    (pending / "notes.txt").write_text("not a spool member", encoding="utf-8")
    (pending / "dangling").symlink_to(pending / "gone.json")
    (spool_root / "claude-code-some-session.jsonl").write_text("{}\n", encoding="utf-8")

    summary = compact_legacy_spool(spool_root)

    refused_counts: dict[str, int] = summary["refused"]  # type: ignore[assignment]
    refused_sizes: dict[str, int] = summary["refused_bytes"]  # type: ignore[assignment]
    folded = int(summary["folded"])  # type: ignore[call-overload]
    folded_bytes = int(summary["folded_bytes"])  # type: ignore[call-overload]
    assert folded == 5
    assert summary["scanned"] == folded + sum(refused_counts.values()) == 10
    assert summary["scanned_bytes"] == folded_bytes + sum(refused_sizes.values())
    assert refused_counts["spool member is not a regular file"] == 1
    assert set(refused_sizes) == set(refused_counts)
    assert folded_bytes > 0


def test_compact_bounds_one_carrier_and_stays_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fold splits at its size bound and a re-run folds nothing twice."""

    _archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    pending = spool_root / "pending" / "2026-09-14"
    pending.mkdir(parents=True)
    for index in range(30):
        (pending / f"{index:032x}.json").write_text(
            json.dumps(
                {
                    "event_id": f"{index:032x}",
                    "event_type": "PreToolUse",
                    "session_id": "legacy-session",
                    "timestamp": "2026-09-14T08:00:00Z",
                    "provider": "claude-code",
                    "payload": {},
                }
            ),
            encoding="utf-8",
        )

    summary = compact_legacy_spool(spool_root, max_bytes=400)
    assert summary["folded"] == 30
    assert len(summary["carriers"]) > 1  # type: ignore[arg-type]
    for carrier in _carriers(spool_root):
        assert carrier.stat().st_size <= 400 + 256

    again = compact_legacy_spool(spool_root)
    assert again["folded"] == 0


@pytest.mark.parametrize("events", [10_000])
def test_compact_and_materialize_ten_thousand_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: int
) -> None:
    """k3ahm AC3: 10,000 events compact and materialize at well over 2,000/s.

    The timed window is exactly compaction and materialization. Archive
    bootstrap and carrier acquisition are fixed per-test costs, not per-event
    throughput, so they run outside it. The floor fails on a regression of
    the route's shape -- a return to per-event blob publication, per-event
    commits or per-event statements -- rather than on a busy machine.
    Anti-vacuity: restore the per-event write and this is red by two orders
    of magnitude.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    pending = spool_root / "pending" / "2026-09-14"
    pending.mkdir(parents=True)
    for index in range(events):
        (pending / f"{index:032x}.json").write_text(
            json.dumps(
                {
                    "event_id": f"{index:032x}",
                    "event_type": "PostToolUse" if index % 2 else "PreToolUse",
                    "session_id": f"legacy-session-{index // 40}",
                    "timestamp": "2026-09-14T08:00:00Z",
                    "provider": "claude-code",
                    "payload": {"tool_name": "Bash", "tool_input": {"command": "true"}},
                }
            ),
            encoding="utf-8",
        )

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(archive_root)
    started = time.perf_counter()
    assert compact_legacy_spool(spool_root)["folded"] == events
    compact_elapsed = time.perf_counter() - started
    assert acquire_hook_carriers(archive_root) >= 1
    started = time.perf_counter()
    assert materialize_acquired_hook_carriers(archive_root) == events
    elapsed = compact_elapsed + (time.perf_counter() - started)
    assert events / elapsed > 2_000, f"{events / elapsed:.0f} events/s"


# ── validation, unchanged semantics ───────────────────────────────────────


def test_hermes_per_turn_end_and_durable_finalize_remain_distinct_event_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fs1.7 AC: on_session_end (per turn) is never conflated with on_session_finalize."""

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for event_id, event_type in (
        ("turn-end-1", PER_TURN_END),
        ("turn-end-2", PER_TURN_END),
        ("finalize-1", DURABLE_FINALIZE),
    ):
        append_hook_event(
            event_id=event_id,
            provider="hermes",
            event_type=event_type,
            session_id="hermes-session-1",
            timestamp=_TIMESTAMP,
            payload={},
            root=spool_root,
        )
    assert materialize_hook_carriers(archive_root) == 3
    with sqlite3.connect(archive_root / "source.db") as conn:
        rows = conn.execute(
            "SELECT event_type, COUNT(*) FROM raw_hook_events WHERE session_native_id = 'hermes-session-1' "
            "GROUP BY event_type ORDER BY event_type"
        ).fetchall()
    assert rows == [(PER_TURN_END, 2), (DURABLE_FINALIZE, 1)]


def test_hook_payload_rejects_duplicated_transcript_text(tmp_path: Path) -> None:
    """fs1.7 AC: event bodies carry ids/hashes/timings/outcomes, never a transcript."""

    with pytest.raises(HookSpoolRecordError, match="duplicated transcript"):
        append_hook_event(
            event_id="oversized-payload",
            provider="hermes",
            event_type="tool_finish",
            session_id="hermes-session-1",
            timestamp=_TIMESTAMP,
            payload={"text": "x" * 5000},
            root=tmp_path / "hooks",
        )


def test_hook_payload_allows_short_evidence_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary short ids/summaries are not mistaken for duplicated transcripts."""

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    append_hook_event(
        event_id="short-payload",
        provider="hermes",
        event_type="tool_finish",
        session_id="hermes-session-1",
        timestamp=_TIMESTAMP,
        payload={"tool_call_id": "call-1", "content": "exit 0"},
        root=spool_root,
    )
    assert materialize_hook_carriers(archive_root) == 1


def test_a_camelcase_envelope_is_a_spelling_not_a_malformation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored envelope is canonical snake_case; the payload stays verbatim.

    The payload is the harness's own evidence -- normalizing it would destroy
    the record of which generation was emitted.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    payload = {"toolName": "Bash", "toolInput": {"command": "true"}}
    carrier = carrier_path(spool_root, "claude-code")
    carrier.parent.mkdir(parents=True, exist_ok=True)
    append_carrier_line(
        carrier,
        validated_hook_record(
            {
                "eventId": "d" * 32,
                "eventType": "PreToolUse",
                "sessionId": "camel-session",
                "timestamp": _TIMESTAMP,
                "provider": "claude-code",
                "payload": payload,
            }
        ),
    )
    assert materialize_hook_carriers(archive_root) == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        stored = conn.execute("SELECT hook_event_id, payload_json FROM raw_hook_events").fetchone()
    assert stored[0] == f"hook:{'d' * 32}"
    assert json.loads(stored[1])["payload"] == payload


def test_a_journal_record_is_refused_by_the_validator() -> None:
    """The per-session journal envelope is not an ingest surface.

    It carries no event identity, and idempotence is keyed on one --
    ``hook:<event_id>`` is the source-tier key a replay reuses. Deriving an id
    from the content would mint a second identity for events the archive
    already holds under their producer's own. ``docs/hooks.md`` carries the
    census.

    Anti-vacuity: accepting the journal shape -- by making ``event_id``
    optional or deriving one -- makes this red, which is the point. Reversing
    the verdict is a decision, not a refactoring.
    """

    with pytest.raises(HookSpoolRecordError, match="no event_id"):
        validated_hook_record(
            {
                "event_type": "PreToolUse",
                "session_id": "0fe73aeb-5d82-4124-a24a-d764a94fbf05",
                "timestamp": "2026-08-11T20:00:23Z",
                "provider": "claude-code",
                "payload": {"tool_name": "Bash"},
            }
        )


def test_an_unmapped_provider_raises_rather_than_defaulting_to_codex() -> None:
    """Defense in depth behind ``validated_record``'s provider gate.

    A future drift between the supported-provider set and the origin mapping
    must raise, not silently misclassify an unknown provider as Codex.
    """

    from polylogue.sources.hooks import hook_event_origin

    with pytest.raises(HookSpoolRecordError, match="no origin mapping"):
        hook_event_origin("gemini-cli")


def test_find_carrier_event_reads_only_the_carrier_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``carriers/`` is a destination; ``acknowledged/`` is a fold record."""

    _archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    append_hook_event(
        event_type="Stop",
        session_id="s",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={},
        root=spool_root,
        event_id="a" * 32,
    )
    acknowledged = spool_root / "acknowledged" / "2026-09-14"
    acknowledged.mkdir(parents=True)
    (acknowledged / "b.json").write_text(
        json.dumps(
            {
                "event_id": "b" * 32,
                "event_type": "Stop",
                "session_id": "s",
                "timestamp": _TIMESTAMP,
                "provider": "codex",
                "payload": {},
            }
        ),
        encoding="utf-8",
    )

    assert find_carrier_event(spool_root, "a" * 32) is not None
    assert find_carrier_event(spool_root, "b" * 32) is None


# ── the installed and published producers ─────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "session_id"),
    [("claude-code", "claude-session"), ("codex", "codex-session")],
)
def test_hook_entrypoint_appends_and_materializes_configured_runtime_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    session_id: str,
) -> None:
    """The installed hook command (which bakes ``--sidecar-dir``, polylogue-o7hx)
    reaches the durable source-tier receipt path."""

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.stdin", StringIO(f'{{"session_id":"{session_id}","tool_name":"exec"}}'))

    assert hook_producer_main(["PostToolUse", "--provider", provider, "--sidecar-dir", str(spool_root)]) == 0

    carriers = _carriers(spool_root)
    assert len(carriers) == 1
    record = json.loads(carriers[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["provider"] == provider
    assert record["session_id"] == session_id
    assert record["event_type"] == "PostToolUse"
    assert materialize_hook_carriers(archive_root) == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT session_native_id FROM raw_hook_events").fetchone() == (session_id,)


@pytest.mark.parametrize(
    ("command", "extra_env"),
    [
        (
            (sys.executable, "-m", "polylogue_hooks.cli", "PostToolUse", "--provider", "claude-code"),
            {"PYTHONPATH": str(Path("packaging/polylogue-hooks/src").resolve())},
        ),
        ((str(Path("contrib/polylogue-hook").resolve()), "PostToolUse", "--provider", "codex"), {}),
    ],
)
def test_published_hook_adapters_fall_back_to_the_archive_root_env_var(
    tmp_path: Path,
    command: tuple[str, ...],
    extra_env: dict[str, str],
) -> None:
    """Without an explicit ``--sidecar-dir``, the standalone adapters still
    derive their carrier root from ``POLYLOGUE_ARCHIVE_ROOT`` -- the one env
    var already required to isolate a scratch daemon, not a separate
    hook-specific knob (polylogue-o7hx)."""

    scratch_archive_root = tmp_path / "scratch-archive-root"
    subprocess_tmp = tmp_path / "tmp"
    subprocess_tmp.mkdir()
    environment = (
        os.environ | extra_env | {"TMPDIR": str(subprocess_tmp), "POLYLOGUE_ARCHIVE_ROOT": str(scratch_archive_root)}
    )
    result = subprocess.run(
        command,
        input='{"session_id":"external-session","tool_name":"exec"}',
        text=True,
        env=environment,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(_carriers(scratch_archive_root / "hooks")) == 1


@pytest.mark.parametrize(
    ("command", "extra_env"),
    [
        (
            (sys.executable, "-m", "polylogue_hooks.cli", "PostToolUse", "--provider", "claude-code"),
            {"PYTHONPATH": str(Path("packaging/polylogue-hooks/src").resolve())},
        ),
        ((str(Path("contrib/polylogue-hook").resolve()), "PostToolUse", "--provider", "codex"), {}),
        (
            (sys.executable, "-m", "polylogue_hooks.cli", "tool_finish", "--provider", "hermes"),
            {"PYTHONPATH": str(Path("packaging/polylogue-hooks/src").resolve())},
        ),
        ((str(Path("contrib/polylogue-hook").resolve()), "tool_finish", "--provider", "hermes"), {}),
    ],
)
def test_published_hook_adapters_append_then_materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
    extra_env: dict[str, str],
) -> None:
    """Every documented non-bundled executable (incl. the Hermes prototype)
    reaches the durable receipt path via ``--sidecar-dir``, the concrete
    resolved path an installer bakes in at install time (polylogue-o7hx)."""

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    subprocess_tmp = tmp_path / "tmp"
    subprocess_tmp.mkdir()
    environment = os.environ | extra_env | {"TMPDIR": str(subprocess_tmp)}
    result = subprocess.run(
        (*command, "--sidecar-dir", str(spool_root)),
        input='{"session_id":"external-session","tool_name":"exec"}',
        text=True,
        env=environment,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(_carriers(spool_root)) == 1
    assert materialize_hook_carriers(archive_root) == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT session_native_id FROM raw_hook_events").fetchone() == ("external-session",)


@pytest.mark.parametrize(
    ("command", "extra_env"),
    [
        (
            (sys.executable, "-m", "polylogue_hooks.cli", "tool_finish", "--provider", "hermes"),
            {"PYTHONPATH": str(Path("packaging/polylogue-hooks/src").resolve())},
        ),
        ((str(Path("contrib/polylogue-hook").resolve()), "tool_finish", "--provider", "hermes"), {}),
    ],
)
def test_published_hook_adapters_refuse_duplicated_transcript_payloads(
    tmp_path: Path,
    command: tuple[str, ...],
    extra_env: dict[str, str],
) -> None:
    """The standalone (non-bundled) producers enforce the same rule."""

    spool_root = tmp_path / "hooks"
    subprocess_tmp = tmp_path / "tmp"
    subprocess_tmp.mkdir()
    environment = os.environ | extra_env | {"TMPDIR": str(subprocess_tmp)}
    result = subprocess.run(
        (*command, "--sidecar-dir", str(spool_root)),
        input=json.dumps({"session_id": "external-session", "text": "x" * 5000}),
        text=True,
        env=environment,
        check=False,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "duplicated transcript" in result.stderr
    assert _carriers(spool_root) == []


def test_transcript_duplication_policy_stays_in_sync_across_hook_producers() -> None:
    """The no-duplicated-transcript field list/threshold is copied, not shared, across three
    independent runtime boundaries (main package, standalone pip package, dependency-free bash
    script) -- deliberately, since ``packaging/polylogue-hooks`` documents itself as having *no
    dependency on the main polylogue distribution* and ``contrib/polylogue-hook`` is not Python at
    all, so a single importable helper cannot span all three. This test is the drift guard that
    duplication-without-sharing still needs: if ``polylogue.sources.hook_producer.TRANSCRIPT_LIKE_KEYS``
    (or its threshold) ever changes without updating the two mirrored copies, this fails loudly
    instead of the gap only surfacing if someone happens to craft a payload using exactly the
    added/removed field name.
    """

    from polylogue.sources.hook_producer import MAX_TRANSCRIPT_LIKE_FIELD_CHARS, TRANSCRIPT_LIKE_KEYS

    packaging_src = Path("packaging/polylogue-hooks/src").resolve()
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, polylogue_hooks.cli as m; "
            "print(json.dumps({'keys': list(m._TRANSCRIPT_LIKE_KEYS), "
            "'threshold': m._MAX_TRANSCRIPT_LIKE_FIELD_CHARS}))",
        ],
        env=os.environ | {"PYTHONPATH": str(packaging_src)},
        check=True,
        capture_output=True,
        text=True,
    )
    packaging_policy = json.loads(probe.stdout)
    assert packaging_policy["keys"] == list(TRANSCRIPT_LIKE_KEYS)
    assert packaging_policy["threshold"] == MAX_TRANSCRIPT_LIKE_FIELD_CHARS

    contrib_source = Path("contrib/polylogue-hook").resolve().read_text(encoding="utf-8")
    keys_match = re.search(r"for _key in \(([^)]*)\):", contrib_source)
    threshold_match = re.search(r"if _size > (\d+):", contrib_source)
    assert keys_match is not None, "contrib/polylogue-hook: transcript-key loop not found"
    assert threshold_match is not None, "contrib/polylogue-hook: transcript threshold not found"
    contrib_keys = [item.strip().strip('"') for item in keys_match.group(1).split(",") if item.strip()]
    assert contrib_keys == list(TRANSCRIPT_LIKE_KEYS)
    assert int(threshold_match.group(1)) == MAX_TRANSCRIPT_LIKE_FIELD_CHARS


def test_the_carrier_publish_path_is_the_only_one_left() -> None:
    """k3ahm AC4: the drain route is gone from the tree, grep-provable.

    Anti-vacuity: reintroduce any of these names and this is red. The list is
    the deletion ledger, held where a reviewer reads it.
    """

    retired = (
        "drain_hook_event_spool",
        "HookSpoolDrainResult",
        "enqueue_hook_event",
        "pending_hook_spool_dir",
        "acknowledged_hook_spool_dir",
        "read_hook_spool_record",
        "hook_spool_pending_depth",
        "hook_spool_has_pending_events",
        "hook_watch_sources",
        "HookSpoolIntakeAdapter",
        "_drain_hook_spools",
        "_iter_pending_event_paths",
        "_prune_empty_shards",
        "_HOOK_SPOOL_DRAIN_BATCH_LIMIT",
    )
    offenders: list[str] = []
    for path in Path("polylogue").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path}: {name}" for name in retired if name in text)
    assert offenders == []


def test_the_carrier_route_admits_before_it_materializes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Acquisition and materialization are two steps, in that order.

    Anti-vacuity: materialize at capture time again -- the defect this whole
    change removes -- and the archive holds events before any carrier has been
    acquired, making the intermediate count non-zero.
    """

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(3):
        append_hook_event(
            event_type="Stop",
            session_id="s",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={},
            root=spool_root,
            event_id=f"{index:032x}",
        )
    assert acquire_hook_carriers(archive_root) == 1
    assert hook_event_count(archive_root) == 0
    assert materialize_hook_carriers(archive_root) == 3


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("fault_depth", [None, 0, 1, 2, 3])
def test_compaction_settles_every_carrier_ancestor_before_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool, fault_depth: int | None
) -> None:
    """A failed ancestor barrier retains the original, including on a retry."""
    import stat

    root = tmp_path / "spool"
    pending = root / "pending" / "2026-09-14"
    pending.mkdir(parents=True)
    original = pending / "event.json"
    original.write_text(_envelope(0), encoding="utf-8")
    directories = [
        root,
        root / "carriers",
        root / "carriers" / "claude-code",
        root / "carriers" / "claude-code" / "2026-09-14",
    ]
    if preexisting:
        directories[-1].mkdir(parents=True)
    events: list[Path | str] = []
    real_sync, real_replace = os.fsync, os.replace
    fault = directories[fault_depth] if fault_depth is not None else None

    def sync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory = Path(os.readlink(f"/proc/self/fd/{fd}"))
            assert original.exists()
            events.append(directory)
            if directory == fault:
                raise OSError("injected directory barrier failure")
        else:
            events.append("file")
        real_sync(fd)

    def replace(source: object, destination: object) -> None:
        if Path(source) == original:  # type: ignore[arg-type]
            assert events == ["file", *directories]
            events.append("retire")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    if fault is not None:
        with pytest.raises(OSError, match="injected directory barrier"):
            compact_legacy_spool(root)
        assert original.exists()
        assert "retire" not in events
        fault = None
        events.clear()
    result = compact_legacy_spool(root)
    assert result["retired"] == 1
    assert not original.exists()
    assert events == ["file", *directories, "retire"]


def test_ordinary_hook_emission_does_not_synchronize_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.hook_producer import append_event

    def forbidden(_fd: int) -> None:
        pytest.fail("ordinary hook emission acquired a durability barrier")

    monkeypatch.setattr(os, "fsync", forbidden)
    record = json.loads(_envelope(0))
    append_event(root=str(tmp_path), **record)
    assert len(list(tmp_path.rglob("*.ndjson"))) == 1


def test_recreated_carrier_retains_new_event_at_the_old_byte_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing a day/PID path cannot certify a different event from its old offset."""
    from polylogue.storage.hook_event_authority import census_hook_event_authority

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    arguments: dict[str, Any] = {
        "event_type": "PostToolUse",
        "session_id": "replacement-session",
        "provider": "codex",
        "timestamp": _TIMESTAMP,
        "root": spool_root,
    }
    path = append_hook_event(**arguments, payload={"text": "old"}, event_id="1" * 32)
    assert materialize_hook_carriers(archive_root) == 1
    old_bytes = path.read_bytes()
    path.unlink()
    replacement = append_hook_event(**arguments, payload={"text": "new"}, event_id="2" * 32)
    assert replacement == path
    assert len(path.read_bytes()) == len(old_bytes)
    assert materialize_hook_carriers(archive_root) == 2
    # A fresh route owner repeats acquisition/discovery/materialization.
    assert materialize_hook_carriers(archive_root) == 2
    with sqlite3.connect(archive_root / "source.db") as conn:
        hashes = conn.execute(
            "SELECT DISTINCT hex(blob_hash) FROM raw_sessions WHERE source_path = ?", (str(path),)
        ).fetchall()
        assert len(hashes) == 2
        assert conn.execute("SELECT COUNT(*) FROM hook_event_carriers").fetchone() == (2,)
        assert {row[0] for row in conn.execute("SELECT hook_event_id FROM raw_hook_events")} == {
            "hook:" + "1" * 32,
            "hook:" + "2" * 32,
        }
        assert census_hook_event_authority(conn).issues == ()
    from polylogue.storage.blob_store import BlobStore

    store = BlobStore(archive_root / "blob")
    assert {store.read_all(row[0].lower()) for row in hashes} == {old_bytes, path.read_bytes()}


def test_pending_hook_discovery_passes_complete_pages_and_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid first page is not an empty pending population."""
    from polylogue.operations.hook_event_derivation import converge_hook_carriers, discover_pending_hook_carriers

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(4):
        path = append_hook_event(
            event_type="PostToolUse",
            session_id="paged-session",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={"sequence": index},
            root=spool_root,
            event_id=f"{index:032x}",
        )
        # Four independent producer carriers, all produced by the normal encoder.
        path.rename(path.with_name(f"neutral-{index}.ndjson"))
    assert acquire_hook_carriers(archive_root) == 4
    all_pending = discover_pending_hook_carriers(archive_root, 4)
    assert len(all_pending) == 4
    for key, _cost in all_pending[:3]:
        assert converge_hook_carriers(archive_root, raw_ids=(key,), limit=1).done == 1
    assert discover_pending_hook_carriers(archive_root, 1) == all_pending[3:]
    assert materialize_acquired_hook_carriers(archive_root) == 4
    assert discover_pending_hook_carriers(archive_root, 1) == ()
    assert materialize_acquired_hook_carriers(archive_root) == 4


@pytest.mark.parametrize("materialize_prefix", [True, False])
def test_cancelled_hook_discovery_leaves_later_pending_for_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, materialize_prefix: bool
) -> None:
    """Cancellation between completed pages cannot acknowledge later work."""
    from polylogue.core.compute import DaemonOperationCancelled
    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.operations.hook_event_derivation import converge_hook_carriers, discover_pending_hook_carriers
    from polylogue.storage.derived.hook_events import HookEventsDerivation

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    for index in range(2):
        path = append_hook_event(
            event_type="PostToolUse",
            session_id="cancelled-discovery",
            provider="codex",
            timestamp=_TIMESTAMP,
            payload={"sequence": index},
            root=spool_root,
            event_id=f"{index:032x}",
        )
        path.rename(path.with_name(f"neutral-{index}.ndjson"))
    assert acquire_hook_carriers(archive_root) == 2
    pending = discover_pending_hook_carriers(archive_root, 2)
    if materialize_prefix:
        assert converge_hook_carriers(archive_root, raw_ids=(pending[0][0],), limit=1).done == 1
    cancelled = threading.Event()
    original = HookEventsDerivation.inspect

    def cancel_after_inspection(self: Any, frame: object, keys: Any) -> Any:
        result = original(self, frame, keys)
        cancelled.set()
        return result

    token = compute_cancel.set(cancelled)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(HookEventsDerivation, "inspect", cancel_after_inspection)
            with pytest.raises(DaemonOperationCancelled):
                discover_pending_hook_carriers(archive_root, 1)
    finally:
        compute_cancel.reset(token)
    assert hook_event_count(archive_root) == int(materialize_prefix)
    assert discover_pending_hook_carriers(archive_root, 1) == (pending[1:] if materialize_prefix else pending[:1])
    assert materialize_acquired_hook_carriers(archive_root) == 2


def test_hook_identity_with_changed_payload_remains_a_reported_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Digest inspection cannot certify a reused identity with changed content."""
    from polylogue.operations.hook_event_derivation import converge_hook_carriers, discover_pending_hook_carriers

    archive_root, spool_root = _scratch(tmp_path, monkeypatch)
    path = append_hook_event(
        event_type="PostToolUse",
        session_id="immutable-event",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={"text": "old"},
        root=spool_root,
        event_id="e" * 32,
    )
    assert materialize_hook_carriers(archive_root) == 1
    path.unlink()
    append_hook_event(
        event_type="PostToolUse",
        session_id="immutable-event",
        provider="codex",
        timestamp=_TIMESTAMP,
        payload={"text": "new"},
        root=spool_root,
        event_id="e" * 32,
    )
    assert acquire_hook_carriers(archive_root) == 1
    pending = discover_pending_hook_carriers(archive_root, 1)
    assert len(pending) == 1
    report = converge_hook_carriers(archive_root, raw_ids=(pending[0][0],), limit=1)
    assert report.failed == 1 and report.done == 0
    assert discover_pending_hook_carriers(archive_root, 1) == pending
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)
        rows = conn.execute("SELECT payload_json FROM raw_hook_events").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0])["payload"]["text"] == "old"
