"""Production-route tests for append-only hook-event carriers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

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
from tests.infra.hook_carriers import acquire_hook_carriers, hook_event_count, materialize_hook_carriers

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
        suffixes=(".ndjson",),
        source_id="primary-hook-spool:codex",
        role="primary-writable",
    )
    watcher = LiveWatcher(
        SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=archive_root / "index.db")),
        (source,),
    )
    metrics = asyncio.run(watcher._ingest_files([carrier]))
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
    ]


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
    assert recorded == sorted(f"{relative}#{line.byte_offset:012d}" for line in lines)


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
    """k3ahm AC3: 10,000 folded events materialize at well over 2,000/s.

    The threshold is deliberately far below the measured rate (about 6,000/s
    on the development workstation for the whole acquire+materialize route)
    so this fails on a regression of the route's shape -- a return to
    per-event blob publication or per-event commits -- rather than on a busy
    machine. Anti-vacuity: restore the per-event write and this is red by two
    orders of magnitude.
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

    assert compact_legacy_spool(spool_root)["folded"] == events
    started = time.perf_counter()
    assert materialize_hook_carriers(archive_root) == events
    elapsed = time.perf_counter() - started
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
