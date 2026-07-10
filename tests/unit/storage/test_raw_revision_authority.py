from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.archive.revision_authority import (
    HistoricalRawRevision,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    classify_historical_full_revisions,
)
from polylogue.core.enums import Origin, Provider
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch_support import _AppendPlan
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session


def test_historical_full_classifier_proves_unique_prefix_chain_independent_of_order() -> None:
    revisions = [
        HistoricalRawRevision("middle", b"one\ntwo\n"),
        HistoricalRawRevision("newest", b"one\ntwo\nthree\n"),
        HistoricalRawRevision("oldest", b"one\n"),
    ]

    def normalized(items: list[HistoricalRawRevision]) -> set[tuple[str, str | None, RawRevisionAuthority]]:
        return {
            (decision.raw_id, decision.predecessor_raw_id, decision.authority)
            for decision in classify_historical_full_revisions(items)
        }

    assert (
        normalized(revisions)
        == normalized(list(reversed(revisions)))
        == {
            ("oldest", None, RawRevisionAuthority.BYTE_PROVEN),
            ("middle", "oldest", RawRevisionAuthority.BYTE_PROVEN),
            ("newest", "middle", RawRevisionAuthority.BYTE_PROVEN),
        }
    )


@pytest.mark.parametrize("payloads", [[b"same", b"same"], [b"left", b"right"], [b"root", b"root-left", b"root-right"]])
def test_historical_classifier_quarantines_unprovable_authority(payloads: list[bytes]) -> None:
    decisions = classify_historical_full_revisions(
        [HistoricalRawRevision(f"raw-{index}", payload) for index, payload in enumerate(payloads)]
    )
    assert decisions
    assert {decision.authority for decision in decisions} == {RawRevisionAuthority.QUARANTINED}


def test_append_envelope_requires_baseline_and_exact_forward_offsets() -> None:
    with pytest.raises(ValueError, match="baseline and offsets"):
        RawRevisionEnvelope("codex:session", RawRevisionKind.APPEND, "rev-2", 2)


def test_source_writer_persists_typed_revision_envelope() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SOURCE_DDL)
    envelope = RawRevisionEnvelope(
        "codex:session-1",
        RawRevisionKind.APPEND,
        "sha256:revision-2",
        2,
        predecessor_raw_id="raw-full",
        baseline_raw_id="raw-full",
        append_start_offset=100,
        append_end_offset=150,
    )
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/capture/session.jsonl",
        source_index=-1,
        payload=b"append bytes",
        acquired_at_ms=10,
        revision=envelope,
    )
    assert conn.execute(
        """SELECT logical_source_key, revision_kind, source_revision, predecessor_raw_id,
                  baseline_raw_id, append_start_offset, append_end_offset,
                  acquisition_generation, revision_authority
           FROM raw_sessions WHERE raw_id = ?""",
        (raw_id,),
    ).fetchone() == ("codex:session-1", "append", "sha256:revision-2", "raw-full", "raw-full", 100, 150, 2, "asserted")


def test_unenveloped_raw_write_is_quarantined() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SOURCE_DDL)
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/capture/legacy.jsonl",
        source_index=0,
        payload=b"legacy",
        acquired_at_ms=10,
    )
    assert conn.execute(
        "SELECT revision_kind, revision_authority FROM raw_sessions WHERE raw_id = ?", (raw_id,)
    ).fetchone() == ("unknown", "quarantined")


def test_live_append_acquisition_binds_exact_offsets_to_authoritative_baseline(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    full_payload = b"x" * 100
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=full_payload,
            source_path=str(tmp_path / "session.jsonl"),
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            baseline_raw_id,
            RawRevisionEnvelope("codex:session-1", RawRevisionKind.FULL, "full-revision", 1),
        )

    append_payload = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"new"}]}}\n'
    )
    path = tmp_path / "session.jsonl"
    path.write_bytes(full_payload + append_payload)
    stat = path.stat()
    plan = _AppendPlan(
        path=path,
        source_name="codex",
        start_offset=len(full_payload),
        last_complete_newline=stat.st_size,
        stat_size=stat.st_size,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        payload=append_payload,
        payload_hash="append-hash",
        cursor_fingerprint="full-revision",
        bytes_read=len(append_payload),
        source_generation=2,
    )
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    owner = SimpleNamespace(
        _cursor=cursor,
        _polylogue=SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path)),
    )

    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.succeeded == [plan]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """SELECT predecessor_raw_id, baseline_raw_id, append_start_offset,
                      append_end_offset, acquisition_generation, revision_authority
               FROM raw_sessions WHERE revision_kind = 'append'"""
        ).fetchone()
    assert row == (baseline_raw_id, baseline_raw_id, len(full_payload), stat.st_size, 2, "asserted")
