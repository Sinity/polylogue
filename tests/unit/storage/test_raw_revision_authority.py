from __future__ import annotations

import sqlite3
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest

from polylogue.archive.revision_authority import (
    HistoricalRawRevision,
    HistoricalRawRevisionStream,
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    append_source_revision,
    classify_historical_full_revision_streams,
    classify_historical_full_revisions,
)
from polylogue.core.enums import Origin, Provider
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch_support import _AppendPlan
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_write import bind_source_raw_revision, write_source_raw_session


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


def test_streamed_historical_full_classifier_matches_byte_proof_without_eager_payloads() -> None:
    payloads = {
        "oldest": b"one\n",
        "middle": b"one\ntwo\n",
        "newest": b"one\ntwo\nthree\n",
    }
    opened: list[str] = []

    def opener(raw_id: str) -> Callable[[], BinaryIO]:
        def open_payload() -> BinaryIO:
            opened.append(raw_id)
            from io import BytesIO

            return BytesIO(payloads[raw_id])

        return open_payload

    streamed = classify_historical_full_revision_streams(
        [HistoricalRawRevisionStream(raw_id, len(payload), opener(raw_id)) for raw_id, payload in payloads.items()]
    )
    eager = classify_historical_full_revisions(
        [HistoricalRawRevision(raw_id, payload) for raw_id, payload in payloads.items()]
    )

    assert {(item.raw_id, item.predecessor_raw_id, item.authority) for item in streamed} == {
        (item.raw_id, item.predecessor_raw_id, item.authority) for item in eager
    }
    assert opened


@pytest.mark.parametrize("payloads", [[b"same", b"same"], [b"left", b"right"], [b"root", b"root-left", b"root-right"]])
def test_historical_classifier_quarantines_unprovable_authority(payloads: list[bytes]) -> None:
    decisions = classify_historical_full_revisions(
        [HistoricalRawRevision(f"raw-{index}", payload) for index, payload in enumerate(payloads)]
    )
    assert decisions
    assert {decision.authority for decision in decisions} == {RawRevisionAuthority.QUARANTINED}


def test_append_envelope_requires_predecessor_revision_and_exact_forward_offsets() -> None:
    with pytest.raises(ValueError, match="predecessor revision and offsets"):
        RawRevisionEnvelope("codex:session", RawRevisionKind.APPEND, "rev-2", 2)


def test_quarantined_append_records_observation_without_claiming_raw_parent() -> None:
    envelope = RawRevisionEnvelope(
        "codex:session",
        RawRevisionKind.APPEND,
        "rev-2",
        2,
        predecessor_source_revision="rev-1",
        append_start_offset=100,
        append_end_offset=150,
        authority=RawRevisionAuthority.QUARANTINED,
    )

    assert envelope.predecessor_raw_id is None
    assert envelope.baseline_raw_id is None


def test_replay_eligible_append_requires_bound_raw_parent() -> None:
    with pytest.raises(ValueError, match="baseline and raw predecessor"):
        RawRevisionEnvelope(
            "codex:session",
            RawRevisionKind.APPEND,
            "rev-2",
            2,
            predecessor_source_revision="rev-1",
            append_start_offset=100,
            append_end_offset=150,
            authority=RawRevisionAuthority.BYTE_PROVEN,
        )


def test_source_writer_persists_typed_revision_envelope() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SOURCE_DDL)
    envelope = RawRevisionEnvelope(
        "codex:session-1",
        RawRevisionKind.APPEND,
        "sha256:revision-2",
        2,
        predecessor_source_revision="sha256:revision-1",
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
        """SELECT logical_source_key, revision_kind, source_revision, predecessor_source_revision,
                  predecessor_raw_id,
                  baseline_raw_id, append_start_offset, append_end_offset,
                  acquisition_generation, revision_authority
           FROM raw_sessions WHERE raw_id = ?""",
        (raw_id,),
    ).fetchone() == (
        "codex:session-1",
        "append",
        "sha256:revision-2",
        "sha256:revision-1",
        "raw-full",
        "raw-full",
        100,
        150,
        2,
        "asserted",
    )


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


def test_raw_revision_material_preserves_capture_mode(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    payload = b'{"chunkedPrompt":{"chunks":[]}}'
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.DRIVE,
            payload=payload,
            source_path="/captures/live-drive.json",
            acquired_at_ms=1,
        )
        provider, observed_payload, _source_path, _kind = archive.raw_revision_material(raw_id)

    assert provider is Provider.DRIVE
    assert observed_payload == payload


def test_revision_binding_is_idempotent_only_for_the_exact_envelope() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SOURCE_DDL)
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/capture/session.jsonl",
        source_index=0,
        payload=b"raw",
        acquired_at_ms=10,
    )
    first = RawRevisionEnvelope("codex:session-1", RawRevisionKind.FULL, "revision-1", 0)
    bind_source_raw_revision(conn, raw_id, first)

    bind_source_raw_revision(conn, raw_id, first)
    with pytest.raises(ValueError, match="already authoritative"):
        bind_source_raw_revision(
            conn,
            raw_id,
            RawRevisionEnvelope("codex:session-1", RawRevisionKind.FULL, "revision-1", 99),
        )
    with pytest.raises(ValueError, match="already authoritative"):
        bind_source_raw_revision(
            conn,
            raw_id,
            RawRevisionEnvelope("codex:session-1", RawRevisionKind.FULL, "different", 1),
        )
    with pytest.raises(ValueError, match="already authoritative or missing"):
        bind_source_raw_revision(conn, "missing", first)


def test_provisional_revision_rebind_accepts_only_classifier_refinement() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SOURCE_DDL)
    raw_id = write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path="/capture/session.jsonl",
        source_index=0,
        payload=b"raw",
        acquired_at_ms=10,
    )
    provisional = RawRevisionEnvelope(
        "codex:session-1",
        RawRevisionKind.FULL,
        "revision-1",
        0,
        authority=RawRevisionAuthority.QUARANTINED,
    )
    bind_source_raw_revision(conn, raw_id, provisional)
    conn.execute(
        """UPDATE raw_sessions
           SET revision_authority = 'byte_proven', predecessor_raw_id = ?,
               baseline_raw_id = ?, acquisition_generation = 4
           WHERE raw_id = ?""",
        ("older-full", "older-full", raw_id),
    )

    bind_source_raw_revision(conn, raw_id, provisional)

    assert conn.execute(
        """SELECT logical_source_key, revision_kind, source_revision,
                  predecessor_raw_id, baseline_raw_id, acquisition_generation,
                  revision_authority
           FROM raw_sessions WHERE raw_id = ?""",
        (raw_id,),
    ).fetchone() == (
        "codex:session-1",
        "full",
        "revision-1",
        "older-full",
        "older-full",
        4,
        "byte_proven",
    )
    with pytest.raises(ValueError, match="already authoritative"):
        bind_source_raw_revision(
            conn,
            raw_id,
            RawRevisionEnvelope(
                "codex:session-1",
                RawRevisionKind.FULL,
                "different-revision",
                0,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )


@pytest.mark.parametrize("write_mode", ["payload", "blob-ref"])
def test_reacquiring_same_raw_cannot_reset_its_authoritative_envelope(
    tmp_path: Path,
    write_mode: str,
) -> None:
    initialize_active_archive_root(tmp_path)
    payload = b'{"type":"session_meta","payload":{"id":"session-1"}}\n'
    first = RawRevisionEnvelope(
        "codex:session-1",
        RawRevisionKind.FULL,
        "revision-1",
        0,
        authority=RawRevisionAuthority.BYTE_PROVEN,
    )
    conflicting = RawRevisionEnvelope(
        "codex:session-1",
        RawRevisionKind.FULL,
        "revision-2",
        99,
        authority=RawRevisionAuthority.BYTE_PROVEN,
    )

    def acquire(archive: ArchiveStore, acquired_at_ms: int) -> str:
        if write_mode == "payload":
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=str(tmp_path / "session.jsonl"),
                acquired_at_ms=acquired_at_ms,
            )
        return archive.write_raw_blob_ref(
            provider=Provider.CODEX,
            blob_hash_hex=sha256(payload).hexdigest(),
            blob_size=len(payload),
            source_path=str(tmp_path / "session.jsonl"),
            acquired_at_ms=acquired_at_ms,
        )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = acquire(archive, 1)
        archive.bind_raw_revision(raw_id, first)

        reacquired_raw_id = acquire(archive, 2)
        assert reacquired_raw_id == raw_id
        with pytest.raises(ValueError, match="already authoritative"):
            archive.bind_raw_revision(raw_id, conflicting)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """SELECT source_revision, acquisition_generation, revision_authority
               FROM raw_sessions WHERE raw_id = ?""",
            (raw_id,),
        ).fetchone() == ("revision-1", 0, "byte_proven")


def test_live_append_acquisition_binds_exact_offsets_to_authoritative_baseline(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    full_payload = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"old","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
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
        b'{"type":"response_item","payload":{"type":"message","id":"new","role":"user","content":'
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
            """SELECT predecessor_source_revision, predecessor_raw_id, baseline_raw_id, append_start_offset,
                      append_end_offset, acquisition_generation, revision_authority
               FROM raw_sessions WHERE revision_kind = 'append'"""
        ).fetchone()
    assert row == (
        "full-revision",
        baseline_raw_id,
        baseline_raw_id,
        len(full_payload),
        stat.st_size,
        1,
        "byte_proven",
    )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        assert set(conn.execute("SELECT decision FROM raw_revision_applications").fetchall()) == {
            ("selected_baseline",),
            ("applied_append",),
        }
        append_application = conn.execute(
            "SELECT raw_id FROM raw_revision_applications WHERE decision = 'applied_append'"
        ).fetchone()
        assert (
            conn.execute(
                "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'codex:session-1'"
            ).fetchone()
            == append_application
        )


def test_live_append_retains_cursor_identity_until_baseline_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    append_payload = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-02T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"new"}]}}\n'
    )
    path = tmp_path / "session.jsonl"
    path.write_bytes((b"x" * 100) + append_payload)
    stat = path.stat()
    plan = _AppendPlan(
        path=path,
        source_name="codex",
        start_offset=100,
        last_complete_newline=stat.st_size,
        stat_size=stat.st_size,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        payload=append_payload,
        payload_hash="append-hash",
        cursor_fingerprint="late-full-revision",
        bytes_read=len(append_payload),
    )
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    owner = SimpleNamespace(
        _cursor=cursor,
        _polylogue=SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=cursor._db_path)),
    )
    events: list[str] = []
    original_bind = ArchiveStore.bind_raw_revision
    original_index = ArchiveStore.write_parsed_for_retained_raw

    def recording_bind(self: ArchiveStore, raw_id: str, revision: RawRevisionEnvelope) -> None:
        events.append("bind")
        original_bind(self, raw_id, revision)

    def recording_index(self: ArchiveStore, *args: Any, **kwargs: Any) -> tuple[str, str]:
        events.append("index")
        return original_index(self, *args, **kwargs)

    monkeypatch.setattr(ArchiveStore, "bind_raw_revision", recording_bind)
    monkeypatch.setattr(ArchiveStore, "write_parsed_for_retained_raw", recording_index)

    result = ingest_append_plans(cast(Any, owner), [plan])

    assert result.succeeded == []
    assert result.deferred == [plan]
    assert events == ["bind"]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        observed = conn.execute(
            """SELECT source_revision, predecessor_source_revision, predecessor_raw_id,
                      baseline_raw_id, append_start_offset, append_end_offset, revision_authority
               FROM raw_sessions WHERE revision_kind = 'append'"""
        ).fetchone()
    assert observed == (
        append_source_revision("late-full-revision", "append-hash"),
        "late-full-revision",
        None,
        None,
        100,
        stat.st_size,
        "quarantined",
    )

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"x" * 100,
            source_path=str(path),
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            baseline_raw_id,
            RawRevisionEnvelope(
                "codex:session-1",
                RawRevisionKind.FULL,
                "late-full-revision",
                0,
            ),
        )
        assert archive.raw_append_revision_parent("codex:session-1", 100, "late-full-revision") == (
            baseline_raw_id,
            baseline_raw_id,
            1,
        )


def test_append_parent_requires_exact_cursor_revision(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"baseline",
            source_path="session.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope("codex:session-1", RawRevisionKind.FULL, "revision-a", 0),
        )

        assert archive.raw_append_revision_parent("codex:session-1", 8, "revision-b") is None
        assert archive.raw_append_revision_parent("codex:session-1", 8, "revision-a") == (
            raw_id,
            raw_id,
            1,
        )
        assert append_source_revision("revision-a", "payload") != append_source_revision("revision-a", "other-payload")
