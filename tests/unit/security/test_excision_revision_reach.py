"""Excising a session must reach every revision of its raw evidence.

``sessions.raw_id`` names only the most recently applied revision. Superseded
baselines and append fragments stay live until an explicit retention run
compacts them, each with its own ``blob_refs`` rows and its own content hash
-- an append revision is a byte-prefix of its successor but hashes
differently, so the head's ``excised_content`` marker does not cover it.
Excising only the head therefore left the earlier bytes both readable and
freely re-ingestible while the receipt reported success.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.security.excision import (
    apply_session_excision,
    plan_session_excision,
    resolve_session_excision_target,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_LOGICAL_KEY = "codex-session:multi-revision"


def _seed_two_revisions(archive_root: Path) -> tuple[str, str, str, bytes, bytes]:
    """Two live raw revisions of one logical source, plus its index session."""
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    baseline_payload = b'{"event": "first"}\n'
    head_payload = b'{"event": "first"}\n{"event": "second"}\n'

    source_conn = sqlite3.connect(source_db)
    source_conn.execute("PRAGMA foreign_keys = ON")
    try:
        baseline_raw_id = write_source_raw_session(
            source_conn,
            origin="codex-session",
            source_path="/fake/multi-revision.jsonl",
            source_index=0,
            payload=baseline_payload,
            acquired_at_ms=1_000,
            native_id="multi-revision",
            raw_id="raw-baseline",
            revision=RawRevisionEnvelope(
                logical_source_key=_LOGICAL_KEY,
                kind=RawRevisionKind.FULL,
                source_revision="rev-0",
                acquisition_generation=0,
            ),
        )
        head_raw_id = write_source_raw_session(
            source_conn,
            origin="codex-session",
            source_path="/fake/multi-revision.jsonl",
            source_index=0,
            payload=head_payload,
            acquired_at_ms=2_000,
            native_id="multi-revision",
            raw_id="raw-head",
            # A real append fragment: its predecessor is the baseline above,
            # and its own payload hash differs from the baseline's even though
            # the baseline's bytes are a prefix of it. That is exactly why the
            # head's excised_content marker cannot cover the baseline.
            revision=RawRevisionEnvelope(
                logical_source_key=_LOGICAL_KEY,
                kind=RawRevisionKind.APPEND,
                source_revision="rev-1",
                acquisition_generation=1,
                predecessor_source_revision="rev-0",
                predecessor_raw_id=baseline_raw_id,
                baseline_raw_id=baseline_raw_id,
                append_start_offset=len(baseline_payload),
                append_end_offset=len(head_payload),
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_conn = sqlite3.connect(index_db)
    index_conn.execute("PRAGMA foreign_keys = ON")
    try:
        index_conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
            "VALUES ('multi-revision', 'codex-session', ?, 'Multi revision', zeroblob(32), 1000, 2000)",
            (head_raw_id,),
        )
        session_id = str(
            index_conn.execute(
                "SELECT session_id FROM sessions WHERE native_id = 'multi-revision'",
            ).fetchone()[0]
        )
        index_conn.commit()
    finally:
        index_conn.close()
    return session_id, baseline_raw_id, head_raw_id, baseline_payload, head_payload


def _raw_ids(archive_root: Path) -> set[str]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return {str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions")}


def _blob_ref_ids(archive_root: Path) -> set[str]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return {str(row[0]) for row in conn.execute("SELECT ref_id FROM blob_refs")}


def _excised_hashes(archive_root: Path) -> set[bytes]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return {bytes(row[0]) for row in conn.execute("SELECT removed_hash FROM excised_content")}


def test_excision_reaches_every_revision_of_the_session(tmp_path: Path) -> None:
    """A superseded baseline must not survive its session's excision.

    Anti-vacuity: reverting ``resolve_session_excision_target`` to seed raw
    ids from ``sessions.raw_id`` alone leaves ``raw-baseline`` in
    ``raw_sessions`` with a live ``blob_refs`` row and its payload hash absent
    from ``excised_content`` -- all three assertions below go red. A resolver
    that simply deleted every raw row would not pass either: an unrelated
    session's raw evidence is asserted untouched.
    """
    session_id, baseline_raw_id, head_raw_id, baseline_payload, head_payload = _seed_two_revisions(tmp_path)

    # An unrelated logical source, to prove the closure is scoped.
    source_conn = sqlite3.connect(tmp_path / "source.db")
    source_conn.execute("PRAGMA foreign_keys = ON")
    try:
        write_source_raw_session(
            source_conn,
            origin="codex-session",
            source_path="/fake/unrelated.jsonl",
            source_index=0,
            payload=b'{"event": "unrelated"}\n',
            acquired_at_ms=3_000,
            native_id="unrelated",
            raw_id="raw-unrelated",
        )
        source_conn.commit()
    finally:
        source_conn.close()

    target = resolve_session_excision_target(tmp_path, session_id)
    resolved = {raw.raw_id for raw in target.raw_targets}
    assert resolved == {baseline_raw_id, head_raw_id}, "the superseded baseline was not resolved"

    plan = plan_session_excision(tmp_path, session_id)
    assert plan.source_raw_rows == 2

    receipt = apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt.found

    assert _raw_ids(tmp_path) == {"raw-unrelated"}, "an earlier revision survived the excision"
    assert _blob_ref_ids(tmp_path) == {"raw-unrelated"}
    excised = _excised_hashes(tmp_path)
    assert hashlib.sha256(baseline_payload).digest() in excised, "the baseline's own hash was never marked excised"
    assert hashlib.sha256(head_payload).digest() in excised


def test_excision_reports_hook_evidence_it_does_not_remove(tmp_path: Path) -> None:
    """A session's hook payloads survive excision, so the receipt says so.

    Hook events are durable and session-addressable by
    ``(origin, session_native_id)`` but carry no ``raw_sessions`` row, so no
    raw target reaches them. Until excision covers them, reporting unqualified
    success is the defect; naming them as a counted residual is the honest
    partial result.

    Anti-vacuity: dropping ``retained_hook_event_ids`` from the resolver, or
    ``retained_hook_events`` from the receipt, makes ``receipt.complete`` true
    again while the hook row is still readable below.
    """
    session_id, _baseline, _head, _bp, _hp = _seed_two_revisions(tmp_path)

    source_conn = sqlite3.connect(tmp_path / "source.db")
    try:
        with source_conn:
            source_conn.execute(
                "INSERT INTO raw_hook_events ("
                "  hook_event_id, origin, native_id, session_native_id, source_path,"
                "  event_type, payload_json, observed_at_ms"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "hook-1",
                    "codex-session",
                    "hook-native-1",
                    "multi-revision",
                    "/fake/hooks.jsonl",
                    "PreToolUse",
                    '{"tool_input": "content the operator asked to forget"}',
                    2_500,
                ),
            )
    finally:
        source_conn.close()

    plan = plan_session_excision(tmp_path, session_id)
    assert plan.retained_hook_events == ("hook-1",)

    receipt = apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt.retained_hook_events == ("hook-1",)
    assert receipt.complete is False, "an excision that leaves hook payloads must not report completeness"
    assert receipt.as_dict()["retained_hook_events"] == ["hook-1"]

    # The residual is real: the payload is still readable.
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute("SELECT payload_json FROM raw_hook_events WHERE hook_event_id = 'hook-1'").fetchall()
    assert len(rows) == 1
