"""Excision reach is derived from the schema, not from a hand-kept list.

Every earlier instance of "the receipt said complete and the evidence
survived" was a relation nobody remembered to add to the excision path: hook
events (polylogue-bhhsa), fact-tier TODO snapshots (polylogue-si5kj),
container payloads held live through ``source_items`` (polylogue-q4f6d). The
membership list itself was the defect (polylogue-9lrqs), so these tests pin
the derivation: a session-keyed relation with no declared reach must make the
excision refuse, and the declared reach must match the live schema.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.security.excision import (
    apply_session_excision,
    plan_session_excision,
    resolve_session_excision_target,
)
from polylogue.security.excision_carriers import (
    SESSION_CARRIERS,
    CarrierReach,
    UnclassifiedSessionCarrierError,
    audit_session_carriers,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.schema_inventory import canonical_schema_objects
from polylogue.storage.sqlite.archive_tiers.source import RETIRED_SOURCE_SCHEMA_OBJECTS
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_NATIVE_ID = "session-under-excision"
_OTHER_NATIVE_ID = "session-sharing-the-container"
_PAYLOAD = b'{"conversation": "the one being forgotten"}'
_OTHER_PAYLOAD = b'{"conversation": "someone else in the same export"}'


def _seed_archive(tmp_path: Path) -> tuple[str, str]:
    """Two sessions acquired from one container export, plus an index row each."""
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)

    blob_store = BlobStore(tmp_path / "blob")
    blob_store.write_from_bytes(_PAYLOAD)
    blob_store.write_from_bytes(_OTHER_PAYLOAD)

    conn = sqlite3.connect(source_db)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        for raw_id, native_id, payload, index in (
            ("raw-target", _NATIVE_ID, _PAYLOAD, 0),
            ("raw-other", _OTHER_NATIVE_ID, _OTHER_PAYLOAD, 1),
        ):
            write_source_raw_session(
                conn,
                origin="chatgpt-export",
                source_path="/exports/conversations.json",
                source_index=index,
                payload=payload,
                acquired_at_ms=1_000 + index,
                native_id=native_id,
                raw_id=raw_id,
            )
        conn.commit()
    finally:
        conn.close()

    index_conn = sqlite3.connect(index_db)
    index_conn.execute("PRAGMA foreign_keys = ON")
    session_ids: list[str] = []
    try:
        for raw_id, native_id in (("raw-target", _NATIVE_ID), ("raw-other", _OTHER_NATIVE_ID)):
            index_conn.execute(
                "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
                "VALUES (?, 'chatgpt-export', ?, 'Container member', zeroblob(32), 1000, 2000)",
                (native_id, raw_id),
            )
            session_ids.append(
                str(
                    index_conn.execute(
                        "SELECT session_id FROM sessions WHERE native_id = ?",
                        (native_id,),
                    ).fetchone()[0]
                )
            )
        index_conn.commit()
    finally:
        index_conn.close()
    return session_ids[0], session_ids[1]


def _seed_container(tmp_path: Path) -> None:
    """One manifest item covering both raw acquisitions, as a container export."""
    container_hash = hashlib.sha256(b"the whole conversations.json export").digest()
    conn = sqlite3.connect(tmp_path / "source.db")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            "INSERT INTO source_generations (source_generation_id, manifest_digest, addressing_mode, "
            "item_count, created_at_ms) VALUES ('gen-1', ?, 'path', 1, 1000)",
            ("a" * 64,),
        )
        conn.execute(
            "INSERT INTO source_items (source_generation_id, source_item_id, logical_coordinate, "
            "addressing_mode, origin, source_path, disposition, outcome_code, stage, raw_id, blob_hash, "
            "observed_at_ms, updated_at_ms) VALUES ('gen-1', 'item-1', '/exports/conversations.json', "
            "'path', 'chatgpt-export', '/exports/conversations.json', 'admitted', 'success', "
            "'parse', 'raw-target', ?, 1000, 1000)",
            (container_hash,),
        )
        for coordinate, raw_id, payload in (
            ("record-0", "raw-target", _PAYLOAD),
            ("record-1", "raw-other", _OTHER_PAYLOAD),
        ):
            conn.execute(
                "INSERT INTO source_item_raw_members (source_generation_id, source_item_id, "
                "record_coordinate, raw_id, raw_blob_hash) VALUES ('gen-1', 'item-1', ?, ?, ?)",
                (coordinate, raw_id, hashlib.sha256(payload).digest()),
            )
        conn.commit()
    finally:
        conn.close()


def _source_conn(tmp_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "source.db")


#: The shape a migrated historical source tier still carries for the retired
#: inbound span table (polylogue-enrpa). Fresh generations omit it.
_MIGRATED_OTLP_SPANS_DDL = """
CREATE TABLE otlp_spans (
    span_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    parent_span_id    TEXT,
    origin            TEXT,
    session_native_id TEXT,
    name              TEXT NOT NULL,
    kind              TEXT,
    attributes_json   TEXT NOT NULL DEFAULT '{}',
    events_json       TEXT NOT NULL DEFAULT '[]',
    started_at_ms     INTEGER,
    ended_at_ms       INTEGER,
    received_at_ms    INTEGER NOT NULL
) STRICT
"""


def test_every_session_keyed_relation_in_the_live_schema_is_declared(tmp_path: Path) -> None:
    """A fresh source tier must have a declared reach for every session key.

    Anti-vacuity: add a table to ``SOURCE_DDL`` with a ``raw_id`` column, or
    with ``(origin, session_native_id)``, and omit it from
    ``SESSION_CARRIERS`` -- this goes red naming that table. Widening the
    detection to zero columns makes it vacuous, which the
    ``declared`` assertion below catches.
    """
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    conn = _source_conn(tmp_path)
    try:
        audit = audit_session_carriers(conn)
    finally:
        conn.close()
    assert audit.undeclared == ()
    assert audit.misdeclared == ()
    # The detection actually found the known carriers, so "ok" is not vacuous.
    assert {"raw_sessions", "raw_hook_events", "source_items"} <= set(audit.declared)
    # ``otlp_spans`` is retired from fresh DDL (polylogue-enrpa), so a fresh
    # tier has nothing to classify under that name.
    assert "otlp_spans" not in set(audit.declared)


def test_fresh_source_tier_omits_the_retired_inbound_span_storage(tmp_path: Path) -> None:
    """``otlp_spans`` is declared retired, not silently dropped.

    It had canonical DDL, two indexes and migration fixtures but no
    production writer and no production reader other than excision, and no
    inbound OTLP receiver route was reachable (polylogue-enrpa). Fresh
    generations omit it; migrated historical tiers keep it and stay readable
    because durable fresh-DDL parity excludes the explicitly retired set.

    Anti-vacuity: put the table back into ``SOURCE_DDL`` and the fresh live
    schema grows it again, which the ``sqlite_master`` assertion catches;
    delete the ``RETIRED_SOURCE_SCHEMA_OBJECTS`` entries instead and the
    migrated-tier parity proof reports them as unexpected objects.
    """
    for ref in ("table:otlp_spans", "index:idx_otlp_spans_trace", "index:idx_otlp_spans_session"):
        assert ref in RETIRED_SOURCE_SCHEMA_OBJECTS
    declared = {obj.object_ref for obj in canonical_schema_objects(ArchiveTier.SOURCE)}
    assert not declared & {"table:otlp_spans", "index:idx_otlp_spans_trace", "index:idx_otlp_spans_session"}

    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    conn = _source_conn(tmp_path)
    try:
        live = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    assert "otlp_spans" not in live
    assert "idx_otlp_spans_trace" not in live
    assert "idx_otlp_spans_session" not in live


def test_an_undeclared_session_keyed_table_makes_excision_refuse(tmp_path: Path) -> None:
    """The lint is the point: a new keyed relation fails closed.

    Anti-vacuity: delete the ``audit_session_carriers`` call from
    ``resolve_session_excision_target`` and this excision reports a completed
    target while ``raw_future_evidence`` keeps its row.
    """
    session_id, _other = _seed_archive(tmp_path)
    conn = _source_conn(tmp_path)
    try:
        conn.execute(
            "CREATE TABLE raw_future_evidence ("
            "  evidence_id TEXT PRIMARY KEY,"
            "  origin TEXT NOT NULL,"
            "  session_native_id TEXT NOT NULL,"
            "  payload_json TEXT NOT NULL) STRICT"
        )
        conn.commit()
        audit = audit_session_carriers(conn)
    finally:
        conn.close()
    assert audit.undeclared == ("raw_future_evidence",)
    assert not audit.ok

    with pytest.raises(UnclassifiedSessionCarrierError) as excinfo:
        resolve_session_excision_target(tmp_path, session_id)
    assert "raw_future_evidence" in str(excinfo.value)

    with pytest.raises(UnclassifiedSessionCarrierError):
        apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")


def test_a_raw_cascade_declaration_is_checked_against_the_live_foreign_key() -> None:
    """Declaring raw-cascade does not make it true.

    Anti-vacuity: replacing the audit's ``PRAGMA foreign_key_list`` check with
    an unconditional pass makes this green while a ``SET NULL`` relation
    strands rows the excision believes it cascaded away.
    """
    cascade_declared = next(
        table for table, carrier in SESSION_CARRIERS.items() if carrier.reach is CarrierReach.RAW_CASCADE
    )
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, origin TEXT, blob_hash BLOB)")
        conn.execute(
            f'CREATE TABLE "{cascade_declared}" (row_id TEXT PRIMARY KEY, '
            "raw_id TEXT REFERENCES raw_sessions(raw_id) ON DELETE SET NULL)"
        )
        audit = audit_session_carriers(conn)
    finally:
        conn.close()
    assert audit.misdeclared == (cascade_declared,)
    with pytest.raises(UnclassifiedSessionCarrierError, match="ON DELETE CASCADE"):
        audit.raise_if_unreachable()


def test_excision_removes_the_sessions_telemetry_spans(tmp_path: Path) -> None:
    """A migrated tier that still carries ``otlp_spans`` is still excised.

    ``otlp_spans`` is retired from fresh DDL (polylogue-enrpa), so this seeds
    the table the way a migrated historical source tier carries it. The
    session key is ``(origin, session_native_id)`` and there is no raw row,
    so nothing cascades it away.

    Anti-vacuity: drop the ``otlp_spans`` deletion from the apply and the
    excised session's span attributes stay readable under its native id,
    while the unrelated session's span must survive either way.
    """
    session_id, _other = _seed_archive(tmp_path)
    conn = _source_conn(tmp_path)
    try:
        conn.execute(_MIGRATED_OTLP_SPANS_DDL)
        for span_id, native_id in (("span-a", _NATIVE_ID), ("span-b", _OTHER_NATIVE_ID)):
            conn.execute(
                "INSERT INTO otlp_spans (span_id, trace_id, origin, session_native_id, name, "
                "attributes_json, received_at_ms) VALUES (?, 'trace-1', 'chatgpt-export', ?, "
                "'tool.call', '{\"prompt\": \"private\"}', 1000)",
                (span_id, native_id),
            )
        conn.commit()
    finally:
        conn.close()

    plan = plan_session_excision(tmp_path, session_id)
    assert plan.source_otlp_spans == 1

    receipt = apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt.counts["source_otlp_spans"] == 1

    conn = _source_conn(tmp_path)
    try:
        remaining = {str(row[0]) for row in conn.execute("SELECT span_id FROM otlp_spans")}
    finally:
        conn.close()
    assert remaining == {"span-b"}


def test_container_membership_is_excised_per_member(tmp_path: Path) -> None:
    """A shared container keeps its row; the excised member's does not.

    ``source_items`` is a blob-liveness owner, so leaving the container row
    keyed to the excised raw kept the bytes GC-rooted and the receipt still
    claimed success (polylogue-q4f6d).

    Anti-vacuity: drop the container disposition from the apply and
    ``source_item_raw_members`` still holds ``record-0`` for the excised
    session, and the receipt reports ``complete`` with no named residual.
    """
    session_id, other_session_id = _seed_archive(tmp_path)
    _seed_container(tmp_path)

    plan = plan_session_excision(tmp_path, session_id)
    assert plan.source_container_members == 1
    assert plan.source_container_items == 0
    assert plan.retained_source_containers == ("gen-1:item-1",)

    receipt = apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt.counts["source_container_members"] == 1
    assert receipt.retained_source_containers == ("gen-1:item-1",)
    assert not receipt.complete, "a container still holding the excised bytes is not a complete excision"

    conn = _source_conn(tmp_path)
    try:
        members = {str(row[0]) for row in conn.execute("SELECT record_coordinate FROM source_item_raw_members")}
        items = int(conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0])
        excised = {bytes(row[0]) for row in conn.execute("SELECT removed_hash FROM excised_content")}
    finally:
        conn.close()
    assert members == {"record-1"}, "the excised session's container member survived"
    assert items == 1, "the container is still needed by a live member"
    assert hashlib.sha256(_PAYLOAD).digest() in excised

    # Excising the last live member releases the container itself.
    second = apply_session_excision(tmp_path, other_session_id, reason="test", actor="user:local")
    assert second.counts["source_container_items"] == 1
    assert second.retained_source_containers == ()
    assert second.complete

    conn = _source_conn(tmp_path)
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0]) == 0
        assert int(conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone()[0]) == 0
    finally:
        conn.close()


def test_excision_drops_publication_reservations_for_removed_blobs(tmp_path: Path) -> None:
    """A reservation must not keep publishing a hash whose evidence is gone.

    Anti-vacuity: remove the ``blob_publication_reservations`` delete and the
    excised payload's hash stays reserved; the unrelated session's
    reservation must survive either way.
    """
    session_id, _other = _seed_archive(tmp_path)
    conn = _source_conn(tmp_path)
    try:
        for publication_id, payload in (("pub-a", _PAYLOAD), ("pub-b", _OTHER_PAYLOAD)):
            conn.execute(
                "INSERT INTO blob_publication_reservations (publication_id, blob_hash, size_bytes, "
                "publisher_id, reserved_at_ms) VALUES (?, ?, ?, 'publisher', 1000)",
                (publication_id, hashlib.sha256(payload).digest(), len(payload)),
            )
        conn.commit()
    finally:
        conn.close()

    receipt = apply_session_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt.counts["source_publication_reservations"] == 1

    conn = _source_conn(tmp_path)
    try:
        remaining = {str(row[0]) for row in conn.execute("SELECT publication_id FROM blob_publication_reservations")}
    finally:
        conn.close()
    assert remaining == {"pub-b"}
