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

import asyncio
import hashlib
import sqlite3
import uuid
from pathlib import Path

import pytest

from polylogue.security.excision_carriers import (
    SESSION_CARRIERS,
    CarrierReach,
    UnclassifiedSessionCarrierError,
    audit_session_carriers,
)
from polylogue.storage.accepted_marker_inputs import (
    AcceptedMarkerInputExcisedError,
    MixedAcceptedMarkerInputError,
    append_accepted_marker_input,
    persist_pending_marker_input_sync,
    prepare_accepted_marker_input,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture
from tests.infra.excision import (
    plan_session_excision_from_root,
    resolve_session_excision_target_from_root,
)
from tests.infra.excision_execution import execute_excision
from tests.infra.sync_as_async import AsyncConnectionView

_NATIVE_ID = "session-under-excision"
_OTHER_NATIVE_ID = "session-sharing-the-container"
_PAYLOAD = b'{"conversation": "the one being forgotten"}'
_OTHER_PAYLOAD = b'{"conversation": "someone else in the same export"}'


def _seed_archive(tmp_path: Path) -> tuple[str, str]:
    """Two sessions acquired from one container export, plus an index row each."""
    initialize_active_archive_root(tmp_path)
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_runtime_source_fixture(source_db)
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
                canonical_source_path="/exports/conversations.json",
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


def test_every_session_keyed_relation_in_the_live_schema_is_declared(tmp_path: Path) -> None:
    """A fresh source tier must have a declared reach for every session key.

    Anti-vacuity: add a table to ``SOURCE_DDL`` with a ``raw_id`` column, or
    with ``(origin, session_native_id)``, and omit it from
    ``SESSION_CARRIERS`` -- this goes red naming that table. Widening the
    detection to zero columns makes it vacuous, which the
    ``declared`` assertion below catches.
    """
    source_db = tmp_path / "source.db"
    initialize_runtime_source_fixture(source_db)
    conn = _source_conn(tmp_path)
    try:
        audit = audit_session_carriers(conn)
    finally:
        conn.close()
    assert audit.undeclared == ()
    assert audit.misdeclared == ()
    # The detection actually found the known carriers, so "ok" is not vacuous.
    assert {
        "raw_sessions",
        "raw_existence_changes",
        "raw_hook_events",
        "source_items",
        "pending_accepted_marker_inputs",
        "accepted_marker_inputs",
        "excised_marker_inputs",
    } <= set(audit.declared)


def test_excision_erases_marker_carriers_and_keeps_only_terminal_evidence(tmp_path: Path) -> None:
    """The public excision path cannot leave or replay sealed marker bytes.

    Anti-vacuity: remove the terminal insert, the carrier deletion, or the
    retry guard and this test respectively finds payload-bearing source rows,
    an empty terminal table, or accepts a replay after index replacement.
    """
    session_id, other_session_id = _seed_archive(tmp_path)
    target = prepare_accepted_marker_input(
        "raw-target", [{"session_id": session_id, "candidates": [{"body": "secret marker"}]}]
    )
    accepted_target = prepare_accepted_marker_input(
        "raw-target",
        [{"session_id": session_id, "candidates": [{"body": "accepted secret marker"}]}],
        request_facts={"revision": "accepted"},
    )
    other = prepare_accepted_marker_input(
        "raw-other", [{"session_id": other_session_id, "candidates": [{"body": "keep marker"}]}]
    )
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("BEGIN IMMEDIATE")
        persist_pending_marker_input_sync(source, target, expected_incarnation_id=str(uuid.uuid4()))
        asyncio.run(append_accepted_marker_input(AsyncConnectionView(source), accepted_target))
        asyncio.run(append_accepted_marker_input(AsyncConnectionView(source), other))
    from tests.infra.excision_embeddings import seed_excision_marker_witnesses

    seed_excision_marker_witnesses(tmp_path, (target, accepted_target, other))

    with sqlite3.connect(tmp_path / "index.db") as index:
        outside_marker = index.execute(
            "SELECT * FROM ingest_marker_witnesses WHERE request_key=?", (other.identity,)
        ).fetchone()
        assert outside_marker is not None
    plan = plan_session_excision_from_root(tmp_path, session_id)
    assert plan.source_marker_inputs_pending == 1
    assert plan.source_marker_inputs_accepted == 1
    receipt = execute_excision(tmp_path, session_id, reason="marker secret", actor="user:local")
    assert receipt["counts"]["source_marker_inputs_pending"] == 1
    assert receipt["counts"]["source_marker_inputs_accepted"] == 1
    assert receipt["counts"]["index_marker_witnesses"] == 2

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM pending_accepted_marker_inputs").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs").fetchone() == (1,)
        assert source.execute(
            "SELECT raw_id, carrier_digest, state, stream_id, accepted_sequence, excised_at_ms "
            "FROM excised_marker_inputs WHERE identity = ?",
            (target.identity,),
        ).fetchone() == ("raw-target", target.payload_sha256, "pending", None, None, receipt["excised_at_ms"])
        accepted_tombstone = source.execute(
            "SELECT raw_id, carrier_digest, state, stream_id, accepted_sequence, excised_at_ms "
            "FROM excised_marker_inputs WHERE identity = ?",
            (accepted_target.identity,),
        ).fetchone()
        assert accepted_tombstone is not None
        assert accepted_tombstone[:3] == ("raw-target", accepted_target.payload_sha256, "accepted")
        assert accepted_tombstone[3] and accepted_tombstone[4]
        assert accepted_tombstone[5] == receipt["excised_at_ms"]
        columns = {str(row[1]) for row in source.execute("PRAGMA table_info(excised_marker_inputs)")}
        assert columns.isdisjoint({"payload", "request_facts", "sessions", "candidates", "provenance"})
        source.execute("BEGIN IMMEDIATE")
        with pytest.raises(AcceptedMarkerInputExcisedError):
            persist_pending_marker_input_sync(source, target, expected_incarnation_id=str(uuid.uuid4()))
        with pytest.raises(AcceptedMarkerInputExcisedError):
            asyncio.run(append_accepted_marker_input(AsyncConnectionView(source), accepted_target))
        source.rollback()
        replacement = prepare_accepted_marker_input(
            "raw-other",
            [{"session_id": other_session_id, "candidates": [{"body": "new marker"}]}],
            request_facts={"revision": "after-excision"},
        )
        source.execute("BEGIN IMMEDIATE")
        replacement_sequence = asyncio.run(append_accepted_marker_input(AsyncConnectionView(source), replacement))
        source.commit()
        assert replacement_sequence > int(accepted_tombstone[4])
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT COUNT(*) FROM ingest_marker_witnesses WHERE request_key IN (?, ?)",
            (target.identity, accepted_target.identity),
        ).fetchone() == (0,)
        assert (
            index.execute("SELECT * FROM ingest_marker_witnesses WHERE request_key=?", (other.identity,)).fetchone()
            == outside_marker
        )


def test_mixed_marker_carrier_refuses_before_any_session_tier_mutates(tmp_path: Path) -> None:
    """A sealed shared request must not be partially redacted in an excision.

    Anti-vacuity: move marker resolution after source/index mutation, or omit
    the retained-session check, and the raw/index assertions observe a partial
    apply or a silently erased carrier.
    """
    session_id, other_session_id = _seed_archive(tmp_path)
    mixed = prepare_accepted_marker_input(
        "raw-target",
        [
            {"session_id": session_id, "candidates": [{"body": "target"}]},
            {"session_id": other_session_id, "candidates": [{"body": "retain"}]},
        ],
    )
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("BEGIN IMMEDIATE")
        persist_pending_marker_input_sync(source, mixed, expected_incarnation_id=str(uuid.uuid4()))
    with pytest.raises(MixedAcceptedMarkerInputError, match="retained sessions"):
        execute_excision(tmp_path, session_id, reason="mixed", actor="user:local")
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = 'raw-target'").fetchone() == (1,)
        assert source.execute(
            "SELECT COUNT(*) FROM pending_accepted_marker_inputs WHERE request_key = ?", (mixed.identity,)
        ).fetchone() == (1,)
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)).fetchone() == (1,)


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
        resolve_session_excision_target_from_root(tmp_path, session_id)
    assert "raw_future_evidence" in str(excinfo.value)

    with pytest.raises(UnclassifiedSessionCarrierError):
        execute_excision(tmp_path, session_id, reason="test", actor="user:local")


def test_raw_existence_journal_is_declared_as_excised(tmp_path: Path) -> None:
    """The deletion journal is scrubbed by name when its raw row is excised.

    the audited Excision operation deletes the journal rows for every excised raw
    id (``test_excision`` asserts none survive), so the declaration must say
    EXCISED. Anti-vacuity: remove the declaration and the fresh source schema
    is rejected as an undeclared session carrier; declare any other reach and
    the final assertion fails.
    """
    initialize_runtime_source_fixture(tmp_path / "source.db")
    with _source_conn(tmp_path) as conn:
        audit = audit_session_carriers(conn)
    assert audit.ok
    assert "raw_existence_changes" in audit.declared
    assert SESSION_CARRIERS["raw_existence_changes"].reach is CarrierReach.EXCISED


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

    plan = plan_session_excision_from_root(tmp_path, session_id)
    assert plan.source_container_members == 1
    assert plan.source_container_items == 0
    assert plan.retained_source_containers == ("gen-1:item-1",)

    receipt = execute_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt["counts"]["source_container_members"] == 1
    assert receipt["retained_source_containers"] == ["gen-1:item-1"]
    assert not receipt["complete"], "a container still holding the excised bytes is not a complete excision"

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
    second = execute_excision(tmp_path, other_session_id, reason="test", actor="user:local")
    assert second["counts"]["source_container_items"] == 1
    assert second["retained_source_containers"] == []
    assert second["complete"]
    conn = _source_conn(tmp_path)
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM source_items").fetchone()[0]) == 0
        assert int(conn.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone()[0]) == 0
    finally:
        conn.close()


def test_lineage_cascade_releases_container_shared_only_by_cascade_targets(tmp_path: Path) -> None:
    """Container liveness is resolved against the complete lineage cascade.

    Anti-vacuity: resolving each session independently leaves the shared item
    retained because the other cascade member still appears live at preflight.
    """

    parent_id, child_id = _seed_archive(tmp_path)
    _seed_container(tmp_path)
    index = sqlite3.connect(tmp_path / "index.db")
    try:
        index.execute(
            "INSERT INTO session_links (src_session_id, dst_origin, dst_native_id, link_type, "
            "resolved_dst_session_id, branch_point_message_id, inheritance, status, method, confidence, "
            "evidence_json, observed_at_ms, resolved_at_ms) "
            "VALUES (?, 'chatgpt-export', ?, 'branch', ?, NULL, 'prefix-sharing', NULL, NULL, 1.0, '[]', 1, NULL)",
            (child_id, _NATIVE_ID, parent_id),
        )
        index.commit()
    finally:
        index.close()

    plan = plan_session_excision_from_root(tmp_path, parent_id, cascade_lineage=True)
    assert plan.source_container_items == 1
    assert plan.retained_source_containers == ()
    receipt = execute_excision(tmp_path, parent_id, reason="lineage", actor="user:local", cascade_lineage=True)
    assert receipt["counts"]["source_container_items"] == 1
    source = _source_conn(tmp_path)
    try:
        assert source.execute("SELECT COUNT(*) FROM source_items").fetchone() == (0,)
    finally:
        source.close()


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

    receipt = execute_excision(tmp_path, session_id, reason="test", actor="user:local")
    assert receipt["counts"]["source_publication_reservations"] == 1

    conn = _source_conn(tmp_path)
    try:
        remaining = {str(row[0]) for row in conn.execute("SELECT publication_id FROM blob_publication_reservations")}
    finally:
        conn.close()
    assert remaining == {"pub-b"}
