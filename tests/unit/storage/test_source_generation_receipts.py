"""Supplied-reader source-43 publication receipt laws."""

from __future__ import annotations

import sqlite3

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT
from polylogue.storage.source_generation_receipts import SourceGenerationBlocker, source_generation_receipt
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    RevisionApplicationReceipt,
    record_revision_application_sync,
)
from polylogue.storage.sqlite.archive_tiers.source_items import (
    complete_source_item_enumeration,
    publish_source_generation,
    record_source_item_raw_member,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _connections() -> tuple[sqlite3.Connection, sqlite3.Connection, str]:
    source = sqlite3.connect(":memory:")
    index = sqlite3.connect(":memory:")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    initialize_archive_tier(index, ArchiveTier.INDEX)
    source.execute("PRAGMA foreign_keys=ON")
    (item_id,) = publish_source_generation(
        source,
        source_generation_id="source-43",
        manifest_digest="a" * 64,
        addressing_mode="physical-file-v1",
        coordinates=("synthetic-export.json",),
        input_blob_hashes={"synthetic-export.json": b"i" * 32},
        enumeration_fingerprint="b" * 64,
        observed_at_ms=1,
    )
    source.execute(
        """
        INSERT INTO raw_sessions(
            raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms,
            logical_source_key, revision_kind, source_revision, acquisition_generation
        ) VALUES ('raw-1', 'codex-session', '/synthetic/export.json', ?, 1, 1,
                  'codex:session-1', 'full', 'revision-1', 1)
        """,
        (b"r" * 32,),
    )
    source.execute(
        """
        INSERT INTO raw_session_memberships(
            raw_id, logical_source_key, provider_session_id, source_revision,
            normalized_content_hash, message_count, acquisition_generation, decision, decided_at_ms
        ) VALUES ('raw-1', 'codex:session-1', 'session-1', 'revision-1', ?, 1, 1, 'applied', 1)
        """,
        (b"c" * 32,),
    )
    source.execute(
        """
        INSERT INTO raw_authority_parser_census(
            raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
        ) VALUES ('raw-1', ?, 'complete', '["codex:session-1"]', '', 1)
        """,
        (RAW_AUTHORITY_PARSER_FINGERPRINT,),
    )
    source.execute(
        """
        INSERT INTO raw_membership_census(
            raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
        ) VALUES ('raw-1', ?, 'complete', 1, 1, '')
        """,
        (RAW_AUTHORITY_PARSER_FINGERPRINT,),
    )
    record_source_item_raw_member(
        source,
        source_generation_id="source-43",
        source_item_id=item_id,
        record_coordinate="record:0",
        raw_id="raw-1",
        raw_blob_hash=b"r" * 32,
    )
    complete_source_item_enumeration(
        source,
        source_generation_id="source-43",
        source_item_id=item_id,
        enumeration_fingerprint="b" * 64,
        record_coordinates=("record:0",),
        enumerated_at_ms=2,
    )
    source.commit()
    index.execute(
        "INSERT INTO sessions(native_id, origin, raw_id, content_hash) VALUES ('session-1', 'codex-session', 'raw-1', ?)",
        (b"c" * 32,),
    )
    receipt = RevisionApplicationReceipt(
        raw_id="raw-1",
        session_id="codex-session:session-1",
        logical_source_key="codex-session:session-1",
        source_revision="revision-1",
        acquisition_generation=1,
        decision=ApplicationDecision.SELECTED_BASELINE,
        accepted_raw_id="raw-1",
        accepted_source_revision="revision-1",
        accepted_content_hash=b"c" * 32,
        accepted_frontier_kind="semantic",
        accepted_frontier=1,
    )
    record_revision_application_sync(index, receipt, decided_at_ms=2)
    index.execute(
        """
        INSERT INTO candidate_source_membership(raw_id, blob_hash, blob_size, source_snapshot, status)
        VALUES ('raw-1', ?, 1, 'source-snapshot-1', 'committed')
        """,
        (b"r" * 32,),
    )
    return source, index, item_id


def test_receipt_requires_exact_source43_member_parser_application_head_and_session() -> None:
    """An ANY application or a parsed_at marker must not stand in for current evidence."""
    source, index, _item_id = _connections()

    receipt = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    assert receipt.complete is True
    assert receipt.confirmed_raw_ids == ("raw-1",)
    assert receipt.unresolved_raw_ids == ()
    assert receipt.source_marker_missing_raw_ids == ("raw-1",)
    assert receipt.index_generation_binding.active_generation == "index-generation-1"
    assert receipt.index_generation_binding.source_snapshots == ("source-snapshot-1",)
    logical = receipt.items[0].raws[0].logicals[0]
    assert logical.application_ids
    assert logical.head_session_ids == ("codex-session:session-1",)
    assert logical.session_ids == ("codex-session:session-1",)

    index.execute("UPDATE raw_revision_applications SET accepted_raw_id = 'other-raw' WHERE raw_id = 'raw-1'")
    stale = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )
    assert SourceGenerationBlocker.APPLICATION_STALE in stale.items[0].raws[0].logicals[0].blockers
    assert stale.unresolved_raw_ids == ("raw-1",)


def test_receipt_rejects_incomplete_enumeration_despite_current_raw_witnesses() -> None:
    """Index witnesses cannot make an unexhausted source-43 input complete."""
    source, index, item_id = _connections()
    source.execute(
        "UPDATE source_items SET enumerated_at_ms = NULL WHERE source_generation_id = 'source-43' AND source_item_id = ?",
        (item_id,),
    )

    receipt = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    item = receipt.items[0]
    assert receipt.enumeration_complete is False
    assert receipt.complete is False
    assert SourceGenerationBlocker.ENUMERATION_INCOMPLETE in item.blockers
    assert item.raws[0].raw_id == "raw-1"


def test_receipt_reports_retired_source_member_by_coordinate_not_invented_raw_id() -> None:
    """A NULL source-43 foreign key is durable retirement evidence, not a retryable raw gap."""
    source, index, _item_id = _connections()
    source.execute("DELETE FROM raw_sessions WHERE raw_id = 'raw-1'")

    receipt = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    assert receipt.retired_raw_ids == ()
    assert receipt.enumeration_complete is True
    assert receipt.retired_coordinates[0].record_coordinate == "record:0"
    assert SourceGenerationBlocker.RETIRED_MEMBER in receipt.items[0].blockers
    assert receipt.confirmed_raw_ids == ()


def test_receipt_requires_membership_census_and_keeps_byte_governed_logical_denominator() -> None:
    """A byte fragment with a durable key still needs an exact current index witness."""
    source, index, _item_id = _connections()
    source.execute("DELETE FROM raw_session_memberships WHERE raw_id = 'raw-1'")
    source.execute("UPDATE raw_sessions SET source_index = -1 WHERE raw_id = 'raw-1'")
    source.execute(
        """
        UPDATE raw_membership_census
        SET status = 'failed', member_count = 0,
            detail = 'append fragments are governed by byte revision authority'
        WHERE raw_id = 'raw-1'
        """
    )

    receipt = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    raw = receipt.items[0].raws[0]
    assert raw.parser_complete is True
    assert [logical.logical_source_key for logical in raw.logicals] == ["codex-session:session-1"]
    assert raw.logicals[0].complete is True

    source.execute("UPDATE raw_membership_census SET member_count = 1 WHERE raw_id = 'raw-1'")
    mismatch = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )
    assert mismatch.items[0].raws[0].parser_complete is False
    assert mismatch.unresolved_raw_ids == ("raw-1",)


def test_receipt_preserves_an_applied_prefix_that_leads_to_the_current_head() -> None:
    """A later head must not turn a source member's immutable prefix receipt into a false gap."""
    source, index, _item_id = _connections()
    source.execute(
        """
        INSERT INTO raw_sessions(
            raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms,
            logical_source_key, revision_kind, source_revision, predecessor_raw_id,
            acquisition_generation
        ) VALUES ('raw-2', 'codex-session', '/synthetic/export.json', ?, 2, 2,
                  'codex:session-1', 'append', 'revision-2', 'raw-1', 2)
        """,
        (b"s" * 32,),
    )
    index.execute(
        "UPDATE sessions SET raw_id = 'raw-2', content_hash = ? WHERE session_id = 'codex-session:session-1'",
        (b"d" * 32,),
    )
    record_revision_application_sync(
        index,
        RevisionApplicationReceipt(
            raw_id="raw-2",
            session_id="codex-session:session-1",
            logical_source_key="codex-session:session-1",
            source_revision="revision-2",
            acquisition_generation=2,
            decision=ApplicationDecision.APPLIED_APPEND,
            accepted_raw_id="raw-2",
            accepted_source_revision="revision-2",
            accepted_content_hash=b"d" * 32,
            accepted_frontier_kind="semantic",
            accepted_frontier=2,
        ),
        decided_at_ms=3,
    )

    receipt = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    logical = receipt.items[0].raws[0].logicals[0]
    assert logical.accepted_raw_id == "raw-2"
    assert logical.complete is True
    assert receipt.complete is True


def test_receipt_rejects_ambiguous_application_that_mimics_the_current_head() -> None:
    """Matching accepted fields cannot make an ambiguous replay receipt terminal."""
    source, index, _item_id = _connections()
    index.execute("UPDATE raw_revision_applications SET decision = 'ambiguous' WHERE raw_id = 'raw-1'")
    source.execute("UPDATE raw_session_memberships SET decision = 'ambiguous' WHERE raw_id = 'raw-1'")

    receipt = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    logical = receipt.items[0].raws[0].logicals[0]
    assert SourceGenerationBlocker.APPLICATION_STALE in logical.blockers
    assert logical.complete is False


def test_receipt_rejects_old_same_raw_receipt_after_reparse_changes_current_head_content() -> None:
    """A self receipt for old content is not a strict-prefix witness for the same raw head."""
    source, index, _item_id = _connections()
    index.execute("UPDATE sessions SET content_hash = ? WHERE session_id = 'codex-session:session-1'", (b"d" * 32,))
    record_revision_application_sync(
        index,
        RevisionApplicationReceipt(
            raw_id="raw-1",
            session_id="codex-session:session-1",
            logical_source_key="codex-session:session-1",
            source_revision="revision-1",
            acquisition_generation=1,
            decision=ApplicationDecision.REPARSE_REAFFIRMATION,
            accepted_raw_id="raw-1",
            accepted_source_revision="revision-1",
            accepted_content_hash=b"d" * 32,
            accepted_frontier_kind="semantic",
            accepted_frontier=1,
        ),
        decided_at_ms=3,
    )

    receipt = source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    logical = receipt.items[0].raws[0].logicals[0]
    assert SourceGenerationBlocker.APPLICATION_STALE in logical.blockers
    assert logical.complete is False
