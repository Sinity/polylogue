"""Supplied-reader source-43 publication receipt laws."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.operations.daemon_ingest import _spool_source_receipt
from polylogue.operations.ingest_inputs import spool_connection
from polylogue.storage.raw_authority import raw_authority_parser_fingerprint
from polylogue.storage.source_generation_receipts import (
    SourceGenerationBlocker,
    _raw_receipt,
    source_generation_receipt_page,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
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
from tests.infra.source_builders import observe_source_generation_receipt


def _connections() -> tuple[sqlite3.Connection, sqlite3.Connection, str]:
    source = sqlite3.connect(":memory:")
    index = sqlite3.connect(":memory:")
    source.executescript(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE])
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
            raw_id, parser_fingerprint, status, logical_keys_json, detail
        ) VALUES ('raw-1', ?, 'complete', '["codex:session-1"]', '')
        """,
        (raw_authority_parser_fingerprint(),),
    )
    source.execute(
        """
        INSERT INTO raw_membership_census(
            raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
        ) VALUES ('raw-1', ?, 'complete', 1, 1, '')
        """,
        (raw_authority_parser_fingerprint(),),
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
    return source, index, item_id


def test_receipt_requires_exact_source43_member_parser_application_head_and_session() -> None:
    """An ANY application or a parsed_at marker must not stand in for current evidence."""
    source, index, _item_id = _connections()

    receipt = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    assert receipt.complete is True
    assert receipt.confirmed_raw_ids == ("raw-1",)
    assert receipt.unresolved_raw_ids == ()
    assert receipt.source_marker_missing_raw_ids == ("raw-1",)
    assert receipt.active_generation == "index-generation-1"
    logical = receipt.items[0].raws[0].logicals[0]
    assert logical.application_ids
    assert logical.head_session_ids == ("codex-session:session-1",)
    assert logical.session_ids == ("codex-session:session-1",)

    index.execute("UPDATE raw_revision_applications SET accepted_raw_id = 'other-raw' WHERE raw_id = 'raw-1'")
    stale = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )
    assert SourceGenerationBlocker.APPLICATION_STALE in stale.items[0].raws[0].logicals[0].blockers
    assert stale.unresolved_raw_ids == ("raw-1",)


def test_parser_census_writer_preserves_inherited_duplicate_receipt_spelling(tmp_path: Path) -> None:
    """Failed inherited census keeps its original sorted canonical JSON evidence."""
    from contextlib import closing

    from polylogue.storage.sqlite.archive_tiers.revision_governance import record_current_parser_source_census
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    with write_lease("test.inherited-parser-census", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with closing(
                source.execute(
                    "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms,logical_source_key,revision_kind) "
                    "VALUES ('raw-1','codex-session','synthetic/export.json',?,1,1,'codex:session-1','full')",
                    (b"r" * 32,),
                )
            ):
                pass
            source.commit()
        for keys, expected in (
            (
                ("codex:session-1", "codex-session:session-1"),
                ("failed", '["codex-session:session-1", "codex-session:session-1"]'),
            ),
            (("codex:session-1",), ("complete", '["codex-session:session-1"]')),
        ):
            with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
                with seal.original_read_snapshot(), seal.source_producer():
                    record_current_parser_source_census(seal, "raw-1", inherited_logical_keys=keys)
                permit = seal.prepare_source_mutation()
                with permit.hold_authority(), permit.mutation_connection() as source:
                    with closing(source.execute("BEGIN IMMEDIATE")):
                        pass
                    permit.apply_source_statements(source)
                    permit.allow_commit(source)
                    source.commit()
                    seal.accept_known_tier_commit(permit.committed())
                with seal.original_read_snapshot():
                    with seal.original_rows(
                        "source",
                        "SELECT status,logical_keys_json FROM raw_authority_parser_census WHERE raw_id='raw-1'",
                    ) as rows:
                        assert tuple(next(rows)) == expected


def test_receipt_counts_all_application_witnesses_and_preserves_ambiguity() -> None:
    """A late second current witness cannot hide behind many stale applications."""
    source, index, _item = _connections()
    try:
        original = index.execute("SELECT decision_id FROM raw_revision_applications").fetchone()[0]
        for number in range(1_024):
            index.execute(
                "INSERT INTO raw_revision_applications "
                "SELECT ?, raw_id, session_id, logical_source_key, ?, acquisition_generation, decision, "
                "accepted_raw_id, accepted_source_revision, accepted_content_hash, accepted_frontier_kind, "
                "accepted_frontier, ?, predecessor_raw_id, append_end_offset, detail, decided_at_ms "
                "FROM raw_revision_applications WHERE decision_id=?",
                (f"z-{number:05d}", f"stale-{number}", f"baseline-{number}", original),
            )
        receipt = observe_source_generation_receipt(
            source, index, source_generation_id="source-43", active_generation="index-1"
        )
        assert receipt.complete
        assert len(receipt.items[0].raws[0].logicals[0].application_ids) == 1_025
        index.execute("UPDATE raw_revision_applications SET source_revision='revision-1' WHERE decision_id='z-01023'")
        ambiguous = observe_source_generation_receipt(
            source, index, source_generation_id="source-43", active_generation="index-1"
        )
        assert not ambiguous.complete
        assert ambiguous.items[0].raws[0].logicals[0].blockers == (SourceGenerationBlocker.APPLICATION_AMBIGUOUS,)
    finally:
        source.close()
        index.close()


def test_receipt_spools_one_raws_complete_logical_denominator_without_collecting_it(tmp_path: Path) -> None:
    from polylogue.operations.ingest_inputs import spool_connection

    source, index, _item = _connections()
    keys = ["codex:session-1"]
    for number in range(1_024):
        native_id = f"logical-{number:05d}"
        keys.append(f"codex:{native_id}")
        source.execute(
            "INSERT INTO raw_session_memberships(raw_id, logical_source_key, provider_session_id, source_revision, "
            "normalized_content_hash, message_count, acquisition_generation, decision, decided_at_ms) "
            "VALUES ('raw-1', ?, ?, 'revision-1', ?, 1, 1, 'applied', 1)",
            (keys[-1], native_id, b"c" * 32),
        )
    source.execute(
        "UPDATE raw_authority_parser_census SET logical_keys_json=? WHERE raw_id='raw-1'", (json.dumps(sorted(keys)),)
    )
    source.execute("UPDATE raw_membership_census SET member_count=? WHERE raw_id='raw-1'", (len(keys),))
    assert source.in_transaction
    spool = _spool_source_receipt(source, index, "source-43", tmp_path / "logical-receipt.sqlite")
    try:
        assert source.in_transaction
        assert spool.unresolved_raw_count == 1
        assert spool.confirmed_raw_count == 0
        with spool_connection(spool.path, read_only=True) as observed:
            assert observed.execute("SELECT COUNT(*) FROM logicals").fetchone() == (1_025,)
            assert observed.execute("SELECT COUNT(*) FROM logicals WHERE complete=1").fetchone() == (1,)
            assert observed.execute("SELECT parser_complete FROM raws WHERE raw_id='raw-1'").fetchone() == (1,)
    finally:
        spool.close()


@pytest.mark.parametrize(
    "recorded",
    [
        '["codex:session-1","codex:session-1"]',
        '["codex:session-1","codex:logical-earlier"]',
        '["codex-session:session-1","codex:session-1"]',
    ],
)
def test_receipt_refuses_invalid_order_or_duplicate_canonical_identity(recorded: str) -> None:
    source, index, _item = _connections()
    source.execute("UPDATE raw_authority_parser_census SET logical_keys_json=? WHERE raw_id='raw-1'", (recorded,))
    receipt = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-1"
    )
    assert not receipt.complete
    assert receipt.unresolved_raw_ids == ("raw-1",)
    assert receipt.items[0].raws[0].parser_blockers == (SourceGenerationBlocker.PARSER_CENSUS_MISMATCH,)


def test_receipt_rejects_incomplete_enumeration_despite_current_raw_witnesses() -> None:
    """Index witnesses cannot make an unexhausted source-43 input complete."""
    source, index, item_id = _connections()
    source.execute(
        "UPDATE source_items SET enumerated_at_ms = NULL WHERE source_generation_id = 'source-43' AND source_item_id = ?",
        (item_id,),
    )

    receipt = observe_source_generation_receipt(
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

    receipt = observe_source_generation_receipt(
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
    source.execute(
        "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms, "
        "logical_source_key, revision_kind, source_revision, acquisition_generation, revision_authority) "
        "VALUES ('raw-0', 'codex-session', '/synthetic/export.json', ?, 1, 0, "
        "'codex:session-1', 'full', 'revision-0', 0, 'byte_proven')",
        (b"q" * 32,),
    )
    source.execute(
        "UPDATE raw_sessions SET source_index=-1, revision_kind='append', revision_authority='byte_proven', "
        "predecessor_raw_id='raw-0', baseline_raw_id='raw-0' WHERE raw_id='raw-1'"
    )
    source.execute(
        """
        UPDATE raw_membership_census
        SET status = 'failed', member_count = 0, revision_authority = 'byte_proven',
            detail = 'append fragments are governed by byte revision authority'
        WHERE raw_id = 'raw-1'
        """
    )

    receipt = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    raw = receipt.items[0].raws[0]
    assert raw.parser_complete is True
    assert [logical.logical_source_key for logical in raw.logicals] == ["codex-session:session-1"]
    assert raw.logicals[0].complete is True

    source.execute("UPDATE raw_membership_census SET member_count = 1 WHERE raw_id = 'raw-1'")
    mismatch = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )
    assert mismatch.items[0].raws[0].parser_complete is False
    assert mismatch.unresolved_raw_ids == ("raw-1",)


@pytest.mark.parametrize("corruption", [None, "cycle", "missing-predecessor"])
def test_receipt_preserves_an_applied_prefix_that_leads_to_the_current_head(corruption: str | None) -> None:
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

    if corruption == "cycle":
        source.execute("UPDATE raw_sessions SET predecessor_raw_id='raw-2' WHERE raw_id='raw-2'")
    elif corruption == "missing-predecessor":
        source.execute("UPDATE raw_sessions SET predecessor_raw_id=NULL WHERE raw_id='raw-2'")

    receipt = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    logical = receipt.items[0].raws[0].logicals[0]
    assert logical.accepted_raw_id == "raw-2"
    assert logical.complete is (corruption is None)
    assert receipt.complete is (corruption is None)
    if corruption is not None:
        assert SourceGenerationBlocker.APPLICATION_STALE in logical.blockers
    else:
        chain_read = False

        def trace(statement: str) -> None:
            nonlocal chain_read
            if "SELECT raw_id, logical_source_key, revision_kind, revision_authority, source_index" in statement:
                chain_read = True

        def stop() -> None:
            if chain_read:
                raise InterruptedError("synthetic predecessor cancellation")

        source.set_trace_callback(trace)
        with pytest.raises(InterruptedError):
            with _raw_receipt(source, index, "raw-1", check_stop=stop) as raw:
                assert raw is not None
                tuple(raw.logicals)
        assert chain_read
        source.set_trace_callback(None)
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 2


def test_receipt_rejects_ambiguous_application_that_mimics_the_current_head() -> None:
    """Matching accepted fields cannot make an ambiguous replay receipt terminal."""
    source, index, _item_id = _connections()
    index.execute("UPDATE raw_revision_applications SET decision = 'ambiguous' WHERE raw_id = 'raw-1'")
    source.execute("UPDATE raw_session_memberships SET decision = 'ambiguous' WHERE raw_id = 'raw-1'")

    receipt = observe_source_generation_receipt(
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

    receipt = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    logical = receipt.items[0].raws[0].logicals[0]
    assert SourceGenerationBlocker.APPLICATION_STALE in logical.blockers
    assert logical.complete is False


def test_the_retired_candidate_membership_relation_is_neither_created_nor_read() -> None:
    """polylogue-79yii: an unwritten relation may not be reported as a measurement.

    ``candidate_source_membership`` was generation-local resume state for the
    index-rebuild transaction lifecycle. PR #5336 retired its only writers, and
    it had no production caller before that. The receipt reader folded it into
    ``index_generation_binding.source_snapshots``, which could then only be the
    empty set -- an unwritten relation presented as a measured "no source
    snapshots". Nothing outside the module ever read the value, so the table,
    the read, and the reported field are all retired together.

    Anti-vacuity, in both directions, which is why this asserts the schema and
    the read rather than an empty result (an empty-result assertion passed
    before the retirement too, and proved nothing):

    * restoring the ``CREATE TABLE`` to ``INDEX_DDL`` turns the first
      assertion red;
    * restoring the reader's query without the table makes
      the actual paged receipt owner raise ``no such table`` instead of
      returning a complete receipt, turning the second half red;
    * reintroducing the reported field as an always-empty stand-in turns the
      last assertion red.
    """
    source, index, _item_id = _connections()

    declared = {str(row[0]) for row in index.execute("SELECT name FROM sqlite_schema")}
    assert "candidate_source_membership" not in declared
    assert "idx_candidate_source_membership_pending" not in declared

    receipt = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )

    assert receipt.complete is True
    assert receipt.active_generation == "index-generation-1"
    assert not hasattr(receipt, "index_generation_binding")


def test_receipt_pages_large_denominator_and_reduces_raw_ids_globally(tmp_path: Path) -> None:
    source = sqlite3.connect(":memory:")
    index = sqlite3.connect(":memory:")
    source.executescript(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE])
    initialize_archive_tier(index, ArchiveTier.INDEX)
    coordinates = tuple(f"input:{number:05d}" for number in range(10_241))
    ids = publish_source_generation(
        source,
        source_generation_id="large",
        manifest_digest="a" * 64,
        addressing_mode="physical-file-v1",
        coordinates=coordinates,
        input_blob_hashes=dict.fromkeys(coordinates, b"i" * 32),
        enumeration_fingerprint="b" * 64,
        observed_at_ms=1,
    )
    source.execute(
        "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
        "VALUES ('shared-raw', 'codex-session', '/synthetic/shared', ?, 1, 1)",
        (b"r" * 32,),
    )
    for item_id in (ids[0], ids[-1]):
        record_source_item_raw_member(
            source,
            source_generation_id="large",
            source_item_id=item_id,
            record_coordinate="record:0",
            raw_id="shared-raw",
            raw_blob_hash=b"r" * 32,
        )
    cursor = None
    page_count = 0
    while True:
        page = source_generation_receipt_page(source, source_generation_id="large", after=cursor)
        if not page.items:
            break
        assert len(page.items) <= 256
        page_count += 1
        cursor = page.next_cursor
    assert page_count == 41
    spool = _spool_source_receipt(source, index, "large", tmp_path / "receipt.sqlite")
    try:
        assert spool.item_count == 10_241
        assert spool.unresolved_raw_count == 1
        assert spool.confirmed_raw_count == 0
        assert spool.raw_page() == ("shared-raw",)
        # Parser census completion does not imply Index publication. The
        # accepted generation still offers this Raw to its resident owner.
        with spool_connection(spool.path) as pending:
            pending.execute("UPDATE raws SET parser_complete=1 WHERE raw_id='shared-raw'")
        assert spool.raw_page() == ("shared-raw",)
        with sqlite3.connect(spool.path) as check:
            assert check.execute("SELECT COUNT(*) FROM items").fetchone() == (10_241,)
    finally:
        spool.close()


def test_receipt_spools_nested_members_from_the_callers_uncommitted_source_snapshot(tmp_path: Path) -> None:
    from polylogue.operations.ingest_inputs import spool_connection

    source, index, item_id = _connections()
    source.execute(
        "UPDATE source_items SET enumerated_record_count=NULL, enumeration_digest=NULL, enumerated_at_ms=NULL, "
        "enumerated_member_count=NULL, enumeration_member_digest=NULL WHERE source_item_id=?",
        (item_id,),
    )
    for ordinal in range(513):
        raw_id = f"nested:{ordinal:04d}"
        source.execute(
            "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
            "VALUES (?, 'codex-session', '/synthetic/nested', ?, 1, 1)",
            (raw_id, b"n" * 32),
        )
        record_source_item_raw_member(
            source,
            source_generation_id="source-43",
            source_item_id=item_id,
            record_coordinate=f"record:nested:{ordinal:04d}",
            raw_id=raw_id,
            raw_blob_hash=b"n" * 32,
        )
    complete_source_item_enumeration(
        source,
        source_generation_id="source-43",
        source_item_id=item_id,
        enumeration_fingerprint="b" * 64,
        record_coordinates=(
            str(row[0])
            for row in source.execute(
                "SELECT record_coordinate FROM source_item_raw_members WHERE source_item_id=?", (item_id,)
            )
        ),
        enumerated_at_ms=2,
    )
    assert source.in_transaction
    spool = _spool_source_receipt(source, index, "source-43", tmp_path / "nested.sqlite")
    try:
        assert source.in_transaction
        assert spool.enumeration_complete
        assert spool.unresolved_raw_count == 513
        assert spool.confirmed_raw_count == 1
        with spool_connection(spool.path, read_only=True) as observed:
            assert observed.execute("SELECT raw_count, unresolved_count FROM items").fetchone() == (514, 513)
            assert observed.execute("SELECT COUNT(*) FROM item_raws").fetchone() == (514,)
    finally:
        spool.close()
        source.rollback()


def test_receipt_cancellation_preserves_source_snapshot_and_settles_private_spool(tmp_path: Path) -> None:
    from polylogue.operations.ingest_inputs import unlink_spool
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

    source, index, _item_id = _connections()
    path = tmp_path / "cancelled-receipt.sqlite"
    calls = 0

    def stop() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("synthetic receipt cancellation")

    with pytest.raises(InterruptedError):
        _spool_source_receipt(source, index, "source-43", path, check_stop=stop)
    assert not retained_native_sql_owners_for_lifetime(path)
    assert source.execute("SELECT COUNT(*) FROM source_item_raw_members").fetchone() == (1,)
    unlink_spool(path)


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "UPDATE raw_sessions SET revision_authority='quarantined' WHERE raw_id='raw-1'",
        "UPDATE raw_sessions SET logical_source_key='codex:other-session' WHERE raw_id='raw-1'",
        "UPDATE raw_sessions SET source_index=-1 WHERE raw_id='raw-1'",
        "UPDATE raw_sessions SET predecessor_raw_id='raw-2' WHERE raw_id='raw-2'",
        "UPDATE raw_sessions SET predecessor_raw_id=NULL WHERE raw_id='raw-2'",
        "UPDATE raw_sessions SET baseline_raw_id='raw-2' WHERE raw_id='raw-2'",
    ],
    ids=[
        "exact",
        "quarantined-baseline",
        "different-session",
        "negative-baseline",
        "cycle",
        "missing-predecessor",
        "wrong-baseline",
    ],
)
def test_byte_fragment_receipt_requires_the_exact_durable_baseline_chain(corruption: str | None) -> None:
    """A failed fragment census proves completion only with its complete BYTE_PROVEN chain."""
    source, index, _item = _connections()
    try:
        source.execute("UPDATE raw_sessions SET revision_authority='byte_proven' WHERE raw_id='raw-1'")
        source.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                logical_source_key, revision_kind, source_revision, acquisition_generation,
                revision_authority, predecessor_raw_id, baseline_raw_id
            ) VALUES ('raw-2', 'codex-session', '/synthetic/export.json', -1, ?, 1, 2,
                      'codex:session-1', 'append', 'revision-2', 2, 'byte_proven', 'raw-1', 'raw-1')
            """,
            (b"s" * 32,),
        )
        source.execute(
            "INSERT INTO raw_authority_parser_census(raw_id, parser_fingerprint, status, logical_keys_json, detail) "
            "VALUES ('raw-2', ?, 'complete', '[\"codex-session:session-1\"]', '')",
            (raw_authority_parser_fingerprint(),),
        )
        source.execute(
            "INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, "
            "censused_at_ms, detail, revision_authority) VALUES ('raw-2', ?, 'failed', 0, 2, '', 'byte_proven')",
            (raw_authority_parser_fingerprint(),),
        )
        if corruption is not None:
            source.execute(corruption)
        with _raw_receipt(source, index, "raw-2", check_stop=None) as receipt:
            assert receipt is not None
            assert receipt.parser_complete is (corruption is None)
            assert receipt.parser_blockers == (
                () if corruption is None else (SourceGenerationBlocker.PARSER_CENSUS_MISMATCH,)
            )
        if corruption is None:
            chain_read = False

            def trace(statement: str) -> None:
                nonlocal chain_read
                if "SELECT raw_id, logical_source_key, revision_kind, revision_authority, source_index" in statement:
                    chain_read = True

            def stop() -> None:
                if chain_read:
                    raise InterruptedError("synthetic byte-chain cancellation")

            source.set_trace_callback(trace)
            with pytest.raises(InterruptedError):
                with _raw_receipt(source, index, "raw-2", check_stop=stop):
                    pytest.fail("cancelled chain published a receipt")
            assert chain_read
            source.set_trace_callback(None)
            assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 2
    finally:
        source.close()
        index.close()


@pytest.mark.parametrize("corruption", [None, "missing_parser", "stale_parser", "membership_count"])
def test_primary_revision_receipt_requires_original_current_parser_identity(corruption: str | None) -> None:
    source, index, _item_id = _connections()
    source.execute("DELETE FROM raw_session_memberships WHERE raw_id='raw-1'")
    source.execute("DELETE FROM raw_membership_census WHERE raw_id='raw-1'")
    if corruption == "missing_parser":
        source.execute("DELETE FROM raw_authority_parser_census WHERE raw_id='raw-1'")
    elif corruption == "stale_parser":
        source.execute("UPDATE raw_authority_parser_census SET parser_fingerprint='stale' WHERE raw_id='raw-1'")
    elif corruption == "membership_count":
        source.execute(
            "INSERT INTO raw_membership_census(raw_id,parser_fingerprint,status,member_count,censused_at_ms,detail) "
            "VALUES ('raw-1',?,'complete',1,1,'')",
            (raw_authority_parser_fingerprint(),),
        )
    receipt = observe_source_generation_receipt(
        source, index, source_generation_id="source-43", active_generation="index-generation-1"
    )
    assert receipt.complete is (corruption is None)
    assert receipt.unresolved_raw_ids == (() if corruption is None else ("raw-1",))


@pytest.mark.parametrize(
    "corruption", [None, "head_generation", "original_generation", "accepted_generation", "accepted_frontier"]
)
def test_superseded_member_keeps_its_original_generation_when_naming_a_newer_head(corruption: str | None) -> None:
    """Supersession binds the original event and the newer accepted Source identity separately."""
    source, index, _item = _connections()
    try:
        source.execute(
            "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms,"
            "logical_source_key,revision_kind,source_revision,acquisition_generation) "
            "VALUES ('raw-2','codex-session','/synthetic/later.json',?,2,2,'codex:session-1','full','revision-2',2)",
            (b"s" * 32,),
        )
        source.execute(
            "INSERT INTO raw_session_memberships(raw_id,logical_source_key,provider_session_id,source_revision,"
            "normalized_content_hash,message_count,acquisition_generation,decision,decided_at_ms) "
            "VALUES ('raw-2','codex:session-1','session-1','revision-2',?,2,2,'applied',3)",
            (b"d" * 32,),
        )
        source.execute("UPDATE raw_session_memberships SET decision='superseded_prefix' WHERE raw_id='raw-1'")
        for raw_id, revision, generation, decision in (
            ("raw-1", "revision-1", 1, ApplicationDecision.SUPERSEDED),
            ("raw-2", "revision-2", 2, ApplicationDecision.SELECTED_BASELINE),
        ):
            record_revision_application_sync(
                index,
                RevisionApplicationReceipt(
                    raw_id=raw_id,
                    session_id="codex-session:session-1",
                    logical_source_key="codex-session:session-1",
                    source_revision=revision,
                    acquisition_generation=generation,
                    decision=decision,
                    accepted_raw_id="raw-2",
                    accepted_source_revision="revision-2",
                    accepted_content_hash=b"d" * 32,
                    accepted_frontier_kind="semantic",
                    accepted_frontier=2,
                ),
                decided_at_ms=3,
            )
        index.execute("UPDATE sessions SET raw_id='raw-2',content_hash=?", (b"d" * 32,))
        if corruption == "head_generation":
            index.execute("UPDATE raw_revision_heads SET acquisition_generation=3")
        elif corruption == "original_generation":
            index.execute("UPDATE raw_revision_applications SET acquisition_generation=3 WHERE raw_id='raw-1'")
        elif corruption == "accepted_generation":
            source.execute("UPDATE raw_session_memberships SET acquisition_generation=3 WHERE raw_id='raw-2'")
        elif corruption == "accepted_frontier":
            source.execute("UPDATE raw_session_memberships SET message_count=3 WHERE raw_id='raw-2'")
        receipt = observe_source_generation_receipt(
            source, index, source_generation_id="source-43", active_generation="index-generation-1"
        )
        logical = receipt.items[0].raws[0].logicals[0]
        assert logical.accepted_raw_id == "raw-2"
        assert receipt.complete is (corruption is None)
        if corruption is not None:
            assert SourceGenerationBlocker.APPLICATION_STALE in logical.blockers
        else:
            assert receipt.confirmed_raw_ids == ("raw-1",)
            assert receipt.unresolved_raw_ids == ()
    finally:
        source.close()
        index.close()


@pytest.mark.parametrize("corruption", [None, "frontier", "generation"])
def test_primary_byte_head_preserves_distinct_session_and_blob_hashes(corruption: str | None) -> None:
    """The original byte revision proves generation/frontier, not the parsed session hash."""
    source, index, _item = _connections()
    try:
        source.execute("DELETE FROM raw_session_memberships WHERE raw_id='raw-1'")
        source.execute("DELETE FROM raw_membership_census WHERE raw_id='raw-1'")
        index.execute("UPDATE raw_revision_heads SET accepted_frontier_kind='byte'")
        index.execute("UPDATE raw_revision_applications SET accepted_frontier_kind='byte'")
        if corruption == "frontier":
            index.execute("UPDATE raw_revision_heads SET accepted_frontier=2")
            index.execute("UPDATE raw_revision_applications SET accepted_frontier=2")
        elif corruption == "generation":
            index.execute("UPDATE raw_revision_heads SET acquisition_generation=2")
        receipt = observe_source_generation_receipt(
            source, index, source_generation_id="source-43", active_generation="index-generation-1"
        )
        assert receipt.complete is (corruption is None)
        if corruption is not None:
            assert SourceGenerationBlocker.APPLICATION_STALE in receipt.items[0].raws[0].logicals[0].blockers
    finally:
        source.close()
        index.close()
