from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import MessageOwnerResolution, _event_payload_hash
from polylogue.sources.parse_accounting_spool import SpilledParseAccountingOutcomes, SqliteParseAccountingWriter
from polylogue.sources.parsers.base import ParsedSessionEvent
from polylogue.sources.parsers.base_models import (
    AdmissionDisposition,
    AdmissionOutcome,
    AdmissionRefusalReason,
    AdmissionUnit,
    AdmissionUnknownReason,
    ParseAccounting,
    ParsedMessage,
    ParsedSession,
)
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.sources.streamed_event_payload import SqliteJsonArrayWriter, StreamedJsonArray
from polylogue.storage.sqlite.queries.session_events import read_session_events


def test_streamed_event_payload_replays_hashes_and_prepared_event_round_trips(tmp_path: Path) -> None:
    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        array_writer = SqliteJsonArrayWriter(store.conn)
        array_writer.extend(f"parent-{index}" for index in range(5000))
        parents = array_writer.finish()
        assert isinstance(parents, StreamedJsonArray)

        event = ParsedSessionEvent(
            event_type="antigravity_parent_reference",
            payload={"references": parents, "parent_provider_ids": parents, "parent_observed": True},
        )
        events = store.new_event_sink()
        events.append(event)

        restored = events[0]
        streamed = restored.payload["references"]
        assert isinstance(streamed, StreamedJsonArray)
        assert len(streamed) == 5000
        assert next(streamed.iter_values()) == "parent-0"
        assert list(streamed.iter_values())[-1] == "parent-4999"
        assert _event_payload_hash(event.event_type, event.payload) == _event_payload_hash(
            event.event_type,
            {
                "references": [f"parent-{index}" for index in range(5000)],
                "parent_provider_ids": [f"parent-{index}" for index in range(5000)],
                "parent_observed": True,
            },
        )
        encoded = json.loads(store.conn.execute("SELECT event_json FROM prepared_event").fetchone()[0])
        assert encoded["$polylogue_prepared_event"] == 1
        assert "references" not in encoded["event"]["payload"]
        assert _event_payload_hash(restored.event_type, restored.payload) == _event_payload_hash(
            event.event_type, event.payload
        )
        empty_writer = SqliteJsonArrayWriter(store.conn)
        empty = empty_writer.finish()
        empty_event = ParsedSessionEvent(event_type="antigravity_parent_reference", payload={"references": empty})
        events.append(empty_event)
        empty_restored = events[1]
        assert isinstance(empty_restored.payload["references"], StreamedJsonArray)
        assert len(empty_restored.payload["references"]) == 0
        assert _event_payload_hash(empty_event.event_type, empty_event.payload) == _event_payload_hash(
            empty_event.event_type, {"references": []}
        )
    finally:
        store.close()


def test_parse_accounting_spill_preserves_complete_outcomes_and_conservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        total = 5000
        writer = SqliteParseAccountingWriter(store.conn, {AdmissionUnit.PART: total})
        for ordinal in range(total):
            disposition = (
                AdmissionDisposition.MATERIALIZED,
                AdmissionDisposition.TYPED_UNKNOWN,
                AdmissionDisposition.TYPED_REFUSAL,
            )[ordinal % 3]
            reason = (
                AdmissionUnknownReason.UNSUPPORTED_SHAPE
                if disposition is AdmissionDisposition.TYPED_UNKNOWN
                else AdmissionRefusalReason.MALFORMED
                if disposition is AdmissionDisposition.TYPED_REFUSAL
                else None
            )
            writer.append(
                AdmissionOutcome(
                    unit=AdmissionUnit.PART,
                    ordinal=ordinal,
                    key=f"part-{ordinal}",
                    disposition=disposition,
                    reason=reason,
                )
            )
        accounting = ParseAccounting(expected={AdmissionUnit.PART: total}, outcomes=writer.finish())
        accounting.assert_conserved()
        assert len(accounting.outcomes) == total
        assert (
            sum(outcome.disposition is AdmissionDisposition.TYPED_UNKNOWN for outcome in accounting.iter_outcomes())
            == 1667
        )
        assert (
            sum(outcome.disposition is AdmissionDisposition.TYPED_REFUSAL for outcome in accounting.iter_outcomes())
            == 1666
        )
        store.conn.commit()
        restored = ParseAccounting.from_prepared_payload(accounting.to_prepared_payload(), store.path)
        restored.assert_conserved()
        first_binding = accounting.stable_binding_digest()
        replayed = restored.iter_outcomes()
        assert next(replayed).key == "part-0"
        assert sum(1 for _ in replayed) == total - 1

        second_writer = SqliteParseAccountingWriter(store.conn, {AdmissionUnit.PART: total})
        for ordinal in range(total):
            disposition = (
                AdmissionDisposition.MATERIALIZED,
                AdmissionDisposition.TYPED_UNKNOWN,
                AdmissionDisposition.TYPED_REFUSAL,
            )[ordinal % 3]
            reason = (
                AdmissionUnknownReason.UNSUPPORTED_SHAPE
                if disposition is AdmissionDisposition.TYPED_UNKNOWN
                else AdmissionRefusalReason.MALFORMED
                if disposition is AdmissionDisposition.TYPED_REFUSAL
                else None
            )
            second_writer.append(
                AdmissionOutcome(
                    unit=AdmissionUnit.PART,
                    ordinal=ordinal,
                    key=f"part-{ordinal}",
                    disposition=disposition,
                    reason=reason,
                )
            )
        second = ParseAccounting(expected={AdmissionUnit.PART: total}, outcomes=second_writer.finish())
        second_outcomes = second.outcomes
        first_outcomes = accounting.outcomes
        assert isinstance(first_outcomes, SpilledParseAccountingOutcomes)
        assert isinstance(second_outcomes, SpilledParseAccountingOutcomes)
        assert second_outcomes.accounting_id != first_outcomes.accounting_id
        assert second.stable_binding_digest() == first_binding
        inline_rows = [
            AdmissionOutcome.model_validate(row) for row in second_outcomes.iter_unit(AdmissionUnit.PART.value)
        ]
        inline_rows.reverse()
        inline = ParseAccounting(expected={AdmissionUnit.PART: total}, outcomes=inline_rows)
        assert inline.stable_binding_digest() == first_binding

        def unexpected_full_iteration(self: SpilledParseAccountingOutcomes) -> object:
            raise AssertionError("marker binding must stream spilled outcomes by unit")

        monkeypatch.setattr(SpilledParseAccountingOutcomes, "__iter__", unexpected_full_iteration)

        from polylogue.sources.revision_backfill import _accepted_marker_request_session_binding

        message_sink = store.new_sink()
        message_sink.append(ParsedMessage(provider_message_id="streamed-1", role=Role.USER, text="bound text"))
        array_writer = SqliteJsonArrayWriter(store.conn)
        array_writer.extend(f"event-{index}" for index in range(2048))
        event_sink = store.new_event_sink()
        event_sink.append(
            ParsedSessionEvent(event_type="marker-binding-test", payload={"values": array_writer.finish()})
        )
        store.conn.commit()
        session = ParsedSession(
            source_name=Provider.ANTIGRAVITY,
            provider_session_id="spilled-accounting-session",
            messages=[],
            unit_accounting=second,
        ).model_copy(update={"messages": message_sink, "session_events": event_sink})
        binding = _accepted_marker_request_session_binding(session)
        assert binding["unit_accounting_digest"] == first_binding
        assert binding["content_hash"]
        assert "unit_accounting" not in binding
    finally:
        store.close()


def test_session_event_reads_restore_streamed_payload_keys(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "index.sqlite")
    connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT NOT NULL);
            CREATE TABLE session_events (
                event_id TEXT, session_id TEXT, source_message_id TEXT,
                source_message_provider_id TEXT, position INTEGER, event_type TEXT,
                payload_json TEXT, occurred_at_ms INTEGER, boundary_start_position INTEGER,
                boundary_end_position INTEGER, boundary_message_id TEXT
            );
            CREATE TABLE session_event_array_items (
                session_id TEXT, event_position INTEGER, payload_key TEXT,
                item_ordinal INTEGER, value_json TEXT
            );
            """
        )
        connection.execute("INSERT INTO sessions VALUES ('s-1', 'antigravity-session')")
        connection.execute(
            "INSERT INTO session_events VALUES ('s-1:4', 's-1', NULL, NULL, 4, "
            "'antigravity_parent_reference', '{\"parent_observed\":true}', NULL, NULL, NULL, NULL)"
        )
        connection.executemany(
            "INSERT INTO session_events VALUES (?, 's-1', NULL, NULL, ?, 'ordinary', '{}', NULL, NULL, NULL, NULL)",
            ((f"s-1:{position}", position) for position in range(5, 1005)),
        )
        connection.executemany(
            "INSERT INTO session_event_array_items VALUES ('s-1', 4, ?, ?, ?)",
            [
                ("references", 0, '{"id":"parent-a"}'),
                ("references", 1, '"parent-b"'),
                ("parent_provider_ids", 0, '"parent-a"'),
                ("parent_provider_ids", 1, '"parent-b"'),
            ],
        )
        records = read_session_events(connection, "s-1")
        assert records[0].payload == {
            "parent_observed": True,
            "references": [{"id": "parent-a"}, "parent-b"],
            "parent_provider_ids": ["parent-a", "parent-b"],
        }
    finally:
        connection.close()


def test_archive_writer_stores_streamed_event_arrays_and_reader_restores_them(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import TABLE_SPECS
    from polylogue.storage.sqlite.archive_tiers.write import _write_session_events

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    destination = sqlite3.connect(":memory:")
    destination.row_factory = sqlite3.Row
    try:
        destination.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT NOT NULL)")
        destination.execute("INSERT INTO sessions VALUES ('s-1', 'antigravity-session')")
        for table in ("session_events", "session_event_array_items", "session_agent_policies"):
            destination.execute(f"CREATE TABLE {table} ({TABLE_SPECS[table].ddl_body}) STRICT")

        array_writer = SqliteJsonArrayWriter(store.conn)
        array_writer.extend({"provider_id": f"parent-{index}"} for index in range(1000))
        parents = array_writer.finish()
        event = ParsedSessionEvent(
            event_type="antigravity_parent_reference",
            payload={"references": parents, "parent_provider_ids": parents, "observed": True},
        )
        empty_owners = MessageOwnerResolution(
            keys=(),
            by_physical_coordinate={},
            ambiguous_physical_coordinates=set(),
            by_stable_key={},
            ambiguous_stable_keys=set(),
            ambiguous_keys=set(),
            unique_provider_keys={},
            ambiguous_provider_ids=set(),
        )
        _write_session_events(
            destination,
            "s-1",
            [],
            [event],
            owner_resolution=empty_owners,
            content_identities=[],
        )

        stored = destination.execute("SELECT payload_json FROM session_events").fetchone()[0]
        assert json.loads(stored) == {"observed": True}
        assert destination.execute("SELECT COUNT(*) FROM session_event_array_items").fetchone()[0] == 2000
        restored = read_session_events(destination, "s-1")[0].payload
        references = restored["references"]
        parent_provider_ids = restored["parent_provider_ids"]
        assert isinstance(references, list)
        assert isinstance(parent_provider_ids, list)
        assert references[0] == {"provider_id": "parent-0"}
        assert references[-1] == {"provider_id": "parent-999"}
        assert parent_provider_ids == references
        from polylogue.operations.orchestration import iter_orchestration_events

        orchestration_event = next(iter_orchestration_events(destination, "s-1"))
        orchestration_references = orchestration_event.payload["references"]
        assert isinstance(orchestration_references, list)
        assert orchestration_references == references

        _write_session_events(
            destination,
            "s-1",
            [],
            [ParsedSessionEvent(event_type="antigravity_parent_reference", payload={"ordinary": True})],
            owner_resolution=empty_owners,
            content_identities=[],
        )
        assert read_session_events(destination, "s-1")[0].payload == {"ordinary": True}
        assert destination.execute("SELECT COUNT(*) FROM session_event_array_items").fetchone()[0] == 0

        empty_writer = SqliteJsonArrayWriter(store.conn)
        empty = empty_writer.finish()
        _write_session_events(
            destination,
            "s-1",
            [],
            [ParsedSessionEvent(event_type="antigravity_parent_reference", payload={"references": empty})],
            owner_resolution=empty_owners,
            content_identities=[],
        )
        empty_payload = read_session_events(destination, "s-1")[0].payload
        assert empty_payload == {"references": []}
        assert destination.execute("SELECT COUNT(*) FROM session_event_array_items").fetchone()[0] == 0
    finally:
        destination.close()
        store.close()
