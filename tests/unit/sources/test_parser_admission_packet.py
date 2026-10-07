"""Red-first regressions for parser admission conservation."""

from __future__ import annotations

from typing import Any

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources import dispatch
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import (
    AdmissionDisposition,
    AdmissionLedger,
    AdmissionOutcome,
    AdmissionUnit,
    AdmissionUnknownReason,
    ParseAccounting,
    ParsedMessage,
    ParsedSession,
    content_blocks_from_segments,
)
from polylogue.sources.parsers.chatgpt import extract_messages_from_mapping
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session


def test_unknown_structured_segment_is_retained_as_typed_evidence() -> None:
    blocks = content_blocks_from_segments([{"type": "future_asset", "asset_id": "asset-1"}])

    assert len(blocks) == 1
    assert blocks[0].type is BlockType.DOCUMENT
    assert blocks[0].metadata == {
        "admission_disposition": "typed_unknown",
        "unknown_reason": "unrecognized_type",
        "wire_type": "future_asset",
    }


def test_chatgpt_keeps_message_with_only_unknown_content_part() -> None:
    mapping = {
        "root": {"id": "root", "message": None, "parent": None, "children": ["u"]},
        "u": {
            "id": "u",
            "parent": "root",
            "children": ["a"],
            "message": {
                "id": "u-msg",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["keep me"]},
            },
        },
        "a": {
            "id": "a",
            "parent": "u",
            "children": [],
            "message": {
                "id": "a-msg",
                "author": {"role": "assistant"},
                "content": {"content_type": "future_asset", "parts": [{"asset_id": "asset-1"}]},
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert [message.provider_message_id for message in messages] == ["u-msg", "a-msg"]
    assert not messages[1].text
    assert messages[1].blocks[0].metadata == {
        "admission_disposition": "typed_unknown",
        "unknown_reason": "unrecognized_type",
        "wire_type": "future_asset",
    }


def test_unknown_late_outer_envelope_is_retained_as_typed_event() -> None:
    records = [
        {
            "type": "session_meta",
            "payload": {"id": "codex-unknown-outer", "timestamp": "2026-01-01T00:00:00Z"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        },
        {"type": "future_outer", "payload": {"marker": "late"}},
    ]

    session = parse_codex(records, "fallback")

    unknown = [event for event in session.session_events if event.event_type == "codex_unknown_outer_record"]
    assert len(unknown) == 1
    assert unknown[0].payload["source_index"] == 3
    assert unknown[0].payload["wire_type"] == "future_outer"


def test_codex_keeps_message_with_only_unknown_structured_block() -> None:
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "u-1",
                "role": "user",
                "content": [{"type": "input_text", "text": "keep me"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "a-1",
                "role": "assistant",
                "content": [{"type": "future_asset", "asset_id": "asset-1"}],
            },
        },
    ]

    session = parse_codex(records, "fallback")

    assert [message.provider_message_id for message in session.messages] == ["u-1", "a-1"]
    assert session.messages[1].blocks[0].metadata == {
        "admission_disposition": "typed_unknown",
        "unknown_reason": "unrecognized_type",
        "wire_type": "future_asset",
    }


def test_chatgpt_bundle_reports_rejected_siblings_even_with_valid_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[dict[str, Any]] = []

    def capture_warning(message: str, *args: object, **kwargs: object) -> None:
        warnings.append({"message": message, "args": args, **kwargs})

    monkeypatch.setattr(dispatch.logger, "warning", capture_warning)
    payloads = [
        {
            "id": "valid",
            "mapping": {
                "node": {
                    "id": "node",
                    "message": {
                        "id": "valid-message",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["neutral test content"]},
                    },
                }
            },
        },
        *[
            {"id": f"drift-{index}", "mapping": {"node": {"id": "node", "message": {"author": "future"}}}}
            for index in range(5)
        ],
    ]

    sessions = parse_payload("chatgpt", payloads, "bundle")

    assert len(sessions) == 1
    assert len(warnings) == 1
    assert warnings[0]["args"][:3] == ("bundle", 5, 6)


def test_admission_ledger_has_closed_terminal_dispositions() -> None:
    ledger = AdmissionLedger()
    ledger.expect(AdmissionUnit.MESSAGE, 2)
    ledger.materialized(AdmissionUnit.MESSAGE, 0, "m-0")
    ledger.unknown(AdmissionUnit.MESSAGE, 1, "m-1")

    accounting = ledger.close()

    assert {outcome.disposition for outcome in accounting.iter_outcomes()} == {
        AdmissionDisposition.MATERIALIZED,
        AdmissionDisposition.TYPED_UNKNOWN,
    }
    accounting.assert_conserved()


def test_writer_refuses_nonconserving_parse_before_sqlite_mutation() -> None:
    # Construct a deliberately incomplete accounting object without using the
    # ledger's close assertion; this is the mutation-resistant red twin.
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="conservation",
        messages=[ParsedMessage(provider_message_id="m-0", role=Role.USER, text="hello")],
        unit_accounting=ParseAccounting(
            expected={AdmissionUnit.MESSAGE: 2},
            outcomes=[
                AdmissionOutcome(
                    unit=AdmissionUnit.MESSAGE,
                    ordinal=0,
                    key="m-0",
                    disposition=AdmissionDisposition.MATERIALIZED,
                )
            ],
        ),
    )

    # Preparation reads the Index schema; the refusal still precedes every write.
    conn = connect_measured(":memory:")
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        baseline = conn.total_changes
        with pytest.raises(ValueError, match="parse admission conservation refused"):
            write_fixture_index_session(conn, session, standalone_memory=True)
        assert conn.execute("SELECT 1").fetchone() == (1,)
        assert conn.total_changes == baseline
    finally:
        close_fixture_index_connection(conn)


# polylogue-ro922. A session file is untrusted input, so the admission ledger's
# cost per input record is a memory amplifier. At head the ledger retained one
# Pydantic ``AdmissionOutcome`` per outer record for the whole parse and
# ``close()`` copied the list: 200k records measured 52.8 MB of traced Python
# allocations for a proof that is one ordinal interval. The bound below is the
# ceiling that reverting the compact-range representation blows.
_LEDGER_RECORD_COUNT = 200_000
_LEDGER_PEAK_BYTES_MAX = 4 * 1024 * 1024


def test_admission_ledger_cost_does_not_scale_with_materialized_records() -> None:
    """Anti-vacuity: restoring one retained AdmissionOutcome per record blows the traced-peak bound.

    A counter bug in the compact representation instead breaks
    ``assert_conserved`` -- the denominator and the per-unit terms it checks
    are reconstructed from the same ranges.
    """
    import tracemalloc

    ledger = AdmissionLedger()
    ledger.expect(AdmissionUnit.OUTER_RECORD, _LEDGER_RECORD_COUNT)

    tracemalloc.start()
    try:
        for ordinal in range(_LEDGER_RECORD_COUNT - 1):
            assert ledger.next_ordinal(AdmissionUnit.OUTER_RECORD) == ordinal
            ledger.materialized(AdmissionUnit.OUTER_RECORD, ordinal, "parsed")
        ledger.unknown(AdmissionUnit.OUTER_RECORD, _LEDGER_RECORD_COUNT - 1, "future_record")
        accounting = ledger.close()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < _LEDGER_PEAK_BYTES_MAX, f"admission ledger traced peak {peak} exceeds {_LEDGER_PEAK_BYTES_MAX}"
    # The conservation contract is unchanged: every denominator member still
    # has exactly one terminal disposition, and the evidence-bearing one is
    # still a real outcome.
    accounting.assert_conserved()
    assert accounting.expected[AdmissionUnit.OUTER_RECORD] == _LEDGER_RECORD_COUNT
    assert len(accounting.outcomes) == 1
    exceptional = next(iter(accounting.outcomes))
    assert exceptional.disposition is AdmissionDisposition.TYPED_UNKNOWN
    assert exceptional.ordinal == _LEDGER_RECORD_COUNT - 1
    assert accounting.materialized_ordinals[AdmissionUnit.OUTER_RECORD] == [(0, _LEDGER_RECORD_COUNT - 1)]
    assert sum(1 for _ in accounting.iter_outcomes()) == _LEDGER_RECORD_COUNT


class _UnreadOutcomes(list[Any]):
    """A recorded-outcome container that fails if anything reads it."""

    def __iter__(self) -> Any:
        raise AssertionError("next_ordinal re-read the recorded outcomes")

    def __len__(self) -> int:
        raise AssertionError("next_ordinal re-counted the recorded outcomes")

    def __getitem__(self, index: Any) -> Any:
        raise AssertionError("next_ordinal indexed the recorded outcomes")


def test_admission_ledger_next_ordinal_reads_no_recorded_outcome() -> None:
    """Operation-count regression: the next ordinal costs no pass over prior outcomes.

    Alternating dispositions keep every outcome as its own retained model and
    materialized run, so a recount has the most to read. Anti-vacuity:
    restore the sum-based ordinal (counting the retained outcomes plus the
    materialized runs for the unit on every call, quadratic in records) and
    the guarded containers raise.
    """
    unit = AdmissionUnit.OUTER_RECORD
    record_count = 1_000
    ledger = AdmissionLedger()
    ledger.expect(unit, record_count)
    for ordinal in range(record_count):
        if ordinal % 2:
            ledger.unknown(unit, ordinal, "future_record")
        else:
            ledger.materialized(unit, ordinal, "parsed")

    outcomes, materialized = ledger._outcomes, ledger._materialized
    ledger._outcomes = _UnreadOutcomes()
    guarded: dict[AdmissionUnit, list[list[int]]] = {key: _UnreadOutcomes() for key in materialized}
    ledger._materialized = guarded
    try:
        assert ledger.next_ordinal(unit) == record_count
    finally:
        ledger._outcomes, ledger._materialized = outcomes, materialized

    accounting = ledger.close()
    accounting.assert_conserved()
    assert len(accounting.outcomes) == record_count // 2
    assert sum(1 for _ in accounting.iter_outcomes()) == record_count


def test_admission_conservation_rejects_a_range_that_overcounts() -> None:
    """Anti-vacuity: dropping the range arithmetic from assert_conserved makes this pass silently."""
    overcounting = ParseAccounting(
        expected={AdmissionUnit.OUTER_RECORD: 3},
        materialized_ordinals={AdmissionUnit.OUTER_RECORD: [(0, 4)]},
    )
    with pytest.raises(ValueError, match="invalid materialized admission range"):
        overcounting.assert_conserved()

    overlapping = ParseAccounting(
        expected={AdmissionUnit.OUTER_RECORD: 3},
        outcomes=[
            AdmissionOutcome(
                unit=AdmissionUnit.OUTER_RECORD,
                ordinal=1,
                key="dup",
                disposition=AdmissionDisposition.TYPED_UNKNOWN,
                reason=AdmissionUnknownReason.UNRECOGNIZED_TYPE,
            )
        ],
        materialized_ordinals={AdmissionUnit.OUTER_RECORD: [(0, 3)]},
    )
    with pytest.raises(ValueError, match=r"duplicate admission outcome for outer_record\[1\]"):
        overlapping.assert_conserved()
