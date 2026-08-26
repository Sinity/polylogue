"""Red-first regressions for parser admission conservation."""

from __future__ import annotations

import sqlite3
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
    ParseAccounting,
    ParsedMessage,
    ParsedSession,
    content_blocks_from_segments,
)
from polylogue.sources.parsers.chatgpt import extract_messages_from_mapping
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


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
        {"id": "valid", "mapping": {"node": {"id": "node", "message": None}}},
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

    assert {outcome.disposition for outcome in accounting.outcomes} == {
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

    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="parse admission conservation refused"):
            write_parsed_session_to_archive(conn, session)
        assert conn.execute("SELECT 1").fetchone() == (1,)
    finally:
        conn.close()
