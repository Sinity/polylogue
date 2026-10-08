"""Admission conservation and authoredness at production parser boundaries."""

import pytest

from polylogue.archive.message.types import MessageType
from polylogue.core.enums import MaterialOrigin, Provider
from polylogue.sources.parsers.base import (
    AdmissionDisposition,
    AdmissionOutcome,
    AdmissionRefusalReason,
    AdmissionUnit,
    ParseAccounting,
    ParsedSession,
)
from polylogue.sources.parsers.browser_capture import parse as parse_capture
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session
from tests.infra.source_parser_cases import browser_thinking_turn, case


@pytest.mark.parametrize("representation", ["range", "outcome"])
def test_writer_refuses_out_of_denominator_ordinal(representation: str) -> None:
    """Equal counts formerly admitted an ordinal outside the observed domain."""
    accounting = ParseAccounting(expected={AdmissionUnit.MESSAGE: 1})
    if representation == "range":
        accounting.materialized_ordinals = {AdmissionUnit.MESSAGE: [(2, 3)]}
    else:
        accounting.outcomes = [
            AdmissionOutcome(
                unit=AdmissionUnit.MESSAGE,
                ordinal=2,
                key="nonexistent",
                disposition=AdmissionDisposition.MATERIALIZED,
            )
        ]
    session = ParsedSession(source_name=Provider.CODEX, provider_session_id="refused", messages=[])
    # Preparation reads the Index schema; admission must still refuse before
    # the first SQL write, so no change follows the bootstrap baseline.
    conn = connect_measured(":memory:")
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        baseline = conn.total_changes
        with pytest.raises(ValueError):
            write_fixture_index_session(conn, session, unit_accounting=accounting, standalone_memory=True)
        assert conn.total_changes == baseline
    finally:
        close_fixture_index_connection(conn)


def test_capture_thinking_is_not_runtime_context() -> None:
    """Reasoning-only context markers formerly changed authoredness."""
    session = parse_capture(browser_thinking_turn(), "fallback")
    [message] = session.messages
    assert message.message_type is MessageType.MESSAGE
    assert message.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert len(message.blocks) == 2


def test_empty_tool_part_is_refused_beside_valid_content() -> None:
    """The absent tool-use payload formerly counted as a materialized part."""
    payload = case("codex")
    payload[1]["payload"]["content"] = case("empty_tool_content")
    session = parse_codex(payload, "fallback")
    assert session.messages[0].text == "keep this"
    assert session.unit_accounting is not None
    refused = [row for row in session.unit_accounting.outcomes if row.unit is AdmissionUnit.PART]
    assert len(refused) == 1
    assert refused[0].disposition is AdmissionDisposition.TYPED_REFUSAL
    assert refused[0].reason == AdmissionRefusalReason.MALFORMED
