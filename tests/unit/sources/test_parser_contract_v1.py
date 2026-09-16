from __future__ import annotations

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_models import IMPLAUSIBLE_OCCURRED_AT_MS_CEILING


def test_parsed_message_exposes_optional_archive_contract_fields() -> None:
    message = ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")

    assert message.position is None
    assert message.variant_index is None
    assert message.is_active_path is None
    assert message.is_active_leaf is None
    assert message.model_effort is None
    assert message.duration_ms is None
    assert message.occurred_at_ms is None


def test_parsed_message_derives_occurred_at_ms_from_timestamp() -> None:
    message = ParsedMessage(
        provider_message_id="m1",
        role=Role.USER,
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert message.occurred_at_ms == 1_767_225_600_000


def test_parsed_message_keeps_explicit_occurred_at_ms() -> None:
    message = ParsedMessage(
        provider_message_id="m1",
        role=Role.USER,
        timestamp="2026-01-01T00:00:00+00:00",
        occurred_at_ms=123,
    )

    assert message.occurred_at_ms == 123


@pytest.mark.parametrize("field", ["occurred_at_ms", "position", "variant_index", "duration_ms"])
def test_parsed_message_rejects_negative_archive_integer_fields(field: str) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        if field == "occurred_at_ms":
            ParsedMessage(provider_message_id="m1", role=Role.USER, occurred_at_ms=-1)
        elif field == "position":
            ParsedMessage(provider_message_id="m1", role=Role.USER, position=-1)
        elif field == "variant_index":
            ParsedMessage(provider_message_id="m1", role=Role.USER, variant_index=-1)
        elif field == "duration_ms":
            ParsedMessage(provider_message_id="m1", role=Role.USER, duration_ms=-1)
        else:
            raise AssertionError(f"unknown field: {field}")


def test_parsed_message_refuses_an_implausible_future_occurred_at_ms() -> None:
    """polylogue-xj7ru: only negatives were bounded, so freshness had no ceiling.

    ``occurred_at_ms`` becomes ``sessions.updated_at_ms`` when the export
    carries no session-level timestamp, so one crafted or corrupt far-future
    record pinned that session at the top of freshness ordering permanently.
    The parser boundary now refuses it -- it does not clamp, which would
    silently rewrite provider evidence.

    Anti-vacuity: delete the ``plausible_occurred_at_ms`` validator and the
    microsecond-as-milliseconds value below is accepted as a year-57000
    timestamp.
    """
    microseconds_read_as_milliseconds = 1_767_225_600_000_000

    with pytest.raises(ValueError, match="plausibility ceiling"):
        ParsedMessage(
            provider_message_id="m1",
            role=Role.USER,
            occurred_at_ms=microseconds_read_as_milliseconds,
        )

    # The ceiling itself is admissible; this is a bound, not an off-by-one cut.
    assert (
        ParsedMessage(
            provider_message_id="m1",
            role=Role.USER,
            occurred_at_ms=IMPLAUSIBLE_OCCURRED_AT_MS_CEILING,
        ).occurred_at_ms
        == IMPLAUSIBLE_OCCURRED_AT_MS_CEILING
    )


def test_parsed_session_tracks_active_leaf_without_requiring_parser_changes() -> None:
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="c1",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER)],
    )

    assert session.active_leaf_message_provider_id is None
