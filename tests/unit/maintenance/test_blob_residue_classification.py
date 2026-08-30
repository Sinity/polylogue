"""Synthetic tests for normalized blob-residue comparison."""

from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.maintenance.blob_residue_comparison import (
    ComparisonOutcome,
    ContributionComparison,
    NormalizedContribution,
    compare_normalized_contributions,
    parse_production_route,
)
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "claude-code" / "claude-normalization-main.jsonl"


def _session(*texts: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="session-1",
        messages=[
            ParsedMessage(provider_message_id=f"message-{index}", role=Role.USER, text=text)
            for index, text in enumerate(texts)
        ],
    )


def _comparison(stored: ParsedSession, current: ParsedSession) -> ContributionComparison:
    return compare_normalized_contributions(
        NormalizedContribution.from_sessions([stored]),
        NormalizedContribution.from_sessions([current]),
    )


def test_header_only_parser_normalization_is_reproduced() -> None:
    stored = _session("same material")
    current = stored.model_copy(update={"title": "a current header"})

    result = _comparison(stored, current)

    assert result.outcome is ComparisonOutcome.REPRODUCED_NORMALIZED
    assert result.differing_fields == ()


def test_current_normalized_material_strictly_extends_stored_material() -> None:
    result = _comparison(_session("first"), _session("first", "second"))

    assert result.outcome is ComparisonOutcome.SUPERSEDED_PREFIX
    assert result.extended_fields == ("messages",)


def test_real_message_difference_remains_content_divergent_and_named() -> None:
    result = _comparison(_session("stored"), _session("current"))

    assert result.outcome is ComparisonOutcome.CONTENT_DIVERGENT
    assert result.differing_fields == ("messages",)


def test_missing_session_is_not_accepted_as_current_prefix() -> None:
    stored = _session("stored")
    current = stored.model_copy(update={"provider_session_id": "different-session"})

    result = _comparison(stored, current)

    assert result.outcome is ComparisonOutcome.CONTENT_DIVERGENT
    assert "sessions" in result.differing_fields


def test_production_route_witness_uses_detector_and_parser_admission() -> None:
    route, observation = parse_production_route(_FIXTURE, provider_hint=Provider.CLAUDE_CODE)

    assert route.error is None
    assert route.route == "stream.parse.accepted"
    assert route.detector_evidence
    assert len(route.sessions) == 2
    assert observation["sha256"]
