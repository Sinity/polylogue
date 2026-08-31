"""Synthetic tests for normalized blob-residue comparison."""

import sqlite3
from pathlib import Path
from typing import cast

from pytest import MonkeyPatch

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.maintenance.blob_residue_comparison import (
    AuthorityOutcome,
    ComparisonOutcome,
    ContributionComparison,
    NormalizedContribution,
    compare_normalized_contributions,
    extend_census,
    parse_production_route,
)
from polylogue.sources.parsers import codex_state
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.blob_store import BlobStore

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


def test_large_route_uses_the_production_streaming_branch(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_bytes(_FIXTURE.read_bytes())
    monkeypatch.setattr("polylogue.maintenance.blob_residue_comparison._STREAMING_FULL_INGEST_BYTES", 1)

    route, observation = parse_production_route(path, provider_hint=Provider.CLAUDE_CODE)

    assert route.error is None
    assert route.route == "stream.parse.accepted"
    assert len(route.sessions) == 2
    assert observation["size_bytes"] == path.stat().st_size


def test_extend_census_leaves_source_missing_records_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "capture.jsonl"
    source.write_bytes(_FIXTURE.read_bytes())
    store = BlobStore(tmp_path / "blob")
    blob_hash, _size = store.write_from_path(source)
    missing = {"cohort": "source_missing", "blob_hash": "missing", "recorded_source": None}
    present = {
        "cohort": "claude_leading_record_or_prefix",
        "origin": "claude-code-session",
        "capture_mode": None,
        "blob_hash": blob_hash,
        "recorded_source": str(source),
    }
    census = {"candidate_hash_digest": "fixture", "records": [present.copy() for _ in range(577)] + [missing]}

    receipt = extend_census(census, blob_root=store.root)

    records = receipt["records"]
    comparison = cast(dict[str, object], receipt["normalized_comparison"])
    assert isinstance(records, list)
    assert isinstance(comparison, dict)
    last_record = cast(dict[str, object], records[-1])
    assert last_record["blob_hash"] == "missing"
    assert last_record["authority_outcome"] == AuthorityOutcome.UNRESOLVED_BLOCKER.value
    assert comparison["present_source_candidate_count"] == 577
    assert comparison["source_missing_candidate_count_untouched"] == 1
    assert comparison["candidate_record_count"] == 578
    assert comparison["unresolved_candidate_count"] == 1
    authority_values = cast(list[str], comparison["authority_outcome_values"])
    assert set(authority_values) == {
        "current_source_reacquirable",
        "restored_and_reacquired",
        "positively_excluded",
        "unresolved_blocker",
    }


def test_present_candidate_count_is_derived_and_route_errors_block(tmp_path: Path) -> None:
    source = tmp_path / "not-a-session.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)
    census = {
        "candidate_hash_digest": "fixture",
        "records": [
            {
                "cohort": "single_document_or_cache",
                "origin": "claude-code-session",
                "capture_mode": None,
                "blob_hash": blob_hash,
                "size_bytes": size,
                "recorded_source": str(source),
            }
        ],
    }

    receipt = extend_census(census, blob_root=store.root)

    comparison = cast(dict[str, object], receipt["normalized_comparison"])
    assert comparison["present_source_candidate_count"] == 1
    assert comparison["candidate_distinct_bytes"] == size
    assert comparison["unresolved_candidate_count"] == 1
    first_record = cast(dict[str, object], cast(list[object], receipt["records"])[0])
    assert first_record["authority_outcome"] == AuthorityOutcome.UNRESOLVED_BLOCKER.value


def test_sqlite_route_uses_immutable_read_only_connections(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    path = tmp_path / "state_5.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE threads (id TEXT)")
        connection.execute("CREATE TABLE thread_spawn_edges (id TEXT)")

    immutable_args: list[bool] = []
    original_connect = codex_state._connect_readonly

    def connect_readonly(path: Path, *, timeout: float = 1.0, immutable: bool = False) -> sqlite3.Connection:
        immutable_args.append(immutable)
        return original_connect(path, timeout=timeout, immutable=immutable)

    monkeypatch.setattr(codex_state, "_connect_readonly", connect_readonly)

    route, _observation = parse_production_route(path, provider_hint=Provider.CODEX)

    assert route.error is None
    assert route.route == "codex_state.thread_state.non_session"
    assert immutable_args
    assert all(immutable_args)
