"""Synthetic tests for normalized blob-residue comparison."""

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

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


def test_duplicate_carriers_count_once_and_source_missing_stays_unresolved(tmp_path: Path) -> None:
    """577 records naming one hash are one object; the missing source blocks."""
    source = tmp_path / "capture.jsonl"
    source.write_bytes(_FIXTURE.read_bytes())
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)
    missing = {
        "cohort": "source_missing",
        "blob_hash": "missing",
        "recorded_source": None,
        "size_bytes": 7,
        "authority_outcome": AuthorityOutcome.UNRESOLVED_BLOCKER.value,
    }
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
    assert last_record == missing
    assert comparison["present_source_candidate_count"] == 577
    assert comparison["source_missing_candidate_count_unresolved"] == 1
    assert comparison["candidate_record_count"] == 578
    assert comparison["candidate_distinct_bytes"] == size + 7
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
    codex_state_module = cast(Any, codex_state)
    original_shape = codex_state_module.logical_source_shape

    def logical_shape(path: Path, *, immutable: bool = False) -> dict[str, tuple[str, ...]]:
        immutable_args.append(immutable)
        return original_shape(path, immutable=immutable)

    monkeypatch.setattr(codex_state_module, "logical_source_shape", logical_shape)

    route, _observation = parse_production_route(path, provider_hint=Provider.CODEX)

    assert route.error is None
    assert route.route == "codex_state.thread_state.non_session"
    assert immutable_args
    assert all(immutable_args)


def _blob_census(*records: dict[str, object]) -> dict[str, object]:
    return {"candidate_hash_digest": "fixture", "records": list(records)}


def _rewritten_header_source(tmp_path: Path) -> Path:
    """A carrier whose leading record is rewritten without changing material."""
    lines = _FIXTURE.read_bytes().splitlines(keepends=True)
    head = json.loads(lines[0])
    head["cwd"] = "/a/relocated/checkout"
    source = tmp_path / "rewritten.jsonl"
    source.write_bytes(json.dumps(head).encode("utf-8") + b"\n" + b"".join(lines[1:]))
    return source


def test_mutable_header_rewrite_is_reacquirable_despite_differing_bytes(tmp_path: Path) -> None:
    """A byte-only oracle would call this divergent; the material is unchanged."""
    stored = tmp_path / "stored.jsonl"
    stored.write_bytes(_FIXTURE.read_bytes())
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(stored)
    current = _rewritten_header_source(tmp_path)
    assert current.read_bytes() != stored.read_bytes()

    receipt = extend_census(
        _blob_census(
            {
                "cohort": "claude_leading_record_or_prefix",
                "origin": "claude-code-session",
                "capture_mode": None,
                "blob_hash": blob_hash,
                "size_bytes": size,
                "recorded_source": str(current),
            },
        ),
        blob_root=store.root,
    )

    comparison = cast(dict[str, object], receipt["normalized_comparison"])
    record = cast(dict[str, object], cast(list[object], receipt["records"])[0])
    assert record["authority_outcome"] == AuthorityOutcome.CURRENT_SOURCE_REACQUIRABLE.value
    assert comparison["unresolved_candidate_count"] == 0
    assert comparison["accepted"] is True


def _codex_sqlite(path: Path, *tables: str) -> Path:
    with sqlite3.connect(path) as connection:
        for table in tables:
            connection.execute(f"CREATE TABLE {table} (id TEXT)")
    return path


def test_declared_out_of_scope_state_db_is_positively_excluded(tmp_path: Path) -> None:
    """Exclusion cites the declared taxonomy reason, never a parse failure."""
    source = _codex_sqlite(tmp_path / "logs_2.sqlite", "logs")
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)

    receipt = extend_census(
        _blob_census(
            {
                "cohort": "mutable_sqlite_snapshot",
                "origin": "codex-session",
                "capture_mode": None,
                "blob_hash": blob_hash,
                "size_bytes": size,
                "recorded_source": str(source),
            },
        ),
        blob_root=store.root,
    )

    record = cast(dict[str, object], cast(list[object], receipt["records"])[0])
    comparison = cast(dict[str, object], record["normalized_comparison"])
    stored_route = cast(dict[str, object], comparison["stored_route"])
    exclusion = cast(dict[str, str], stored_route["exclusion"])
    assert record["authority_outcome"] == AuthorityOutcome.POSITIVELY_EXCLUDED.value
    assert exclusion["taxonomy"] == "codex_state.logs.out_of_scope"
    assert "not" in exclusion["reason"] and "session evidence" in exclusion["reason"]
    assert cast(dict[str, object], receipt["normalized_comparison"])["accepted"] is True


def test_unrecognized_state_db_is_never_excluded(tmp_path: Path) -> None:
    """Only a declared kind excludes; an unknown shape stays a blocker."""
    source = _codex_sqlite(tmp_path / "unknown_9.sqlite", "some_other_table")
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)

    receipt = extend_census(
        _blob_census(
            {
                "cohort": "mutable_sqlite_snapshot",
                "origin": "codex-session",
                "capture_mode": None,
                "blob_hash": blob_hash,
                "size_bytes": size,
                "recorded_source": str(source),
            },
        ),
        blob_root=store.root,
    )

    record = cast(dict[str, object], cast(list[object], receipt["records"])[0])
    comparison = cast(dict[str, object], receipt["normalized_comparison"])
    assert record["authority_outcome"] == AuthorityOutcome.UNRESOLVED_BLOCKER.value
    assert comparison["accepted"] is False


def _restored_record(blob_hash: str, size: int, source: Path, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "cohort": "source_restored",
        "origin": "claude-code-session",
        "capture_mode": None,
        "blob_hash": blob_hash,
        "size_bytes": size,
        "recorded_source": str(source),
        "restoration": {
            "destination": "hook_event_spool",
            "logical_id": "restored-1",
            "outcome": "restored",
        },
    }
    record.update(overrides)
    return record


def test_completed_restoration_that_reacquires_is_its_own_outcome(tmp_path: Path) -> None:
    source = tmp_path / "restored.jsonl"
    source.write_bytes(_FIXTURE.read_bytes())
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)

    receipt = extend_census(_blob_census(_restored_record(blob_hash, size, source)), blob_root=store.root)

    record = cast(dict[str, object], cast(list[object], receipt["records"])[0])
    comparison = cast(dict[str, object], record["normalized_comparison"])
    assert record["authority_outcome"] == AuthorityOutcome.RESTORED_AND_REACQUIRED.value
    assert cast(dict[str, str], comparison["completed_restoration"])["outcome"] == "restored"


def test_restoration_claim_without_reacquisition_still_blocks(tmp_path: Path) -> None:
    """A restoration record never substitutes for the reacquisition proof."""
    source = tmp_path / "restored.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)

    receipt = extend_census(_blob_census(_restored_record(blob_hash, size, source)), blob_root=store.root)

    record = cast(dict[str, object], cast(list[object], receipt["records"])[0])
    assert record["authority_outcome"] == AuthorityOutcome.UNRESOLVED_BLOCKER.value


def test_restoration_outcome_outside_the_apply_vocabulary_is_not_a_restoration(tmp_path: Path) -> None:
    source = tmp_path / "restored.jsonl"
    source.write_bytes(_FIXTURE.read_bytes())
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)
    blocked = _restored_record(blob_hash, size, source)
    blocked["restoration"] = {"destination": "hook_event_spool", "logical_id": "r", "outcome": "blocked"}

    receipt = extend_census(_blob_census(blocked), blob_root=store.root)

    record = cast(dict[str, object], cast(list[object], receipt["records"])[0])
    assert record["authority_outcome"] == AuthorityOutcome.CURRENT_SOURCE_REACQUIRABLE.value


def test_input_claimed_outcomes_are_replaced_by_derived_evidence(tmp_path: Path) -> None:
    """A census that misclassifies itself cannot import its own verdict."""
    source = tmp_path / "not-a-session.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)
    present: dict[str, object] = {
        "cohort": "single_document_or_cache",
        "origin": "claude-code-session",
        "capture_mode": None,
        "blob_hash": blob_hash,
        "size_bytes": size,
        "recorded_source": str(source),
        "authority_outcome": AuthorityOutcome.POSITIVELY_EXCLUDED.value,
    }
    missing: dict[str, object] = {
        "cohort": "source_missing",
        "blob_hash": "missing",
        "recorded_source": None,
        "size_bytes": 3,
        "authority_outcome": AuthorityOutcome.CURRENT_SOURCE_REACQUIRABLE.value,
    }

    receipt = extend_census(_blob_census(present, missing), blob_root=store.root)

    records = cast(list[object], receipt["records"])
    comparison = cast(dict[str, object], receipt["normalized_comparison"])
    assert cast(dict[str, object], records[0])["authority_outcome"] == AuthorityOutcome.UNRESOLVED_BLOCKER.value
    assert cast(dict[str, object], records[1]) == missing
    assert comparison["authority_outcome_counts"] == {
        "current_source_reacquirable": 0,
        "positively_excluded": 0,
        "restored_and_reacquired": 0,
        "unresolved_blocker": 2,
    }
    assert comparison["accepted"] is False


def test_null_declared_size_falls_back_to_the_other_declared_count(tmp_path: Path) -> None:
    source = _codex_sqlite(tmp_path / "logs_2.sqlite", "logs")
    store = BlobStore(tmp_path / "blob")
    blob_hash, size = store.write_from_path(source)

    receipt = extend_census(
        _blob_census(
            {
                "cohort": "mutable_sqlite_snapshot",
                "origin": "codex-session",
                "capture_mode": None,
                "blob_hash": blob_hash,
                "size_bytes": None,
                "blob_size_bytes": size + 11,
                "recorded_source": str(source),
            },
        ),
        blob_root=store.root,
    )

    comparison = cast(dict[str, object], receipt["normalized_comparison"])
    assert comparison["candidate_distinct_bytes"] == size + 11


def test_source_disappearance_blocks_its_candidate_without_aborting_the_census(tmp_path: Path) -> None:
    """One removed carrier blocks itself; the surviving candidate still resolves."""
    survivor = tmp_path / "survivor.jsonl"
    survivor.write_bytes(_FIXTURE.read_bytes())
    vanished = tmp_path / "vanished.jsonl"
    vanished.write_bytes(_FIXTURE.read_bytes())
    store = BlobStore(tmp_path / "blob")
    survivor_hash, survivor_size = store.write_from_path(survivor)
    vanished_hash, vanished_size = store.write_from_path(vanished)
    vanished.unlink()

    def record(blob_hash: str, size: int, source: Path) -> dict[str, object]:
        return {
            "cohort": "source_missing_carrier",
            "origin": "claude-code-session",
            "capture_mode": None,
            "blob_hash": blob_hash,
            "size_bytes": size,
            "recorded_source": str(source),
        }

    receipt = extend_census(
        _blob_census(record(survivor_hash, survivor_size, survivor), record(vanished_hash, vanished_size, vanished)),
        blob_root=store.root,
    )

    records = cast(list[object], receipt["records"])
    comparison = cast(dict[str, object], receipt["normalized_comparison"])
    assert (
        cast(dict[str, object], records[0])["authority_outcome"] == AuthorityOutcome.CURRENT_SOURCE_REACQUIRABLE.value
    )
    blocked = cast(dict[str, object], records[1])
    assert blocked["authority_outcome"] == AuthorityOutcome.UNRESOLVED_BLOCKER.value
    blocked_route = cast(dict[str, object], cast(dict[str, object], blocked["normalized_comparison"])["current_route"])
    assert blocked_route["route"] == "carrier.unreadable"
    assert comparison["unresolved_candidate_count"] == 1
