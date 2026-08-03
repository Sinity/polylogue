"""Regression coverage for polylogue-ih67 AC#6: Codex UUID-title census.

``compute_codex_title_census`` must report resolved/unresolved counts and
classify every unresolved session by structural REASON (never claiming an
undifferentiated "unresolved" bucket, and never reading message content or
file paths -- only ``sessions`` table counts/enum columns).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.codex_title_census import (
    CodexHookEventTitleCoverage,
    CodexTitleCensus,
    compare_censuses,
    compute_codex_hook_event_title_coverage,
    compute_codex_title_census,
)
from polylogue.core.enums import BlockType, MaterialOrigin, Origin, Provider, Role, TitleSource
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveHookEvent,
    write_source_hook_event,
)
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _human_message(text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id="m1",
        role=Role.USER,
        text=text,
        position=0,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _assistant_message(text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text=text,
        position=0,
        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _write(
    conn: sqlite3.Connection,
    *,
    native_id: str,
    title: str,
    title_source: TitleSource | None = None,
    messages: list[ParsedMessage] | None = None,
) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=title,
        title_source=title_source,
        messages=messages or [],
    )
    write_parsed_session_to_archive(conn, session)


def _seed_corpus(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # 1. Resolved via provider thread name / authored history.
        _write(conn, native_id="native-a", title="Custom Title", title_source=TitleSource.ORIGIN)
        # 2. Unresolved: no messages materialized at all.
        _write(conn, native_id="native-b", title="native-b", messages=[])
        # 3. Unresolved: has messages, but none human-authored.
        _write(
            conn,
            native_id="native-c",
            title="native-c",
            messages=[_assistant_message("hello")],
        )
        # 4. Unresolved: human-authored message present, never enriched (title_source None).
        _write(
            conn,
            native_id="native-d",
            title="native-d",
            title_source=None,
            messages=[_human_message("please fix the thing")],
        )
        # 5. Unresolved anomaly: enrichment ran (title_source set) but title
        #    is still the native id -- simulates a synthesis-failed edge case.
        _write(
            conn,
            native_id="native-e",
            title="native-e",
            title_source=TitleSource.HEURISTIC,
            messages=[_human_message("   ")],
        )
        conn.commit()
    finally:
        conn.close()


def test_census_classifies_resolved_and_every_unresolved_reason(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass  # bootstrap schema

    _seed_corpus(db_path)

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        census = compute_codex_title_census(conn)

    assert census.total_codex_sessions == 5
    assert census.resolved_count == 1
    assert census.unresolved_count == 4
    assert census.resolved_by_title_source == {"origin": 1}
    assert census.unresolved_by_reason == {
        "no_messages_materialized": 1,
        "no_human_authored_message": 1,
        "not_yet_reprocessed_with_assembly": 1,
        "human_authored_present_synthesis_failed": 1,
    }
    # Never claims full coverage without naming what remains.
    assert census.resolved_fraction < 1.0
    assert sum(census.unresolved_by_reason.values()) == census.unresolved_count


def test_census_round_trips_through_json_dict(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass
    _seed_corpus(db_path)

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        census = compute_codex_title_census(conn)

    restored = CodexTitleCensus.from_dict(census.to_dict())
    assert restored == census


def _write_codex_thread_title_hook_event(source_db_path: Path, *, thread_id: str, title: str) -> None:
    """Write a real ``codex_thread_title`` raw_hook_events row (polylogue-0jf4
    shape) directly into ``source.db``."""
    payload: dict[str, object] = {"thread_id": thread_id, "title": title}
    with sqlite3.connect(source_db_path) as conn:
        write_source_hook_event(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="synthetic:state-db",
            payload=json_dumps(payload),
            acquired_at_ms=1_000,
            raw_id=f"raw-{thread_id}",
            hook_event=ArchiveHookEvent(
                hook_event_id=f"codex-thread-title:{thread_id}",
                origin=Origin.CODEX_SESSION,
                source_path="synthetic:state-db",
                event_type="codex_thread_title",
                payload=payload,
                observed_at_ms=1_000,
                native_id=f"{thread_id}:codex_thread_title",
                session_native_id=thread_id,
            ),
        )


def json_dumps(payload: dict[str, object]) -> bytes:
    import json as _json

    return _json.dumps(payload).encode("utf-8")


def test_hook_event_title_coverage_reports_lower_bound(tmp_path: Path) -> None:
    """bd polylogue-foee AC#3: unresolved sessions whose thread id is covered
    by an acquired ``codex_thread_title`` hook event are counted as coverage
    -- a resolved session or one lacking a matching hook event is excluded."""
    db_path = tmp_path / "index.db"
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass
    _seed_corpus(db_path)

    # native-b/c/d/e are unresolved; only native-b and native-d get a
    # matching hook event. native-a is already resolved and must not count
    # even though we also give it a hook event.
    _write_codex_thread_title_hook_event(tmp_path / "source.db", thread_id="native-a", title="ignored")
    _write_codex_thread_title_hook_event(tmp_path / "source.db", thread_id="native-b", title="Recovered B")
    _write_codex_thread_title_hook_event(tmp_path / "source.db", thread_id="native-d", title="Recovered D")

    with (
        sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as index_conn,
        sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True) as source_conn,
    ):
        coverage = compute_codex_hook_event_title_coverage(index_conn, source_conn)

    assert coverage.unresolved_count == 4
    assert coverage.covered_by_hook_event_count == 2
    assert coverage.coverage_fraction == 0.5
    restored = coverage.to_dict()
    assert restored["covered_by_hook_event_count"] == 2


def test_hook_event_title_coverage_fraction_reads_complete_at_zero_unresolved() -> None:
    """Zero unresolved sessions means nothing left to cover -- 1.0, not 0.0.

    Matches the sibling CodexTitleCensus.resolved_fraction convention: an
    empty denominator means "fully done", never "zero coverage" (an operator
    reading "0/0 unresolved (0.0%) covered" could otherwise misread complete
    resolution as no coverage at all).
    """
    coverage = CodexHookEventTitleCoverage(unresolved_count=0, covered_by_hook_event_count=0)
    assert coverage.coverage_fraction == 1.0
    assert coverage.to_dict()["coverage_fraction"] == 1.0


def test_compare_censuses_reports_newly_resolved_delta() -> None:
    before = CodexTitleCensus(
        total_codex_sessions=10,
        resolved_count=2,
        unresolved_count=8,
        unresolved_by_reason={"not_yet_reprocessed_with_assembly": 8},
    )
    after = CodexTitleCensus(
        total_codex_sessions=10,
        resolved_count=9,
        unresolved_count=1,
        resolved_by_title_source={"origin": 7, "heuristic": 2},
        unresolved_by_reason={"no_human_authored_message": 1},
    )
    delta = compare_censuses(before, after)
    assert delta.newly_resolved_count == 7
    payload = delta.to_dict()
    assert payload["newly_resolved_count"] == 7
