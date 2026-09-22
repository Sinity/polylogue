"""Why the writer did or did not consume prepared rows (polylogue-i07pw AC1).

A parse worker that builds rows and seals a shard only removes writer-side
work on the writes that actually consume them. Profiles of real ingest showed
``_build_message_rows``/``_build_block_rows`` running on the writer thread and
``copy_shard_session_rows`` never running -- so the miss path was the common
path -- but nothing recorded *which gate* declined, and the 2026-09-16 design
note asks for that counter by name: "add a counted event on the prepared=None
fallback naming the gate that failed (session_row_existed / hash mismatch /
slicing / no shard)".

These tests pin each reason against the condition that produces it. That is
what makes the counter usable as evidence: AC1 is proven by the declined
reasons reading zero on a stratified run, and a non-zero reason names the gate
to fix. A counter that reported a single undifferentiated "fallback" would
satisfy no part of that.

They also cover the merge-append carrier split, which polylogue-fdzb3 AC1
depends on and which no existing test reached: an append whose pinned frontier
still matches must reuse the carrier, and one whose frontier moved must
recompute.

Anti-vacuity: delete a ``_record_prepared_disposition`` call and its test goes
red with an empty or differently-keyed counter.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import MaterialOrigin, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    PREPARED_ACCEPTED_DISPOSITIONS,
    prepare_session_rows,
    prepared_row_dispositions,
    reset_prepared_row_dispositions,
    write_parsed_session_to_archive,
)


@pytest.fixture(autouse=True)
def _fresh_counters() -> None:
    reset_prepared_row_dispositions()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _message(native_id: str, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=Role.USER,
        text=text,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
        position=position,
    )


def _session(provider_session_id: str, texts: list[str], *, id_offset: int = 0) -> ParsedSession:
    """``id_offset`` keeps an appended turn's native ids distinct from the stored ones."""
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=provider_session_id,
        title="dispositions",
        messages=[_message(f"m{index + id_offset}", text, index) for index, text in enumerate(texts)],
    )


def test_prepared_rows_are_recorded_as_consumed(tmp_path: Path) -> None:
    session = _session("consumed", ["one", "two"])
    prepared = prepare_session_rows(session)
    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(
            conn, session, content_hash=str(session_content_hash(session)), prepared=prepared
        )
    finally:
        conn.close()

    assert prepared_row_dispositions() == {"prepared_rows": 1}
    assert set(prepared_row_dispositions()) <= PREPARED_ACCEPTED_DISPOSITIONS


def test_no_shard_is_recorded_as_absent(tmp_path: Path) -> None:
    session = _session("absent", ["one"])
    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(conn, session, content_hash=str(session_content_hash(session)))
    finally:
        conn.close()

    assert prepared_row_dispositions() == {"absent": 1}


def test_changed_content_is_recorded_as_a_hash_mismatch(tmp_path: Path) -> None:
    """The prefetch race: the raw changed between warm() and the writer hold."""
    original = _session("mutates", ["the body the worker saw"])
    stale_prepared = prepare_session_rows(original)
    mutated = original.model_copy(update={"messages": [_message("m0", "a different body", 0)]})
    assert session_content_hash(mutated) != session_content_hash(original)

    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(
            conn, mutated, content_hash=str(session_content_hash(mutated)), prepared=stale_prepared
        )
    finally:
        conn.close()

    assert prepared_row_dispositions() == {"content_hash_mismatch": 1}


def test_prefix_sharing_child_is_recorded_as_slicing(tmp_path: Path) -> None:
    """A fork's messages are sliced against its archived parent after preparation."""
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="slice-parent",
        messages=[_message("a", "A", 0), _message("b", "B", 1)],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="slice-child",
        parent_session_provider_id="slice-parent",
        messages=[_message("a", "A", 0), _message("b", "B", 1), _message("c", "C", 2)],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(conn, parent, content_hash=str(session_content_hash(parent)))
        reset_prepared_row_dispositions()
        write_parsed_session_to_archive(
            conn,
            child,
            content_hash=str(session_content_hash(child)),
            prepared=prepare_session_rows(child),
        )
    finally:
        conn.close()

    assert prepared_row_dispositions() == {"prefix_sharing": 1}


def test_append_with_a_matching_frontier_consumes_the_carrier(tmp_path: Path) -> None:
    """polylogue-fdzb3 AC1 for the append path: pinned frontier still valid."""
    first = _session("append", ["one"])
    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(conn, first, content_hash=str(session_content_hash(first)))
        appended = _session("append", ["two"], id_offset=1)
        # Pin the carrier to the frontier the writer will observe, exactly as
        # a preparation running against this connection would.
        session_id = "codex-session:append"
        prepared = prepare_session_rows(
            appended,
            position_offset=archive_tier_write._next_message_position(conn, session_id),
            content_occurrence_offsets=archive_tier_write._stored_content_occurrences(conn, session_id),
        )
        reset_prepared_row_dispositions()
        write_parsed_session_to_archive(
            conn,
            appended,
            content_hash=str(session_content_hash(appended)),
            prepared=prepared,
            merge_append=True,
        )
        stored = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()

    assert prepared_row_dispositions() == {"append_prepared_rows": 1}
    assert stored == 2


def test_append_pinned_to_a_stale_frontier_is_recorded_and_recomputed(tmp_path: Path) -> None:
    """A carrier pinned to the wrong offset must recompute, and say so."""
    first = _session("stale-append", ["one"])
    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(conn, first, content_hash=str(session_content_hash(first)))
        appended = _session("stale-append", ["two"], id_offset=1)
        # Pinned against an empty session: the live frontier is position 1.
        stale = prepare_session_rows(appended, position_offset=0, content_occurrence_offsets={})
        assert stale.position_offset == 0
        reset_prepared_row_dispositions()
        write_parsed_session_to_archive(
            conn,
            appended,
            content_hash=str(session_content_hash(appended)),
            prepared=stale,
            merge_append=True,
        )
        positions = [row[0] for row in conn.execute("SELECT position FROM messages ORDER BY position")]
    finally:
        conn.close()

    assert prepared_row_dispositions() == {"append_frontier_stale": 1}
    # The recompute is what keeps the append honest: the rows land past the
    # stored frontier rather than colliding with position 0.
    assert positions == [0, 1]


def test_a_shard_carrier_is_not_row_tuples_for_an_append(tmp_path: Path) -> None:
    """Shard bindings address rows for a full replace; an append needs tuples."""
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionShardRows
    from polylogue.storage.sqlite.archive_tiers.write_shard import ShardSessionRows

    first = _session("shard-append", ["one"])
    appended = _session("shard-append", ["two"], id_offset=1)
    prepared = prepare_session_rows(appended, position_offset=1, content_occurrence_offsets={})
    shard_like = PreparedSessionShardRows(
        session_id=prepared.session_id,
        session_content_hash=prepared.session_content_hash,
        schema="shard",
        entry=ShardSessionRows(
            session_id=prepared.session_id,
            session_content_hash=prepared.session_content_hash,
            message_lo=1,
            message_hi=len(prepared.message_rows),
            block_lo=1,
            block_hi=len(prepared.block_rows),
            content_identities=prepared.content_identities,
        ),
        content_identities=prepared.content_identities,
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_parsed_session_to_archive(conn, first, content_hash=str(session_content_hash(first)))
        reset_prepared_row_dispositions()
        write_parsed_session_to_archive(
            conn,
            appended,
            content_hash=str(session_content_hash(appended)),
            prepared=shard_like,
            merge_append=True,
        )
    finally:
        conn.close()

    assert prepared_row_dispositions() == {"append_carrier_not_row_tuples": 1}


def test_dispositions_accumulate_across_writes(tmp_path: Path) -> None:
    """One terminal reason per session write, so a page's counts sum to its sessions."""
    conn = _connect(tmp_path / "index.db")
    try:
        for index in range(3):
            session = _session(f"sum-{index}", ["body"])
            prepared = prepare_session_rows(session) if index else None
            write_parsed_session_to_archive(
                conn, session, content_hash=str(session_content_hash(session)), prepared=prepared
            )
        corrupted = _session("sum-corrupt", ["body"])
        write_parsed_session_to_archive(
            conn,
            corrupted,
            content_hash=str(session_content_hash(corrupted)),
            prepared=replace(prepare_session_rows(corrupted), session_content_hash=b"not-this-session"),
        )
    finally:
        conn.close()

    counts = prepared_row_dispositions()
    assert counts == {"absent": 1, "prepared_rows": 2, "content_hash_mismatch": 1}
    assert sum(counts.values()) == 4
