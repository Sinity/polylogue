"""Diagnostic benchmark for polylogue-3wb: rebuild graph_resolve tail latency.

Live evidence (2026-07-03, active v24 rebuild): batch 311 took 274s with 260s
in ``append.index.graph_resolve`` for a single Codex session; batch 316 took
427s. This reproduces the mechanism at controlled scale using
``_resolve_session_graph``/``_reextract_prefix_tail_db`` directly
(``polylogue/storage/sqlite/archive_tiers/write.py``).

Root cause (traced via per-substage timings, see ``add_timing`` calls in
``_reextract_prefix_tail_db``): when a session's *children* (resumes/forks)
are ingested before the *parent* during a rebuild, each child is stored whole
(#2467 deferred-tail workaround) rather than as a divergent tail. When the
parent finally arrives, ``_resolve_session_graph`` walks every such orphaned
child and normalizes it: deletes the child's own duplicate copy of the shared
prefix, remaps the child's ``session_events`` refs onto the parent's rows, and
deletes prefix-scoped dependents (blocks/attachment_refs/paste_spans/
web_content_constructs). This is genuine row-mutation work proportional to
``orphaned_children x shared_prefix_size`` -- not an accidental full scan.

This benchmark exists to (a) prove the scaling is linear in child count, not
quadratic (ruling out an accidental O(n^2) bug in the resolve path itself),
and (b) give a concrete before/after harness for whatever mitigates the
*frequency* of the expensive path (see polylogue-3wb bead notes for the
proposed follow-up: rebuild replay currently processes logical sources in
lexicographic ``sorted(logical_keys)`` order --
``polylogue/sources/revision_backfill.py`` -- with no parent-before-child
ordering, so this path triggers far more often during a cold rebuild than it
would with lineage-aware scheduling).

Every SQL statement on this hot path was audited via ``EXPLAIN QUERY PLAN``
against the live archive and confirmed to already use an index (SEARCH, not
SCAN) except ``web_content_constructs`` -- fixed by polylogue-rgbj's
``idx_web_constructs_message``, which this benchmark's fixture does not
exercise (Codex sessions have no ``web_content_constructs`` rows; that
construct type is ChatGPT/browser-capture-specific). Restructuring the SQL
here (batched vs. per-child statements, IN-list vs. range subqueries) was
measured to change wall time by less than 10%, confirming the cost is real
B-tree mutation work rather than query-shape/parameter-marshaling overhead.

Run with:
    pytest tests/benchmarks/test_graph_resolve_deferred_tail.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_INDEX_DDL = ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX]

_PARENT_MESSAGES = 3000
_CHILD_TAIL_MESSAGES = 5


def _build_deferred_tail_fixture(db_path: Path, *, n_children: int) -> tuple[sqlite3.Connection, str, str, str]:
    """Seed a parent session plus N children stored whole with an unresolved
    parent-link edge -- the exact on-disk shape #2467's deferred-tail
    extraction repairs once the parent becomes known.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_INDEX_DDL)

    parent_native_id = "big"
    parent_origin = "codex-session"
    parent_session_id = f"{parent_origin}:{parent_native_id}"

    conn.execute(
        "INSERT INTO sessions (origin, native_id, title, content_hash) VALUES (?, ?, 't', ?)",
        (parent_origin, parent_native_id, b"p" * 32),
    )

    parent_messages = []
    parent_blocks = []
    for position in range(_PARENT_MESSAGES):
        native_id = f"m{position}"
        message_id = f"{parent_session_id}:{native_id}"
        role = "user" if position % 2 == 0 else "assistant"
        parent_messages.append((parent_session_id, native_id, position, role, b"x" * 32))
        parent_blocks.append((message_id, parent_session_id, 0, "text", f"text-{position}"))
    conn.executemany(
        "INSERT INTO messages (session_id, native_id, position, role, content_hash) VALUES (?, ?, ?, ?, ?)",
        parent_messages,
    )
    conn.executemany(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, ?, ?, ?)",
        parent_blocks,
    )

    total_child_messages = _PARENT_MESSAGES + _CHILD_TAIL_MESSAGES
    for child_index in range(n_children):
        child_native_id = f"child{child_index}"
        child_session_id = f"{parent_origin}:{child_native_id}"
        conn.execute(
            "INSERT INTO sessions (origin, native_id, title, content_hash) VALUES (?, ?, 't', ?)",
            (parent_origin, child_native_id, bytes([child_index % 256]) * 32),
        )
        child_messages = []
        child_blocks = []
        child_events = []
        for position in range(total_child_messages):
            native_id = f"m{position}"
            message_id = f"{child_session_id}:{native_id}"
            role = "user" if position % 2 == 0 else "assistant"
            # Shared-prefix positions reuse the parent's exact text so the
            # composed-signature comparison walks the full shared prefix;
            # tail positions diverge (this child's own new content).
            text = f"text-{position}" if position < _PARENT_MESSAGES else f"child{child_index}-own-{position}"
            child_messages.append((child_session_id, native_id, position, role, b"x" * 32))
            child_blocks.append((message_id, child_session_id, 0, "text", text))
            child_events.append((child_session_id, message_id, position, "message_event", "e"))
        conn.executemany(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) VALUES (?, ?, ?, ?, ?)",
            child_messages,
        )
        conn.executemany(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, ?, ?, ?)",
            child_blocks,
        )
        conn.executemany(
            "INSERT INTO session_events (session_id, source_message_id, position, event_type, summary, payload_json) "
            "VALUES (?, ?, ?, ?, ?, '{}')",
            child_events,
        )
        # Unresolved parent-link edge (inheritance IS NULL): the exact
        # #2467 "child ingested before parent" state.
        conn.execute(
            """
            INSERT INTO session_links (
                src_session_id, dst_origin, dst_native_id, link_type,
                branch_point_message_id, inheritance,
                status, method, confidence, evidence_json, observed_at_ms
            ) VALUES (?, ?, ?, 'resume', NULL, NULL, NULL, 'parser-parent', 1.0, '{}', 0)
            """,
            (child_session_id, parent_origin, parent_native_id),
        )
    conn.commit()
    return conn, parent_session_id, parent_native_id, parent_origin


def _time_graph_resolve(n_children: int, tmp_path: Path) -> float:
    conn, session_id, native_id, origin = _build_deferred_tail_fixture(
        tmp_path / f"deferred_tail_{n_children}.db", n_children=n_children
    )
    try:
        started_at = time.perf_counter()
        archive_tier_write._resolve_session_graph(conn, session_id, native_id, origin, cache={})
        conn.commit()
        return time.perf_counter() - started_at
    finally:
        conn.close()


@pytest.mark.benchmark
def test_graph_resolve_deferred_tail_scales_linearly_not_quadratically(tmp_path: Path) -> None:
    """graph_resolve cost for N orphaned children must scale ~linearly in N.

    Guards against a *new* accidental quadratic regression in
    ``_resolve_session_graph``/``_reextract_prefix_tail_db`` (e.g. losing the
    ``composed_cache`` sharing across siblings, or re-walking already-resolved
    children). The existing linear cost (per-child O(shared_prefix_size) row
    mutation) is a known, evidence-backed characteristic documented in
    polylogue-3wb -- this test is not expected to make it fast, only to catch
    it getting asymptotically worse.
    """
    small_seconds = _time_graph_resolve(5, tmp_path)
    large_seconds = _time_graph_resolve(40, tmp_path)

    # 8x the children at the same per-child prefix size. Linear cost implies
    # ~8x wall time; quadratic implies ~64x. Allow generous headroom (20x)
    # for fixed per-call overhead and host noise while still catching a
    # genuine quadratic regression.
    ratio = large_seconds / max(small_seconds, 1e-9)
    print(
        f"\ngraph_resolve deferred-tail scaling: 5 children={small_seconds:.4f}s, "
        f"40 children={large_seconds:.4f}s, ratio={ratio:.1f}x (linear expects ~8x)"
    )
    assert ratio < 20, (
        f"graph_resolve cost grew {ratio:.1f}x for 8x the orphaned children "
        f"(5={small_seconds:.4f}s, 40={large_seconds:.4f}s) -- expected roughly linear "
        "(~8x); this smells like a new quadratic regression in "
        "_resolve_session_graph/_reextract_prefix_tail_db, not the known linear "
        "per-child cost documented in polylogue-3wb."
    )
