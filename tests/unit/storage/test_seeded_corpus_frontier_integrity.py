"""Fixture authority: the shared seeded corpus must satisfy frontier integrity.

polylogue-ku00r asked whether the shared seeded corpus violates raw frontier
integrity -- reported as 2 broken accepted heads -- and, if so, whether the
invariant is over-strict for a legitimately-shaped small archive or the seeding
helper builds a genuinely inconsistent chain.

Neither, as measured here: the corpus is clean, and these tests pin that so it
cannot regress silently again. The suite has no other check that the corpus
every snapshot test reads is itself production-valid, which is why the original
symptom could only surface as a confusing snapshot diff.

The corpus is two ChatGPT-export sessions whose raws are FULL baselines
(``acquisition_generation=0``, no predecessor, ``asserted`` authority), and
``raw_revision_heads`` is empty. An empty head table is CORRECT for a plain
import: accepted heads are established by the revision-replay/membership
governance path, not by ordinary ingest. So ``_check_broken_active_chains``
validates the two session raws through ``_validate_active_revision_chain`` and
finds nothing broken -- there is no byte head to validate against.

The second test is the red twin. Without it, the first test cannot distinguish
"the invariant holds" from "the invariant is asleep" -- and an integrity check
that silently passes everything is the more dangerous failure, because it is
what would let a genuinely broken corpus back in.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from polylogue.storage.raw_retention import RawFrontierIntegritySnapshot, raw_frontier_integrity_snapshot
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _snapshot(root: Path) -> RawFrontierIntegritySnapshot:
    conn = open_readonly_connection(root / "source.db")
    try:
        return raw_frontier_integrity_snapshot(
            conn,
            index_db_path=root / "index.db",
            ops_db_path=root / "ops.db",
        )
    finally:
        conn.close()


def test_seeded_corpus_satisfies_raw_frontier_integrity(
    named_seeded_archive: Callable[[str], Path],
) -> None:
    """polylogue-ku00r: the corpus every snapshot test reads is production-valid."""
    root = named_seeded_archive("cli-chatgpt").parent

    snapshot = _snapshot(root)

    assert snapshot.broken_head_status == "healthy", snapshot.broken_head_reason
    assert snapshot.broken_head_count == 0
    assert snapshot.broken_head_samples == ()
    # Anti-vacuity on the count itself: a corpus that checked nothing would also
    # report zero broken heads.
    assert snapshot.broken_head_checked_count == 2


def test_broken_predecessor_chain_in_the_same_corpus_is_reported(
    named_seeded_archive: Callable[[str], Path],
) -> None:
    """Red twin: the invariant is not vacuously green on this corpus shape.

    Point one session's raw at a predecessor that does not exist. That is a
    genuinely broken active chain, and the check must name it -- otherwise the
    green result above proves nothing about the corpus.
    """
    root = named_seeded_archive("cli-chatgpt").parent

    conn = sqlite3.connect(root / "source.db")
    with conn:
        raw_id = str(conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id LIMIT 1").fetchone()[0])
        conn.execute(
            """
            UPDATE raw_sessions
            SET revision_kind = 'append',
                predecessor_raw_id = 'polylogue-ku00r-missing-predecessor',
                baseline_raw_id = 'polylogue-ku00r-missing-predecessor'
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
    conn.close()

    snapshot = _snapshot(root)

    assert snapshot.broken_head_status != "healthy"
    assert snapshot.broken_head_count >= 1
    assert any(raw_id == sample.accepted_raw_id for sample in snapshot.broken_head_samples), (
        snapshot.broken_head_samples
    )
