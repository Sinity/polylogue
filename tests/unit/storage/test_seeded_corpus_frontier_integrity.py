"""Fixture authority: the shared seeded corpus must satisfy frontier integrity.

polylogue-ku00r asked whether the shared seeded corpus violates raw frontier
integrity -- reported as 2 broken accepted heads -- and, if so, whether the
invariant is over-strict for a legitimately-shaped small archive or the seeding
helper builds a genuinely inconsistent chain.

Neither, as measured here: the corpus is clean, and these tests pin that so it
cannot regress silently again. The suite has no other check that the corpus
every snapshot test reads is itself production-valid, which is why the original
symptom could only surface as a confusing snapshot diff.

The corpus is two ChatGPT-export sessions. Single-pass ingest settles each
through membership governance, so ``raw_revision_heads`` holds two semantic
heads and their raws are retired from byte-revision governance. The frontier
check visits both session raws and finds nothing broken; a semantic head has
no byte predecessor chain to validate.

The second test is the red twin. Without it, the first test cannot distinguish
"the invariant holds" from "the invariant is asleep" -- and an integrity check
that silently passes everything is the more dangerous failure, because it is
what would let a genuinely broken corpus back in.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from polylogue.storage.raw_retention import RawFrontierIntegritySnapshot, raw_frontier_integrity_snapshot
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.workload_artifacts import SeededArchiveClone, SeededArchiveQueryLease

pytest_plugins = ("tests.infra.corpus_fixtures",)


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
    named_seeded_archive_ro: Callable[[str], SeededArchiveQueryLease],
) -> None:
    """polylogue-ku00r: the corpus every snapshot test reads is production-valid."""
    root = named_seeded_archive_ro("cli-chatgpt").root

    snapshot = _snapshot(root)

    assert snapshot.broken_head_status == "healthy", snapshot.broken_head_reason
    assert snapshot.broken_head_count == 0
    assert snapshot.broken_head_samples == ()
    # Anti-vacuity on the count itself: a corpus that checked nothing would also
    # report zero broken heads.
    assert snapshot.broken_head_checked_count == 2


def test_broken_predecessor_chain_in_the_same_corpus_is_reported(
    named_seeded_archive_rw: Callable[[str], SeededArchiveClone],
) -> None:
    """Red twin: the invariant is not vacuously green on this corpus shape.

    Remove the raw one accepted head names from the source tier. That is a
    genuinely broken active chain, and the check must name it -- otherwise the
    green result above proves nothing about the corpus. The corpus heads are
    membership-governed semantic heads, which carry no byte predecessor chain,
    so a forged byte predecessor on their raw is not evidence they consult.
    """
    root = named_seeded_archive_rw("cli-chatgpt").root

    with closing(sqlite3.connect(root / "index.db")) as index:
        raw_id = str(
            index.execute("SELECT accepted_raw_id FROM raw_revision_heads ORDER BY accepted_raw_id LIMIT 1").fetchone()[
                0
            ]
        )
    conn = sqlite3.connect(root / "source.db")
    with conn:
        conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (raw_id,))
    conn.close()

    snapshot = _snapshot(root)

    assert snapshot.broken_head_status != "healthy"
    assert snapshot.broken_head_count >= 1
    assert any(raw_id == sample.accepted_raw_id for sample in snapshot.broken_head_samples), (
        snapshot.broken_head_samples
    )
