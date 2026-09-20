"""``record_manual_continuation`` must write a row the index tier accepts.

Two independent writer/vocabulary disagreements made the route unusable; the
first masked the second.

``session_links.status`` carries ``TopologyEdgeStatus`` -- an exceptional
marker (repaired / quarantined / authority-contradicted) recording *why* an
edge needed intervention. Ordinary resolvedness is carried by
``resolved_dst_session_id IS NOT NULL``. The route wrote ``status='resolved'``,
a value the column's generated CHECK has never admitted, so every call raised
``sqlite3.IntegrityError`` and the operator's handoff assertion was written
while the lineage edge it describes was not (polylogue-pkst AC2: the generated
DDL and the normal writer must agree).

The handoff assertion then wrote ``author_ref="service:polylogue"``, and
``service`` is not a declared ``ObjectRef`` kind, so ``upsert_assertion``
raised on the ref parse. Both are the same class of defect: a writer naming a
value its declared vocabulary never admitted.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _archive_with_sessions(tmp_path: Path) -> Polylogue:
    with ArchiveStore(tmp_path):
        pass
    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        for native_id in ("child", "parent"):
            conn.execute(
                "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, 'codex-session', zeroblob(32))",
                (native_id,),
            )
        conn.commit()
    finally:
        conn.close()
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")


def test_manual_continuation_records_a_storable_lineage_edge(tmp_path: Path) -> None:
    """The edge lands, marked spawned-fresh and resolved to the named parent.

    Anti-vacuity: restore ``status='resolved'`` in the INSERT and this fails
    with ``IntegrityError: CHECK constraint failed``; restore
    ``author_ref="service:polylogue"`` and it fails with ``unsupported object
    ref kind``; drop ``resolved_dst_session_id`` instead and the resolved
    parent assertion goes red, so the test is not merely proving the write
    did not raise. The handoff assertion is read back so the second fix is
    covered by an outcome, not by the absence of an exception.
    """
    archive = _archive_with_sessions(tmp_path)

    asyncio.run(archive.record_manual_continuation("codex-session:child", "codex-session:parent"))

    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        rows = conn.execute(
            "SELECT src_session_id, link_type, inheritance, status, resolved_dst_session_id, "
            "branch_point_message_id, method FROM session_links"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        (
            "codex-session:child",
            "continuation",
            "spawned-fresh",
            None,
            "codex-session:parent",
            None,
            "manual-continuation",
        )
    ]

    user = sqlite3.connect(tmp_path / "user.db")
    try:
        handoffs = user.execute("SELECT target_ref, kind, author_ref, author_kind, status FROM assertions").fetchall()
    finally:
        user.close()
    assert handoffs == [("session:codex-session:child", "handoff", "actor:polylogue", "service", "candidate")]


def test_manual_continuation_still_refuses_an_absent_parent(tmp_path: Path) -> None:
    """The route's own preconditions are unchanged by the storage fix."""
    archive = _archive_with_sessions(tmp_path)

    with pytest.raises(ValueError, match="parent session does not exist"):
        asyncio.run(archive.record_manual_continuation("codex-session:child", "codex-session:absent"))
