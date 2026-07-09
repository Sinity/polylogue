"""Race evidence for polylogue-9e5.4.3 (get->modify->put audit, polylogue-9e5.4).

``_reconcile_embedding_config_change`` (``polylogue/daemon/convergence_stages.py``)
runs on every archive-embed freshness *check* call, not just at daemon
startup. When it detects the configured embedding model/dimension no longer
matches ``message_embeddings_meta``, it bulk-marks every ``embedding_status``
row ``needs_reindex = 1`` so those sessions get re-embedded.

``_record_archive_embedding_success`` (``polylogue/storage/embeddings/materialization.py``)
is the terminal write of one archive-session embed pass: it unconditionally
sets ``needs_reindex = 0`` on success, regardless of whether the row's
reindex requirement was set *after* that embed pass started reading messages
(i.e. the embeddings it just computed may already be stale under the newly
configured model).

Neither write is conditioned on the other (no generation/version column
gates the transition), so if the bulk reindex mark lands in between an
in-flight embed pass's read and its terminal success write, the success
write silently clobbers the reindex requirement — the session is left
marked "fresh" while holding embeddings computed under the superseded
model/dimension.

This test is evidence for the bug bead, not a fix. It reproduces the
*mechanism* directly against the real ``embedding_status`` DDL and the real
``_record_archive_embedding_success`` function, using the same SQL
``_reconcile_embedding_config_change`` runs at
``polylogue/daemon/convergence_stages.py:676`` for the bulk mark (reproduced
verbatim here rather than importing the full function, which pulls in
config loading and the sqlite-vec runtime — orthogonal to the race itself).
A fix landing under the follow-up bead should update this assertion.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.embeddings.materialization import _record_archive_embedding_success

# Mirrors polylogue/storage/sqlite/archive_tiers/embeddings.py::EMBEDDINGS_DDL,
# minus the vec0 virtual table and message_embeddings_meta (not needed to
# exercise this race).
_EMBEDDING_STATUS_DDL = """
CREATE TABLE embedding_status (
    session_id                 TEXT PRIMARY KEY,
    origin                     TEXT NOT NULL DEFAULT '',
    message_count_embedded     INTEGER NOT NULL DEFAULT 0 CHECK(message_count_embedded >= 0),
    last_embedded_at_ms        INTEGER,
    needs_reindex              INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1)),
    error_message              TEXT
) STRICT;
"""

# Verbatim from _reconcile_embedding_config_change (convergence_stages.py:676),
# the bulk mark issued when a configured model/dimension change is detected.
_BULK_MARK_NEEDS_REINDEX_SQL = "UPDATE embedding_status SET needs_reindex = 1, error_message = NULL"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_EMBEDDING_STATUS_DDL)
    conn.commit()
    return conn


def test_embedding_success_write_clobbers_concurrent_reindex_request(tmp_path: Path) -> None:
    session_id = "codex-session:race-session"
    db_path = tmp_path / "embeddings.db"
    conn = _connect(db_path)

    # Baseline: session was embedded successfully under the currently
    # configured model, needs_reindex=0.
    _record_archive_embedding_success(conn, session_id=session_id, origin="codex-session", message_count=3)
    row = conn.execute("SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)).fetchone()
    assert row is not None
    assert row["needs_reindex"] == 0

    # --- Actor A (an in-flight _archive_embed_execute* pass) has already
    # read this session's messages and is generating embeddings under the
    # OLD model/dimension (represented here just by the passage of time
    # between the baseline write above and Actor A's terminal write below).
    #
    # --- Actor B (a freshness *check* probe running _reconcile_embedding_
    # config_change, e.g. triggered by an operator changing
    # POLYLOGUE_EMBEDDING_MODEL) detects the configured model no longer
    # matches message_embeddings_meta and bulk-marks every row for reindex,
    # landing on its own connection/transaction WHILE Actor A is still
    # mid-flight.
    actor_b_conn = sqlite3.connect(db_path)
    actor_b_conn.execute(_BULK_MARK_NEEDS_REINDEX_SQL)
    actor_b_conn.commit()
    actor_b_conn.close()

    row_after_b = conn.execute(
        "SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
    ).fetchone()
    assert row_after_b is not None
    assert row_after_b["needs_reindex"] == 1, "Actor B's reindex request should be visible before Actor A finishes"

    # Actor A now finishes the embed pass it started before Actor B's mark
    # landed, and writes its terminal success — unconditionally clearing
    # needs_reindex regardless of Actor B's intervening request.
    _record_archive_embedding_success(conn, session_id=session_id, origin="codex-session", message_count=3)

    final_row = conn.execute(
        "SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
    ).fetchone()
    assert final_row is not None
    # Bug reproduced: Actor B's reindex requirement is silently lost even
    # though Actor A's embeddings were computed under the pre-change model.
    # A correct implementation would leave needs_reindex == 1 here (or
    # detect the staleness and re-embed before clearing the flag).
    assert final_row["needs_reindex"] == 0, (
        "race reproduced: expected Actor A's blind success write to clobber Actor B's "
        "reindex request (0); got a value suggesting the race no longer reproduces"
    )

    conn.close()
