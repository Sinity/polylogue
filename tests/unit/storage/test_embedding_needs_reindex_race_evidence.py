"""Race evidence for polylogue-y337 (get->modify->put audit, polylogue-9e5.4).

``_reconcile_embedding_config_change`` (``polylogue/daemon/convergence_stages.py``)
runs on every archive-embed freshness *check* call, not just at daemon
startup. When it detects the configured embedding model/dimension no longer
matches ``message_embeddings_meta``, it bulk-marks every ``embedding_status``
row ``needs_reindex = 1`` so those sessions get re-embedded.

``_record_archive_embedding_success`` (``polylogue/storage/embeddings/materialization.py``)
is the terminal write of one archive-session embed pass. The fix for
polylogue-y337 makes that write's ``needs_reindex`` clear *conditional* on
the model the pass actually used (``model=`` kwarg, threaded from
``text_provider.model`` at the real call site) still matching the
*currently configured* model at write time: if the model moved on since the
pass started reading messages, the just-written embeddings are already
stale, so ``needs_reindex`` is left at 1 instead of being blindly cleared.

This test exercises that fix directly against the real ``embedding_status``
DDL and the real ``_record_archive_embedding_success`` function, using the
same SQL ``_reconcile_embedding_config_change`` runs at
``polylogue/daemon/convergence_stages.py:676`` for the bulk mark (reproduced
verbatim here rather than importing the full function, which pulls in
config loading and the sqlite-vec runtime — orthogonal to the race itself).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.storage.embeddings import materialization
from polylogue.storage.embeddings.materialization import _record_archive_embedding_success


@dataclass(frozen=True, slots=True)
class _FakeCfg:
    """Minimal stand-in for ``PolylogueConfig`` exposing only what
    ``_record_archive_embedding_success`` reads."""

    embedding_model: str


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


def test_embedding_success_write_does_not_clobber_concurrent_reindex_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "codex-session:race-session"
    db_path = tmp_path / "embeddings.db"
    conn = _connect(db_path)

    old_model = "voyage-3"
    new_model = "voyage-4"

    # Baseline: session was embedded successfully under the currently
    # configured model (old_model), needs_reindex=0.
    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: _FakeCfg(embedding_model=old_model))
    _record_archive_embedding_success(
        conn, session_id=session_id, origin="codex-session", message_count=3, model=old_model
    )
    row = conn.execute("SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)).fetchone()
    assert row is not None
    assert row["needs_reindex"] == 0

    # --- Actor A (an in-flight _archive_embed_execute* pass) has already
    # read this session's messages and is generating embeddings under
    # old_model (represented here just by the passage of time between the
    # baseline write above and Actor A's terminal write below — it captured
    # ``old_model`` from its own ``text_provider.model`` before Actor B's
    # config change landed).
    #
    # --- Actor B (a freshness *check* probe running _reconcile_embedding_
    # config_change, e.g. triggered by an operator changing
    # POLYLOGUE_EMBEDDING_MODEL) detects the configured model no longer
    # matches message_embeddings_meta and bulk-marks every row for reindex,
    # landing on its own connection/transaction WHILE Actor A is still
    # mid-flight. The config is now `new_model`.
    actor_b_conn = sqlite3.connect(db_path)
    actor_b_conn.execute(_BULK_MARK_NEEDS_REINDEX_SQL)
    actor_b_conn.commit()
    actor_b_conn.close()
    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: _FakeCfg(embedding_model=new_model))

    row_after_b = conn.execute(
        "SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
    ).fetchone()
    assert row_after_b is not None
    assert row_after_b["needs_reindex"] == 1, "Actor B's reindex request should be visible before Actor A finishes"

    # Actor A now finishes the embed pass it started before Actor B's mark
    # landed and writes its terminal success — still tagged with the
    # ``old_model`` it actually embedded under. Because the currently
    # configured model has since moved on to ``new_model``, the fix must NOT
    # clear needs_reindex: Actor A's embeddings are already stale.
    _record_archive_embedding_success(
        conn, session_id=session_id, origin="codex-session", message_count=3, model=old_model
    )

    final_row = conn.execute(
        "SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
    ).fetchone()
    assert final_row is not None
    assert final_row["needs_reindex"] == 1, (
        "fix regressed: Actor A's stale-model success write must not clobber Actor B's still-pending reindex request"
    )

    conn.close()


def test_embedding_success_write_clears_reindex_when_model_matches_current_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: a success write under the *currently* configured
    model must still clear needs_reindex normally — the fix should not make
    every embed pass permanently sticky."""
    session_id = "codex-session:fresh-session"
    db_path = tmp_path / "embeddings.db"
    conn = _connect(db_path)

    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: _FakeCfg(embedding_model="voyage-4"))
    conn.execute(
        "INSERT INTO embedding_status (session_id, origin, needs_reindex) VALUES (?, ?, 1)",
        (session_id, "codex-session"),
    )
    conn.commit()

    _record_archive_embedding_success(
        conn, session_id=session_id, origin="codex-session", message_count=5, model="voyage-4"
    )

    final_row = conn.execute(
        "SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
    ).fetchone()
    assert final_row is not None
    assert final_row["needs_reindex"] == 0

    conn.close()
