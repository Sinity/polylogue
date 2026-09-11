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
from typing import Any, cast

import pytest

from polylogue.storage.embeddings import materialization
from polylogue.storage.embeddings.materialization import _record_archive_embedding_success
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


@dataclass(frozen=True, slots=True)
class _FakeCfg:
    """Minimal stand-in for ``PolylogueConfig`` exposing only what
    ``_record_archive_embedding_success`` reads."""

    embedding_model: str


# Verbatim from _reconcile_embedding_config_change (convergence_stages.py:676),
# the bulk mark issued when a configured model/dimension change is detected.
_BULK_MARK_NEEDS_REINDEX_SQL = "UPDATE embedding_status SET needs_reindex = 1, error_message = NULL"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        conn.close()
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
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


# ── Drift during lease-free computation (polylogue-c0l7n) ───────────────────
#
# The provider call now runs with no writer lease and no generation lock, so
# the window in which the session's source can move under an in-flight embed
# pass is real and open by design. Publication must therefore refuse a session
# whose inputs moved, rather than land vectors computed from an older session.

_DRIFT_TEXT = "The prose this attempt was computed from."
_REPLACEMENT_TEXT = "Different prose, written while the provider was working."


def _write_single_message_session(root: Path, *, native_id: str, text: str) -> str:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    with ArchiveStore(root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )


def test_source_mutation_during_provider_call_is_refused_not_published(tmp_path: Path) -> None:
    """A session edited while its vectors were being computed stays unpublished.

    Anti-vacuity: delete the source-hash/message-count revalidation in
    ``_finalize_archive_embedding_attempt`` and this pass reports ``embedded``
    while ``embedding_status`` claims freshness for a session whose prose has
    already changed.
    """
    from polylogue.config import load_polylogue_config
    from polylogue.storage.embeddings.materialization import embed_archive_session_sync
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    root = tmp_path / "archive"
    session_id = _write_single_message_session(root, native_id="drift-source", text=_DRIFT_TEXT)
    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    probe = sqlite3.connect(embeddings_db)
    loaded, error = try_load_sqlite_vec(probe)
    probe.close()
    if not loaded:
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")

    configured_model = load_polylogue_config().embedding_model

    class _MutatingProvider:
        model = configured_model
        dimension = 1024

        def __init__(self) -> None:
            self.calls = 0

        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            # Exactly the window this change opens: no writer lease, no
            # generation lock, so an unrelated writer really can land here.
            self.calls += 1
            with sqlite3.connect(index_db) as conn:
                conn.execute("UPDATE blocks SET text = ? WHERE session_id = ?", (_REPLACEMENT_TEXT, session_id))
                conn.commit()
            return [[0.25] * self.dimension for _ in texts]

    provider = _MutatingProvider()
    outcome = embed_archive_session_sync(index_db, cast(Any, provider), session_id)

    assert provider.calls == 1
    assert outcome.status == "error"
    assert outcome.error is not None
    assert "source or recipe changed" in outcome.error

    with sqlite3.connect(embeddings_db) as conn:
        conn.row_factory = sqlite3.Row
        status = conn.execute(
            "SELECT needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
        ).fetchone()
        state = conn.execute(
            "SELECT attempt_state FROM embedding_derivation_state WHERE session_id = ?", (session_id,)
        ).fetchone()
    assert status is not None
    assert status["needs_reindex"] == 1, "a refused publication must not leave a fresh-status receipt"
    assert state is not None
    assert state["attempt_state"] == "pending", "the superseded attempt must be re-reserved for retry"
