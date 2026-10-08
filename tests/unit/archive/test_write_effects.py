from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from polylogue.archive.write_effects import (
    WRITE_EFFECT_REGISTRY,
    WriteEffect,
    WriteEffectContext,
    commit_archive_write_effects,
)
from polylogue.archive.write_gateway import WriteEffectReceipt, WriteOperation, WriteResult
from polylogue.storage.sqlite.connection import open_connection


def test_registry_declares_the_canonical_effects_in_order() -> None:
    """The registry is the single source of truth for effect order and phase."""
    assert [effect.name for effect in WRITE_EFFECT_REGISTRY] == [
        "ensure_fts_triggers",
        "repair_message_fts",
        "invalidate_search_cache",
        "announce_ingest_committed",
        "invalidate_session_insights",
    ]
    assert [effect.phase for effect in WRITE_EFFECT_REGISTRY] == [
        "in-transaction",
        "in-transaction",
        "post-commit",
        # The bus announcement is post-commit on purpose: a subscriber woken by
        # an uncommitted write could read rows that still roll back.
        "post-commit",
        "in-transaction",
    ]
    assert [effect.failure_policy for effect in WRITE_EFFECT_REGISTRY] == [
        "abort",
        "abort",
        "abort",
        "log-and-continue",
        "log-and-continue",
    ]


def test_write_result_receipt_annotation_resolves_at_runtime() -> None:
    assert get_type_hints(WriteResult)["effect_receipts"] == tuple[WriteEffectReceipt, ...]


def test_commit_write_effects_positive_case_runs_fts_repair_and_cache_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded positive case: changed session ids drive both the in-transaction
    FTS repair effect and the post-commit cache-invalidation effect."""
    invalidated: list[bool] = []
    repaired: list[tuple[str, ...]] = []

    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, ids: repaired.append(tuple(ids)),
    )
    monkeypatch.setattr(
        "polylogue.storage.search.cache.invalidate_search_cache",
        lambda: invalidated.append(True),
    )

    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = commit_archive_write_effects(
            conn,
            WriteOperation.INGEST,
            {"changed_session_ids": ("c2", "c1", "c1")},
        )

    assert result.status == "committed"
    assert result.rows_affected == 2
    assert repaired == [("c1", "c2")]
    assert invalidated == [True]
    insight_invalidation = next(
        receipt for receipt in result.effect_receipts if receipt.name == "invalidate_session_insights"
    )
    assert insight_invalidation.disposition == "applied"
    assert insight_invalidation.phase == "in-transaction"


def test_commit_write_effects_degraded_case_skips_conditional_effects_when_no_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degraded/empty case: no changed session ids means the always-run
    trigger-ensure effect still fires but the two conditional effects
    (repair_message_fts, invalidate_search_cache) do not."""
    ensured: list[bool] = []
    invalidated: list[bool] = []
    repaired: list[bool] = []

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync",
        lambda _conn: ensured.append(True),
    )
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, _ids: repaired.append(True),
    )
    monkeypatch.setattr(
        "polylogue.storage.search.cache.invalidate_search_cache",
        lambda: invalidated.append(True),
    )

    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = commit_archive_write_effects(conn, WriteOperation.INGEST, {"changed_session_ids": ()})

    assert result.status == "committed"
    assert result.rows_affected == 0
    assert ensured == [True]
    assert repaired == []
    assert invalidated == []
    assert result.effect_receipts[-1].disposition == "skipped"


def test_repair_message_fts_should_run_honors_explicit_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repaired: list[bool] = []
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, _ids: repaired.append(True),
    )
    monkeypatch.setattr("polylogue.storage.search.cache.invalidate_search_cache", lambda: None)

    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        commit_archive_write_effects(
            conn,
            WriteOperation.INGEST,
            {"changed_session_ids": ("c1",), "repair_message_fts": False},
        )

    assert repaired == []


def test_log_and_continue_failure_policy_does_not_poison_later_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A log-and-continue effect that raises must not prevent a later
    in-registry effect from running — this is the failure-isolation
    property the bead's design calls out as the real bug class fixed by
    declaring failure policy per effect rather than one bare function body."""
    from polylogue.archive import write_effects as write_effects_module

    ran_after: list[bool] = []

    def _boom(_ctx: WriteEffectContext) -> None:
        raise RuntimeError("simulated effect failure")

    def _after(_ctx: WriteEffectContext) -> bool:
        ran_after.append(True)
        return False

    isolated_registry = (
        WriteEffect(name="boom", phase="in-transaction", run=_boom, failure_policy="log-and-continue"),
        WriteEffect(name="after", phase="in-transaction", run=lambda _ctx: None, should_run=_after),
    )
    monkeypatch.setattr(write_effects_module, "WRITE_EFFECT_REGISTRY", isolated_registry)

    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = commit_archive_write_effects(conn, WriteOperation.INGEST, {"changed_session_ids": ()})

    assert result.status == "committed"
    assert ran_after == [True]


def test_abort_failure_policy_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.archive import write_effects as write_effects_module

    def _boom(_ctx: WriteEffectContext) -> None:
        raise RuntimeError("simulated effect failure")

    monkeypatch.setattr(
        write_effects_module,
        "WRITE_EFFECT_REGISTRY",
        (WriteEffect(name="boom", phase="in-transaction", run=_boom, failure_policy="abort"),),
    )

    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError, match="simulated effect failure"):
            commit_archive_write_effects(conn, WriteOperation.INGEST, {"changed_session_ids": ()})
        # The caller owns the connection lifecycle: an aborted effect leaves its
        # transaction open, and the owner settles it before close.
        assert conn.in_transaction
        conn.rollback()


def test_tolerated_effect_failure_has_failed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.archive import write_effects as module

    effect = WriteEffect(
        name="tolerated",
        phase="in-transaction",
        run=lambda _ctx: (_ for _ in ()).throw(RuntimeError("effect broke")),
        failure_policy="log-and-continue",
    )
    monkeypatch.setattr(module, "WRITE_EFFECT_REGISTRY", (effect,))
    with open_connection(tmp_path / "archive.db") as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = commit_archive_write_effects(conn, WriteOperation.INGEST, {})
    assert result.effect_receipts == (
        module.WriteEffectReceipt("tolerated", "in-transaction", "failed", error="effect broke"),
    )


def test_insight_invalidation_rides_the_admitted_transaction(tmp_path: Path) -> None:
    """Profile invalidation commits or rolls back with the caller's write.

    #5727 moved this effect from an async-deferred reopen by path into the
    admitted transaction, so there is no later delivery to follow a repointed
    index. Anti-vacuity: an effect that opened its own connection by path would
    commit independently, and the rollback below would leave the key cleared.
    """
    import sqlite3
    from typing import Any, cast

    from polylogue.archive.write_effects import WriteEffectContext, _invalidate_insights_effect

    path = tmp_path / "index.db"
    with sqlite3.connect(path) as setup:
        setup.execute("CREATE TABLE session_profiles (session_id TEXT, source_sort_key TEXT, source_updated_at TEXT)")
        setup.execute("INSERT INTO session_profiles VALUES ('s1', 'k', 'u')")

    def invalidate(conn: sqlite3.Connection) -> None:
        _invalidate_insights_effect(
            WriteEffectContext(
                conn=conn,
                op=cast(Any, None),
                payload={},
                changed_session_ids=("s1",),
                staleness_key="k",
                run_archive_effects=True,
            )
        )

    def stored_key() -> object:
        with sqlite3.connect(path) as reader:
            return reader.execute("SELECT source_sort_key FROM session_profiles").fetchone()[0]

    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        invalidate(conn)
        conn.execute("ROLLBACK")
        assert stored_key() == "k"

        conn.execute("BEGIN IMMEDIATE")
        invalidate(conn)
        conn.execute("COMMIT")
    finally:
        conn.close()
    assert stored_key() is None
