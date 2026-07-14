from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.write_effects import (
    WRITE_EFFECT_REGISTRY,
    WriteEffect,
    WriteEffectContext,
    commit_archive_write_effects,
)
from polylogue.archive.write_gateway import WriteOperation
from polylogue.storage.sqlite.connection import open_connection


def test_registry_declares_the_three_canonical_effects_in_order() -> None:
    """Slice 1 of polylogue-0aj: the registry is the single source of truth
    for what commit_archive_write_effects runs, and this pins its shape so a
    future addition/removal is a deliberate, reviewable diff here."""
    assert [effect.name for effect in WRITE_EFFECT_REGISTRY] == [
        "ensure_fts_triggers",
        "repair_message_fts",
        "invalidate_search_cache",
    ]
    assert [effect.phase for effect in WRITE_EFFECT_REGISTRY] == [
        "in-transaction",
        "in-transaction",
        "post-commit",
    ]
    # Failure policy defaults to "abort" unless an effect opts into
    # log-and-continue; all three canonical effects are today's
    # already-established abort-on-failure behavior.
    assert all(effect.failure_policy == "abort" for effect in WRITE_EFFECT_REGISTRY)


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
        lambda _conn, ids, **_kwargs: repaired.append(tuple(ids)),
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
        lambda _conn, _ids, **_kwargs: repaired.append(True),
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


def test_repair_message_fts_should_run_honors_explicit_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repaired: list[bool] = []
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, _ids, **_kwargs: repaired.append(True),
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
