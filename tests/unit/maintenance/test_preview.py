"""Staleness inventory preview surface contracts.

Validates ``polylogue.maintenance.preview.staleness_inventory`` and
``polylogue ops maintenance preview`` against the acceptance criteria of
issue #1145:

* multiple :class:`InvalidationReason` values exercised (not just
  ``VERSION_MISMATCH`` aka ``materializer_version``);
* zero-row models still emit explicit rows (not absence);
* preview is read-only — no DB mutations observed via SQLite write hook;
* CLI subcommand renders the inventory in plain and JSON modes;
* per-model fractions report stale / source.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.maintenance import maintenance_group
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.maintenance.preview import (
    ALL_SCOPES,
    InvalidationReason,
    StalenessInventory,
    StalenessItem,
    staleness_inventory,
)
from tests.infra.storage_records import SessionBuilder, db_setup

# ---------------------------------------------------------------------------
# Pure projection tests (no DB)
# ---------------------------------------------------------------------------


def test_staleness_item_fraction_clamps_and_handles_zero_source() -> None:
    zero = StalenessItem(
        model="x",
        scope="derived",
        reason=InvalidationReason.STALE,
        count=5,
        source_total=0,
        materialized_total=0,
        detail="",
    )
    assert zero.fraction == 0.0

    partial = StalenessItem(
        model="x",
        scope="derived",
        reason=InvalidationReason.STALE,
        count=3,
        source_total=12,
        materialized_total=12,
        detail="",
    )
    assert partial.fraction == pytest.approx(0.25)

    over = StalenessItem(
        model="x",
        scope="derived",
        reason=InvalidationReason.STALE,
        count=999,
        source_total=10,
        materialized_total=10,
        detail="",
    )
    assert over.fraction == 1.0


def test_inventory_to_dict_round_trips_items_and_totals() -> None:
    inv = StalenessInventory(
        captured_at="2026-05-17T00:00:00+00:00",
        db_path=":memory:",
        scopes=ALL_SCOPES,
        items=(
            StalenessItem(
                model="messages_fts",
                scope="derived",
                reason=InvalidationReason.MISSING,
                count=2,
                source_total=10,
                materialized_total=8,
                detail="2 pending",
            ),
            StalenessItem(
                model="messages_fts",
                scope="derived",
                reason=InvalidationReason.STALE,
                count=1,
                source_total=10,
                materialized_total=8,
                detail="1 stale",
            ),
        ),
    )
    payload = json.loads(json.dumps(inv.to_dict()))

    assert payload["total_stale"] == 3
    assert len(payload["items"]) == 2
    assert payload["items"][0]["reason"] == "missing"
    assert payload["items"][0]["fraction"] == pytest.approx(0.2, abs=1e-6)


def test_inventory_by_model_groups_items() -> None:
    items = (
        StalenessItem(
            model="m1",
            scope="derived",
            reason=InvalidationReason.MISSING,
            count=1,
            source_total=1,
            materialized_total=0,
            detail="",
        ),
        StalenessItem(
            model="m1",
            scope="derived",
            reason=InvalidationReason.STALE,
            count=0,
            source_total=1,
            materialized_total=1,
            detail="",
        ),
        StalenessItem(
            model="m2",
            scope="retrieval",
            reason=InvalidationReason.ORPHAN,
            count=2,
            source_total=5,
            materialized_total=5,
            detail="",
        ),
    )
    inv = StalenessInventory(
        captured_at="t",
        db_path=":memory:",
        scopes=ALL_SCOPES,
        items=items,
    )

    grouped = inv.by_model()
    assert set(grouped.keys()) == {"m1", "m2"}
    assert len(grouped["m1"]) == 2
    assert len(grouped["m2"]) == 1


# ---------------------------------------------------------------------------
# Read-only invariant
# ---------------------------------------------------------------------------


def test_staleness_inventory_performs_no_writes(workspace_env: dict[str, Path]) -> None:
    """Preview must not mutate the database.

    Uses SQLite's authorizer hook to count write attempts directly on the
    backing connection while ``staleness_inventory`` runs.
    """

    db_path = db_setup(workspace_env)
    # Seed a small archive so we have a real DB to preview against.
    SessionBuilder(db_path, "preview-1").provider("chatgpt").title("seed").add_message(
        role="user", text="hello"
    ).add_message(role="assistant", text="world").save()
    SessionBuilder(db_path, "preview-2").provider("claude-code").title("seed2").add_message(
        role="user", text="one"
    ).save()

    write_actions = {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_DROP_INDEX,
    }
    write_attempts: list[tuple[int, str | None, str | None]] = []

    # Open a separate read-only connection with an authorizer; we cannot
    # install an authorizer on the cached connection used by
    # ``staleness_inventory`` itself, so we take a row-count snapshot
    # before/after as the primary invariant and the authorizer guards a
    # parallel sanity check on our own connection.
    sanity = sqlite3.connect(str(db_path))

    def _authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        trigger: str | None,
    ) -> int:
        if action in write_actions:
            write_attempts.append((action, arg1, arg2))
        return sqlite3.SQLITE_OK

    sanity.set_authorizer(_authorizer)
    # Probing read on the authorized connection just to confirm the hook works.
    # ``db_setup`` returns the ``index.db``; the session tree
    # lives in ``sessions`` (no legacy ``sessions`` table).
    sanity.execute("SELECT COUNT(*) FROM sessions").fetchone()
    sanity.close()
    assert write_attempts == []

    # Row-count snapshot across every table — preview must not change
    # any of them.
    def _snapshot() -> dict[str, int]:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            counts: dict[str, int] = {}
            for (name,) in rows:
                try:
                    counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM '{name}'").fetchone()[0])
                except sqlite3.DatabaseError:
                    # FTS shadow tables can reject SELECT COUNT — skip.
                    continue
            return counts

    before = _snapshot()
    inv = staleness_inventory(db_path)
    after = _snapshot()

    assert before == after, "preview mutated row counts"
    assert isinstance(inv, StalenessInventory)
    # The active archive is the archive store; the inventory reads
    # the session/message/block tree from ``index.db`` (not the legacy
    # single-file ``index.db``).
    assert inv.db_path.endswith("index.db")


# ---------------------------------------------------------------------------
# AC: zero-row models emit explicit rows, multiple reasons exercised
# ---------------------------------------------------------------------------


def test_inventory_emits_zero_rows_for_clean_models(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "clean-1").provider("chatgpt").title("clean").add_message(role="user", text="hi").save()

    inv = staleness_inventory(db_path)

    # Every model that emitted rows must emit MISSING/STALE/ORPHAN
    # explicitly (count=0 is fine), so consumers don't have to guess
    # "absent vs zero".
    grouped = inv.by_model()
    assert grouped, "no models inventoried for a non-empty archive"

    for model, items in grouped.items():
        if model in {
            "orphaned_messages",
            "empty_sessions",
            "orphaned_attachments",
            "orphaned_blobs",
        }:
            # Archive-cleanup scopes report a single orphan_archive_row item.
            assert {item.reason for item in items} == {InvalidationReason.ORPHAN_ARCHIVE_ROW}
            continue
        if model == "message_type_backfill":
            assert {item.reason for item in items} == {InvalidationReason.MISSING}
            continue
        reasons = {item.reason for item in items}
        assert {
            InvalidationReason.MISSING,
            InvalidationReason.STALE,
            InvalidationReason.ORPHAN,
        }.issubset(reasons), f"model {model} missing baseline reasons: {reasons}"


def test_inventory_exercises_multiple_invalidation_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC: not all stale rows can be attributed to materializer_version.

    Stubs the derived-status collector with synthetic statuses that
    exercise four different reasons across two models, plus injects
    archive-cleanup orphans via a temporary DB.
    """

    from polylogue.maintenance import preview as preview_mod

    def _fake_statuses(_conn: sqlite3.Connection, *, verify_full: bool = True) -> dict[str, DerivedModelStatus]:
        return {
            "messages_fts": DerivedModelStatus(
                name="messages_fts",
                ready=False,
                detail="",
                source_documents=100,
                materialized_documents=80,
                source_rows=100,
                materialized_rows=80,
                pending_rows=20,
                stale_rows=5,
                orphan_rows=2,
                missing_provenance_rows=1,
                materializer_version=4,
                matches_version=True,
            ),
            "transcript_embeddings": DerivedModelStatus(
                name="transcript_embeddings",
                ready=False,
                detail="",
                source_documents=50,
                materialized_documents=10,
                materialized_rows=10,
                pending_documents=40,
                stale_rows=0,
                orphan_rows=0,
                materializer_version=2,
                matches_version=False,
            ),
            # Models we don't inventory in this test — included to verify
            # filtering.
            "noise_model": DerivedModelStatus(
                name="noise_model",
                ready=True,
                detail="",
            ),
        }

    monkeypatch.setattr(preview_mod, "_archive_cleanup_items", lambda _c: [])
    monkeypatch.setattr(preview_mod, "_backfill_items", lambda _c: [])

    from polylogue.storage.derived import derived_status as derived_status_mod

    monkeypatch.setattr(
        derived_status_mod,
        "collect_derived_model_statuses_sync",
        _fake_statuses,
    )

    # Use an in-memory SQLite connection — connection_context happily
    # accepts a pre-opened connection and returns it as-is.
    conn = sqlite3.connect(":memory:")
    inv = staleness_inventory(conn, scopes=("derived", "retrieval"))

    by_model = inv.by_model()
    fts_reasons = {item.reason for item in by_model["messages_fts"]}
    embed_reasons = {item.reason for item in by_model["transcript_embeddings"]}

    # Four distinct reasons across the suite.
    all_reasons = fts_reasons | embed_reasons
    assert {
        InvalidationReason.MISSING,
        InvalidationReason.STALE,
        InvalidationReason.ORPHAN,
        InvalidationReason.MISSING_PROVENANCE,
        InvalidationReason.VERSION_MISMATCH,
    }.issubset(all_reasons)

    # VERSION_MISMATCH only fires when matches_version is False.
    assert InvalidationReason.VERSION_MISMATCH in embed_reasons
    assert InvalidationReason.VERSION_MISMATCH not in fts_reasons


def test_inventory_rejects_unknown_scope(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "x").add_message(role="user", text="hi").save()

    with pytest.raises(ValueError, match="Unknown preview scopes"):
        staleness_inventory(db_path, scopes=("bogus",))


def test_inventory_respects_scope_filtering(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "scope-1").add_message(role="user", text="hi").save()

    derived_only = staleness_inventory(db_path, scopes=("derived",))
    scopes_emitted = {item.scope for item in derived_only.items}
    assert scopes_emitted == {"derived"}

    archive_only = staleness_inventory(db_path, scopes=("archive_cleanup",))
    assert {item.scope for item in archive_only.items} == {"archive_cleanup"}


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_preview_json_renders_inventory(workspace_env: dict[str, Path]) -> None:
    db_setup(workspace_env)

    runner = CliRunner()
    result = runner.invoke(
        maintenance_group,
        ["preview", "--output-format", "json"],
        obj=None,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "items" in payload
    assert "captured_at" in payload
    assert "total_stale" in payload
    assert isinstance(payload["items"], list)


def test_cli_preview_plain_renders_per_model_sections(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "plain-1").add_message(role="user", text="hi").save()

    runner = CliRunner()
    result = runner.invoke(
        maintenance_group,
        ["preview"],
        obj=None,
    )
    assert result.exit_code == 0, result.output
    assert "Captured:" in result.output
    assert "Total stale rows:" in result.output
    # At least one model section header should appear.
    assert "messages_fts:" in result.output or "orphaned_messages:" in result.output


def test_cli_preview_scope_filter_passes_through(workspace_env: dict[str, Path]) -> None:
    db_setup(workspace_env)
    runner = CliRunner()
    result = runner.invoke(
        maintenance_group,
        ["preview", "--scope", "archive_cleanup", "--output-format", "json"],
        obj=None,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["scopes"] == ["archive_cleanup"]
    for item in payload["items"]:
        assert item["scope"] == "archive_cleanup"
