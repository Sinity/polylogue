"""Cohort money and origin logical counts through the public insight reader."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from polylogue.analysis.archive import (
    CostRollupInsight,
    CostRollupInsightQuery,
    SessionTagRollupInsight,
    SessionTagRollupQuery,
)
from polylogue.analysis.insight_reads import read_insight_page
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder


def _session(db_path: Path, name: str, *, dollars: float | None = None, provider: str = "codex") -> str:
    builder = SessionBuilder(db_path, name).provider(provider).reported_cost_usd(dollars)
    timestamp = "2026-03-01T10:00:00+00:00"
    builder.created_at(timestamp).updated_at(timestamp).add_message(
        "u1", role="user", text="neutral", timestamp=timestamp
    ).save()
    return builder.native_session_id()


def _usage(
    conn: sqlite3.Connection, session: str, model: str, *, provider: float | None = None, catalog: float | None = None
) -> None:
    conn.execute(
        """INSERT INTO session_model_usage
           (session_id, model_name, input_tokens, output_tokens, provider_usage_observed,
            provider_lanes_complete, provider_cost_usd, catalog_cost_usd)
           VALUES (?, ?, 10, 5, 1, 1, ?, ?)""",
        (session, model, provider, catalog),
    )


def _costs(root: Path, request: CostRollupInsightQuery | None = None) -> list[CostRollupInsight]:
    with (
        ArchiveStore(root, read_only=True) as archive,
        patch.object(archive, "read_session", side_effect=AssertionError("payload hydrated")),
    ):
        rows = read_insight_page(archive, request or CostRollupInsightQuery())
        assert all(isinstance(row, CostRollupInsight) for row in rows)
        return [row for row in rows if isinstance(row, CostRollupInsight)]


def test_reported_session_money_sums_before_model_cohort_reduction(cli_workspace: dict[str, Path]) -> None:
    db = cli_workspace["db_path"]
    first = _session(db, "first", dollars=1)
    second = _session(db, "second", dollars=2)
    other = _session(db, "other-origin", dollars=100, provider="claude-code")
    with sqlite3.connect(db) as conn:
        for session in (first, second, other):
            _usage(conn, session, "neutral-model")
    rows = _costs(cli_workspace["archive_root"], CostRollupInsightQuery(origin="codex-session", model="neutral-model"))
    assert len(rows) == 1
    row = rows[0]
    assert row.session_count == row.priced_session_count == 2
    assert row.total_usd == row.basis.provider_reported_usd == 3
    assert row.status_counts == {"exact": 2}


def test_money_precedence_is_settled_for_each_session(cli_workspace: dict[str, Path]) -> None:
    db = cli_workspace["db_path"]
    provider = _session(db, "provider", dollars=50)
    reported = _session(db, "reported", dollars=2)
    catalog = _session(db, "catalog")
    with sqlite3.connect(db) as conn:
        _usage(conn, provider, "neutral-model", provider=1, catalog=70)
        _usage(conn, reported, "neutral-model", catalog=80)
        _usage(conn, catalog, "neutral-model", catalog=3)
    row = _costs(cli_workspace["archive_root"])[0]
    assert row.total_usd == 6
    assert row.basis.provider_reported_usd == 3
    assert row.basis.catalog_priced_usd == 3
    assert row.status_counts == {"exact": 2, "priced": 1}


def test_multimodel_report_is_not_repeated_and_unknown_is_unavailable(cli_workspace: dict[str, Path]) -> None:
    db = cli_workspace["db_path"]
    multi = _session(db, "multiple-models", dollars=99)
    unknown = _session(db, "unknown")
    zero = _session(db, "measured-zero", dollars=0)
    with sqlite3.connect(db) as conn:
        _usage(conn, multi, "neutral-a", provider=1)
        _usage(conn, multi, "neutral-b", catalog=2)
        _usage(conn, unknown, "neutral-a")
        _usage(conn, zero, "neutral-a")
    rows = _costs(cli_workspace["archive_root"])
    assert sum(row.total_usd for row in rows) == 3
    first = next(row for row in rows if row.model_name == "neutral-a")
    assert first.status_counts == {"exact": 2, "unavailable": 1}
    assert first.priced_session_count == 2
    assert first.unavailable_session_count == 1
    assert _costs(cli_workspace["archive_root"], CostRollupInsightQuery(model="neutral-b"))[0].total_usd == 2


def test_origin_tag_counts_selected_physical_rows_and_distinct_logical_roots(cli_workspace: dict[str, Path]) -> None:
    db = cli_workspace["db_path"]
    root = _session(db, "root")
    child = _session(db, "child")
    _session(db, "independent")
    _session(db, "other-origin", provider="claude-code")
    old = _session(db, "outside-date")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE sessions SET root_session_id = ? WHERE session_id IN (?, ?)", (root, root, child))
        conn.execute("UPDATE sessions SET updated_at_ms = 1 WHERE session_id = ?", (old,))
    with (
        ArchiveStore(cli_workspace["archive_root"], read_only=True) as archive,
        patch.object(archive, "read_session", side_effect=AssertionError("payload hydrated")),
    ):
        request = SessionTagRollupQuery(origin="codex-session", since="2026-03-01", until="2026-03-02", query="origin:")
        rows = read_insight_page(archive, request)
        assert len(rows) == 1, rows
        row = rows[0]
        assert isinstance(row, SessionTagRollupInsight)
        assert row.tag == "origin:codex-session"
        assert row.session_count == 3
        assert row.logical_session_count == 2
        all_rows = read_insight_page(archive, SessionTagRollupQuery(query="origin:"))
        other = next(
            item
            for item in all_rows
            if isinstance(item, SessionTagRollupInsight) and item.tag == "origin:claude-code-session"
        )
        assert other.session_count == other.logical_session_count == 1
