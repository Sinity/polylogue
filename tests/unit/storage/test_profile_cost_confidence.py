"""Absent cost evidence must not read as a known zero."""

from __future__ import annotations

import sqlite3

from polylogue.storage.sqlite.queries.mappers_insight_profiles import _cost_is_estimated


def _row(**fields: object) -> sqlite3.Row:
    """A real sqlite3.Row carrying exactly the given columns."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    if fields:
        names = ", ".join(f"? AS {name}" for name in fields)
        cursor = conn.execute(f"SELECT {names}", tuple(fields.values()))
    else:
        cursor = conn.execute("SELECT 1 AS present")
    row: sqlite3.Row = cursor.fetchone()
    return row


def test_a_profile_without_a_stated_cost_reports_an_estimate() -> None:
    """Only a provider-reported figure makes a cost known.

    Anti-vacuity: defaulting the absent column to False makes each of these
    claim the session is known to have cost nothing, which is what a
    subscription session -- priced only as an API equivalent -- never is.
    """
    assert _cost_is_estimated(_row()) is True
    assert _cost_is_estimated(_row(cost_provenance="unknown")) is True
    assert _cost_is_estimated(_row(cost_provenance="mixed")) is True
    assert _cost_is_estimated(_row(cost_provenance="provider_reported")) is False


def test_a_stored_flag_outranks_the_provenance_fallback() -> None:
    """A materialized value is evidence; the fallback only fills its absence.

    Anti-vacuity: reading provenance first would discard what the writer
    recorded, so a reported cost later marked estimated would silently flip.
    """
    assert _cost_is_estimated(_row(cost_is_estimated=1, cost_provenance="provider_reported")) is True
    assert _cost_is_estimated(_row(cost_is_estimated=0, cost_provenance="unknown")) is False
