"""Absent cost evidence must not read as a known zero."""

from __future__ import annotations

import sqlite3

from polylogue.analysis.archive_models import SessionEvidencePayload
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


def test_stored_evidence_supplies_absent_columns() -> None:
    """``session_profiles`` persists neither column; the payload holds both.

    ``session_profile_insert_columns`` writes no ``cost_is_estimated`` and no
    ``cost_provenance``, so on the production ``SELECT * FROM session_profiles``
    path (rebuild and thread reads) both lookups above miss and every row -- a
    provider-reported charge included -- came back as an estimate. Portfolio and
    postmortem read the field directly and relabel a whole rollup from one such
    row.

    Anti-vacuity: remove the ``stated_evidence`` arm and the first assertion
    goes back to ``True``. The second and third assertions are the opposite
    direction: a synthesized payload is not a statement about cost, and a
    stored column still outranks the payload.
    """
    reported = SessionEvidencePayload(cost_is_estimated=False, cost_provenance="provider_reported")
    assert _cost_is_estimated(_row(), reported) is False
    assert _cost_is_estimated(_row(), None) is True
    assert _cost_is_estimated(_row(cost_is_estimated=1), reported) is True
