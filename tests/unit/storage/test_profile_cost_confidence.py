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


_ESTIMATED = SessionEvidencePayload(cost_is_estimated=True, cost_provenance="unknown")
_REPORTED = SessionEvidencePayload(cost_is_estimated=False, cost_provenance="provider_reported")


def test_a_provenance_column_decides_before_the_payload() -> None:
    """Only a provider-reported figure makes a cost known.

    Anti-vacuity: reading the payload first makes the second and third rows
    claim a known cost for a subscription session, which is only ever priced
    as an API equivalent.
    """
    assert _cost_is_estimated(_row(), _ESTIMATED) is True
    assert _cost_is_estimated(_row(cost_provenance="unknown"), _REPORTED) is True
    assert _cost_is_estimated(_row(cost_provenance="mixed"), _REPORTED) is True
    assert _cost_is_estimated(_row(cost_provenance="provider_reported"), _ESTIMATED) is False


def test_a_stored_flag_outranks_the_provenance_column() -> None:
    """A materialized value is evidence; provenance only fills its absence.

    Anti-vacuity: reading provenance first would discard what the writer
    recorded, so a reported cost later marked estimated would silently flip.
    """
    assert _cost_is_estimated(_row(cost_is_estimated=1, cost_provenance="provider_reported"), _REPORTED) is True
    assert _cost_is_estimated(_row(cost_is_estimated=0, cost_provenance="unknown"), _ESTIMATED) is False


def test_stored_evidence_supplies_absent_columns() -> None:
    """``session_profiles`` persists neither column; the payload holds both.

    ``SESSION_PROFILE_INSERT_COLUMNS`` writes no ``cost_is_estimated`` and no
    ``cost_provenance``, so on the production ``SELECT * FROM session_profiles``
    path (rebuild and thread reads) both lookups above miss and the stored
    evidence answers. Portfolio and postmortem read the field directly and
    relabel a whole rollup from one such row.

    Anti-vacuity: return ``True`` when both columns are absent and the first
    assertion fails; ignore the payload and one of the two fails either way.
    """
    assert _cost_is_estimated(_row(), _REPORTED) is False
    assert _cost_is_estimated(_row(), _ESTIMATED) is True
