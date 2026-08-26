"""Coverage and anti-vacuity tests for the audit schema disposition."""

from __future__ import annotations

from dataclasses import replace

import pytest

from polylogue.storage.sqlite.archive_tiers.schema_disposition import (
    assert_complete_audit_disposition,
    audit_column_dispositions,
    canonical_audit_columns,
)


def test_audit_disposition_covers_canonical_ddl_exactly_once() -> None:
    rows = audit_column_dispositions()
    assert_complete_audit_disposition(rows)
    assert len(rows) == len(canonical_audit_columns()) == 134
    assert len({row.ref for row in rows}) == len(rows)
    assert {row.disposition for row in rows} == {"KEEP-WIRED"}
    for row in rows:
        assert all(
            value
            for value in (
                row.writer,
                row.reader,
                row.authority_role,
                row.retention,
                row.continuity_or_receipt,
                row.live_denominator,
                row.evidence,
            )
        )


@pytest.mark.parametrize("mutation", ["omitted", "duplicate", "extra", "unclear"])
def test_audit_disposition_rejects_incomplete_or_unresolved_inventory(mutation: str) -> None:
    rows = list(audit_column_dispositions())
    if mutation == "omitted":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "extra":
        rows.append(replace(rows[0], column="undeclared_column"))
    else:
        rows[0] = replace(rows[0], disposition="UNCLEAR")

    with pytest.raises(ValueError):
        assert_complete_audit_disposition(rows)


def test_audit_disposition_rejects_purge_without_copy_forward_owner() -> None:
    row = replace(audit_column_dispositions()[0], disposition="PURGE")
    with pytest.raises(ValueError, match="60i5 copy-forward owner"):
        assert_complete_audit_disposition([row, *audit_column_dispositions()[1:]])
