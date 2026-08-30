"""Coverage and anti-vacuity tests for the audit schema disposition."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from polylogue.storage.sqlite.archive_tiers import AUDIT_COLUMN_DISPOSITIONS
from polylogue.storage.sqlite.archive_tiers.schema_disposition import (
    assert_complete_audit_disposition,
    assert_complete_schema_dispositions,
    audit_column_dispositions,
    canonical_audit_columns,
    schema_disposition_report,
    schema_dispositions,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_archive_tier_schema_assembly_publishes_complete_disposition() -> None:
    """The canonical tier map exposes the disposition it validates at import."""
    assert audit_column_dispositions() == AUDIT_COLUMN_DISPOSITIONS
    assert_complete_audit_disposition(AUDIT_COLUMN_DISPOSITIONS)


def test_audit_disposition_covers_canonical_ddl_exactly_once() -> None:
    rows = audit_column_dispositions()
    assert_complete_audit_disposition(rows)
    assert len(rows) == len(canonical_audit_columns()) == 134
    assert len({row.ref for row in rows}) == len(rows)
    assert {row.disposition for row in rows} == {"KEEP"}
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
        # Anti-vacuity: deliberately construct a disposition outside the
        # declared vocabulary to prove the runtime validator rejects it.
        rows[0] = replace(rows[0], disposition=cast(Any, "UNCLEAR"))

    with pytest.raises(ValueError):
        assert_complete_audit_disposition(rows)


def test_audit_disposition_rejects_purge_without_copy_forward_owner() -> None:
    row = replace(audit_column_dispositions()[0], disposition="PURGE")
    with pytest.raises(ValueError, match="60i5 copy-forward owner"):
        assert_complete_audit_disposition([row, *audit_column_dispositions()[1:]])


def test_six_tier_disposition_is_ddl_derived_and_settles_special_groups() -> None:
    rows = schema_dispositions()
    assert_complete_schema_dispositions(rows)
    assert len(rows) == len({row.object_ref for row in rows})
    assert {row.disposition for row in rows} <= {"KEEP", "COMPLETE", "PURGE", "DERIVE", "TRANSITION"}
    by_ref = {row.object_ref: row for row in rows}

    assert by_ref["source:table:excised_content"].disposition == "TRANSITION"
    assert by_ref["source:table:raw_failure_disposition_receipts"].disposition == "KEEP"
    assert by_ref["user:table:holdout_access_receipts"].disposition == "COMPLETE"
    assert not any("raw_unknown_export_reclassification_receipts" in row.object_ref for row in rows)
    assert not any("dominant_repo_id" in row.object_ref for row in rows)
    assert all(
        row.semantic_owner
        and row.implementation_bead
        and row.definition_sha256
        and row.producer
        and row.consumer
        and row.live_row_denominator
        and row.campaign_action
        and row.successor_or_authorization
        for row in rows
    )


def test_six_tier_disposition_rejects_undeclared_object() -> None:
    rows = list(schema_dispositions())
    rows.append(replace(rows[0], object_ref="source:table:undeclared"))

    with pytest.raises(ValueError, match="undeclared schema objects"):
        assert_complete_schema_dispositions(rows)


def test_six_tier_report_is_generated_from_the_complete_disposition() -> None:
    report = schema_disposition_report()

    assert report["complete"] is True
    assert report["object_count"] == len(schema_dispositions())
    counts = cast("dict[str, int]", report["disposition_counts"])
    objects = cast("list[object]", report["objects"])
    assert sum(counts.values()) == report["object_count"]
    assert report["unknown_count"] == 0
    assert set(cast("dict[str, object]", report["tier_counts"])) == {tier.value for tier in ArchiveTier}
    assert len(objects) == report["object_count"]
