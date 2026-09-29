"""Capability evidence comes from measured fields and canonical readiness."""

from __future__ import annotations

from typing import Any, cast

import pytest

from polylogue.archive.query.capability_catalog import capability_detail_page
from polylogue.readiness import ReadinessCheck, ReadinessReport, VerifyStatus
from polylogue.sources.origin_specs import origin_specs


def _page(**kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], capability_detail_page(**kwargs))


def _items(**kwargs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = _page(offset=offset, **kwargs)
        rows.extend(cast(list[dict[str, Any]], page["items"]))
        cursor = page["next_offset"]
        if cursor is None:
            return rows
        assert isinstance(cursor, int) and cursor > offset
        offset = cursor


def test_nonempty_archive_does_not_invent_field_or_block_observations() -> None:
    """Session/message totals do not measure tags, tools, vectors, or content blocks."""
    rows = _items(stats={"total_sessions": 7, "total_messages": 11})
    fields = [row for row in rows if row["kind"] != "unit"]
    assert fields
    assert all(row["observed_count"] is None for row in fields)
    units = {row["name"]: row for row in rows if row["kind"] == "unit"}
    assert units["message"]["observed_count"] == 11
    assert units["block"]["observed_count"] is None
    assert all(row["status"] == "unknown" for row in rows)


@pytest.mark.parametrize("value", [True, -1, "7"])
def test_invalid_measurements_are_unknown(value: object) -> None:
    rows = _items(stats={"total_messages": value})
    message = next(row for row in rows if row["kind"] == "unit" and row["name"] == "message")
    assert message["observed_count"] is None


def test_origin_and_readiness_evidence_is_shared_once_per_page() -> None:
    """Restoring per-item evidence multiplies origin metadata by the page size."""
    page = _page(stats={"total_sessions": 2})
    rows = cast(list[dict[str, Any]], page["items"])
    assert len(rows) == 25
    assert all("evidence" not in row for row in rows)
    evidence = cast(dict[str, Any], page["evidence"])
    origins = evidence["origins"]
    assert origins
    assert {item["origin"] for item in origins} == {spec.origin.value for spec in origin_specs() if spec.public_filter}
    assert evidence["readiness"] == {"source": "unmeasured"}
    assert page["snapshot"]["freshness"] == "unknown"


def test_canonical_degradation_overrides_positive_counts() -> None:
    """A successful stats read cannot advertise current capability over a failed index."""
    report = ReadinessReport(timestamp=100, checks=[ReadinessCheck("fts_sync", VerifyStatus.ERROR)])
    page = _page(stats={"total_sessions": 4, "total_messages": 9}, readiness=report)
    rows = cast(list[dict[str, Any]], page["items"])
    assert rows
    assert all(row["status"] == "stale_or_degraded" for row in rows)
    assert page["snapshot"]["freshness"] == "stale_or_degraded"
    assert page["evidence"]["readiness"]["checks"] == [{"name": "fts_sync", "status": "error", "count": 0}]
    unknown = _page(stats={"total_sessions": 4, "total_messages": 9}, readiness=ReadinessReport(timestamp=100))
    assert unknown["snapshot"]["freshness"] == "unknown"
    assert page["snapshot"]["id"] != unknown["snapshot"]["id"]


@pytest.mark.parametrize(("count", "status"), [(0, "supported_but_absent"), (3, "supported_and_observed")])
def test_current_canonical_readiness_keeps_real_unit_measurements(count: int, status: str) -> None:
    """Unknown evidence must not erase a measured ready component either."""
    report = ReadinessReport(
        timestamp=100,
        raw_materialization_readiness={
            "available": True,
            "raw_artifact_count": 1,
            "materialized_raw_artifact_count": 1,
            "raw_authority_parser_census": {"available": True},
        },
    )
    assert report.archive_convergence["materialization_ready"] is True
    page = _page(stats={"total_messages": count, "total_blocks": count}, readiness=report)
    assert page["snapshot"]["freshness"] == "request-current"
    rows = _items(stats={"total_messages": count, "total_blocks": count}, readiness=report)
    measured = [row for row in rows if row["kind"] == "unit" and row["name"] in {"message", "block"}]
    assert len(measured) == 2
    assert all(row["observed_count"] == count and row["status"] == status for row in measured)
