"""The frontier certificate's recorded findings render each status category.

Status reads the completed inspection's Ops mark instead of repeating the
corpus inspection. Anti-vacuity: drop the findings from the mark (or render
only the certificate's state) and a broken head reads as ``unknown`` with no
count, sample or reason; accept an undeclared document and a malformed mark
reads as a clean category.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from polylogue.storage.raw_retention import (
    BrokenAppendHeadSample,
    FrontierInspectionFindings,
    _raw_frontier_integrity_from_coverage,
)

_SAMPLE = BrokenAppendHeadSample(
    logical_source_key="codex:session-1",
    accepted_raw_id="raw-append",
    reason="active index raw is missing from source tier: raw-append",
)


_BASE = FrontierInspectionFindings(
    mode="full",
    head_checks=3,
    blocking_heads=0,
    broken_heads=0,
    broken_head_samples=(),
    cursor_checks=2,
    cursor_ahead=0,
    cursor_ahead_comparisons=0,
    cursor_gaps=0,
    missing_session_raws=0,
)


def _coverage(findings: FrontierInspectionFindings | None, *, state: str) -> dict[str, object]:
    coverage: dict[str, object] = {"available": True, "current": True, "healthy": state == "healthy", "state": state}
    if findings is not None:
        coverage["findings"] = findings
    return coverage


def test_findings_round_trip_through_their_recorded_document() -> None:
    findings = replace(
        _BASE, broken_heads=1, broken_head_samples=(_SAMPLE,), cursor_ahead=1, cursor_ahead_comparisons=2
    )

    assert FrontierInspectionFindings.from_document(findings.to_document()) == findings


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("broken_heads"),
        lambda payload: payload.update(extra=1),
        lambda payload: payload.update(broken_heads=-1),
        lambda payload: payload.update(broken_heads=True),
        lambda payload: payload.update(mode="current"),
        lambda payload: payload.update(broken_head_samples=[{"reason": "x"}]),
        lambda payload: payload.update(broken_heads=0),
    ],
    ids=["missing-count", "undeclared-key", "negative", "boolean", "mode", "sample-shape", "samples-exceed-count"],
)
def test_an_undeclared_findings_document_is_refused(mutation: Callable[[dict[str, object]], object]) -> None:
    payload = json.loads(replace(_BASE, broken_heads=1, broken_head_samples=(_SAMPLE,)).to_document())
    mutation(payload)

    with pytest.raises(ValueError):
        FrontierInspectionFindings.from_document(json.dumps(payload))


def test_a_blocked_certificate_renders_its_broken_heads() -> None:
    findings = replace(_BASE, broken_heads=1, broken_head_samples=(_SAMPLE,))

    projection = _raw_frontier_integrity_from_coverage(_coverage(findings, state="blocked"), {})

    assert projection.overall_status == "violated"
    assert projection.broken_head_status == "violated"
    assert projection.broken_head_count == 1
    assert projection.broken_head_checked_count == 3
    assert projection.broken_head_samples == (_SAMPLE,)
    assert "broken predecessor chain" in projection.broken_head_reason
    assert projection.cursor_ahead_status == "healthy"


def test_a_blocking_obligation_without_a_category_still_refuses_the_certificate() -> None:
    projection = _raw_frontier_integrity_from_coverage(_coverage(replace(_BASE, blocking_heads=1), state="blocked"), {})

    assert projection.broken_head_status == "healthy"
    assert projection.overall_status == "violated"


def test_an_unreadable_findings_document_leaves_every_category_unknown() -> None:
    coverage = _coverage(None, state="blocked")
    coverage["detail"] = "frontier inspection findings unreadable: undeclared shape"

    projection = _raw_frontier_integrity_from_coverage(coverage, {})

    assert projection.broken_head_status == "unknown"
    assert projection.cursor_ahead_status == "unknown"
    assert projection.overall_status == "violated"
    assert projection.broken_head_count == 0
