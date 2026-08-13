"""Unit tests for the executable continuity-evidence artifact.

Anti-vacuity: the discovery-coverage lane runs against the real shipped
``QUERY_DISCOVERY_EXAMPLES`` catalog and t8t's real ``CONTINUITY_SCENARIOS``
declarations (not stand-ins). The mutation case removes a real discovery
example and requires the executable report to fail rather than asserting a
fixed catalog shape.
"""

from __future__ import annotations

import pytest

from devtools import continuity_evidence as mcr
from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES
from polylogue.core.json import JSONDocument
from polylogue.product.continuity_scenarios import CONTINUITY_SCENARIOS

# ── Discovery-coverage lane ────────────────────────────────────────────


def test_discovery_coverage_passes_for_shipped_continuity_scenarios() -> None:
    report = mcr.check_discovery_coverage(CONTINUITY_SCENARIOS)

    assert report.checked_steps > 0
    assert report.gaps == ()
    assert report.status == "pass"
    assert report.to_dict()["status"] == "pass"


def test_discovery_coverage_flags_a_regressed_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove every declared "runs" query example -- a real shipped scenario
    # (postmortem) issues a `runs where ...` query step, so this must surface
    # as a named gap rather than a silent pass.
    filtered = tuple(
        example
        for example in QUERY_DISCOVERY_EXAMPLES
        if not (example.route == "query" and example.unit_source == "runs")
    )
    assert len(filtered) < len(QUERY_DISCOVERY_EXAMPLES)
    monkeypatch.setattr(mcr, "QUERY_DISCOVERY_EXAMPLES", filtered)

    report = mcr.check_discovery_coverage(CONTINUITY_SCENARIOS)

    assert report.status == "fail"
    assert any(gap.plan_atom == "query:runs" for gap in report.gaps)
    assert report.covered_steps == report.checked_steps - len(report.gaps)


# ── Redaction ──────────────────────────────────────────────────────────


def test_redact_report_hashes_evidence_prose_but_preserves_refs_and_counts() -> None:
    document: JSONDocument = {
        "status": "pass",
        "count": 3,
        "label": "polylogue-7fj status: 'in_progress' -> 'closed'",
        "nested": {"claim_text": "Beads issue polylogue-7fj was recorded as closed.", "ref": "commit:abc123"},
        "list": [{"reason": "Focused parser checks passed."}, {"ref": "beads-issue:polylogue-7fj"}],
    }

    redacted = mcr.redact_report(document)

    assert isinstance(redacted, dict)
    assert redacted["status"] == "pass"
    assert redacted["count"] == 3
    label = redacted["label"]
    assert isinstance(label, str) and label.startswith("redacted:sha256:")
    nested = redacted["nested"]
    assert isinstance(nested, dict)
    claim_text = nested["claim_text"]
    assert isinstance(claim_text, str) and claim_text.startswith("redacted:sha256:")
    assert nested["ref"] == "commit:abc123"
    entries = redacted["list"]
    assert isinstance(entries, list)
    first_entry = entries[0]
    assert isinstance(first_entry, dict)
    reason = first_entry["reason"]
    assert isinstance(reason, str) and reason.startswith("redacted:sha256:")
    second_entry = entries[1]
    assert isinstance(second_entry, dict)
    assert second_entry["ref"] == "beads-issue:polylogue-7fj"


def test_redact_report_is_deterministic() -> None:
    document: JSONDocument = {"label": "same text twice"}
    assert mcr.redact_report(document) == mcr.redact_report(dict(document))
