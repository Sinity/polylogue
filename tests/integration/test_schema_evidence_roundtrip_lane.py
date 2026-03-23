"""Integration smoke tests for the schema-and-evidence roundtrip proof lane."""

from __future__ import annotations

import pytest

from polylogue.schemas.roundtrip_proof import prove_schema_evidence_roundtrip_suite

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("provider", ["chatgpt", "claude-code"])
def test_roundtrip_proof_lane_smoke(provider: str) -> None:
    report = prove_schema_evidence_roundtrip_suite(providers=[provider], count=1)

    assert report.summary["provider_count"] == 1
    assert report.summary["clean"] is True

    provider_report = report.provider_reports[provider]
    assert provider_report.is_clean is True
    assert provider_report.package_version
    assert provider_report.stages["selection"].status == "ok"
    assert provider_report.stages["synthetic"].status == "ok"
    assert provider_report.stages["acquisition"].status == "ok"
    assert provider_report.stages["validation"].status == "ok"
    assert provider_report.stages["parse_dispatch"].status == "ok"
    assert provider_report.stages["prepare_persist"].status == "ok"
    assert provider_report.stages["corpus_verification"].status == "ok"
    assert provider_report.stages["artifact_proof"].status == "ok"
    assert provider_report.summary["artifact_count"] == 1
    assert provider_report.summary["parsed_conversations"] >= 1
    assert provider_report.summary["persisted_conversations"] >= 1
