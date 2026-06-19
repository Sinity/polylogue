from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from polylogue.sources import provider_completeness as module
from polylogue.sources.provider_completeness import (
    accepted_blockers,
    provider_package_completeness,
)


def test_provider_completeness_reports_representative_modes() -> None:
    report = provider_package_completeness()
    by_ref = {row.package_ref: row for row in report.rows}

    codex = by_ref["provider-package:codex-session/session-jsonl@v1"]
    chatgpt = by_ref["provider-package:chatgpt-export/takeout-json@v1"]
    browser = by_ref["provider-package:browser-capture/live-receiver@v1"]

    assert codex.origin == "codex-session"
    assert codex.capture_mode == "session-jsonl"
    assert codex.detector.status == "complete"
    assert codex.parser.owner_path == "polylogue/sources/parsers/codex.py"
    assert codex.import_explain.status == "complete"

    assert chatgpt.origin == "chatgpt-export"
    assert chatgpt.schema_package.status == "complete"

    assert browser.maturity == "proposed"
    assert browser.status == "proposed"
    assert browser.schema_package.status == "missing"
    assert not browser.blockers


def test_provider_completeness_origin_filter_accepts_origin_and_provider() -> None:
    by_origin = provider_package_completeness(origin="codex-session")
    by_provider = provider_package_completeness(origin="codex")

    assert [row.package_ref for row in by_origin.rows] == ["provider-package:codex-session/session-jsonl@v1"]
    assert [row.package_ref for row in by_provider.rows] == ["provider-package:codex-session/session-jsonl@v1"]


def test_provider_completeness_json_payload_shape() -> None:
    report = provider_package_completeness(origin="codex-session")
    payload = json.loads(report.to_json())

    assert payload["mode"] == "provider-package-completeness"
    assert payload["totals"]["total"] == 1
    assert payload["rows"][0]["query_units"]["status"] == "complete"
    assert payload["rows"][0]["provider_wire"] == "codex"


def test_provider_completeness_check_blocks_accepted_missing_required_item(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    report = provider_package_completeness(origin="codex-session")

    assert report.rows[0].maturity == "accepted"
    assert report.rows[0].blockers
    assert accepted_blockers(report)
