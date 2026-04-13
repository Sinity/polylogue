from __future__ import annotations

import json
from pathlib import Path

from devtools.campaign_report import (
    generate_campaign_json,
    generate_campaign_markdown,
    save_campaign_reports,
)
from devtools.synthetic_benchmark_runtime import CampaignResult


def _sample_result(name: str) -> CampaignResult:
    return CampaignResult(
        campaign_name=name,
        scale_level="small",
        metrics={"wall_s": 1.25, "items": 12},
        db_stats={"db_size_bytes": 1024 * 1024, "messages_count": 42},
        timestamp="2026-04-11T00:00:00+00:00",
        origin="authored.synthetic-benchmark",
        path_targets=["synthetic-startup-loop"],
        artifact_targets=["message_fts"],
        operation_targets=["index.message-fts-rebuild"],
        tags=["benchmark", "synthetic"],
    )


def test_generate_campaign_markdown_renders_summary_and_db_stats() -> None:
    rendered = generate_campaign_markdown([_sample_result("fts-rebuild"), _sample_result("filter-scan")])

    assert "# Benchmark Campaign Report" in rendered
    assert "| Campaign | Key Metric | Value |" in rendered
    assert "| db_size_bytes | 1.0 MB | 1.0 MB |" in rendered
    assert "| path targets | `synthetic-startup-loop` |" in rendered


def test_generate_campaign_json_emits_campaign_payload() -> None:
    payload = json.loads(generate_campaign_json([_sample_result("fts-rebuild")]))

    assert payload["scale_level"] == "small"
    assert payload["campaigns"][0]["campaign_name"] == "fts-rebuild"
    assert payload["campaigns"][0]["origin"] == "authored.synthetic-benchmark"
    assert payload["campaigns"][0]["path_targets"] == ["synthetic-startup-loop"]


def test_save_campaign_reports_writes_markdown_and_json(tmp_path: Path) -> None:
    saved = save_campaign_reports([_sample_result("fts-rebuild")], tmp_path)

    assert {path.suffix for path in saved} == {".md", ".json"}
    for path in saved:
        assert path.exists()
