"""Tests for versioned insight export bundle contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.insights.export_bundles import InsightExportBundleError, InsightExportBundleRequest
from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import ConversationBuilder


def _json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _jsonl_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        assert isinstance(payload, dict)
        rows.append(payload)
    return rows


def _seed_export_insights(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "codex-export")
        .provider("codex")
        .title("Codex Export")
        .created_at("2026-03-02T09:00:00+00:00")
        .updated_at("2026-03-02T09:15:00+00:00")
        .add_message("u1", role="user", text="Inspect insight export code.", timestamp="2026-03-02T09:00:00+00:00")
        .add_message(
            "a1",
            role="assistant",
            text="Editing the bundle writer.",
            timestamp="2026-03-02T09:10:00+00:00",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "semantic_type": "file_edit",
                        "input": {"path": "/workspace/polylogue/polylogue/product_export_bundles.py"},
                    }
                ]
            },
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "claude-export")
        .provider("claude-code")
        .title("Claude Export")
        .created_at("2026-03-03T09:00:00+00:00")
        .updated_at("2026-03-03T09:10:00+00:00")
        .add_message("u2", role="user", text="Run bundle tests.", timestamp="2026-03-03T09:00:00+00:00")
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_session_insights_sync(conn)


@pytest.mark.asyncio
async def test_insight_export_bundle_writes_bounded_insights(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_export_insights(db_path)
    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    target = cli_workspace["archive_root"] / "exports" / "bundle"

    result = await archive.export_insight_bundle(
        InsightExportBundleRequest(
            output_path=target,
            insights=("profiles", "work-events"),
            provider="codex",
            since="2026-03-01",
            until="2026-03-31",
        )
    )

    assert result.output_path == target
    manifest = _json_file(target / "manifest.json")
    assert manifest["bundle_version"] == 1
    assert manifest["query"] == {
        "insights": ["session_profiles", "session_work_events"],
        "provider": "codex",
        "since": "2026-03-01",
        "until": "2026-03-31",
    }
    assert (target / "coverage.json").exists()
    assert (target / "README.md").exists()
    assert (target / "schemas" / "session_profiles.schema.json").exists()
    assert (target / "schemas" / "session_work_events.schema.json").exists()
    profiles = _jsonl_file(target / "insights" / "session_profiles.jsonl")
    events = _jsonl_file(target / "insights" / "session_work_events.jsonl")
    assert {profile["provider_name"] for profile in profiles} == {"codex"}
    assert profiles
    assert events
    summaries = manifest["insights"]
    assert isinstance(summaries, list)
    assert {summary["insight_name"] for summary in summaries if isinstance(summary, dict)} == {
        "session_profiles",
        "session_work_events",
    }


@pytest.mark.asyncio
async def test_insight_export_bundle_protects_existing_targets(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_export_insights(db_path)
    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    target = cli_workspace["archive_root"] / "exports" / "existing"
    target.mkdir(parents=True)
    marker = target / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(InsightExportBundleError, match="already exists"):
        await archive.export_insight_bundle(InsightExportBundleRequest(output_path=target, insights=("profiles",)))

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
async def test_insight_export_bundle_records_stale_readiness(cli_workspace: dict[str, Path]) -> None:
    db_path = cli_workspace["db_path"]
    _seed_export_insights(db_path)
    with open_connection(db_path) as conn:
        conn.execute("UPDATE conversations SET sort_key = sort_key + 1 WHERE conversation_id = ?", ("codex-export",))
        conn.commit()
    archive = Polylogue(archive_root=cli_workspace["archive_root"], db_path=db_path)
    target = cli_workspace["archive_root"] / "exports" / "stale-bundle"

    await archive.export_insight_bundle(InsightExportBundleRequest(output_path=target, insights=("profiles",)))

    coverage = _json_file(target / "coverage.json")
    coverage_products = coverage["insights"]
    assert isinstance(coverage_products, list)
    assert coverage_products[0]["verdict"] == "stale"
    manifest = _json_file(target / "manifest.json")
    manifest_products = manifest["insights"]
    assert isinstance(manifest_products, list)
    assert manifest_products[0]["row_count"] == 0
    assert manifest_products[0]["errors"]
    assert (target / "insights" / "session_profiles.jsonl").read_text(encoding="utf-8") == ""
