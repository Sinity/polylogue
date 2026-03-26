"""Tests for the durable archive data products CLI surfaces."""

from __future__ import annotations

import json

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.storage.action_event_lifecycle import rebuild_action_event_read_model_sync
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.backends.queries.maintenance_runs import record_maintenance_run_sync
from polylogue.storage.session_product_lifecycle import (
    rebuild_session_products_sync,
    session_product_status_sync,
)
from polylogue.storage.store import MaintenanceRunRecord
from tests.infra.storage_records import ConversationBuilder


def _extract_json(output: str) -> dict[str, object]:
    data = json.loads(output)
    if isinstance(data, dict) and data.get("status") == "ok":
        return data["result"]
    return data


def _seed_products(cli_workspace) -> None:
    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-root")
        .provider("claude-code")
        .title("Root Thread")
        .created_at("2026-03-01T10:00:00+00:00")
        .updated_at("2026-03-01T10:10:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Plan the refactor and inspect README",
            timestamp="2026-03-01T10:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Inspecting and editing files",
            timestamp="2026-03-01T10:05:00+00:00",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "semantic_type": "file_read",
                        "input": {"path": "/realm/project/polylogue/README.md"},
                    },
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "semantic_type": "file_edit",
                        "input": {"path": "/realm/project/polylogue/README.md"},
                    },
                ]
            },
        )
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-child")
        .provider("claude-code")
        .title("Child Thread")
        .parent_conversation("conv-root")
        .branch_type("continuation")
        .created_at("2026-03-01T11:00:00+00:00")
        .updated_at("2026-03-01T11:05:00+00:00")
        .add_message(
            "u2",
            role="user",
            text="Run tests for the refactor",
            timestamp="2026-03-01T11:00:00+00:00",
        )
        .add_message(
            "a2",
            role="assistant",
            text="Running pytest",
            timestamp="2026-03-01T11:04:00+00:00",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "semantic_type": "shell",
                        "input": {"command": "pytest -q tests/unit/cli/test_products.py"},
                    }
                ]
            },
        )
        .save()
    )
    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        rebuild_action_event_read_model_sync(conn)


def _record_maintenance_lineage(cli_workspace) -> None:
    with open_connection(cli_workspace["db_path"]) as conn:
        record_maintenance_run_sync(
            conn,
            MaintenanceRunRecord(
                maintenance_run_id="maint-test-001",
                executed_at="2026-03-24T10:00:00+00:00",
                mode="preview",
                preview=True,
                repair_selected=True,
                cleanup_selected=False,
                vacuum_requested=False,
                target_names=("session_products",),
                success=True,
                manifest={
                    "preview_counts": {"session_products": 2},
                    "results": [{"name": "session_products", "repaired_count": 2}],
                },
            ),
        )
        record_maintenance_run_sync(
            conn,
            MaintenanceRunRecord(
                maintenance_run_id="maint-test-002",
                executed_at="2026-03-24T10:05:00+00:00",
                mode="apply",
                preview=False,
                repair_selected=True,
                cleanup_selected=False,
                vacuum_requested=False,
                target_names=("session_products",),
                success=True,
                manifest={"results": [{"name": "session_products", "repaired_count": 2}]},
            ),
        )
        record_maintenance_run_sync(
            conn,
            MaintenanceRunRecord(
                maintenance_run_id="maint-test-003",
                executed_at="2026-03-24T10:10:00+00:00",
                mode="preview",
                preview=True,
                repair_selected=True,
                cleanup_selected=False,
                vacuum_requested=False,
                target_names=("session_products",),
                success=True,
                manifest={
                    "preview_counts": {"session_products": 0},
                    "results": [{"name": "session_products", "repaired_count": 0}],
                },
            ),
        )


def test_products_profiles_json(cli_workspace):
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "profiles", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert payload["count"] == 2
    first = payload["session_profiles"][0]
    assert first["contract_version"] == 4
    assert first["product_kind"] == "session_profile"
    assert first["semantic_tier"] == "merged"
    assert first["evidence"]["canonical_session_date"] == "2026-03-01"
    assert first["inference"]["engaged_duration_ms"] >= 0
    assert "evidence" in first
    assert "inference" in first
    assert "provenance" in first


def test_products_enrichments_json(cli_workspace):
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "products",
            "enrichments",
            "--provider",
            "claude-code",
            "--session-date-since",
            "2026-03-01",
            "--json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert payload["count"] == 2
    first = payload["session_enrichments"][0]
    assert first["contract_version"] == 4
    assert first["product_kind"] == "session_enrichment"
    assert first["semantic_tier"] == "enrichment"
    assert first["enrichment_provenance"]["enrichment_family"] == "scored_session_enrichment"
    assert first["enrichment"]["support_level"] in {"weak", "moderate", "strong"}
    assert "input_band_summary" in first["enrichment"]


def test_products_profiles_json_supports_explicit_evidence_and_inference_tiers(cli_workspace):
    _seed_products(cli_workspace)

    runner = CliRunner()
    evidence_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "evidence", "--json"],
        catch_exceptions=False,
    )
    inference_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "inference", "--json"],
        catch_exceptions=False,
    )

    assert evidence_result.exit_code == 0
    assert inference_result.exit_code == 0

    evidence_payload = _extract_json(evidence_result.output)
    inference_payload = _extract_json(inference_result.output)
    evidence_profile = evidence_payload["session_profiles"][0]
    inference_profile = inference_payload["session_profiles"][0]

    assert evidence_profile["semantic_tier"] == "evidence"
    assert evidence_profile["evidence"]["canonical_session_date"] == "2026-03-01"
    assert evidence_profile["inference"] is None
    assert evidence_profile["inference_provenance"] is None

    assert inference_profile["semantic_tier"] == "inference"
    assert inference_profile["evidence"] is None
    assert inference_profile["inference"]["engaged_duration_ms"] >= 0
    assert inference_profile["inference_provenance"]["inference_family"] == "heuristic_session_semantics"


def test_products_profiles_json_handles_blank_tier_search_text_from_migrated_rows(cli_workspace):
    _seed_products(cli_workspace)
    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("UPDATE session_profiles SET evidence_search_text = '', inference_search_text = ''")
        conn.commit()

    runner = CliRunner()
    evidence_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "evidence", "--json"],
        catch_exceptions=False,
    )
    inference_result = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "inference", "--json"],
        catch_exceptions=False,
    )

    assert evidence_result.exit_code == 0
    assert inference_result.exit_code == 0
    assert _extract_json(evidence_result.output)["count"] == 2
    assert _extract_json(inference_result.output)["count"] == 2


def test_products_reconstructs_tiered_payloads_from_blank_migrated_rows(cli_workspace):
    _seed_products(cli_workspace)
    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("UPDATE session_profiles SET evidence_payload_json = '{}', inference_payload_json = '{}'")
        conn.execute("UPDATE session_work_events SET evidence_payload_json = '{}', inference_payload_json = '{}'")
        conn.execute("UPDATE session_phases SET evidence_payload_json = '{}', inference_payload_json = '{}'")
        conn.commit()

    runner = CliRunner()
    evidence_profiles = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "evidence", "--json"],
        catch_exceptions=False,
    )
    inference_profiles = runner.invoke(
        cli,
        ["products", "profiles", "--tier", "inference", "--json"],
        catch_exceptions=False,
    )
    work_events = runner.invoke(cli, ["products", "work-events", "--json"], catch_exceptions=False)
    phases = runner.invoke(cli, ["products", "phases", "--json"], catch_exceptions=False)

    assert evidence_profiles.exit_code == 0
    assert inference_profiles.exit_code == 0
    assert work_events.exit_code == 0
    assert phases.exit_code == 0

    evidence_profiles_payload = _extract_json(evidence_profiles.output)["session_profiles"]
    inference_profiles_payload = _extract_json(inference_profiles.output)["session_profiles"]
    work_event = _extract_json(work_events.output)["session_work_events"][0]
    phase = _extract_json(phases.output)["session_phases"][0]

    assert all(item["evidence"]["message_count"] >= 1 for item in evidence_profiles_payload)
    assert all(item["evidence"]["canonical_session_date"] == "2026-03-01" for item in evidence_profiles_payload)
    assert all(item["inference"]["work_event_count"] >= 0 for item in inference_profiles_payload)
    assert work_event["evidence"]["start_index"] >= 0
    assert work_event["inference"]["kind"] in {"implementation", "testing", "planning", "debugging"}
    assert phase["evidence"]["message_range"][0] >= 0
    assert phase["inference"]["kind"]


def test_products_profile_date_filters_and_phases_json(cli_workspace):
    _seed_products(cli_workspace)

    runner = CliRunner()
    profiles = runner.invoke(
        cli,
        [
            "products",
            "profiles",
            "--session-date-since",
            "2026-03-01",
            "--session-date-until",
            "2026-03-01",
            "--json",
        ],
        catch_exceptions=False,
    )
    phases = runner.invoke(cli, ["products", "phases", "--json"], catch_exceptions=False)

    assert profiles.exit_code == 0
    assert phases.exit_code == 0

    profile_payload = _extract_json(profiles.output)
    phase_payload = _extract_json(phases.output)
    assert profile_payload["count"] == 2
    assert phase_payload["count"] >= 1
    assert phase_payload["session_phases"][0]["product_kind"] == "session_phase"


def test_session_product_rebuild_supports_legacy_payload_columns(cli_workspace):
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("ALTER TABLE session_profiles ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE session_work_events ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE session_phases ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.commit()
        rebuild_session_products_sync(conn)
        status = session_product_status_sync(conn)

    assert status["profile_row_count"] == 2
    assert status["profile_rows_ready"] is True
    assert status["work_event_inference_rows_ready"] is True
    assert status["phase_inference_rows_ready"] is True


def test_products_threads_and_status_json(cli_workspace):
    _seed_products(cli_workspace)

    runner = CliRunner()
    threads = runner.invoke(cli, ["products", "threads", "--json"], catch_exceptions=False)
    status = runner.invoke(cli, ["products", "status", "--json"], catch_exceptions=False)

    assert threads.exit_code == 0
    assert status.exit_code == 0

    threads_payload = _extract_json(threads.output)
    status_payload = _extract_json(status.output)
    assert threads_payload["count"] == 1
    assert threads_payload["work_threads"][0]["product_kind"] == "work_thread"
    assert status_payload["session_products"]["stale_profile_row_count"] == 0
    assert status_payload["session_products"]["stale_work_event_inference_count"] == 0
    assert status_payload["session_products"]["stale_phase_inference_count"] == 0
    assert status_payload["session_products"]["profile_merged_fts_duplicate_count"] == 0
    assert status_payload["session_products"]["profile_rows_ready"] is True
    assert status_payload["session_products"]["profile_merged_fts_ready"] is True
    assert status_payload["session_products"]["profile_evidence_fts_ready"] is True
    assert status_payload["session_products"]["profile_inference_fts_ready"] is True
    assert status_payload["session_products"]["profile_enrichment_fts_ready"] is True
    assert status_payload["session_products"]["phase_inference_rows_ready"] is True
    assert status_payload["session_products"]["threads_ready"] is True
    assert status_payload["session_products"]["tag_rollups_ready"] is True
    assert status_payload["session_products"]["day_summaries_ready"] is True
    assert status_payload["session_products"]["week_summaries_ready"] is True
    assert status_payload["archive_debt"]["tracked_items"] >= 1
    assert status_payload["archive_debt"]["actionable_items"] == 0
    assert status_payload["archive_debt"]["issue_rows"] == 0


def test_products_tag_and_summary_rollups_json(cli_workspace):
    _seed_products(cli_workspace)

    runner = CliRunner()
    tags = runner.invoke(cli, ["products", "tags", "--json"], catch_exceptions=False)
    days = runner.invoke(cli, ["products", "day-summaries", "--json"], catch_exceptions=False)
    weeks = runner.invoke(cli, ["products", "week-summaries", "--json"], catch_exceptions=False)

    assert tags.exit_code == 0
    assert days.exit_code == 0
    assert weeks.exit_code == 0

    tag_payload = _extract_json(tags.output)
    day_payload = _extract_json(days.output)
    week_payload = _extract_json(weeks.output)
    assert any(item["tag"] == "provider:claude-code" for item in tag_payload["session_tag_rollups"])
    assert day_payload["count"] == 1
    assert day_payload["day_session_summaries"][0]["product_kind"] == "day_session_summary"
    assert week_payload["count"] == 1
    assert week_payload["week_session_summaries"][0]["product_kind"] == "week_session_summary"


def test_products_maintenance_json(cli_workspace):
    _record_maintenance_lineage(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "maintenance", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert payload["count"] >= 1
    assert payload["maintenance_runs"][0]["product_kind"] == "maintenance_run"
    assert payload["maintenance_runs"][0]["target_names"] == ["session_products"]


def test_products_analytics_json(cli_workspace):
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["products", "analytics", "--provider", "claude-code", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert payload["count"] == 1
    item = payload["provider_analytics"][0]
    assert item["product_kind"] == "provider_analytics"
    assert item["provider_name"] == "claude-code"
    assert item["conversation_count"] == 2
    assert item["tool_use_count"] == 2


def test_products_debt_json(cli_workspace):
    _seed_products(cli_workspace)
    _record_maintenance_lineage(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "debt", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = _extract_json(result.output)
    assert payload["count"] >= 1
    assert {item["product_kind"] for item in payload["archive_debt"]} == {"archive_debt"}
    session_product_item = next(item for item in payload["archive_debt"] if item["debt_name"] == "session_products")
    assert session_product_item["governance_stage"] == "validated"
    assert session_product_item["lineage"]["latest_preview_at"] is not None
    assert session_product_item["lineage"]["latest_successful_apply_at"] is not None
    assert session_product_item["lineage"]["latest_validation_at"] is not None
    assert session_product_item["lineage"]["latest_validation_issue_count"] == 0
    assert session_product_item["lineage"]["latest_successful_validation_at"] is not None


def test_session_product_status_accepts_epoch_backed_conversation_timestamps(cli_workspace):
    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-epoch")
        .provider("claude-code")
        .title("Epoch-backed timestamps")
        .created_at("1740823200.0")
        .updated_at("1740826800.0")
        .add_message("u1", role="user", text="Inspect the archive state")
        .add_message(
            "a1",
            role="assistant",
            text="Inspecting files and planning repairs",
            provider_meta={
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "semantic_type": "file_read",
                        "input": {"path": "/realm/project/polylogue/README.md"},
                    }
                ]
            },
        )
        .save()
    )

    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        status = session_product_status_sync(conn)

    assert status["profile_row_count"] == 1
    assert status["stale_profile_row_count"] == 0
    assert status["stale_work_event_inference_count"] == 0
    assert status["stale_phase_inference_count"] == 0
    assert status["profile_rows_ready"] is True
    assert status["work_event_inference_rows_ready"] is True
    assert status["phase_inference_rows_ready"] is True
    assert status["profile_merged_fts_duplicate_count"] == 0


def test_targeted_session_product_rebuild_does_not_duplicate_profile_fts(cli_workspace):
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        rebuild_session_products_sync(conn, conversation_ids=["conv-root"])
        status = session_product_status_sync(conn)

    assert status["profile_row_count"] == 2
    assert status["profile_merged_fts_count"] == 2
    assert status["profile_merged_fts_duplicate_count"] == 0
    assert status["profile_merged_fts_ready"] is True
