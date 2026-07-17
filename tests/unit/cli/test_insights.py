"""Tests for the durable archive datan insights CLI surfaces."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner, Result

from polylogue.api.archive import _rebuild_archive_session_insights
from polylogue.cli.click_app import cli
from polylogue.cli.commands.insights import _make_callback
from polylogue.insights.archive import ArchiveCoverageInsight
from polylogue.insights.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION
from polylogue.insights.registry import get_insight_type, insight_items_payload
from polylogue.storage.insights.session.runtime import SessionInsightCounts, SessionInsightStatusSnapshot
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import upsert_session_profile_costs
from tests.infra.archive_scenarios import native_session_id_for, open_index_db
from tests.infra.json_contracts import (
    extract_json_result,
    json_array,
    json_int,
    json_number,
    json_object,
    json_object_list,
)
from tests.infra.storage_records import SessionBuilder

CliWorkspace = dict[str, Path]

# session ids for the seeded sessions. The CLI surfaces
# now emit archive ``<origin>:ext-<conv-id>`` ids, so read-side assertions key on
# these rather than the raw builder tokens.
NID_ROOT = native_session_id_for("claude-code", "conv-root")
NID_CHILD = native_session_id_for("claude-code", "conv-child")
NID_EXACT_COST = native_session_id_for("claude-code", "conv-exact-cost")
NID_PRICED_COST = native_session_id_for("chatgpt", "conv-priced-cost")
NID_UNAVAILABLE_COST = native_session_id_for("chatgpt", "conv-unavailable-cost")
NID_EPOCH = native_session_id_for("claude-code", "conv-epoch")
NID_HEAVY = native_session_id_for("codex", "conv-heavy")


def _rebuild_insights(db_path: Path, **kwargs: Any) -> SessionInsightCounts:
    """Materialize session insights for a seeded ``index.db``.

    This is the archive equivalent of ``rebuild_session_insights_sync``
    over a v22 connection: it opens the archive at ``db_path``'s root
    and runs the archive session-insight materializer, populating ``session_profiles``,
    ``session_work_events``, ``session_phases``, and ``threads``.
    """
    with ArchiveStore.open_existing(db_path.parent, read_only=False) as archive:
        return _rebuild_archive_session_insights(archive, **kwargs)


def _insight_status(db_path: Path) -> SessionInsightStatusSnapshot:
    """Archive session-insight readiness snapshot for a seeded ``index.db``."""
    with ArchiveStore.open_existing(db_path.parent, read_only=False) as archive:
        return archive.session_insight_status()


def _exception_message(result: Result) -> str:
    return str(result.exception) if result.exception is not None else result.output.strip()


def test_insight_items_payload_can_render_cli_and_mcp_keys() -> None:
    product = ArchiveCoverageInsight(
        group_by="origin",
        bucket="claude-code-session",
        origin="claude-code-session",
        session_count=1,
        message_count=2,
        user_message_count=1,
        assistant_message_count=1,
        avg_messages_per_session=2.0,
        avg_user_words=3.0,
        avg_assistant_words=4.0,
        tool_use_count=1,
        thinking_count=0,
        total_sessions_with_tools=1,
        total_sessions_with_thinking=0,
        tool_use_percentage=100.0,
        thinking_percentage=0.0,
    )
    insight_type = get_insight_type("archive_coverage")

    cli_payload = insight_items_payload([product], insight_type)
    mcp_payload = insight_items_payload([product], insight_type, item_key="items")

    assert cli_payload["total"] == 1
    assert json_object_list(cli_payload["archive_coverage"])[0]["insight_kind"] == "archive_coverage"
    assert mcp_payload["total"] == 1
    assert json_object_list(mcp_payload["items"])[0]["origin"] == "claude-code-session"


def _seed_products(cli_workspace: CliWorkspace) -> None:
    db_path = cli_workspace["db_path"]
    (
        SessionBuilder(db_path, "conv-root")
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
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Read",
                    "semantic_type": "file_read",
                    "input": {"path": "/workspace/polylogue/README.md"},
                },
                {
                    "type": "tool_use",
                    "tool_name": "Edit",
                    "semantic_type": "file_edit",
                    "input": {"path": "/workspace/polylogue/README.md"},
                },
            ],
        )
        .save()
    )
    (
        SessionBuilder(db_path, "conv-child")
        .provider("claude-code")
        .title("Child Thread")
        .parent_session("ext-conv-root")
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
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Bash",
                    "semantic_type": "shell",
                    "input": {"command": "pytest -q tests/unit/cli/test_insights.py"},
                }
            ],
        )
        .save()
    )
    _rebuild_insights(db_path)


def _seed_cost_products(cli_workspace: CliWorkspace) -> None:
    db_path = cli_workspace["db_path"]
    (
        SessionBuilder(db_path, "conv-exact-cost")
        .provider("claude-code")
        .title("Exact Cost")
        .metadata({"total_cost_usd": 1.25, "model": "claude-sonnet-4-5"})
        .updated_at("2026-03-01T12:00:00+00:00")
        .add_message("u1", role="user", text="Run exact-cost task", timestamp="2026-03-01T11:55:00+00:00")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-priced-cost")
        .provider("chatgpt")
        .title("Priced Cost")
        .metadata({"model": "openai/gpt-4o-2024-08-06", "usage": {"input_tokens": 1000, "output_tokens": 500}})
        .updated_at("2026-03-01T13:00:00+00:00")
        .add_message("u2", role="user", text="Run priced-cost task", timestamp="2026-03-01T12:55:00+00:00")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-unavailable-cost")
        .provider("chatgpt")
        .title("Unavailable Cost")
        .updated_at("2026-03-01T14:00:00+00:00")
        .add_message("u3", role="user", text="Run unavailable-cost task", timestamp="2026-03-01T13:55:00+00:00")
        .save()
    )
    _rebuild_insights(db_path)
    with open_index_db(db_path) as conn:
        upsert_session_profile_costs(
            conn,
            NID_EXACT_COST,
            cost_usd=1.25,
            cost_is_estimated=False,
            cost_provenance="exact",
        )
        upsert_session_profile_costs(
            conn,
            NID_PRICED_COST,
            cost_usd=0.0075,
            cost_is_estimated=True,
            cost_provenance="priced",
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO session_model_usage (
                session_id, model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                cost_credits, cost_provenance
            ) VALUES (?, 'openai/gpt-4o-2024-08-06', 1000, 500, 0, 0, 0, 'origin_reported')
            """,
            (NID_PRICED_COST,),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO session_provider_usage_events (
                session_id, source_message_id, position, provider_event_type, model_name,
                last_input_tokens, last_output_tokens, last_cached_input_tokens,
                last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
                total_input_tokens, total_output_tokens, total_cached_input_tokens,
                total_cache_write_tokens, total_reasoning_output_tokens, total_tokens,
                occurred_at_ms
            ) VALUES (
                ?, NULL, 0, 'message_usage', 'openai/gpt-4o-2024-08-06',
                1000, 500, 200, 50, 75, 1750,
                1000, 500, 200, 50, 75, 1750,
                1772389200000
            )
            """,
            (NID_PRICED_COST,),
        )
        conn.commit()


def test_insights_profiles_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "insights", "profiles", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["total"]) == 2
    profiles = json_object_list(payload["session_profiles"])
    first = next(item for item in profiles if item["session_id"] == NID_ROOT)
    evidence = json_object(first["evidence"])
    inference = json_object(first["inference"])
    assert json_int(first["contract_version"]) == ARCHIVE_INSIGHT_CONTRACT_VERSION
    assert first["insight_kind"] == "session_profile"
    assert first["semantic_tier"] == "merged"
    assert evidence["canonical_session_date"] == "2026-03-01"
    assert evidence["first_message_at"] == "2026-03-01T10:00:00+00:00"
    assert evidence["last_message_at"] == "2026-03-01T10:05:00+00:00"
    assert json_int(evidence["timestamped_message_count"]) == 2
    assert json_int(evidence["untimestamped_message_count"]) == 0
    assert evidence["timestamp_coverage"] == "complete"
    assert json_int(evidence["wall_duration_ms"]) == 300000
    assert json_number(inference["engaged_duration_ms"]) >= 0
    assert "evidence" in first
    assert "inference" in first
    assert "provenance" in first
    provenance = json_object(first["provenance"])
    assert provenance["input_high_water_mark_source"] == "provider_ts"
    assert provenance["time_confidence"] == "recorded"


def test_insights_costs_json(cli_workspace: CliWorkspace) -> None:
    _seed_cost_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "insights", "costs", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    costs = json_object_list(payload["session_costs"])
    exact = next(item for item in costs if item["session_id"] == NID_EXACT_COST)
    priced = next(item for item in costs if item["session_id"] == NID_PRICED_COST)
    unavailable = next(item for item in costs if item["session_id"] == NID_UNAVAILABLE_COST)
    exact_estimate = json_object(exact["estimate"])
    priced_estimate = json_object(priced["estimate"])
    unavailable_estimate = json_object(unavailable["estimate"])
    assert exact["insight_kind"] == "session_cost"
    assert exact_estimate["status"] == "exact"
    assert json_number(exact_estimate["total_usd"]) == pytest.approx(1.25)
    assert priced_estimate["status"] == "priced"
    assert priced_estimate["normalized_model"] == "gpt-4o"
    assert json_number(priced_estimate["total_usd"]) == pytest.approx(0.0075)
    assert unavailable_estimate["status"] == "unavailable"
    assert "missing_token_usage" in json_array(unavailable_estimate["missing_reasons"])


def test_insights_cost_rollups_json(cli_workspace: CliWorkspace) -> None:
    _seed_cost_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "insights", "cost-rollups", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    rollups = json_object_list(payload["cost_rollups"])
    claude = next(item for item in rollups if item["origin"] == "claude-code-session")
    gpt = next(item for item in rollups if item["normalized_model"] == "gpt-4o")
    unknown = next(item for item in rollups if item["origin"] == "chatgpt-export" and item["normalized_model"] is None)
    assert claude["insight_kind"] == "cost_rollup"
    assert json_number(claude["total_usd"]) == pytest.approx(1.25)
    assert json_int(claude["priced_session_count"]) == 1
    assert json_number(gpt["total_usd"]) == pytest.approx(0.0075)
    assert json_int(unknown["unavailable_session_count"]) == 1


def test_insights_usage_timeline_json(cli_workspace: CliWorkspace) -> None:
    _seed_cost_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--origin", "chatgpt-export", "analyze", "insights", "usage-timeline", "--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    rows = json_object_list(payload["usage_timeline"])
    row = next(item for item in rows if item["origin"] == "chatgpt-export")
    usage = json_object(row["usage"])
    assert row["insight_kind"] == "usage_timeline"
    assert row["bucket"] == "2026-03"
    assert row["normalized_model"] == "gpt-4o"
    assert json_int(row["event_count"]) == 1
    assert json_int(row["session_count"]) >= 1
    assert json_int(usage["input_tokens"]) == 1000
    assert json_int(usage["cache_read_tokens"]) == 200
    assert json_int(row["reasoning_output_tokens"]) == 75


def test_usage_timeline_first_page_skips_later_provider_events(cli_workspace: CliWorkspace) -> None:
    _seed_cost_products(cli_workspace)
    db_path = cli_workspace["db_path"]
    with open_index_db(db_path) as conn:
        conn.execute(
            "UPDATE session_provider_usage_events SET occurred_at_ms = ? WHERE session_id = ?",
            (1775067600000, NID_PRICED_COST),
        )
        conn.commit()

    with ArchiveStore.open_existing(db_path.parent, read_only=True) as archive:
        cutoff_ms, skip_event_scan = archive._usage_timeline_event_scan_cutoff_ms(
            origin="chatgpt-export",
            model=None,
            group_by="month-origin-model",
            since_ms=None,
            until_ms=None,
            limit=1,
        )
        rows = archive.list_usage_timeline_insights(origin="chatgpt-export", limit=1)

    assert cutoff_ms is not None
    assert skip_event_scan is True
    assert len(rows) == 1
    assert rows[0].bucket == "2026-03"
    assert rows[0].origin == "chatgpt-export"
    assert rows[0].event_count == 0
    assert rows[0].stored_cost_usd == 0.0
    assert rows[0].cost_provenance_counts == {"origin_reported": 1}


def test_insights_profiles_support_wallclock_filters_and_sort(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "analyze",
            "insights",
            "profiles",
            "--sort",
            "wallclock",
            "--min-wallclock-seconds",
            "250",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    profiles = json_object_list(payload["session_profiles"])
    assert json_int(payload["total"]) == 1
    assert profiles[0]["session_id"] == NID_ROOT

    result = runner.invoke(
        cli,
        ["analyze", "insights", "profiles", "--sort", "wallclock", "--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    profiles = json_object_list(payload["session_profiles"])
    assert [item["session_id"] for item in profiles] == [NID_ROOT, NID_CHILD]


def test_insights_profiles_plain_output_shows_session_time_axis(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "insights", "profiles", "--sort", "wallclock"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "first=2026-03-01T10:00:00+00:00" in result.output
    assert "last=2026-03-01T10:05:00+00:00" in result.output
    assert "wall_s=300" in result.output
    assert "ts_cov=complete" in result.output


def test_insights_profiles_format_json_alias(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "insights", "profiles", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["total"]) == 2


def test_insights_profiles_inherit_root_format_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--format", "json", "analyze", "insights", "profiles", "--limit", "1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["total"]) == 1


def test_analyze_rejects_aggregate_options_before_insights_subcommand(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--format", "json", "insights", "profiles"], catch_exceptions=False)

    assert result.exit_code == 2
    assert "put read-model options after `analyze insights <command>`" in result.output


def test_insights_status_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["ops", "insights", "status", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert payload["aggregate_verdict"] == "degraded"
    insights = {item["insight_name"]: item for item in json_object_list(payload["insights"])}
    assert set(insights) >= {
        "session_profiles",
        "session_work_events",
        "session_phases",
        "threads",
        "session_tag_rollups",
        "archive_coverage",
    }
    assert insights["session_profiles"]["verdict"] == "degraded"
    assert json_int(insights["session_profiles"]["degraded_count"]) == 2
    assert json_int(insights["session_work_events"]["row_count"]) >= 1


def test_insights_status_inherits_root_format_and_filters(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--format",
            "json",
            "--origin",
            "claude-code-session",
            "ops",
            "insights",
            "status",
            "--insight",
            "session-work-events",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert payload["origin"] == "claude-code-session"
    insights = json_object_list(payload["insights"])
    assert len(insights) == 1
    assert insights[0]["insight_name"] == "session_work_events"
    coverage = json_object_list(insights[0]["origin_coverage"])
    assert coverage[0]["origin"] == "claude-code-session"


def test_insights_status_plain(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["ops", "insights", "status", "--insight", "profiles"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Insight Readiness: degraded" in result.output
    assert "session_profiles: degraded" in result.output


def test_insights_audit_json(cli_workspace: CliWorkspace) -> None:
    """``insights audit`` returns the per-product rigor profile (#1275)."""

    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["ops", "insights", "audit", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0, _exception_message(result)
    payload = extract_json_result(result.output)
    entries = {item["insight_name"]: item for item in json_object_list(payload["entries"])}
    assert "session_profiles" in entries
    assert "session_work_events" in entries
    profiles = entries["session_profiles"]
    assert profiles["has_evidence_payload"] is True
    assert profiles["has_inference_payload"] is True
    assert json_int(profiles["sample_size"]) >= 1
    assert json_int(profiles["evidence_count"]) >= 1
    # Fallback markers are declared for work events.
    we = entries["session_work_events"]
    assert we["has_fallback_markers"] is True
    assert "confidence_distribution" in we
    assert "version_targets" in profiles


def test_insights_audit_plain_renders_summary(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["ops", "insights", "audit"], catch_exceptions=False)

    assert result.exit_code == 0, _exception_message(result)
    assert "Insight Rigor Audit" in result.output
    assert "session_profiles" in result.output
    assert "evidence:" in result.output


def test_insights_export_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)
    target = cli_workspace["archive_root"] / "exports" / "insights-bundle"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ops", "insights", "export", "--out", str(target), "--insight", "profiles", "--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert payload["output_path"] == str(target)
    assert (target / "manifest.json").exists()
    assert (target / "coverage.json").exists()
    assert (target / "insights" / "session_profiles.jsonl").exists()


def test_insights_profiles_json_includes_folded_enrichment(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--origin",
            "claude-code-session",
            "analyze",
            "insights",
            "profiles",
            "--session-date-since",
            "2026-03-01",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["total"]) == 2
    first = json_object_list(payload["session_profiles"])[0]
    enrichment_provenance = json_object(first["enrichment_provenance"])
    enrichment = json_object(first["enrichment"])
    assert json_int(first["contract_version"]) == ARCHIVE_INSIGHT_CONTRACT_VERSION
    assert first["insight_kind"] == "session_profile"
    assert first["semantic_tier"] == "merged"
    assert enrichment_provenance["enrichment_family"] == "archive"
    assert enrichment["support_level"] in {"weak", "moderate", "strong"}
    assert "input_band_summary" in enrichment


def test_insights_callback_rejects_unknown_query_fields() -> None:
    callback = _make_callback(get_insight_type("session_profiles"))
    env = SimpleNamespace(polylogue=MagicMock())
    # Build a minimal Click context to satisfy @click.pass_context
    mock_ctx = click.Context(click.Command("profiles"))
    mock_ctx.obj = env
    # No parent — root filter inheritance is a no-op

    with pytest.raises(
        SystemExit, match="insights profiles: Unknown query field\\(s\\) for session_profiles: refined_work_kind"
    ):
        wrapped = getattr(callback, "__wrapped__", None)
        assert callable(wrapped)
        wrapped(mock_ctx, refined_work_kind="planning")


def test_insights_profiles_json_supports_explicit_evidence_and_inference_tiers(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    evidence_result = runner.invoke(
        cli,
        ["analyze", "insights", "profiles", "--tier", "evidence", "--format", "json"],
        catch_exceptions=False,
    )
    inference_result = runner.invoke(
        cli,
        ["analyze", "insights", "profiles", "--tier", "inference", "--format", "json"],
        catch_exceptions=False,
    )

    assert evidence_result.exit_code == 0
    assert inference_result.exit_code == 0

    evidence_payload = extract_json_result(evidence_result.output)
    inference_payload = extract_json_result(inference_result.output)
    evidence_profile = json_object_list(evidence_payload["session_profiles"])[0]
    inference_profile = json_object_list(inference_payload["session_profiles"])[0]

    assert evidence_profile["semantic_tier"] == "evidence"
    assert json_object(evidence_profile["evidence"])["canonical_session_date"] == "2026-03-01"
    assert evidence_profile["inference"] is None
    assert evidence_profile["inference_provenance"] is None

    assert inference_profile["semantic_tier"] == "inference"
    assert inference_profile["evidence"] is None
    assert json_number(json_object(inference_profile["inference"])["engaged_duration_ms"]) >= 0
    assert json_object(inference_profile["inference_provenance"])["inference_family"] == "archive"


def test_insights_profiles_json_handles_blank_tier_search_text_from_migrated_rows(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)
    with open_index_db(cli_workspace["db_path"]) as conn:
        conn.execute("UPDATE session_profiles SET search_text = ''")
        conn.commit()

    runner = CliRunner()
    evidence_result = runner.invoke(
        cli,
        ["analyze", "insights", "profiles", "--tier", "evidence", "--format", "json"],
        catch_exceptions=False,
    )
    inference_result = runner.invoke(
        cli,
        ["analyze", "insights", "profiles", "--tier", "inference", "--format", "json"],
        catch_exceptions=False,
    )

    assert evidence_result.exit_code == 0
    assert inference_result.exit_code == 0
    assert json_int(extract_json_result(evidence_result.output)["total"]) == 2
    assert json_int(extract_json_result(inference_result.output)["total"]) == 2


# Retired (#1743): tiered-payload reconstruction from blanked
# provenance/evidence/inference columns was a legacy rebuild-recovery path.
# stores the tiered payloads canonically (no base-column
# reconstruction fallback), so blanking them is not a supported state to recover.


def test_insights_work_events_filter_by_canonical_session_date(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    matched = runner.invoke(
        cli,
        [
            "analyze",
            "insights",
            "work-events",
            "--session-date-since",
            "2026-03-01",
            "--session-date-until",
            "2026-03-01",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    missed = runner.invoke(
        cli,
        [
            "analyze",
            "insights",
            "work-events",
            "--session-date-since",
            "2026-03-02",
            "--session-date-until",
            "2026-03-02",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert matched.exit_code == 0
    assert missed.exit_code == 0
    assert json_int(extract_json_result(matched.output)["total"]) >= 1
    assert json_int(extract_json_result(missed.output)["total"]) == 0


def test_insights_profile_date_filters_and_phases_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    profiles = runner.invoke(
        cli,
        [
            "analyze",
            "insights",
            "profiles",
            "--session-date-since",
            "2026-03-01",
            "--session-date-until",
            "2026-03-01",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    phases = runner.invoke(cli, ["analyze", "insights", "phases", "--format", "json"], catch_exceptions=False)

    assert profiles.exit_code == 0
    assert phases.exit_code == 0

    profile_payload = extract_json_result(profiles.output)
    phase_payload = extract_json_result(phases.output)
    assert json_int(profile_payload["total"]) == 2
    assert json_int(phase_payload["total"]) >= 1
    assert json_object_list(phase_payload["session_phases"])[0]["insight_kind"] == "session_phase"


def test_session_insight_rebuild_pages_full_rebuild(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    counts = _rebuild_insights(cli_workspace["db_path"])
    status = _insight_status(cli_workspace["db_path"])

    assert counts.profiles == 2
    assert counts.work_events >= 1
    assert counts.phases >= 1
    assert status.profile_row_count == 2
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True


def test_session_insight_rebuild_sync_reports_progress(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    observed: list[tuple[int, str | None]] = []

    _rebuild_insights(
        cli_workspace["db_path"],
        progress_callback=lambda amount, desc=None: observed.append((json_int(amount), desc)),
    )

    # Bounded-WAL model (#2458): the full rebuild no longer clears per-session
    # insight tables upfront. It upserts/commits per chunk and prunes orphan
    # rows after the loop, emitting a per-table "pruned orphans" heartbeat. The
    # old upfront "cleared session_*" heartbeats must be gone.
    prune_events = [event for event in observed if event[1] and event[1].startswith("rebuild: pruned orphans from ")]
    assert [desc for _, desc in prune_events] == [
        "rebuild: pruned orphans from session_work_events",
        "rebuild: pruned orphans from session_phases",
        "rebuild: pruned orphans from session_latency_profiles",
        "rebuild: pruned orphans from session_profiles",
    ]
    assert not [event for event in observed if event[1] and event[1].startswith("rebuild: cleared session_")]


def test_session_insight_rebuild_materializes_profile_and_repo_for_git_session(
    cli_workspace: CliWorkspace,
) -> None:
    db_path = cli_workspace["db_path"]
    builder = (
        SessionBuilder(db_path, "conv-heavy")
        .provider("codex")
        .title("Heavy Provider Meta")
        .created_at("2026-03-01T10:00:00+00:00")
        .updated_at("2026-03-01T10:10:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Continue work on /realm/project/sinex and inspect the branch.",
            timestamp="2026-03-01T10:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Inspecting the repository state.",
            timestamp="2026-03-01T10:05:00+00:00",
        )
    )
    # Archive repo derivation reads ParsedSession git fields (written at
    # ingest into ``session_repos``), not a legacy ``repo_names_json`` profile
    # column.
    builder.conv = builder.conv.model_copy(
        update={
            "git_repository_url": "git@github.com:Sinity/sinex.git",
            "git_branch": "master",
        }
    )
    builder.save()

    counts = _rebuild_insights(db_path)

    assert counts.profiles == 1
    with open_index_db(db_path) as conn:
        profile = conn.execute(
            "SELECT session_id FROM session_profiles WHERE session_id = ?",
            (NID_HEAVY,),
        ).fetchone()
        repo_names = [
            str(r["repo_name"])
            for r in conn.execute(
                """
                SELECT repos.repo_name
                FROM session_repos
                JOIN repos
                  ON repos.repo_id = session_repos.repo_id
                WHERE session_repos.session_id = ?
                """,
                (NID_HEAVY,),
            ).fetchall()
        ]
    assert profile is not None
    assert "sinex" in repo_names


def test_insights_threads_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    threads = runner.invoke(cli, ["analyze", "insights", "threads", "--format", "json"], catch_exceptions=False)

    assert threads.exit_code == 0

    threads_payload = extract_json_result(threads.output)
    assert json_int(threads_payload["total"]) == 1
    thread = json_object_list(threads_payload["threads"])[0]
    assert thread["insight_kind"] == "thread"
    thread_payload = json_object(thread["thread"])
    # Archive thread payloads attribute support to the archive thread tables
    # rather than the legacy ``explicit_lineage`` lineage-scoring vocabulary.
    assert "archive_threads" in json_array(thread_payload["support_signals"])
    members = json_object_list(thread_payload["member_evidence"])
    assert [member["session_id"] for member in members] == [NID_ROOT, NID_CHILD]
    assert members[0]["role"] == "root"
    assert members[1]["role"] == "parent_continuation"
    assert members[1]["parent_id"] == NID_ROOT


def test_insights_tag_and_summary_rollups_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    tags = runner.invoke(cli, ["analyze", "insights", "tags", "--format", "json"], catch_exceptions=False)
    days = runner.invoke(
        cli,
        ["analyze", "insights", "coverage", "--group-by", "day", "--format", "json"],
        catch_exceptions=False,
    )
    weeks = runner.invoke(
        cli,
        ["analyze", "insights", "coverage", "--group-by", "week", "--format", "json"],
        catch_exceptions=False,
    )

    assert tags.exit_code == 0
    assert days.exit_code == 0
    assert weeks.exit_code == 0

    tag_payload = extract_json_result(tags.output)
    day_payload = extract_json_result(days.output)
    week_payload = extract_json_result(weeks.output)
    assert any(
        item["tag"] == "origin:claude-code-session" for item in json_object_list(tag_payload["session_tag_rollups"])
    )
    assert json_int(day_payload["total"]) == 1
    assert json_object_list(day_payload["archive_coverage"])[0]["insight_kind"] == "archive_coverage"
    assert json_object_list(day_payload["archive_coverage"])[0]["group_by"] == "day"
    assert json_int(week_payload["total"]) == 1
    assert json_object_list(week_payload["archive_coverage"])[0]["group_by"] == "week"


def test_insights_analytics_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--origin",
            "claude-code-session",
            "analyze",
            "insights",
            "coverage",
            "--group-by",
            "origin",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["total"]) == 1
    item = json_object_list(payload["archive_coverage"])[0]
    assert item["insight_kind"] == "archive_coverage"
    assert item["origin"] == "claude-code-session"
    assert json_int(item["session_count"]) == 2
    assert json_int(item["tool_use_count"]) == 2


def test_insights_debt_json(cli_workspace: CliWorkspace) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "insights", "debt", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["total"]) >= 1
    first = json_object_list(payload["archive_debt"])[0]
    assert first["insight_kind"] == "archive_debt"
    assert "maintenance_target" in first


def test_session_insight_status_accepts_epoch_backed_session_timestamps(cli_workspace: CliWorkspace) -> None:
    db_path = cli_workspace["db_path"]
    (
        SessionBuilder(db_path, "conv-epoch")
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
                        "input": {"path": "/workspace/polylogue/README.md"},
                    }
                ]
            },
        )
        .save()
    )

    _rebuild_insights(db_path)
    status = _insight_status(db_path)

    assert status.profile_row_count == 1
    assert status.stale_profile_row_count == 0
    assert status.stale_work_event_inference_count == 0
    assert status.stale_phase_inference_count == 0
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True
    assert status.profile_merged_fts_duplicate_count == 0


def test_targeted_session_insight_rebuild_does_not_duplicate_profile_fts(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    _rebuild_insights(cli_workspace["db_path"], session_ids=[NID_ROOT])
    status = _insight_status(cli_workspace["db_path"])

    assert status.profile_row_count == 2
    assert status.profile_rows_ready is True


def test_session_insight_status_marks_missing_profile_rows_not_ready(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    # Archive readiness is structural: a session without a materialized
    # ``session_profiles`` row makes the profile surface not-ready. (The legacy
    # per-row ``materializer_version`` staleness column does not exist in the v1
    # schema; missing/orphan rows are the archive incompleteness signal.)
    with open_index_db(cli_workspace["db_path"]) as conn:
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (NID_CHILD,))
        conn.commit()

    status = _insight_status(cli_workspace["db_path"])

    assert status.profile_row_count == 1
    assert status.missing_profile_row_count == 1
    assert status.profile_rows_ready is False


def test_insights_timeline_json_emits_fidelity_tags(cli_workspace: CliWorkspace) -> None:
    """`analyze insights timeline <conv-id> --format json` emits per-entry fidelity tags."""
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["analyze", "insights", "timeline", NID_ROOT, "--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert payload["session_id"] == NID_ROOT
    counts = json_object(payload["fidelity_counts"])
    # Both fidelity buckets are present in the contract output even if zero.
    assert "hook" in counts
    assert "sort_key" in counts
    entries = json_object_list(payload["entries"])
    # Every entry carries a fidelity tag drawn from the contract set.
    for entry in entries:
        assert entry["fidelity"] in {"hook", "sort_key"}
        assert "timing_provenance" in entry
        assert entry["source"] in {"work_event", "phase"}


def test_insights_timeline_plain_includes_legend(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["analyze", "insights", "timeline", NID_ROOT],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert f"Timeline for {NID_ROOT}" in result.output
    assert "legend:" in result.output
