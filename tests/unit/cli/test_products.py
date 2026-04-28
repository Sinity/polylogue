"""Tests for the durable archive data products CLI surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from polylogue.cli.commands.products import _make_callback
from polylogue.products.archive import ProviderAnalyticsProduct
from polylogue.products.archive_models import ARCHIVE_PRODUCT_CONTRACT_VERSION
from polylogue.products.registry import get_product_type, product_items_payload
from polylogue.storage.action_events.rebuild_runtime import rebuild_action_event_read_model_sync
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.products.session.rebuild import rebuild_session_products_sync
from polylogue.storage.products.session.status import session_product_status_sync
from polylogue.storage.runtime.store_constants import SESSION_PRODUCT_MATERIALIZER_VERSION
from tests.infra.json_contracts import (
    extract_json_result,
    json_array,
    json_int,
    json_number,
    json_object,
    json_object_list,
)
from tests.infra.storage_records import ConversationBuilder

CliWorkspace = dict[str, Path]


def _exception_message(result: Result) -> str:
    return str(result.exception) if result.exception is not None else result.output.strip()


def test_product_items_payload_can_render_cli_and_mcp_keys() -> None:
    product = ProviderAnalyticsProduct(
        provider_name="claude-code",
        conversation_count=1,
        message_count=2,
        user_message_count=1,
        assistant_message_count=1,
        avg_messages_per_conversation=2.0,
        avg_user_words=3.0,
        avg_assistant_words=4.0,
        tool_use_count=1,
        thinking_count=0,
        total_conversations_with_tools=1,
        total_conversations_with_thinking=0,
        tool_use_percentage=100.0,
        thinking_percentage=0.0,
    )
    product_type = get_product_type("provider_analytics")

    cli_payload = product_items_payload([product], product_type)
    mcp_payload = product_items_payload([product], product_type, item_key="items")

    assert cli_payload["count"] == 1
    assert json_object_list(cli_payload["provider_analytics"])[0]["product_kind"] == "provider_analytics"
    assert mcp_payload["count"] == 1
    assert json_object_list(mcp_payload["items"])[0]["provider_name"] == "claude-code"


def _seed_products(cli_workspace: CliWorkspace) -> None:
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
                        "input": {"path": "/workspace/polylogue/README.md"},
                    },
                    {
                        "type": "tool_use",
                        "tool_name": "Edit",
                        "semantic_type": "file_edit",
                        "input": {"path": "/workspace/polylogue/README.md"},
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


def _seed_cost_products(cli_workspace: CliWorkspace) -> None:
    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-exact-cost")
        .provider("claude-code")
        .title("Exact Cost")
        .provider_meta({"total_cost_usd": 1.25, "model": "claude-sonnet-4-5"})
        .updated_at("2026-03-01T12:00:00+00:00")
        .add_message("u1", role="user", text="Run exact-cost task", timestamp="2026-03-01T11:55:00+00:00")
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-priced-cost")
        .provider("chatgpt")
        .title("Priced Cost")
        .provider_meta({"model": "openai/gpt-4o-2024-08-06", "usage": {"input_tokens": 1000, "output_tokens": 500}})
        .updated_at("2026-03-01T13:00:00+00:00")
        .add_message("u2", role="user", text="Run priced-cost task", timestamp="2026-03-01T12:55:00+00:00")
        .save()
    )
    (
        ConversationBuilder(db_path, "conv-unavailable-cost")
        .provider("chatgpt")
        .title("Unavailable Cost")
        .updated_at("2026-03-01T14:00:00+00:00")
        .add_message("u3", role="user", text="Run unavailable-cost task", timestamp="2026-03-01T13:55:00+00:00")
        .save()
    )


def test_products_profiles_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "profiles", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["count"]) == 2
    profiles = json_object_list(payload["session_profiles"])
    first = next(item for item in profiles if item["conversation_id"] == "conv-root")
    evidence = json_object(first["evidence"])
    inference = json_object(first["inference"])
    assert json_int(first["contract_version"]) == ARCHIVE_PRODUCT_CONTRACT_VERSION
    assert first["product_kind"] == "session_profile"
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


def test_products_costs_json(cli_workspace: CliWorkspace) -> None:
    _seed_cost_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "costs", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    costs = json_object_list(payload["session_costs"])
    exact = next(item for item in costs if item["conversation_id"] == "conv-exact-cost")
    priced = next(item for item in costs if item["conversation_id"] == "conv-priced-cost")
    unavailable = next(item for item in costs if item["conversation_id"] == "conv-unavailable-cost")
    exact_estimate = json_object(exact["estimate"])
    priced_estimate = json_object(priced["estimate"])
    unavailable_estimate = json_object(unavailable["estimate"])
    assert exact["product_kind"] == "session_cost"
    assert exact_estimate["status"] == "exact"
    assert json_number(exact_estimate["total_usd"]) == pytest.approx(1.25)
    assert priced_estimate["status"] == "priced"
    assert priced_estimate["normalized_model"] == "gpt-4o"
    assert json_number(priced_estimate["total_usd"]) == pytest.approx(0.0075)
    assert unavailable_estimate["status"] == "unavailable"
    assert "missing_token_usage" in json_array(unavailable_estimate["missing_reasons"])


def test_products_cost_rollups_json(cli_workspace: CliWorkspace) -> None:
    _seed_cost_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "cost-rollups", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    rollups = json_object_list(payload["cost_rollups"])
    claude = next(item for item in rollups if item["provider_name"] == "claude-code")
    gpt = next(item for item in rollups if item["normalized_model"] == "gpt-4o")
    unknown = next(item for item in rollups if item["provider_name"] == "chatgpt" and item["normalized_model"] is None)
    assert claude["product_kind"] == "cost_rollup"
    assert json_number(claude["total_usd"]) == pytest.approx(1.25)
    assert json_int(claude["priced_session_count"]) == 1
    assert json_number(gpt["total_usd"]) == pytest.approx(0.0075)
    assert json_int(unknown["unavailable_session_count"]) == 1


def test_products_profiles_support_wallclock_filters_and_sort(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["products", "profiles", "--sort", "wallclock", "--min-wallclock-seconds", "250", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    profiles = json_object_list(payload["session_profiles"])
    assert json_int(payload["count"]) == 1
    assert profiles[0]["conversation_id"] == "conv-root"

    result = runner.invoke(
        cli,
        ["products", "profiles", "--sort", "wallclock", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    profiles = json_object_list(payload["session_profiles"])
    assert [item["conversation_id"] for item in profiles] == ["conv-root", "conv-child"]


def test_products_profiles_plain_output_shows_session_time_axis(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "profiles", "--sort", "wallclock"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "first=2026-03-01T10:00:00+00:00" in result.output
    assert "last=2026-03-01T10:05:00+00:00" in result.output
    assert "wall_s=300" in result.output
    assert "ts_cov=complete" in result.output


def test_products_profiles_format_json_alias(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "profiles", "--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["count"]) == 2


def test_products_profiles_inherit_root_format_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--format", "json", "products", "profiles", "--limit", "1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["count"]) == 1


def test_products_status_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "status", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert payload["aggregate_verdict"] == "ready"
    products = {item["product_name"]: item for item in json_object_list(payload["products"])}
    assert set(products) >= {
        "session_profiles",
        "session_enrichments",
        "session_work_events",
        "session_phases",
        "work_threads",
        "session_tag_rollups",
        "day_session_summaries",
        "week_session_summaries",
        "provider_analytics",
    }
    assert products["session_profiles"]["verdict"] == "ready"
    assert json_int(products["session_work_events"]["row_count"]) >= 1


def test_products_status_inherits_root_format_and_filters(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--format",
            "json",
            "--provider",
            "claude-code",
            "products",
            "status",
            "--product",
            "session-work-events",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert payload["provider"] == "claude-code"
    products = json_object_list(payload["products"])
    assert len(products) == 1
    assert products[0]["product_name"] == "session_work_events"
    coverage = json_object_list(products[0]["provider_coverage"])
    assert coverage[0]["provider_name"] == "claude-code"


def test_products_status_plain(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(cli, ["products", "status", "--product", "profiles"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Product Readiness: ready" in result.output
    assert "session_profiles: ready" in result.output


def test_products_export_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)
    target = cli_workspace["archive_root"] / "exports" / "products-bundle"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["products", "export", "--out", str(target), "--product", "profiles", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert payload["output_path"] == str(target)
    assert (target / "manifest.json").exists()
    assert (target / "coverage.json").exists()
    assert (target / "products" / "session_profiles.jsonl").exists()


def test_products_enrichments_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--provider",
            "claude-code",
            "products",
            "enrichments",
            "--session-date-since",
            "2026-03-01",
            "--json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["count"]) == 2
    first = json_object_list(payload["session_enrichments"])[0]
    enrichment_provenance = json_object(first["enrichment_provenance"])
    enrichment = json_object(first["enrichment"])
    assert json_int(first["contract_version"]) == ARCHIVE_PRODUCT_CONTRACT_VERSION
    assert first["product_kind"] == "session_enrichment"
    assert first["semantic_tier"] == "enrichment"
    assert enrichment_provenance["enrichment_family"] == "scored_session_enrichment"
    assert enrichment["support_level"] in {"weak", "moderate", "strong"}
    assert "input_band_summary" in enrichment


def test_products_callback_rejects_unknown_query_fields() -> None:
    callback = _make_callback(get_product_type("session_enrichments"))
    env = SimpleNamespace(operations=MagicMock())
    # Build a minimal Click context to satisfy @click.pass_context
    mock_ctx = click.Context(click.Command("enrichments"))
    mock_ctx.obj = env
    # No parent — root filter inheritance is a no-op

    with pytest.raises(
        SystemExit, match="products enrichments: Unknown query field\\(s\\) for session_enrichments: refined_work_kind"
    ):
        wrapped = getattr(callback, "__wrapped__", None)
        assert callable(wrapped)
        wrapped(mock_ctx, json_mode=False, refined_work_kind="planning")


def test_products_profiles_json_supports_explicit_evidence_and_inference_tiers(cli_workspace: CliWorkspace) -> None:
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
    assert json_object(inference_profile["inference_provenance"])["inference_family"] == "heuristic_session_semantics"


def test_products_profiles_json_handles_blank_tier_search_text_from_migrated_rows(cli_workspace: CliWorkspace) -> None:
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
    assert json_int(extract_json_result(evidence_result.output)["count"]) == 2
    assert json_int(extract_json_result(inference_result.output)["count"]) == 2


def test_products_reconstructs_tiered_payloads_from_blank_migrated_rows(cli_workspace: CliWorkspace) -> None:
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

    evidence_profiles_payload = json_object_list(extract_json_result(evidence_profiles.output)["session_profiles"])
    inference_profiles_payload = json_object_list(extract_json_result(inference_profiles.output)["session_profiles"])
    work_event = json_object_list(extract_json_result(work_events.output)["session_work_events"])[0]
    phase = json_object_list(extract_json_result(phases.output)["session_phases"])[0]

    assert all(json_int(json_object(item["evidence"])["message_count"]) >= 1 for item in evidence_profiles_payload)
    assert all(
        json_object(item["evidence"])["canonical_session_date"] == "2026-03-01" for item in evidence_profiles_payload
    )
    assert all(json_int(json_object(item["inference"])["work_event_count"]) >= 0 for item in inference_profiles_payload)
    assert json_int(json_object(work_event["evidence"])["start_index"]) >= 0
    assert json_object(work_event["inference"])["kind"] in {"implementation", "testing", "planning", "debugging"}
    assert json_int(json_array(json_object(phase["evidence"])["message_range"])[0]) >= 0
    assert json_number(json_object(phase["inference"])["confidence"]) >= 0.0


def test_products_profile_date_filters_and_phases_json(cli_workspace: CliWorkspace) -> None:
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

    profile_payload = extract_json_result(profiles.output)
    phase_payload = extract_json_result(phases.output)
    assert json_int(profile_payload["count"]) == 2
    assert json_int(phase_payload["count"]) >= 1
    assert json_object_list(phase_payload["session_phases"])[0]["product_kind"] == "session_phase"


def test_session_product_rebuild_supports_legacy_payload_columns(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("ALTER TABLE session_profiles ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE session_work_events ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE session_phases ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'")
        conn.commit()
        rebuild_session_products_sync(conn)
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 2
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True


def test_session_product_rebuild_pages_full_rebuild(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        counts = rebuild_session_products_sync(conn, page_size=1)
        status = session_product_status_sync(conn)

    assert counts.profiles == 2
    assert counts.work_events >= 1
    assert counts.phases >= 1
    assert status.profile_row_count == 2
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True


def test_session_product_rebuild_sync_reports_progress(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    observed: list[tuple[int, str | None]] = []

    with open_connection(cli_workspace["db_path"]) as conn:
        rebuild_session_products_sync(
            conn,
            page_size=1,
            progress_callback=lambda amount, desc=None: observed.append((json_int(amount), desc)),
            progress_total=2,
        )

    assert observed == [
        (1, "Materializing: 1/2"),
        (1, "Materializing: 2/2"),
    ]


def test_session_product_rebuild_preserves_profile_semantics_without_loading_full_provider_meta(
    cli_workspace: CliWorkspace,
) -> None:
    db_path = cli_workspace["db_path"]
    (
        ConversationBuilder(db_path, "conv-heavy")
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
        .save()
    )
    huge_provider_meta = {
        "git": {
            "branch": "master",
            "repository_url": "git@github.com:Sinity/sinex.git",
        },
        "context_compactions": [{"summary": "Earlier context collapsed."}],
        "raw": {"payload": "x" * 200_000},
    }

    with open_connection(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET provider_meta = ? WHERE conversation_id = ?",
            (json.dumps(huge_provider_meta), "conv-heavy"),
        )
        counts = rebuild_session_products_sync(conn, page_size=1)
        row = conn.execute(
            "SELECT repo_names_json, evidence_payload_json FROM session_profiles WHERE conversation_id = ?",
            ("conv-heavy",),
        ).fetchone()

    assert counts.profiles == 1
    assert row is not None
    repo_names = json_array(json.loads(row["repo_names_json"]))
    evidence_payload = json_object(json.loads(row["evidence_payload_json"]))
    assert "sinex" in repo_names
    assert json_int(evidence_payload["compaction_count"]) == 1


def test_products_threads_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    threads = runner.invoke(cli, ["products", "threads", "--json"], catch_exceptions=False)

    assert threads.exit_code == 0

    threads_payload = extract_json_result(threads.output)
    assert json_int(threads_payload["count"]) == 1
    thread = json_object_list(threads_payload["work_threads"])[0]
    assert thread["product_kind"] == "work_thread"
    thread_payload = json_object(thread["thread"])
    assert thread_payload["support_level"] == "strong"
    assert "explicit_lineage" in json_array(thread_payload["support_signals"])
    members = json_object_list(thread_payload["member_evidence"])
    assert [member["conversation_id"] for member in members] == ["conv-root", "conv-child"]
    assert members[0]["role"] == "root"
    assert members[1]["role"] == "parent_continuation"
    assert members[1]["parent_id"] == "conv-root"


def test_products_tag_and_summary_rollups_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    tags = runner.invoke(cli, ["products", "tags", "--json"], catch_exceptions=False)
    days = runner.invoke(cli, ["products", "day-summaries", "--json"], catch_exceptions=False)
    weeks = runner.invoke(cli, ["products", "week-summaries", "--json"], catch_exceptions=False)

    assert tags.exit_code == 0
    assert days.exit_code == 0
    assert weeks.exit_code == 0

    tag_payload = extract_json_result(tags.output)
    day_payload = extract_json_result(days.output)
    week_payload = extract_json_result(weeks.output)
    assert any(item["tag"] == "provider:claude-code" for item in json_object_list(tag_payload["session_tag_rollups"]))
    assert json_int(day_payload["count"]) == 1
    assert json_object_list(day_payload["day_session_summaries"])[0]["product_kind"] == "day_session_summary"
    assert json_int(week_payload["count"]) == 1
    assert json_object_list(week_payload["week_session_summaries"])[0]["product_kind"] == "week_session_summary"


def test_products_analytics_json(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--provider", "claude-code", "products", "analytics", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["count"]) == 1
    item = json_object_list(payload["provider_analytics"])[0]
    assert item["product_kind"] == "provider_analytics"
    assert item["provider_name"] == "claude-code"
    assert json_int(item["conversation_count"]) == 2
    assert json_int(item["tool_use_count"]) == 2


def test_products_debt_json(cli_workspace: CliWorkspace) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["products", "debt", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = extract_json_result(result.output)
    assert json_int(payload["count"]) >= 1
    first = json_object_list(payload["archive_debt"])[0]
    assert first["product_kind"] == "archive_debt"
    assert "maintenance_target" in first


def test_session_product_status_accepts_epoch_backed_conversation_timestamps(cli_workspace: CliWorkspace) -> None:
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
                        "input": {"path": "/workspace/polylogue/README.md"},
                    }
                ]
            },
        )
        .save()
    )

    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 1
    assert status.stale_profile_row_count == 0
    assert status.stale_work_event_inference_count == 0
    assert status.stale_phase_inference_count == 0
    assert status.profile_rows_ready is True
    assert status.work_event_inference_rows_ready is True
    assert status.phase_inference_rows_ready is True
    assert status.profile_merged_fts_duplicate_count == 0


def test_targeted_session_product_rebuild_does_not_duplicate_profile_fts(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        rebuild_session_products_sync(conn, conversation_ids=["conv-root"])
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 2
    assert status.profile_merged_fts_count == 2
    assert status.profile_merged_fts_duplicate_count == 0
    assert status.profile_merged_fts_ready is True


def test_session_product_status_marks_older_materializer_versions_stale(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        rebuild_session_products_sync(conn)
        conn.execute(
            "UPDATE session_profiles SET materializer_version = ?",
            (SESSION_PRODUCT_MATERIALIZER_VERSION - 1,),
        )
        conn.commit()
        status = session_product_status_sync(conn)

    assert status.profile_row_count == 2
    assert status.stale_profile_row_count == 2
    assert status.profile_rows_ready is False


@pytest.mark.parametrize(
    ("argv", "sql", "expected"),
    [
        (
            ["products", "profiles", "--json"],
            "UPDATE session_profiles SET materializer_version = ?",
            "Session-profile rows are incomplete.",
        ),
        (
            ["products", "work-events", "--json"],
            "UPDATE session_work_events SET materializer_version = ?",
            "Session work-event rows are incomplete.",
        ),
        (
            ["products", "phases", "--json"],
            "UPDATE session_phases SET materializer_version = ?",
            "Session phase rows are incomplete.",
        ),
        (
            ["products", "threads", "--json"],
            "UPDATE work_threads SET materializer_version = ?",
            "Work-thread rows are incomplete.",
        ),
        (
            ["products", "tags", "--json"],
            "UPDATE session_tag_rollups SET materializer_version = ?",
            "Session tag rollups are incomplete.",
        ),
        (
            ["products", "day-summaries", "--json"],
            "UPDATE day_session_summaries SET materializer_version = ?",
            "Day session summaries are incomplete.",
        ),
        (
            ["products", "week-summaries", "--json"],
            "UPDATE day_session_summaries SET materializer_version = ?",
            "Week session summaries are incomplete.",
        ),
    ],
)
def test_products_reject_stale_session_product_surfaces(
    cli_workspace: CliWorkspace, argv: list[str], sql: str, expected: str
) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute(sql, (SESSION_PRODUCT_MATERIALIZER_VERSION - 1,))
        conn.commit()

    runner = CliRunner()
    result = runner.invoke(cli, argv, catch_exceptions=False)

    assert result.exit_code == 1
    message = _exception_message(result)
    assert expected in message
    assert "polylogue doctor --repair --target session_products" in message


def test_products_profiles_reject_incomplete_profile_search_index(cli_workspace: CliWorkspace) -> None:
    _seed_products(cli_workspace)

    with open_connection(cli_workspace["db_path"]) as conn:
        conn.execute("DELETE FROM session_profiles_fts")
        conn.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["products", "profiles", "--query", "refactor", "--json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    message = _exception_message(result)
    assert "Session-profile merged search index is incomplete." in message
    assert "polylogue doctor --repair --target session_products" in message
